"""Finite-budget predictor and policy training helpers."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .contracts import Method
from .environment import NativeCellGrid
from .episodes import run_episode
from .features import FeatureBank
from .models import CAIActor, CommonCAIPredictor
from .perception import highest_reliable_candidate_mask
from .policies import fixed_action_order


def state_dict_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError("model state contains a non-tensor value")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(slots=True)
class PredictorTrainingResult:
    model: CommonCAIPredictor
    losses: tuple[float, ...]
    validation_scores: tuple[tuple[int, float], ...]
    updates_completed: int
    selected_update: int


@dataclass(slots=True)
class ActorTrainingResult:
    model: CAIActor
    training_losses: tuple[float, ...]
    validation_scores: tuple[tuple[int, float], ...]
    updates_completed: int
    selected_update: int


def _random_measured_masks(
    rng: np.random.Generator,
    *,
    batch_size: int,
    mask_counts: tuple[int, ...],
) -> np.ndarray:
    masks = np.zeros((batch_size, 64), dtype=bool)
    for row in range(batch_size):
        count = int(rng.choice(mask_counts))
        if count:
            masks[row, rng.choice(64, size=count, replace=False)] = True
    return masks


def _exact_costs(
    bank: FeatureBank, indices: np.ndarray, measured: np.ndarray
) -> np.ndarray:
    output = np.empty(len(indices), dtype=np.float32)
    for row, index in enumerate(indices):
        shape = tuple(int(item) for item in bank.native_shapes[int(index)])
        grid = NativeCellGrid.from_shape(shape)
        output[row] = grid.measured_cost(
            {int(cell) for cell in np.flatnonzero(measured[row])}
        )
    return output


def _predictor_validation_score(
    model: CommonCAIPredictor,
    bank: FeatureBank,
    *,
    indices: np.ndarray,
    mask_counts: tuple[int, ...],
    device: str,
) -> float:
    order = fixed_action_order(Method.GEOMETRY_SPREAD, seed=0)
    domain_errors: dict[str, list[float]] = {}
    model.eval()
    with torch.inference_mode():
        for index in indices:
            surface = torch.from_numpy(bank.surface_tokens[int(index) : int(index) + 1]).to(device)
            cscan = torch.from_numpy(bank.cscan_tokens[int(index) : int(index) + 1]).to(device)
            grid = NativeCellGrid.from_shape(
                tuple(int(item) for item in bank.native_shapes[int(index)])
            )
            for count in mask_counts:
                selected = set(order[:count])
                measured_np = np.zeros((1, 64), dtype=bool)
                if selected:
                    measured_np[0, list(selected)] = True
                cost = grid.measured_cost(selected)
                prediction = float(
                    model(
                        surface,
                        cscan,
                        torch.from_numpy(measured_np).to(device),
                        cost=torch.tensor([cost], dtype=torch.float32, device=device),
                    )[0].cpu()
                )
                domain_errors.setdefault(bank.dataset_ids[int(index)], []).append(
                    abs(prediction - float(bank.targets_mpa[int(index)]))
                )
    model.train()
    return float(np.mean([np.mean(values) for values in domain_errors.values()]))


def train_predictor(
    bank: FeatureBank,
    *,
    fit_indices: np.ndarray,
    updates: int,
    batch_size: int,
    width: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    seed: int,
    device: str,
    validation_indices: np.ndarray | None = None,
    validation_interval: int = 250,
    mask_counts: tuple[int, ...] = (0, 1, 2, 4, 8, 12, 16, 32, 64),
) -> PredictorTrainingResult:
    fit = np.asarray(fit_indices, dtype=np.int64)
    validation = (
        np.asarray(validation_indices, dtype=np.int64)
        if validation_indices is not None
        else fit
    )
    if (
        updates < 1
        or batch_size < 1
        or len(fit) < 1
        or len(validation) < 1
        or validation_interval < 1
        or not mask_counts
        or any(type(count) is not int or not 0 <= count <= 64 for count in mask_counts)
    ):
        raise ValueError("predictor training request is invalid")
    if any(bank.splits[int(index)] != "TRAIN" for index in fit):
        raise ValueError("predictor fitting is restricted to TRAIN")
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    target_mean = float(np.mean(bank.targets_mpa[fit]))
    target_scale = max(float(np.std(bank.targets_mpa[fit])), 1.0)
    model = CommonCAIPredictor(
        token_dimension=bank.token_dimension,
        width=width,
        target_mean=target_mean,
        target_scale=target_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    surface = torch.from_numpy(bank.surface_tokens).to(device)
    cscan = torch.from_numpy(bank.cscan_tokens).to(device)
    targets = torch.from_numpy(bank.targets_mpa).to(device)
    losses: list[float] = []
    validation_scores: list[tuple[int, float]] = []
    best_score = float("inf")
    best_update = 0
    best_state: dict[str, torch.Tensor] | None = None
    model.train()
    for update in range(1, updates + 1):
        batch_indices = np_rng.choice(fit, size=batch_size, replace=True)
        measured_np = _random_measured_masks(
            np_rng, batch_size=batch_size, mask_counts=mask_counts
        )
        measured = torch.from_numpy(measured_np).to(device)
        index_tensor = torch.from_numpy(batch_indices).to(device)
        costs = torch.from_numpy(_exact_costs(bank, batch_indices, measured_np)).to(device)
        prediction = model(
            surface[index_tensor], cscan[index_tensor], measured, cost=costs
        )
        loss = torch.mean(((prediction - targets[index_tensor]) / target_scale) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if update % validation_interval == 0 or update == updates:
            score = _predictor_validation_score(
                model,
                bank,
                indices=validation,
                mask_counts=mask_counts,
                device=device,
            )
            validation_scores.append((update, score))
            if score < best_score:
                best_score = score
                best_update = update
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError("predictor validation produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return PredictorTrainingResult(
        model,
        tuple(losses),
        tuple(validation_scores),
        updates,
        best_update,
    )


def _actor_for_method(
    method: Method, *, token_dimension: int, width: int
) -> CAIActor:
    if method is Method.VLM_CAI_FEEDBACK_AGENT:
        flags = (True, True)
    elif method is Method.NO_VLM_FEEDBACK:
        flags = (False, True)
    elif method is Method.VLM_OPEN_LOOP:
        flags = (True, False)
    elif method is Method.LEARNED_STATIC:
        flags = (False, False)
    else:
        raise ValueError("actor training requires a learned method")
    return CAIActor(
        token_dimension=token_dimension,
        width=width,
        use_vlm=flags[0],
        use_feedback=flags[1],
    )


def _cell_cost_matrix(bank: FeatureBank, indices: np.ndarray) -> np.ndarray:
    rows = []
    for index in indices:
        shape = tuple(int(item) for item in bank.native_shapes[int(index)])
        rows.append([cell.cost for cell in NativeCellGrid.from_shape(shape).cells])
    return np.asarray(rows, dtype=np.float32)


def _validation_actor_score(
    bank: FeatureBank,
    *,
    indices: np.ndarray,
    method: Method,
    predictor: CommonCAIPredictor,
    actor: CAIActor,
    seed: int,
    endpoint_budget: float,
    device: str,
) -> float:
    values = [
        run_episode(
            bank,
            specimen_index=int(index),
            method=method,
            predictor=predictor,
            actor=actor,
            seed=seed,
            endpoint_budget=endpoint_budget,
            device=device,
            sample_actions=False,
        ).normalized_error_area_mpa
        for index in indices
    ]
    return float(np.mean(values))


def train_actor(
    bank: FeatureBank,
    *,
    method: Method,
    reward_predictors: tuple[CommonCAIPredictor, ...],
    reward_query_folds: tuple[np.ndarray, ...],
    validation_indices: np.ndarray,
    updates: int,
    batch_size: int,
    width: int,
    endpoint_budget: float,
    validation_interval: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    entropy_weight: float,
    value_weight: float,
    seed: int,
    device: str,
    validation_predictor: CommonCAIPredictor | None = None,
) -> ActorTrainingResult:
    if (
        updates < 1
        or batch_size < 1
        or validation_interval < 1
        or len(reward_predictors) != len(reward_query_folds)
        or not reward_predictors
    ):
        raise ValueError("actor training request is invalid")
    for fold in reward_query_folds:
        if len(fold) < 1 or any(bank.splits[int(index)] != "TRAIN" for index in fold):
            raise ValueError("actor rewards must use TRAIN OOF folds")
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    actor = _actor_for_method(
        method, token_dimension=bank.token_dimension, width=width
    ).to(device)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for predictor in reward_predictors:
        predictor.to(device).eval()
        for parameter in predictor.parameters():
            parameter.requires_grad_(False)
    validation_model = validation_predictor or reward_predictors[0]
    validation_model.to(device).eval()

    surface_all = torch.from_numpy(bank.surface_tokens).to(device)
    cscan_all = torch.from_numpy(bank.cscan_tokens).to(device)
    indicator_all = torch.from_numpy(bank.vlm_indicators).to(device)
    confidence_all = torch.from_numpy(bank.vlm_confidences).to(device)
    available_all = torch.from_numpy(bank.vlm_available).to(device)
    no_reliable_all = torch.from_numpy(bank.vlm_no_reliable).to(device)
    reliable_all = torch.from_numpy(
        highest_reliable_candidate_mask(bank.vlm_indicators, bank.vlm_confidences)
    ).to(device)
    targets_all = torch.from_numpy(bank.targets_mpa).to(device)
    target_scale = max(
        float(np.std(bank.targets_mpa[bank.indices("TRAIN")])), 1.0
    )
    losses: list[float] = []
    validation_scores: list[tuple[int, float]] = []
    best_score = float("inf")
    best_update = 0
    best_state: dict[str, torch.Tensor] | None = None

    for update in range(1, updates + 1):
        fold_index = (update - 1) % len(reward_query_folds)
        fold = reward_query_folds[fold_index]
        predictor = reward_predictors[fold_index]
        chosen = np_rng.choice(fold, size=batch_size, replace=True).astype(np.int64)
        index_tensor = torch.from_numpy(chosen).to(device)
        surface = surface_all[index_tensor]
        cscan = cscan_all[index_tensor]
        indicators = indicator_all[index_tensor]
        confidences = confidence_all[index_tensor]
        available = available_all[index_tensor]
        no_reliable = no_reliable_all[index_tensor]
        reliable_candidates = reliable_all[index_tensor]
        targets = targets_all[index_tensor]
        cell_costs = torch.from_numpy(_cell_cost_matrix(bank, chosen)).to(device)
        measured = torch.zeros(batch_size, 64, dtype=torch.bool, device=device)
        action_history = torch.zeros(
            batch_size, 64, dtype=torch.float32, device=device
        )
        costs = torch.zeros(batch_size, dtype=torch.float32, device=device)
        with torch.no_grad():
            predictions = predictor(surface, cscan, measured, cost=costs)
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []

        for action_count in range(64):
            legal = (~measured) & (
                costs.unsqueeze(1) + cell_costs <= endpoint_budget + 1e-12
            )
            if not bool(torch.all(torch.any(legal, dim=1))):
                break
            proposal = legal
            if action_count == 0 and method.uses_vlm:
                restricted = legal & reliable_candidates
                has_reliable = torch.any(restricted, dim=1, keepdim=True)
                proposal = torch.where(has_reliable, restricted, legal)
            scores, value = actor(
                surface,
                cscan,
                measured,
                action_history,
                indicators,
                confidences,
                available,
                no_reliable,
                predictions,
                costs,
                endpoint_budget - costs,
            )
            distribution = torch.distributions.Categorical(
                logits=scores.masked_fill(~proposal, -torch.inf)
            )
            actions = distribution.sample()
            log_probs.append(distribution.log_prob(actions))
            entropies.append(distribution.entropy())
            values.append(value)
            delta = cell_costs.gather(1, actions.unsqueeze(1)).squeeze(1)
            measured = measured.clone()
            measured.scatter_(1, actions.unsqueeze(1), True)
            action_history = action_history.clone()
            action_history.scatter_(
                1, actions.unsqueeze(1), float(action_count + 1) / 64.0
            )
            costs = costs + delta
            with torch.no_grad():
                predictions = predictor(surface, cscan, measured, cost=costs)
                error = torch.abs(predictions - targets)
                rewards.append(-error * (delta / endpoint_budget) / target_scale)

        if not rewards:
            raise ValueError("actor training episode contains no affordable action")
        rewards[-1] = rewards[-1] - torch.abs(predictions - targets) / target_scale
        returns: list[torch.Tensor] = []
        running = torch.zeros_like(rewards[-1])
        for reward in reversed(rewards):
            running = reward + running
            returns.append(running)
        returns.reverse()
        log_prob_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        value_tensor = torch.stack(values)
        return_tensor = torch.stack(returns).detach()
        advantage = return_tensor - value_tensor.detach()
        loss = (
            -(log_prob_tensor * advantage).mean()
            + value_weight * torch.mean((value_tensor - return_tensor) ** 2)
            - entropy_weight * entropy_tensor.mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        if update % validation_interval == 0 or update == updates:
            actor.eval()
            score = _validation_actor_score(
                bank,
                indices=np.asarray(validation_indices, dtype=np.int64),
                method=method,
                predictor=validation_model,
                actor=actor,
                seed=seed,
                endpoint_budget=endpoint_budget,
                device=device,
            )
            validation_scores.append((update, score))
            if score < best_score:
                best_score = score
                best_update = update
                best_state = copy.deepcopy(actor.state_dict())
            actor.train()
    if best_state is None:
        raise RuntimeError("actor validation did not produce a checkpoint")
    actor.load_state_dict(best_state)
    actor.eval()
    return ActorTrainingResult(
        model=actor,
        training_losses=tuple(losses),
        validation_scores=tuple(validation_scores),
        updates_completed=updates,
        selected_update=best_update,
    )


__all__ = [
    "ActorTrainingResult",
    "PredictorTrainingResult",
    "fixed_action_order",
    "state_dict_sha256",
    "train_predictor",
]
