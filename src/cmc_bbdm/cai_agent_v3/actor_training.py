"""Stage-gated REINFORCE pilots for the CAI Agent v3 visible-state policies."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cmc_bbdm.cai_active_image.contracts import Method
from cmc_bbdm.cai_active_image.policies import fixed_action_order

from .feature_bank import V3FeatureBank, load_feature_bank
from .files import read_csv, sha256_file, write_csv, write_json
from .gates import policy_pilot_gate
from .metrics import (
    left_error_area_mpa,
    torch_policy_cost_to_go,
    trajectory_objective_mpa,
)
from .models import MeanFeedbackActor, SpatialCAIActor, TrueStaticActor
from .policy import vlm_first_action_mask
from .predictor_training import (
    _cell_costs,
    load_predictor_checkpoint,
)

_ENDPOINT_BUDGET = 0.25
_VALIDATION_INTERVAL = 250
_PATIENCE = 4
_BATCH_SIZE = 16
_MAX_TOTAL_OPTIMIZER_UPDATES = 28_100
_POLICY_EXPANSION_MAX_UPDATES = 9_000

_PILOT_SPECS = (
    ("VLM_SPATIAL_FEEDBACK", 1250),
    ("NO_VLM_SPATIAL_FEEDBACK", 1250),
    ("VLM_SPATIAL_OPEN_LOOP", 1250),
    ("LEARNED_STATIC_TRUE", 750),
    ("VLM_MEAN_FEEDBACK", 1250),
)


def _legal_action_mask(
    measured: np.ndarray,
    exact_costs: np.ndarray,
    per_cell_costs: np.ndarray,
) -> np.ndarray:
    if (
        measured.ndim != 2
        or exact_costs.shape != (len(measured),)
        or per_cell_costs.shape != measured.shape
        or measured.dtype != np.bool_
        or exact_costs.dtype != np.float64
        or per_cell_costs.dtype != np.float64
    ):
        raise ValueError("policy legality inputs are invalid")
    return (~measured) & (
        exact_costs[:, None] + per_cell_costs <= _ENDPOINT_BUDGET + 1e-12
    )


def _append_ledger(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _optimizer_update_upper_bound(path: Path) -> int:
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            actual = row.get("actual_optimizer_updates")
            if isinstance(actual, int):
                total += actual
            elif actual is None and isinstance(
                row.get("actual_optimizer_updates_upper_bound"), int
            ):
                total += int(row["actual_optimizer_updates_upper_bound"])
    return total


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


class _VLMFeatures:
    def __init__(
        self,
        bank: V3FeatureBank,
        path: Path,
        *,
        required_splits: tuple[str, ...] = ("TRAIN", "VALID"),
    ) -> None:
        rows = {row["specimen_key"]: row for row in read_csv(path)}
        required = {
            bank.specimen_keys[int(index)]
            for split in required_splits
            for index in bank.indices(split)
        }
        if not required <= set(rows):
            raise ValueError("fit VLM actor features are incomplete")
        self.indicator = np.zeros((len(bank.specimen_keys), 64), dtype=np.float32)
        self.confidence = np.zeros((len(bank.specimen_keys), 64), dtype=np.float32)
        self.available = np.zeros(len(bank.specimen_keys), dtype=bool)
        self.no_reliable = np.zeros(len(bank.specimen_keys), dtype=bool)
        self.cache_keys = [""] * len(bank.specimen_keys)
        for index, key in enumerate(bank.specimen_keys):
            row = rows.get(key)
            if row is None:
                continue
            indicator = np.asarray(
                [float(value) for value in row["region_indicator"].split(";")],
                dtype=np.float32,
            )
            confidence = np.asarray(
                [float(value) for value in row["confidence"].split(";")],
                dtype=np.float32,
            )
            if indicator.shape != (64,) or confidence.shape != (64,):
                raise ValueError("VLM actor feature row has an invalid grid")
            self.indicator[index] = indicator
            self.confidence[index] = confidence
            self.available[index] = row["vlm_available"] == "True"
            self.no_reliable[index] = row["no_reliable_cue"] == "True"
            self.cache_keys[index] = row["cache_key"]


def _actor(method: str, *, target_mean: float, target_scale: float) -> nn.Module:
    kwargs = {"target_mean": target_mean, "target_scale": target_scale}
    if method == "VLM_SPATIAL_FEEDBACK":
        return SpatialCAIActor(use_vlm=True, use_feedback=True, **kwargs)
    if method == "NO_VLM_SPATIAL_FEEDBACK":
        return SpatialCAIActor(use_vlm=False, use_feedback=True, **kwargs)
    if method == "VLM_SPATIAL_OPEN_LOOP":
        return SpatialCAIActor(use_vlm=True, use_feedback=False, **kwargs)
    if method == "LEARNED_STATIC_TRUE":
        return TrueStaticActor()
    if method == "VLM_MEAN_FEEDBACK":
        return MeanFeedbackActor(**kwargs)
    raise ValueError(f"unknown Actor method: {method}")


def _uses_vlm(method: str) -> bool:
    return method in {
        "GDFS_ADAPTED_FROZEN_PREDICTOR_S1",
        "VLM_SPATIAL_FEEDBACK",
        "VLM_SPATIAL_OPEN_LOOP",
        "VLM_MEAN_FEEDBACK",
    }


def _sample_specimens(
    rng: np.random.Generator, bank: V3FeatureBank, *, batch_size: int
) -> np.ndarray:
    train = bank.indices("TRAIN")
    domains = {
        domain: np.asarray(
            [index for index in train if bank.dataset_ids[int(index)] == domain],
            dtype=np.int64,
        )
        for domain in sorted({bank.dataset_ids[int(index)] for index in train})
    }
    output = []
    domain_names = tuple(domains)
    for _ in range(batch_size):
        domain = domain_names[int(rng.integers(0, len(domain_names)))]
        values = domains[domain]
        output.append(int(values[int(rng.integers(0, len(values)))]))
    return np.asarray(output, dtype=np.int64)


def _predict_by_fold(
    predictors: dict[int, nn.Module],
    folds: np.ndarray,
    surface: torch.Tensor,
    cscan: torch.Tensor,
    measured: torch.Tensor,
    costs: torch.Tensor,
) -> torch.Tensor:
    predictions = torch.empty(len(folds), dtype=surface.dtype, device=surface.device)
    with torch.no_grad():
        for fold, predictor in predictors.items():
            rows = np.flatnonzero(folds == fold)
            if len(rows) == 0:
                continue
            indices = torch.from_numpy(rows).to(surface.device)
            predictions[indices] = predictor(
                surface[indices],
                cscan[indices],
                measured[indices],
                cost=costs[indices],
            )
    return predictions


def _actor_forward(
    actor: nn.Module,
    method: str,
    *,
    surface: torch.Tensor,
    cscan: torch.Tensor,
    measured: torch.Tensor,
    history: torch.Tensor,
    indicator: torch.Tensor,
    confidence: torch.Tensor,
    available: torch.Tensor,
    no_reliable: torch.Tensor,
    current_prediction: torch.Tensor,
    costs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if method == "LEARNED_STATIC_TRUE":
        scores = actor(batch_size=len(surface), device=surface.device)
        return scores, torch.zeros(
            len(surface), dtype=surface.dtype, device=surface.device
        )
    return actor(
        surface,
        cscan,
        measured,
        history,
        indicator,
        confidence,
        available,
        no_reliable,
        current_prediction.detach(),
        costs,
        torch.full_like(costs, _ENDPOINT_BUDGET) - costs,
    )


def _training_rollout_loss(
    actor: nn.Module,
    method: str,
    bank: V3FeatureBank,
    features: _VLMFeatures,
    predictors: dict[int, nn.Module],
    fold_by_index: np.ndarray,
    cell_costs: np.ndarray,
    specimen_indices: np.ndarray,
    *,
    device: str,
    entropy_weight: float,
    target_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = len(specimen_indices)
    surface = torch.from_numpy(bank.surface_tokens[specimen_indices]).to(device)
    cscan = torch.from_numpy(bank.cscan_tokens[specimen_indices]).to(device)
    indicator = torch.from_numpy(features.indicator[specimen_indices]).to(device)
    confidence = torch.from_numpy(features.confidence[specimen_indices]).to(device)
    available = torch.from_numpy(features.available[specimen_indices]).to(device)
    no_reliable = torch.from_numpy(features.no_reliable[specimen_indices]).to(device)
    per_cell_cost = cell_costs[specimen_indices]
    folds = fold_by_index[specimen_indices]
    measured = torch.zeros((batch, 64), dtype=torch.bool, device=device)
    history = torch.zeros((batch, 64), dtype=torch.float32, device=device)
    costs = torch.zeros(batch, dtype=torch.float32, device=device)
    exact_costs = np.zeros(batch, dtype=np.float64)
    current_prediction = _predict_by_fold(
        predictors, folds, surface, cscan, measured, costs
    )
    state_costs = [[0.0] for _ in range(batch)]
    state_predictions = [[float(value)] for value in current_prediction.cpu()]
    log_probs: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    values: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    entropies: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    action_count = torch.zeros(batch, dtype=torch.int64, device=device)
    while True:
        measured_np = measured.detach().cpu().numpy()
        legal_np = _legal_action_mask(measured_np, exact_costs, per_cell_cost)
        active_rows = np.flatnonzero(legal_np.any(axis=1))
        if len(active_rows) == 0:
            break
        legal = torch.from_numpy(legal_np).to(device)
        proposal, _ = vlm_first_action_mask(
            legal=legal,
            use_vlm=_uses_vlm(method),
            action_count=action_count,
            indicator=indicator,
            confidence=confidence,
            available=available,
            no_reliable=no_reliable,
        )
        scores, critic = _actor_forward(
            actor,
            method,
            surface=surface,
            cscan=cscan,
            measured=measured,
            history=history,
            indicator=indicator,
            confidence=confidence,
            available=available,
            no_reliable=no_reliable,
            current_prediction=current_prediction,
            costs=costs,
        )
        active = torch.from_numpy(active_rows).to(device)
        masked_scores = scores[active].masked_fill(~proposal[active], -torch.inf)
        distribution = torch.distributions.Categorical(logits=masked_scores)
        actions = distribution.sample()
        for offset, row in enumerate(active_rows):
            log_probs[int(row)].append(distribution.log_prob(actions)[offset])
            values[int(row)].append(critic[active][offset])
            entropies[int(row)].append(distribution.entropy()[offset])
        measured = measured.clone()
        history = history.clone()
        costs = costs.clone()
        for offset, row in enumerate(active_rows):
            cell = int(actions[offset])
            measured[int(row), cell] = True
            action_count[int(row)] += 1
            history[int(row), cell] = action_count[int(row)].to(torch.float32) / 64.0
            next_cost = float(np.sum(per_cell_cost[int(row), measured_np[int(row)]]))
            next_cost += float(per_cell_cost[int(row), cell])
            exact_costs[int(row)] = next_cost
            costs[int(row)] = next_cost
        current_prediction = _predict_by_fold(
            predictors, folds, surface, cscan, measured, costs
        )
        for row in active_rows:
            state_costs[int(row)].append(float(exact_costs[int(row)]))
            state_predictions[int(row)].append(float(current_prediction[int(row)]))
    actor_terms = []
    value_terms = []
    entropy_terms = []
    objectives = []
    for row in range(batch):
        returns = (
            torch_policy_cost_to_go(
                torch.tensor(state_costs[row], dtype=torch.float64, device=device),
                torch.tensor(
                    state_predictions[row], dtype=torch.float32, device=device
                ),
                torch.tensor(
                    float(bank.targets_mpa[int(specimen_indices[row])]),
                    dtype=torch.float32,
                    device=device,
                ),
                budget=_ENDPOINT_BUDGET,
            )
            / target_scale
        )
        log_probability = torch.stack(log_probs[row])
        entropy = torch.stack(entropies[row])
        value = torch.stack(values[row])
        actor_terms.append(torch.sum(log_probability * (returns - value).detach()))
        value_terms.append(torch.mean(torch.square(value - returns.detach())))
        entropy_terms.append(torch.mean(entropy))
        objectives.append(float(returns[0]))
    actor_loss = torch.stack(actor_terms).mean()
    value_loss = torch.stack(value_terms).mean()
    entropy = torch.stack(entropy_terms).mean()
    if method == "LEARNED_STATIC_TRUE":
        value_loss = value_loss * 0.0
    total = actor_loss + 0.5 * value_loss - entropy_weight * entropy
    return total, {
        "actor_loss": float(actor_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "mean_scaled_objective": float(np.mean(objectives)),
        "min_action_count": float(min(len(values) for values in log_probs)),
        "max_action_count": float(max(len(values) for values in log_probs)),
    }


def _evaluate_one(
    actor: nn.Module | None,
    method: str,
    predictor: nn.Module,
    bank: V3FeatureBank,
    features: _VLMFeatures,
    cell_costs: np.ndarray,
    specimen: int,
    *,
    device: str,
    random_seed: int = 0,
    target_mpa: float | None = None,
) -> tuple[float, tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    surface = torch.from_numpy(bank.surface_tokens[specimen : specimen + 1]).to(device)
    cscan = torch.from_numpy(bank.cscan_tokens[specimen : specimen + 1]).to(device)
    measured = torch.zeros((1, 64), dtype=torch.bool, device=device)
    history = torch.zeros((1, 64), dtype=torch.float32, device=device)
    indicator = torch.from_numpy(features.indicator[specimen : specimen + 1]).to(device)
    confidence = torch.from_numpy(features.confidence[specimen : specimen + 1]).to(
        device
    )
    available = torch.from_numpy(features.available[specimen : specimen + 1]).to(device)
    no_reliable = torch.from_numpy(features.no_reliable[specimen : specimen + 1]).to(
        device
    )
    costs = [0.0]
    cells: list[int] = []
    with torch.inference_mode():
        prediction = float(
            predictor(surface, cscan, measured, cost=torch.zeros(1, device=device))[0]
        )
    predictions = [prediction]
    order = (
        fixed_action_order(Method(method), seed=random_seed)
        if method in {"CENTER_FIRST", "GEOMETRY_SPREAD", "SERPENTINE", "RANDOM"}
        else ()
    )
    cursor = 0
    while True:
        legal_np = (~measured.cpu().numpy()[0]) & (
            costs[-1] + cell_costs[specimen] <= _ENDPOINT_BUDGET + 1e-12
        )
        if not bool(legal_np.any()):
            break
        if actor is None:
            while cursor < len(order) and not legal_np[order[cursor]]:
                cursor += 1
            if cursor == len(order):
                break
            cell = order[cursor]
            cursor += 1
        else:
            legal = torch.from_numpy(legal_np[None, :]).to(device)
            proposal, _ = vlm_first_action_mask(
                legal=legal,
                use_vlm=_uses_vlm(method),
                action_count=torch.tensor([len(cells)], device=device),
                indicator=indicator,
                confidence=confidence,
                available=available,
                no_reliable=no_reliable,
            )
            with torch.inference_mode():
                scores, _ = _actor_forward(
                    actor,
                    method,
                    surface=surface,
                    cscan=cscan,
                    measured=measured,
                    history=history,
                    indicator=indicator,
                    confidence=confidence,
                    available=available,
                    no_reliable=no_reliable,
                    current_prediction=torch.tensor([predictions[-1]], device=device),
                    costs=torch.tensor([costs[-1]], device=device),
                )
                cell = int(scores[0].masked_fill(~proposal[0], -torch.inf).argmax())
        measured[0, cell] = True
        history[0, cell] = (len(cells) + 1) / 64.0
        cells.append(cell)
        next_cost = float(np.sum(cell_costs[specimen, cells]))
        costs.append(next_cost)
        with torch.inference_mode():
            prediction = float(
                predictor(
                    surface,
                    cscan,
                    measured,
                    cost=torch.tensor([next_cost], dtype=torch.float32, device=device),
                )[0]
            )
        predictions.append(prediction)
    target = (
        float(bank.targets_mpa[specimen]) if target_mpa is None else float(target_mpa)
    )
    if not math.isfinite(target):
        raise ValueError("policy evaluation target is not finite")
    area = left_error_area_mpa(costs, predictions, target, end=_ENDPOINT_BUDGET)
    return area, tuple(cells), tuple(costs), tuple(predictions)


def evaluate_policy(
    actor: nn.Module | None,
    method: str,
    predictor: nn.Module,
    bank: V3FeatureBank,
    features: _VLMFeatures,
    cell_costs: np.ndarray,
    *,
    device: str,
    split: str = "VALID",
    targets_mpa: np.ndarray | None = None,
) -> tuple[float, list[dict[str, object]]]:
    target_values = bank.targets_mpa if targets_mpa is None else np.asarray(targets_mpa)
    if target_values.shape != (len(bank.specimen_keys),):
        raise ValueError("policy evaluation target array is invalid")
    runs = range(5) if method == "RANDOM" else range(1)
    rows = []
    for specimen in bank.indices(split):
        target = float(target_values[int(specimen)])
        if not math.isfinite(target):
            raise ValueError("policy evaluation target is not finite")
        for run in runs:
            area, cells, costs, predictions = _evaluate_one(
                actor,
                method,
                predictor,
                bank,
                features,
                cell_costs,
                int(specimen),
                device=device,
                random_seed=2026091250 + run,
                target_mpa=target,
            )
            rows.append(
                {
                    "specimen_key": bank.specimen_keys[int(specimen)],
                    "dataset_id": bank.dataset_ids[int(specimen)],
                    "capture_group_id": bank.capture_group_ids[int(specimen)],
                    "method": method,
                    "run": run,
                    "target_mpa": target,
                    "left_error_area_mpa": area,
                    "early_left_error_area_mpa": left_error_area_mpa(
                        costs,
                        predictions,
                        target,
                        end=0.0625,
                    ),
                    "trajectory_objective_mpa": trajectory_objective_mpa(
                        costs,
                        predictions,
                        target,
                        budget=_ENDPOINT_BUDGET,
                    ),
                    "final_error_mpa": abs(predictions[-1] - target),
                    "action_count": len(cells),
                    "final_cost": costs[-1],
                    "unused_budget": _ENDPOINT_BUDGET - costs[-1],
                    "cells": ";".join(str(cell) for cell in cells),
                    "costs": ";".join(str(cost) for cost in costs),
                    "predictions_mpa": ";".join(
                        str(prediction) for prediction in predictions
                    ),
                }
            )
    score = _domain_equal_episode_score(rows, "left_error_area_mpa")
    return score, rows


def _domain_equal_episode_score(rows: list[dict[str, object]], metric: str) -> float:
    specimen_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        specimen_values[(str(row["dataset_id"]), str(row["specimen_key"]))].append(
            float(row[metric])
        )
    domains: dict[str, list[float]] = defaultdict(list)
    for (domain, _), values in specimen_values.items():
        domains[domain].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in domains.values()]))


def _load_oof_predictors(
    root: Path, *, device: str
) -> tuple[dict[int, nn.Module], np.ndarray]:
    output = root / "results/cai_agent_v3/new_protocol"
    readiness = json.loads((output / "oof_readiness.json").read_text(encoding="utf-8"))
    if readiness.get("status") != "REWARD_MODELS_READY":
        raise ValueError("reward predictor gate is not ready")
    predictors = {}
    for manifest in readiness["fold_manifests"]:
        model, _ = load_predictor_checkpoint(
            root / manifest["checkpoint_path"], device=device
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        predictors[int(manifest["fold"])] = model
    bank = load_feature_bank(project_root=root)
    fold_by_index = np.full(len(bank.specimen_keys), -1, dtype=np.int64)
    key_index = {key: index for index, key in enumerate(bank.specimen_keys)}
    for row in read_csv(output / "oof_fold_manifest.csv"):
        fold_by_index[key_index[row["specimen_key"]]] = int(row["fold"])
    if np.any(fold_by_index[bank.indices("TRAIN")] < 0):
        raise ValueError("TRAIN OOF routing is incomplete")
    return predictors, fold_by_index


def _train_actor(
    method: str,
    max_updates: int,
    bank: V3FeatureBank,
    features: _VLMFeatures,
    p_all: nn.Module,
    oof_predictors: dict[int, nn.Module],
    fold_by_index: np.ndarray,
    cell_costs: np.ndarray,
    *,
    seed_panel: int,
    training_seed: int,
    device: str,
) -> tuple[
    nn.Module, dict[str, object], list[dict[str, object]], list[dict[str, object]]
]:
    train_targets = bank.targets_mpa[bank.indices("TRAIN")]
    target_mean = float(np.mean(train_targets))
    target_scale = max(float(np.std(train_targets)), 1.0)
    actor = _actor(method, target_mean=target_mean, target_scale=target_scale).to(
        device
    )
    rng = np.random.default_rng(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-4)
    best_score = math.inf
    best_state = None
    best_update = 0
    stale = 0
    progress = []
    best_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for update in range(1, max_updates + 1):
        indices = _sample_specimens(rng, bank, batch_size=_BATCH_SIZE)
        entropy_weight = 0.01 * (1.0 - (update - 1) / max(max_updates - 1, 1))
        loss, terms = _training_rollout_loss(
            actor,
            method,
            bank,
            features,
            oof_predictors,
            fold_by_index,
            cell_costs,
            indices,
            device=device,
            entropy_weight=entropy_weight,
            target_scale=target_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        if update % _VALIDATION_INTERVAL == 0 or update == max_updates:
            actor.eval()
            score, validation_rows = evaluate_policy(
                actor,
                method,
                p_all,
                bank,
                features,
                cell_costs,
                device=device,
            )
            actor.train()
            progress.append(
                {
                    "method": method,
                    "seed_panel": seed_panel,
                    "training_seed": training_seed,
                    "update": update,
                    "validation_area_mpa": score,
                    "total_loss": float(loss.detach()),
                    "entropy_weight": entropy_weight,
                    **terms,
                }
            )
            if score < best_score - 1e-12:
                best_score = score
                best_state = copy.deepcopy(actor.state_dict())
                best_update = update
                best_rows = validation_rows
                stale = 0
            else:
                stale += 1
            if stale >= _PATIENCE:
                break
    if best_state is None:
        raise ValueError("Actor pilot produced no checkpoint")
    actor.load_state_dict(best_state)
    actor.eval()
    for row in best_rows:
        row["seed_panel"] = seed_panel
        row["training_seed"] = training_seed
    manifest = {
        "method": method,
        "seed_panel": seed_panel,
        "training_seed": training_seed,
        "updates_completed": update,
        "selected_update": best_update,
        "validation_area_mpa": best_score,
        "validation_early_area_mpa": _domain_equal_episode_score(
            best_rows, "early_left_error_area_mpa"
        ),
        "parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
        "state_dict_sha256": _state_dict_sha256(actor.state_dict()),
        "elapsed_seconds": time.perf_counter() - started,
        "architecture": repr(actor),
        "target_scale_mpa": target_scale,
    }
    return actor, manifest, progress, best_rows


def load_actor_checkpoint(
    path: str | Path, bank: V3FeatureBank, *, device: str
) -> tuple[nn.Module, dict[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    method = payload.get("method")
    if payload.get("schema_version") != 3 or method not in {
        row[0] for row in _PILOT_SPECS
    }:
        raise ValueError("v3 Actor checkpoint identity is invalid")
    train_targets = bank.targets_mpa[bank.indices("TRAIN")]
    actor = _actor(
        str(method),
        target_mean=float(np.mean(train_targets)),
        target_scale=max(float(np.std(train_targets)), 1.0),
    ).to(device)
    actor.load_state_dict(payload["state_dict"])
    actor.eval()
    manifest = dict(payload["manifest"])
    if _state_dict_sha256(actor.state_dict()) != manifest["state_dict_sha256"]:
        raise ValueError("v3 Actor checkpoint state hash changed")
    return actor, manifest


def precheck_policy_training(
    *, project_root: str | Path, device: str
) -> dict[str, object]:
    """Run one optimizer update for each fixed policy architecture."""

    root = Path(project_root).resolve(strict=True)
    output = root / "results/cai_agent_v3/new_protocol"
    predictor_gate = json.loads(
        (output / "predictor_gate.json").read_text(encoding="utf-8")
    )
    oof_gate = json.loads((output / "oof_readiness.json").read_text(encoding="utf-8"))
    vlm_gate = json.loads(
        (output / "vlm_manifest_fit.json").read_text(encoding="utf-8")
    )
    if (
        predictor_gate.get("status") != "PREDICTOR_READY"
        or oof_gate.get("status") != "REWARD_MODELS_READY"
        or vlm_gate.get("status") != "REAL_FROZEN_VLM_PERCEPTION_COMPLETE"
    ):
        raise ValueError("policy precheck inputs are not ready")
    bank = load_feature_bank(project_root=root)
    features = _VLMFeatures(bank, output / "vlm_actor_features_fit.csv")
    p_all_manifest = next(
        manifest
        for manifest in predictor_gate["candidate_manifests"]
        if manifest["model"] == predictor_gate["selected_p_all"]
    )
    p_all, _ = load_predictor_checkpoint(
        root / p_all_manifest["checkpoint_path"], device=device
    )
    for parameter in p_all.parameters():
        parameter.requires_grad_(False)
    oof_predictors, fold_by_index = _load_oof_predictors(root, device=device)
    cell_costs = _cell_costs(bank)
    ledger = root / "results/cai_agent_v3/compute_ledger.jsonl"
    rows = []
    for offset, (method, _) in enumerate(_PILOT_SPECS):
        job = f"policy_interface_precheck_{method}"
        started = time.perf_counter()
        _append_ledger(
            ledger,
            {
                "job": job,
                "stage": "W3_PRECHECK",
                "device": device,
                "actual_optimizer_updates": 0,
                "status": "STARTED",
            },
        )
        try:
            _, manifest, progress, validation_rows = _train_actor(
                method,
                1,
                bank,
                features,
                p_all,
                oof_predictors,
                fold_by_index,
                cell_costs,
                seed_panel=1,
                training_seed=2026091201 + offset,
                device=device,
            )
        except Exception as exc:
            _append_ledger(
                ledger,
                {
                    "job": job,
                    "stage": "W3_PRECHECK",
                    "device": device,
                    "actual_optimizer_updates": 0,
                    "status": "FAILED_BEFORE_COMPLETION",
                    "reason": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        row = {
            "method": method,
            "updates_completed": manifest["updates_completed"],
            "validation_area_mpa": manifest["validation_area_mpa"],
            "progress_rows": len(progress),
            "validation_episode_rows": len(validation_rows),
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        _append_ledger(
            ledger,
            {
                "job": job,
                "stage": "W3_PRECHECK",
                "device": device,
                "actual_optimizer_updates": 1,
                "status": "COMPLETED",
                "elapsed_seconds": row["elapsed_seconds"],
            },
        )
    payload = {
        "status": "POLICY_INTERFACE_PRECHECK_PASSED",
        "actual_optimizer_updates": len(rows),
        "models": rows,
    }
    write_json(output / "policy_precheck.json", payload)
    return payload


def run_policy_pilots(*, project_root: str | Path, device: str) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    output = root / "results/cai_agent_v3/new_protocol"
    predictor_gate = json.loads(
        (output / "predictor_gate.json").read_text(encoding="utf-8")
    )
    oof_gate = json.loads((output / "oof_readiness.json").read_text(encoding="utf-8"))
    vlm_gate = json.loads(
        (output / "vlm_manifest_fit.json").read_text(encoding="utf-8")
    )
    if (
        predictor_gate.get("status") != "PREDICTOR_READY"
        or oof_gate.get("status") != "REWARD_MODELS_READY"
        or vlm_gate.get("status") != "REAL_FROZEN_VLM_PERCEPTION_COMPLETE"
    ):
        payload = {
            "status": "POLICY_PILOT_BLOCKED_INPUT",
            "predictor_status": predictor_gate.get("status"),
            "reward_status": oof_gate.get("status"),
            "vlm_status": vlm_gate.get("status"),
        }
        write_json(output / "policy_pilot_gate.json", payload)
        return payload
    bank = load_feature_bank(project_root=root)
    features = _VLMFeatures(bank, output / "vlm_actor_features_fit.csv")
    p_all_manifest = next(
        manifest
        for manifest in predictor_gate["candidate_manifests"]
        if manifest["model"] == predictor_gate["selected_p_all"]
    )
    p_all, _ = load_predictor_checkpoint(
        root / p_all_manifest["checkpoint_path"], device=device
    )
    for parameter in p_all.parameters():
        parameter.requires_grad_(False)
    oof_predictors, fold_by_index = _load_oof_predictors(root, device=device)
    cell_costs = _cell_costs(bank)
    model_dir = output / "models"
    ledger = root / "results/cai_agent_v3/compute_ledger.jsonl"
    manifests = []
    progress_rows = []
    episode_rows = []
    actors = {}
    for offset, (method, max_updates) in enumerate(_PILOT_SPECS):
        seed_panel = 1
        training_seed = 2026091301 + offset
        started_at = time.time()
        _append_ledger(
            ledger,
            {
                "job": f"policy_pilot_{method}_seed1",
                "stage": "W3",
                "device": device,
                "actual_optimizer_updates": 0,
                "status": "STARTED",
            },
        )
        actor, manifest, progress, rows = _train_actor(
            method,
            max_updates,
            bank,
            features,
            p_all,
            oof_predictors,
            fold_by_index,
            cell_costs,
            seed_panel=seed_panel,
            training_seed=training_seed,
            device=device,
        )
        checkpoint = model_dir / f"actor_{method.lower()}_seed1.pt"
        torch.save(
            {
                "schema_version": 3,
                "method": method,
                "seed_panel": seed_panel,
                "training_seed": training_seed,
                "state_dict": actor.state_dict(),
                "manifest": manifest,
            },
            checkpoint,
        )
        manifest["checkpoint_path"] = checkpoint.relative_to(root).as_posix()
        manifest["checkpoint_sha256"] = sha256_file(checkpoint)
        manifests.append(manifest)
        progress_rows.extend(progress)
        episode_rows.extend(rows)
        actors[method] = actor
        _append_ledger(
            ledger,
            {
                "job": f"policy_pilot_{method}_seed1",
                "stage": "W3",
                "device": device,
                "actual_optimizer_updates": manifest["updates_completed"],
                "elapsed_seconds": time.time() - started_at,
                "status": "COMPLETED",
                "checkpoint_sha256": manifest["checkpoint_sha256"],
            },
        )
    fixed_scores = {}
    fixed_early_scores = {}
    for method in ("CENTER_FIRST", "GEOMETRY_SPREAD", "SERPENTINE", "RANDOM"):
        score, rows = evaluate_policy(
            None,
            method,
            p_all,
            bank,
            features,
            cell_costs,
            device=device,
        )
        fixed_scores[method] = score
        fixed_early_scores[method] = _domain_equal_episode_score(
            rows, "early_left_error_area_mpa"
        )
        for row in rows:
            row["seed_panel"] = 0
            row["training_seed"] = 0
        episode_rows.extend(rows)
    learned_scores = {
        str(manifest["method"]): float(manifest["validation_area_mpa"])
        for manifest in manifests
    }
    learned_early_scores = {
        str(manifest["method"]): float(manifest["validation_early_area_mpa"])
        for manifest in manifests
    }
    nonadaptive = {
        **fixed_scores,
        "LEARNED_STATIC_TRUE": learned_scores["LEARNED_STATIC_TRUE"],
    }
    best_nonadaptive = min(nonadaptive, key=lambda name: (nonadaptive[name], name))
    gate = policy_pilot_gate(
        main_area_mpa=learned_scores["VLM_SPATIAL_FEEDBACK"],
        best_nonadaptive_area_mpa=nonadaptive[best_nonadaptive],
        open_loop_area_mpa=learned_scores["VLM_SPATIAL_OPEN_LOOP"],
    )
    metrics_rows = [
        {
            "method": method,
            "role": (
                "MAIN"
                if method == "VLM_SPATIAL_FEEDBACK"
                else "STRUCTURE_DIAGNOSTIC"
                if method == "VLM_MEAN_FEEDBACK"
                else "NONADAPTIVE"
                if method in nonadaptive
                else "ABLATION"
            ),
            "validation_area_mpa": score,
            "validation_early_area_mpa": (
                fixed_early_scores[method]
                if method in fixed_early_scores
                else learned_early_scores[method]
            ),
            "best_nonadaptive": method == best_nonadaptive,
        }
        for method, score in sorted({**fixed_scores, **learned_scores}.items())
    ]
    write_csv(output / "policy_pilot_metrics.csv", metrics_rows)
    write_csv(output / "policy_training_progress.csv", progress_rows)
    write_csv(output / "policy_validation_episodes.csv", episode_rows)
    payload = {
        "status": gate.status,
        "passed": gate.passed,
        "reasons": gate.reasons,
        "main_method": "VLM_SPATIAL_FEEDBACK",
        "main_validation_area_mpa": learned_scores["VLM_SPATIAL_FEEDBACK"],
        "main_validation_early_area_mpa": learned_early_scores["VLM_SPATIAL_FEEDBACK"],
        "open_loop_validation_area_mpa": learned_scores["VLM_SPATIAL_OPEN_LOOP"],
        "no_vlm_validation_area_mpa": learned_scores["NO_VLM_SPATIAL_FEEDBACK"],
        "no_vlm_validation_early_area_mpa": learned_early_scores[
            "NO_VLM_SPATIAL_FEEDBACK"
        ],
        "mean_feedback_validation_area_mpa": learned_scores["VLM_MEAN_FEEDBACK"],
        "best_nonadaptive": best_nonadaptive,
        "best_nonadaptive_validation_area_mpa": nonadaptive[best_nonadaptive],
        "main_minus_best_nonadaptive_effect_mpa": nonadaptive[best_nonadaptive]
        - learned_scores["VLM_SPATIAL_FEEDBACK"],
        "feedback_effect_mpa": learned_scores["VLM_SPATIAL_OPEN_LOOP"]
        - learned_scores["VLM_SPATIAL_FEEDBACK"],
        "early_vlm_effect_mpa": learned_early_scores["NO_VLM_SPATIAL_FEEDBACK"]
        - learned_early_scores["VLM_SPATIAL_FEEDBACK"],
        "actor_manifests": manifests,
        "actual_optimizer_updates": sum(
            int(manifest["updates_completed"]) for manifest in manifests
        ),
        "test_accessed": False,
    }
    write_json(output / "policy_pilot_gate.json", payload)
    return payload


def run_policy_expansion(*, project_root: str | Path, device: str) -> dict[str, object]:
    """Train only the pre-authorized seed-2/3 policies after the W3 gate."""

    root = Path(project_root).resolve(strict=True)
    output = root / "results/cai_agent_v3/new_protocol"
    pilot_gate = json.loads(
        (output / "policy_pilot_gate.json").read_text(encoding="utf-8")
    )
    review_path = root / "artifacts/cai_agent_v3/requirements_review_w3.json"
    review = (
        json.loads(review_path.read_text(encoding="utf-8"))
        if review_path.is_file()
        else {}
    )
    if pilot_gate.get("status") != "POLICY_PILOT_SUPPORTED":
        payload = {
            "status": "NOT_EXECUTED_POLICY_PILOT_NOT_SUPPORTED",
            "actual_optimizer_updates": 0,
            "test_accessed": False,
        }
        write_json(output / "policy_expansion.json", payload)
        return payload
    if review.get("status") != "CORE_REVIEW_PASS":
        raise ValueError("policy expansion requires the completed W3 core review")
    ledger = root / "results/cai_agent_v3/compute_ledger.jsonl"
    used_upper_bound = _optimizer_update_upper_bound(ledger)
    remaining = _MAX_TOTAL_OPTIMIZER_UPDATES - used_upper_bound
    if remaining < _POLICY_EXPANSION_MAX_UPDATES:
        payload = {
            "status": "RESOURCE_LIMITED",
            "actual_optimizer_updates": 0,
            "optimizer_updates_used_upper_bound": used_upper_bound,
            "optimizer_update_limit": _MAX_TOTAL_OPTIMIZER_UPDATES,
            "planned_expansion_updates": _POLICY_EXPANSION_MAX_UPDATES,
            "optimizer_updates_remaining": remaining,
            "test_accessed": False,
        }
        write_json(output / "policy_expansion.json", payload)
        _append_ledger(
            ledger,
            {
                "job": "policy_seed_expansion",
                "stage": "W5",
                "actual_optimizer_updates": 0,
                "status": "RESOURCE_LIMITED",
            },
        )
        return payload
    bank = load_feature_bank(project_root=root)
    features = _VLMFeatures(bank, output / "vlm_actor_features_fit.csv")
    predictor_gate = json.loads(
        (output / "predictor_gate.json").read_text(encoding="utf-8")
    )
    p_all_manifest = next(
        manifest
        for manifest in predictor_gate["candidate_manifests"]
        if manifest["model"] == predictor_gate["selected_p_all"]
    )
    p_all, _ = load_predictor_checkpoint(
        root / p_all_manifest["checkpoint_path"], device=device
    )
    for parameter in p_all.parameters():
        parameter.requires_grad_(False)
    oof_predictors, fold_by_index = _load_oof_predictors(root, device=device)
    cell_costs = _cell_costs(bank)
    model_dir = output / "models"
    progress_rows: list[dict[str, object]] = list(
        read_csv(output / "policy_training_progress.csv")
    )
    episode_rows: list[dict[str, object]] = list(
        read_csv(output / "policy_validation_episodes.csv")
    )
    manifests = []
    specs = (
        ("VLM_SPATIAL_FEEDBACK", 1250),
        ("NO_VLM_SPATIAL_FEEDBACK", 1250),
        ("VLM_SPATIAL_OPEN_LOOP", 1250),
        ("LEARNED_STATIC_TRUE", 750),
    )
    for seed_panel in (2, 3):
        for offset, (method, max_updates) in enumerate(specs):
            training_seed = 2026091300 + seed_panel * 10 + offset
            job = f"policy_expansion_{method}_seed{seed_panel}"
            started = time.perf_counter()
            _append_ledger(
                ledger,
                {
                    "job": job,
                    "stage": "W5",
                    "device": device,
                    "actual_optimizer_updates": 0,
                    "status": "STARTED",
                },
            )
            actor, manifest, progress, rows = _train_actor(
                method,
                max_updates,
                bank,
                features,
                p_all,
                oof_predictors,
                fold_by_index,
                cell_costs,
                seed_panel=seed_panel,
                training_seed=training_seed,
                device=device,
            )
            checkpoint = model_dir / f"actor_{method.lower()}_seed{seed_panel}.pt"
            torch.save(
                {
                    "schema_version": 3,
                    "method": method,
                    "seed_panel": seed_panel,
                    "training_seed": training_seed,
                    "state_dict": actor.state_dict(),
                    "manifest": manifest,
                },
                checkpoint,
            )
            manifest["checkpoint_path"] = checkpoint.relative_to(root).as_posix()
            manifest["checkpoint_sha256"] = sha256_file(checkpoint)
            manifests.append(manifest)
            progress_rows.extend(progress)
            episode_rows.extend(rows)
            _append_ledger(
                ledger,
                {
                    "job": job,
                    "stage": "W5",
                    "device": device,
                    "actual_optimizer_updates": manifest["updates_completed"],
                    "elapsed_seconds": time.perf_counter() - started,
                    "status": "COMPLETED",
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                },
            )
    write_csv(output / "policy_training_progress.csv", progress_rows)
    write_csv(output / "policy_validation_episodes.csv", episode_rows)
    nonadaptive_scores = {}
    for method in (
        "CENTER_FIRST",
        "GEOMETRY_SPREAD",
        "SERPENTINE",
        "RANDOM",
        "LEARNED_STATIC_TRUE",
    ):
        rows = [row for row in episode_rows if row["method"] == method]
        nonadaptive_scores[method] = _domain_equal_episode_score(
            rows, "left_error_area_mpa"
        )
    locked_best = min(
        nonadaptive_scores,
        key=lambda method: (nonadaptive_scores[method], method),
    )
    payload = {
        "status": "POLICY_SEEDS_1_TO_3_LOCKED",
        "expanded_seed_panels": [2, 3],
        "expanded_methods": [row[0] for row in specs],
        "new_actor_manifests": manifests,
        "seed1_actor_manifests": pilot_gate["actor_manifests"],
        "locked_best_nonadaptive": locked_best,
        "nonadaptive_validation_area_mpa": nonadaptive_scores,
        "actual_optimizer_updates": sum(
            int(manifest["updates_completed"]) for manifest in manifests
        ),
        "test_accessed": False,
    }
    write_json(output / "policy_expansion.json", payload)
    return payload


__all__ = [
    "evaluate_policy",
    "load_actor_checkpoint",
    "precheck_policy_training",
    "run_policy_expansion",
    "run_policy_pilots",
]
