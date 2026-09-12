"""One bounded GDFS-inspired frozen-predictor selector pilot."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .actor_training import (
    _actor_forward,
    _append_ledger,
    _cell_costs,
    _legal_action_mask,
    _load_oof_predictors,
    _predict_by_fold,
    _sample_specimens,
    _state_dict_sha256,
    _VLMFeatures,
    evaluate_policy,
)
from .feature_bank import V3FeatureBank, load_feature_bank
from .files import sha256_file, write_csv, write_json
from .models import SpatialCAIActor
from .policy import vlm_first_action_mask
from .predictor_training import load_predictor_checkpoint

_METHOD = "GDFS_ADAPTED_FROZEN_PREDICTOR_S1"
_ENDPOINT_BUDGET = 0.25
_MAX_UPDATES = 1250
_VALIDATION_INTERVAL = 250
_PATIENCE = 4
_BATCH_SIZE = 16
_CONCRETE_GAMMA = 0.2


def concrete_group_selection(
    logits: torch.Tensor, *, temperature: float, deterministic: bool
) -> torch.Tensor:
    """Return one differentiable 64-group Concrete selection per episode."""

    value = float(temperature)
    if logits.ndim != 2 or logits.shape[1] != 64 or not 0.0 < value <= 1.0:
        raise ValueError("Concrete grouped selection input is invalid")
    if deterministic:
        return torch.softmax(logits / (_CONCRETE_GAMMA * value), dim=-1)
    distribution = torch.distributions.RelaxedOneHotCategorical(
        torch.tensor(value, dtype=logits.dtype, device=logits.device),
        logits=logits / _CONCRETE_GAMMA,
    )
    return distribution.rsample()


def _predict_soft_by_fold(
    predictors: dict[int, nn.Module],
    folds: np.ndarray,
    surface: torch.Tensor,
    cscan: torch.Tensor,
    measured_weight: torch.Tensor,
    costs: torch.Tensor,
) -> torch.Tensor:
    predictions = torch.zeros(len(folds), dtype=surface.dtype, device=surface.device)
    for fold, predictor in predictors.items():
        rows = np.flatnonzero(folds == fold)
        if len(rows) == 0:
            continue
        indices = torch.from_numpy(rows).to(surface.device)
        prediction = predictor.forward_soft(
            surface[indices],
            cscan[indices],
            measured_weight[indices],
            cost=costs[indices],
        )
        predictions = predictions.index_copy(0, indices, prediction)
    return predictions


def _gdfs_training_loss(
    actor: nn.Module,
    bank: V3FeatureBank,
    features: _VLMFeatures,
    predictors: dict[int, nn.Module],
    fold_by_index: np.ndarray,
    cell_costs: np.ndarray,
    specimen_indices: np.ndarray,
    *,
    temperature: float,
    target_scale: float,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = len(specimen_indices)
    surface = torch.from_numpy(bank.surface_tokens[specimen_indices]).to(device)
    cscan = torch.from_numpy(bank.cscan_tokens[specimen_indices]).to(device)
    indicator = torch.from_numpy(features.indicator[specimen_indices]).to(device)
    confidence = torch.from_numpy(features.confidence[specimen_indices]).to(device)
    available = torch.from_numpy(features.available[specimen_indices]).to(device)
    no_reliable = torch.from_numpy(features.no_reliable[specimen_indices]).to(device)
    per_cell_cost_exact = cell_costs[specimen_indices]
    per_cell_cost = torch.from_numpy(per_cell_cost_exact).to(
        device=device, dtype=torch.float32
    )
    folds = fold_by_index[specimen_indices]
    targets = torch.from_numpy(bank.targets_mpa[specimen_indices]).to(device)
    measured = torch.zeros((batch, 64), dtype=torch.bool, device=device)
    history = torch.zeros((batch, 64), dtype=torch.float32, device=device)
    costs = torch.zeros(batch, dtype=torch.float32, device=device)
    exact_costs = np.zeros(batch, dtype=np.float64)
    action_count = torch.zeros(batch, dtype=torch.int64, device=device)
    current_prediction = _predict_by_fold(
        predictors, folds, surface, cscan, measured, costs
    )
    episode_losses: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    while True:
        legal = torch.from_numpy(
            _legal_action_mask(
                measured.detach().cpu().numpy(),
                exact_costs,
                per_cell_cost_exact,
            )
        ).to(device)
        active_rows = torch.nonzero(legal.any(dim=1), as_tuple=False).flatten()
        if active_rows.numel() == 0:
            break
        proposal, _ = vlm_first_action_mask(
            legal=legal,
            use_vlm=True,
            action_count=action_count,
            indicator=indicator,
            confidence=confidence,
            available=available,
            no_reliable=no_reliable,
        )
        scores, _ = _actor_forward(
            actor,
            _METHOD,
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
        active_scores = scores[active_rows].masked_fill(
            ~proposal[active_rows], -torch.inf
        )
        soft_active = concrete_group_selection(
            active_scores, temperature=temperature, deterministic=False
        )
        soft_selection = torch.zeros_like(scores).index_copy(
            0, active_rows, soft_active
        )
        soft_mask = torch.maximum(measured.to(surface.dtype), soft_selection)
        soft_costs = costs + torch.sum(soft_selection * per_cell_cost, dim=1)
        soft_predictions = _predict_soft_by_fold(
            predictors, folds, surface, cscan, soft_mask, soft_costs
        )
        normalized_error = (soft_predictions - targets) / target_scale
        step_loss = F.smooth_l1_loss(
            normalized_error,
            torch.zeros_like(normalized_error),
            beta=1.0,
            reduction="none",
        )
        for row in active_rows.tolist():
            episode_losses[row].append(step_loss[row])

        actions = soft_active.argmax(dim=1)
        measured = measured.clone()
        history = history.clone()
        costs = costs.clone()
        for offset, row in enumerate(active_rows.tolist()):
            cell = int(actions[offset])
            measured[row, cell] = True
            action_count[row] += 1
            history[row, cell] = action_count[row].to(torch.float32) / 64.0
            exact_costs[row] = float(
                np.sum(per_cell_cost_exact[row, measured[row].cpu().numpy()])
            )
            costs[row] = exact_costs[row]
        current_prediction = _predict_by_fold(
            predictors, folds, surface, cscan, measured, costs
        )
    if any(not losses for losses in episode_losses):
        raise ValueError("GDFS training produced an empty episode")
    per_episode = torch.stack([torch.stack(losses).mean() for losses in episode_losses])
    loss = per_episode.mean()
    return loss, {
        "temperature": temperature,
        "mean_proxy_huber": float(loss.detach()),
        "mean_action_count": float(np.mean([len(losses) for losses in episode_losses])),
    }


def _validation_soft_hard_proxy(
    actor: nn.Module,
    predictor: nn.Module,
    bank: V3FeatureBank,
    features: _VLMFeatures,
    cell_costs: np.ndarray,
    *,
    temperature: float,
    target_scale: float,
    device: str,
) -> tuple[float, float]:
    soft_losses = []
    hard_losses = []
    for specimen in bank.indices("VALID"):
        index = int(specimen)
        surface = torch.from_numpy(bank.surface_tokens[index : index + 1]).to(device)
        cscan = torch.from_numpy(bank.cscan_tokens[index : index + 1]).to(device)
        indicator = torch.from_numpy(features.indicator[index : index + 1]).to(device)
        confidence = torch.from_numpy(features.confidence[index : index + 1]).to(device)
        available = torch.from_numpy(features.available[index : index + 1]).to(device)
        no_reliable = torch.from_numpy(features.no_reliable[index : index + 1]).to(
            device
        )
        per_cell_cost_exact = cell_costs[index : index + 1]
        per_cell_cost = torch.from_numpy(per_cell_cost_exact).to(
            device=device, dtype=torch.float32
        )
        measured = torch.zeros((1, 64), dtype=torch.bool, device=device)
        history = torch.zeros((1, 64), dtype=torch.float32, device=device)
        costs = torch.zeros(1, dtype=torch.float32, device=device)
        exact_costs = np.zeros(1, dtype=np.float64)
        action_count = torch.zeros(1, dtype=torch.int64, device=device)
        target = torch.tensor(
            [bank.targets_mpa[index]], dtype=torch.float32, device=device
        )
        with torch.inference_mode():
            current_prediction = predictor(surface, cscan, measured, cost=costs)
            episode_soft = []
            episode_hard = []
            while True:
                legal = torch.from_numpy(
                    _legal_action_mask(
                        measured.cpu().numpy(), exact_costs, per_cell_cost_exact
                    )
                ).to(device)
                if not bool(legal.any()):
                    break
                proposal, _ = vlm_first_action_mask(
                    legal=legal,
                    use_vlm=True,
                    action_count=action_count,
                    indicator=indicator,
                    confidence=confidence,
                    available=available,
                    no_reliable=no_reliable,
                )
                scores, _ = _actor_forward(
                    actor,
                    _METHOD,
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
                masked_scores = scores.masked_fill(~proposal, -torch.inf)
                soft = concrete_group_selection(
                    masked_scores, temperature=temperature, deterministic=True
                )
                soft_mask = torch.maximum(measured.to(surface.dtype), soft)
                soft_cost = costs + torch.sum(soft * per_cell_cost, dim=1)
                soft_prediction = predictor.forward_soft(
                    surface, cscan, soft_mask, cost=soft_cost
                )
                episode_soft.append(
                    F.smooth_l1_loss(
                        (soft_prediction - target) / target_scale,
                        torch.zeros_like(target),
                        beta=1.0,
                    )
                )
                cell = int(soft.argmax(dim=1)[0])
                measured[0, cell] = True
                action_count[0] += 1
                history[0, cell] = action_count[0].to(torch.float32) / 64.0
                exact_costs[0] = float(
                    np.sum(per_cell_cost_exact[0, measured[0].cpu().numpy()])
                )
                costs[0] = exact_costs[0]
                current_prediction = predictor(surface, cscan, measured, cost=costs)
                episode_hard.append(
                    F.smooth_l1_loss(
                        (current_prediction - target) / target_scale,
                        torch.zeros_like(target),
                        beta=1.0,
                    )
                )
        soft_losses.append(float(torch.stack(episode_soft).mean()))
        hard_losses.append(float(torch.stack(episode_hard).mean()))
    return float(np.mean(soft_losses)), float(np.mean(hard_losses))


def _train_gdfs(
    bank: V3FeatureBank,
    features: _VLMFeatures,
    p_all: nn.Module,
    oof_predictors: dict[int, nn.Module],
    fold_by_index: np.ndarray,
    cell_costs: np.ndarray,
    *,
    max_updates: int,
    device: str,
) -> tuple[
    nn.Module, dict[str, object], list[dict[str, object]], list[dict[str, object]]
]:
    train_targets = bank.targets_mpa[bank.indices("TRAIN")]
    target_mean = float(np.mean(train_targets))
    target_scale = max(float(np.std(train_targets)), 1.0)
    actor = SpatialCAIActor(
        use_vlm=True,
        use_feedback=True,
        target_mean=target_mean,
        target_scale=target_scale,
    ).to(device)
    training_seed = 2026091401
    rng = np.random.default_rng(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-4)
    predictor_hashes_before = {
        f"oof_fold{fold}": _state_dict_sha256(predictor.state_dict())
        for fold, predictor in oof_predictors.items()
    }
    predictor_hashes_before["p_all"] = _state_dict_sha256(p_all.state_dict())
    best_score = math.inf
    best_state = None
    best_update = 0
    best_rows: list[dict[str, object]] = []
    stale = 0
    progress = []
    started = time.perf_counter()
    for update in range(1, max_updates + 1):
        temperature = 0.1 ** ((update - 1) / max(max_updates - 1, 1))
        indices = _sample_specimens(rng, bank, batch_size=_BATCH_SIZE)
        loss, terms = _gdfs_training_loss(
            actor,
            bank,
            features,
            oof_predictors,
            fold_by_index,
            cell_costs,
            indices,
            temperature=temperature,
            target_scale=target_scale,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        if update % _VALIDATION_INTERVAL == 0 or update == max_updates:
            actor.eval()
            hard_area, rows = evaluate_policy(
                actor,
                _METHOD,
                p_all,
                bank,
                features,
                cell_costs,
                device=device,
            )
            soft_huber, hard_huber = _validation_soft_hard_proxy(
                actor,
                p_all,
                bank,
                features,
                cell_costs,
                temperature=temperature,
                target_scale=target_scale,
                device=device,
            )
            actor.train()
            progress.append(
                {
                    "method": _METHOD,
                    "seed_panel": 1,
                    "training_seed": training_seed,
                    "update": update,
                    "temperature": temperature,
                    "validation_hard_area_mpa": hard_area,
                    "validation_soft_proxy_huber": soft_huber,
                    "validation_hard_proxy_huber": hard_huber,
                    "soft_minus_hard_proxy_huber": soft_huber - hard_huber,
                    **terms,
                }
            )
            if hard_area < best_score - 1e-12:
                best_score = hard_area
                best_state = copy.deepcopy(actor.state_dict())
                best_update = update
                best_rows = rows
                stale = 0
            else:
                stale += 1
            if stale >= _PATIENCE:
                break
    if best_state is None:
        raise ValueError("GDFS pilot produced no checkpoint")
    actor.load_state_dict(best_state)
    actor.eval()
    predictor_hashes_after = {
        f"oof_fold{fold}": _state_dict_sha256(predictor.state_dict())
        for fold, predictor in oof_predictors.items()
    }
    predictor_hashes_after["p_all"] = _state_dict_sha256(p_all.state_dict())
    manifest = {
        "method": _METHOD,
        "adaptation_identity": "FROZEN_PREDICTOR_VLM_COST_GROUPED_CONCRETE_GREEDY_RELAXATION",
        "source_commit": "e2b6f7403fdac4d217ac2ec5dea96acd60240b60",
        "seed_panel": 1,
        "training_seed": training_seed,
        "updates_completed": update,
        "selected_update": best_update,
        "validation_hard_area_mpa": best_score,
        "parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
        "state_dict_sha256": _state_dict_sha256(actor.state_dict()),
        "predictor_hashes_before": predictor_hashes_before,
        "predictor_hashes_after": predictor_hashes_after,
        "predictor_parameters_unchanged": predictor_hashes_before
        == predictor_hashes_after,
        "selector_observation": "hard-visible only",
        "training_relaxation": "full hidden C-scan tokens only through one 64x1 Concrete group mask",
        "validation_acquisition": "hard",
        "temperature_schedule": "geometric 1.0 to 0.1",
        "concrete_gamma": _CONCRETE_GAMMA,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return actor, manifest, progress, best_rows


def _load_inputs(
    root: Path, *, device: str
) -> tuple[
    Path,
    V3FeatureBank,
    _VLMFeatures,
    nn.Module,
    dict[int, nn.Module],
    np.ndarray,
    np.ndarray,
]:
    output = root / "results/cai_agent_v3/new_protocol"
    policy_gate = json.loads((output / "policy_pilot_gate.json").read_text())
    if policy_gate.get("status") not in {
        "POLICY_PILOT_SUPPORTED",
        "POLICY_PILOT_NOT_SUPPORTED",
    }:
        raise ValueError("GDFS requires a completed W3 pilot")
    predictor_gate = json.loads((output / "predictor_gate.json").read_text())
    if predictor_gate.get("status") != "PREDICTOR_READY":
        raise ValueError("GDFS requires a ready common predictor")
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
    for predictor in oof_predictors.values():
        if not hasattr(predictor, "forward_soft"):
            raise ValueError("selected reward predictor lacks the soft-mask adapter")
    return (
        output,
        bank,
        features,
        p_all,
        oof_predictors,
        fold_by_index,
        _cell_costs(bank),
    )


def precheck_gdfs_training(
    *, project_root: str | Path, device: str
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    output, bank, features, p_all, oof_predictors, folds, costs = _load_inputs(
        root, device=device
    )
    actor, manifest, progress, rows = _train_gdfs(
        bank,
        features,
        p_all,
        oof_predictors,
        folds,
        costs,
        max_updates=1,
        device=device,
    )
    del actor
    payload = {
        "status": "GDFS_INTERFACE_PRECHECK_PASSED",
        "actual_optimizer_updates": 1,
        "predictor_parameters_unchanged": manifest["predictor_parameters_unchanged"],
        "progress_rows": len(progress),
        "validation_episode_rows": len(rows),
    }
    write_json(output / "gdfs_precheck.json", payload)
    _append_ledger(
        root / "results/cai_agent_v3/compute_ledger.jsonl",
        {
            "job": "gdfs_interface_precheck",
            "stage": "W4_PRECHECK",
            "device": device,
            "actual_optimizer_updates": 1,
            "status": "COMPLETED",
        },
    )
    return payload


def run_gdfs_pilot(*, project_root: str | Path, device: str) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    output, bank, features, p_all, oof_predictors, folds, costs = _load_inputs(
        root, device=device
    )
    ledger = root / "results/cai_agent_v3/compute_ledger.jsonl"
    _append_ledger(
        ledger,
        {
            "job": "gdfs_adapted_frozen_predictor_seed1",
            "stage": "W4",
            "device": device,
            "actual_optimizer_updates": 0,
            "status": "STARTED",
        },
    )
    started = time.perf_counter()
    actor, manifest, progress, rows = _train_gdfs(
        bank,
        features,
        p_all,
        oof_predictors,
        folds,
        costs,
        max_updates=_MAX_UPDATES,
        device=device,
    )
    checkpoint = output / "models/gdfs_adapted_frozen_predictor_seed1.pt"
    torch.save(
        {
            "schema_version": 3,
            "method": _METHOD,
            "state_dict": actor.state_dict(),
            "manifest": manifest,
        },
        checkpoint,
    )
    manifest["checkpoint_path"] = checkpoint.relative_to(root).as_posix()
    manifest["checkpoint_sha256"] = sha256_file(checkpoint)
    write_csv(output / "gdfs_training_progress.csv", progress)
    write_csv(output / "gdfs_validation_episodes.csv", rows)
    payload = {
        "status": (
            "GDFS_ADAPTER_COMPLETE"
            if manifest["predictor_parameters_unchanged"]
            else "GDFS_ADAPTER_INCOMPLETE"
        ),
        "manifest": manifest,
        "actual_optimizer_updates": manifest["updates_completed"],
        "test_accessed": False,
        "scientific_role": "learning-method diagnostic outside the three main comparisons",
    }
    write_json(output / "gdfs_pilot.json", payload)
    _append_ledger(
        ledger,
        {
            "job": "gdfs_adapted_frozen_predictor_seed1",
            "stage": "W4",
            "device": device,
            "actual_optimizer_updates": manifest["updates_completed"],
            "elapsed_seconds": time.perf_counter() - started,
            "status": payload["status"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
        },
    )
    return payload


__all__ = [
    "concrete_group_selection",
    "precheck_gdfs_training",
    "run_gdfs_pilot",
]
