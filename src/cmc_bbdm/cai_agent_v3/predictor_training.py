"""Fixed v3 Ridge diagnostics and finite neural predictor comparison."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from cmc_bbdm.cai_active_image.contracts import Method
from cmc_bbdm.cai_active_image.environment import NativeCellGrid
from cmc_bbdm.cai_active_image.policies import fixed_action_order

from .feature_bank import V3FeatureBank, load_feature_bank
from .files import read_csv, sha256_file, write_csv, write_json
from .gates import choose_common_predictor
from .metrics import left_error_area_mpa
from .models import MeanSCPredictor, SpatialPredictor

_ENDPOINT_BUDGET = 0.25
_VALIDATION_INTERVAL = 250
_VALIDATION_PATIENCE = 4
_CANDIDATE_UPDATES = 2000
_BATCH_SIZE = 32
_MODEL_SEEDS = {
    "MEAN_SC": 2026091201,
    "SPATIAL_SC": 2026091202,
    "SPATIAL_C": 2026091203,
}
_ROUTES = ("CENTER_FIRST", "GEOMETRY_SPREAD", "SERPENTINE", "RANDOM")


@dataclass(frozen=True, slots=True)
class ValidationLibrary:
    specimen_indices: np.ndarray
    route_names: tuple[str, ...]
    state_indices: np.ndarray
    masks: np.ndarray
    costs: np.ndarray


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _append_ledger(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _cell_costs(bank: V3FeatureBank) -> np.ndarray:
    return np.asarray(
        [
            [
                cell.cost
                for cell in NativeCellGrid.from_shape(
                    tuple(int(value) for value in shape)
                ).cells
            ]
            for shape in bank.native_shapes
        ],
        dtype=np.float64,
    )


def _random_order(specimen_key: str) -> tuple[int, ...]:
    seed = int.from_bytes(
        hashlib.sha256(f"cai-agent-v3-valid-random|{specimen_key}".encode()).digest()[
            :8
        ],
        "big",
    )
    return fixed_action_order(Method.RANDOM, seed=seed)


def _route_order(route: str, specimen_key: str) -> tuple[int, ...]:
    if route == "RANDOM":
        return _random_order(specimen_key)
    return fixed_action_order(Method(route), seed=0)


def build_prefix_library(
    bank: V3FeatureBank,
    cell_costs: np.ndarray,
    specimen_subset: np.ndarray,
) -> ValidationLibrary:
    specimen_indices: list[int] = []
    route_names: list[str] = []
    state_indices: list[int] = []
    masks: list[np.ndarray] = []
    costs: list[float] = []
    for index in np.asarray(specimen_subset, dtype=np.int64):
        specimen = int(index)
        for route in _ROUTES:
            mask = np.zeros(64, dtype=bool)
            current = 0.0
            state_index = 0
            specimen_indices.append(specimen)
            route_names.append(route)
            state_indices.append(state_index)
            masks.append(mask.copy())
            costs.append(current)
            for cell in _route_order(route, bank.specimen_keys[specimen]):
                proposed = current + float(cell_costs[specimen, cell])
                if proposed > _ENDPOINT_BUDGET + 1e-12:
                    continue
                mask[cell] = True
                current = proposed
                state_index += 1
                specimen_indices.append(specimen)
                route_names.append(route)
                state_indices.append(state_index)
                masks.append(mask.copy())
                costs.append(current)
    return ValidationLibrary(
        specimen_indices=np.asarray(specimen_indices, dtype=np.int64),
        route_names=tuple(route_names),
        state_indices=np.asarray(state_indices, dtype=np.int64),
        masks=np.asarray(masks, dtype=bool),
        costs=np.asarray(costs, dtype=np.float64),
    )


def build_validation_library(
    bank: V3FeatureBank, cell_costs: np.ndarray
) -> ValidationLibrary:
    return build_prefix_library(bank, cell_costs, bank.indices("VALID"))


def _sample_specimens(
    rng: np.random.Generator,
    bank: V3FeatureBank,
    domain_indices: Mapping[str, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    domains = tuple(sorted(domain_indices))
    selected = []
    for _ in range(batch_size):
        domain = domains[int(rng.integers(0, len(domains)))]
        candidates = domain_indices[domain]
        selected.append(int(candidates[int(rng.integers(0, len(candidates)))]))
    return np.asarray(selected, dtype=np.int64)


def _sample_training_masks(rng: np.random.Generator, *, batch_size: int) -> np.ndarray:
    masks = np.zeros((batch_size, 64), dtype=bool)
    fixed = {
        "CENTER_FIRST": fixed_action_order(Method.CENTER_FIRST, seed=0),
        "GEOMETRY_SPREAD": fixed_action_order(Method.GEOMETRY_SPREAD, seed=0),
        "SERPENTINE": fixed_action_order(Method.SERPENTINE, seed=0),
    }
    routes = ("RANDOM", *fixed)
    for row in range(batch_size):
        draw = float(rng.random())
        if draw < 0.10:
            continue
        if draw < 0.70:
            count = int(rng.integers(1, 17))
            route = routes[int(rng.integers(0, len(routes)))]
            order = (
                tuple(int(value) for value in rng.permutation(64))
                if route == "RANDOM"
                else fixed[route]
            )
        elif draw < 0.90:
            count = int(rng.integers(17, 49))
            order = tuple(int(value) for value in rng.permutation(64))
        else:
            count = 64
            order = tuple(range(64))
        masks[row, list(order[:count])] = True
    return masks


def _fixed_ridge_masks(rng: np.random.Generator) -> np.ndarray:
    """Return exactly 32 fixed states spanning zero, partial and full input."""

    masks = np.zeros((32, 64), dtype=bool)
    fixed_orders = (
        fixed_action_order(Method.CENTER_FIRST, seed=0),
        fixed_action_order(Method.GEOMETRY_SPREAD, seed=0),
        fixed_action_order(Method.SERPENTINE, seed=0),
    )
    row = 3
    for small_index in range(19):
        count = 1 + small_index % 16
        if small_index % 4 == 0:
            order = tuple(int(value) for value in rng.permutation(64))
        else:
            order = fixed_orders[(small_index - 1) % len(fixed_orders)]
        masks[row, list(order[:count])] = True
        row += 1
    for _ in range(6):
        count = int(rng.integers(17, 49))
        order = rng.permutation(64)
        masks[row, order[:count]] = True
        row += 1
    masks[row:] = True
    return masks


def _predict_batches(
    model: nn.Module,
    bank: V3FeatureBank,
    specimen_indices: np.ndarray,
    masks: np.ndarray,
    costs: np.ndarray,
    *,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(specimen_indices), batch_size):
            stop = start + batch_size
            indices = specimen_indices[start:stop]
            prediction = model(
                torch.from_numpy(bank.surface_tokens[indices]).to(device),
                torch.from_numpy(bank.cscan_tokens[indices]).to(device),
                torch.from_numpy(masks[start:stop]).to(device),
                cost=torch.from_numpy(costs[start:stop]).to(
                    device=device, dtype=torch.float32
                ),
            )
            output.append(prediction.detach().cpu().numpy())
    return np.concatenate(output).astype(np.float64, copy=False)


def _constant_metrics(
    bank: V3FeatureBank, fit_indices: np.ndarray | None = None
) -> dict[str, float]:
    fit = bank.indices("TRAIN") if fit_indices is None else fit_indices
    train_targets = bank.targets_mpa[fit].astype(np.float64)
    valid_targets = bank.targets_mpa[bank.indices("VALID")].astype(np.float64)
    mean = float(np.mean(train_targets))
    median = float(np.median(train_targets))
    return {
        "train_target_mean_mpa": mean,
        "train_target_median_mpa": median,
        "train_target_std_mpa": float(np.std(train_targets)),
        "train_mean_full_mae_mpa": float(np.mean(np.abs(valid_targets - mean))),
        "train_mean_full_mse_mpa2": float(np.mean(np.square(valid_targets - mean))),
        "train_median_full_mae_mpa": float(np.mean(np.abs(valid_targets - median))),
        "train_median_full_mse_mpa2": float(np.mean(np.square(valid_targets - median))),
    }


def evaluate_predictor(
    model: nn.Module,
    bank: V3FeatureBank,
    library: ValidationLibrary,
    constants: Mapping[str, float],
    *,
    device: str,
) -> dict[str, float]:
    predictions = _predict_batches(
        model,
        bank,
        library.specimen_indices,
        library.masks,
        library.costs,
        device=device,
    )
    trajectories: dict[tuple[int, str], list[int]] = defaultdict(list)
    for row, (index, route) in enumerate(
        zip(library.specimen_indices, library.route_names, strict=True)
    ):
        trajectories[(int(index), route)].append(row)
    areas: dict[tuple[int, str], float] = {}
    endpoints: dict[tuple[int, str], float] = {}
    zero_predictions: dict[int, float] = {}
    for identity, rows in trajectories.items():
        index, _ = identity
        ordered = sorted(rows, key=lambda row: int(library.state_indices[row]))
        target = float(bank.targets_mpa[index])
        areas[identity] = left_error_area_mpa(
            library.costs[ordered], predictions[ordered], target, end=_ENDPOINT_BUDGET
        )
        endpoints[identity] = float(predictions[ordered[-1]])
        zero_predictions.setdefault(index, float(predictions[ordered[0]]))
    specimen_areas = {
        index: float(np.mean([areas[(index, route)] for route in _ROUTES]))
        for index in zero_predictions
    }
    domain_areas = [
        np.mean(
            [
                area
                for index, area in specimen_areas.items()
                if bank.dataset_ids[index] == domain
            ]
        )
        for domain in sorted({bank.dataset_ids[index] for index in specimen_areas})
    ]
    valid_indices = bank.indices("VALID")
    full_masks = np.ones((len(valid_indices), 64), dtype=bool)
    full_costs = np.ones(len(valid_indices), dtype=np.float32)
    full_predictions = _predict_batches(
        model,
        bank,
        valid_indices,
        full_masks,
        full_costs,
        device=device,
    )
    targets = bank.targets_mpa[valid_indices].astype(np.float64)
    zero_errors = [
        abs(zero_predictions[int(index)] - float(bank.targets_mpa[int(index)]))
        for index in valid_indices
    ]
    result = {
        "valid_area_mpa": float(np.mean(domain_areas)),
        "zero_mae_mpa": float(np.mean(zero_errors)),
        "full_mae_mpa": float(np.mean(np.abs(full_predictions - targets))),
        "full_mse_mpa2": float(np.mean(np.square(full_predictions - targets))),
        "full_rmse_mpa": float(np.sqrt(np.mean(np.square(full_predictions - targets)))),
        "full_r2": float(r2_score(targets, full_predictions)),
        "center_endpoint_mae_mpa": float(
            np.mean(
                [
                    abs(endpoints[(int(index), "CENTER_FIRST")] - target)
                    for index, target in zip(valid_indices, targets, strict=True)
                ]
            )
        ),
        "geometry_endpoint_mae_mpa": float(
            np.mean(
                [
                    abs(endpoints[(int(index), "GEOMETRY_SPREAD")] - target)
                    for index, target in zip(valid_indices, targets, strict=True)
                ]
            )
        ),
    }
    result.update(
        {
            name: float(constants[name])
            for name in (
                "train_median_full_mae_mpa",
                "train_mean_full_mse_mpa2",
            )
        }
    )
    return result


def _fit_projection_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    scaler = StandardScaler().fit(train_x)
    scaled = scaler.transform(train_x)
    rank = int(np.linalg.matrix_rank(scaled))
    components = min(16, len(train_y) - 1, rank)
    if components < 1:
        raise ValueError("Ridge diagnostic PCA has no valid component")
    pca = PCA(n_components=components, svd_solver="full").fit(scaled)
    ridge = Ridge(alpha=10.0).fit(
        pca.transform(scaled), train_y, sample_weight=sample_weight
    )
    return ridge.predict(pca.transform(scaler.transform(valid_x))), components


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    return {
        "mae_mpa": float(np.mean(np.abs(residual))),
        "rmse_mpa": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": float(r2_score(target, prediction)),
    }


def fit_ridge_diagnostics(
    bank: V3FeatureBank,
    library: ValidationLibrary,
    cell_costs: np.ndarray,
) -> list[dict[str, object]]:
    train = bank.indices("TRAIN")
    valid = bank.indices("VALID")
    train_y = bank.targets_mpa[train].astype(np.float64)
    valid_y = bank.targets_mpa[valid].astype(np.float64)
    rows: list[dict[str, object]] = []
    full_inputs = {
        "RIDGE_SURFACE_FULL": (
            bank.full_surface_tokens[train],
            bank.full_surface_tokens[valid],
        ),
        "RIDGE_CSCAN_FULL": (
            bank.full_cscan_tokens[train],
            bank.full_cscan_tokens[valid],
        ),
        "RIDGE_SURFACE_CSCAN_FULL": (
            np.concatenate(
                (bank.full_surface_tokens[train], bank.full_cscan_tokens[train]), axis=1
            ),
            np.concatenate(
                (bank.full_surface_tokens[valid], bank.full_cscan_tokens[valid]), axis=1
            ),
        ),
    }
    for name, (train_x, valid_x) in full_inputs.items():
        prediction, components = _fit_projection_ridge(
            train_x.astype(np.float64), train_y, valid_x.astype(np.float64)
        )
        rows.append(
            {
                "model": name,
                "condition": "FULL_INPUT_DIAGNOSTIC_ONLY",
                "pca_components": components,
                "ridge_alpha": 10.0,
                "train_physical_n": len(train),
                "train_states_per_specimen": 1,
                **_regression_metrics(valid_y, prediction),
                "valid_area_mpa": "",
            }
        )
    train_features: list[np.ndarray] = []
    train_targets: list[float] = []
    train_weights: list[float] = []
    for specimen in train:
        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(
                    f"cai-agent-v3-ridge-partial|{bank.specimen_keys[int(specimen)]}".encode()
                ).digest()[:8],
                "big",
            )
        )
        masks = _fixed_ridge_masks(rng)
        for mask in masks:
            observed = bank.cscan_tokens[int(specimen), mask]
            mean = (
                observed.mean(axis=0)
                if len(observed)
                else np.zeros(512, dtype=np.float32)
            )
            cost = float(np.sum(cell_costs[int(specimen)] * mask))
            train_features.append(
                np.concatenate(
                    (
                        bank.full_surface_tokens[int(specimen)],
                        mean,
                        mask.astype(np.float32),
                        np.asarray([cost, float(mask.any())], dtype=np.float32),
                    )
                )
            )
            train_targets.append(float(bank.targets_mpa[int(specimen)]))
            train_weights.append(1.0 / 32.0)
    valid_features = []
    for specimen, mask, cost in zip(
        library.specimen_indices, library.masks, library.costs, strict=True
    ):
        observed = bank.cscan_tokens[int(specimen), mask]
        mean = (
            observed.mean(axis=0) if len(observed) else np.zeros(512, dtype=np.float32)
        )
        valid_features.append(
            np.concatenate(
                (
                    bank.full_surface_tokens[int(specimen)],
                    mean,
                    mask.astype(np.float32),
                    np.asarray([cost, float(mask.any())], dtype=np.float32),
                )
            )
        )
    partial_prediction, components = _fit_projection_ridge(
        np.asarray(train_features, dtype=np.float64),
        np.asarray(train_targets, dtype=np.float64),
        np.asarray(valid_features, dtype=np.float64),
        sample_weight=np.asarray(train_weights, dtype=np.float64),
    )
    trajectories: dict[tuple[int, str], list[int]] = defaultdict(list)
    for row, (index, route) in enumerate(
        zip(library.specimen_indices, library.route_names, strict=True)
    ):
        trajectories[(int(index), route)].append(row)
    areas = []
    for (index, _), indices in trajectories.items():
        ordered = sorted(indices, key=lambda row: library.state_indices[row])
        areas.append(
            left_error_area_mpa(
                library.costs[ordered],
                partial_prediction[ordered],
                float(bank.targets_mpa[index]),
                end=_ENDPOINT_BUDGET,
            )
        )
    rows.append(
        {
            "model": "RIDGE_PARTIAL",
            "condition": "VISIBLE_CELL_MEAN_PLUS_FULL_SURFACE_DIAGNOSTIC",
            "pca_components": components,
            "ridge_alpha": 10.0,
            "train_physical_n": len(train),
            "train_states_per_specimen": 32,
            **_regression_metrics(
                bank.targets_mpa[library.specimen_indices].astype(np.float64),
                partial_prediction,
            ),
            "valid_area_mpa": float(np.mean(areas)),
        }
    )
    return rows


def _model(name: str, constants: Mapping[str, float]) -> nn.Module:
    kwargs = {
        "target_mean": constants["train_target_mean_mpa"],
        "target_scale": max(constants["train_target_std_mpa"], 1.0),
    }
    if name == "MEAN_SC":
        return MeanSCPredictor(**kwargs)
    if name == "SPATIAL_SC":
        return SpatialPredictor(use_surface=True, **kwargs)
    if name == "SPATIAL_C":
        return SpatialPredictor(use_surface=False, **kwargs)
    raise ValueError(f"unknown predictor candidate: {name}")


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _train_candidate(
    name: str,
    bank: V3FeatureBank,
    library: ValidationLibrary,
    cell_costs: np.ndarray,
    constants: Mapping[str, float],
    *,
    device: str,
    fit_indices: np.ndarray | None = None,
    seed_override: int | None = None,
    max_updates: int = _CANDIDATE_UPDATES,
    validation_interval: int = _VALIDATION_INTERVAL,
    validation_patience: int = _VALIDATION_PATIENCE,
) -> tuple[nn.Module, dict[str, object], list[dict[str, object]]]:
    seed = _MODEL_SEEDS[name] if seed_override is None else seed_override
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = _model(name, constants).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train = bank.indices("TRAIN") if fit_indices is None else fit_indices
    domains = {
        domain: np.asarray(
            [index for index in train if bank.dataset_ids[int(index)] == domain],
            dtype=np.int64,
        )
        for domain in sorted({bank.dataset_ids[int(index)] for index in train})
    }
    scale = max(constants["train_target_std_mpa"], 1.0)
    progress: list[dict[str, object]] = []
    best_area = math.inf
    best_update = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()
    for update in range(1, max_updates + 1):
        indices = _sample_specimens(rng, bank, domains, _BATCH_SIZE)
        masks = _sample_training_masks(rng, batch_size=_BATCH_SIZE)
        costs = np.sum(cell_costs[indices] * masks, axis=1).astype(np.float32)
        prediction = model(
            torch.from_numpy(bank.surface_tokens[indices]).to(device),
            torch.from_numpy(bank.cscan_tokens[indices]).to(device),
            torch.from_numpy(masks).to(device),
            cost=torch.from_numpy(costs).to(device),
        )
        target = torch.from_numpy(bank.targets_mpa[indices]).to(device)
        loss = nn.functional.huber_loss(
            (prediction - target) / scale,
            torch.zeros_like(prediction),
            delta=1.0,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % validation_interval == 0 or update == max_updates:
            metrics = evaluate_predictor(model, bank, library, constants, device=device)
            model.train()
            progress.append(
                {
                    "model": name,
                    "update": update,
                    "train_loss": float(loss.detach()),
                    **metrics,
                }
            )
            if metrics["valid_area_mpa"] < best_area - 1e-12:
                best_area = metrics["valid_area_mpa"]
                best_update = update
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= validation_patience:
                break
    if best_state is None:
        raise ValueError("predictor candidate produced no selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_metrics = evaluate_predictor(model, bank, library, constants, device=device)
    manifest: dict[str, object] = {
        "model": name,
        "seed": seed,
        "updates_completed": update,
        "selected_update": best_update,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "fit_physical_n": len(train),
        "fit_capture_group_n": len(
            {bank.capture_group_ids[int(index)] for index in train}
        ),
        "state_dict_sha256": _state_dict_sha256(model.state_dict()),
        "metrics": final_metrics,
        "architecture": repr(model),
    }
    return model, manifest, progress


def precheck_predictor_training(
    *, project_root: str | Path, device: str
) -> dict[str, object]:
    """Run one optimizer update per candidate to verify the formal data path."""

    root = Path(project_root).resolve(strict=True)
    bank = load_feature_bank(project_root=root)
    cell_costs = _cell_costs(bank)
    library = build_validation_library(bank, cell_costs)
    constants = _constant_metrics(bank)
    rows = []
    for offset, name in enumerate(("MEAN_SC", "SPATIAL_SC", "SPATIAL_C")):
        _, manifest, _ = _train_candidate(
            name,
            bank,
            library,
            cell_costs,
            constants,
            device=device,
            seed_override=2026091291 + offset,
            max_updates=1,
            validation_interval=1,
            validation_patience=1,
        )
        rows.append(
            {
                "model": name,
                "actual_optimizer_updates": manifest["updates_completed"],
                "finite_metrics": all(
                    math.isfinite(float(value))
                    for value in manifest["metrics"].values()
                ),
                "parameter_count": manifest["parameter_count"],
            }
        )
    payload = {
        "status": "PREDICTOR_PRECHECK_PASS",
        "device": device,
        "actual_optimizer_updates": sum(
            int(row["actual_optimizer_updates"]) for row in rows
        ),
        "models": rows,
    }
    write_json(
        root / "results/cai_agent_v3/new_protocol/predictor_precheck.json", payload
    )
    _append_ledger(
        root / "results/cai_agent_v3/compute_ledger.jsonl",
        {
            "job": "predictor_interface_precheck",
            "stage": "W2_PRECHECK",
            "device": device,
            "actual_optimizer_updates": payload["actual_optimizer_updates"],
            "status": "COMPLETED",
            "ended_at": _utc_now(),
        },
    )
    return payload


def load_predictor_checkpoint(
    path: str | Path, *, device: str
) -> tuple[nn.Module, dict[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema_version") != 3 or payload.get("model_name") not in {
        "MEAN_SC",
        "SPATIAL_SC",
        "SPATIAL_C",
    }:
        raise ValueError("v3 predictor checkpoint identity is invalid")
    name = str(payload["model_name"])
    state = payload["state_dict"]
    mean = float(state["target_mean"])
    scale = float(state["target_scale"])
    constants = {
        "train_target_mean_mpa": mean,
        "train_target_std_mpa": scale,
    }
    model = _model(name, constants).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, payload["manifest"]


def run_predictor_candidates(
    *, project_root: str | Path, device: str
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    output = root / "results/cai_agent_v3/new_protocol"
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    ledger = root / "results/cai_agent_v3/compute_ledger.jsonl"
    bank = load_feature_bank(project_root=root)
    cell_costs = _cell_costs(bank)
    library = build_validation_library(bank, cell_costs)
    constants = _constant_metrics(bank)
    np.savez_compressed(
        output / "validation_prefix_library.npz",
        specimen_indices=library.specimen_indices,
        route_names=np.asarray(library.route_names),
        state_indices=library.state_indices,
        masks=library.masks,
        costs=library.costs,
    )
    ridge_started = time.perf_counter()
    ridge_rows = fit_ridge_diagnostics(bank, library, cell_costs)
    write_csv(output / "ridge_diagnostics.csv", ridge_rows)
    _append_ledger(
        ledger,
        {
            "job": "fixed_ridge_diagnostics",
            "stage": "W2",
            "device": "cpu",
            "cpu_workers": 4,
            "actual_optimizer_updates": 0,
            "ridge_fit_count": 4,
            "elapsed_seconds": time.perf_counter() - ridge_started,
            "status": "COMPLETED",
            "recorded_at": _utc_now(),
        },
    )
    manifests = []
    all_progress: list[dict[str, object]] = []
    for name in ("MEAN_SC", "SPATIAL_SC", "SPATIAL_C"):
        job_started = _utc_now()
        _append_ledger(
            ledger,
            {
                "job": f"predictor_candidate_{name}",
                "stage": "W2",
                "device": device,
                "actual_optimizer_updates": 0,
                "status": "STARTED",
                "started_at": job_started,
            },
        )
        model, manifest, progress = _train_candidate(
            name, bank, library, cell_costs, constants, device=device
        )
        checkpoint = model_dir / f"predictor_{name.lower()}.pt"
        torch.save(
            {
                "schema_version": 3,
                "model_name": name,
                "state_dict": model.state_dict(),
                "manifest": manifest,
            },
            checkpoint,
        )
        manifest["checkpoint_path"] = checkpoint.relative_to(root).as_posix()
        manifest["checkpoint_sha256"] = sha256_file(checkpoint)
        manifests.append(manifest)
        all_progress.extend(progress)
        _append_ledger(
            ledger,
            {
                "job": f"predictor_candidate_{name}",
                "stage": "W2",
                "device": device,
                "actual_optimizer_updates": manifest["updates_completed"],
                "status": "COMPLETED",
                "started_at": job_started,
                "ended_at": _utc_now(),
                "elapsed_seconds": manifest["elapsed_seconds"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
            },
        )
    candidates = [
        (
            str(manifest["model"]),
            manifest["metrics"],
            int(manifest["parameter_count"]),
        )
        for manifest in manifests
    ]
    selected, gates = choose_common_predictor(candidates)
    comparison_rows = []
    for manifest in manifests:
        name = str(manifest["model"])
        metrics = manifest["metrics"]
        gate = gates[name]
        comparison_rows.append(
            {
                "model": name,
                "selected_as_p_all": name == selected,
                "gate_status": gate.status,
                "gate_reasons": ";".join(gate.reasons),
                "parameter_count": manifest["parameter_count"],
                "updates_completed": manifest["updates_completed"],
                "selected_update": manifest["selected_update"],
                **metrics,
                "checkpoint_path": manifest["checkpoint_path"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
            }
        )
    write_csv(output / "predictor_training_progress.csv", all_progress)
    write_csv(output / "predictor_comparison.csv", comparison_rows)
    gate_payload = {
        "status": "PREDICTOR_READY" if selected is not None else "PREDICTOR_NOT_READY",
        "selected_p_all": selected,
        "selection_rule": "MIN_VALID_PREFIX_AREA_THEN_PARAMETERS_THEN_NAME",
        "constants": constants,
        "candidate_manifests": manifests,
        "gates": {
            name: {
                "status": gate.status,
                "passed": gate.passed,
                "reasons": gate.reasons,
            }
            for name, gate in gates.items()
        },
        "validation_prefix_library_sha256": sha256_file(
            output / "validation_prefix_library.npz"
        ),
        "test_labels_or_metrics_used": False,
    }
    write_json(output / "predictor_gate.json", gate_payload)
    return gate_payload


def _oof_assignments(bank: V3FeatureBank) -> dict[int, int]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in bank.indices("TRAIN"):
        grouped[bank.dataset_ids[int(index)]][
            bank.capture_group_ids[int(index)]
        ].append(int(index))
    assignments: dict[int, int] = {}
    for domain, groups in sorted(grouped.items()):
        ordered = sorted(
            groups.items(),
            key=lambda item: (
                hashlib.sha256(f"cai-agent-v3-oof|{item[0]}".encode()).hexdigest(),
                item[0],
            ),
        )
        fold_sizes = [0, 0, 0]
        for group, members in ordered:
            fold = min(range(3), key=lambda value: (fold_sizes[value], value))
            for index in members:
                assignments[index] = fold
            fold_sizes[fold] += len(members)
        if any(size == 0 for size in fold_sizes):
            raise ValueError(f"OOF fold is empty in domain {domain}")
    return assignments


def run_oof_reward_predictors(
    *, project_root: str | Path, device: str
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    output = root / "results/cai_agent_v3/new_protocol"
    gate = json.loads((output / "predictor_gate.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PREDICTOR_READY" or not gate.get("selected_p_all"):
        payload = {
            "status": "NOT_EXECUTED_PREDICTOR_NOT_READY",
            "selected_structure": gate.get("selected_p_all"),
            "actual_optimizer_updates": 0,
        }
        write_json(output / "oof_readiness.json", payload)
        return payload
    bank = load_feature_bank(project_root=root)
    cell_costs = _cell_costs(bank)
    valid_library = build_validation_library(bank, cell_costs)
    assignments = _oof_assignments(bank)
    fold_rows = [
        {
            "specimen_key": bank.specimen_keys[index],
            "dataset_id": bank.dataset_ids[index],
            "capture_group_id": bank.capture_group_ids[index],
            "fold": fold,
        }
        for index, fold in sorted(assignments.items())
    ]
    write_csv(output / "oof_fold_manifest.csv", fold_rows)
    selected = str(gate["selected_p_all"])
    model_dir = output / "models"
    ledger = root / "results/cai_agent_v3/compute_ledger.jsonl"
    manifests = []
    progress_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    all_ready = True
    train_indices = bank.indices("TRAIN")
    for fold in range(3):
        query = np.asarray(
            [index for index in train_indices if assignments[int(index)] == fold],
            dtype=np.int64,
        )
        fit = np.asarray(
            [index for index in train_indices if assignments[int(index)] != fold],
            dtype=np.int64,
        )
        fit_groups = {bank.capture_group_ids[int(index)] for index in fit}
        query_groups = {bank.capture_group_ids[int(index)] for index in query}
        if fit_groups & query_groups:
            raise ValueError("OOF capture group leaks between fit and query")
        constants = _constant_metrics(bank, fit)
        started_at = _utc_now()
        job = f"oof_reward_predictor_{selected}_fold{fold}"
        _append_ledger(
            ledger,
            {
                "job": job,
                "stage": "W2",
                "device": device,
                "actual_optimizer_updates": 0,
                "status": "STARTED",
                "started_at": started_at,
            },
        )
        model, manifest, progress = _train_candidate(
            selected,
            bank,
            valid_library,
            cell_costs,
            constants,
            device=device,
            fit_indices=fit,
            seed_override=2026091211 + fold,
        )
        checkpoint = model_dir / f"reward_predictor_{selected.lower()}_fold{fold}.pt"
        torch.save(
            {
                "schema_version": 3,
                "model_name": selected,
                "state_dict": model.state_dict(),
                "manifest": manifest,
                "query_fold": fold,
            },
            checkpoint,
        )
        manifest.update(
            {
                "fold": fold,
                "fit_capture_group_n": len(fit_groups),
                "query_capture_group_n": len(query_groups),
                "query_physical_n": len(query),
                "fit_domain_n": len({bank.dataset_ids[int(index)] for index in fit}),
                "query_domain_n": len(
                    {bank.dataset_ids[int(index)] for index in query}
                ),
                "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        readiness = choose_common_predictor(
            [(selected, manifest["metrics"], int(manifest["parameter_count"]))]
        )[1][selected]
        manifest["readiness_status"] = readiness.status
        manifest["readiness_reasons"] = readiness.reasons
        all_ready &= readiness.passed
        manifests.append(manifest)
        for row in progress:
            progress_rows.append({"fold": fold, **row})
        query_library = build_prefix_library(bank, cell_costs, query)
        predictions = _predict_batches(
            model,
            bank,
            query_library.specimen_indices,
            query_library.masks,
            query_library.costs,
            device=device,
        )
        for row_index, prediction in enumerate(predictions):
            specimen = int(query_library.specimen_indices[row_index])
            target = float(bank.targets_mpa[specimen])
            state_rows.append(
                {
                    "fold": fold,
                    "specimen_key": bank.specimen_keys[specimen],
                    "dataset_id": bank.dataset_ids[specimen],
                    "capture_group_id": bank.capture_group_ids[specimen],
                    "route": query_library.route_names[row_index],
                    "state_index": int(query_library.state_indices[row_index]),
                    "cost": float(query_library.costs[row_index]),
                    "prediction_mpa": float(prediction),
                    "target_mpa": target,
                    "absolute_error_mpa": abs(float(prediction) - target),
                    "state_role": "PREFIX",
                }
            )
        full_predictions = _predict_batches(
            model,
            bank,
            query,
            np.ones((len(query), 64), dtype=bool),
            np.ones(len(query), dtype=np.float32),
            device=device,
        )
        for specimen, prediction in zip(query, full_predictions, strict=True):
            target = float(bank.targets_mpa[int(specimen)])
            state_rows.append(
                {
                    "fold": fold,
                    "specimen_key": bank.specimen_keys[int(specimen)],
                    "dataset_id": bank.dataset_ids[int(specimen)],
                    "capture_group_id": bank.capture_group_ids[int(specimen)],
                    "route": "FULL_64",
                    "state_index": 64,
                    "cost": 1.0,
                    "prediction_mpa": float(prediction),
                    "target_mpa": target,
                    "absolute_error_mpa": abs(float(prediction) - target),
                    "state_role": "FULL_INPUT_DIAGNOSTIC",
                }
            )
        _append_ledger(
            ledger,
            {
                "job": job,
                "stage": "W2",
                "device": device,
                "actual_optimizer_updates": manifest["updates_completed"],
                "status": "COMPLETED",
                "started_at": started_at,
                "ended_at": _utc_now(),
                "elapsed_seconds": manifest["elapsed_seconds"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
            },
        )
    write_csv(output / "oof_predictor_training_progress.csv", progress_rows)
    write_csv(output / "oof_state_predictions.csv", state_rows)
    payload = {
        "status": "REWARD_MODELS_READY" if all_ready else "REWARD_MODELS_NOT_READY",
        "selected_structure": selected,
        "fold_manifests": manifests,
        "group_isolation": True,
        "query_physical_n": sum(int(row["query_physical_n"]) for row in manifests),
        "actual_optimizer_updates": sum(
            int(row["updates_completed"]) for row in manifests
        ),
        "test_labels_or_metrics_used": False,
    }
    write_json(output / "oof_readiness.json", payload)
    return payload


def refresh_cost_precision_evaluations(
    *, project_root: str | Path, device: str
) -> dict[str, object]:
    """Re-evaluate frozen W2 checkpoints with float64 native-raster costs."""

    root = Path(project_root).resolve(strict=True)
    output = root / "results/cai_agent_v3/new_protocol"
    bank = load_feature_bank(project_root=root)
    cell_costs = _cell_costs(bank)
    exact_library = build_validation_library(bank, cell_costs)
    stored_library_path = output / "validation_prefix_library.npz"
    with np.load(stored_library_path, allow_pickle=False) as stored:
        stored_masks = np.asarray(stored["masks"])
        stored_costs = np.asarray(stored["costs"])
    if stored_masks.shape != exact_library.masks.shape:
        raise ValueError("validation library shape changed during precision refresh")
    changed_rows = np.any(stored_masks != exact_library.masks, axis=1)
    np.savez_compressed(
        stored_library_path,
        specimen_indices=exact_library.specimen_indices,
        route_names=np.asarray(exact_library.route_names),
        state_indices=exact_library.state_indices,
        masks=exact_library.masks,
        costs=exact_library.costs,
    )
    constants = _constant_metrics(bank)
    gate_path = output / "predictor_gate.json"
    predictor_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    candidate_deltas = []
    candidates = []
    comparison_rows = []
    for manifest in predictor_gate["candidate_manifests"]:
        previous = dict(manifest["metrics"])
        model, _ = load_predictor_checkpoint(
            root / manifest["checkpoint_path"], device=device
        )
        metrics = evaluate_predictor(
            model, bank, exact_library, constants, device=device
        )
        manifest["metrics"] = metrics
        candidate_deltas.append(
            {
                "model": manifest["model"],
                "old_valid_area_mpa": previous["valid_area_mpa"],
                "exact_valid_area_mpa": metrics["valid_area_mpa"],
                "valid_area_delta_mpa": metrics["valid_area_mpa"]
                - previous["valid_area_mpa"],
                "old_full_mae_mpa": previous["full_mae_mpa"],
                "exact_full_mae_mpa": metrics["full_mae_mpa"],
            }
        )
        candidates.append(
            (str(manifest["model"]), metrics, int(manifest["parameter_count"]))
        )
    selected, gate_results = choose_common_predictor(candidates)
    for manifest in predictor_gate["candidate_manifests"]:
        name = str(manifest["model"])
        result = gate_results[name]
        metrics = manifest["metrics"]
        comparison_rows.append(
            {
                "model": name,
                "selected_as_p_all": name == selected,
                "gate_status": result.status,
                "gate_reasons": ";".join(result.reasons),
                "parameter_count": manifest["parameter_count"],
                "updates_completed": manifest["updates_completed"],
                "selected_update": manifest["selected_update"],
                **metrics,
                "checkpoint_path": manifest["checkpoint_path"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
            }
        )
    previous_selected = predictor_gate["selected_p_all"]
    predictor_gate["status"] = (
        "PREDICTOR_READY" if selected is not None else "PREDICTOR_NOT_READY"
    )
    predictor_gate["selected_p_all"] = selected
    predictor_gate["gates"] = {
        name: {
            "status": result.status,
            "passed": result.passed,
            "reasons": result.reasons,
        }
        for name, result in gate_results.items()
    }
    predictor_gate["validation_prefix_library_sha256"] = sha256_file(
        stored_library_path
    )
    predictor_gate["cost_precision"] = "FLOAT64_NATIVE_PIXEL_FRACTIONS"
    predictor_gate["checkpoint_selection_history"] = (
        "UPDATE_SELECTED_WITH_FLOAT32_COSTS; FROZEN_CHECKPOINT_REEVALUATED_WITH_FLOAT64_COSTS"
    )
    write_csv(output / "predictor_comparison.csv", comparison_rows)
    write_json(gate_path, predictor_gate)

    oof_path = output / "oof_readiness.json"
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    key_index = {key: index for index, key in enumerate(bank.specimen_keys)}
    query_fold = {
        key_index[row["specimen_key"]]: int(row["fold"])
        for row in read_csv(output / "oof_fold_manifest.csv")
    }
    oof_deltas = []
    all_oof_ready = True
    for manifest in oof["fold_manifests"]:
        previous = dict(manifest["metrics"])
        model, _ = load_predictor_checkpoint(
            root / manifest["checkpoint_path"], device=device
        )
        fit_indices = np.asarray(
            [
                int(index)
                for index in bank.indices("TRAIN")
                if query_fold[int(index)] != int(manifest["fold"])
            ],
            dtype=np.int64,
        )
        fold_constants = _constant_metrics(bank, fit_indices)
        metrics = evaluate_predictor(
            model, bank, exact_library, fold_constants, device=device
        )
        gate = choose_common_predictor(
            [(str(manifest["model"]), metrics, int(manifest["parameter_count"]))]
        )[1][str(manifest["model"])]
        manifest["metrics"] = metrics
        manifest["gate"] = {
            "status": gate.status,
            "passed": gate.passed,
            "reasons": gate.reasons,
        }
        manifest["readiness_status"] = gate.status
        manifest["readiness_reasons"] = gate.reasons
        all_oof_ready = all_oof_ready and gate.passed
        oof_deltas.append(
            {
                "fold": manifest["fold"],
                "old_valid_area_mpa": previous["valid_area_mpa"],
                "exact_valid_area_mpa": metrics["valid_area_mpa"],
                "valid_area_delta_mpa": metrics["valid_area_mpa"]
                - previous["valid_area_mpa"],
            }
        )
    oof["status"] = (
        "REWARD_MODELS_READY" if all_oof_ready else "REWARD_MODELS_NOT_READY"
    )
    oof["cost_precision"] = "FLOAT64_NATIVE_PIXEL_FRACTIONS"
    oof["checkpoint_selection_history"] = (
        "UPDATE_SELECTED_WITH_FLOAT32_COSTS; FROZEN_CHECKPOINT_REEVALUATED_WITH_FLOAT64_COSTS"
    )
    write_json(oof_path, oof)
    payload = {
        "status": (
            "EXACT_NATIVE_COST_REEVALUATION_COMPLETE"
            if selected == previous_selected and all_oof_ready
            else "EXACT_NATIVE_COST_REEVALUATION_BLOCKED"
        ),
        "optimizer_updates": 0,
        "previous_selected_p_all": previous_selected,
        "exact_selected_p_all": selected,
        "validation_rows_with_mask_change": int(np.sum(changed_rows)),
        "validation_specimens_with_mask_change": len(
            set(exact_library.specimen_indices[changed_rows].tolist())
        ),
        "maximum_stored_cost_delta": float(
            np.max(np.abs(stored_costs - exact_library.costs))
        ),
        "candidate_metric_deltas": candidate_deltas,
        "oof_metric_deltas": oof_deltas,
        "model_parameters_changed": False,
        "test_labels_or_metrics_used": False,
    }
    write_json(output / "cost_precision_audit.json", payload)
    _append_ledger(
        root / "results/cai_agent_v3/compute_ledger.jsonl",
        {
            "job": "w2_exact_native_cost_reevaluation",
            "stage": "W2_CORRECTION",
            "device": device,
            "actual_optimizer_updates": 0,
            "status": payload["status"],
        },
    )
    if payload["status"] != "EXACT_NATIVE_COST_REEVALUATION_COMPLETE":
        raise ValueError(
            "exact native-cost reevaluation changed a W2 readiness decision"
        )
    return payload


__all__ = [
    "ValidationLibrary",
    "build_prefix_library",
    "build_validation_library",
    "evaluate_predictor",
    "fit_ridge_diagnostics",
    "load_predictor_checkpoint",
    "precheck_predictor_training",
    "refresh_cost_precision_evaluations",
    "run_oof_reward_predictors",
    "run_predictor_candidates",
]
