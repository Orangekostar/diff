"""Recompute C evidence, figures, cases, and the local result index."""

from __future__ import annotations

import csv
import gzip
import html
import importlib.util
import json
import math
import os
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from scripts.cai_c_retrain.context import TaskContext, atomic_json, sha256_file
from scripts.cai_c_retrain.vlm import render_c_inputs

METHODS = (
    "CENTER_FIRST",
    "GEOMETRY_SPREAD",
    "SERPENTINE",
    "RANDOM",
    "LEARNED_STATIC_TRUE",
    "NO_VLM_SPATIAL_FEEDBACK",
    "VLM_MEAN_FEEDBACK",
    "VLM_SPATIAL_FEEDBACK",
    "VLM_SPATIAL_OPEN_LOOP",
)
MAIN = "VLM_SPATIAL_FEEDBACK"
NONADAPTIVE = METHODS[:5]
BUDGETS = np.asarray([0.0, 0.0625, 0.125, 0.1875, 0.25], dtype=np.float64)
STAGE_EDGES = (0.0, 0.0625, 0.125, 0.1875, 0.25)
CASE_KEYS = (
    "74t7kcdgkr:c8-16",
    "cgtnjyggtm:q24-48",
    "w68dtmpfyf:q16-29",
)
METHOD_LABELS = {
    "CENTER_FIRST": "Center-first",
    "GEOMETRY_SPREAD": "Geometry-spread",
    "SERPENTINE": "Serpentine",
    "RANDOM": "Random",
    "LEARNED_STATIC_TRUE": "Learned-static",
    "NO_VLM_SPATIAL_FEEDBACK": "No-VLM feedback",
    "VLM_MEAN_FEEDBACK": "C mean feedback",
    "VLM_SPATIAL_FEEDBACK": "C spatial feedback",
    "VLM_SPATIAL_OPEN_LOOP": "C spatial open-loop",
}
METHOD_COLORS = {
    "CENTER_FIRST": "#8C8C8C",
    "GEOMETRY_SPREAD": "#6F7C85",
    "SERPENTINE": "#B17C45",
    "RANDOM": "#9B8AA5",
    "LEARNED_STATIC_TRUE": "#4E8B75",
    "NO_VLM_SPATIAL_FEEDBACK": "#3D78A6",
    "VLM_MEAN_FEEDBACK": "#D08C60",
    "VLM_SPATIAL_FEEDBACK": "#B33B3B",
    "VLM_SPATIAL_OPEN_LOOP": "#25858A",
}


def _values(text: str, cast=float) -> np.ndarray:
    return np.asarray([cast(value) for value in str(text).split(";") if value != ""])


def _trajectory(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    costs = _values(row["costs"])
    predictions = _values(row["predictions_mpa"])
    target = float(row["target_mpa"])
    if (
        costs.ndim != 1
        or len(costs) == 0
        or costs.shape != predictions.shape
        or costs[0] != 0.0
        or np.any(np.diff(costs) <= 0)
        or costs[-1] > 0.25 + 1e-12
        or not np.isfinite(costs).all()
        or not np.isfinite(predictions).all()
        or not math.isfinite(target)
    ):
        raise ValueError("cost trajectory is invalid")
    return costs, predictions, target


def _held(
    costs: np.ndarray, predictions: np.ndarray, budgets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    requested = np.asarray(budgets, dtype=np.float64)
    if np.any(requested < 0) or np.any(requested > 0.25):
        raise ValueError("budget is outside the observed partial horizon")
    indices = np.searchsorted(costs, requested, side="right") - 1
    return predictions[indices], costs[indices]


def _left_area(
    costs: np.ndarray, predictions: np.ndarray, target: float, end: float
) -> float:
    if not 0 < end <= 0.25:
        raise ValueError("area endpoint is invalid")
    points = np.asarray([0.0, *costs[(costs > 0) & (costs < end)], end])
    total = 0.0
    for left, right in pairwise(points):
        index = int(np.searchsorted(costs, left, side="right") - 1)
        total += (right - left) * abs(float(predictions[index]) - target)
    return total / end


def curve_metrics(
    rows: list[dict[str, Any]], budgets: list[float] | np.ndarray
) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("curve rows are empty")
    grid = np.asarray(budgets, dtype=np.float64)
    keys = sorted({str(row["specimen_key"]) for row in rows})
    runs = sorted({int(row["run"]) for row in rows})
    lookup = {(int(row["run"]), str(row["specimen_key"])): row for row in rows}
    if set(lookup) != {(run, key) for run in runs for key in keys}:
        raise ValueError("method repeats do not form a complete specimen panel")
    targets = np.asarray(
        [float(lookup[(runs[0], key)]["target_mpa"]) for key in keys], dtype=np.float64
    )
    predictions = np.empty((len(runs), len(keys), len(grid)), dtype=np.float64)
    actual = np.empty_like(predictions)
    for run_index, run in enumerate(runs):
        for key_index, key in enumerate(keys):
            costs, values, target = _trajectory(lookup[(run, key)])
            if target != targets[key_index]:
                raise ValueError("target differs between method repeats")
            predictions[run_index, key_index], actual[run_index, key_index] = _held(
                costs, values, grid
            )
    residual = predictions - targets[None, :, None]
    physical_abs = np.mean(np.abs(residual), axis=0)
    physical_sq = np.mean(np.square(residual), axis=0)
    denominator = np.sum(np.square(targets - targets.mean()))
    r2 = (
        np.mean(1.0 - np.sum(np.square(residual), axis=1) / denominator, axis=0)
        if denominator > 0
        else np.full(len(grid), np.nan)
    )
    return {
        "keys": np.asarray(keys),
        "targets": targets,
        "predictions": predictions,
        "residual": residual,
        "physical_abs": physical_abs,
        "physical_sq": physical_sq,
        "mae": physical_abs.mean(axis=0),
        "mse": physical_sq.mean(axis=0),
        "rmse": np.sqrt(physical_sq.mean(axis=0)),
        "r2": r2,
        "actual_cost_mean": actual.mean(axis=(0, 1)),
        "actual_cost_min": actual.min(axis=(0, 1)),
        "actual_cost_max": actual.max(axis=(0, 1)),
    }


def derive_event_grid(rows: list[dict[str, Any]]) -> np.ndarray:
    values = {0.0, 0.25}
    for row in rows:
        costs = _values(row["costs"])
        if len(costs) == 0 or costs[0] != 0 or np.any(np.diff(costs) <= 0):
            raise ValueError("cost trajectory is invalid")
        values.update(float(cost) for cost in costs if 0 <= cost <= 0.25)
    return np.asarray(sorted(values), dtype=np.float64)


def reindex_bootstrap_weights(
    weights: np.ndarray,
    old_keys: np.ndarray,
    new_keys: np.ndarray,
    new_domains: np.ndarray,
    new_groups: np.ndarray,
    *,
    old_domains: np.ndarray,
    old_groups: np.ndarray,
) -> np.ndarray:
    if weights.shape[1] != len(old_keys) or len(set(old_keys)) != len(old_keys):
        raise ValueError("stored bootstrap weights are invalid")
    lookup = {str(key): index for index, key in enumerate(old_keys)}
    if set(map(str, old_keys)) != set(map(str, new_keys)):
        raise ValueError("bootstrap specimen keys changed")
    order = np.asarray([lookup[str(key)] for key in new_keys], dtype=np.int64)
    if not np.array_equal(np.asarray(old_domains)[order], np.asarray(new_domains)):
        raise ValueError("bootstrap domains changed")
    if not np.array_equal(np.asarray(old_groups)[order], np.asarray(new_groups)):
        raise ValueError("bootstrap capture groups changed")
    return np.asarray(weights)[:, order]


def _first_quality(
    costs: np.ndarray, maes: np.ndarray, target: float
) -> dict[str, Any]:
    if costs.shape != maes.shape or np.any(np.diff(costs) <= 0):
        raise ValueError("quality curve is invalid")
    found = np.flatnonzero(maes <= target)
    if not len(found):
        return {
            "cost": None,
            "mae": None,
            "status": "NOT_REACHED_WITHIN_OBSERVED_RANGE",
            "later_recrosses_target": None,
        }
    index = int(found[0])
    return {
        "cost": float(costs[index]),
        "mae": float(maes[index]),
        "status": "REACHED",
        "later_recrosses_target": bool(np.any(maes[index + 1 :] > target)),
    }


def equal_quality_comparison(
    costs: np.ndarray,
    *,
    main_mae: np.ndarray,
    control_mae: np.ndarray,
    target: float,
) -> dict[str, Any]:
    main = _first_quality(np.asarray(costs), np.asarray(main_mae), target)
    control = _first_quality(np.asarray(costs), np.asarray(control_mae), target)
    if main["cost"] is None or control["cost"] is None:
        absolute = relative = None
        saving_status = "ONE_OR_BOTH_NOT_REACHED"
    else:
        absolute = float(control["cost"] - main["cost"])
        relative = (
            None if control["cost"] == 0 else float(1 - main["cost"] / control["cost"])
        )
        saving_status = "ZERO_CONTROL_COST" if control["cost"] == 0 else "DEFINED"
    return {
        "target_mae": target,
        "main_cost": main["cost"],
        "main_status": main["status"],
        "main_later_recrosses_target": main["later_recrosses_target"],
        "control_cost": control["cost"],
        "control_status": control["status"],
        "control_later_recrosses_target": control["later_recrosses_target"],
        "absolute_saving": absolute,
        "relative_saving": relative,
        "saving_status": saving_status,
    }


def timing_decomposition(
    costs: list[float] | np.ndarray,
    errors: list[float] | np.ndarray,
    *,
    budget: float = 0.25,
    stage_edges: tuple[float, ...] = STAGE_EDGES,
) -> dict[str, Any]:
    cost = np.asarray(costs, dtype=np.float64)
    error = np.asarray(errors, dtype=np.float64)
    if (
        len(cost) == 0
        or cost.shape != error.shape
        or cost[0] != 0
        or cost[-1] > budget
        or np.any(np.diff(cost) <= 0)
        or np.any(error < 0)
    ):
        raise ValueError("timing trajectory is invalid")
    area = (
        np.sum(np.diff(cost) * error[:-1]) + (budget - cost[-1]) * error[-1]
    ) / budget
    terms = (1.0 - cost[1:] / budget) * (error[:-1] - error[1:])
    stages = np.zeros(len(stage_edges) - 1, dtype=np.float64)
    for completion, term in zip(cost[1:], terms, strict=True):
        index = int(np.searchsorted(stage_edges, completion, side="left") - 1)
        if not 0 <= index < len(stages):
            raise ValueError("timing completion falls outside stage edges")
        stages[index] += term
    if not math.isclose(area, float(error[0] - terms.sum()), abs_tol=1e-10):
        raise ValueError("timing identity failed")
    return {
        "initial_error": float(error[0]),
        "area": float(area),
        "terms": terms.tolist(),
        "stage_contributions": stages.tolist(),
    }


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty evidence table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _domain_equal(values: np.ndarray, domains: np.ndarray) -> float:
    return float(
        np.mean([np.mean(values[domains == domain]) for domain in sorted(set(domains))])
    )


def _bootstrap_domain_equal(
    values: np.ndarray, weights: np.ndarray, domains: np.ndarray
) -> np.ndarray:
    per_domain = []
    for domain in sorted(set(domains)):
        mask = domains == domain
        local_weights = weights[:, mask].astype(np.float64)
        denominator = local_weights.sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError("bootstrap produced an empty domain")
        per_domain.append((local_weights @ values[mask]) / denominator)
    return np.mean(per_domain, axis=0)


def _area_vectors(
    rows: list[dict[str, Any]], keys: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        costs, predictions, target = _trajectory(row)
        grouped[str(row["specimen_key"])].append(
            (
                _left_area(costs, predictions, target, 0.25),
                _left_area(costs, predictions, target, 0.0625),
            )
        )
    if set(grouped) != set(keys):
        raise ValueError("area panel differs from the 50 VALID specimens")
    values = np.asarray([np.mean(grouped[key], axis=0) for key in keys])
    return values[:, 0], values[:, 1]


def _load_bootstrap(
    context: TaskContext,
    keys: list[str],
    domains: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, str]:
    source = context.path("old_evidence") / "bootstrap_group_weights.npz"
    try:
        with np.load(source, allow_pickle=False) as archive:
            weights = reindex_bootstrap_weights(
                archive["weights"],
                archive["specimen_keys"],
                np.asarray(keys),
                domains,
                groups,
                old_domains=archive["domains"],
                old_groups=archive["capture_groups"],
            )
        if weights.shape != (5000, 50):
            raise ValueError("stored bootstrap dimensions changed")
        return weights, "REUSED_OLD_WEIGHTS_AFTER_EXPLICIT_KEY_REINDEX"
    except (KeyError, OSError, ValueError):
        rng = np.random.default_rng(2026091401)
        weights = np.zeros((5000, len(keys)), dtype=np.int16)
        for domain in sorted(set(domains)):
            names = sorted(set(groups[domains == domain]))
            draw = rng.integers(0, len(names), size=(5000, len(names)))
            for index, group in enumerate(names):
                weights[:, groups == group] = np.sum(draw == index, axis=1)[:, None]
        return weights, "REGENERATED_ONCE_BECAUSE_OLD_WEIGHTS_DID_NOT_MATCH"


def _full_reference(
    context: TaskContext,
    keys: list[str],
    metadata: dict[str, dict[str, str]],
    targets: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    pointer = json.loads(
        (context.path("old_w3") / "p_all_saved_reference.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        pointer["checkpoint_path"] != context.scope["predictors"]["common_checkpoint"]
        or pointer["checkpoint_sha256"]
        != context.scope["predictors"]["common_checkpoint_sha256"]
    ):
        raise ValueError("full-input pointer differs from the frozen W2 predictor")
    index_rows = _read_csv(context.path("data") / "feature_bank_index.csv")
    with np.load(
        context.root / pointer["source_predictions"], allow_pickle=False
    ) as archive:
        indices = archive["full_specimen_indices"]
        predictions = archive["full_predictions_mpa"]
        source_targets = archive["full_targets_mpa"]
    if not len(indices) == len(predictions) == len(source_targets) == 50:
        raise ValueError("full-input saved arrays changed")
    records = {}
    for source_index, prediction, target in zip(
        indices, predictions, source_targets, strict=True
    ):
        row = index_rows[int(source_index)]
        key = row["specimen_key"]
        if row["split"] != "VALID" or key not in metadata:
            raise ValueError("full-input mapping includes a non-VALID specimen")
        records[key] = {
            "specimen_key": key,
            "dataset_id": metadata[key]["dataset_id"],
            "capture_group_id": metadata[key]["capture_group_id"],
            "full_prediction_mpa": float(prediction),
            "target_mpa": float(target),
            "abs_error_mpa": abs(float(prediction) - float(target)),
            "cost": 1.0,
        }
    if set(records) != set(keys):
        raise ValueError("full-input keys differ from the VALID panel")
    ordered = [records[key] for key in keys]
    full_predictions = np.asarray([row["full_prediction_mpa"] for row in ordered])
    full_targets = np.asarray([row["target_mpa"] for row in ordered])
    if not np.allclose(full_targets, targets, rtol=0, atol=0):
        raise ValueError("full-input targets differ from policy targets")
    residual = full_predictions - targets
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.sum(np.square(targets - targets.mean())))
    metrics = {
        "mae": float(np.mean(np.abs(residual))),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": float(1 - np.sum(np.square(residual)) / denominator),
    }
    for field, pointer_field in (
        ("mae", "full_mae_mpa"),
        ("rmse", "full_rmse_mpa"),
        ("r2", "full_r2"),
    ):
        if abs(metrics[field] - float(pointer[pointer_field])) > 1e-9:
            raise ValueError("full-input frozen metric changed")
    return metrics, ordered


def _analyze_tables(
    context: TaskContext,
    episodes: list[dict[str, str]],
    historical: list[dict[str, str]],
) -> dict[str, Any]:
    evidence_root = context.path("evidence")
    index = [
        row
        for row in _read_csv(context.path("data") / "feature_bank_index.csv")
        if row["split"] == "VALID"
    ]
    keys = sorted(row["specimen_key"] for row in index)
    metadata = {row["specimen_key"]: row for row in index}
    if (
        len(keys) != 50
        or len({metadata[key]["capture_group_id"] for key in keys}) != 48
    ):
        raise ValueError("VALID identity changed")
    domains = np.asarray([metadata[key]["dataset_id"] for key in keys])
    groups = np.asarray([metadata[key]["capture_group_id"] for key in keys])
    by_method = {
        method: [row for row in episodes if row["method"] == method]
        for method in METHODS
    }
    if any(
        len(rows) != (250 if method == "RANDOM" else 50)
        for method, rows in by_method.items()
    ):
        raise ValueError("650-row primary method counts changed")
    targets = np.asarray(
        [
            float(
                next(row for row in episodes if row["specimen_key"] == key)[
                    "target_mpa"
                ]
            )
            for key in keys
        ]
    )
    event_grid = derive_event_grid(episodes)
    primary = {
        method: curve_metrics(rows, BUDGETS) for method, rows in by_method.items()
    }
    event = {
        method: curve_metrics(rows, event_grid) for method, rows in by_method.items()
    }
    for method in METHODS:
        if primary[method]["keys"].tolist() != keys:
            raise ValueError(f"method key order changed: {method}")
    areas = {method: _area_vectors(rows, keys) for method, rows in by_method.items()}
    weights, bootstrap_source = _load_bootstrap(context, keys, domains, groups)
    _write_npz(
        evidence_root / "bootstrap_group_weights.npz",
        weights=weights,
        specimen_keys=np.asarray(keys),
        domains=domains,
        capture_groups=groups,
    )
    full_metrics, full_rows = _full_reference(context, keys, metadata, targets)

    method_summary = []
    same_cost = []
    per_domain = []
    event_rows = []
    for method in METHODS:
        area, early = areas[method]
        method_summary.append(
            {
                "method": method,
                "prior_version": context.scope["prior_version"]
                if method.startswith("VLM_")
                else "UNCHANGED_CONTROL",
                "area_mpa": _domain_equal(area, domains),
                "early_area_mpa": _domain_equal(early, domains),
                "endpoint_mae_mpa": float(primary[method]["mae"][-1]),
                "endpoint_rmse_mpa": float(primary[method]["rmse"][-1]),
                "endpoint_r2": float(primary[method]["r2"][-1]),
                "physical_n": 50,
            }
        )
        for budget_index, budget in enumerate(BUDGETS):
            same_cost.append(
                {
                    "method": method,
                    "budget": float(budget),
                    "mae_mpa": float(primary[method]["mae"][budget_index]),
                    "rmse_mpa": float(primary[method]["rmse"][budget_index]),
                    "r2": float(primary[method]["r2"][budget_index]),
                    "actual_cost_mean": float(
                        primary[method]["actual_cost_mean"][budget_index]
                    ),
                    "actual_cost_min": float(
                        primary[method]["actual_cost_min"][budget_index]
                    ),
                    "actual_cost_max": float(
                        primary[method]["actual_cost_max"][budget_index]
                    ),
                    "physical_n": 50,
                    "aggregation": "REPEAT_LOSS_MEAN_THEN_PHYSICAL_SPECIMEN_MEAN",
                }
            )
            for domain in sorted(set(domains)):
                mask = domains == domain
                residual = primary[method]["residual"][:, mask, budget_index]
                physical_abs = np.mean(np.abs(residual), axis=0)
                physical_sq = np.mean(np.square(residual), axis=0)
                denominator = np.sum(np.square(targets[mask] - targets[mask].mean()))
                per_domain.append(
                    {
                        "method": method,
                        "dataset_id": domain,
                        "budget": float(budget),
                        "physical_n": int(mask.sum()),
                        "mae_mpa": float(physical_abs.mean()),
                        "rmse_mpa": float(np.sqrt(physical_sq.mean())),
                        "r2": float(
                            np.mean(
                                1 - np.sum(np.square(residual), axis=1) / denominator
                            )
                        )
                        if denominator > 0
                        else None,
                        "area_mpa": float(area[mask].mean()),
                        "early_area_mpa": float(early[mask].mean()),
                    }
                )
        for budget_index, budget in enumerate(event_grid):
            event_rows.append(
                {
                    "method": method,
                    "budget": float(budget),
                    "mae_mpa": float(event[method]["mae"][budget_index]),
                    "rmse_mpa": float(event[method]["rmse"][budget_index]),
                    "r2": float(event[method]["r2"][budget_index]),
                    "actual_cost_mean": float(
                        event[method]["actual_cost_mean"][budget_index]
                    ),
                }
            )

    best_nonadaptive = min(
        NONADAPTIVE,
        key=lambda method: (
            next(row["area_mpa"] for row in method_summary if row["method"] == method),
            NONADAPTIVE.index(method),
        ),
    )
    contrasts = (
        ("best_nonadaptive", best_nonadaptive, 0),
        ("feedback", "VLM_SPATIAL_OPEN_LOOP", 0),
        ("vlm_early", "NO_VLM_SPATIAL_FEEDBACK", 1),
        ("spatial_vs_mean", "VLM_MEAN_FEEDBACK", 0),
    )
    effects = []
    for name, control, column in contrasts:
        difference = areas[control][column] - areas[MAIN][column]
        draws = _bootstrap_domain_equal(difference, weights, domains)
        effects.append(
            {
                "effect": name,
                "main_method": MAIN,
                "comparator": control,
                "metric": "early_area_mpa" if column else "area_mpa",
                "gain_mpa": _domain_equal(difference, domains),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "positive_favors_C_main": True,
                "bootstrap_replicates": 5000,
                "bootstrap_seed": 2026091401,
            }
        )
    paired = []
    for control in METHODS:
        if control == MAIN:
            continue
        for budget_index, budget in enumerate(BUDGETS):
            difference = (
                primary[control]["physical_abs"][:, budget_index]
                - primary[MAIN]["physical_abs"][:, budget_index]
            )
            draws = _bootstrap_domain_equal(difference, weights, domains)
            paired.append(
                {
                    "main_method": MAIN,
                    "comparator": control,
                    "budget": float(budget),
                    "mae_gain_mpa": float(difference.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "positive_favors_C_main": True,
                }
            )

    historical_by_method = {
        method: [row for row in historical if row["method"] == method]
        for method in (
            "VLM_SPATIAL_FEEDBACK",
            "VLM_SPATIAL_OPEN_LOOP",
            "VLM_MEAN_FEEDBACK",
        )
    }
    version_rows = []
    version_domain_rows = []
    for method, old_rows in historical_by_method.items():
        old_primary = curve_metrics(old_rows, BUDGETS)
        old_area, old_early = _area_vectors(old_rows, keys)
        for metric, old_values, new_values in (
            ("area_mpa", old_area, areas[method][0]),
            ("early_area_mpa", old_early, areas[method][1]),
            (
                "endpoint_abs_error_mpa",
                old_primary["physical_abs"][:, -1],
                primary[method]["physical_abs"][:, -1],
            ),
        ):
            difference = old_values - new_values
            draws = _bootstrap_domain_equal(difference, weights, domains)
            version_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "historical_A": _domain_equal(old_values, domains)
                    if metric != "endpoint_abs_error_mpa"
                    else float(old_values.mean()),
                    "current_C": _domain_equal(new_values, domains)
                    if metric != "endpoint_abs_error_mpa"
                    else float(new_values.mean()),
                    "A_minus_C": _domain_equal(difference, domains)
                    if metric != "endpoint_abs_error_mpa"
                    else float(difference.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                }
            )
            for domain in sorted(set(domains)):
                mask = domains == domain
                version_domain_rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "dataset_id": domain,
                        "physical_n": int(mask.sum()),
                        "A_minus_C": float(difference[mask].mean()),
                    }
                )

    timing_episode = []
    event_index = []
    acquisition_events = []
    for source_row, row in enumerate(episodes):
        costs, predictions, target = _trajectory(row)
        errors = np.abs(predictions - target)
        timing = timing_decomposition(costs, errors)
        timing_episode.append(
            {
                "method": row["method"],
                "dataset_id": row["dataset_id"],
                "specimen_key": row["specimen_key"],
                "run": row["run"],
                "initial_error_mpa": timing["initial_error"],
                "area_mpa": timing["area"],
                **{
                    f"stage_{index + 1}_mpa": value
                    for index, value in enumerate(timing["stage_contributions"])
                },
            }
        )
        cells = _values(row["cells"], int)
        event_index.append(
            {
                "source_row": source_row,
                "specimen_key": row["specimen_key"],
                "dataset_id": row["dataset_id"],
                "capture_group_id": row["capture_group_id"],
                "method": row["method"],
                "run": row["run"],
                "action_count": len(cells),
                "final_cost": float(costs[-1]),
                "target_mpa": target,
            }
        )
        for index, cell in enumerate(cells):
            acquisition_events.append(
                {
                    "specimen_key": row["specimen_key"],
                    "dataset_id": row["dataset_id"],
                    "capture_group_id": row["capture_group_id"],
                    "method": row["method"],
                    "run": row["run"],
                    "action_index": index + 1,
                    "cell": int(cell),
                    "before_cost": float(costs[index]),
                    "after_cost": float(costs[index + 1]),
                    "error_before_mpa": float(errors[index]),
                    "error_after_mpa": float(errors[index + 1]),
                    "error_reduction_mpa": float(errors[index] - errors[index + 1]),
                }
            )
    timing_rows = []
    maximum_identity_residual = 0.0
    for method in METHODS:
        method_rows = [row for row in timing_episode if row["method"] == method]
        specimen_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in method_rows:
            specimen_values[row["specimen_key"]].append(row)
        aggregates = {}
        for key in keys:
            rows = specimen_values[key]
            aggregates[key] = {
                field: float(np.mean([float(row[field]) for row in rows]))
                for field in (
                    "initial_error_mpa",
                    "area_mpa",
                    "stage_1_mpa",
                    "stage_2_mpa",
                    "stage_3_mpa",
                    "stage_4_mpa",
                )
            }
        summary = {
            field: _domain_equal(
                np.asarray([aggregates[key][field] for key in keys]), domains
            )
            for field in next(iter(aggregates.values()))
        }
        residual = summary["area_mpa"] - (
            summary["initial_error_mpa"]
            - sum(summary[f"stage_{index}_mpa"] for index in range(1, 5))
        )
        maximum_identity_residual = max(maximum_identity_residual, abs(residual))
        for index in range(4):
            timing_rows.append(
                {
                    "method": method,
                    "stage": index + 1,
                    "lower_cost": STAGE_EDGES[index],
                    "upper_cost": STAGE_EDGES[index + 1],
                    "contribution_mpa": summary[f"stage_{index + 1}_mpa"],
                    "initial_error_mpa": summary["initial_error_mpa"],
                    "area_mpa": summary["area_mpa"],
                }
            )
    if maximum_identity_residual > 1e-9:
        raise ValueError("aggregated timing identity failed")

    q_values = np.concatenate(
        [event[method]["mae"] for method in METHODS]
        + [np.asarray([full_metrics["mae"]])]
    )
    uniform_targets = [
        float(value)
        for value in range(math.floor(q_values.min()), math.ceil(q_values.max()) + 1)
    ]
    anchors = [
        {
            "target_id": f"ANCHOR_{method}_{budget}",
            "target_mae": float(primary[method]["mae"][index]),
            "source_method": method,
            "source_budget": float(budget),
        }
        for method in NONADAPTIVE
        for index, budget in enumerate(BUDGETS)
        if budget > 0
    ]
    anchors.append(
        {
            "target_id": "ANCHOR_FULL",
            "target_mae": full_metrics["mae"],
            "source_method": "FULL_SCAN_P_ALL",
            "source_budget": 1.0,
        }
    )

    def quality_rows(targets_to_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for target_row in targets_to_run:
            target = float(target_row["target_mae"])
            for grid_name, grid, collection in (
                ("PRIMARY_FIVE_BUDGETS", BUDGETS, primary),
                ("SECONDARY_EVENT_GRID", event_grid, event),
            ):
                main = _first_quality(grid, collection[MAIN]["mae"], target)
                for method in (*METHODS, "FULL_SCAN_P_ALL"):
                    found = (
                        _first_quality(
                            np.asarray([1.0]), np.asarray([full_metrics["mae"]]), target
                        )
                        if method == "FULL_SCAN_P_ALL"
                        else _first_quality(grid, collection[method]["mae"], target)
                    )
                    if main["cost"] is None or found["cost"] is None:
                        absolute = relative = None
                    else:
                        absolute = float(found["cost"] - main["cost"])
                        relative = (
                            None
                            if found["cost"] == 0
                            else float(1 - main["cost"] / found["cost"])
                        )
                    output.append(
                        {
                            **target_row,
                            "grid": grid_name,
                            "method": method,
                            "cost": found["cost"],
                            "attained_mae": found["mae"],
                            "status": found["status"],
                            "later_recrosses_target": found["later_recrosses_target"],
                            "main_cost": main["cost"],
                            "absolute_saving": absolute,
                            "relative_saving": relative,
                        }
                    )
        return output

    grid_targets = [
        {
            "target_id": f"Q_{value:g}",
            "target_mae": value,
            "target_type": "UNIFORM_1_MPA",
        }
        for value in uniform_targets
    ]
    table_map = {
        "method_summary.csv": method_summary,
        "same_cost_metrics.csv": same_cost,
        "same_cost_metrics_by_domain.csv": per_domain,
        "cost_error_event_curves.csv": event_rows,
        "mechanism_effects.csv": effects,
        "same_cost_paired_summary.csv": paired,
        "version_A_C_comparison.csv": version_rows,
        "version_A_C_by_domain.csv": version_domain_rows,
        "timing_contributions.csv": timing_rows,
        "frozen_episode_index.csv": event_index,
        "acquisition_events.csv": acquisition_events,
        "full_scan_predictions.csv": full_rows,
        "quality_targets.csv": grid_targets + anchors,
        "equal_quality_grid.csv": quality_rows(grid_targets),
        "equal_quality_anchors.csv": quality_rows(anchors),
    }
    for name, rows in table_map.items():
        _write_csv(evidence_root / name, rows)
    atomic_json(
        evidence_root / "full_scan_reference.json",
        {
            "method": "FULL_SCAN_P_ALL",
            "cost": 1.0,
            **full_metrics,
            "physical_n": 50,
            "area_mpa": None,
        },
    )
    old_same = _read_csv(context.path("old_evidence") / "same_cost_metrics.csv")
    old_lookup = {
        (row["method"], float(row["budget"])): row
        for row in old_same
        if row["method"] in METHODS[:6]
    }
    maximum_control_difference = 0.0
    for row in same_cost:
        if row["method"] not in METHODS[:6]:
            continue
        old = old_lookup[(row["method"], row["budget"])]
        for current_field, old_field in (
            ("mae_mpa", "mae"),
            ("rmse_mpa", "rmse"),
            ("r2", "r2"),
        ):
            maximum_control_difference = max(
                maximum_control_difference,
                abs(float(row[current_field]) - float(old[old_field])),
            )
    if maximum_control_difference > 1e-9:
        raise ValueError("reused control metrics changed from frozen evidence")
    return {
        "keys": keys,
        "metadata": metadata,
        "domains": domains,
        "groups": groups,
        "targets": targets,
        "by_method": by_method,
        "primary": primary,
        "event": event,
        "event_grid": event_grid,
        "areas": areas,
        "full_metrics": full_metrics,
        "best_nonadaptive": best_nonadaptive,
        "bootstrap_source": bootstrap_source,
        "maximum_control_difference": maximum_control_difference,
        "maximum_timing_identity_residual": maximum_identity_residual,
        "quality_target_count": len(grid_targets),
        "anchor_count": len(anchors),
        "table_map": table_map,
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _figure_alignment_tool():
    path = (
        Path(os.environ["NATURE_FIGURE_SKILL_ROOT"])
        if "NATURE_FIGURE_SKILL_ROOT" in os.environ
        else Path.home() / ".codex/skills/nature-figure"
    ) / "scripts/audit_panel_alignment.py"
    if not path.is_file():
        raise FileNotFoundError("nature-figure panel alignment auditor is unavailable")
    spec = importlib.util.spec_from_file_location("c_retrain_panel_alignment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the panel alignment auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_figure(fig: Any, destination: Path, name: str) -> dict[str, Any]:
    from matplotlib import pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    qa = destination / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    auditor = _figure_alignment_tool()
    alignment = auditor.require_matplotlib_panel_alignment(
        fig,
        json_out=qa / f"{name}_alignment.json",
        overlay_svg=qa / f"{name}_alignment_overlay.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    outputs = {}
    for extension in ("png", "svg", "pdf"):
        target = destination / f"{name}.{extension}"
        temporary = target.with_name(f"{target.stem}.tmp.{extension}")
        fig.savefig(
            temporary,
            dpi=300 if extension == "png" else None,
            facecolor="white",
            metadata={"Creator": "CAI C evidence pipeline"}
            if extension in {"png", "pdf"}
            else None,
        )
        temporary.replace(target)
        outputs[extension] = {
            "path": target.name,
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    with Image.open(destination / f"{name}.png") as preview:
        extrema = preview.convert("RGB").getextrema()
        if (
            preview.width < 1000
            or preview.height < 500
            or not any(low != high for low, high in extrema)
        ):
            raise ValueError(f"figure preview is blank or undersized: {name}")
    plt.close(fig)
    return {
        "name": name,
        "outputs": outputs,
        "alignment_verdict": alignment["verdict"],
        "alignment_comparisons": alignment["summary"]["comparisons"],
    }


def _figure_style() -> None:
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _render_summary_figures(
    context: TaskContext, analysis: dict[str, Any]
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _figure_style()
    figure_root = context.path("evidence") / "figures"
    records = []

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    for method in METHODS:
        style = {
            "color": METHOD_COLORS[method],
            "linewidth": 2.0 if method == MAIN else 1.15,
            "alpha": 1.0 if method == MAIN else 0.88,
        }
        axes[0].plot(
            BUDGETS, analysis["primary"][method]["mae"], marker="o", ms=3, **style
        )
        axes[1].plot(
            BUDGETS,
            analysis["primary"][method]["rmse"],
            marker="o",
            ms=3,
            label=METHOD_LABELS[method],
            **style,
        )
    axes[0].set(
        title="a  Pooled MAE",
        ylabel="Error (MPa)",
        xlabel="Acquired native-pixel fraction",
    )
    axes[1].set(
        title="b  Pooled RMSE",
        ylabel="Error (MPa)",
        xlabel="Acquired native-pixel fraction",
    )
    for axis in axes:
        axis.set_xlim(0, 0.25)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(-0.08, -0.24), ncol=3, fontsize=7
    )
    records.append(_save_figure(fig, figure_root, "F1_cost_error_curves"))

    paired = [
        row
        for row in analysis["table_map"]["same_cost_paired_summary.csv"]
        if float(row["budget"]) == 0.25
    ]
    effects = analysis["table_map"]["mechanism_effects.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)
    for axis, rows, value_field, title in (
        (axes[0], paired, "mae_gain_mpa", "a  Endpoint MAE gain"),
        (axes[1], effects, "gain_mpa", "b  Error-area gain"),
    ):
        labels = [
            METHOD_LABELS.get(row.get("comparator", ""), row.get("effect", ""))
            for row in rows
        ]
        values = np.asarray([float(row[value_field]) for row in rows])
        low = np.asarray([float(row["ci_low"]) for row in rows])
        high = np.asarray([float(row["ci_high"]) for row in rows])
        y = np.arange(len(rows))
        axis.errorbar(
            values,
            y,
            xerr=np.vstack([values - low, high - values]),
            fmt="o",
            color="#B33B3B",
            ecolor="#52616B",
            capsize=2,
            markersize=4,
        )
        axis.axvline(0, color="#444444", linewidth=0.7)
        axis.set(
            yticks=y,
            yticklabels=labels,
            xlabel="Control error - C main (MPa)",
            title=title,
        )
        axis.invert_yaxis()
        axis.grid(axis="x", color="#E0E0E0", linewidth=0.5)
    records.append(_save_figure(fig, figure_root, "F2_paired_effects"))

    quality = analysis["table_map"]["equal_quality_grid.csv"]
    selected_methods = (
        MAIN,
        analysis["best_nonadaptive"],
        "NO_VLM_SPATIAL_FEEDBACK",
        "VLM_SPATIAL_OPEN_LOOP",
        "FULL_SCAN_P_ALL",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    for method in selected_methods:
        rows = [
            row
            for row in quality
            if row["grid"] == "SECONDARY_EVENT_GRID" and row["method"] == method
        ]
        reached = [row for row in rows if row["cost"] is not None]
        label = METHOD_LABELS.get(method, "Full input")
        color = METHOD_COLORS.get(method, "#242424")
        axes[0].step(
            [float(row["target_mae"]) for row in reached],
            [float(row["cost"]) for row in reached],
            where="mid",
            label=label,
            color=color,
            linewidth=2 if method == MAIN else 1.1,
        )
    main_comparators = [
        analysis["best_nonadaptive"],
        "NO_VLM_SPATIAL_FEEDBACK",
        "VLM_SPATIAL_OPEN_LOOP",
    ]
    for method in main_comparators:
        rows = [
            row
            for row in quality
            if row["grid"] == "SECONDARY_EVENT_GRID"
            and row["method"] == method
            and row["relative_saving"] is not None
        ]
        axes[1].plot(
            [float(row["target_mae"]) for row in rows],
            [100 * float(row["relative_saving"]) for row in rows],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linewidth=1.3,
        )
    axes[0].set(
        title="a  Earliest observed cost", xlabel="Target MAE (MPa)", ylabel="Cost"
    )
    axes[1].set(
        title="b  C-main cost saving",
        xlabel="Target MAE (MPa)",
        ylabel="Relative saving (%)",
    )
    axes[1].axhline(0, color="#555555", linewidth=0.7)
    for axis in axes:
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.5)
        axis.legend(fontsize=6.8)
    records.append(_save_figure(fig, figure_root, "F3_equal_quality"))

    timing = analysis["table_map"]["timing_contributions.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), constrained_layout=True)
    y = np.arange(len(METHODS))
    offsets = np.linspace(-0.27, 0.27, 4)
    stage_colors = ("#286B8E", "#69A6A6", "#B78C52", "#8B718B")
    for stage in range(1, 5):
        values = [
            float(
                next(
                    row["contribution_mpa"]
                    for row in timing
                    if row["method"] == method and row["stage"] == stage
                )
            )
            for method in METHODS
        ]
        axes[0].barh(
            y + offsets[stage - 1],
            values,
            height=0.17,
            color=stage_colors[stage - 1],
            label=f"Stage {stage}",
        )
    initial = [
        float(
            next(row["initial_error_mpa"] for row in timing if row["method"] == method)
        )
        for method in METHODS
    ]
    area = [
        float(next(row["area_mpa"] for row in timing if row["method"] == method))
        for method in METHODS
    ]
    axes[1].plot(initial, y, "o", color="#6A6A6A", label="Initial error")
    axes[1].plot(area, y, "s", color="#B33B3B", label="Error area")
    axes[0].set(
        title="a  Signed timing contribution",
        xlabel="Contribution (MPa)",
        yticks=y,
        yticklabels=[METHOD_LABELS[m] for m in METHODS],
    )
    axes[1].set(
        title="b  Identity components", xlabel="Error (MPa)", yticks=y, yticklabels=[]
    )
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[0].axvline(0, color="#444444", linewidth=0.7)
    axes[0].legend(ncol=2, fontsize=7)
    axes[1].legend(fontsize=7)
    records.append(_save_figure(fig, figure_root, "F4_timing_contributions"))

    domain_rows = [
        row
        for row in analysis["table_map"]["same_cost_metrics_by_domain.csv"]
        if float(row["budget"]) == 0.25
    ]
    version_rows = [
        row
        for row in analysis["table_map"]["version_A_C_by_domain.csv"]
        if row["metric"] == "area_mpa"
    ]
    domains = sorted({row["dataset_id"] for row in domain_rows})
    version_methods = (MAIN, "VLM_SPATIAL_OPEN_LOOP", "VLM_MEAN_FEEDBACK")
    heat = np.asarray(
        [
            [
                float(
                    next(
                        row["mae_mpa"]
                        for row in domain_rows
                        if row["method"] == method and row["dataset_id"] == domain
                    )
                )
                for domain in domains
            ]
            for method in METHODS
        ]
    )
    version_heat = np.asarray(
        [
            [
                float(
                    next(
                        row["A_minus_C"]
                        for row in version_rows
                        if row["method"] == method and row["dataset_id"] == domain
                    )
                )
                for domain in domains
            ]
            for method in version_methods
        ]
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.1),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.45, 1]},
    )
    left = axes[0].imshow(heat, cmap="viridis", aspect="auto")
    right_limit = max(float(np.max(np.abs(version_heat))), 1e-9)
    right = axes[1].imshow(
        version_heat, cmap="RdBu_r", vmin=-right_limit, vmax=right_limit, aspect="auto"
    )
    axes[0].set(
        title="a  Endpoint MAE by domain",
        xticks=np.arange(6),
        xticklabels=domains,
        yticks=np.arange(9),
        yticklabels=[METHOD_LABELS[m] for m in METHODS],
    )
    axes[1].set(
        title="b  Historical A - current C area",
        xticks=np.arange(6),
        xticklabels=domains,
        yticks=np.arange(3),
        yticklabels=[METHOD_LABELS[m] for m in version_methods],
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=55, labelsize=6.5)
    fig.colorbar(left, ax=axes[0], fraction=0.035, pad=0.02, label="MAE (MPa)")
    fig.colorbar(right, ax=axes[1], fraction=0.05, pad=0.02, label="A - C (MPa)")
    records.append(_save_figure(fig, figure_root, "F5_domain_results"))
    return records


def _cell_box(width: int, height: int, cell: int) -> tuple[int, int, int, int]:
    if not 0 <= cell < 64:
        raise ValueError("cell index is outside the 8x8 grid")
    row, column = divmod(cell, 8)
    return (
        round(column * width / 8),
        round(row * height / 8),
        round((column + 1) * width / 8),
        round((row + 1) * height / 8),
    )


def _draw_grid(
    image: Image.Image, *, color: tuple[int, int, int, int] = (30, 30, 30, 180)
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width = max(1, round(max(image.size) / 500))
    for index in range(1, 8):
        x = round(index * image.width / 8)
        y = round(index * image.height / 8)
        draw.line((x, 0, x, image.height - 1), fill=color, width=width)
        draw.line((0, y, image.width - 1, y), fill=color, width=width)


def _save_case_image(image: Image.Image, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    image.save(temporary, format="PNG", optimize=False, compress_level=9)
    temporary.replace(path)
    if image.width < 100 or image.height < 100 or not image.convert("RGB").getbbox():
        raise ValueError(f"case image is blank or undersized: {path.name}")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _surface_overlay(
    surface: Image.Image,
    indicator: np.ndarray,
    confidence: np.ndarray,
    *,
    first_cell: int | None = None,
) -> Image.Image:
    image = surface.convert("RGB").copy()
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for cell in range(64):
        if indicator[cell] <= 0:
            continue
        x0, y0, x1, y1 = _cell_box(image.width, image.height, cell)
        alpha = int(45 + 115 * min(max(float(confidence[cell]), 0.0), 1.0))
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(232, 178, 44, alpha))
    image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    _draw_grid(image, color=(255, 255, 255, 155))
    if first_cell is not None:
        draw = ImageDraw.Draw(image, "RGBA")
        x0, y0, x1, y1 = _cell_box(image.width, image.height, first_cell)
        width = max(3, round(max(image.size) / 250))
        draw.rectangle(
            (x0, y0, x1 - 1, y1 - 1), outline=(185, 35, 35, 255), width=width
        )
    return image


def _measured_overlay(cscan: Image.Image, cells: list[int]) -> Image.Image:
    source = cscan.convert("RGB")
    image = Image.new("RGB", source.size, (218, 220, 222))
    for cell in cells:
        box = _cell_box(source.width, source.height, cell)
        image.paste(source.crop(box), box[:2])
    _draw_grid(image, color=(50, 50, 50, 190))
    draw = ImageDraw.Draw(image, "RGBA")
    width = max(2, round(max(image.size) / 350))
    for cell in cells:
        x0, y0, x1, y1 = _cell_box(source.width, source.height, cell)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 122, 91, 255), width=width)
    return image


def _render_cases(context: TaskContext, analysis: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    queue = {
        row["specimen_key"]: row
        for row in _read_csv(context.path("data") / "candidate_queue.csv")
    }
    features = {
        row["specimen_key"]: row
        for row in _read_csv(context.path("vlm") / "vlm_actor_features_fit.csv")
    }
    source_manifest = json.loads(
        (context.path("data") / "feature_bank_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_root = Path(source_manifest["encoder_execution_root"])
    cases_root = context.path("evidence") / "cases"
    manifest_rows = []
    state_rows = []
    images_by_key: dict[str, dict[str, Path]] = {}
    for key in CASE_KEYS:
        row = next(
            episode
            for episode in analysis["by_method"][MAIN]
            if episode["specimen_key"] == key
        )
        source = queue[key]
        feature = features[key]
        if source["split"] != "VALID" or feature["split"] != "VALID":
            raise ValueError(f"case is not in the frozen VALID panel: {key}")
        surface_path = source_root / source["impacted_surface_path"]
        cscan_path = source_root / source["registered_cscan_crop_path"]
        if sha256_file(surface_path) != source["surface_sha256"]:
            raise ValueError(f"case surface hash changed: {key}")
        if sha256_file(cscan_path) != source["registered_cscan_crop_sha256"]:
            raise ValueError(f"case C-scan hash changed: {key}")
        with Image.open(surface_path) as image:
            rendered = render_c_inputs(image)
            surface = rendered.clean.copy()
        with Image.open(cscan_path) as image:
            cscan = image.convert("RGB").copy()
        cells = _values(row["cells"], int).astype(int).tolist()
        costs = _values(row["costs"])
        predictions = _values(row["predictions_mpa"])
        if (
            len(cells) < 8
            or len(costs) != len(cells) + 1
            or len(predictions) != len(costs)
        ):
            raise ValueError(f"case trajectory cannot supply required states: {key}")
        indicator = _values(feature["region_indicator"])
        confidence = _values(feature["confidence"])
        if indicator.shape != (64,) or confidence.shape != (64,):
            raise ValueError(f"case prior is not a 64-cell vector: {key}")
        case_dir = cases_root / key.replace(":", "__")
        candidates_path = case_dir / "surface_candidates.png"
        first_path = case_dir / "first_action.png"
        output_records = {
            "surface_candidates": _save_case_image(
                _surface_overlay(surface, indicator, confidence), candidates_path
            ),
            "first_action": _save_case_image(
                _surface_overlay(surface, indicator, confidence, first_cell=cells[0]),
                first_path,
            ),
        }
        paths = {"surface_candidates": candidates_path, "first_action": first_path}
        steps = (1, 4, 8, len(cells))
        for count in steps:
            name = "measured_endpoint" if count == len(cells) else f"measured_{count}"
            path = case_dir / f"{name}.png"
            paths[name] = path
            output_records[name] = _save_case_image(
                _measured_overlay(cscan, cells[:count]), path
            )
            state_rows.append(
                {
                    "specimen_key": key,
                    "state": name,
                    "acquired_count": count,
                    "measured_cells": ";".join(map(str, cells[:count])),
                    "cost": float(costs[count]),
                    "prediction_mpa": float(predictions[count]),
                    "target_mpa": float(row["target_mpa"]),
                    "next_cell": cells[count] if count < len(cells) else None,
                }
            )
        fig, axis = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
        axis.step(
            costs,
            predictions,
            where="post",
            color="#B33B3B",
            linewidth=1.8,
            label="C prediction",
        )
        axis.axhline(
            float(row["target_mpa"]),
            color="#202020",
            linestyle="--",
            linewidth=1,
            label="VALID target",
        )
        for count in steps:
            axis.plot(
                costs[count], predictions[count], "o", color="#25858A", markersize=4
            )
            axis.annotate(
                str(count),
                (costs[count], predictions[count]),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set(
            xlim=(0, 0.25),
            xlabel="Acquired native-pixel fraction",
            ylabel="CAI prediction (MPa)",
            title=f"C process: {key}",
        )
        axis.legend(fontsize=7)
        process_path = case_dir / "cai_process.png"
        temporary = process_path.with_name("cai_process.tmp.png")
        case_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(temporary, dpi=220, facecolor="white")
        plt.close(fig)
        temporary.replace(process_path)
        output_records["cai_process"] = {
            "path": process_path.name,
            "sha256": sha256_file(process_path),
            "bytes": process_path.stat().st_size,
        }
        paths["cai_process"] = process_path
        images_by_key[key] = paths
        manifest_rows.append(
            {
                "specimen_key": key,
                "status": "NEW_C_CASE_COMPLETE",
                "method": MAIN,
                "selected_update": row.get("selected_update", ""),
                "action_count": len(cells),
                "first_cell": cells[0],
                "final_cost": float(costs[-1]),
                "surface_source_sha256": source["surface_sha256"],
                "cscan_source_sha256": source["registered_cscan_crop_sha256"],
                "outputs": output_records,
            }
        )
    _write_csv(context.path("evidence") / "case_same_cost_states.csv", state_rows)

    fig, axes = plt.subplots(3, 4, figsize=(7.2, 6.9), constrained_layout=True)
    for row_index, key in enumerate(CASE_KEYS):
        for column_index, name in enumerate(
            ("surface_candidates", "measured_1", "measured_8", "cai_process")
        ):
            axis = axes[row_index, column_index]
            with Image.open(images_by_key[key][name]) as image:
                axis.imshow(image.convert("RGB"), interpolation="none")
            axis.set_axis_off()
            if row_index == 0:
                axis.set_title(
                    (
                        "C prior",
                        "1 acquisition",
                        "8 acquisitions",
                        "Prediction process",
                    )[column_index]
                )
            if column_index == 0:
                axis.text(
                    -0.04,
                    0.5,
                    key,
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8,
                )
    figure_record = _save_figure(
        fig, context.path("evidence") / "figures", "F6_case_progression"
    )
    manifest = {
        "schema_version": 1,
        "status": "THREE_NEW_C_CASES_COMPLETE",
        "case_source": "SELECTED_NEW_C_MAIN_EPISODES_AND_C_PRIORS",
        "new_model_forwards": 0,
        "cases": manifest_rows,
        "summary_figure": figure_record,
    }
    atomic_json(cases_root / "case_manifest.json", manifest)
    return manifest


def _html_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    heading = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{heading}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _render_index(
    context: TaskContext,
    analysis: dict[str, Any],
    figures: list[dict[str, Any]],
    cases: dict[str, Any],
) -> Path:
    evidence_root = context.path("evidence")
    vlm_manifest = json.loads(
        (context.path("vlm") / "vlm_manifest_fit.json").read_text(encoding="utf-8")
    )
    status_rows = []
    for state in vlm_manifest["states"]:
        slug = state["specimen_key"].replace(":", "__")
        thumbnail = f"../vlm/inputs/{slug}/numbered_thumbnail.png"
        status_rows.append(
            {
                "thumbnail": f'<a href="{html.escape(thumbnail)}"><img src="{html.escape(thumbnail)}" alt="C input thumbnail"></a>',
                "specimen_key": state["specimen_key"],
                "status": state["status"],
                "attempts": state["attempts"],
                "reused": state["reused_from_pilot"],
            }
        )
    status_heading = "".join(
        f"<th>{name}</th>"
        for name in ("Input", "Specimen", "Status", "Attempts", "Pilot reuse")
    )
    status_body = "".join(
        "<tr>"
        f"<td>{row['thumbnail']}</td>"
        f"<td>{html.escape(str(row['specimen_key']))}</td>"
        f"<td>{html.escape(str(row['status']))}</td>"
        f"<td>{row['attempts']}</td>"
        f"<td>{row['reused']}</td>"
        "</tr>"
        for row in status_rows
    )
    figure_cards = "".join(
        '<figure><a href="figures/{0}.pdf"><img src="figures/{0}.png" alt="{0}"></a><figcaption>{0}</figcaption></figure>'.format(
            html.escape(record["name"])
        )
        for record in figures
    )
    case_cards = "".join(
        '<article><h3>{key}</h3><div class="case-grid">{images}</div></article>'.format(
            key=html.escape(case["specimen_key"]),
            images="".join(
                '<a href="cases/{slug}/{name}.png"><img src="cases/{slug}/{name}.png" alt="{key} {name}"></a>'.format(
                    slug=html.escape(case["specimen_key"].replace(":", "__")),
                    name=html.escape(name),
                    key=html.escape(case["specimen_key"]),
                )
                for name in (
                    "surface_candidates",
                    "first_action",
                    "measured_1",
                    "measured_4",
                    "measured_8",
                    "measured_endpoint",
                    "cai_process",
                )
            ),
        )
        for case in cases["cases"]
    )
    downloads = "".join(
        f'<li><a href="{html.escape(path.name)}">{html.escape(path.name)}</a></li>'
        for path in sorted(evidence_root.iterdir())
        if path.is_file() and path.name != "index.html"
    )
    methods = _html_table(
        analysis["table_map"]["method_summary.csv"],
        (
            "method",
            "area_mpa",
            "early_area_mpa",
            "endpoint_mae_mpa",
            "endpoint_rmse_mpa",
            "endpoint_r2",
        ),
    )
    versions = _html_table(
        analysis["table_map"]["version_A_C_comparison.csv"],
        (
            "method",
            "metric",
            "historical_A",
            "current_C",
            "A_minus_C",
            "ci_low",
            "ci_high",
        ),
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CAI C retrain evidence</title>
<style>
:root {{ color-scheme: light; --ink:#172126; --muted:#5c686e; --line:#c8d0d3; --accent:#a73434; --paper:#fff; --band:#f2f5f4; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font:14px/1.45 system-ui,sans-serif; }}
header {{ border-bottom:3px solid var(--accent); padding:20px max(24px,calc((100% - 1180px)/2)); }}
main {{ max-width:1180px; margin:auto; padding:18px 24px 48px; }} h1 {{ margin:0; font-size:25px; }} h2 {{ margin:30px 0 10px; font-size:18px; }}
.meta {{ color:var(--muted); }} .figures,.case-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
figure {{ margin:0; border:1px solid var(--line); padding:8px; }} figure img,.case-grid img {{ width:100%; height:auto; display:block; }}
figcaption {{ margin-top:5px; font-weight:600; }} table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th,td {{ padding:6px 7px; border:1px solid var(--line); text-align:left; vertical-align:top; }} th {{ background:var(--band); position:sticky; top:0; }}
.status {{ max-height:520px; overflow:auto; border:1px solid var(--line); }} .status table {{ border:0; }} .status img {{ width:72px; height:72px; object-fit:contain; }}
article {{ margin:14px 0 26px; }} a {{ color:#165b72; }} ul {{ columns:2; }}
@media (max-width:650px) {{ main {{ padding:14px; }} ul {{ columns:1; }} table {{ display:block; overflow-x:auto; }} }}
</style>
</head>
<body><header><h1>CAI C retrain evidence</h1><div class="meta">P0 + exact R1, TRAIN/VALID only, fixed selected VALID panel</div></header>
<main>
<h2>Current C figures</h2><div class="figures">{figure_cards}</div>
<h2>Nine-method primary summary</h2>{methods}
<h2>Historical A / current C comparison</h2>{versions}
<h2>Three selected C cases</h2>{case_cards}
<h2>211 prior states</h2><div class="status"><table><thead><tr>{status_heading}</tr></thead><tbody>{status_body}</tbody></table></div>
<h2>Downloads</h2><ul>{downloads}</ul>
</main></body></html>
"""
    path = evidence_root / "index.html"
    _atomic_text(path, document)
    return path


def _output_hashes(root: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    skipped = exclude or set()
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in skipped
    }


def analyze_stage(context: TaskContext) -> dict[str, Any]:
    assembly_path = context.path("w3") / "assembly_manifest.json"
    vlm_manifest_path = context.path("vlm") / "vlm_manifest_fit.json"
    feature_path = context.path("vlm") / "vlm_actor_features_fit.csv"
    if not all(
        path.is_file() for path in (assembly_path, vlm_manifest_path, feature_path)
    ):
        raise RuntimeError(
            "assembly and complete C prior must exist before evidence analysis"
        )
    assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
    vlm_manifest = json.loads(vlm_manifest_path.read_text(encoding="utf-8"))
    if assembly.get("status") != "PRIMARY_MATRIX_COMPLETE":
        raise ValueError("primary trajectory matrix is incomplete")
    if (
        vlm_manifest.get("status") != "C_PRIOR_COMPLETE"
        or vlm_manifest.get("rows") != 211
    ):
        raise ValueError("C prior is incomplete")
    signature_inputs = [
        assembly_path,
        vlm_manifest_path,
        feature_path,
        context.config_path,
    ]
    signature = context.phase_signature("analyze", signature_inputs)
    evidence_root = context.path("evidence")
    manifest_path = evidence_root / "analysis_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "C_EVIDENCE_COMPLETE":
            if existing.get("input_signature") != signature:
                raise ValueError("completed evidence input signature changed")
            for relative, digest in existing["outputs"].items():
                path = evidence_root / relative
                if not path.is_file() or sha256_file(path) != digest:
                    raise ValueError(f"completed evidence output changed: {relative}")
            return existing
    evidence_root.mkdir(parents=True, exist_ok=True)
    context.transition("analyze", "RUNNING")
    episodes = _read_gzip_csv(context.path("w3") / "policy_validation_episodes.csv.gz")
    historical = _read_gzip_csv(
        context.path("w3") / "historical_A_comparison/policy_validation_episodes.csv.gz"
    )
    if len(episodes) != 650 or len(historical) != 150:
        raise ValueError("assembled evidence input counts changed")
    analysis = _analyze_tables(context, episodes, historical)
    figures = _render_summary_figures(context, analysis)
    cases = _render_cases(context, analysis)
    index_path = _render_index(context, analysis, figures, cases)
    manifest = {
        "schema_version": 1,
        "task_id": context.task_id,
        "status": "C_EVIDENCE_COMPLETE",
        "prior_version": context.scope["prior_version"],
        "execution_code_sha256": sha256_file(Path(__file__)),
        "input_signature": signature,
        "primary_episode_rows": len(episodes),
        "historical_A_rows": len(historical),
        "physical_valid_n": 50,
        "capture_group_n": 48,
        "domain_n": 6,
        "method_n": 9,
        "best_nonadaptive": analysis["best_nonadaptive"],
        "bootstrap": {
            "replicates": 5000,
            "seed": 2026091401,
            "source": analysis["bootstrap_source"],
        },
        "event_grid_points": len(analysis["event_grid"]),
        "quality_target_count": analysis["quality_target_count"],
        "anchor_count": analysis["anchor_count"],
        "maximum_reused_control_metric_difference": analysis[
            "maximum_control_difference"
        ],
        "maximum_timing_identity_residual": analysis[
            "maximum_timing_identity_residual"
        ],
        "figure_count": len(figures) + 1,
        "case_count": len(cases["cases"]),
        "new_model_forwards": 0,
        "index": index_path.relative_to(evidence_root).as_posix(),
        "outputs": {},
    }
    manifest["outputs"] = _output_hashes(
        evidence_root, exclude={manifest_path.relative_to(evidence_root).as_posix()}
    )
    atomic_json(manifest_path, manifest)
    context.transition(
        "analyze",
        "COMPLETE",
        primary_episode_rows=650,
        method_n=9,
        figure_count=manifest["figure_count"],
        case_count=3,
    )
    return manifest


__all__ = [
    "analyze_stage",
    "curve_metrics",
    "derive_event_grid",
    "equal_quality_comparison",
    "reindex_bootstrap_weights",
    "timing_decomposition",
]
