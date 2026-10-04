"""Pure trajectory and statistical calculations for order analysis."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cmc_bbdm.cai_agent_v3.metrics import left_error_area_mpa
from scripts.cai_order_mechanism.inputs import (
    ARCHIVED_METHODS,
    MAIN_METHOD,
    TaskContext,
    load_bootstrap_weights,
    read_csv_rows,
    read_gzip_csv_rows,
    write_csv_atomic,
    write_gzip_csv_atomic,
)

HORIZON = 0.25
EARLY_HORIZON = 0.0625
STAGE_EDGES = np.asarray([0.0, 0.0625, 0.125, 0.1875, 0.25], dtype=np.float64)
DISPLAY_LABELS = {
    "CENTER_FIRST": "Center-first",
    "GEOMETRY_SPREAD": "Geometry-spread",
    "SERPENTINE": "Serpentine",
    "RANDOM": "Random",
    "LEARNED_STATIC_TRUE": "Learned-static",
    MAIN_METHOD: "Proposed / 本文方法",
}


def _split_values(text: object, cast: type = float) -> np.ndarray:
    values = [cast(value) for value in str(text).split(";") if value != ""]
    return np.asarray(values)


@dataclass(frozen=True, slots=True)
class Episode:
    specimen_key: str
    dataset_id: str
    capture_group_id: str
    method: str
    run: int
    target_mpa: float
    cells: tuple[int, ...]
    costs: np.ndarray
    predictions_mpa: np.ndarray


@dataclass(frozen=True, slots=True)
class TimingResult:
    initial_error: float
    final_error: float
    area_mpa: float
    stage_index: np.ndarray
    raw_event: np.ndarray
    weighted_event: np.ndarray
    raw_stage: np.ndarray
    raw_positive: np.ndarray
    raw_negative: np.ndarray
    weighted_stage: np.ndarray
    weighted_positive: np.ndarray
    weighted_negative: np.ndarray


def parse_episode(row: Mapping[str, object]) -> Episode:
    required = (
        "specimen_key",
        "dataset_id",
        "capture_group_id",
        "method",
        "run",
        "target_mpa",
        "cells",
        "costs",
        "predictions_mpa",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"episode is missing fields: {missing}")
    cells_array = _split_values(row["cells"], int).astype(np.int64, copy=False)
    cells = tuple(int(value) for value in cells_array)
    costs = _split_values(row["costs"]).astype(np.float64, copy=False)
    predictions = _split_values(row["predictions_mpa"]).astype(np.float64, copy=False)
    target = float(row["target_mpa"])
    if any(cell < 0 or cell >= 64 for cell in cells):
        raise ValueError("actions must be cell indices in [0, 63]")
    if len(cells) != len(set(cells)):
        raise ValueError("actions must be unique")
    if len(costs) != len(cells) + 1 or len(predictions) != len(cells) + 1:
        raise ValueError("costs and predictions must contain actions plus one values")
    if (
        len(costs) == 0
        or costs[0] != 0.0
        or np.any(np.diff(costs) <= 0.0)
        or costs[-1] > HORIZON + 1e-12
        or not np.isfinite(costs).all()
    ):
        raise ValueError("cost trajectory is invalid")
    if not np.isfinite(predictions).all() or not math.isfinite(target):
        raise ValueError("prediction trajectory is invalid")
    return Episode(
        specimen_key=str(row["specimen_key"]),
        dataset_id=str(row["dataset_id"]),
        capture_group_id=str(row["capture_group_id"]),
        method=str(row["method"]),
        run=int(row["run"]),
        target_mpa=target,
        cells=cells,
        costs=costs,
        predictions_mpa=predictions,
    )


def timing_decomposition(
    costs: np.ndarray,
    predictions_mpa: np.ndarray,
    target_mpa: float,
    *,
    horizon: float = HORIZON,
    stage_edges: np.ndarray = STAGE_EDGES,
) -> TimingResult:
    cost = np.asarray(costs, dtype=np.float64)
    prediction = np.asarray(predictions_mpa, dtype=np.float64)
    if (
        cost.ndim != 1
        or cost.shape != prediction.shape
        or len(cost) < 2
        or cost[0] != 0.0
        or cost[-1] > horizon + 1e-12
        or np.any(np.diff(cost) <= 0.0)
        or not np.isfinite(cost).all()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("timing trajectory is invalid")
    edges = np.asarray(stage_edges, dtype=np.float64)
    if not np.array_equal(edges, np.asarray([0.0, 0.0625, 0.125, 0.1875, horizon])):
        raise ValueError("stage edges changed")
    errors = np.abs(prediction - float(target_mpa))
    area = float(
        (np.sum(np.diff(cost) * errors[:-1]) + (horizon - cost[-1]) * errors[-1])
        / horizon
    )
    raw = errors[:-1] - errors[1:]
    weighted = (1.0 - cost[1:] / horizon) * raw
    stage = np.searchsorted(edges, cost[1:], side="left") - 1
    if np.any((stage < 0) | (stage >= len(edges) - 1)):
        raise ValueError("action completion falls outside stage edges")

    def aggregate(values: np.ndarray, mode: str = "net") -> np.ndarray:
        if mode == "positive":
            values = np.maximum(values, 0.0)
        elif mode == "negative":
            values = np.minimum(values, 0.0)
        return np.bincount(stage, weights=values, minlength=len(edges) - 1).astype(
            np.float64
        )

    tolerance = 1e-8
    if abs(float(raw.sum()) - float(errors[0] - errors[-1])) > tolerance:
        raise ValueError("raw event identity failed")
    if abs(float(weighted.sum()) - float(errors[0] - area)) > tolerance:
        raise ValueError("weighted event identity failed")
    return TimingResult(
        initial_error=float(errors[0]),
        final_error=float(errors[-1]),
        area_mpa=area,
        stage_index=stage.astype(np.int64),
        raw_event=raw,
        weighted_event=weighted,
        raw_stage=aggregate(raw),
        raw_positive=aggregate(raw, "positive"),
        raw_negative=aggregate(raw, "negative"),
        weighted_stage=aggregate(weighted),
        weighted_positive=aggregate(weighted, "positive"),
        weighted_negative=aggregate(weighted, "negative"),
    )


def domain_equal(values: np.ndarray, domains: np.ndarray) -> float:
    samples = np.asarray(values, dtype=np.float64)
    labels = np.asarray(domains)
    if samples.ndim != 1 or labels.shape != samples.shape or len(samples) == 0:
        raise ValueError("domain-equal inputs are invalid")
    unique = sorted(set(map(str, labels)))
    if not unique:
        raise ValueError("domain panel is empty")
    return float(np.mean([samples[labels.astype(str) == domain].mean() for domain in unique]))


def bootstrap_domain_equal(
    values: np.ndarray, weights: np.ndarray, domains: np.ndarray
) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    matrix = np.asarray(weights)
    labels = np.asarray(domains).astype(str)
    if samples.ndim != 1 or labels.shape != samples.shape:
        raise ValueError("bootstrap values and domains differ")
    if matrix.ndim != 2 or matrix.shape[1] != len(samples) or np.any(matrix < 0):
        raise ValueError("bootstrap weights are invalid")
    estimates = []
    for domain in sorted(set(labels)):
        mask = labels == domain
        local = matrix[:, mask].astype(np.float64)
        denominator = local.sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError("bootstrap produced an empty domain")
        estimates.append((local @ samples[mask]) / denominator)
    return np.mean(estimates, axis=0)


def endpoint_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or truth.ndim != 1 or values.shape[1] != len(truth):
        raise ValueError("endpoint predictions must be repeat by specimen")
    residual = values - truth[None, :]
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denominator = float(np.sum(np.square(truth - truth.mean())))
    r2 = (
        None
        if denominator == 0.0
        else float(np.mean(1.0 - np.sum(np.square(residual), axis=1) / denominator))
    )
    return {"mae_mpa": mae, "rmse_mpa": rmse, "r2": r2}


def first_quality_crossing(
    costs: np.ndarray, values: np.ndarray, target: float
) -> dict[str, float | str | bool | None]:
    x = np.asarray(costs, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if (
        x.ndim != 1
        or x.shape != y.shape
        or len(x) == 0
        or np.any(np.diff(x) <= 0)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("quality curve is invalid")
    found = np.flatnonzero(y <= float(target))
    if len(found) == 0:
        return {
            "cost": None,
            "value": None,
            "status": "NOT_REACHED_WITHIN_OBSERVED_RANGE",
            "later_recrosses_target": None,
        }
    index = int(found[0])
    return {
        "cost": float(x[index]),
        "value": float(y[index]),
        "status": "REACHED",
        "later_recrosses_target": bool(np.any(y[index + 1 :] > float(target))),
    }


def compare_quality(
    costs: np.ndarray,
    *,
    main_values: np.ndarray,
    comparator_values: np.ndarray,
    target: float,
) -> dict[str, object]:
    main = first_quality_crossing(costs, main_values, target)
    comparator = first_quality_crossing(costs, comparator_values, target)
    if main["cost"] is None or comparator["cost"] is None:
        absolute = relative = None
        saving_status = "ONE_OR_BOTH_NOT_REACHED"
    else:
        main_cost = float(main["cost"])
        comparator_cost = float(comparator["cost"])
        absolute = comparator_cost - main_cost
        relative = None if comparator_cost == 0.0 else 1.0 - main_cost / comparator_cost
        saving_status = "ZERO_COMPARATOR_COST" if comparator_cost == 0.0 else "DEFINED"
    return {
        "target_mpa": float(target),
        "main_cost": main["cost"],
        "main_status": main["status"],
        "main_later_recrosses_target": main["later_recrosses_target"],
        "comparator_cost": comparator["cost"],
        "comparator_status": comparator["status"],
        "comparator_later_recrosses_target": comparator["later_recrosses_target"],
        "absolute_saving": absolute,
        "relative_saving": relative,
        "saving_status": saving_status,
    }


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _method_order(method: str) -> int:
    return ARCHIVED_METHODS.index(method)


def _group_mean(
    rows: list[dict[str, Any]], key_fields: tuple[str, ...], value_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    output = []
    for key, members in grouped.items():
        result = dict(zip(key_fields, key, strict=True))
        result.update(
            {
                field: float(np.mean([float(row[field]) for row in members]))
                for field in value_fields
            }
        )
        result["repeat_count"] = len(members)
        output.append(result)
    return output


def _archive_panel(context: TaskContext) -> tuple[list[dict[str, str]], list[Episode]]:
    rows = [
        row
        for row in read_gzip_csv_rows(context.source("sources.assembled_episodes"))
        if row["method"] in ARCHIVED_METHODS
    ]
    if len(rows) != 500:
        raise ValueError("prepared archived panel no longer contains 500 rows")
    rows.sort(
        key=lambda row: (
            _method_order(row["method"]),
            row["specimen_key"],
            int(row["run"]),
        )
    )
    return rows, [parse_episode(row) for row in rows]


def derive_archived(context: TaskContext) -> list[Path]:
    results = context.results_root
    rows, episodes = _archive_panel(context)
    cohort = {
        row["specimen_key"]: row
        for row in read_csv_rows(results / "cohort_manifest.csv")
    }
    keys = sorted(cohort)
    domains = np.asarray([cohort[key]["dataset_id"] for key in keys])
    groups = np.asarray([cohort[key]["capture_group_id"] for key in keys])
    weights = load_bootstrap_weights(context, keys, domains, groups)
    cell_pixels = {
        key: np.asarray(
            [int(value) for value in cohort[key]["cell_pixel_counts"].split(";")],
            dtype=np.int64,
        )
        for key in keys
    }

    archived_rows = []
    event_rows: list[dict[str, Any]] = []
    stage_episode_rows: list[dict[str, Any]] = []
    episode_results: dict[tuple[str, str, int], TimingResult] = {}
    episode_lookup: dict[tuple[str, str, int], Episode] = {}
    for source_row, episode in zip(rows, episodes, strict=True):
        timing = timing_decomposition(
            episode.costs, episode.predictions_mpa, episode.target_mpa
        )
        identity = (episode.method, episode.specimen_key, episode.run)
        episode_results[identity] = timing
        episode_lookup[identity] = episode
        archived_rows.append(
            {
                "identity_family": "ARCHIVED_METHODS",
                "cohort": "VALID50_CAPTURE48_DOMAIN6",
                "clock": "ARCHIVED_FLOAT64",
                "method_display_label": DISPLAY_LABELS[episode.method],
                **source_row,
            }
        )
        errors = np.abs(episode.predictions_mpa - episode.target_mpa)
        pixels = cell_pixels[episode.specimen_key]
        for action_index, cell in enumerate(episode.cells, start=1):
            stage = int(timing.stage_index[action_index - 1])
            event_rows.append(
                {
                    "identity_family": "ARCHIVED_METHODS",
                    "cohort": "VALID50_CAPTURE48_DOMAIN6",
                    "clock": "ARCHIVED_FLOAT64",
                    "specimen_key": episode.specimen_key,
                    "dataset_id": episode.dataset_id,
                    "capture_group_id": episode.capture_group_id,
                    "method": episode.method,
                    "method_display_label": DISPLAY_LABELS[episode.method],
                    "run": episode.run,
                    "action_index": action_index,
                    "cell": cell,
                    "before_cost": float(episode.costs[action_index - 1]),
                    "after_cost": float(episode.costs[action_index]),
                    "before_prediction_mpa": float(
                        episode.predictions_mpa[action_index - 1]
                    ),
                    "after_prediction_mpa": float(
                        episode.predictions_mpa[action_index]
                    ),
                    "before_error_mpa": float(errors[action_index - 1]),
                    "after_error_mpa": float(errors[action_index]),
                    "raw_delta_mpa": float(timing.raw_event[action_index - 1]),
                    "weighted_gain_mpa": float(
                        timing.weighted_event[action_index - 1]
                    ),
                    "stage": stage + 1,
                    "stage_interval": (
                        f"({STAGE_EDGES[stage]},{STAGE_EDGES[stage + 1]}]"
                    ),
                    "new_pixels": int(pixels[cell]),
                    "units": "MPa; cost=fraction; pixels=count",
                }
            )
        for stage in range(4):
            event_mask = timing.stage_index == stage
            stage_episode_rows.append(
                {
                    "identity_family": "ARCHIVED_METHODS",
                    "cohort": "VALID50_CAPTURE48_DOMAIN6",
                    "clock": "ARCHIVED_FLOAT64",
                    "specimen_key": episode.specimen_key,
                    "dataset_id": episode.dataset_id,
                    "capture_group_id": episode.capture_group_id,
                    "method": episode.method,
                    "method_display_label": DISPLAY_LABELS[episode.method],
                    "run": episode.run,
                    "stage": stage + 1,
                    "lower_cost": float(STAGE_EDGES[stage]),
                    "upper_cost": float(STAGE_EDGES[stage + 1]),
                    "raw_net_mpa": float(timing.raw_stage[stage]),
                    "raw_positive_mpa": float(timing.raw_positive[stage]),
                    "raw_negative_mpa": float(timing.raw_negative[stage]),
                    "weighted_net_mpa": float(timing.weighted_stage[stage]),
                    "weighted_positive_mpa": float(
                        timing.weighted_positive[stage]
                    ),
                    "weighted_negative_mpa": float(
                        timing.weighted_negative[stage]
                    ),
                    "event_count": int(event_mask.sum()),
                    "new_pixels": int(
                        sum(
                            pixels[cell]
                            for cell, keep in zip(
                                episode.cells, event_mask, strict=True
                            )
                            if keep
                        )
                    ),
                    "initial_error_mpa": timing.initial_error,
                    "final_error_mpa": timing.final_error,
                    "area_mpa": timing.area_mpa,
                    "raw_identity_residual_mpa": float(
                        timing.raw_stage.sum()
                        - (timing.initial_error - timing.final_error)
                    ),
                    "weighted_identity_residual_mpa": float(
                        timing.weighted_stage.sum()
                        - (timing.initial_error - timing.area_mpa)
                    ),
                    "units": "MPa; pixels=count",
                    "aggregation": "TRAJECTORY_EVENT_SUM",
                }
            )

    metric_fields = (
        "raw_net_mpa",
        "raw_positive_mpa",
        "raw_negative_mpa",
        "weighted_net_mpa",
        "weighted_positive_mpa",
        "weighted_negative_mpa",
        "event_count",
        "new_pixels",
        "initial_error_mpa",
        "final_error_mpa",
        "area_mpa",
    )
    specimen_base = _group_mean(
        stage_episode_rows,
        ("method", "specimen_key", "dataset_id", "capture_group_id", "stage"),
        metric_fields,
    )
    stage_specimen_rows = [
        {
            "identity_family": "ARCHIVED_METHODS",
            "cohort": "VALID50_CAPTURE48_DOMAIN6",
            "clock": "ARCHIVED_FLOAT64",
            "method_display_label": DISPLAY_LABELS[row["method"]],
            **row,
            "units": "MPa; pixels=count",
            "aggregation": "REPEAT_MEAN_WITHIN_SPECIMEN",
        }
        for row in sorted(
            specimen_base,
            key=lambda value: (
                _method_order(value["method"]),
                value["specimen_key"],
                value["stage"],
            ),
        )
    ]

    summary_rows = []
    for method in ARCHIVED_METHODS:
        for stage in range(1, 5):
            panel = [
                row
                for row in stage_specimen_rows
                if row["method"] == method and row["stage"] == stage
            ]
            for domain in sorted(set(domains)):
                local = [row for row in panel if row["dataset_id"] == domain]
                summary_rows.append(
                    {
                        "identity_family": "ARCHIVED_METHODS",
                        "cohort": "VALID50_CAPTURE48_DOMAIN6",
                        "clock": "ARCHIVED_FLOAT64",
                        "summary_scope": "DOMAIN",
                        "dataset_id": domain,
                        "method": method,
                        "method_display_label": DISPLAY_LABELS[method],
                        "stage": stage,
                        "physical_n": len(local),
                        **{
                            field: float(np.mean([row[field] for row in local]))
                            for field in metric_fields
                        },
                        "units": "MPa; pixels=count",
                        "aggregation": (
                            "REPEAT_MEAN_THEN_SPECIMEN_MEAN_WITHIN_DOMAIN"
                        ),
                    }
                )
            summary_rows.append(
                {
                    "identity_family": "ARCHIVED_METHODS",
                    "cohort": "VALID50_CAPTURE48_DOMAIN6",
                    "clock": "ARCHIVED_FLOAT64",
                    "summary_scope": "SIX_DOMAIN_EQUAL",
                    "dataset_id": "ALL_SIX_EQUAL",
                    "method": method,
                    "method_display_label": DISPLAY_LABELS[method],
                    "stage": stage,
                    "physical_n": 50,
                    **{
                        field: domain_equal(
                            np.asarray([row[field] for row in panel]),
                            np.asarray([row["dataset_id"] for row in panel]),
                        )
                        for field in metric_fields
                    },
                    "units": "MPa; pixels=count",
                    "aggregation": (
                        "REPEAT_MEAN_THEN_DOMAIN_MEAN_THEN_SIX_DOMAIN_EQUAL"
                    ),
                }
            )

    existing = read_csv_rows(context.source("sources.existing_stage_summary"))
    existing_lookup = {
        (row["method"], int(row["stage"])): float(row["contribution_mpa"])
        for row in existing
        if row["method"] in ARCHIVED_METHODS
    }
    for row in summary_rows:
        if row["summary_scope"] == "SIX_DOMAIN_EQUAL":
            expected = existing_lookup[(row["method"], row["stage"])]
            if abs(row["weighted_net_mpa"] - expected) > 1e-9:
                raise ValueError(
                    "new weighted-stage result does not reproduce saved evidence"
                )

    specimen_lookup = {
        (row["method"], row["specimen_key"], row["stage"]): row
        for row in stage_specimen_rows
    }
    totals: dict[tuple[str, str], dict[str, float]] = {}
    for method in ARCHIVED_METHODS:
        for key in keys:
            local = [
                result
                for (candidate, specimen, _run), result in episode_results.items()
                if candidate == method and specimen == key
            ]
            totals[(method, key)] = {
                "initial": float(np.mean([result.initial_error for result in local])),
                "final": float(np.mean([result.final_error for result in local])),
                "area": float(np.mean([result.area_mpa for result in local])),
            }
    contrast_rows = []
    for control in ARCHIVED_METHODS:
        if control == MAIN_METHOD:
            continue
        per_metric_total = {
            "weighted_net_mpa": np.zeros(50, dtype=np.float64),
            "raw_net_mpa": np.zeros(50, dtype=np.float64),
        }
        first_contrast_index = len(contrast_rows)
        for stage in range(1, 5):
            for metric, definition in (
                (
                    "weighted_net_mpa",
                    "PROPOSED_MINUS_CONTROL_WEIGHTED_CONTRIBUTION",
                ),
                ("raw_net_mpa", "PROPOSED_MINUS_CONTROL_RAW_ERROR_CHANGE"),
            ):
                difference = np.asarray(
                    [
                        specimen_lookup[(MAIN_METHOD, key, stage)][metric]
                        - specimen_lookup[(control, key, stage)][metric]
                        for key in keys
                    ],
                    dtype=np.float64,
                )
                per_metric_total[metric] += difference
                draws = bootstrap_domain_equal(difference, weights, domains)
                contrast_rows.append(
                    {
                        "identity_family": "ARCHIVED_METHODS",
                        "cohort": "VALID50_CAPTURE48_DOMAIN6",
                        "clock": "ARCHIVED_FLOAT64",
                        "main_method": MAIN_METHOD,
                        "main_display_label": DISPLAY_LABELS[MAIN_METHOD],
                        "comparator": control,
                        "comparator_display_label": DISPLAY_LABELS[control],
                        "stage": stage,
                        "metric": metric,
                        "difference_definition": definition,
                        "estimate_mpa": domain_equal(difference, domains),
                        "ci_low_mpa": float(np.quantile(draws, 0.025)),
                        "ci_high_mpa": float(np.quantile(draws, 0.975)),
                        "positive_favors": "PROPOSED",
                        "bootstrap_replicates": 5000,
                        "units": "MPa",
                        "aggregation": (
                            "REPEAT_MEAN_THEN_DOMAIN_EQUAL_PAIRED_BOOTSTRAP"
                        ),
                    }
                )
        weighted_residual = []
        raw_residual = []
        for index, key in enumerate(keys):
            main = totals[(MAIN_METHOD, key)]
            comparator = totals[(control, key)]
            expected_weighted = (
                comparator["area"]
                - main["area"]
                + main["initial"]
                - comparator["initial"]
            )
            expected_raw = (
                comparator["final"]
                - main["final"]
                + main["initial"]
                - comparator["initial"]
            )
            weighted_residual.append(
                per_metric_total["weighted_net_mpa"][index] - expected_weighted
            )
            raw_residual.append(
                per_metric_total["raw_net_mpa"][index] - expected_raw
            )
        max_weighted = float(np.max(np.abs(weighted_residual)))
        max_raw = float(np.max(np.abs(raw_residual)))
        if max_weighted > 1e-8 or max_raw > 1e-8:
            raise ValueError("paired stage identity failed")
        for row in contrast_rows[first_contrast_index:]:
            row["max_weighted_identity_residual_mpa"] = max_weighted
            row["max_raw_identity_residual_mpa"] = max_raw

    terminal_rows = []
    main_by_key = {key: episode_lookup[(MAIN_METHOD, key, 0)] for key in keys}
    terminal_value_fields = (
        "jaccard",
        "main_cell_count",
        "comparator_cell_count",
        "main_final_pixels",
        "comparator_final_pixels",
        "pixel_difference_comparator_minus_proposed",
        "cost_difference_comparator_minus_proposed",
        "final_error_difference_comparator_minus_proposed_mpa",
    )
    for control in ARCHIVED_METHODS:
        if control == MAIN_METHOD:
            continue
        run_rows = []
        for key in keys:
            main = main_by_key[key]
            main_set = set(main.cells)
            main_pixels = int(cell_pixels[key][list(main.cells)].sum())
            candidates = [
                episode
                for (method, specimen, _run), episode in episode_lookup.items()
                if method == control and specimen == key
            ]
            for candidate in candidates:
                comparator_set = set(candidate.cells)
                comparator_pixels = int(
                    cell_pixels[key][list(candidate.cells)].sum()
                )
                row = {
                    "identity_family": "ARCHIVED_METHODS",
                    "cohort": "VALID50_CAPTURE48_DOMAIN6",
                    "clock": "ARCHIVED_FLOAT64",
                    "record_level": "RUN",
                    "specimen_key": key,
                    "dataset_id": main.dataset_id,
                    "capture_group_id": main.capture_group_id,
                    "main_method": MAIN_METHOD,
                    "comparator": control,
                    "run": candidate.run,
                    "jaccard": len(main_set & comparator_set)
                    / len(main_set | comparator_set),
                    "main_cell_count": len(main_set),
                    "comparator_cell_count": len(comparator_set),
                    "main_final_pixels": main_pixels,
                    "comparator_final_pixels": comparator_pixels,
                    "pixel_difference_comparator_minus_proposed": (
                        comparator_pixels - main_pixels
                    ),
                    "cost_difference_comparator_minus_proposed": float(
                        candidate.costs[-1] - main.costs[-1]
                    ),
                    "final_error_difference_comparator_minus_proposed_mpa": (
                        abs(float(candidate.predictions_mpa[-1] - candidate.target_mpa))
                        - abs(float(main.predictions_mpa[-1] - main.target_mpa))
                    ),
                    "difference_definition": (
                        "COMPARATOR_MINUS_PROPOSED_POSITIVE_ERROR_FAVORS_PROPOSED"
                    ),
                    "units": (
                        "Jaccard=fraction; pixels=count; cost=fraction; error=MPa"
                    ),
                }
                run_rows.append(row)
                terminal_rows.append(row)
        for key in keys:
            local = [row for row in run_rows if row["specimen_key"] == key]
            terminal_rows.append(
                {
                    **{
                        field: local[0][field]
                        for field in (
                            "identity_family",
                            "cohort",
                            "clock",
                            "specimen_key",
                            "dataset_id",
                            "capture_group_id",
                            "main_method",
                            "comparator",
                            "difference_definition",
                            "units",
                        )
                    },
                    "record_level": "SPECIMEN_REPEAT_MEAN",
                    "run": "MEAN",
                    **{
                        field: float(np.mean([row[field] for row in local]))
                        for field in terminal_value_fields
                    },
                }
            )

    return [
        write_gzip_csv_atomic(
            results / "archived_cohort_episodes.csv.gz", archived_rows
        ),
        write_gzip_csv_atomic(results / "archived_events.csv.gz", event_rows),
        write_csv_atomic(results / "stage_episode_metrics.csv", stage_episode_rows),
        write_csv_atomic(results / "stage_specimen_metrics.csv", stage_specimen_rows),
        write_csv_atomic(results / "stage_summary.csv", summary_rows),
        write_csv_atomic(results / "stage_paired_contrasts.csv", contrast_rows),
        write_csv_atomic(results / "terminal_set_comparison.csv", terminal_rows),
        _write_npz_atomic(
            results / "bootstrap_group_weights.npz",
            weights=weights,
            specimen_keys=np.asarray(keys),
            domains=domains,
            capture_groups=groups,
        ),
    ]


def bootstrap_pooled(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or samples.shape != (matrix.shape[1],):
        raise ValueError("pooled bootstrap inputs are invalid")
    denominator = matrix.sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError("pooled bootstrap draw is empty")
    return (matrix @ samples) / denominator


def _event_grid(episodes: Sequence[Episode]) -> np.ndarray:
    values = {0.0, 0.25}
    for episode in episodes:
        values.update(float(cost) for cost in episode.costs if 0.0 <= cost <= 0.25)
    return np.asarray(sorted(values), dtype=np.float64)


def _curve_table(
    family: str,
    series_name: str,
    display_label: str,
    episodes: Sequence[Episode],
    grid: np.ndarray,
    *,
    grid_type: str,
) -> list[dict[str, Any]]:
    from cmc_bbdm.cai_agent_v3.metrics import prediction_at_budget

    keys = sorted({episode.specimen_key for episode in episodes})
    runs = sorted({episode.run for episode in episodes})
    lookup = {(episode.run, episode.specimen_key): episode for episode in episodes}
    if set(lookup) != {(run, key) for run in runs for key in keys}:
        raise ValueError(f"curve series is not a complete repeat panel: {series_name}")
    targets = np.asarray([lookup[(runs[0], key)].target_mpa for key in keys])
    denominator = float(np.sum(np.square(targets - targets.mean())))
    output = []
    for budget in grid:
        prediction = np.empty((len(runs), len(keys)), dtype=np.float64)
        actual = np.empty_like(prediction)
        for run_index, run in enumerate(runs):
            for key_index, key in enumerate(keys):
                episode = lookup[(run, key)]
                prediction[run_index, key_index] = prediction_at_budget(
                    episode.costs, episode.predictions_mpa, float(budget)
                )
                state = int(np.searchsorted(episode.costs, budget, side="right") - 1)
                actual[run_index, key_index] = episode.costs[state]
        residual = prediction - targets[None, :]
        per_specimen_abs = np.mean(np.abs(residual), axis=0)
        per_specimen_sq = np.mean(np.square(residual), axis=0)
        r2 = (
            None
            if denominator == 0.0
            else float(
                np.mean(
                    1.0 - np.sum(np.square(residual), axis=1) / denominator
                )
            )
        )
        output.append(
            {
                "identity_family": family,
                "cohort": "VALID50_CAPTURE48_DOMAIN6",
                "clock": (
                    "ARCHIVED_FLOAT64"
                    if family == "ARCHIVED_METHODS"
                    else "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL"
                ),
                "series": series_name,
                "series_display_label": display_label,
                "repeat_count": len(runs),
                "grid_type": grid_type,
                "requested_cost": float(budget),
                "mae_mpa": float(per_specimen_abs.mean()),
                "rmse_mpa": float(np.sqrt(per_specimen_sq.mean())),
                "r2": r2,
                "actual_cost_mean": float(actual.mean()),
                "actual_cost_min": float(actual.min()),
                "actual_cost_max": float(actual.max()),
                "physical_n": len(keys),
                "units": "error=MPa; cost=fraction",
                "aggregation": "REPEAT_LOSS_MEAN_THEN_PHYSICAL_SPECIMEN_POOL",
            }
        )
    return output


def analyze_results(context: TaskContext) -> list[Path]:
    results = context.results_root
    _archived_rows, archived_episodes = _archive_panel(context)
    reorder_rows = read_gzip_csv_rows(results / "reorder_trajectories.csv.gz")
    if len(reorder_rows) != 350:
        raise ValueError("reorder trajectory panel must contain 350 rows")
    order_episodes = []
    for row in reorder_rows:
        order_episodes.append(
            parse_episode(
                {
                    **row,
                    "method": row["variant"],
                    "run": row["repeat"],
                }
            )
        )
    cohort = {
        row["specimen_key"]: row
        for row in read_csv_rows(results / "cohort_manifest.csv")
    }
    keys = sorted(cohort)
    domains = np.asarray([cohort[key]["dataset_id"] for key in keys])
    groups = np.asarray([cohort[key]["capture_group_id"] for key in keys])
    weights = load_bootstrap_weights(context, keys, domains, groups)

    order_rows = []
    for episode, source in zip(order_episodes, reorder_rows, strict=True):
        timing = timing_decomposition(
            episode.costs, episode.predictions_mpa, episode.target_mpa
        )
        calculated_early = left_error_area_mpa(
            episode.costs,
            episode.predictions_mpa,
            episode.target_mpa,
            end=EARLY_HORIZON,
        )
        if (
            abs(timing.area_mpa - float(source["left_error_area_mpa"])) > 1e-10
            or abs(calculated_early - float(source["early_left_error_area_mpa"]))
            > 1e-10
        ):
            raise ValueError("stored reorder metric differs from recomputation")
        order_rows.append(
            {
                "identity_family": "FIXED_SET_ORDERS",
                "cohort": "VALID50_CAPTURE48_DOMAIN6",
                "clock": "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL",
                "record_level": "TRAJECTORY",
                "order_id": source["order_id"],
                "specimen_key": episode.specimen_key,
                "dataset_id": episode.dataset_id,
                "capture_group_id": episode.capture_group_id,
                "variant": source["variant"],
                "variant_display_label": source["variant_display_label"],
                "repeat": episode.run,
                "repeat_count": 1,
                "area_mpa": timing.area_mpa,
                "early_area_mpa": calculated_early,
                "objective_mpa": float(source["trajectory_objective_mpa"]),
                "initial_error_mpa": timing.initial_error,
                "final_error_mpa": timing.final_error,
                "action_count": len(episode.cells),
                "final_pixels": int(source["final_pixels"]),
                **{
                    f"raw_stage_{stage + 1}_mpa": float(timing.raw_stage[stage])
                    for stage in range(4)
                },
                **{
                    f"weighted_stage_{stage + 1}_mpa": float(
                        timing.weighted_stage[stage]
                    )
                    for stage in range(4)
                },
                "units": "error,area,objective,stage=MPa; pixels=count",
                "aggregation": "TRAJECTORY",
            }
        )

    value_fields = (
        "area_mpa",
        "early_area_mpa",
        "objective_mpa",
        "initial_error_mpa",
        "final_error_mpa",
        "action_count",
        "final_pixels",
        *(f"raw_stage_{stage}_mpa" for stage in range(1, 5)),
        *(f"weighted_stage_{stage}_mpa" for stage in range(1, 5)),
    )
    permuted = [row for row in order_rows if row["variant"] == "PERMUTED"]
    permuted_mean = _group_mean(
        permuted,
        ("specimen_key", "dataset_id", "capture_group_id"),
        value_fields,
    )
    for row in permuted_mean:
        order_rows.append(
            {
                "identity_family": "FIXED_SET_ORDERS",
                "cohort": "VALID50_CAPTURE48_DOMAIN6",
                "clock": "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL",
                "record_level": "REPEAT_MEAN",
                "order_id": f"{row['specimen_key']}|PERMUTED_MEAN",
                "variant": "PERMUTED_MEAN",
                "variant_display_label": "Permuted order (5-repeat mean)",
                "repeat": "MEAN_0_TO_4",
                **row,
                "units": "error,area,objective,stage=MPa; pixels=count",
                "aggregation": "FIVE_PRESET_PERMUTATION_REPEAT_MEAN",
            }
        )

    selected_order_rows = [
        row
        for row in order_rows
        if row["variant"] in {"NATIVE_REPLAY", "REVERSE"}
        or row["record_level"] == "REPEAT_MEAN"
    ]
    series_definitions = {
        "NATIVE_REPLAY": (
            "Native order",
            [episode for episode in order_episodes if episode.method == "NATIVE_REPLAY"],
        ),
        "REVERSE": (
            "Reverse order",
            [episode for episode in order_episodes if episode.method == "REVERSE"],
        ),
        "PERMUTED_MEAN": (
            "Permuted order (5-repeat mean)",
            [episode for episode in order_episodes if episode.method == "PERMUTED"],
        ),
    }
    order_summary = []
    order_by_domain = []
    for series, (label, episodes) in series_definitions.items():
        panel = [row for row in selected_order_rows if row["variant"] == series]
        curve_endpoint = _curve_table(
            "FIXED_SET_ORDERS",
            series,
            label,
            episodes,
            np.asarray([0.25]),
            grid_type="ENDPOINT_ONLY",
        )[0]
        order_summary.append(
            {
                "identity_family": "FIXED_SET_ORDERS",
                "cohort": "VALID50_CAPTURE48_DOMAIN6",
                "clock": "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL",
                "variant": series,
                "variant_display_label": label,
                "repeat_count": len(episodes) // 50,
                "area_mpa": domain_equal(
                    np.asarray([row["area_mpa"] for row in panel]),
                    np.asarray([row["dataset_id"] for row in panel]),
                ),
                "early_area_mpa": domain_equal(
                    np.asarray([row["early_area_mpa"] for row in panel]),
                    np.asarray([row["dataset_id"] for row in panel]),
                ),
                "endpoint_mae_mpa": curve_endpoint["mae_mpa"],
                "endpoint_rmse_mpa": curve_endpoint["rmse_mpa"],
                "endpoint_r2": curve_endpoint["r2"],
                **{
                    f"weighted_stage_{stage}_mpa": domain_equal(
                        np.asarray(
                            [row[f"weighted_stage_{stage}_mpa"] for row in panel]
                        ),
                        np.asarray([row["dataset_id"] for row in panel]),
                    )
                    for stage in range(1, 5)
                },
                "physical_n": 50,
                "units": "error,area,stage=MPa",
                "aggregation": (
                    "REPEAT_MEAN_THEN_DOMAIN_MEAN_THEN_SIX_DOMAIN_EQUAL_FOR_AREA;"
                    "REPEAT_LOSS_MEAN_THEN_PHYSICAL_POOL_FOR_ENDPOINT"
                ),
            }
        )
        for domain in sorted(set(domains)):
            local = [row for row in panel if row["dataset_id"] == domain]
            order_by_domain.append(
                {
                    "identity_family": "FIXED_SET_ORDERS",
                    "cohort": "VALID50_CAPTURE48_DOMAIN6",
                    "clock": "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL",
                    "record_type": "SERIES_ESTIMATE",
                    "dataset_id": domain,
                    "variant": series,
                    "variant_display_label": label,
                    "area_mpa": float(np.mean([row["area_mpa"] for row in local])),
                    "early_area_mpa": float(
                        np.mean([row["early_area_mpa"] for row in local])
                    ),
                    "physical_n": len(local),
                    "units": "area=MPa",
                    "aggregation": "REPEAT_MEAN_THEN_SPECIMEN_MEAN_WITHIN_DOMAIN",
                }
            )

    specimen_lookup = {
        (row["variant"], row["specimen_key"]): row for row in selected_order_rows
    }
    contrast_specs = (
        ("PRIMARY_PERMUTED_MEAN_MINUS_NATIVE", "PERMUTED_MEAN"),
        ("SECONDARY_REVERSE_MINUS_NATIVE", "REVERSE"),
    )
    paired_rows = []
    for contrast, comparator in contrast_specs:
        metric_specs = [
            ("area_mpa", "DOMAIN_EQUAL"),
            ("early_area_mpa", "DOMAIN_EQUAL"),
            ("objective_mpa", "DOMAIN_EQUAL"),
            ("final_error_mpa", "POOLED_MAE"),
            *((f"weighted_stage_{stage}_mpa", "DOMAIN_EQUAL") for stage in range(1, 5)),
        ]
        for metric, estimand in metric_specs:
            difference = np.asarray(
                [
                    specimen_lookup[(comparator, key)][metric]
                    - specimen_lookup[("NATIVE_REPLAY", key)][metric]
                    for key in keys
                ]
            )
            if estimand == "DOMAIN_EQUAL":
                estimate = domain_equal(difference, domains)
                draws = bootstrap_domain_equal(difference, weights, domains)
            else:
                estimate = float(difference.mean())
                draws = bootstrap_pooled(difference, weights)
            paired_rows.append(
                {
                    "identity_family": "FIXED_SET_ORDERS",
                    "cohort": "VALID50_CAPTURE48_DOMAIN6",
                    "clock": "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL",
                    "contrast": contrast,
                    "main_variant": "NATIVE_REPLAY",
                    "comparator_variant": comparator,
                    "metric": metric,
                    "estimand": estimand,
                    "difference_definition": (
                        "COMPARATOR_MINUS_NATIVE_POSITIVE_FAVORS_NATIVE"
                    ),
                    "estimate_mpa": estimate,
                    "ci_low_mpa": float(np.quantile(draws, 0.025)),
                    "ci_high_mpa": float(np.quantile(draws, 0.975)),
                    "positive_favors": "NATIVE_REPLAY",
                    "bootstrap_replicates": 5000,
                    "units": "MPa",
                    "aggregation": (
                        "FIXED_CAPTURE_GROUP_WITHIN_DOMAIN_PAIRED_BOOTSTRAP"
                    ),
                }
            )
        for domain in sorted(set(domains)):
            local_keys = [key for key in keys if cohort[key]["dataset_id"] == domain]
            for metric in ("area_mpa", "early_area_mpa"):
                difference = [
                    specimen_lookup[(comparator, key)][metric]
                    - specimen_lookup[("NATIVE_REPLAY", key)][metric]
                    for key in local_keys
                ]
                order_by_domain.append(
                    {
                        "identity_family": "FIXED_SET_ORDERS",
                        "cohort": "VALID50_CAPTURE48_DOMAIN6",
                        "clock": (
                            "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL"
                        ),
                        "record_type": "PAIRED_EFFECT",
                        "dataset_id": domain,
                        "contrast": contrast,
                        "main_variant": "NATIVE_REPLAY",
                        "comparator_variant": comparator,
                        "metric": metric,
                        "difference_definition": (
                            "COMPARATOR_MINUS_NATIVE_POSITIVE_FAVORS_NATIVE"
                        ),
                        "estimate_mpa": float(np.mean(difference)),
                        "physical_n": len(local_keys),
                        "units": "MPa",
                        "aggregation": "SPECIMEN_PAIRED_MEAN_WITHIN_DOMAIN",
                    }
                )

    j_residual = []
    endpoint_residual = []
    for key in keys:
        native = specimen_lookup[("NATIVE_REPLAY", key)]
        perm = specimen_lookup[("PERMUTED_MEAN", key)]
        j_residual.append(
            (perm["objective_mpa"] - native["objective_mpa"])
            - (perm["area_mpa"] - native["area_mpa"])
        )
        endpoint_residual.append(perm["final_error_mpa"] - native["final_error_mpa"])
    structure_max = max(float(np.max(np.abs(j_residual))), float(np.max(np.abs(endpoint_residual))))
    if structure_max > 1e-4:
        raise ValueError("fixed-set endpoint/J/A structure identity failed")
    for row in paired_rows:
        row["max_endpoint_and_j_minus_a_residual_mpa"] = structure_max

    archived_series = {
        method: (
            DISPLAY_LABELS[method],
            [episode for episode in archived_episodes if episode.method == method],
        )
        for method in ARCHIVED_METHODS
    }
    family_series = {
        "ARCHIVED_METHODS": archived_series,
        "FIXED_SET_ORDERS": series_definitions,
    }
    cap_grid = np.asarray([0.0, 0.0625, 0.125, 0.1875, 0.25])
    cap_rows = []
    curve_rows = []
    family_grids = {}
    for family, definitions in family_series.items():
        family_episodes = [episode for _label, series in definitions.values() for episode in series]
        event_grid = _event_grid(family_episodes)
        family_grids[family] = event_grid
        for series, (label, episodes) in definitions.items():
            cap_rows.extend(
                _curve_table(
                    family,
                    series,
                    label,
                    episodes,
                    cap_grid,
                    grid_type="FIVE_FIXED_CAPS",
                )
            )
            curve_rows.extend(
                _curve_table(
                    family,
                    series,
                    label,
                    episodes,
                    event_grid,
                    grid_type="FULL_EVENT_UNION",
                )
            )

    quality_targets = [float(value) for value in range(41, 62)] + [
        41.69001007080078
    ]
    matched_rows = []
    curve_sources = {"FIVE_FIXED_CAPS": cap_rows, "FULL_EVENT_UNION": curve_rows}
    for family, definitions in family_series.items():
        main_series = MAIN_METHOD if family == "ARCHIVED_METHODS" else "NATIVE_REPLAY"
        comparators = [series for series in definitions if series != main_series]
        for grid_type, source_rows in curve_sources.items():
            family_rows = [
                row
                for row in source_rows
                if row["identity_family"] == family and row["grid_type"] == grid_type
            ]
            costs = np.asarray(
                [
                    row["requested_cost"]
                    for row in family_rows
                    if row["series"] == main_series
                ]
            )
            main_values = np.asarray(
                [
                    row["mae_mpa"]
                    for row in family_rows
                    if row["series"] == main_series
                ]
            )
            for comparator in comparators:
                comparator_values = np.asarray(
                    [
                        row["mae_mpa"]
                        for row in family_rows
                        if row["series"] == comparator
                    ]
                )
                for target in quality_targets:
                    comparison = compare_quality(
                        costs,
                        main_values=main_values,
                        comparator_values=comparator_values,
                        target=target,
                    )
                    matched_rows.append(
                        {
                            "identity_family": family,
                            "cohort": "VALID50_CAPTURE48_DOMAIN6",
                            "clock": (
                                "ARCHIVED_FLOAT64"
                                if family == "ARCHIVED_METHODS"
                                else (
                                    "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL"
                                )
                            ),
                            "grid_type": grid_type,
                            "main_series": main_series,
                            "comparator_series": comparator,
                            "difference_definition": (
                                "COMPARATOR_COST_MINUS_MAIN_COST_POSITIVE_FAVORS_MAIN"
                            ),
                            **comparison,
                            "units": "quality=MPa; cost,saving=fraction",
                            "aggregation": (
                                "QUEUE_MAE_FIRST_CROSSING_NOT_SPECIMEN_ORACLE"
                            ),
                        }
                    )

    return [
        write_csv_atomic(results / "order_specimen_metrics.csv", order_rows),
        write_csv_atomic(results / "order_summary.csv", order_summary),
        write_csv_atomic(results / "order_paired_contrasts.csv", paired_rows),
        write_csv_atomic(results / "order_by_domain.csv", order_by_domain),
        write_csv_atomic(results / "same_cap_metrics.csv", cap_rows),
        write_gzip_csv_atomic(results / "cost_error_curves.csv.gz", curve_rows),
        write_csv_atomic(results / "matched_quality.csv", matched_rows),
    ]


__all__ = [
    "EARLY_HORIZON",
    "HORIZON",
    "STAGE_EDGES",
    "Episode",
    "TimingResult",
    "analyze_results",
    "bootstrap_domain_equal",
    "bootstrap_pooled",
    "compare_quality",
    "derive_archived",
    "domain_equal",
    "endpoint_metrics",
    "first_quality_crossing",
    "parse_episode",
    "timing_decomposition",
]
