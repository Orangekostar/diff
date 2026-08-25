"""Specimen-first and equal-domain metrics for dynamic action valuation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import rankdata

from .dynamic_data import DynamicStateGroup


class MAVISDynamicMetricError(ValueError):
    """Raised when P3 score arrays or statistical units are invalid."""


_METRICS = (
    "next_action_regret",
    "one_step_cai_utility",
    "spearman",
    "ndcg",
    "recall_at_k",
)


@dataclass(frozen=True, slots=True)
class DynamicMetricTables:
    per_specimen: pl.DataFrame
    per_domain: pl.DataFrame
    aggregate: pl.DataFrame


def _spearman(scores: np.ndarray, teacher: np.ndarray) -> float:
    score_ranks = rankdata(scores, method="average")
    teacher_ranks = rankdata(teacher, method="average")
    score_centered = score_ranks - np.mean(score_ranks)
    teacher_centered = teacher_ranks - np.mean(teacher_ranks)
    denominator = float(
        np.linalg.norm(score_centered) * np.linalg.norm(teacher_centered)
    )
    if denominator == 0.0:
        return 1.0 if np.array_equal(score_ranks, teacher_ranks) else 0.0
    return float(np.dot(score_centered, teacher_centered) / denominator)


def _ndcg(scores: np.ndarray, teacher: np.ndarray) -> float:
    gains = teacher - np.min(teacher)
    if np.max(gains) == 0.0:
        return 1.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    predicted_order = np.argsort(-scores, kind="stable")
    ideal_order = np.argsort(-teacher, kind="stable")
    observed = float(np.sum(gains[predicted_order] * discounts))
    ideal = float(np.sum(gains[ideal_order] * discounts))
    if not ideal > 0.0:
        raise MAVISDynamicMetricError("P3 ideal DCG is invalid")
    return observed / ideal


def _recall(scores: np.ndarray, teacher: np.ndarray, k: int) -> float:
    count = min(k, scores.size)
    predicted = set(np.argsort(-scores, kind="stable")[:count].tolist())
    expected = set(np.argsort(-teacher, kind="stable")[:count].tolist())
    return len(predicted & expected) / count


def evaluate_dynamic_scores(
    groups: tuple[DynamicStateGroup, ...],
    score_arrays: tuple[np.ndarray, ...],
    *,
    mode: str,
    recall_k: int,
) -> pl.DataFrame:
    if (
        type(groups) is not tuple
        or not groups
        or any(type(group) is not DynamicStateGroup for group in groups)
        or type(score_arrays) is not tuple
        or len(score_arrays) != len(groups)
        or type(mode) is not str
        or not mode
        or type(recall_k) is not int
        or recall_k <= 0
    ):
        raise MAVISDynamicMetricError("P3 metric request is invalid")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for group, raw_scores in zip(groups, score_arrays, strict=True):
        scores = np.asarray(raw_scores, dtype=np.float64)
        teacher = np.asarray(group.teacher_values, dtype=np.float64)
        if (
            group.state_id in seen
            or scores.shape != teacher.shape
            or scores.ndim != 1
            or scores.size == 0
            or not np.all(np.isfinite(scores))
            or not np.all(np.isfinite(teacher))
        ):
            raise MAVISDynamicMetricError("P3 state score roster is invalid")
        seen.add(group.state_id)
        selected = int(np.argmax(scores))
        best = int(np.argmax(teacher))
        regret = float(teacher[best] - teacher[selected])
        if regret < -1.0e-12:
            raise MAVISDynamicMetricError("P3 next-action regret is invalid")
        rows.append(
            {
                "schema_version": 1,
                "outer_domain": group.outer_domain,
                "domain_id": group.domain_id,
                "specimen_id": group.specimen_id,
                "state_id": group.state_id,
                "mode": mode,
                "candidate_count": len(group.candidates),
                "selected_candidate_index": selected,
                "selected_cell_index": group.candidates[selected].cell_index,
                "selected_from_level": group.candidates[selected].from_level,
                "selected_to_level": group.candidates[selected].to_level,
                "selected_exact_added_cost": (
                    group.candidates[selected].exact_added_cost
                ),
                "oracle_candidate_index": best,
                "next_action_regret": max(regret, 0.0),
                "one_step_cai_utility": float(teacher[selected]),
                "spearman": _spearman(scores, teacher),
                "ndcg": _ndcg(scores, teacher),
                "recall_at_k": _recall(scores, teacher, recall_k),
                "recall_k": min(recall_k, scores.size),
                "teacher_fold_count": group.teacher_fold_count,
                "group_state_sha256": group.state_sha256,
            }
        )
    table = pl.DataFrame(rows, infer_schema_length=None).sort(
        ["outer_domain", "specimen_id", "state_id", "mode"]
    )
    if table.select(pl.any_horizontal(pl.selectors.numeric().is_nan())).to_series().any():
        raise MAVISDynamicMetricError("P3 metric table contains NaN")
    return table


def aggregate_dynamic_metrics(per_state: pl.DataFrame) -> DynamicMetricTables:
    required = {
        "outer_domain",
        "domain_id",
        "specimen_id",
        "state_id",
        "mode",
        *_METRICS,
    }
    if (
        not isinstance(per_state, pl.DataFrame)
        or per_state.height == 0
        or not required <= set(per_state.columns)
        or per_state.select(pl.any_horizontal(pl.selectors.numeric().is_nan()))
        .to_series()
        .any()
    ):
        raise MAVISDynamicMetricError("P3 per-state metric table is invalid")
    keys = ["outer_domain", "domain_id", "mode"]
    per_specimen = (
        per_state.group_by([*keys, "specimen_id"])
        .agg(
            pl.len().alias("state_count"),
            *(pl.col(metric).mean().alias(metric) for metric in _METRICS),
        )
        .with_columns(pl.lit("physical_specimen").alias("statistical_unit"))
        .sort([*keys, "specimen_id"])
    )
    per_domain = (
        per_specimen.group_by(keys)
        .agg(
            pl.len().alias("specimen_count"),
            *(pl.col(metric).mean().alias(metric) for metric in _METRICS),
        )
        .with_columns(pl.lit("held_out_domain").alias("statistical_unit"))
        .sort(keys)
    )
    aggregate = (
        per_domain.group_by(["mode"])
        .agg(
            pl.len().alias("domain_count"),
            *(pl.col(metric).mean().alias(metric) for metric in _METRICS),
        )
        .with_columns(pl.lit("equal_domain").alias("statistical_unit"))
        .sort("mode")
    )
    for table in (per_specimen, per_domain, aggregate):
        if table.select(pl.any_horizontal(pl.selectors.numeric().is_nan())).to_series().any():
            raise MAVISDynamicMetricError("P3 aggregate metric table contains NaN")
    return DynamicMetricTables(
        per_specimen=per_specimen,
        per_domain=per_domain,
        aggregate=aggregate,
    )


def bootstrap_dynamic_contrasts(
    per_specimen: pl.DataFrame,
    *,
    reference_mode: str,
    control_modes: tuple[str, ...],
    domain_order: tuple[str, ...],
    replicates: int,
    seed: int,
) -> pl.DataFrame:
    required = {
        "outer_domain",
        "specimen_id",
        "mode",
        "next_action_regret",
        "one_step_cai_utility",
    }
    if (
        not isinstance(per_specimen, pl.DataFrame)
        or per_specimen.height == 0
        or not required <= set(per_specimen.columns)
        or type(reference_mode) is not str
        or not reference_mode
        or type(control_modes) is not tuple
        or not control_modes
        or reference_mode in control_modes
        or len(set(control_modes)) != len(control_modes)
        or type(domain_order) is not tuple
        or not domain_order
        or len(set(domain_order)) != len(domain_order)
        or type(replicates) is not int
        or replicates <= 0
        or type(seed) is not int
    ):
        raise MAVISDynamicMetricError("P3 bootstrap request is invalid")
    modes = {reference_mode, *control_modes}
    if not modes <= set(per_specimen.get_column("mode").unique()):
        raise MAVISDynamicMetricError("P3 bootstrap mode roster is incomplete")
    lookup: dict[tuple[str, str, str], tuple[float, float]] = {}
    specimens: dict[str, tuple[str, ...]] = {}
    for domain in domain_order:
        table = per_specimen.filter(pl.col("outer_domain") == domain)
        ids = tuple(sorted(table.get_column("specimen_id").unique()))
        if not ids:
            raise MAVISDynamicMetricError("P3 bootstrap domain roster is empty")
        specimens[domain] = ids
        for specimen_id in ids:
            for mode in modes:
                row = table.filter(
                    (pl.col("specimen_id") == specimen_id)
                    & (pl.col("mode") == mode)
                )
                if row.height != 1:
                    raise MAVISDynamicMetricError(
                        "P3 bootstrap specimen/mode roster is incomplete"
                    )
                lookup[(domain, specimen_id, mode)] = (
                    float(row.item(0, "next_action_regret")),
                    float(row.item(0, "one_step_cai_utility")),
                )
    generator = np.random.Generator(np.random.PCG64(seed))
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        sampled = {
            domain: tuple(
                ids[index]
                for index in generator.integers(0, len(ids), size=len(ids))
            )
            for domain, ids in specimens.items()
        }
        for control in control_modes:
            regret_differences: list[float] = []
            utility_differences: list[float] = []
            for domain in domain_order:
                reference_values = [
                    lookup[(domain, specimen_id, reference_mode)]
                    for specimen_id in sampled[domain]
                ]
                control_values = [
                    lookup[(domain, specimen_id, control)]
                    for specimen_id in sampled[domain]
                ]
                regret_differences.append(
                    float(
                        np.mean([value[0] for value in control_values])
                        - np.mean([value[0] for value in reference_values])
                    )
                )
                utility_differences.append(
                    float(
                        np.mean([value[1] for value in reference_values])
                        - np.mean([value[1] for value in control_values])
                    )
                )
            rows.append(
                {
                    "replicate": replicate,
                    "reference_mode": reference_mode,
                    "control_mode": control,
                    "control_minus_reference_regret": float(
                        np.mean(regret_differences, dtype=np.float64)
                    ),
                    "reference_minus_control_utility": float(
                        np.mean(utility_differences, dtype=np.float64)
                    ),
                }
            )
    output = pl.DataFrame(rows).sort(["control_mode", "replicate"])
    if output.select(pl.any_horizontal(pl.selectors.numeric().is_nan())).to_series().any():
        raise MAVISDynamicMetricError("P3 bootstrap contains NaN")
    return output


__all__ = [
    "DynamicMetricTables",
    "MAVISDynamicMetricError",
    "aggregate_dynamic_metrics",
    "bootstrap_dynamic_contrasts",
    "evaluate_dynamic_scores",
]
