"""Specimen-first, domain-balanced MRIS metrics and paired bootstrap."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl


class MAVISMRISMetricError(ValueError):
    """Raised when MRIS predictions cannot support registered inference."""


_REQUIRED_PREDICTION_COLUMNS = {
    "outer_domain",
    "state_id",
    "specimen_id",
    "trajectory_id",
    "method",
    "nominal_checkpoint",
    "exact_acquired_cost",
    "native_count",
    "effective_budget",
    "mode",
    "target",
    "prediction",
    "absolute_error",
    "model_state_sha256",
}


@dataclass(frozen=True, slots=True)
class MRISMetricTables:
    per_specimen_metrics: pl.DataFrame
    domain_metrics: pl.DataFrame
    aggregate_metrics: pl.DataFrame
    domain_auebc: pl.DataFrame
    aggregate_auebc: pl.DataFrame


def _domain_order(value: object) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or len(value) < 2
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise MAVISMRISMetricError("MRIS domain order is invalid")
    return value


def _auebc(cost: np.ndarray, error: np.ndarray) -> float:
    order = np.argsort(cost, kind="stable")
    x = np.asarray(cost[order], dtype=np.float64)
    y = np.asarray(error[order], dtype=np.float64)
    if (
        x.ndim != 1
        or x.size == 0
        or y.shape != x.shape
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(np.diff(x) <= 0.0)
    ):
        raise MAVISMRISMetricError("MRIS curve is invalid")
    if x.size == 1:
        return float(y[0])
    return float(np.trapezoid(y, x=x) / (x[-1] - x[0]))


def evaluate_mris_predictions(
    predictions: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
) -> MRISMetricTables:
    domains = _domain_order(domain_order)
    if (
        not isinstance(predictions, pl.DataFrame)
        or predictions.height == 0
        or not _REQUIRED_PREDICTION_COLUMNS <= set(predictions.columns)
        or set(predictions.get_column("outer_domain").unique()) != set(domains)
        or predictions.get_column("state_id").n_unique()
        * predictions.get_column("mode").n_unique()
        != predictions.height
    ):
        raise MAVISMRISMetricError("MRIS prediction roster is invalid")
    numeric = predictions.select(
        "nominal_checkpoint",
        "exact_acquired_cost",
        "native_count",
        "effective_budget",
        "target",
        "prediction",
        "absolute_error",
    )
    if (
        numeric.select(pl.any_horizontal(pl.all().is_nan().any())).item()
        or not predictions.select(
            (
                (pl.col("native_count") > 0)
                & (pl.col("exact_acquired_cost") > 0)
                & (
                    (
                        pl.col("effective_budget")
                        - pl.col("exact_acquired_cost") / pl.col("native_count")
                    ).abs()
                    <= 1.0e-15
                )
                & (
                    (
                        pl.col("absolute_error")
                        - (pl.col("target") - pl.col("prediction")).abs()
                    ).abs()
                    <= 1.0e-15
                )
            ).all()
        ).item()
    ):
        raise MAVISMRISMetricError("MRIS prediction values are invalid")
    per_specimen = (
        predictions.group_by(
            "outer_domain",
            "specimen_id",
            "mode",
            "nominal_checkpoint",
        )
        .agg(
            pl.col("exact_acquired_cost").mean().alias("mean_exact_acquired_cost"),
            pl.col("effective_budget").mean().alias("mean_effective_budget"),
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("trajectory_id").n_unique().alias("trajectory_count"),
            pl.len().alias("state_count"),
        )
        .sort(["outer_domain", "specimen_id", "mode", "nominal_checkpoint"])
    )
    domain_metrics = (
        per_specimen.group_by("outer_domain", "mode", "nominal_checkpoint")
        .agg(
            pl.col("specimen_id").n_unique().alias("specimen_count"),
            pl.col("mean_exact_acquired_cost")
            .mean()
            .alias("mean_exact_acquired_cost"),
            pl.col("mean_effective_budget").mean().alias("mean_effective_budget"),
            pl.col("mae").mean().alias("mae"),
        )
        .sort(["outer_domain", "mode", "nominal_checkpoint"])
    )
    aggregate = (
        domain_metrics.group_by("mode", "nominal_checkpoint")
        .agg(
            pl.col("outer_domain").n_unique().alias("domain_count"),
            pl.col("mean_exact_acquired_cost")
            .mean()
            .alias("mean_exact_acquired_cost"),
            pl.col("mean_effective_budget").mean().alias("mean_effective_budget"),
            pl.col("mae").mean().alias("domain_balanced_mae"),
            pl.col("mae").max().alias("worst_domain_mae"),
        )
        .sort(["mode", "nominal_checkpoint"])
    )
    if aggregate.get_column("domain_count").unique().to_list() != [len(domains)]:
        raise MAVISMRISMetricError("MRIS aggregate domain roster is incomplete")
    domain_auebc_rows: list[dict[str, object]] = []
    for (domain, mode), group in domain_metrics.group_by(
        "outer_domain", "mode", maintain_order=True
    ):
        domain_auebc_rows.append(
            {
                "outer_domain": domain,
                "mode": mode,
                "auebc": _auebc(
                    group.get_column("mean_effective_budget").to_numpy(),
                    group.get_column("mae").to_numpy(),
                ),
            }
        )
    domain_auebc = pl.DataFrame(domain_auebc_rows).sort(["outer_domain", "mode"])
    aggregate_auebc = (
        domain_auebc.group_by("mode")
        .agg(
            pl.col("outer_domain").n_unique().alias("domain_count"),
            pl.col("auebc").mean().alias("domain_balanced_auebc"),
            pl.col("auebc").max().alias("worst_domain_auebc"),
        )
        .sort("mode")
    )
    return MRISMetricTables(
        per_specimen_metrics=per_specimen,
        domain_metrics=domain_metrics,
        aggregate_metrics=aggregate,
        domain_auebc=domain_auebc,
        aggregate_auebc=aggregate_auebc,
    )


def _specimen_curve_auebc(table: pl.DataFrame, specimen_id: str, mode: str) -> float:
    curve = table.filter(
        (pl.col("specimen_id") == specimen_id) & (pl.col("mode") == mode)
    ).sort("nominal_checkpoint")
    return _auebc(
        curve.get_column("mean_effective_budget").to_numpy(),
        curve.get_column("mae").to_numpy(),
    )


def bootstrap_mris_contrasts(
    per_specimen_metrics: pl.DataFrame,
    *,
    reference_mode: str,
    control_modes: tuple[str, ...],
    domain_order: tuple[str, ...],
    replicates: int,
    seed: int,
) -> pl.DataFrame:
    domains = _domain_order(domain_order)
    if (
        not isinstance(per_specimen_metrics, pl.DataFrame)
        or per_specimen_metrics.height == 0
        or type(reference_mode) is not str
        or not reference_mode
        or type(control_modes) is not tuple
        or not control_modes
        or reference_mode in control_modes
        or len(set(control_modes)) != len(control_modes)
        or type(replicates) is not int
        or replicates <= 0
        or type(seed) is not int
    ):
        raise MAVISMRISMetricError("MRIS bootstrap request is invalid")
    modes = set(per_specimen_metrics.get_column("mode").unique())
    if not {reference_mode, *control_modes} <= modes:
        raise MAVISMRISMetricError("MRIS bootstrap mode roster is incomplete")
    specimen_auebc: dict[tuple[str, str, str], float] = {}
    specimens_by_domain: dict[str, tuple[str, ...]] = {}
    for domain in domains:
        domain_table = per_specimen_metrics.filter(pl.col("outer_domain") == domain)
        specimens = tuple(sorted(domain_table.get_column("specimen_id").unique()))
        if not specimens:
            raise MAVISMRISMetricError("MRIS bootstrap domain roster is empty")
        specimens_by_domain[domain] = specimens
        for specimen_id in specimens:
            for mode in (reference_mode, *control_modes):
                specimen_auebc[(domain, specimen_id, mode)] = _specimen_curve_auebc(
                    domain_table,
                    specimen_id,
                    mode,
                )
    generator = np.random.Generator(np.random.PCG64(seed))
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        sampled = {
            domain: tuple(
                specimens[index]
                for index in generator.integers(
                    0,
                    len(specimens),
                    size=len(specimens),
                )
            )
            for domain, specimens in specimens_by_domain.items()
        }
        reference_domains = [
            float(
                np.mean(
                    [
                        specimen_auebc[(domain, specimen_id, reference_mode)]
                        for specimen_id in sampled[domain]
                    ],
                    dtype=np.float64,
                )
            )
            for domain in domains
        ]
        reference = float(np.mean(reference_domains, dtype=np.float64))
        for control_mode in control_modes:
            control_domains = [
                float(
                    np.mean(
                        [
                            specimen_auebc[(domain, specimen_id, control_mode)]
                            for specimen_id in sampled[domain]
                        ],
                        dtype=np.float64,
                    )
                )
                for domain in domains
            ]
            control = float(np.mean(control_domains, dtype=np.float64))
            delta = control - reference
            if not all(math.isfinite(value) for value in (reference, control, delta)):
                raise MAVISMRISMetricError("MRIS bootstrap draw is invalid")
            rows.append(
                {
                    "replicate": replicate,
                    "reference_mode": reference_mode,
                    "control_mode": control_mode,
                    "reference_auebc": reference,
                    "control_auebc": control,
                    "control_minus_reference_auebc": delta,
                }
            )
    return pl.DataFrame(rows).sort(["control_mode", "replicate"])


__all__ = [
    "MAVISMRISMetricError",
    "MRISMetricTables",
    "bootstrap_mris_contrasts",
    "evaluate_mris_predictions",
]
