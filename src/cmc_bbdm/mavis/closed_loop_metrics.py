"""Specimen-first CAI and reconstruction metrics for closed-loop MAVIS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


class MAVISClosedLoopMetricError(ValueError):
    """Raised when closed-loop curves violate exact-cost inference semantics."""


_COLUMNS = {
    "outer_domain",
    "specimen_id",
    "method",
    "nominal_checkpoint",
    "initial_budget",
    "action_count",
    "exact_acquired_cost",
    "native_count",
    "effective_budget",
    "target",
    "prediction",
    "absolute_error",
    "reconstruction_mse",
}


@dataclass(frozen=True, slots=True)
class ClosedLoopMetricTables:
    per_specimen_curve: pl.DataFrame
    domain_curve: pl.DataFrame
    aggregate_curve: pl.DataFrame
    specimen_auebc: pl.DataFrame
    domain_auebc: pl.DataFrame
    aggregate_auebc: pl.DataFrame


def _roster(value: object, label: str) -> tuple[str, ...] | tuple[float, ...]:
    if type(value) is not tuple or not value or len(set(value)) != len(value):
        raise MAVISClosedLoopMetricError(f"closed-loop {label} roster is invalid")
    return value


def _auebc(cost: np.ndarray, values: np.ndarray) -> float:
    order = np.argsort(cost, kind="stable")
    x = np.asarray(cost[order], dtype=np.float64)
    y = np.asarray(values[order], dtype=np.float64)
    if (
        x.ndim != 1
        or x.size == 0
        or y.shape != x.shape
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(np.diff(x) <= 0.0)
    ):
        raise MAVISClosedLoopMetricError("closed-loop curve is invalid")
    if x.size == 1:
        return float(y[0])
    return float(np.trapezoid(y, x=x) / (x[-1] - x[0]))


def evaluate_closed_loop_predictions(
    predictions: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    method_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
) -> ClosedLoopMetricTables:
    domains = _roster(domain_order, "domain")
    methods = _roster(method_order, "method")
    caps = _roster(checkpoints, "checkpoint")
    if (
        not isinstance(predictions, pl.DataFrame)
        or predictions.height == 0
        or not _COLUMNS <= set(predictions.columns)
        or set(predictions.get_column("outer_domain").unique()) != set(domains)
        or set(predictions.get_column("method").unique()) != set(methods)
        or set(predictions.get_column("nominal_checkpoint").unique()) != set(caps)
    ):
        raise MAVISClosedLoopMetricError("closed-loop prediction roster is invalid")
    valid = predictions.select(
        (
            (pl.col("native_count") > 0)
            & (pl.col("exact_acquired_cost") > 0)
            & (pl.col("initial_budget") > 0.0)
            & (pl.col("initial_budget") <= 1.0)
            & (pl.col("action_count") >= 0)
            & (
                (
                    pl.col("effective_budget")
                    - pl.col("exact_acquired_cost") / pl.col("native_count")
                ).abs()
                <= 1.0e-15
            )
            & (
                (pl.col("effective_budget") <= pl.col("nominal_checkpoint") + 1.0e-15)
                | (
                    (pl.col("action_count") == 0)
                    & (
                        (pl.col("nominal_checkpoint") - pl.col("initial_budget")).abs()
                        <= 1.0e-15
                    )
                )
            )
            & (
                (
                    pl.col("absolute_error")
                    - (pl.col("target") - pl.col("prediction")).abs()
                ).abs()
                <= 1.0e-12
            )
            & (pl.col("reconstruction_mse") >= 0.0)
        ).all()
    ).item()
    if (
        not valid
        or predictions.select(pl.any_horizontal(pl.selectors.numeric().is_nan()))
        .to_series()
        .any()
    ):
        raise MAVISClosedLoopMetricError("closed-loop prediction values are invalid")
    per_specimen = (
        predictions.group_by(
            "outer_domain",
            "specimen_id",
            "method",
            "nominal_checkpoint",
        )
        .agg(
            pl.col("exact_acquired_cost").mean().alias("mean_exact_acquired_cost"),
            pl.col("effective_budget").mean().alias("mean_effective_budget"),
            pl.col("absolute_error").mean().alias("cai_mae"),
            pl.col("reconstruction_mse").mean().alias("reconstruction_mse"),
            pl.len().alias("trajectory_count"),
        )
        .sort(["outer_domain", "specimen_id", "method", "nominal_checkpoint"])
    )
    per_specimen_counts = per_specimen.group_by(
        "outer_domain", "specimen_id", "method"
    ).len()
    if per_specimen_counts.get_column("len").unique().to_list() != [len(caps)]:
        raise MAVISClosedLoopMetricError("closed-loop specimen curve is incomplete")
    domain_curve = (
        per_specimen.group_by("outer_domain", "method", "nominal_checkpoint")
        .agg(
            pl.col("specimen_id").n_unique().alias("specimen_count"),
            pl.col("mean_exact_acquired_cost")
            .mean()
            .alias("mean_exact_acquired_cost"),
            pl.col("mean_effective_budget").mean().alias("mean_effective_budget"),
            pl.col("cai_mae").mean().alias("cai_mae"),
            pl.col("reconstruction_mse").mean().alias("reconstruction_mse"),
        )
        .sort(["outer_domain", "method", "nominal_checkpoint"])
    )
    aggregate_curve = (
        domain_curve.group_by("method", "nominal_checkpoint")
        .agg(
            pl.col("outer_domain").n_unique().alias("domain_count"),
            pl.col("mean_exact_acquired_cost")
            .mean()
            .alias("mean_exact_acquired_cost"),
            pl.col("mean_effective_budget").mean().alias("mean_effective_budget"),
            pl.col("cai_mae").mean().alias("domain_balanced_cai_mae"),
            pl.col("cai_mae").max().alias("worst_domain_cai_mae"),
            pl.col("reconstruction_mse")
            .mean()
            .alias("domain_balanced_reconstruction_mse"),
        )
        .sort(["method", "nominal_checkpoint"])
    )
    if aggregate_curve.get_column("domain_count").unique().to_list() != [len(domains)]:
        raise MAVISClosedLoopMetricError("closed-loop aggregate domains are incomplete")
    specimen_rows: list[dict[str, object]] = []
    for (domain, specimen, method), table in per_specimen.group_by(
        "outer_domain", "specimen_id", "method", maintain_order=True
    ):
        specimen_rows.append(
            {
                "outer_domain": domain,
                "specimen_id": specimen,
                "method": method,
                "cai_auebc": _auebc(
                    table.get_column("mean_effective_budget").to_numpy(),
                    table.get_column("cai_mae").to_numpy(),
                ),
                "reconstruction_auebc": _auebc(
                    table.get_column("mean_effective_budget").to_numpy(),
                    table.get_column("reconstruction_mse").to_numpy(),
                ),
                "statistical_unit": "physical_specimen",
            }
        )
    specimen_auebc = pl.DataFrame(specimen_rows).sort(
        ["outer_domain", "specimen_id", "method"]
    )
    domain_auebc = (
        specimen_auebc.group_by("outer_domain", "method")
        .agg(
            pl.col("specimen_id").n_unique().alias("specimen_count"),
            pl.col("cai_auebc").mean().alias("cai_auebc"),
            pl.col("reconstruction_auebc").mean().alias("reconstruction_auebc"),
        )
        .with_columns(pl.lit("held_out_domain").alias("statistical_unit"))
        .sort(["outer_domain", "method"])
    )
    aggregate_auebc = (
        domain_auebc.group_by("method")
        .agg(
            pl.col("outer_domain").n_unique().alias("domain_count"),
            pl.col("cai_auebc").mean().alias("domain_balanced_cai_auebc"),
            pl.col("cai_auebc").max().alias("worst_domain_cai_auebc"),
            pl.col("reconstruction_auebc")
            .mean()
            .alias("domain_balanced_reconstruction_auebc"),
        )
        .with_columns(pl.lit("equal_domain").alias("statistical_unit"))
        .sort("method")
    )
    return ClosedLoopMetricTables(
        per_specimen_curve=per_specimen,
        domain_curve=domain_curve,
        aggregate_curve=aggregate_curve,
        specimen_auebc=specimen_auebc,
        domain_auebc=domain_auebc,
        aggregate_auebc=aggregate_auebc,
    )


def bootstrap_closed_loop_contrasts(
    specimen_auebc: pl.DataFrame,
    *,
    reference_method: str,
    control_methods: tuple[str, ...],
    domain_order: tuple[str, ...],
    replicates: int,
    seed: int,
) -> pl.DataFrame:
    domains = _roster(domain_order, "domain")
    controls = _roster(control_methods, "control method")
    required = {
        "outer_domain",
        "specimen_id",
        "method",
        "cai_auebc",
        "reconstruction_auebc",
    }
    if (
        not isinstance(specimen_auebc, pl.DataFrame)
        or specimen_auebc.height == 0
        or not required <= set(specimen_auebc.columns)
        or type(reference_method) is not str
        or not reference_method
        or reference_method in controls
        or type(replicates) is not int
        or replicates < 1
        or type(seed) is not int
        or set(specimen_auebc.get_column("outer_domain").unique()) != set(domains)
        or specimen_auebc.unique(
            subset=["outer_domain", "specimen_id", "method"]
        ).height
        != specimen_auebc.height
    ):
        raise MAVISClosedLoopMetricError("closed-loop bootstrap request is invalid")
    methods = {reference_method, *controls}
    selected = specimen_auebc.filter(pl.col("method").is_in(methods))
    if set(selected.get_column("method").unique()) != methods:
        raise MAVISClosedLoopMetricError("closed-loop bootstrap method roster is invalid")

    paired: dict[tuple[str, str], np.ndarray] = {}
    for domain in domains:
        reference = selected.filter(
            (pl.col("outer_domain") == domain)
            & (pl.col("method") == reference_method)
        ).select(
            "specimen_id",
            pl.col("cai_auebc").alias("reference_cai"),
            pl.col("reconstruction_auebc").alias("reference_reconstruction"),
        )
        for control in controls:
            control_table = selected.filter(
                (pl.col("outer_domain") == domain) & (pl.col("method") == control)
            ).select(
                "specimen_id",
                pl.col("cai_auebc").alias("control_cai"),
                pl.col("reconstruction_auebc").alias("control_reconstruction"),
            )
            joined = reference.join(
                control_table,
                on="specimen_id",
                how="inner",
                validate="1:1",
            )
            if (
                joined.height == 0
                or joined.height != reference.height
                or joined.height != control_table.height
            ):
                raise MAVISClosedLoopMetricError(
                    "closed-loop bootstrap specimen pairing is incomplete"
                )
            differences = joined.select(
                (pl.col("control_cai") - pl.col("reference_cai")).alias("cai"),
                (
                    pl.col("control_reconstruction")
                    - pl.col("reference_reconstruction")
                ).alias("reconstruction"),
            ).to_numpy()
            if not np.all(np.isfinite(differences)):
                raise MAVISClosedLoopMetricError(
                    "closed-loop bootstrap values are invalid"
                )
            paired[(str(domain), str(control))] = differences

    generator = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for control in controls:
        for replicate in range(replicates):
            domain_differences: list[np.ndarray] = []
            for domain in domains:
                differences = paired[(str(domain), str(control))]
                indices = generator.integers(0, differences.shape[0], differences.shape[0])
                domain_differences.append(
                    np.mean(differences[indices], axis=0, dtype=np.float64)
                )
            domain_values = np.stack(domain_differences)
            rows.append(
                {
                    "control_method": control,
                    "reference_method": reference_method,
                    "replicate": replicate,
                    "control_minus_reference_cai_auebc": float(
                        np.mean(domain_values[:, 0], dtype=np.float64)
                    ),
                    "control_minus_reference_reconstruction_auebc": float(
                        np.mean(domain_values[:, 1], dtype=np.float64)
                    ),
                    "improved_domain_count": int(
                        np.count_nonzero(domain_values[:, 0] > 0.0)
                    ),
                    "domain_count": len(domains),
                    "statistical_unit": "paired_physical_specimen_within_domain",
                }
            )
    return pl.DataFrame(rows).sort(["control_method", "replicate"])


__all__ = [
    "ClosedLoopMetricTables",
    "MAVISClosedLoopMetricError",
    "bootstrap_closed_loop_contrasts",
    "evaluate_closed_loop_predictions",
]
