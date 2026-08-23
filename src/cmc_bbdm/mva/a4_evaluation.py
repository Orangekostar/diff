"""Aggregation and independent A4/A5 decisions for global MVA masks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import polars as pl

from .budget_metrics import auebc, simulated_saving, sufficiency_budget
from .statistics import (
    BootstrapEffect,
    paired_domain_bootstrap,
    synchronized_bootstrap_indices,
)

GLOBAL_METHODS = (
    "global_appearance_mask",
    "global_reconstruction_mask",
    "global_mechanical_mask",
)
AGGREGATED_METHODS = (
    "uniform",
    *GLOBAL_METHODS,
    "mechanical_oracle",
    "random_median",
)


@dataclass(frozen=True, slots=True)
class A4GateInputs:
    uniform_effect: BootstrapEffect
    reconstruction_effect: BootstrapEffect
    appearance_effect: BootstrapEffect
    adaptive_gap_effect: BootstrapEffect
    global_mechanical_auebc: float
    mechanical_oracle_auebc: float


@dataclass(frozen=True, slots=True)
class A4GateResult:
    global_mask_status: str
    a5_status: str
    uniform_pass: bool
    reconstruction_pass: bool
    appearance_pass: bool
    adaptive_gap_pass: bool
    relative_adaptive_gap: float
    uniform_effect: BootstrapEffect
    reconstruction_effect: BootstrapEffect
    appearance_effect: BootstrapEffect
    adaptive_gap_effect: BootstrapEffect


@dataclass(frozen=True, slots=True)
class A4Aggregation:
    domain_order: tuple[str, ...]
    checkpoints: tuple[float, ...]
    random_seeds: tuple[int, ...]
    curves: tuple[dict[str, object], ...]
    domain_metrics: tuple[dict[str, object], ...]
    budget_metrics: tuple[dict[str, object], ...]
    specimen_metrics: tuple[dict[str, object], ...]
    bootstrap_effects: tuple[BootstrapEffect, ...]
    gate: A4GateResult
    state_sha256: str


def _validate_effect(value: BootstrapEffect, label: str) -> BootstrapEffect:
    if type(value) is not BootstrapEffect:
        raise ValueError(f"{label} must be a bootstrap effect")
    numeric = (
        value.point_estimate,
        value.lower,
        value.upper,
        *value.domain_effects,
    )
    if (
        type(value.effect_id) is not str
        or not value.effect_id
        or len(value.domain_effects) != 6
        or not all(math.isfinite(float(item)) for item in numeric)
        or value.lower > value.upper
        or value.improved_domains
        != sum(item > 0.0 for item in value.domain_effects)
        or not math.isclose(
            value.point_estimate,
            float(np.mean(value.domain_effects, dtype=np.float64)),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or len(value.indices_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value.indices_sha256
        )
    ):
        raise ValueError(f"{label} is inconsistent")
    return value


def evaluate_a4_gate(inputs: A4GateInputs) -> A4GateResult:
    """Apply the preregistered global-mask and A5-authorization gates."""

    if type(inputs) is not A4GateInputs:
        raise ValueError("issued A4 gate inputs are required")
    uniform = _validate_effect(inputs.uniform_effect, "uniform effect")
    reconstruction = _validate_effect(
        inputs.reconstruction_effect, "reconstruction effect"
    )
    appearance = _validate_effect(inputs.appearance_effect, "appearance effect")
    adaptive = _validate_effect(inputs.adaptive_gap_effect, "adaptive gap effect")
    if len(
        {
            uniform.indices_sha256,
            reconstruction.indices_sha256,
            appearance.indices_sha256,
            adaptive.indices_sha256,
        }
    ) != 1:
        raise ValueError("A4 effects must share one bootstrap index matrix")
    global_area = float(inputs.global_mechanical_auebc)
    oracle_area = float(inputs.mechanical_oracle_auebc)
    if (
        not math.isfinite(global_area)
        or not math.isfinite(oracle_area)
        or global_area <= 0.0
        or oracle_area < 0.0
        or not math.isclose(
            adaptive.point_estimate,
            global_area - oracle_area,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError("adaptive gap areas are inconsistent")

    uniform_pass = (
        uniform.point_estimate > 0.0
        and uniform.lower > 0.0
        and uniform.improved_domains >= 4
    )
    reconstruction_pass = (
        reconstruction.point_estimate > 0.0 and reconstruction.lower > 0.0
    )
    appearance_pass = appearance.point_estimate > 0.0 and appearance.lower > 0.0
    global_status = (
        "MVA_A4_GLOBAL_GO"
        if all((uniform_pass, reconstruction_pass, appearance_pass))
        else "MVA_A4_GLOBAL_NO_GO"
    )

    relative_gap = float(adaptive.point_estimate / global_area)
    adaptive_pass = (
        adaptive.point_estimate > 0.0
        and relative_gap >= 0.03
        and adaptive.lower > 0.0
        and adaptive.improved_domains >= 4
    )
    a5_status = "MVA_A5_AUTHORIZED" if adaptive_pass else "MVA_A5_NOT_AUTHORIZED"
    return A4GateResult(
        global_mask_status=global_status,
        a5_status=a5_status,
        uniform_pass=uniform_pass,
        reconstruction_pass=reconstruction_pass,
        appearance_pass=appearance_pass,
        adaptive_gap_pass=adaptive_pass,
        relative_adaptive_gap=relative_gap,
        uniform_effect=uniform,
        reconstruction_effect=reconstruction,
        appearance_effect=appearance,
        adaptive_gap_effect=adaptive,
    )


def _finite_column(table: pl.DataFrame, name: str) -> bool:
    return bool(table.select(pl.col(name).is_finite().all()).item())


def _validate_state_tables(
    a4_states: pl.DataFrame,
    reference_states: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
    random_seeds: tuple[int, ...],
) -> tuple[pl.DataFrame, pl.DataFrame, tuple[str, ...]]:
    if (
        not isinstance(a4_states, pl.DataFrame)
        or not isinstance(reference_states, pl.DataFrame)
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
        or not checkpoints
        or tuple(sorted(set(checkpoints))) != checkpoints
        or 0.0625 not in checkpoints
        or 0.25 not in checkpoints
        or not random_seeds
        or tuple(sorted(set(random_seeds))) != random_seeds
    ):
        raise ValueError("A4 aggregation authority changed")
    required = {
        "dataset_id",
        "effective_budget",
        "method",
        "nominal_checkpoint",
        "normalized_rgb_mse",
        "p_a_absolute_error",
        "p_b_absolute_error",
        "p_b_predictor_state_sha256",
        "specimen_id",
        "ssim",
    }
    if not required <= set(a4_states.columns) or not required | {"seed"} <= set(
        reference_states.columns
    ):
        raise ValueError("A4 aggregation state schema changed")
    a4 = a4_states.filter(pl.col("method").is_in(list(GLOBAL_METHODS)))
    reference = reference_states.filter(
        pl.col("method").is_in(["uniform", "random", "mechanical_oracle"])
        & pl.col("nominal_checkpoint").is_in(list(checkpoints))
    )
    specimens = tuple(sorted(str(value) for value in a4["specimen_id"].unique()))
    reference_specimens = tuple(
        sorted(str(value) for value in reference["specimen_id"].unique())
    )
    specimen_count = len(specimens)
    if (
        specimen_count < 1
        or specimens != reference_specimens
        or set(a4["dataset_id"].unique()) != set(domain_order)
        or set(reference["dataset_id"].unique()) != set(domain_order)
        or set(a4["nominal_checkpoint"].unique()) != set(checkpoints)
        or set(reference["nominal_checkpoint"].unique()) != set(checkpoints)
        or set(a4["method"].unique()) != set(GLOBAL_METHODS)
        or set(reference["method"].unique())
        != {"uniform", "random", "mechanical_oracle"}
        or a4.height != specimen_count * len(GLOBAL_METHODS) * len(checkpoints)
        or reference.height
        != specimen_count * (2 + len(random_seeds)) * len(checkpoints)
        or a4.unique(
            subset=["specimen_id", "method", "nominal_checkpoint"]
        ).height
        != a4.height
        or reference.unique(
            subset=["specimen_id", "method", "seed", "nominal_checkpoint"]
        ).height
        != reference.height
    ):
        raise ValueError("A4 aggregation state roster changed")
    observed_random_seeds = tuple(
        int(value)
        for value in reference.filter(pl.col("method") == "random")["seed"]
        .unique()
        .sort()
    )
    if observed_random_seeds != random_seeds:
        raise ValueError("A4 random seed roster changed")
    for name in (
        "effective_budget",
        "nominal_checkpoint",
        "p_a_absolute_error",
        "p_b_absolute_error",
    ):
        if not _finite_column(a4, name) or not _finite_column(reference, name):
            raise ValueError("A4 aggregation contains nonfinite values")
    for name in ("normalized_rgb_mse", "ssim"):
        if not _finite_column(a4, name):
            raise ValueError("A4 image metrics contain nonfinite values")
    if (
        a4.filter(
            (pl.col("effective_budget") <= 0.0)
            | (pl.col("p_a_absolute_error") < 0.0)
            | (pl.col("p_b_absolute_error") < 0.0)
            | (pl.col("normalized_rgb_mse") < 0.0)
        ).height
        or reference.filter(
            (pl.col("effective_budget") <= 0.0)
            | (pl.col("p_a_absolute_error") < 0.0)
            | (pl.col("p_b_absolute_error") < 0.0)
        ).height
    ):
        raise ValueError("A4 aggregation metric range changed")

    specimen_domains = a4.group_by("specimen_id").agg(
        pl.col("dataset_id").n_unique().alias("domains")
    )
    if specimen_domains.filter(pl.col("domains") != 1).height:
        raise ValueError("A4 specimen domain roster changed")
    for domain in domain_order:
        for checkpoint in checkpoints:
            a4_hashes = set(
                a4.filter(
                    (pl.col("dataset_id") == domain)
                    & (pl.col("nominal_checkpoint") == checkpoint)
                )["p_b_predictor_state_sha256"]
            )
            reference_hashes = set(
                reference.filter(
                    (pl.col("dataset_id") == domain)
                    & (pl.col("nominal_checkpoint") == checkpoint)
                )["p_b_predictor_state_sha256"]
            )
            if (
                len(a4_hashes) != 1
                or a4_hashes != reference_hashes
                or any(
                    type(value) is not str
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef" for character in value
                    )
                    for value in a4_hashes
                )
            ):
                raise ValueError("A4 P-B checkpoint head changed")
    return a4, reference, specimens


def _domain_curves(
    table: pl.DataFrame,
    *,
    method: str,
    error_column: str,
    seed: int | None = None,
) -> dict[str, dict[float, float]]:
    selected = table.filter(pl.col("method") == method)
    if seed is not None:
        selected = selected.filter(pl.col("seed") == seed)
    values: dict[tuple[str, float], list[float]] = {}
    for row in selected.sort(
        ["dataset_id", "nominal_checkpoint", "specimen_id"]
    ).select("dataset_id", "nominal_checkpoint", error_column).iter_rows(named=True):
        values.setdefault(
            (str(row["dataset_id"]), float(row["nominal_checkpoint"])), []
        ).append(float(row[error_column]))
    output: dict[str, dict[float, float]] = {}
    for (domain, checkpoint), items in values.items():
        output.setdefault(domain, {})[checkpoint] = float(
            np.mean(np.asarray(items, dtype=np.float64), dtype=np.float64)
        )
    return output


def _random_domain_curves(
    table: pl.DataFrame, *, error_column: str
) -> dict[int, dict[str, dict[float, float]]]:
    values: dict[tuple[int, str, float], list[float]] = {}
    selected = table.filter(pl.col("method") == "random").sort(
        ["seed", "dataset_id", "nominal_checkpoint", "specimen_id"]
    )
    for row in selected.select(
        "seed", "dataset_id", "nominal_checkpoint", error_column
    ).iter_rows(named=True):
        values.setdefault(
            (
                int(row["seed"]),
                str(row["dataset_id"]),
                float(row["nominal_checkpoint"]),
            ),
            [],
        ).append(float(row[error_column]))
    output: dict[int, dict[str, dict[float, float]]] = {}
    for (seed, domain, checkpoint), items in values.items():
        output.setdefault(seed, {}).setdefault(domain, {})[checkpoint] = float(
            np.mean(np.asarray(items, dtype=np.float64), dtype=np.float64)
        )
    return output


def _equal_curve(
    domain_curve: dict[str, dict[float, float]],
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
) -> np.ndarray:
    try:
        return np.asarray(
            [
                np.mean(
                    [domain_curve[domain][checkpoint] for domain in domain_order],
                    dtype=np.float64,
                )
                for checkpoint in checkpoints
            ],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ValueError("A4 domain curve roster changed") from error


def _equal_metric(
    table: pl.DataFrame,
    *,
    method: str,
    checkpoint: float,
    column: str,
) -> float:
    selected = table.filter(
        (pl.col("method") == method)
        & (pl.col("nominal_checkpoint") == checkpoint)
    ).sort(["dataset_id", "specimen_id"])
    values: dict[str, list[float]] = {}
    for row in selected.select("dataset_id", column).iter_rows(named=True):
        values.setdefault(str(row["dataset_id"]), []).append(float(row[column]))
    domain_means = np.asarray(
        [
            np.mean(np.asarray(values[domain], dtype=np.float64), dtype=np.float64)
            for domain in sorted(values)
        ],
        dtype=np.float64,
    )
    return float(np.mean(domain_means, dtype=np.float64))


def _aggregation_state(result: A4Aggregation) -> str:
    payload = {
        "bootstrap_effects": [asdict(value) for value in result.bootstrap_effects],
        "budget_metrics": result.budget_metrics,
        "checkpoints": result.checkpoints,
        "curves": result.curves,
        "domain_metrics": result.domain_metrics,
        "domain_order": result.domain_order,
        "gate": asdict(result.gate),
        "random_seeds": result.random_seeds,
        "specimen_metrics": result.specimen_metrics,
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()


def aggregate_a4_tables(
    a4_states: pl.DataFrame,
    reference_states: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
    random_seeds: tuple[int, ...],
    full_mae: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> A4Aggregation:
    """Aggregate validated A4 states with checksum-bound A2 references."""

    full = float(full_mae)
    if not math.isfinite(full) or full <= 0.0:
        raise ValueError("A4 FULL MAE is invalid")
    a4, reference, specimens = _validate_state_tables(
        a4_states,
        reference_states,
        domain_order=domain_order,
        checkpoints=checkpoints,
        random_seeds=random_seeds,
    )
    deterministic_tables = {
        "uniform": reference,
        **{method: a4 for method in GLOBAL_METHODS},
        "mechanical_oracle": reference,
    }
    domain_curves_by_protocol: dict[
        str, dict[str, dict[str, dict[float, float]]]
    ] = {"P-B": {}, "P-A": {}}
    equal_curves_by_protocol: dict[str, dict[str, np.ndarray]] = {
        "P-B": {},
        "P-A": {},
    }
    curves: list[dict[str, object]] = []
    for protocol, error_column in (
        ("P-B", "p_b_absolute_error"),
        ("P-A", "p_a_absolute_error"),
    ):
        for method, table in deterministic_tables.items():
            domain_curve = _domain_curves(
                table, method=method, error_column=error_column
            )
            equal = _equal_curve(
                domain_curve, domain_order=domain_order, checkpoints=checkpoints
            )
            domain_curves_by_protocol[protocol][method] = domain_curve
            equal_curves_by_protocol[protocol][method] = equal
            for index, checkpoint in enumerate(checkpoints):
                selected = table.filter(
                    (pl.col("method") == method)
                    & (pl.col("nominal_checkpoint") == checkpoint)
                )
                curves.append(
                    {
                        "method": method,
                        "protocol": protocol,
                        "nominal_checkpoint": checkpoint,
                        "equal_domain_mae": float(equal[index]),
                        "mae_mean": None,
                        "mae_median": None,
                        "mae_q05": None,
                        "mae_q95": None,
                        "effective_mean": _equal_metric(
                            table,
                            method=method,
                            checkpoint=checkpoint,
                            column="effective_budget",
                        ),
                        "effective_min": float(selected["effective_budget"].min()),
                        "effective_max": float(selected["effective_budget"].max()),
                        "normalized_rgb_mse": (
                            _equal_metric(
                                table,
                                method=method,
                                checkpoint=checkpoint,
                                column="normalized_rgb_mse",
                            )
                            if method in GLOBAL_METHODS
                            else None
                        ),
                        "ssim": (
                            _equal_metric(
                                table,
                                method=method,
                                checkpoint=checkpoint,
                                column="ssim",
                            )
                            if method in GLOBAL_METHODS
                            else None
                        ),
                    }
                )
        seed_domain_curves = _random_domain_curves(
            reference, error_column=error_column
        )
        if set(seed_domain_curves) != set(random_seeds):
            raise ValueError("A4 random curve roster changed")
        seed_equal_curves = {
            seed: _equal_curve(
                seed_domain_curves[seed],
                domain_order=domain_order,
                checkpoints=checkpoints,
            )
            for seed in random_seeds
        }
        random_median = np.median(
            np.vstack([seed_equal_curves[seed] for seed in random_seeds]), axis=0
        )
        equal_curves_by_protocol[protocol]["random_median"] = random_median
        domain_curves_by_protocol[protocol]["random_median"] = {
            domain: {
                checkpoint: float(
                    np.median(
                        [
                            seed_domain_curves[seed][domain][checkpoint]
                            for seed in random_seeds
                        ]
                    )
                )
                for checkpoint in checkpoints
            }
            for domain in domain_order
        }
        stacked = np.vstack([seed_equal_curves[seed] for seed in random_seeds])
        for index, checkpoint in enumerate(checkpoints):
            selected = reference.filter(
                (pl.col("method") == "random")
                & (pl.col("nominal_checkpoint") == checkpoint)
            ).sort(["seed", "dataset_id", "specimen_id"])
            effective_values = selected["effective_budget"].to_numpy()
            curves.append(
                {
                    "method": "random_median",
                    "protocol": protocol,
                    "nominal_checkpoint": checkpoint,
                    "equal_domain_mae": float(random_median[index]),
                    "mae_mean": float(np.mean(stacked[:, index])),
                    "mae_median": float(np.median(stacked[:, index])),
                    "mae_q05": float(np.quantile(stacked[:, index], 0.05)),
                    "mae_q95": float(np.quantile(stacked[:, index], 0.95)),
                    "effective_mean": float(
                        np.mean(effective_values, dtype=np.float64)
                    ),
                    "effective_min": float(selected["effective_budget"].min()),
                    "effective_max": float(selected["effective_budget"].max()),
                    "normalized_rgb_mse": None,
                    "ssim": None,
                }
            )

    primary_domain_curves = domain_curves_by_protocol["P-B"]
    primary_equal_curves = equal_curves_by_protocol["P-B"]
    checkpoint_array = np.asarray(checkpoints, dtype=np.float64)
    domain_auebc = {
        method: {
            domain: auebc(
                checkpoint_array,
                np.asarray(
                    [
                        primary_domain_curves[method][domain][checkpoint]
                        for checkpoint in checkpoints
                    ]
                ),
            )
            for domain in domain_order
        }
        for method in AGGREGATED_METHODS
    }
    areas = {
        method: auebc(checkpoint_array, primary_equal_curves[method])
        for method in AGGREGATED_METHODS
    }
    indices = synchronized_bootstrap_indices(
        seed=bootstrap_seed, resamples=bootstrap_resamples, domains=6
    )
    comparison_specs = (
        ("uniform_minus_global_mechanical_auebc", "uniform"),
        (
            "global_reconstruction_minus_global_mechanical_auebc",
            "global_reconstruction_mask",
        ),
        (
            "global_appearance_minus_global_mechanical_auebc",
            "global_appearance_mask",
        ),
        (
            "global_mechanical_minus_mechanical_oracle_auebc",
            "global_mechanical_mask",
        ),
    )
    effects = tuple(
        paired_domain_bootstrap(
            [domain_auebc[baseline][domain] for domain in domain_order],
            [
                domain_auebc[
                    "mechanical_oracle"
                    if effect_id
                    == "global_mechanical_minus_mechanical_oracle_auebc"
                    else "global_mechanical_mask"
                ][domain]
                for domain in domain_order
            ],
            indices=indices,
            effect_id=effect_id,
        )
        for effect_id, baseline in comparison_specs
    )
    gate = evaluate_a4_gate(
        A4GateInputs(
            uniform_effect=effects[0],
            reconstruction_effect=effects[1],
            appearance_effect=effects[2],
            adaptive_gap_effect=effects[3],
            global_mechanical_auebc=areas["global_mechanical_mask"],
            mechanical_oracle_auebc=areas["mechanical_oracle"],
        )
    )

    domain_metrics: list[dict[str, object]] = []
    for domain in domain_order:
        for method in AGGREGATED_METHODS:
            values = np.asarray(
                [
                    primary_domain_curves[method][domain][checkpoint]
                    for checkpoint in checkpoints
                ]
            )
            domain_metrics.append(
                {
                    "dataset_id": domain,
                    "method": method,
                    "auebc": domain_auebc[method][domain],
                    "mae_0p125": float(values[checkpoints.index(0.125)]),
                    "b_2p5": sufficiency_budget(
                        checkpoint_array, values, full_mae=full, tolerance=0.025
                    ),
                    "b_5": sufficiency_budget(
                        checkpoint_array, values, full_mae=full, tolerance=0.05
                    ),
                    "b_7p5": sufficiency_budget(
                        checkpoint_array, values, full_mae=full, tolerance=0.075
                    ),
                }
            )
    budget_metrics: list[dict[str, object]] = []
    uniform_b5 = sufficiency_budget(
        checkpoint_array,
        primary_equal_curves["uniform"],
        full_mae=full,
        tolerance=0.05,
    )
    for method in AGGREGATED_METHODS:
        values = primary_equal_curves[method]
        b5 = sufficiency_budget(
            checkpoint_array, values, full_mae=full, tolerance=0.05
        )
        budget_metrics.append(
            {
                "method": method,
                "auebc": areas[method],
                "b_2p5": sufficiency_budget(
                    checkpoint_array, values, full_mae=full, tolerance=0.025
                ),
                "b_5": b5,
                "b_7p5": sufficiency_budget(
                    checkpoint_array, values, full_mae=full, tolerance=0.075
                ),
                "saving_vs_uniform_b5": simulated_saving(b5, uniform_b5),
            }
        )

    deterministic_specimen_curves: dict[
        tuple[str, str], dict[float, tuple[str, float]]
    ] = {}
    for method, table in deterministic_tables.items():
        for row in table.filter(pl.col("method") == method).select(
            "specimen_id",
            "dataset_id",
            "nominal_checkpoint",
            "p_b_absolute_error",
        ).iter_rows(named=True):
            deterministic_specimen_curves.setdefault(
                (method, str(row["specimen_id"])), {}
            )[float(row["nominal_checkpoint"])] = (
                str(row["dataset_id"]),
                float(row["p_b_absolute_error"]),
            )
    random_specimen_curves: dict[
        tuple[str, int], dict[float, tuple[str, float]]
    ] = {}
    for row in reference.filter(pl.col("method") == "random").select(
        "specimen_id",
        "dataset_id",
        "seed",
        "nominal_checkpoint",
        "p_b_absolute_error",
    ).iter_rows(named=True):
        random_specimen_curves.setdefault(
            (str(row["specimen_id"]), int(row["seed"])), {}
        )[float(row["nominal_checkpoint"])] = (
            str(row["dataset_id"]),
            float(row["p_b_absolute_error"]),
        )

    specimen_metrics: list[dict[str, object]] = []
    for specimen_id in specimens:
        domain = deterministic_specimen_curves[
            ("global_mechanical_mask", specimen_id)
        ][checkpoints[0]][0]
        for method in AGGREGATED_METHODS:
            if method == "random_median":
                seed_areas = []
                seed_low = []
                for seed in random_seeds:
                    curve = random_specimen_curves[(specimen_id, seed)]
                    values = np.asarray(
                        [curve[checkpoint][1] for checkpoint in checkpoints]
                    )
                    seed_areas.append(auebc(checkpoint_array, values))
                    seed_low.append(float(values[checkpoints.index(0.125)]))
                area = float(np.median(seed_areas))
                low = float(np.median(seed_low))
            else:
                curve = deterministic_specimen_curves[(method, specimen_id)]
                values = np.asarray(
                    [curve[checkpoint][1] for checkpoint in checkpoints]
                )
                area = auebc(checkpoint_array, values)
                low = float(values[checkpoints.index(0.125)])
            specimen_metrics.append(
                {
                    "specimen_id": specimen_id,
                    "dataset_id": domain,
                    "method": method,
                    "auebc": area,
                    "mae_0p125": low,
                }
            )
    result = A4Aggregation(
        domain_order=domain_order,
        checkpoints=checkpoints,
        random_seeds=random_seeds,
        curves=tuple(curves),
        domain_metrics=tuple(domain_metrics),
        budget_metrics=tuple(budget_metrics),
        specimen_metrics=tuple(specimen_metrics),
        bootstrap_effects=effects,
        gate=gate,
        state_sha256="",
    )
    return replace(result, state_sha256=_aggregation_state(result))


__all__ = [
    "AGGREGATED_METHODS",
    "GLOBAL_METHODS",
    "A4Aggregation",
    "A4GateInputs",
    "A4GateResult",
    "aggregate_a4_tables",
    "evaluate_a4_gate",
]
