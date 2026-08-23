"""Formal aggregation and preregistered gate for MVA A5."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import polars as pl

from .a4_evaluation import (
    _domain_curves,
    _equal_curve,
    _equal_metric,
    _random_domain_curves,
)
from .budget_metrics import auebc, simulated_saving, sufficiency_budget
from .statistics import (
    BootstrapEffect,
    paired_domain_bootstrap,
    synchronized_bootstrap_indices,
)

DEPLOYABLE_METHODS = (
    "center_first",
    "observed_gradient",
    "observed_uncertainty",
    "imitation_policy",
)
REFERENCE_METHODS = (
    "uniform",
    "random_median",
    "global_mechanical_mask",
    "mechanical_oracle",
)
AGGREGATED_METHODS = (*DEPLOYABLE_METHODS, *REFERENCE_METHODS)


@dataclass(frozen=True, slots=True)
class A5GateInputs:
    global_effect: BootstrapEffect
    uniform_effect: BootstrapEffect
    policy_oracle_effect: BootstrapEffect
    global_auebc: float
    policy_auebc: float
    oracle_auebc: float


@dataclass(frozen=True, slots=True)
class A5GateResult:
    a5_status: str
    a6_status: str
    global_pass: bool
    uniform_pass: bool
    gap_closure_pass: bool
    gap_closure: float
    global_effect: BootstrapEffect
    uniform_effect: BootstrapEffect
    policy_oracle_effect: BootstrapEffect


@dataclass(frozen=True, slots=True)
class A5Aggregation:
    domain_order: tuple[str, ...]
    checkpoints: tuple[float, ...]
    random_seeds: tuple[int, ...]
    curves: tuple[dict[str, object], ...]
    domain_metrics: tuple[dict[str, object], ...]
    budget_metrics: tuple[dict[str, object], ...]
    specimen_metrics: tuple[dict[str, object], ...]
    bootstrap_effects: tuple[BootstrapEffect, ...]
    gate: A5GateResult
    state_sha256: str


def _effect(value: BootstrapEffect, effect_id: str) -> BootstrapEffect:
    if (
        type(value) is not BootstrapEffect
        or value.effect_id != effect_id
        or len(value.domain_effects) != 6
        or not all(
            math.isfinite(float(item))
            for item in (
                value.point_estimate,
                value.lower,
                value.upper,
                *value.domain_effects,
            )
        )
        or value.lower > value.upper
        or value.improved_domains
        != sum(item > 0.0 for item in value.domain_effects)
        or not math.isclose(
            value.point_estimate,
            float(np.mean(value.domain_effects, dtype=np.float64)),
            abs_tol=1.0e-15,
        )
        or len(value.indices_sha256) != 64
    ):
        raise ValueError("A5 bootstrap effect is invalid")
    return value


def evaluate_a5_gate(inputs: A5GateInputs) -> A5GateResult:
    """Apply the frozen A5 policy and A6-authorization decision."""

    if type(inputs) is not A5GateInputs:
        raise ValueError("issued A5 gate inputs are required")
    global_effect = _effect(inputs.global_effect, "global_minus_policy_auebc")
    uniform_effect = _effect(inputs.uniform_effect, "uniform_minus_policy_auebc")
    policy_oracle = _effect(
        inputs.policy_oracle_effect, "policy_minus_oracle_auebc"
    )
    if len(
        {
            global_effect.indices_sha256,
            uniform_effect.indices_sha256,
            policy_oracle.indices_sha256,
        }
    ) != 1:
        raise ValueError("A5 effects must share one bootstrap matrix")
    global_area = float(inputs.global_auebc)
    policy_area = float(inputs.policy_auebc)
    oracle_area = float(inputs.oracle_auebc)
    if (
        not all(math.isfinite(value) and value > 0.0 for value in (global_area, policy_area, oracle_area))
        or global_area <= oracle_area
        or not math.isclose(
            global_effect.point_estimate,
            global_area - policy_area,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            policy_oracle.point_estimate,
            policy_area - oracle_area,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError("A5 gate areas are inconsistent")
    global_pass = (
        global_effect.point_estimate > 0.0
        and global_effect.lower > 0.0
        and global_effect.improved_domains >= 4
    )
    uniform_pass = (
        uniform_effect.point_estimate > 0.0
        and uniform_effect.lower > 0.0
        and uniform_effect.improved_domains >= 4
    )
    closure = float((global_area - policy_area) / (global_area - oracle_area))
    closure_pass = closure >= 0.20
    go = global_pass and uniform_pass and closure_pass
    return A5GateResult(
        a5_status="MVA_A5_POLICY_GO" if go else "MVA_A5_POLICY_NO_GO",
        a6_status="MVA_A6_AUTHORIZED" if go else "MVA_A6_NOT_AUTHORIZED",
        global_pass=global_pass,
        uniform_pass=uniform_pass,
        gap_closure_pass=closure_pass,
        gap_closure=closure,
        global_effect=global_effect,
        uniform_effect=uniform_effect,
        policy_oracle_effect=policy_oracle,
    )


def _validate_tables(
    a5_states: pl.DataFrame,
    a4_states: pl.DataFrame,
    a2_states: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
    random_seeds: tuple[int, ...],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, tuple[str, ...]]:
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
    if (
        not all(isinstance(value, pl.DataFrame) for value in (a5_states, a4_states, a2_states))
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
        or tuple(sorted(set(checkpoints))) != checkpoints
        or not checkpoints
        or tuple(sorted(set(random_seeds))) != random_seeds
        or not random_seeds
        or any(not required <= set(value.columns) for value in (a5_states, a4_states, a2_states))
        or "seed" not in a2_states.columns
    ):
        raise ValueError("A5 aggregation authority changed")
    a5 = a5_states.filter(pl.col("method").is_in(list(DEPLOYABLE_METHODS)))
    a4 = a4_states.filter(pl.col("method") == "global_mechanical_mask")
    a2 = a2_states.filter(
        pl.col("method").is_in(["uniform", "random", "mechanical_oracle"])
        & pl.col("nominal_checkpoint").is_in(list(checkpoints))
    )
    specimens = tuple(sorted(str(value) for value in a5["specimen_id"].unique()))
    if (
        not specimens
        or specimens != tuple(sorted(str(value) for value in a4["specimen_id"].unique()))
        or specimens != tuple(sorted(str(value) for value in a2["specimen_id"].unique()))
        or set(a5["dataset_id"]) != set(domain_order)
        or set(a5["method"]) != set(DEPLOYABLE_METHODS)
        or set(a5["nominal_checkpoint"]) != set(checkpoints)
        or set(a4["nominal_checkpoint"]) != set(checkpoints)
        or set(a2["method"]) != {"uniform", "random", "mechanical_oracle"}
        or a5.height != len(specimens) * len(DEPLOYABLE_METHODS) * len(checkpoints)
        or a4.height != len(specimens) * len(checkpoints)
        or a2.height != len(specimens) * (2 + len(random_seeds)) * len(checkpoints)
        or a5.unique(subset=["specimen_id", "method", "nominal_checkpoint"]).height != a5.height
        or a4.unique(subset=["specimen_id", "method", "nominal_checkpoint"]).height != a4.height
        or a2.unique(subset=["specimen_id", "method", "seed", "nominal_checkpoint"]).height != a2.height
    ):
        raise ValueError("A5 state roster changed")
    observed_seeds = tuple(
        int(value)
        for value in a2.filter(pl.col("method") == "random")["seed"].unique().sort()
    )
    if observed_seeds != random_seeds:
        raise ValueError("A5 random seed roster changed")
    for table in (a5, a4, a2):
        for column in (
            "effective_budget",
            "p_a_absolute_error",
            "p_b_absolute_error",
        ):
            if not bool(table.select(pl.col(column).is_finite().all()).item()):
                raise ValueError("A5 state metric is nonfinite")
    for domain in domain_order:
        for checkpoint in checkpoints:
            hashes = [
                set(
                    table.filter(
                        (pl.col("dataset_id") == domain)
                        & (pl.col("nominal_checkpoint") == checkpoint)
                    )["p_b_predictor_state_sha256"]
                )
                for table in (a5, a4, a2)
            ]
            if len(hashes[0]) != 1 or hashes[0] != hashes[1] or hashes[0] != hashes[2]:
                raise ValueError("A5 P-B checkpoint predictor changed")
    return a5, a4, a2, specimens


def _image_metric(
    table: pl.DataFrame, method: str, checkpoint: float, column: str
) -> float:
    return _equal_metric(
        table, method=method, checkpoint=checkpoint, column=column
    )


def _aggregation_state(result: A5Aggregation) -> str:
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
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()


def aggregate_a5_tables(
    a5_states: pl.DataFrame,
    a4_states: pl.DataFrame,
    a2_states: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
    random_seeds: tuple[int, ...],
    full_mae: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> A5Aggregation:
    """Aggregate A5 policy states with checksum-bound A2/A4 references."""

    full = float(full_mae)
    if not math.isfinite(full) or full <= 0.0:
        raise ValueError("A5 FULL MAE is invalid")
    a5, a4, a2, specimens = _validate_tables(
        a5_states,
        a4_states,
        a2_states,
        domain_order=domain_order,
        checkpoints=checkpoints,
        random_seeds=random_seeds,
    )
    deterministic_tables = {
        **{method: a5 for method in DEPLOYABLE_METHODS},
        "uniform": a2,
        "global_mechanical_mask": a4,
        "mechanical_oracle": a2,
    }
    domain_by_protocol: dict[str, dict[str, dict[str, dict[float, float]]]] = {
        "P-B": {},
        "P-A": {},
    }
    equal_by_protocol: dict[str, dict[str, np.ndarray]] = {"P-B": {}, "P-A": {}}
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
            domain_by_protocol[protocol][method] = domain_curve
            equal_by_protocol[protocol][method] = equal
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
                        "effective_mean": _image_metric(
                            table, method, checkpoint, "effective_budget"
                        ),
                        "effective_min": float(selected["effective_budget"].min()),
                        "effective_max": float(selected["effective_budget"].max()),
                        "normalized_rgb_mse": _image_metric(
                            table, method, checkpoint, "normalized_rgb_mse"
                        ),
                        "ssim": _image_metric(table, method, checkpoint, "ssim"),
                    }
                )
        seed_domains = _random_domain_curves(a2, error_column=error_column)
        if set(seed_domains) != set(random_seeds):
            raise ValueError("A5 random curve roster changed")
        seed_equal = {
            seed: _equal_curve(
                seed_domains[seed],
                domain_order=domain_order,
                checkpoints=checkpoints,
            )
            for seed in random_seeds
        }
        stacked = np.vstack([seed_equal[seed] for seed in random_seeds])
        random_median = np.median(stacked, axis=0)
        equal_by_protocol[protocol]["random_median"] = random_median
        domain_by_protocol[protocol]["random_median"] = {
            domain: {
                checkpoint: float(
                    np.median(
                        [seed_domains[seed][domain][checkpoint] for seed in random_seeds]
                    )
                )
                for checkpoint in checkpoints
            }
            for domain in domain_order
        }
        for index, checkpoint in enumerate(checkpoints):
            selected = a2.filter(
                (pl.col("method") == "random")
                & (pl.col("nominal_checkpoint") == checkpoint)
            )
            curves.append(
                {
                    "method": "random_median",
                    "protocol": protocol,
                    "nominal_checkpoint": checkpoint,
                    "equal_domain_mae": float(random_median[index]),
                    "mae_mean": float(np.mean(stacked[:, index], dtype=np.float64)),
                    "mae_median": float(np.median(stacked[:, index])),
                    "mae_q05": float(np.quantile(stacked[:, index], 0.05)),
                    "mae_q95": float(np.quantile(stacked[:, index], 0.95)),
                    "effective_mean": float(
                        np.mean(selected["effective_budget"].to_numpy(), dtype=np.float64)
                    ),
                    "effective_min": float(selected["effective_budget"].min()),
                    "effective_max": float(selected["effective_budget"].max()),
                    "normalized_rgb_mse": None,
                    "ssim": None,
                }
            )

    primary_domains = domain_by_protocol["P-B"]
    primary_equal = equal_by_protocol["P-B"]
    budgets = np.asarray(checkpoints, dtype=np.float64)
    domain_areas = {
        method: {
            domain: auebc(
                budgets,
                [primary_domains[method][domain][checkpoint] for checkpoint in checkpoints],
            )
            for domain in domain_order
        }
        for method in AGGREGATED_METHODS
    }
    areas = {
        method: auebc(budgets, primary_equal[method])
        for method in AGGREGATED_METHODS
    }
    indices = synchronized_bootstrap_indices(
        seed=bootstrap_seed, resamples=bootstrap_resamples, domains=6
    )
    effects = (
        paired_domain_bootstrap(
            [domain_areas["global_mechanical_mask"][domain] for domain in domain_order],
            [domain_areas["imitation_policy"][domain] for domain in domain_order],
            indices=indices,
            effect_id="global_minus_policy_auebc",
        ),
        paired_domain_bootstrap(
            [domain_areas["uniform"][domain] for domain in domain_order],
            [domain_areas["imitation_policy"][domain] for domain in domain_order],
            indices=indices,
            effect_id="uniform_minus_policy_auebc",
        ),
        paired_domain_bootstrap(
            [domain_areas["imitation_policy"][domain] for domain in domain_order],
            [domain_areas["mechanical_oracle"][domain] for domain in domain_order],
            indices=indices,
            effect_id="policy_minus_oracle_auebc",
        ),
    )
    gate = evaluate_a5_gate(
        A5GateInputs(
            global_effect=effects[0],
            uniform_effect=effects[1],
            policy_oracle_effect=effects[2],
            global_auebc=areas["global_mechanical_mask"],
            policy_auebc=areas["imitation_policy"],
            oracle_auebc=areas["mechanical_oracle"],
        )
    )
    domain_metrics: list[dict[str, object]] = []
    for domain in domain_order:
        for method in AGGREGATED_METHODS:
            values = np.asarray(
                [primary_domains[method][domain][checkpoint] for checkpoint in checkpoints]
            )
            domain_metrics.append(
                {
                    "dataset_id": domain,
                    "method": method,
                    "auebc": domain_areas[method][domain],
                    "mae_0p125": float(values[checkpoints.index(0.125)]),
                    "b_2p5": sufficiency_budget(
                        budgets, values, full_mae=full, tolerance=0.025
                    ),
                    "b_5": sufficiency_budget(
                        budgets, values, full_mae=full, tolerance=0.05
                    ),
                    "b_7p5": sufficiency_budget(
                        budgets, values, full_mae=full, tolerance=0.075
                    ),
                }
            )
    uniform_b5 = sufficiency_budget(
        budgets, primary_equal["uniform"], full_mae=full, tolerance=0.05
    )
    budget_metrics: list[dict[str, object]] = []
    for method in AGGREGATED_METHODS:
        values = primary_equal[method]
        b5 = sufficiency_budget(budgets, values, full_mae=full, tolerance=0.05)
        budget_metrics.append(
            {
                "method": method,
                "auebc": areas[method],
                "b_2p5": sufficiency_budget(
                    budgets, values, full_mae=full, tolerance=0.025
                ),
                "b_5": b5,
                "b_7p5": sufficiency_budget(
                    budgets, values, full_mae=full, tolerance=0.075
                ),
                "saving_vs_uniform_b5": simulated_saving(b5, uniform_b5),
            }
        )

    specimen_metrics: list[dict[str, object]] = []
    deterministic = {
        (method, str(row["specimen_id"])): None
        for method, table in deterministic_tables.items()
        for row in table.filter(pl.col("method") == method).select("specimen_id").unique().iter_rows(named=True)
    }
    for method, specimen_id in deterministic:
        table = deterministic_tables[method].filter(
            (pl.col("method") == method) & (pl.col("specimen_id") == specimen_id)
        ).sort("nominal_checkpoint")
        values = table["p_b_absolute_error"].to_numpy()
        specimen_metrics.append(
            {
                "specimen_id": specimen_id,
                "dataset_id": str(table["dataset_id"][0]),
                "method": method,
                "auebc": auebc(budgets, values),
                "mae_0p125": float(values[checkpoints.index(0.125)]),
            }
        )
    for specimen_id in specimens:
        selected = a2.filter(
            (pl.col("method") == "random")
            & (pl.col("specimen_id") == specimen_id)
        )
        seed_areas = []
        seed_low = []
        for seed in random_seeds:
            values = selected.filter(pl.col("seed") == seed).sort(
                "nominal_checkpoint"
            )["p_b_absolute_error"].to_numpy()
            seed_areas.append(auebc(budgets, values))
            seed_low.append(float(values[checkpoints.index(0.125)]))
        specimen_metrics.append(
            {
                "specimen_id": specimen_id,
                "dataset_id": str(selected["dataset_id"][0]),
                "method": "random_median",
                "auebc": float(np.median(seed_areas)),
                "mae_0p125": float(np.median(seed_low)),
            }
        )
    specimen_metrics.sort(key=lambda row: (str(row["specimen_id"]), str(row["method"])))
    result = A5Aggregation(
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
    "DEPLOYABLE_METHODS",
    "REFERENCE_METHODS",
    "A5Aggregation",
    "A5GateInputs",
    "A5GateResult",
    "aggregate_a5_tables",
    "evaluate_a5_gate",
]
