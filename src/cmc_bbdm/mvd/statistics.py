"""Equal-domain aggregation and frozen MVD M0 feasibility gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import polars as pl

from cmc_bbdm.mva.budget_metrics import auebc, sufficiency_budget
from cmc_bbdm.mva.statistics import (
    BootstrapEffect,
    paired_domain_bootstrap,
    synchronized_bootstrap_indices,
)

METHODS = (
    "uniform",
    "random_median",
    "one_shot_reconstruction",
    "global_mechanical_mask",
    "one_shot_mechanical_oracle",
    "sequential_mechanical_oracle",
    "FULL",
)
_DETERMINISTIC = tuple(method for method in METHODS if method not in {"random_median", "FULL"})


@dataclass(frozen=True, slots=True)
class M0GateResult:
    status: str
    go: bool
    uniform_pass: bool
    reconstruction_pass: bool
    headroom_pass: bool
    headroom_retention: float | None
    stronger_baseline: str
    stronger_baseline_auebc: float
    one_shot_auebc: float
    sequential_auebc: float
    uniform_effect: BootstrapEffect
    reconstruction_effect: BootstrapEffect


@dataclass(frozen=True, slots=True)
class M0Aggregation:
    domain_order: tuple[str, ...]
    checkpoints: tuple[float, ...]
    random_seeds: tuple[int, ...]
    curves: tuple[dict[str, object], ...]
    domain_metrics: tuple[dict[str, object], ...]
    budget_metrics: tuple[dict[str, object], ...]
    bootstrap_effects: tuple[BootstrapEffect, ...]
    gate: M0GateResult
    state_sha256: str


def _finite(table: pl.DataFrame, columns: tuple[str, ...]) -> bool:
    return all(bool(table.select(pl.col(name).is_finite().all()).item()) for name in columns)


def _canonical_a2(table: pl.DataFrame) -> pl.DataFrame:
    return table.with_columns(
        pl.when(pl.col("method") == "mechanical_oracle")
        .then(pl.lit("sequential_mechanical_oracle"))
        .otherwise(pl.col("method"))
        .alias("method")
    )


def _validate_tables(
    one_shot_states: pl.DataFrame,
    a2_states: pl.DataFrame,
    a4_states: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
    random_seeds: tuple[int, ...],
    evaluator_reproduction: pl.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, tuple[str, ...]]:
    required = {
        "dataset_id",
        "effective_budget",
        "method",
        "nominal_checkpoint",
        "p_a_absolute_error",
        "p_b_absolute_error",
        "p_b_predictor_state_sha256",
        "specimen_id",
    }
    if (
        not all(isinstance(value, pl.DataFrame) for value in (one_shot_states, a2_states, a4_states))
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
        or tuple(sorted(set(checkpoints))) != checkpoints
        or not {0.0625, 0.25} <= set(checkpoints)
        or len(random_seeds) != 100
        or tuple(sorted(set(random_seeds))) != random_seeds
        or any(not required <= set(value.columns) for value in (one_shot_states, a2_states, a4_states))
        or "seed" not in a2_states.columns
    ):
        raise ValueError("M0 aggregation authority changed")
    one_shot = one_shot_states.filter(
        pl.col("method").is_in(
            ["one_shot_reconstruction", "one_shot_mechanical_oracle"]
        )
        & pl.col("nominal_checkpoint").is_in(list(checkpoints))
    )
    a2 = _canonical_a2(a2_states).filter(
        pl.col("method").is_in(["uniform", "random", "sequential_mechanical_oracle"])
        & pl.col("nominal_checkpoint").is_in(list(checkpoints))
    )
    a4 = a4_states.filter(
        (pl.col("method") == "global_mechanical_mask")
        & pl.col("nominal_checkpoint").is_in(list(checkpoints))
    )
    specimens = tuple(sorted(str(value) for value in one_shot["specimen_id"].unique()))
    specimen_count = len(specimens)
    expected_random = tuple(
        int(value)
        for value in a2.filter(pl.col("method") == "random")["seed"].unique().sort()
    )
    if (
        specimen_count < 1
        or specimens != tuple(sorted(str(value) for value in a2["specimen_id"].unique()))
        or specimens != tuple(sorted(str(value) for value in a4["specimen_id"].unique()))
        or set(one_shot["dataset_id"]) != set(domain_order)
        or set(a2["dataset_id"]) != set(domain_order)
        or set(a4["dataset_id"]) != set(domain_order)
        or set(one_shot["method"]) != {"one_shot_reconstruction", "one_shot_mechanical_oracle"}
        or set(a2["method"]) != {"uniform", "random", "sequential_mechanical_oracle"}
        or set(a4["method"]) != {"global_mechanical_mask"}
        or one_shot.height != specimen_count * 2 * len(checkpoints)
        or a2.height != specimen_count * (2 + len(random_seeds)) * len(checkpoints)
        or a4.height != specimen_count * len(checkpoints)
        or one_shot.unique(subset=["specimen_id", "method", "nominal_checkpoint"]).height != one_shot.height
        or a2.unique(subset=["specimen_id", "method", "seed", "nominal_checkpoint"]).height != a2.height
        or a4.unique(subset=["specimen_id", "method", "nominal_checkpoint"]).height != a4.height
        or expected_random != random_seeds
    ):
        raise ValueError("M0 state roster changed")
    numeric = (
        "effective_budget",
        "nominal_checkpoint",
        "p_a_absolute_error",
        "p_b_absolute_error",
    )
    if not all(_finite(value, numeric) for value in (one_shot, a2, a4)):
        raise ValueError("M0 state metrics are nonfinite")
    if any(
        value.filter(
            (pl.col("effective_budget") <= 0.0)
            | (pl.col("p_a_absolute_error") < 0.0)
            | (pl.col("p_b_absolute_error") < 0.0)
        ).height
        for value in (one_shot, a2, a4)
    ):
        raise ValueError("M0 state metric range changed")
    for domain in domain_order:
        for checkpoint in checkpoints:
            hashes = [
                set(
                    table.filter(
                        (pl.col("dataset_id") == domain)
                        & (pl.col("nominal_checkpoint") == checkpoint)
                    )["p_b_predictor_state_sha256"]
                )
                for table in (one_shot, a2, a4)
            ]
            if len(hashes[0]) != 1 or hashes[1] != hashes[2] or any(
                    type(value) is not str
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in hashes[0]
                ):
                raise ValueError("M0 P-B checkpoint head changed")
            if hashes[0] != hashes[1]:
                if evaluator_reproduction is None:
                    raise ValueError("M0 P-B checkpoint head changed")
                audit = evaluator_reproduction.filter(
                    (pl.col("outer_domain") == domain)
                    & (pl.col("nominal_checkpoint") == checkpoint)
                )
                if (
                    audit.height != 1
                    or audit["new_predictor_state_sha256"][0] not in hashes[0]
                    or audit["reference_predictor_state_sha256"][0] not in hashes[1]
                    or float(audit["tolerance"][0]) != 1.0e-12
                    or float(audit["maximum_prediction_delta"][0]) > 1.0e-12
                    or float(audit["maximum_prediction_delta"][0]) < 0.0
                    or float(audit["mean_prediction_delta"][0]) < 0.0
                    or int(audit["target_specimen_count"][0])
                    != one_shot.filter(pl.col("dataset_id") == domain)[
                        "specimen_id"
                    ].n_unique()
                    or set(str(audit["fit_domains"][0]).split("|"))
                    != set(domain_order) - {domain}
                ):
                    raise ValueError("M0 P-B numerical reproduction changed")
    return one_shot, a2, a4, specimens


def _domain_curve(
    table: pl.DataFrame, *, method: str, error_column: str, seed: int | None = None
) -> dict[str, dict[float, float]]:
    selected = table.filter(pl.col("method") == method)
    if seed is not None:
        selected = selected.filter(pl.col("seed") == seed)
    output: dict[str, dict[float, float]] = {}
    for dataset_id in sorted(str(value) for value in selected["dataset_id"].unique()):
        domain = selected.filter(pl.col("dataset_id") == dataset_id)
        for checkpoint in sorted(
            float(value) for value in domain["nominal_checkpoint"].unique()
        ):
            values = (
                domain.filter(pl.col("nominal_checkpoint") == checkpoint)
                .sort("specimen_id")[error_column]
                .to_numpy()
            )
            output.setdefault(dataset_id, {})[checkpoint] = float(
                np.mean(values, dtype=np.float64)
            )
    return output


def _equal_curve(
    values: dict[str, dict[float, float]],
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
) -> np.ndarray:
    try:
        return np.asarray(
            [
                np.mean(
                    [values[domain][checkpoint] for domain in domain_order],
                    dtype=np.float64,
                )
                for checkpoint in checkpoints
            ],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ValueError("M0 domain curve roster changed") from error


def _effective_summary(
    table: pl.DataFrame, *, method: str, checkpoint: float
) -> tuple[float, float, float]:
    selected = table.filter(
        (pl.col("method") == method)
        & (pl.col("nominal_checkpoint") == checkpoint)
    ).sort(["dataset_id", "specimen_id"])
    means = np.asarray(
        [
            np.mean(
                selected.filter(pl.col("dataset_id") == dataset_id)
                .sort("specimen_id")["effective_budget"]
                .to_numpy(),
                dtype=np.float64,
            )
            for dataset_id in sorted(
                str(value) for value in selected["dataset_id"].unique()
            )
        ],
        dtype=np.float64,
    )
    return (
        float(np.mean(means, dtype=np.float64)),
        float(selected["effective_budget"].min()),
        float(selected["effective_budget"].max()),
    )


def _state(result: M0Aggregation) -> str:
    payload = {
        "bootstrap_effects": [asdict(value) for value in result.bootstrap_effects],
        "budget_metrics": result.budget_metrics,
        "checkpoints": result.checkpoints,
        "curves": result.curves,
        "domain_metrics": result.domain_metrics,
        "domain_order": result.domain_order,
        "gate": asdict(result.gate),
        "random_seeds": result.random_seeds,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()


def aggregate_m0_tables(
    one_shot_states: pl.DataFrame,
    a2_states: pl.DataFrame,
    a4_states: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
    random_seeds: tuple[int, ...],
    full_mae: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    minimum_improved_domains: int,
    minimum_headroom_retention: float,
    strong_headroom_retention: float,
    evaluator_reproduction: pl.DataFrame | None = None,
) -> M0Aggregation:
    """Aggregate M0 against hash-bound historical controls and issue its gate."""

    full = float(full_mae)
    if (
        not math.isfinite(full)
        or full <= 0.0
        or minimum_improved_domains != 4
        or minimum_headroom_retention != 0.20
        or strong_headroom_retention != 0.50
    ):
        raise ValueError("M0 gate constants changed")
    one_shot, a2, a4, _specimens = _validate_tables(
        one_shot_states,
        a2_states,
        a4_states,
        domain_order=domain_order,
        checkpoints=checkpoints,
        random_seeds=random_seeds,
        evaluator_reproduction=evaluator_reproduction,
    )
    tables = {
        "uniform": a2,
        "one_shot_reconstruction": one_shot,
        "global_mechanical_mask": a4,
        "one_shot_mechanical_oracle": one_shot,
        "sequential_mechanical_oracle": a2,
    }
    by_protocol: dict[str, dict[str, dict[str, dict[float, float]]]] = {
        "P-B": {},
        "P-A": {},
    }
    equal: dict[str, dict[str, np.ndarray]] = {"P-B": {}, "P-A": {}}
    curves: list[dict[str, object]] = []
    for protocol, error_column in (("P-B", "p_b_absolute_error"), ("P-A", "p_a_absolute_error")):
        for method, table in tables.items():
            domain_values = _domain_curve(table, method=method, error_column=error_column)
            equal_values = _equal_curve(
                domain_values, domain_order=domain_order, checkpoints=checkpoints
            )
            by_protocol[protocol][method] = domain_values
            equal[protocol][method] = equal_values
            for index, checkpoint in enumerate(checkpoints):
                effective_mean, effective_min, effective_max = _effective_summary(
                    table, method=method, checkpoint=checkpoint
                )
                curves.append(
                    {
                        "method": method,
                        "protocol": protocol,
                        "nominal_checkpoint": checkpoint,
                        "equal_domain_mae": float(equal_values[index]),
                        "mae_mean": None,
                        "mae_median": None,
                        "mae_q05": None,
                        "mae_q95": None,
                        "effective_mean": effective_mean,
                        "effective_min": effective_min,
                        "effective_max": effective_max,
                    }
                )
        random_domain = {
            seed: _domain_curve(a2, method="random", error_column=error_column, seed=seed)
            for seed in random_seeds
        }
        by_protocol[protocol]["random_median"] = {
            domain: {
                checkpoint: float(
                    np.median(
                        [random_domain[seed][domain][checkpoint] for seed in random_seeds]
                    )
                )
                for checkpoint in checkpoints
            }
            for domain in domain_order
        }
        seed_equal = np.vstack(
            [
                _equal_curve(
                    random_domain[seed],
                    domain_order=domain_order,
                    checkpoints=checkpoints,
                )
                for seed in random_seeds
            ]
        )
        random_equal = np.median(seed_equal, axis=0)
        equal[protocol]["random_median"] = random_equal
        for index, checkpoint in enumerate(checkpoints):
            selected = a2.filter(
                (pl.col("method") == "random")
                & (pl.col("nominal_checkpoint") == checkpoint)
            ).sort(["seed", "dataset_id", "specimen_id"])
            curves.append(
                {
                    "method": "random_median",
                    "protocol": protocol,
                    "nominal_checkpoint": checkpoint,
                    "equal_domain_mae": float(random_equal[index]),
                    "mae_mean": float(np.mean(seed_equal[:, index], dtype=np.float64)),
                    "mae_median": float(np.median(seed_equal[:, index])),
                    "mae_q05": float(np.quantile(seed_equal[:, index], 0.05)),
                    "mae_q95": float(np.quantile(seed_equal[:, index], 0.95)),
                    "effective_mean": float(np.mean(selected["effective_budget"].to_numpy(), dtype=np.float64)),
                    "effective_min": float(selected["effective_budget"].min()),
                    "effective_max": float(selected["effective_budget"].max()),
                }
            )
        equal[protocol]["FULL"] = np.full(len(checkpoints), full, dtype=np.float64)
        by_protocol[protocol]["FULL"] = {
            domain: {checkpoint: full for checkpoint in checkpoints}
            for domain in domain_order
        }
        for checkpoint in checkpoints:
            curves.append(
                {
                    "method": "FULL",
                    "protocol": protocol,
                    "nominal_checkpoint": checkpoint,
                    "equal_domain_mae": full,
                    "mae_mean": None,
                    "mae_median": None,
                    "mae_q05": None,
                    "mae_q95": None,
                    "effective_mean": 1.0,
                    "effective_min": 1.0,
                    "effective_max": 1.0,
                }
            )

    checkpoint_array = np.asarray(checkpoints, dtype=np.float64)
    primary = by_protocol["P-B"]
    primary_equal = equal["P-B"]
    domain_auebc = {
        method: {
            domain: auebc(
                checkpoint_array,
                [primary[method][domain][checkpoint] for checkpoint in checkpoints],
            )
            for domain in domain_order
        }
        for method in METHODS
    }
    areas = {
        method: auebc(checkpoint_array, primary_equal[method]) for method in METHODS
    }
    indices = synchronized_bootstrap_indices(
        seed=bootstrap_seed, resamples=bootstrap_resamples, domains=len(domain_order)
    )
    effects = tuple(
        paired_domain_bootstrap(
            [domain_auebc[baseline][domain] for domain in domain_order],
            [domain_auebc["one_shot_mechanical_oracle"][domain] for domain in domain_order],
            indices=indices,
            effect_id=f"{baseline}_minus_one_shot_mechanical_oracle_auebc",
        )
        for baseline in ("uniform", "one_shot_reconstruction")
    )
    uniform_pass = (
        effects[0].point_estimate > 0.0
        and effects[0].lower > 0.0
        and effects[0].improved_domains >= minimum_improved_domains
    )
    reconstruction_pass = (
        effects[1].point_estimate > 0.0
        and effects[1].lower > 0.0
        and effects[1].improved_domains >= minimum_improved_domains
    )
    stronger = min(("uniform", "one_shot_reconstruction"), key=areas.__getitem__)
    denominator = areas[stronger] - areas["sequential_mechanical_oracle"]
    retention = (
        None
        if denominator <= 0.0
        else float((areas[stronger] - areas["one_shot_mechanical_oracle"]) / denominator)
    )
    headroom_pass = retention is not None and retention >= minimum_headroom_retention
    go = uniform_pass and reconstruction_pass and headroom_pass
    status = (
        "MVD_ONE_SHOT_STRONG_GO"
        if go and retention is not None and retention >= strong_headroom_retention
        else "MVD_ONE_SHOT_GO"
        if go
        else "MVD_ONE_SHOT_NO_GO"
    )
    gate = M0GateResult(
        status=status,
        go=go,
        uniform_pass=uniform_pass,
        reconstruction_pass=reconstruction_pass,
        headroom_pass=headroom_pass,
        headroom_retention=retention,
        stronger_baseline=stronger,
        stronger_baseline_auebc=areas[stronger],
        one_shot_auebc=areas["one_shot_mechanical_oracle"],
        sequential_auebc=areas["sequential_mechanical_oracle"],
        uniform_effect=effects[0],
        reconstruction_effect=effects[1],
    )
    domain_metrics: list[dict[str, object]] = []
    for domain in domain_order:
        for method in METHODS:
            values = np.asarray(
                [primary[method][domain][checkpoint] for checkpoint in checkpoints],
                dtype=np.float64,
            )
            domain_metrics.append(
                {
                    "dataset_id": domain,
                    "method": method,
                    "auebc": domain_auebc[method][domain],
                    "mae_0p125": float(values[checkpoints.index(0.125)]),
                    "b_5": 1.0
                    if method == "FULL"
                    else sufficiency_budget(checkpoint_array, values, full_mae=full),
                    "uniform_minus_method_auebc": domain_auebc["uniform"][domain]
                    - domain_auebc[method][domain],
                    "reconstruction_minus_method_auebc": domain_auebc[
                        "one_shot_reconstruction"
                    ][domain]
                    - domain_auebc[method][domain],
                }
            )
    budget_metrics = tuple(
        {
            "method": method,
            "auebc": areas[method],
            "b_2p5": 1.0
            if method == "FULL"
            else sufficiency_budget(
                checkpoint_array, primary_equal[method], full_mae=full, tolerance=0.025
            ),
            "b_5": 1.0
            if method == "FULL"
            else sufficiency_budget(checkpoint_array, primary_equal[method], full_mae=full),
            "b_7p5": 1.0
            if method == "FULL"
            else sufficiency_budget(
                checkpoint_array, primary_equal[method], full_mae=full, tolerance=0.075
            ),
            "worst_domain_uniform_minus_method_auebc": min(
                domain_auebc["uniform"][domain] - domain_auebc[method][domain]
                for domain in domain_order
            ),
        }
        for method in METHODS
    )
    result = M0Aggregation(
        domain_order=domain_order,
        checkpoints=checkpoints,
        random_seeds=random_seeds,
        curves=tuple(
            sorted(
                curves,
                key=lambda row: (
                    str(row["protocol"]),
                    METHODS.index(str(row["method"])),
                    float(row["nominal_checkpoint"]),
                ),
            )
        ),
        domain_metrics=tuple(domain_metrics),
        budget_metrics=budget_metrics,
        bootstrap_effects=effects,
        gate=gate,
        state_sha256="",
    )
    return replace(result, state_sha256=_state(result))


__all__ = [
    "METHODS",
    "M0Aggregation",
    "M0GateResult",
    "aggregate_m0_tables",
]
