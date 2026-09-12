"""Pure predeclared stage gates for predictor and policy execution."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True, slots=True)
class GateResult:
    status: str
    passed: bool
    reasons: tuple[str, ...]


def predictor_readiness_gate(
    metrics: Mapping[str, float],
) -> GateResult:
    required = (
        "full_mae_mpa",
        "train_median_full_mae_mpa",
        "full_mse_mpa2",
        "train_mean_full_mse_mpa2",
        "zero_mae_mpa",
        "center_endpoint_mae_mpa",
        "geometry_endpoint_mae_mpa",
        "valid_area_mpa",
    )
    if any(name not in metrics for name in required) or not all(
        math.isfinite(float(metrics[name])) for name in required
    ):
        return GateResult(
            "PREDICTOR_NOT_READY", False, ("NUMERIC_OR_LABEL_CONNECTION_INVALID",)
        )
    failures: list[str] = []
    if not metrics["full_mae_mpa"] < metrics["train_median_full_mae_mpa"]:
        failures.append("FULL_MAE_NOT_BELOW_TRAIN_MEDIAN_CONSTANT")
    if not metrics["full_mse_mpa2"] < metrics["train_mean_full_mse_mpa2"]:
        failures.append("FULL_MSE_NOT_BELOW_TRAIN_MEAN_CONSTANT")
    if not metrics["full_mae_mpa"] <= 0.98 * metrics["zero_mae_mpa"]:
        failures.append("FULL_MAE_IMPROVEMENT_BELOW_2_PERCENT")
    if not metrics["center_endpoint_mae_mpa"] < metrics["zero_mae_mpa"]:
        failures.append("CENTER_ROUTE_NOT_BELOW_ZERO")
    if not metrics["geometry_endpoint_mae_mpa"] < metrics["zero_mae_mpa"]:
        failures.append("GEOMETRY_ROUTE_NOT_BELOW_ZERO")
    return GateResult(
        "PREDICTOR_READY" if not failures else "PREDICTOR_NOT_READY",
        not failures,
        tuple(failures),
    )


def choose_common_predictor(
    candidates: Sequence[tuple[str, Mapping[str, float], int]],
) -> tuple[str | None, dict[str, GateResult]]:
    results = {
        name: predictor_readiness_gate(metrics) for name, metrics, _ in candidates
    }
    ready = [
        (float(metrics["valid_area_mpa"]), parameters, name)
        for name, metrics, parameters in candidates
        if results[name].passed
    ]
    if not ready:
        return None, results
    best_area = min(row[0] for row in ready)
    tied = [row for row in ready if abs(row[0] - best_area) <= 1e-8]
    return min(tied, key=lambda row: (row[1], row[2]))[2], results


def policy_pilot_gate(
    *,
    main_area_mpa: float,
    best_nonadaptive_area_mpa: float,
    open_loop_area_mpa: float,
) -> GateResult:
    values = (main_area_mpa, best_nonadaptive_area_mpa, open_loop_area_mpa)
    if not all(math.isfinite(float(value)) for value in values):
        return GateResult(
            "POLICY_PILOT_NOT_SUPPORTED", False, ("PILOT_METRIC_INVALID",)
        )
    failures = []
    if not main_area_mpa <= 0.98 * best_nonadaptive_area_mpa:
        failures.append("MAIN_IMPROVEMENT_BELOW_2_PERCENT_VS_BEST_NONADAPTIVE")
    if not main_area_mpa < open_loop_area_mpa:
        failures.append("MAIN_NOT_BELOW_VLM_OPEN_LOOP")
    return GateResult(
        "POLICY_PILOT_SUPPORTED" if not failures else "POLICY_PILOT_NOT_SUPPORTED",
        not failures,
        tuple(failures),
    )


_T = TypeVar("_T")


def execute_if_status(
    *, required_status: str, observed_status: str, action: Callable[[], _T]
) -> _T | None:
    """Execute a downstream stage only for the exact registered gate status."""

    if observed_status != required_status:
        return None
    return action()


__all__ = [
    "GateResult",
    "choose_common_predictor",
    "execute_if_status",
    "policy_pilot_gate",
    "predictor_readiness_gate",
]
