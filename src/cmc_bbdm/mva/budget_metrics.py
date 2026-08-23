"""Preregistered error-budget and mechanical-sufficiency metrics."""

from __future__ import annotations

import math

import numpy as np


def _curve(budgets: object, values: object) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(budgets, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("curve inputs must be numeric") from error
    if (
        x.ndim != 1
        or y.shape != x.shape
        or x.size < 1
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("curve inputs must be aligned finite vectors")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("budgets must be strictly increasing")
    if x[0] <= 0.0 or x[-1] > 1.0:
        raise ValueError("budgets must be in (0,1]")
    return x, y


def auebc(
    budgets: object,
    mae: object,
    *,
    lower: float = 0.0625,
    upper: float = 0.25,
) -> float:
    """Trapezoidal area under the MAE-budget curve on a fixed interval."""

    x, y = _curve(budgets, mae)
    low, high = float(lower), float(upper)
    if not math.isfinite(low) or not math.isfinite(high) or not 0.0 < low < high <= 1.0:
        raise ValueError("AUEBC interval is invalid")
    if not np.any(x == low) or not np.any(x == high):
        raise ValueError("AUEBC endpoints are missing")
    selected = (x >= low) & (x <= high)
    return float(np.trapezoid(y[selected], x[selected]))


def sufficiency_budget(
    budgets: object,
    mae: object,
    *,
    full_mae: float,
    tolerance: float = 0.05,
) -> float | None:
    """Return the first observed checkpoint within tolerance of FULL."""

    x, y = _curve(budgets, mae)
    full = float(full_mae)
    margin = float(tolerance)
    if (
        not math.isfinite(full)
        or full <= 0.0
        or not math.isfinite(margin)
        or margin < 0.0
    ):
        raise ValueError("sufficiency threshold must be finite")
    passing = np.flatnonzero(y <= (1.0 + margin) * full)
    return None if not passing.size else float(x[int(passing[0])])


def simulated_saving(
    adaptive_budget: float | None, uniform_budget: float | None
) -> float | None:
    """Return `1 - adaptive/uniform`, preserving unavailable sufficiency."""

    if adaptive_budget is None or uniform_budget is None:
        return None
    adaptive, uniform = float(adaptive_budget), float(uniform_budget)
    if (
        not math.isfinite(adaptive)
        or not math.isfinite(uniform)
        or adaptive <= 0.0
        or uniform <= 0.0
    ):
        raise ValueError("saving budgets must be positive and finite")
    return float(1.0 - adaptive / uniform)


__all__ = ["auebc", "simulated_saving", "sufficiency_budget"]
