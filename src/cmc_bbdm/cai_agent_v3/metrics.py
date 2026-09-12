"""Pure left-constant trajectory metrics shared by v3 training and evaluation."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from itertools import pairwise

import torch


def _validated_states(
    costs: Sequence[float], predictions_mpa: Sequence[float], target_mpa: float
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    x = tuple(float(value) for value in costs)
    predictions = tuple(float(value) for value in predictions_mpa)
    target = float(target_mpa)
    if not x or len(x) != len(predictions):
        raise ValueError("nonempty aligned trajectory states are required")
    if not all(math.isfinite(value) for value in (*x, *predictions, target)):
        raise ValueError("trajectory values must be finite")
    if x[0] != 0.0 or any(right <= left for left, right in pairwise(x)):
        raise ValueError("costs must start at zero and be strictly increasing")
    return x, predictions, target


def prediction_at_budget(
    costs: Sequence[float], predictions_mpa: Sequence[float], budget: float
) -> float:
    """Return the last prediction acquired at or before ``budget``."""

    x, predictions, _ = _validated_states(costs, predictions_mpa, 0.0)
    point = float(budget)
    if not math.isfinite(point) or point < 0.0:
        raise ValueError("budget must be finite and nonnegative")
    return predictions[bisect.bisect_right(x, point) - 1]


def left_error_area_mpa(
    costs: Sequence[float],
    predictions_mpa: Sequence[float],
    target_mpa: float,
    *,
    start: float = 0.0,
    end: float = 0.25,
) -> float:
    """Compute normalized left-constant absolute-error area on ``[start, end]``."""

    x, predictions, target = _validated_states(costs, predictions_mpa, target_mpa)
    lower = float(start)
    upper = float(end)
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower < 0.0
        or lower >= upper
    ):
        raise ValueError("integration interval is invalid")
    points = (lower, *(cost for cost in x if lower < cost < upper), upper)
    weighted_error = 0.0
    for left, right in pairwise(points):
        prediction = predictions[bisect.bisect_right(x, left) - 1]
        weighted_error += (right - left) * abs(prediction - target)
    return weighted_error / (upper - lower)


def _trajectory_terms(
    costs: Sequence[float],
    predictions_mpa: Sequence[float],
    target_mpa: float,
    *,
    budget: float,
    terminal_weight: float,
) -> tuple[tuple[float, ...], float, float]:
    x, predictions, target = _validated_states(costs, predictions_mpa, target_mpa)
    endpoint = float(budget)
    weight = float(terminal_weight)
    if not math.isfinite(endpoint) or endpoint <= 0.0 or x[-1] > endpoint + 1e-12:
        raise ValueError("trajectory exceeds the valid endpoint budget")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("terminal weight is invalid")
    errors = tuple(abs(prediction - target) for prediction in predictions)
    local = tuple(
        (x[index + 1] - x[index]) * errors[index] / endpoint
        for index in range(len(x) - 1)
    )
    tail = max(0.0, endpoint - x[-1]) * errors[-1] / endpoint
    terminal = tail + weight * errors[-1]
    return local, tail, terminal


def trajectory_objective_mpa(
    costs: Sequence[float],
    predictions_mpa: Sequence[float],
    target_mpa: float,
    *,
    budget: float = 0.25,
    terminal_weight: float = 0.25,
) -> float:
    """Return ``A(B) + terminal_weight * e_T`` in MPa."""

    local, _, terminal = _trajectory_terms(
        costs,
        predictions_mpa,
        target_mpa,
        budget=budget,
        terminal_weight=terminal_weight,
    )
    return sum(local) + terminal


def policy_cost_to_go(
    costs: Sequence[float],
    predictions_mpa: Sequence[float],
    target_mpa: float,
    *,
    budget: float = 0.25,
    terminal_weight: float = 0.25,
) -> tuple[float, ...]:
    """Return one cost-to-go value for each executed action."""

    local, _, terminal = _trajectory_terms(
        costs,
        predictions_mpa,
        target_mpa,
        budget=budget,
        terminal_weight=terminal_weight,
    )
    running = terminal
    reverse: list[float] = []
    for value in reversed(local):
        running += value
        reverse.append(running)
    return tuple(reversed(reverse))


def torch_policy_cost_to_go(
    costs: torch.Tensor,
    predictions_mpa: torch.Tensor,
    target_mpa: torch.Tensor,
    *,
    budget: float = 0.25,
    terminal_weight: float = 0.25,
) -> torch.Tensor:
    """Torch adapter of :func:`policy_cost_to_go` for one trajectory."""

    if (
        costs.ndim != 1
        or predictions_mpa.shape != costs.shape
        or target_mpa.numel() != 1
        or costs.numel() < 1
        or not torch.isfinite(costs).all()
        or not torch.isfinite(predictions_mpa).all()
        or not torch.isfinite(target_mpa).all()
        or float(costs[0].detach()) != 0.0
        or (costs.numel() > 1 and not torch.all(costs[1:] > costs[:-1]))
        or float(costs[-1].detach()) > budget + 1e-12
    ):
        raise ValueError("Torch trajectory states are invalid")
    errors = torch.abs(predictions_mpa - target_mpa.reshape(()))
    local = (costs[1:] - costs[:-1]) * errors[:-1] / budget
    terminal = (budget - costs[-1]).clamp_min(0.0) * errors[
        -1
    ] / budget + terminal_weight * errors[-1]
    if local.numel() == 0:
        return local
    return (
        torch.flip(torch.cumsum(torch.flip(local, dims=(0,)), dim=0), dims=(0,))
        + terminal
    )


__all__ = [
    "left_error_area_mpa",
    "policy_cost_to_go",
    "prediction_at_budget",
    "torch_policy_cost_to_go",
    "trajectory_objective_mpa",
]
