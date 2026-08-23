"""Retrospective CAI error-reduction labels for MVA."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MechanicalValue:
    absolute_error_before: float
    absolute_error_after: float
    absolute_error_reduction: float
    squared_error_reduction: float


def mechanical_value(
    *, target: float, current_prediction: float, candidate_prediction: float
) -> MechanicalValue:
    """Calculate primary absolute and secondary squared CAI error reduction."""

    values = (float(target), float(current_prediction), float(candidate_prediction))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("mechanical value inputs must be finite")
    response, current, candidate = values
    before = abs(response - current)
    after = abs(response - candidate)
    return MechanicalValue(
        absolute_error_before=before,
        absolute_error_after=after,
        absolute_error_reduction=before - after,
        squared_error_reduction=(response - current) ** 2 - (response - candidate) ** 2,
    )


__all__ = ["MechanicalValue", "mechanical_value"]
