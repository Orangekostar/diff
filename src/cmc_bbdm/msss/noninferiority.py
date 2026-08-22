"""Frozen relative non-inferiority rule for MSSS candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


class NoninferiorityError(ValueError):
    """Raised when a non-inferiority curve is invalid."""


@dataclass(frozen=True, slots=True)
class NoninferiorityResult:
    candidates: tuple[float, ...]
    scores: tuple[float, ...]
    full_score: float
    margin: float
    threshold: float
    sufficient_candidates: tuple[float, ...]
    selected: float
    over_coarse: float | None
    boundary_confirmed: bool
    plateau: bool


def select_noninferior(
    candidates: Sequence[float],
    scores: Sequence[float],
    *,
    margin: float,
    minimum_plateau_nonfull_candidates: int = 2,
) -> NoninferiorityResult:
    """Select the last eligible candidate in a fine-to-coarse registry."""

    try:
        scale_values = tuple(float(value) for value in candidates)
        score_values = tuple(float(value) for value in scores)
        relative_margin = float(margin)
    except (TypeError, ValueError, OverflowError) as error:
        raise NoninferiorityError("non-inferiority inputs must be numeric") from error
    if (
        not scale_values
        or len(scale_values) != len(score_values)
        or len(set(scale_values)) != len(scale_values)
        or any(not math.isfinite(value) for value in scale_values)
        or any(not math.isfinite(value) or value < 0.0 for value in score_values)
        or not math.isfinite(relative_margin)
        or relative_margin <= 0.0
        or type(minimum_plateau_nonfull_candidates) is not int
        or minimum_plateau_nonfull_candidates < 1
    ):
        raise NoninferiorityError("non-inferiority inputs are invalid")
    threshold = score_values[0] * (1.0 + relative_margin)
    eligible_indices = tuple(
        index for index, score in enumerate(score_values) if score <= threshold
    )
    if 0 not in eligible_indices:
        raise NoninferiorityError("FULL is missing from its sufficient set")
    selected_index = eligible_indices[-1]
    over_index = next(
        (
            index
            for index in range(selected_index + 1, len(scale_values))
            if index not in eligible_indices
        ),
        None,
    )
    longest = 0
    current = 0
    for index in range(1, len(scale_values)):
        if index in eligible_indices:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return NoninferiorityResult(
        candidates=scale_values,
        scores=score_values,
        full_score=score_values[0],
        margin=relative_margin,
        threshold=threshold,
        sufficient_candidates=tuple(scale_values[index] for index in eligible_indices),
        selected=scale_values[selected_index],
        over_coarse=None if over_index is None else scale_values[over_index],
        boundary_confirmed=over_index is not None,
        plateau=longest >= minimum_plateau_nonfull_candidates,
    )


__all__ = ["NoninferiorityError", "NoninferiorityResult", "select_noninferior"]
