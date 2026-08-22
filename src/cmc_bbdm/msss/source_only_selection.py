"""Source-only scale selection with no target-facing API."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .scale_features import ScaleCondition


class SourceSelectionError(ValueError):
    """Raised when source-only scale evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class SourceScaleDecision:
    selected_condition_id: str
    full_condition_id: str
    over_coarse_condition_id: str | None
    boundary_confirmed: bool
    sufficient_sets: tuple[tuple[float, tuple[str, ...]], ...]
    candidate_scores: tuple[tuple[str, float], ...]


def select_source_scale(
    conditions: Sequence[ScaleCondition],
    source_scores: Mapping[str, float],
    *,
    margins: Sequence[float],
    primary_margin: float,
) -> SourceScaleDecision:
    """Select the coarsest non-inferior candidate from source scores only."""

    registry = tuple(conditions)
    if (
        not registry
        or any(type(item) is not ScaleCondition for item in registry)
        or len({item.condition_id for item in registry}) != len(registry)
        or set(source_scores) != {item.condition_id for item in registry}
    ):
        raise SourceSelectionError("source scale registry or scores are incomplete")
    numeric_scores: dict[str, float] = {}
    for condition in registry:
        try:
            score = float(source_scores[condition.condition_id])
        except (TypeError, ValueError, OverflowError) as error:
            raise SourceSelectionError("source scale score must be finite") from error
        if not math.isfinite(score) or score < 0.0:
            raise SourceSelectionError("source scale score must be finite and nonnegative")
        numeric_scores[condition.condition_id] = score
    full = tuple(item for item in registry if item.is_full_identity)
    if len(full) != 1:
        raise SourceSelectionError("axis must contain exactly one FULL identity")
    margin_tuple = tuple(float(value) for value in margins)
    if (
        not margin_tuple
        or any(not math.isfinite(value) or value <= 0.0 for value in margin_tuple)
        or float(primary_margin) not in margin_tuple
    ):
        raise SourceSelectionError("non-inferiority margins are invalid")
    full_score = numeric_scores[full[0].condition_id]
    sufficient_sets: list[tuple[float, tuple[str, ...]]] = []
    by_primary: tuple[ScaleCondition, ...] | None = None
    for margin in margin_tuple:
        eligible = tuple(
            item
            for item in registry
            if numeric_scores[item.condition_id] <= full_score * (1.0 + margin)
        )
        sufficient_sets.append(
            (margin, tuple(item.condition_id for item in eligible))
        )
        if margin == float(primary_margin):
            by_primary = eligible
    if not by_primary:
        raise SourceSelectionError("FULL must remain in the primary sufficient set")
    selected = max(by_primary, key=lambda item: item.coarse_rank)
    eligible_ids = {item.condition_id for item in by_primary}
    outside = tuple(
        item
        for item in registry
        if item.coarse_rank > selected.coarse_rank
        and item.condition_id not in eligible_ids
    )
    over_coarse = min(outside, key=lambda item: item.coarse_rank) if outside else None
    return SourceScaleDecision(
        selected_condition_id=selected.condition_id,
        full_condition_id=full[0].condition_id,
        over_coarse_condition_id=(
            None if over_coarse is None else over_coarse.condition_id
        ),
        boundary_confirmed=over_coarse is not None,
        sufficient_sets=tuple(sufficient_sets),
        candidate_scores=tuple(
            (item.condition_id, numeric_scores[item.condition_id]) for item in registry
        ),
    )


__all__ = ["SourceScaleDecision", "SourceSelectionError", "select_source_scale"]
