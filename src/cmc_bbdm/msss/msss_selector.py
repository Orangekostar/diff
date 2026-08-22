"""Stability and S1 decision gates for MSSS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class MSSSSelectionError(ValueError):
    """Raised when a stability or gate input is invalid."""


@dataclass(frozen=True, slots=True)
class StabilityResult:
    passed: bool
    maximum_in_window: int
    window: tuple[str, ...]
    selected_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class AxisGate:
    status: str
    plateau: bool
    boundary_confirmed: bool
    stable: bool
    mechanically_sufficient: bool
    spatially_specific: bool


@dataclass(frozen=True, slots=True)
class S1Gate:
    status: str
    passing_axes: int
    total_axes: int


def selection_stability(
    selected: Sequence[str],
    *,
    candidate_order: Sequence[str],
    minimum_outer_folds: int = 4,
    window_steps: int = 1,
) -> StabilityResult:
    """Count source selections inside each contiguous candidate window."""

    selections = tuple(selected)
    order = tuple(candidate_order)
    if (
        not selections
        or not order
        or len(set(order)) != len(order)
        or any(value not in order for value in selections)
        or type(minimum_outer_folds) is not int
        or minimum_outer_folds < 1
        or type(window_steps) is not int
        or window_steps < 0
    ):
        raise MSSSSelectionError("selection stability inputs are invalid")
    counts = tuple((candidate, selections.count(candidate)) for candidate in order)
    width = window_steps + 1
    windows = tuple(
        order[start : start + width]
        for start in range(max(1, len(order) - width + 1))
    )
    ranked = tuple(
        (sum(selections.count(candidate) for candidate in window), index, window)
        for index, window in enumerate(windows)
    )
    maximum, _index, best = max(ranked, key=lambda item: (item[0], -item[1]))
    return StabilityResult(
        passed=maximum >= minimum_outer_folds,
        maximum_in_window=maximum,
        window=best,
        selected_counts=counts,
    )


def axis_gate(
    *,
    plateau: bool,
    boundary_confirmed: bool,
    stable: bool,
    mechanically_sufficient: bool,
    spatially_specific: bool,
) -> AxisGate:
    values = (
        plateau,
        boundary_confirmed,
        stable,
        mechanically_sufficient,
        spatially_specific,
    )
    if any(type(value) is not bool for value in values):
        raise MSSSSelectionError("axis gate inputs must be boolean")
    return AxisGate(
        status="PASS" if all(values) else "FAIL",
        plateau=plateau,
        boundary_confirmed=boundary_confirmed,
        stable=stable,
        mechanically_sufficient=mechanically_sufficient,
        spatially_specific=spatially_specific,
    )


def s1_gate(axes: Sequence[AxisGate]) -> S1Gate:
    registry = tuple(axes)
    if len(registry) != 3 or any(type(item) is not AxisGate for item in registry):
        raise MSSSSelectionError("S1 requires exactly three primary axis gates")
    passing = sum(item.status == "PASS" for item in registry)
    status = "STRONG_GO" if passing == 3 else "GO" if passing >= 2 else "NO_GO"
    return S1Gate(status=status, passing_axes=passing, total_axes=3)


__all__ = [
    "AxisGate",
    "MSSSSelectionError",
    "S1Gate",
    "StabilityResult",
    "axis_gate",
    "s1_gate",
    "selection_stability",
]
