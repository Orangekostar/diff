"""Immutable MVA refinement states and unique-location budget accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .acquisition_grid import AcquisitionGrid


class MeasurementStateError(ValueError):
    """Raised when a refinement state or action violates the grid contract."""


@dataclass(frozen=True, slots=True)
class MeasurementState:
    grid_sha256: str
    levels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RefinementAction:
    cell_index: int
    from_level: int
    to_level: int


@dataclass(frozen=True, slots=True)
class BudgetRecord:
    measured_count: int
    native_count: int
    effective_budget: float


def _validate_state(grid: AcquisitionGrid, state: MeasurementState) -> None:
    if type(grid) is not AcquisitionGrid or type(state) is not MeasurementState:
        raise MeasurementStateError("issued grid and state are required")
    if (
        state.grid_sha256 != grid.state_sha256
        or len(state.levels) != 64
        or any(
            type(level) is not int or level not in (0, 1, 2) for level in state.levels
        )
    ):
        raise MeasurementStateError("state does not belong to the grid")


def initial_state(grid: AcquisitionGrid) -> MeasurementState:
    if type(grid) is not AcquisitionGrid:
        raise MeasurementStateError("issued grid is required")
    return MeasurementState(grid_sha256=grid.state_sha256, levels=(0,) * 64)


def legal_actions(
    grid: AcquisitionGrid, state: MeasurementState
) -> tuple[RefinementAction, ...]:
    _validate_state(grid, state)
    return tuple(
        RefinementAction(index, level, level + 1)
        for index, level in enumerate(state.levels)
        if level < 2
    )


def apply_action(
    grid: AcquisitionGrid, state: MeasurementState, action: RefinementAction
) -> MeasurementState:
    _validate_state(grid, state)
    if (
        type(action) is not RefinementAction
        or not 0 <= action.cell_index < 64
        or state.levels[action.cell_index] != action.from_level
        or action.to_level != action.from_level + 1
        or action.to_level not in (1, 2)
    ):
        raise MeasurementStateError("action is not a legal one-level refinement")
    levels = list(state.levels)
    levels[action.cell_index] = action.to_level
    return MeasurementState(grid_sha256=grid.state_sha256, levels=tuple(levels))


def measurement_mask(grid: AcquisitionGrid, state: MeasurementState) -> np.ndarray:
    _validate_state(grid, state)
    mask = np.zeros(grid.native_shape, dtype=np.bool_)
    for cell, level in zip(grid.cells, state.levels, strict=True):
        rows = np.asarray(cell.rows[level], dtype=np.int64)
        columns = np.asarray(cell.columns[level], dtype=np.int64)
        mask[np.ix_(rows, columns)] = True
    return mask


def budget_record(grid: AcquisitionGrid, state: MeasurementState) -> BudgetRecord:
    mask = measurement_mask(grid, state)
    measured = int(np.count_nonzero(mask))
    native = int(mask.size)
    return BudgetRecord(
        measured_count=measured,
        native_count=native,
        effective_budget=float(measured / native),
    )


def action_fits_checkpoint(
    grid: AcquisitionGrid,
    state: MeasurementState,
    action: RefinementAction,
    checkpoint: float,
) -> bool:
    if isinstance(checkpoint, bool) or not isinstance(checkpoint, (int, float)):
        raise MeasurementStateError("checkpoint must be finite")
    cap = float(checkpoint)
    if not math.isfinite(cap) or not 0.0 < cap <= 1.0:
        raise MeasurementStateError("checkpoint must be in (0,1]")
    refined = apply_action(grid, state, action)
    return budget_record(grid, refined).effective_budget <= cap


def fitting_actions(
    grid: AcquisitionGrid,
    state: MeasurementState,
    checkpoint: float,
) -> tuple[RefinementAction, ...]:
    """Return fitting actions after one shared current-mask calculation."""

    if isinstance(checkpoint, bool) or not isinstance(checkpoint, (int, float)):
        raise MeasurementStateError("checkpoint must be finite")
    cap = float(checkpoint)
    if not math.isfinite(cap) or not 0.0 < cap <= 1.0:
        raise MeasurementStateError("checkpoint must be in (0,1]")
    current = measurement_mask(grid, state)
    current_count = int(np.count_nonzero(current))
    native_count = int(current.size)
    output: list[RefinementAction] = []
    for action in legal_actions(grid, state):
        cell = grid.cells[action.cell_index]
        rows = np.asarray(cell.rows[action.to_level], dtype=np.int64)
        columns = np.asarray(cell.columns[action.to_level], dtype=np.int64)
        added = int(np.count_nonzero(~current[np.ix_(rows, columns)]))
        if float((current_count + added) / native_count) <= cap:
            output.append(action)
    return tuple(output)


def candidate_budget_record(
    grid: AcquisitionGrid,
    state: MeasurementState,
    action: RefinementAction,
) -> BudgetRecord:
    """Count one candidate from the current mask without building its full mask."""

    apply_action(grid, state, action)
    current = measurement_mask(grid, state)
    current_count = int(np.count_nonzero(current))
    cell = grid.cells[action.cell_index]
    rows = np.asarray(cell.rows[action.to_level], dtype=np.int64)
    columns = np.asarray(cell.columns[action.to_level], dtype=np.int64)
    added = int(np.count_nonzero(~current[np.ix_(rows, columns)]))
    native = int(current.size)
    measured = current_count + added
    return BudgetRecord(
        measured_count=measured,
        native_count=native,
        effective_budget=float(measured / native),
    )


__all__ = [
    "BudgetRecord",
    "MeasurementState",
    "MeasurementStateError",
    "RefinementAction",
    "action_fits_checkpoint",
    "apply_action",
    "budget_record",
    "candidate_budget_record",
    "fitting_actions",
    "initial_state",
    "legal_actions",
    "measurement_mask",
]
