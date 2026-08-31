"""Immutable zero-start acquisition state with unique-pixel exact cost."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid


class GeneralizedMeasurementStateError(ValueError):
    """Raised when a zero-start state or transition violates its grid."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _state_hash(grid_sha256: str, levels: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "schema": 1,
                "kind": "inspection-agent-generalized-measurement-state",
                "grid_sha256": grid_sha256,
                "levels": levels,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class GeneralizedMeasurementState:
    grid_sha256: str
    levels: tuple[int, ...]
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.grid_sha256)
            or type(self.levels) is not tuple
            or len(self.levels) != 64
            or any(type(level) is not int or level not in (-1, 0, 1, 2) for level in self.levels)
        ):
            raise GeneralizedMeasurementStateError("generalized state is invalid")
        digest = _state_hash(self.grid_sha256, self.levels)
        if self.state_sha256 not in ("", digest):
            raise GeneralizedMeasurementStateError("generalized state hash changed")
        object.__setattr__(self, "state_sha256", digest)


@dataclass(frozen=True, slots=True)
class InspectionCellAction:
    cell_index: int
    from_level: int
    to_level: int

    def __post_init__(self) -> None:
        if (
            type(self.cell_index) is not int
            or not 0 <= self.cell_index < 64
            or type(self.from_level) is not int
            or type(self.to_level) is not int
            or self.from_level not in (-1, 0, 1, 2)
            or self.to_level not in (-1, 0, 1, 2)
        ):
            raise GeneralizedMeasurementStateError("inspection action is invalid")


@dataclass(frozen=True, slots=True)
class GeneralizedBudgetRecord:
    measured_count: int
    native_count: int
    effective_budget: float


def _validate(grid: AcquisitionGrid, state: GeneralizedMeasurementState) -> None:
    if (
        type(grid) is not AcquisitionGrid
        or type(state) is not GeneralizedMeasurementState
        or state.grid_sha256 != grid.state_sha256
    ):
        raise GeneralizedMeasurementStateError("state does not belong to the grid")


def zero_state(grid: AcquisitionGrid) -> GeneralizedMeasurementState:
    if type(grid) is not AcquisitionGrid:
        raise GeneralizedMeasurementStateError("issued grid is required")
    return GeneralizedMeasurementState(grid.state_sha256, (-1,) * 64)


def legal_actions(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
) -> tuple[InspectionCellAction, ...]:
    _validate(grid, state)
    return tuple(
        InspectionCellAction(index, level, level + 1)
        for index, level in enumerate(state.levels)
        if level < 2
    )


def apply_action(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
    action: InspectionCellAction,
) -> GeneralizedMeasurementState:
    _validate(grid, state)
    if (
        type(action) is not InspectionCellAction
        or state.levels[action.cell_index] != action.from_level
        or action.to_level != action.from_level + 1
    ):
        raise GeneralizedMeasurementStateError("action is not a legal one-level transition")
    levels = list(state.levels)
    levels[action.cell_index] = action.to_level
    return GeneralizedMeasurementState(grid.state_sha256, tuple(levels))


def _mutable_mask(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
) -> np.ndarray:
    _validate(grid, state)
    mask = np.zeros(grid.native_shape, dtype=np.bool_)
    for cell, level in zip(grid.cells, state.levels, strict=True):
        if level < 0:
            continue
        rows = np.asarray(cell.rows[level], dtype=np.int64)
        columns = np.asarray(cell.columns[level], dtype=np.int64)
        mask[np.ix_(rows, columns)] = True
    return mask


def measurement_mask(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
) -> np.ndarray:
    mask = _mutable_mask(grid, state)
    mask.setflags(write=False)
    return mask


def budget_record(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
) -> GeneralizedBudgetRecord:
    mask = _mutable_mask(grid, state)
    measured = int(np.count_nonzero(mask))
    native = int(mask.size)
    return GeneralizedBudgetRecord(measured, native, float(measured / native))


def action_added_positions(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
    action: InspectionCellAction,
) -> np.ndarray:
    candidate = apply_action(grid, state, action)
    current_mask = _mutable_mask(grid, state)
    candidate_mask = _mutable_mask(grid, candidate)
    positions = np.argwhere(candidate_mask & ~current_mask).astype("<i8", copy=False)
    output = np.frombuffer(
        np.ascontiguousarray(positions).tobytes(order="C"), dtype="<i8"
    ).reshape(positions.shape)
    output.setflags(write=False)
    return output


def candidate_budget_record(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
    action: InspectionCellAction,
) -> GeneralizedBudgetRecord:
    current = budget_record(grid, state)
    added = action_added_positions(grid, state, action).shape[0]
    measured = current.measured_count + added
    return GeneralizedBudgetRecord(
        measured,
        current.native_count,
        float(measured / current.native_count),
    )


def fitting_actions(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
    checkpoint: float,
) -> tuple[InspectionCellAction, ...]:
    if isinstance(checkpoint, bool) or not isinstance(checkpoint, (int, float)):
        raise GeneralizedMeasurementStateError("checkpoint is invalid")
    cap = float(checkpoint)
    if not math.isfinite(cap) or not 0.0 < cap <= 1.0:
        raise GeneralizedMeasurementStateError("checkpoint is invalid")
    return tuple(
        action
        for action in legal_actions(grid, state)
        if candidate_budget_record(grid, state, action).effective_budget
        <= cap + 1.0e-15
    )


__all__ = [
    "GeneralizedBudgetRecord",
    "GeneralizedMeasurementState",
    "GeneralizedMeasurementStateError",
    "InspectionCellAction",
    "action_added_positions",
    "apply_action",
    "budget_record",
    "candidate_budget_record",
    "fitting_actions",
    "legal_actions",
    "measurement_mask",
    "zero_state",
]
