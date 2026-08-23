"""Nested endpoint-preserving native-raster acquisition geometry."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from cmc_bbdm.cpb_sparse_scan.sampling import sampling_coordinates


class AcquisitionGridError(ValueError):
    """Raised when a requested nested acquisition grid is invalid."""


INITIAL_BUDGETS = (0.015625, 0.03125, 0.0625)
CELL_SHAPE = (8, 8)


@dataclass(frozen=True, slots=True)
class CellLattices:
    index: int
    row: int
    column: int
    rows: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    columns: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class AcquisitionGrid:
    native_shape: tuple[int, int]
    cell_shape: tuple[int, int]
    initial_nominal_budget: float
    actual_initial_budget: float
    level0_rows: tuple[int, ...]
    level0_columns: tuple[int, ...]
    level1_rows: tuple[int, ...]
    level1_columns: tuple[int, ...]
    row_boundaries: tuple[int, ...]
    column_boundaries: tuple[int, ...]
    cells: tuple[CellLattices, ...]
    state_sha256: str


def _validate_dimension(value: object, label: str) -> int:
    if type(value) is not int or value < 9:
        raise AcquisitionGridError(f"{label} must be an integer >= 9")
    return value


def _validate_budget(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcquisitionGridError("initial budget must be registered")
    budget = float(value)
    if not math.isfinite(budget) or budget not in INITIAL_BUDGETS:
        raise AcquisitionGridError("initial budget must be registered")
    return budget


def _boundary_positions(length: int) -> tuple[int, ...]:
    positions = np.rint(np.linspace(0, length - 1, 9)).astype(np.int64)
    if len({int(value) for value in positions}) != 9:
        raise AcquisitionGridError("level-1 axis cannot support eight cells")
    return tuple(int(value) for value in positions)


def _nested_axis_order(values: tuple[int, ...]) -> tuple[int, ...]:
    boundary_positions = _boundary_positions(len(values))
    selected = [values[position] for position in boundary_positions]
    selected_set = set(selected)
    remaining = [value for value in values if value not in selected_set]
    while remaining:
        candidate = max(
            remaining,
            key=lambda value: (
                min(abs(value - chosen) for chosen in selected),
                -value,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return tuple(selected)


def _initial_axis(
    values: tuple[int, ...], native_length: int, budget: float
) -> tuple[int, ...]:
    count = max(9, int(np.rint(math.sqrt(budget) * native_length)))
    if count > len(values):
        raise AcquisitionGridError("initial axis exceeds P5 level-1 lattice")
    selected = tuple(sorted(_nested_axis_order(values)[:count]))
    if selected[0] != 0 or selected[-1] != native_length - 1:
        raise AcquisitionGridError("initial axis lost an endpoint")
    return selected


def _cell_axis(values: tuple[int, ...], lower: int, upper: int) -> tuple[int, ...]:
    output = tuple(value for value in values if lower <= value <= upper)
    if len(output) < 2 or output[0] != lower or output[-1] != upper:
        raise AcquisitionGridError("cell lattice does not retain boundaries")
    return output


def _state_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@lru_cache(maxsize=9)
def build_acquisition_grid(
    height: int, width: int, *, initial_budget: float
) -> AcquisitionGrid:
    """Build one immutable three-level grid from the registered P5 lattice."""

    native_height = _validate_dimension(height, "height")
    native_width = _validate_dimension(width, "width")
    budget = _validate_budget(initial_budget)
    p5 = sampling_coordinates(native_height, native_width, density=0.25)
    level1_rows = p5.rows
    level1_columns = p5.columns
    row_positions = _boundary_positions(len(level1_rows))
    column_positions = _boundary_positions(len(level1_columns))
    row_boundaries = tuple(level1_rows[position] for position in row_positions)
    column_boundaries = tuple(level1_columns[position] for position in column_positions)
    level0_rows = _initial_axis(level1_rows, native_height, budget)
    level0_columns = _initial_axis(level1_columns, native_width, budget)
    if not set(row_boundaries) <= set(level0_rows) or not set(column_boundaries) <= set(
        level0_columns
    ):
        raise AcquisitionGridError("initial lattice lost a cell boundary")

    cells: list[CellLattices] = []
    for row in range(8):
        for column in range(8):
            row_lower, row_upper = row_boundaries[row : row + 2]
            column_lower, column_upper = column_boundaries[column : column + 2]
            rows = (
                _cell_axis(level0_rows, row_lower, row_upper),
                _cell_axis(level1_rows, row_lower, row_upper),
                tuple(range(row_lower, row_upper + 1)),
            )
            columns = (
                _cell_axis(level0_columns, column_lower, column_upper),
                _cell_axis(level1_columns, column_lower, column_upper),
                tuple(range(column_lower, column_upper + 1)),
            )
            cells.append(
                CellLattices(
                    index=row * 8 + column,
                    row=row,
                    column=column,
                    rows=rows,
                    columns=columns,
                )
            )
    actual = float(
        len(level0_rows) * len(level0_columns) / (native_height * native_width)
    )
    state = _state_hash(
        {
            "native_shape": [native_height, native_width],
            "budget": budget,
            "level0_rows": level0_rows,
            "level0_columns": level0_columns,
            "level1_rows": level1_rows,
            "level1_columns": level1_columns,
            "row_boundaries": row_boundaries,
            "column_boundaries": column_boundaries,
        }
    )
    return AcquisitionGrid(
        native_shape=(native_height, native_width),
        cell_shape=CELL_SHAPE,
        initial_nominal_budget=budget,
        actual_initial_budget=actual,
        level0_rows=level0_rows,
        level0_columns=level0_columns,
        level1_rows=level1_rows,
        level1_columns=level1_columns,
        row_boundaries=row_boundaries,
        column_boundaries=column_boundaries,
        cells=tuple(cells),
        state_sha256=state,
    )


__all__ = [
    "CELL_SHAPE",
    "INITIAL_BUDGETS",
    "AcquisitionGrid",
    "AcquisitionGridError",
    "CellLattices",
    "build_acquisition_grid",
]
