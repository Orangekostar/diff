"""Canonical 8x8 acquisition-action geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .registration import SurfaceToCscanTransform


def validate_action_id(cell_id: int) -> int:
    if type(cell_id) is not int or not 0 <= cell_id <= 63:
        raise ValueError("action cell_id must be an integer in 0..63")
    return cell_id


@dataclass(frozen=True, slots=True)
class GridCell:
    cell_id: int
    row: int
    column: int
    x0: float
    y0: float
    x1: float
    y1: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "cell_id": self.cell_id,
            "row": self.row,
            "column": self.column,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }


@dataclass(frozen=True, slots=True)
class Grid8x8:
    width: float
    height: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.width))
            or not math.isfinite(float(self.height))
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("grid extent must be positive and finite")

    def cell_box(self, cell_id: int) -> tuple[float, float, float, float]:
        legal = validate_action_id(cell_id)
        row, column = divmod(legal, 8)
        cell_width = self.width / 8.0
        cell_height = self.height / 8.0
        return (
            column * cell_width,
            row * cell_height,
            (column + 1) * cell_width,
            (row + 1) * cell_height,
        )

    def cells(self) -> tuple[GridCell, ...]:
        records: list[GridCell] = []
        for cell_id in range(64):
            row, column = divmod(cell_id, 8)
            x0, y0, x1, y1 = self.cell_box(cell_id)
            records.append(GridCell(cell_id, row, column, x0, y0, x1, y1))
        return tuple(records)

    def point_to_cell(self, point: tuple[float, float]) -> int:
        if len(point) != 2:
            raise ValueError("grid point must have two coordinates")
        x, y = (float(value) for value in point)
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or x < 0.0
            or y < 0.0
            or x > self.width
            or y > self.height
        ):
            raise ValueError("grid point is outside the legal field")
        column = min(7, int(x / (self.width / 8.0)))
        row = min(7, int(y / (self.height / 8.0)))
        return row * 8 + column

    def cells_for_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[int, ...]:
        if len(box) != 4:
            raise ValueError("grid box must contain x0,y0,x1,y1")
        x0, y0, x1, y1 = (float(value) for value in box)
        if (
            any(not math.isfinite(value) for value in (x0, y0, x1, y1))
            or x0 < 0.0
            or y0 < 0.0
            or x1 > self.width
            or y1 > self.height
        ):
            raise ValueError("grid box is outside the legal field")
        if x0 > x1 or y0 > y1:
            raise ValueError("grid box bounds are not canonical")
        if x0 == x1 or y0 == y1:
            return ()
        result = []
        for cell in self.cells():
            if (
                max(x0, cell.x0) < min(x1, cell.x1)
                and max(y0, cell.y0) < min(y1, cell.y1)
            ):
                result.append(cell.cell_id)
        return tuple(result)

    def render_records(self) -> tuple[dict[str, int | float], ...]:
        return tuple(cell.as_dict() for cell in self.cells())


def surface_box_to_cscan_cells(
    transform: SurfaceToCscanTransform,
    surface_box: tuple[float, float, float, float],
) -> tuple[int, ...]:
    if type(transform) is not SurfaceToCscanTransform:
        raise ValueError("a resolved surface-to-C-scan transform is required")
    mapped = transform.forward_box(surface_box)
    grid = Grid8x8(
        width=float(transform.destination.width_px - 1),
        height=float(transform.destination.height_px - 1),
    )
    return grid.cells_for_box(mapped)


def render_surface_grid(
    transform: SurfaceToCscanTransform,
) -> tuple[dict[str, object], ...]:
    if type(transform) is not SurfaceToCscanTransform:
        raise ValueError("a resolved surface-to-C-scan transform is required")
    grid = Grid8x8(
        width=float(transform.destination.width_px - 1),
        height=float(transform.destination.height_px - 1),
    )
    return tuple(
        {
            "cell_id": cell.cell_id,
            "row": cell.row,
            "column": cell.column,
            "surface_box": transform.inverse_box((cell.x0, cell.y0, cell.x1, cell.y1)),
        }
        for cell in grid.cells()
    )


__all__ = [
    "Grid8x8",
    "GridCell",
    "render_surface_grid",
    "surface_box_to_cscan_cells",
    "validate_action_id",
]
