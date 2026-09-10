"""Exact native-raster 8x8 full-cell acquisition environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NativeCell:
    index: int
    row: int
    column: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    pixel_count: int
    cost: float


@dataclass(frozen=True, slots=True)
class NativeCellGrid:
    native_shape: tuple[int, int]
    cells: tuple[NativeCell, ...]

    @classmethod
    def from_shape(cls, shape: tuple[int, int]) -> NativeCellGrid:
        if (
            type(shape) is not tuple
            or len(shape) != 2
            or any(type(value) is not int or value < 8 for value in shape)
        ):
            raise ValueError("native shape must contain two dimensions >= 8")
        height, width = shape
        row_edges = np.rint(np.linspace(0, height, 9)).astype(np.int64)
        col_edges = np.rint(np.linspace(0, width, 9)).astype(np.int64)
        if np.any(np.diff(row_edges) <= 0) or np.any(np.diff(col_edges) <= 0):
            raise ValueError("native shape cannot form an 8x8 grid")
        total = height * width
        cells: list[NativeCell] = []
        for row in range(8):
            for column in range(8):
                row_start, row_stop = int(row_edges[row]), int(row_edges[row + 1])
                col_start, col_stop = int(col_edges[column]), int(col_edges[column + 1])
                count = (row_stop - row_start) * (col_stop - col_start)
                cells.append(
                    NativeCell(
                        index=row * 8 + column,
                        row=row,
                        column=column,
                        row_start=row_start,
                        row_stop=row_stop,
                        col_start=col_start,
                        col_stop=col_stop,
                        pixel_count=count,
                        cost=count / total,
                    )
                )
        return cls(native_shape=shape, cells=tuple(cells))

    def measured_cost(self, measured_cells: set[int] | frozenset[int]) -> float:
        self._validate_indices(measured_cells)
        return float(sum(self.cells[index].cost for index in measured_cells))

    def legal_mask(
        self,
        measured_cells: set[int] | frozenset[int],
        *,
        endpoint_budget: float,
        tolerance: float = 1e-12,
    ) -> np.ndarray:
        self._validate_indices(measured_cells)
        current = self.measured_cost(measured_cells)
        return np.asarray(
            [
                cell.index not in measured_cells
                and current + cell.cost <= endpoint_budget + tolerance
                for cell in self.cells
            ],
            dtype=bool,
        )

    @staticmethod
    def _validate_indices(indices: set[int] | frozenset[int]) -> None:
        if not isinstance(indices, (set, frozenset)) or any(
            type(index) is not int or not 0 <= index < 64 for index in indices
        ):
            raise ValueError("measured cells must be unique indices in [0, 63]")


def reveal_cells(
    full_scan: np.ndarray,
    grid: NativeCellGrid,
    *,
    measured_cells: set[int] | frozenset[int],
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(full_scan)
    if image.ndim not in {2, 3} or image.shape[:2] != grid.native_shape:
        raise ValueError("full scan and native grid shapes differ")
    grid._validate_indices(measured_cells)
    pixel_mask = np.zeros(grid.native_shape, dtype=bool)
    for index in measured_cells:
        cell = grid.cells[index]
        pixel_mask[cell.row_start : cell.row_stop, cell.col_start : cell.col_stop] = True
    visible = np.zeros_like(image)
    visible[pixel_mask] = image[pixel_mask]
    return visible, pixel_mask


__all__ = ["NativeCell", "NativeCellGrid", "reveal_cells"]
