"""Observed-only candidate features for deployable MVA policies."""

from __future__ import annotations

import math

import numpy as np

from .acquisition_grid import AcquisitionGrid
from .measurement_state import (
    MeasurementState,
    RefinementAction,
    fitting_actions,
    measurement_mask,
)


class CandidateFeatureError(ValueError):
    """Raised when candidate features cannot be built from issued state."""


def _readonly(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.size == 0 and shape == (0, 8):
        array = np.empty(shape, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise CandidateFeatureError("candidate feature array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(shape)
    output.setflags(write=False)
    return output


def _local_statistics(patch: np.ndarray) -> tuple[float, float]:
    scaled = np.asarray(patch, dtype=np.float64) / 255.0
    gradients: list[np.ndarray] = []
    if scaled.shape[0] > 1:
        gradients.append(np.abs(np.diff(scaled, axis=0)).reshape(-1))
    if scaled.shape[1] > 1:
        gradients.append(np.abs(np.diff(scaled, axis=1)).reshape(-1))
    gradient = float(np.mean(np.concatenate(gradients))) if gradients else 0.0
    return gradient, float(np.var(scaled))


def _nearest_measured_distance(
    cell_center: tuple[float, float], measured_coordinates: np.ndarray
) -> float:
    differences = measured_coordinates - np.asarray(cell_center, dtype=np.float64)
    return float(np.sqrt(np.min(np.sum(differences * differences, axis=1))) / math.sqrt(2.0))


def build_candidate_features(
    grid: AcquisitionGrid,
    state: MeasurementState,
    *,
    current_reconstruction: np.ndarray,
    checkpoint: float,
) -> tuple[tuple[RefinementAction, ...], np.ndarray]:
    """Return stable feasible actions and their eight observed-only features."""

    reconstruction = np.asarray(current_reconstruction)
    if reconstruction.shape != (*grid.native_shape, 3):
        raise CandidateFeatureError("current reconstruction shape changed")
    if not np.issubdtype(reconstruction.dtype, np.number):
        raise CandidateFeatureError("current reconstruction must be numeric")
    numeric = np.asarray(reconstruction, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or np.any((numeric < 0.0) | (numeric > 255.0)):
        raise CandidateFeatureError("current reconstruction values are invalid")

    actions = fitting_actions(grid, state, checkpoint)
    current_mask = measurement_mask(grid, state)
    measured = np.argwhere(current_mask).astype(np.float64, copy=False)
    height, width = grid.native_shape
    native_count = int(current_mask.size)
    measured[:, 0] /= height - 1
    measured[:, 1] /= width - 1

    rows: list[tuple[float, ...]] = []
    for action in actions:
        cell = grid.cells[action.cell_index]
        row_lower, row_upper = grid.row_boundaries[cell.row : cell.row + 2]
        column_lower, column_upper = grid.column_boundaries[cell.column : cell.column + 2]
        patch = numeric[
            row_lower : row_upper + 1,
            column_lower : column_upper + 1,
            :,
        ]
        gradient, variance = _local_statistics(patch)
        cell_mask = current_mask[
            row_lower : row_upper + 1,
            column_lower : column_upper + 1,
        ]
        rows_to_add = np.asarray(cell.rows[action.to_level], dtype=np.int64)
        columns_to_add = np.asarray(cell.columns[action.to_level], dtype=np.int64)
        added = int(
            np.count_nonzero(~current_mask[np.ix_(rows_to_add, columns_to_add)])
        )
        center = (
            (row_lower + row_upper) / (2.0 * (height - 1)),
            (column_lower + column_upper) / (2.0 * (width - 1)),
        )
        rows.append(
            (
                cell.row / 7.0,
                cell.column / 7.0,
                action.from_level / 2.0,
                added / native_count,
                float(np.mean(cell_mask)),
                gradient,
                variance,
                _nearest_measured_distance(center, measured),
            )
        )
    return actions, _readonly(rows, (len(actions), 8))


__all__ = [
    "CandidateFeatureError",
    "build_candidate_features",
]
