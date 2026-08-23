"""Deterministic mixed-cell reconstruction with exact measurement restoration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RectBivariateSpline

from cmc_bbdm.cpb_sparse_scan.sampling import (
    INTERPOLATIONS,
    reconstruct_sparse_rgb,
)

from .acquisition_grid import AcquisitionGrid
from .measurement_state import (
    MeasurementState,
    RefinementAction,
    apply_action,
    budget_record,
    measurement_mask,
)


class MVAInterpolationError(ValueError):
    """Raised when image or interpolation state violates MVA semantics."""


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class ReconstructionResult:
    image: np.ndarray
    interpolation: str
    measured_count: int
    native_count: int
    effective_budget: float
    measured_values_exact: bool
    p5_equivalent: bool
    input_sha256: str
    output_sha256: str

    def __eq__(self, other: object) -> bool:
        if type(other) is not ReconstructionResult:
            return NotImplemented
        return (
            np.array_equal(self.image, other.image)
            and self.interpolation == other.interpolation
            and self.measured_count == other.measured_count
            and self.native_count == other.native_count
            and self.effective_budget == other.effective_budget
            and self.measured_values_exact == other.measured_values_exact
            and self.p5_equivalent == other.p5_equivalent
            and self.input_sha256 == other.input_sha256
            and self.output_sha256 == other.output_sha256
        )


@dataclass(slots=True, eq=False)
class RefinementPatchCache:
    """Per-specimen interpolation patches bound to one source object and grid."""

    image: np.ndarray
    grid: AcquisitionGrid
    patches: dict[tuple[int, int, str], np.ndarray] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.image, np.ndarray)
            or self.image.dtype != np.uint8
            or self.image.shape != (*self.grid.native_shape, 3)
        ):
            raise MVAInterpolationError("patch cache source does not match the grid")


def _snapshot(image: object, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise MVAInterpolationError("image must be an RGB uint8 array")
    if image.shape != (*shape, 3) or image.dtype != np.uint8:
        raise MVAInterpolationError("image shape or dtype does not match the grid")
    payload = np.ascontiguousarray(image).tobytes(order="C")
    return np.frombuffer(payload, dtype=np.uint8).reshape(image.shape)


def _nearest_indices(samples: np.ndarray, query: np.ndarray) -> np.ndarray:
    distance = np.abs(query[:, None] - samples[None, :])
    return np.argmin(distance, axis=1)


def _interpolate_rectilinear(
    sampled: np.ndarray,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    target_rows: np.ndarray,
    target_columns: np.ndarray,
    method: str,
) -> np.ndarray:
    row_axis = np.asarray(rows, dtype=np.float64)
    column_axis = np.asarray(columns, dtype=np.float64)
    query_rows = np.asarray(target_rows, dtype=np.float64)
    query_columns = np.asarray(target_columns, dtype=np.float64)
    if method == "nearest":
        row_index = _nearest_indices(row_axis, query_rows)
        column_index = _nearest_indices(column_axis, query_columns)
        values = sampled[np.ix_(row_index, column_index)]
    elif method == "bilinear":
        horizontal = np.empty((len(rows), len(target_columns), 3), dtype=np.float64)
        for row_index in range(len(rows)):
            for channel in range(3):
                horizontal[row_index, :, channel] = np.interp(
                    query_columns,
                    column_axis,
                    sampled[row_index, :, channel],
                )
        values = np.empty((len(target_rows), len(target_columns), 3), dtype=np.float64)
        for column_index in range(len(target_columns)):
            for channel in range(3):
                values[:, column_index, channel] = np.interp(
                    query_rows,
                    row_axis,
                    horizontal[:, column_index, channel],
                )
    elif method == "bicubic":
        values = np.empty((len(target_rows), len(target_columns), 3), dtype=np.float64)
        row_degree = min(3, len(rows) - 1)
        column_degree = min(3, len(columns) - 1)
        for channel in range(3):
            spline = RectBivariateSpline(
                row_axis,
                column_axis,
                sampled[:, :, channel].astype(np.float64),
                kx=row_degree,
                ky=column_degree,
                s=0.0,
            )
            values[:, :, channel] = spline(query_rows, query_columns, grid=True)
    else:  # pragma: no cover - guarded by the public validator
        raise MVAInterpolationError("interpolation is not registered")
    return np.rint(values).clip(0, 255).astype(np.uint8)


def reconstruct_measurement_state(
    image: np.ndarray,
    grid: AcquisitionGrid,
    state: MeasurementState,
    *,
    interpolation: str,
    specimen_id: str,
    dataset_id: str,
) -> ReconstructionResult:
    """Reconstruct a state and restore all observed RGB triplets exactly."""

    if type(grid) is not AcquisitionGrid or type(state) is not MeasurementState:
        raise MVAInterpolationError("issued grid and state are required")
    if interpolation not in INTERPOLATIONS:
        raise MVAInterpolationError("interpolation is not registered")
    if type(specimen_id) is not str or not specimen_id:
        raise MVAInterpolationError("specimen ID is required")
    if type(dataset_id) is not str or not dataset_id:
        raise MVAInterpolationError("dataset ID is required")
    source = _snapshot(image, grid.native_shape)
    mask = measurement_mask(grid, state)
    p5_equivalent = state.levels == (1,) * 64

    if p5_equivalent:
        reconstructed, _record = reconstruct_sparse_rgb(
            source,
            specimen_id=specimen_id,
            dataset_id=dataset_id,
            density=0.25,
            interpolation=interpolation,
        )
    elif state.levels == (2,) * 64:
        reconstructed = source.copy()
    elif state.levels == (0,) * 64:
        sampled = np.ascontiguousarray(
            source[np.ix_(grid.level0_rows, grid.level0_columns)]
        )
        reconstructed = _interpolate_rectilinear(
            sampled,
            grid.level0_rows,
            grid.level0_columns,
            np.arange(grid.native_shape[0], dtype=np.int64),
            np.arange(grid.native_shape[1], dtype=np.int64),
            interpolation,
        )
        reconstructed[mask] = source[mask]
    else:
        reconstructed = np.empty_like(source)
        for cell, level in zip(grid.cells, state.levels, strict=True):
            row_start, row_stop = cell.rows[2][0], cell.rows[2][-1]
            column_start, column_stop = cell.columns[2][0], cell.columns[2][-1]
            query_rows = np.arange(row_start, row_stop + 1, dtype=np.int64)
            query_columns = np.arange(column_start, column_stop + 1, dtype=np.int64)
            rows = cell.rows[level]
            columns = cell.columns[level]
            sampled = np.ascontiguousarray(source[np.ix_(rows, columns)])
            reconstructed[row_start : row_stop + 1, column_start : column_stop + 1] = (
                _interpolate_rectilinear(
                    sampled,
                    rows,
                    columns,
                    query_rows,
                    query_columns,
                    interpolation,
                )
            )
        reconstructed[mask] = source[mask]

    exact = bool(np.array_equal(reconstructed[mask], source[mask]))
    if (
        not exact
        or reconstructed.shape != source.shape
        or reconstructed.dtype != np.uint8
    ):
        raise MVAInterpolationError("reconstruction invariants failed")
    record = budget_record(grid, state)
    payload = np.ascontiguousarray(reconstructed).tobytes(order="C")
    output = np.frombuffer(payload, dtype=np.uint8).reshape(reconstructed.shape)
    output.setflags(write=False)
    return ReconstructionResult(
        image=output,
        interpolation=interpolation,
        measured_count=record.measured_count,
        native_count=record.native_count,
        effective_budget=record.effective_budget,
        measured_values_exact=exact,
        p5_equivalent=p5_equivalent,
        input_sha256=_sha256(source),
        output_sha256=_sha256(output),
    )


def refine_reconstruction(
    image: np.ndarray,
    grid: AcquisitionGrid,
    state: MeasurementState,
    current_reconstruction: np.ndarray,
    action: RefinementAction,
    *,
    interpolation: str,
    current_mask: np.ndarray | None = None,
    patch_cache: RefinementPatchCache | None = None,
) -> np.ndarray:
    """Increment one cell with byte-identical full-reconstruction semantics."""

    source = _snapshot(image, grid.native_shape)
    current = _snapshot(current_reconstruction, grid.native_shape)
    if interpolation not in INTERPOLATIONS:
        raise MVAInterpolationError("interpolation is not registered")
    if patch_cache is not None and (
        type(patch_cache) is not RefinementPatchCache
        or patch_cache.image is not image
        or patch_cache.grid is not grid
    ):
        raise MVAInterpolationError("patch cache source or grid changed")
    candidate_state = apply_action(grid, state, action)
    if current_mask is None:
        observed = measurement_mask(grid, state)
    elif (
        not isinstance(current_mask, np.ndarray)
        or current_mask.dtype != np.bool_
        or current_mask.shape != grid.native_shape
    ):
        raise MVAInterpolationError("current mask does not match the grid")
    else:
        observed = current_mask
    if candidate_state.levels == (1,) * 64:
        candidate, _record = reconstruct_sparse_rgb(
            source,
            specimen_id="mva-incremental",
            dataset_id="mva-incremental",
            density=0.25,
            interpolation=interpolation,
        )
    elif candidate_state.levels == (2,) * 64:
        candidate = source.copy()
    else:
        candidate = current.copy()
        cell = grid.cells[action.cell_index]
        level = action.to_level
        row_start, row_stop = cell.rows[2][0], cell.rows[2][-1]
        column_start, column_stop = cell.columns[2][0], cell.columns[2][-1]
        query_rows = np.arange(row_start, row_stop + 1, dtype=np.int64)
        query_columns = np.arange(column_start, column_stop + 1, dtype=np.int64)
        rows = cell.rows[level]
        columns = cell.columns[level]
        key = (action.cell_index, level, interpolation)
        patch = None if patch_cache is None else patch_cache.patches.get(key)
        if patch is None:
            sampled = np.ascontiguousarray(source[np.ix_(rows, columns)])
            patch = _interpolate_rectilinear(
                sampled,
                rows,
                columns,
                query_rows,
                query_columns,
                interpolation,
            )
            patch.setflags(write=False)
            if patch_cache is not None:
                patch_cache.patches[key] = patch
        owned_row_stop = row_stop + 1 if cell.row == 7 else row_stop
        owned_column_stop = column_stop + 1 if cell.column == 7 else column_stop
        patch_row_stop = patch.shape[0] if cell.row == 7 else patch.shape[0] - 1
        patch_column_stop = patch.shape[1] if cell.column == 7 else patch.shape[1] - 1
        candidate[row_start:owned_row_stop, column_start:owned_column_stop] = patch[
            :patch_row_stop, :patch_column_stop
        ]
        owned_observed = observed[
            row_start:owned_row_stop, column_start:owned_column_stop
        ]
        candidate_view = candidate[
            row_start:owned_row_stop, column_start:owned_column_stop
        ]
        source_view = source[row_start:owned_row_stop, column_start:owned_column_stop]
        candidate_view[owned_observed] = source_view[owned_observed]
        candidate[np.ix_(rows, columns)] = source[np.ix_(rows, columns)]
    payload = np.ascontiguousarray(candidate).tobytes(order="C")
    output = np.frombuffer(payload, dtype=np.uint8).reshape(candidate.shape)
    output.setflags(write=False)
    return output


__all__ = [
    "MVAInterpolationError",
    "ReconstructionResult",
    "RefinementPatchCache",
    "reconstruct_measurement_state",
    "refine_reconstruction",
]
