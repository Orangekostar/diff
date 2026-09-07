"""Observed-only C-scan Reader v2 and public task reports."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.interpolate import interpn

from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid, CellLattices

from .contracts import Split, Task


@dataclass(frozen=True, slots=True)
class BackgroundPrior:
    rgb: np.ndarray
    fit_specimen_keys: tuple[str, ...]
    fit_split: str

    def __post_init__(self) -> None:
        rgb = _readonly(self.rgb, dtype=np.uint8, shape=(3,))
        if (
            type(self.fit_specimen_keys) is not tuple
            or not self.fit_specimen_keys
            or any(not key for key in self.fit_specimen_keys)
            or len(set(self.fit_specimen_keys)) != len(self.fit_specimen_keys)
            or self.fit_split != Split.TRAIN.value
        ):
            raise ValueError("background prior must be fit on TRAIN specimens")
        object.__setattr__(self, "rgb", rgb)


@dataclass(frozen=True, slots=True)
class CellReadout:
    cell_index: int
    level: int
    measured_count: int
    measured_fraction: float
    observed_rgb_mean: tuple[float, float, float]
    observed_rgb_std: tuple[float, float, float]
    signal_mean: float
    signal_max: float
    support_count: int
    estimate_valid: bool
    candidate: bool
    row_spacing: float
    column_spacing: float
    boundary_unverified: bool


@dataclass(frozen=True, slots=True, eq=False)
class ReaderV2Result:
    measured_mask: np.ndarray
    measured_rgb: np.ndarray
    measured_scores: np.ndarray
    estimated_rgb: np.ndarray
    estimated_scores: np.ndarray
    estimate_valid: np.ndarray
    candidate_mask: np.ndarray
    cells: tuple[CellReadout, ...]

    def __post_init__(self) -> None:
        mask = _readonly(self.measured_mask, dtype=np.bool_)
        if mask.ndim != 2 or not mask.size:
            raise ValueError("reader mask is invalid")
        measured_rgb = _readonly(
            self.measured_rgb, dtype=np.uint8, shape=(*mask.shape, 3)
        )
        measured_scores = _readonly(
            self.measured_scores, dtype=np.float64, shape=mask.shape
        )
        estimated_rgb = _readonly(
            self.estimated_rgb, dtype=np.float64, shape=(*mask.shape, 3)
        )
        estimated_scores = _readonly(
            self.estimated_scores, dtype=np.float64, shape=mask.shape
        )
        valid = _readonly(self.estimate_valid, dtype=np.bool_, shape=mask.shape)
        candidate = _readonly(self.candidate_mask, dtype=np.bool_, shape=mask.shape)
        if (
            np.any(candidate & ~valid)
            or np.any(~np.isfinite(estimated_scores[valid]))
            or np.any(np.isfinite(estimated_scores[~valid]))
            or np.any(~np.isfinite(estimated_rgb[valid]))
            or np.any(np.isfinite(estimated_rgb[~valid]))
            or np.any(~np.isfinite(measured_scores[mask]))
            or np.any(measured_scores[~mask] != 0.0)
            or len(self.cells) != 64
        ):
            raise ValueError("reader result is inconsistent")
        object.__setattr__(self, "measured_mask", mask)
        object.__setattr__(self, "measured_rgb", measured_rgb)
        object.__setattr__(self, "measured_scores", measured_scores)
        object.__setattr__(self, "estimated_rgb", estimated_rgb)
        object.__setattr__(self, "estimated_scores", estimated_scores)
        object.__setattr__(self, "estimate_valid", valid)
        object.__setattr__(self, "candidate_mask", candidate)


@dataclass(frozen=True, slots=True, eq=False)
class VisibleTaskReadout:
    measured_mask: np.ndarray
    report: TaskReportV2

    def __post_init__(self) -> None:
        mask = _readonly(self.measured_mask, dtype=np.bool_)
        if mask.ndim != 2 or not mask.size or self.report.predicted_mask.shape != mask.shape:
            raise ValueError("visible task readout is invalid")
        object.__setattr__(self, "measured_mask", mask)


@dataclass(frozen=True, slots=True, eq=False)
class TaskReportV2:
    task: Task
    predicted_mask: np.ndarray
    support_positions: np.ndarray
    candidate_cells: tuple[int, ...]
    unverified_boundary_cells: tuple[int, ...]
    signal_strength: float
    reason_code: str

    def __post_init__(self) -> None:
        prediction = _readonly(self.predicted_mask, dtype=np.bool_)
        positions = _readonly(self.support_positions, dtype=np.int64)
        if (
            type(self.task) is not Task
            or prediction.ndim != 2
            or positions.ndim != 2
            or positions.shape[1:] != (2,)
            or any(not 0 <= cell < 64 for cell in self.candidate_cells)
            or any(not 0 <= cell < 64 for cell in self.unverified_boundary_cells)
            or not math.isfinite(float(self.signal_strength))
            or not 0.0 <= float(self.signal_strength) <= 1.0
            or not self.reason_code
        ):
            raise ValueError("task report v2 is invalid")
        object.__setattr__(self, "predicted_mask", prediction)
        object.__setattr__(self, "support_positions", positions)


def read_visible_evidence(
    *,
    grid: AcquisitionGrid,
    positions: np.ndarray,
    values: np.ndarray,
    cell_levels: tuple[int, ...],
    prior: BackgroundPrior,
    distance_threshold: float,
) -> ReaderV2Result:
    if (
        type(grid) is not AcquisitionGrid
        or type(cell_levels) is not tuple
        or len(cell_levels) != 64
        or any(type(level) is not int or level not in (-1, 0, 1, 2) for level in cell_levels)
        or type(prior) is not BackgroundPrior
        or isinstance(distance_threshold, bool)
        or not 0.0 < float(distance_threshold) < 1.0
    ):
        raise ValueError("Reader v2 request is invalid")
    coordinates = np.asarray(positions)
    samples = np.asarray(values)
    linear_positions = (
        coordinates[:, 0] * grid.native_shape[1] + coordinates[:, 1]
        if coordinates.dtype.kind in "iu"
        and coordinates.ndim == 2
        and coordinates.shape[1:] == (2,)
        else np.empty(0, dtype=np.int64)
    )
    if (
        coordinates.dtype.kind not in "iu"
        or coordinates.ndim != 2
        or coordinates.shape[1:] != (2,)
        or samples.dtype != np.uint8
        or samples.shape != (len(coordinates), 3)
        or (
            coordinates.size
            and (
                np.any(coordinates < 0)
                or np.any(coordinates[:, 0] >= grid.native_shape[0])
                or np.any(coordinates[:, 1] >= grid.native_shape[1])
                or (
                    len(linear_positions) > 1
                    and (
                        np.any(np.diff(linear_positions) <= 0)
                        and len(np.unique(linear_positions)) != len(coordinates)
                    )
                )
            )
        )
    ):
        raise ValueError("Reader v2 visible observations are invalid")
    shape = grid.native_shape
    measured_mask = np.zeros(shape, dtype=np.bool_)
    measured_rgb = np.zeros((*shape, 3), dtype=np.uint8)
    measured_scores = np.zeros(shape, dtype=np.float64)
    if len(coordinates):
        rows, columns = coordinates.T
        measured_mask[rows, columns] = True
        measured_rgb[rows, columns] = samples
        measured_scores[rows, columns] = _distance(samples, prior.rgb)
    estimated_rgb = np.full((*shape, 3), np.nan, dtype=np.float64)
    estimate_valid = np.zeros(shape, dtype=np.bool_)
    if len(coordinates):
        estimated_rgb[rows, columns] = samples
        estimate_valid[rows, columns] = True
    for cell, level in zip(grid.cells, cell_levels, strict=True):
        if level >= 0:
            _interpolate_cell(
                cell,
                grid,
                level=level,
                measured_mask=measured_mask,
                measured_rgb=measured_rgb,
                estimated_rgb=estimated_rgb,
                estimate_valid=estimate_valid,
            )
    estimated_scores = np.full(shape, np.nan, dtype=np.float64)
    estimated_scores[estimate_valid] = _distance(
        estimated_rgb[estimate_valid], prior.rgb
    )
    if len(coordinates):
        estimated_rgb[rows, columns] = samples
        estimated_scores[rows, columns] = measured_scores[rows, columns]
    candidate_mask = estimate_valid & (
        estimated_scores >= float(distance_threshold)
    )
    cells = tuple(
        _summarize_cell(
            grid,
            cell,
            level=level,
            measured_mask=measured_mask,
            measured_rgb=measured_rgb,
            measured_scores=measured_scores,
            estimate_valid=estimate_valid,
            candidate_mask=candidate_mask,
        )
        for cell, level in zip(grid.cells, cell_levels, strict=True)
    )
    return ReaderV2Result(
        measured_mask=measured_mask,
        measured_rgb=measured_rgb,
        measured_scores=measured_scores,
        estimated_rgb=estimated_rgb,
        estimated_scores=estimated_scores,
        estimate_valid=estimate_valid,
        candidate_mask=candidate_mask,
        cells=cells,
    )


def build_task_report_v2(readout: ReaderV2Result, *, task: Task) -> TaskReportV2:
    if type(readout) is not ReaderV2Result or type(task) is not Task:
        raise TypeError("typed Reader v2 result and task are required")
    prediction = np.array(readout.candidate_mask, copy=True)
    if task is Task.LOCATE and np.any(prediction):
        labels, count = ndimage.label(prediction)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        prediction = labels == int(np.argmax(sizes))
    support = np.argwhere(prediction & readout.measured_mask).astype(
        np.int64, copy=False
    )
    candidate_cells = tuple(cell.cell_index for cell in readout.cells if cell.candidate)
    unverified = tuple(
        cell.cell_index for cell in readout.cells if cell.boundary_unverified
    )
    signal = (
        float(np.mean(readout.measured_scores[support[:, 0], support[:, 1]]))
        if len(support)
        else 0.0
    )
    return TaskReportV2(
        task=task,
        predicted_mask=prediction,
        support_positions=support,
        candidate_cells=candidate_cells,
        unverified_boundary_cells=unverified,
        signal_strength=float(np.clip(signal, 0.0, 1.0)),
        reason_code="VISIBLE_CANDIDATE" if candidate_cells else "NO_VISIBLE_CANDIDATE",
    )


def read_visible_task_report(
    *,
    grid: AcquisitionGrid,
    positions: np.ndarray,
    values: np.ndarray,
    cell_levels: tuple[int, ...],
    prior: BackgroundPrior,
    distance_threshold: float,
    task: Task,
) -> VisibleTaskReadout:
    """Build the exact public task report without actor-only dense features."""
    if (
        type(grid) is not AcquisitionGrid
        or type(cell_levels) is not tuple
        or len(cell_levels) != 64
        or any(
            type(level) is not int or level not in (-1, 0, 1, 2)
            for level in cell_levels
        )
        or type(prior) is not BackgroundPrior
        or type(task) is not Task
        or isinstance(distance_threshold, bool)
        or not 0.0 < float(distance_threshold) < 1.0
    ):
        raise ValueError("visible task report request is invalid")
    coordinates = np.asarray(positions)
    samples = np.asarray(values)
    if (
        coordinates.dtype.kind not in "iu"
        or coordinates.ndim != 2
        or coordinates.shape[1:] != (2,)
        or samples.dtype != np.uint8
        or samples.shape != (len(coordinates), 3)
        or (
            coordinates.size
            and (
                np.any(coordinates < 0)
                or np.any(coordinates[:, 0] >= grid.native_shape[0])
                or np.any(coordinates[:, 1] >= grid.native_shape[1])
            )
        )
    ):
        raise ValueError("visible task report observations are invalid")
    linear = (
        coordinates[:, 0] * grid.native_shape[1] + coordinates[:, 1]
    )
    if (
        len(linear) > 1
        and np.any(np.diff(linear) <= 0)
        and len(np.unique(linear)) != len(coordinates)
    ):
        raise ValueError("visible task report observations are invalid")
    measured_mask = np.zeros(grid.native_shape, dtype=np.bool_)
    measured_rgb = np.zeros((*grid.native_shape, 3), dtype=np.uint8)
    measured_scores = _distance(samples, prior.rgb)
    if len(coordinates):
        rows, columns = coordinates.T
        measured_mask[rows, columns] = True
        measured_rgb[rows, columns] = samples
    estimate_valid = np.array(measured_mask, copy=True)
    candidate_mask = np.zeros(grid.native_shape, dtype=np.bool_)
    if len(coordinates):
        candidate_mask[rows, columns] = measured_scores >= float(distance_threshold)
    for cell, level in zip(grid.cells, cell_levels, strict=True):
        if 0 <= level < 2:
            _interpolate_cell_candidates(
                cell,
                grid,
                measured_mask=measured_mask,
                measured_rgb=measured_rgb,
                estimate_valid=estimate_valid,
                candidate_mask=candidate_mask,
                prior=prior,
                distance_threshold=float(distance_threshold),
            )
    if len(coordinates):
        candidate_mask[rows, columns] = measured_scores >= float(distance_threshold)
    candidate_cells = []
    unverified = []
    for cell, level in zip(grid.cells, cell_levels, strict=True):
        row_slice, column_slice = owned_cell_slices(grid, cell)
        is_candidate = bool(
            level >= 0 and np.any(candidate_mask[row_slice, column_slice])
        )
        if is_candidate:
            candidate_cells.append(cell.index)
            if np.any(~estimate_valid[row_slice, column_slice]):
                unverified.append(cell.index)
    prediction = np.array(candidate_mask, copy=True)
    if task is Task.LOCATE and np.any(prediction):
        labels, count = ndimage.label(prediction)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        prediction = labels == int(np.argmax(sizes))
    support = np.argwhere(prediction & measured_mask).astype(np.int64, copy=False)
    if len(support):
        support_linear = support[:, 0] * grid.native_shape[1] + support[:, 1]
        order = np.argsort(linear, kind="stable")
        sorted_linear = linear[order]
        score_indices = np.searchsorted(sorted_linear, support_linear)
        signal = float(np.mean(measured_scores[order][score_indices]))
    else:
        signal = 0.0
    report = TaskReportV2(
        task=task,
        predicted_mask=prediction,
        support_positions=support,
        candidate_cells=tuple(candidate_cells),
        unverified_boundary_cells=tuple(unverified),
        signal_strength=float(np.clip(signal, 0.0, 1.0)),
        reason_code=(
            "VISIBLE_CANDIDATE" if candidate_cells else "NO_VISIBLE_CANDIDATE"
        ),
    )
    return VisibleTaskReadout(measured_mask=measured_mask, report=report)


def owned_cell_slices(
    grid: AcquisitionGrid, cell: CellLattices
) -> tuple[slice, slice]:
    row_stop = grid.row_boundaries[cell.row + 1] + (cell.row == 7)
    column_stop = grid.column_boundaries[cell.column + 1] + (cell.column == 7)
    return (
        slice(grid.row_boundaries[cell.row], row_stop),
        slice(grid.column_boundaries[cell.column], column_stop),
    )


def _interpolate_cell(
    cell: CellLattices,
    grid: AcquisitionGrid,
    *,
    level: int,
    measured_mask: np.ndarray,
    measured_rgb: np.ndarray,
    estimated_rgb: np.ndarray,
    estimate_valid: np.ndarray,
) -> None:
    if level == 2:
        return
    row_lower, row_upper = grid.row_boundaries[cell.row : cell.row + 2]
    column_lower, column_upper = grid.column_boundaries[cell.column : cell.column + 2]
    local_support = measured_mask[
        row_lower : row_upper + 1, column_lower : column_upper + 1
    ]
    support_rows = (
        np.flatnonzero(np.any(local_support, axis=1)).astype(np.int64) + row_lower
    )
    support_columns = (
        np.flatnonzero(np.any(local_support, axis=0)).astype(np.int64)
        + column_lower
    )
    if not len(support_rows) or not len(support_columns):
        return
    expected = len(support_rows) * len(support_columns)
    if np.count_nonzero(local_support) != expected:
        return
    row_slice, column_slice = owned_cell_slices(grid, cell)
    target_rows = np.arange(row_slice.start, row_slice.stop, dtype=np.int64)
    target_columns = np.arange(column_slice.start, column_slice.stop, dtype=np.int64)
    target_rows = target_rows[
        (target_rows >= support_rows[0]) & (target_rows <= support_rows[-1])
    ]
    target_columns = target_columns[
        (target_columns >= support_columns[0])
        & (target_columns <= support_columns[-1])
    ]
    if not len(target_rows) or not len(target_columns):
        return
    lattice = measured_rgb[np.ix_(support_rows, support_columns)].astype(np.float64)
    if len(support_rows) >= 2 and len(support_columns) >= 2:
        target_row_grid, target_column_grid = np.meshgrid(
            target_rows, target_columns, indexing="ij"
        )
        points = np.column_stack(
            (target_row_grid.ravel(), target_column_grid.ravel())
        )
        result = interpn(
            (support_rows, support_columns), lattice, points, method="linear"
        ).reshape(len(target_rows), len(target_columns), 3)
        estimated_rgb[np.ix_(target_rows, target_columns)] = result
        estimate_valid[np.ix_(target_rows, target_columns)] = True
    elif len(support_rows) == 1 and len(support_columns) >= 2:
        row = int(support_rows[0])
        for channel in range(3):
            estimated_rgb[row, target_columns, channel] = np.interp(
                target_columns, support_columns, lattice[0, :, channel]
            )
        estimate_valid[row, target_columns] = True
    elif len(support_rows) >= 2 and len(support_columns) == 1:
        column = int(support_columns[0])
        for channel in range(3):
            estimated_rgb[target_rows, column, channel] = np.interp(
                target_rows, support_rows, lattice[:, 0, channel]
            )
        estimate_valid[target_rows, column] = True


def _interpolate_cell_candidates(
    cell: CellLattices,
    grid: AcquisitionGrid,
    *,
    measured_mask: np.ndarray,
    measured_rgb: np.ndarray,
    estimate_valid: np.ndarray,
    candidate_mask: np.ndarray,
    prior: BackgroundPrior,
    distance_threshold: float,
) -> None:
    row_lower, row_upper = grid.row_boundaries[cell.row : cell.row + 2]
    column_lower, column_upper = grid.column_boundaries[cell.column : cell.column + 2]
    local_support = measured_mask[
        row_lower : row_upper + 1, column_lower : column_upper + 1
    ]
    support_rows = (
        np.flatnonzero(np.any(local_support, axis=1)).astype(np.int64) + row_lower
    )
    support_columns = (
        np.flatnonzero(np.any(local_support, axis=0)).astype(np.int64)
        + column_lower
    )
    if not len(support_rows) or not len(support_columns):
        return
    if np.count_nonzero(local_support) != len(support_rows) * len(support_columns):
        return
    row_slice, column_slice = owned_cell_slices(grid, cell)
    target_rows = np.arange(row_slice.start, row_slice.stop, dtype=np.int64)
    target_columns = np.arange(column_slice.start, column_slice.stop, dtype=np.int64)
    target_rows = target_rows[
        (target_rows >= support_rows[0]) & (target_rows <= support_rows[-1])
    ]
    target_columns = target_columns[
        (target_columns >= support_columns[0])
        & (target_columns <= support_columns[-1])
    ]
    if not len(target_rows) or not len(target_columns):
        return
    lattice = measured_rgb[np.ix_(support_rows, support_columns)].astype(np.float64)
    result: np.ndarray | None = None
    if len(support_rows) >= 2 and len(support_columns) >= 2:
        target_row_grid, target_column_grid = np.meshgrid(
            target_rows, target_columns, indexing="ij"
        )
        points = np.column_stack(
            (target_row_grid.ravel(), target_column_grid.ravel())
        )
        result = interpn(
            (support_rows, support_columns), lattice, points, method="linear"
        ).reshape(len(target_rows), len(target_columns), 3)
    elif len(support_rows) == 1 and len(support_columns) >= 2:
        result = np.column_stack(
            [
                np.interp(target_columns, support_columns, lattice[0, :, channel])
                for channel in range(3)
            ]
        )[None, :, :]
    elif len(support_rows) >= 2 and len(support_columns) == 1:
        result = np.column_stack(
            [
                np.interp(target_rows, support_rows, lattice[:, 0, channel])
                for channel in range(3)
            ]
        )[:, None, :]
    if result is None:
        return
    estimate_valid[np.ix_(target_rows, target_columns)] = True
    candidate_mask[np.ix_(target_rows, target_columns)] = (
        _distance(result, prior.rgb) >= distance_threshold
    )


def _summarize_cell(
    grid: AcquisitionGrid,
    cell: CellLattices,
    *,
    level: int,
    measured_mask: np.ndarray,
    measured_rgb: np.ndarray,
    measured_scores: np.ndarray,
    estimate_valid: np.ndarray,
    candidate_mask: np.ndarray,
) -> CellReadout:
    row_slice, column_slice = owned_cell_slices(grid, cell)
    observed = measured_mask[row_slice, column_slice]
    valid = estimate_valid[row_slice, column_slice]
    candidate = candidate_mask[row_slice, column_slice]
    observed_values = measured_rgb[row_slice, column_slice][observed]
    scores = measured_scores[row_slice, column_slice][observed]
    count = int(np.count_nonzero(observed))
    size = int(observed.size)
    mean = (
        tuple(float(value) for value in observed_values.mean(axis=0))
        if count
        else (0.0, 0.0, 0.0)
    )
    std = (
        tuple(float(value) for value in observed_values.std(axis=0))
        if count
        else (0.0, 0.0, 0.0)
    )
    support_rows = np.flatnonzero(np.any(observed, axis=1))
    support_columns = np.flatnonzero(np.any(observed, axis=0))
    row_spacing = _maximum_spacing(support_rows, max(observed.shape[0] - 1, 1))
    column_spacing = _maximum_spacing(
        support_columns, max(observed.shape[1] - 1, 1)
    )
    is_candidate = bool(level >= 0 and np.any(candidate))
    return CellReadout(
        cell_index=cell.index,
        level=level,
        measured_count=count,
        measured_fraction=float(count / size),
        observed_rgb_mean=mean,
        observed_rgb_std=std,
        signal_mean=float(scores.mean()) if count else 0.0,
        signal_max=float(scores.max()) if count else 0.0,
        support_count=count,
        estimate_valid=bool(level >= 0 and np.any(valid)),
        candidate=is_candidate,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
        boundary_unverified=bool(is_candidate and np.any(~valid)),
    )


def _maximum_spacing(values: np.ndarray, denominator: int) -> float:
    if len(values) < 2:
        return 1.0 if len(values) else 0.0
    return float(np.max(np.diff(values)) / denominator)


def _distance(values: np.ndarray, background: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        np.asarray(values, dtype=np.float64) - np.asarray(background, dtype=np.float64),
        axis=-1,
    ) / math.sqrt(3.0 * 255.0**2)


def _readonly(
    value: object, *, dtype: object, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError("array shape is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    output.setflags(write=False)
    return output


__all__ = [
    "BackgroundPrior",
    "CellReadout",
    "ReaderV2Result",
    "TaskReportV2",
    "VisibleTaskReadout",
    "build_task_report_v2",
    "owned_cell_slices",
    "read_visible_evidence",
    "read_visible_task_report",
]
