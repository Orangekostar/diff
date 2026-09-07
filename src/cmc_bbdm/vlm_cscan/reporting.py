"""Common measured-evidence reports and visible stopping rules."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .contracts import BenchmarkTask, EvidenceMap, TaskReport


class PublicStopTracker:
    def __init__(self, task: BenchmarkTask) -> None:
        if type(task) is not BenchmarkTask:
            raise TypeError("benchmark task is required")
        self.task = task
        self._previous_mask: np.ndarray | None = None
        self._measured_count = -1
        self._stable_updates = 0
        self._final_report: TaskReport | None = None

    def update(
        self,
        report: TaskReport,
        *,
        measured_count: int,
        cell_levels: tuple[int, ...],
    ) -> TaskReport:
        if (
            type(report) is not TaskReport
            or report.task is not self.task
            or type(measured_count) is not int
            or measured_count < 0
            or type(cell_levels) is not tuple
            or len(cell_levels) != 64
            or any(level not in (-1, 0, 1, 2) for level in cell_levels)
        ):
            raise ValueError("public stop update is invalid")
        if self._final_report is not None:
            return self._final_report
        has_new_measurement = measured_count > self._measured_count
        if has_new_measurement:
            if self._previous_mask is not None and np.array_equal(
                report.predicted_mask, self._previous_mask
            ):
                self._stable_updates += 1
            else:
                self._stable_updates = 0
            self._previous_mask = np.array(report.predicted_mask, copy=True)
            self._measured_count = measured_count
        allowed = self._base_allowed(report, cell_levels) and self._stable_updates >= 2
        updated = TaskReport(
            task=report.task,
            predicted_mask=report.predicted_mask,
            support_positions=report.support_positions,
            confidence=report.confidence,
            public_complete=allowed,
            reason_code=(
                f"STOP_{self.task.value}_STABLE"
                if allowed
                else report.reason_code
            ),
        )
        if allowed:
            self._final_report = updated
        return updated

    def _base_allowed(
        self, report: TaskReport, cell_levels: tuple[int, ...]
    ) -> bool:
        supports = report.support_positions
        common = bool(np.any(report.predicted_mask)) and report.confidence >= 0.5
        if self.task is BenchmarkTask.LOCATE:
            return (
                common
                and len(supports) >= 3
                and len(set(supports[:, 0])) >= 2
                and len(set(supports[:, 1])) >= 2
            )
        if not common or any(level < 0 for level in cell_levels):
            return False
        labels, count = ndimage.label(report.predicted_mask)
        supported = (
            {
                int(label)
                for label in labels[supports[:, 0], supports[:, 1]]
                if label
            }
            if len(supports)
            else set()
        )
        return supported == set(range(1, count + 1))


def build_task_report(
    evidence: EvidenceMap,
    grid: AcquisitionGrid,
    *,
    cell_levels: tuple[int, ...],
    task: BenchmarkTask,
    cell_indication_fraction: float,
    minimum_component_pixels: int,
) -> TaskReport:
    if (
        type(evidence) is not EvidenceMap
        or type(grid) is not AcquisitionGrid
        or evidence.states.shape != grid.native_shape
        or type(cell_levels) is not tuple
        or len(cell_levels) != 64
        or any(level not in (-1, 0, 1, 2) for level in cell_levels)
        or type(task) is not BenchmarkTask
        or not 0.0 < float(cell_indication_fraction) <= 1.0
        or type(minimum_component_pixels) is not int
        or minimum_component_pixels < 1
    ):
        raise ValueError("task report request is invalid")
    prediction = np.zeros(grid.native_shape, dtype=np.bool_)
    for cell, level in zip(grid.cells, cell_levels, strict=True):
        if level < 0:
            continue
        row_slice, column_slice = _owned_cell_slices(grid, cell.row, cell.column)
        measured = evidence.measured_mask[row_slice, column_slice]
        indication = evidence.indication_mask[row_slice, column_slice]
        measured_count = int(np.count_nonzero(measured))
        indication_count = int(np.count_nonzero(indication))
        if (
            measured_count == 0
            or indication_count == 0
            or indication_count / measured_count < float(cell_indication_fraction)
        ):
            continue
        if level == 2:
            prediction[row_slice, column_slice] = indication
        else:
            prediction[row_slice, column_slice] = True
    if any(level == 2 for level in cell_levels):
        labels, count = ndimage.label(prediction)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        keep = np.zeros_like(prediction)
        for label_id in range(1, count + 1):
            if sizes[label_id] >= minimum_component_pixels:
                keep |= labels == label_id
        prediction = keep
    if task is BenchmarkTask.LOCATE and np.any(prediction):
        labels, count = ndimage.label(prediction)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        prediction = labels == int(np.argmax(sizes))
    support = np.argwhere(prediction & evidence.indication_mask).astype(
        np.int64, copy=False
    )
    confidence = (
        float(np.clip(np.mean(evidence.scores[support[:, 0], support[:, 1]]) / 0.36, 0.0, 1.0))
        if len(support)
        else 0.0
    )
    return TaskReport(
        task=task,
        predicted_mask=prediction,
        support_positions=support,
        confidence=confidence,
        public_complete=False,
        reason_code="EVIDENCE_PRESENT" if len(support) else "NO_INDICATION_EVIDENCE",
    )


def _owned_cell_slices(
    grid: AcquisitionGrid, row: int, column: int
) -> tuple[slice, slice]:
    row_start = grid.row_boundaries[row]
    row_stop = grid.row_boundaries[row + 1] + (1 if row == 7 else 0)
    column_start = grid.column_boundaries[column]
    column_stop = grid.column_boundaries[column + 1] + (1 if column == 7 else 0)
    return slice(row_start, row_stop), slice(column_start, column_stop)


__all__ = ["PublicStopTracker", "build_task_report"]
