"""Visible rule and learned-stop contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cmc_bbdm.inspection_agent.state import InspectionCellAction

from .contracts import Task
from .readout import TaskReportV2


@dataclass(frozen=True, slots=True)
class StopDecision:
    should_stop: bool
    relevant_update: bool
    stable_relevant_updates: int
    unmet_conditions: tuple[str, ...]
    reason_code: str


class RuleStopController:
    def __init__(self, task: Task) -> None:
        if type(task) is not Task:
            raise TypeError("typed task is required")
        self.task = task
        self._previous_relevant_mask: np.ndarray | None = None
        self._stable_relevant_updates = 0
        self._final: StopDecision | None = None

    def update(
        self,
        report: TaskReportV2,
        *,
        cell_levels: tuple[int, ...],
        last_action: InspectionCellAction | None,
    ) -> StopDecision:
        if (
            type(report) is not TaskReportV2
            or report.task is not self.task
            or type(cell_levels) is not tuple
            or len(cell_levels) != 64
            or any(level not in (-1, 0, 1, 2) for level in cell_levels)
            or (last_action is not None and type(last_action) is not InspectionCellAction)
        ):
            raise ValueError("rule stop update is invalid")
        if self._final is not None:
            return self._final
        relevant_cells = set(report.candidate_cells)
        relevant_cells.update(_neighbor_ring(report.candidate_cells))
        relevant = last_action is not None and last_action.cell_index in relevant_cells
        if relevant:
            if self._previous_relevant_mask is None:
                self._stable_relevant_updates = 0
            elif _stable(self._previous_relevant_mask, report.predicted_mask):
                self._stable_relevant_updates += 1
            else:
                self._stable_relevant_updates = 0
            self._previous_relevant_mask = np.array(report.predicted_mask, copy=True)
        unmet = _unmet_conditions(self.task, report, cell_levels)
        if self._stable_relevant_updates < 2:
            unmet.append("TWO_RELEVANT_STABLE_UPDATES")
        should_stop = not unmet
        decision = StopDecision(
            should_stop=should_stop,
            relevant_update=relevant,
            stable_relevant_updates=self._stable_relevant_updates,
            unmet_conditions=tuple(unmet),
            reason_code=(
                f"STOP_{self.task.value}_VISIBLE_STABLE"
                if should_stop
                else "CONTINUE_VISIBLE_REQUIREMENTS"
            ),
        )
        if should_stop:
            self._final = decision
        return decision


def _unmet_conditions(
    task: Task, report: TaskReportV2, levels: tuple[int, ...]
) -> list[str]:
    unmet: list[str] = []
    if any(level < 0 for level in levels):
        unmet.append("FULL_LEVEL0_COVERAGE")
    supports = report.support_positions
    if len(supports) < 3:
        unmet.append("THREE_SUPPORT_POINTS")
    if len(supports) and len({int(value) for value in supports[:, 0]}) < 2:
        unmet.append("TWO_SUPPORT_ROWS")
    if len(supports) and len({int(value) for value in supports[:, 1]}) < 2:
        unmet.append("TWO_SUPPORT_COLUMNS")
    if not report.candidate_cells:
        unmet.append("VISIBLE_CANDIDATE")
        return unmet
    candidate_level = 1 if task is Task.LOCATE else 2
    if any(levels[cell] < candidate_level for cell in report.candidate_cells):
        unmet.append(f"CANDIDATE_LEVEL{candidate_level}")
    ring_level = 0 if task is Task.LOCATE else 1
    if any(levels[cell] < ring_level for cell in _neighbor_ring(report.candidate_cells)):
        unmet.append(f"NEIGHBOR_RING_LEVEL{ring_level}")
    return unmet


def _stable(previous: np.ndarray, current: np.ndarray) -> bool:
    if previous.shape != current.shape:
        return False
    area_delta = abs(
        np.count_nonzero(previous) / previous.size
        - np.count_nonzero(current) / current.size
    )
    previous_bbox = _normalized_bbox(previous)
    current_bbox = _normalized_bbox(current)
    if previous_bbox is None or current_bbox is None:
        bbox_delta = 0.0 if previous_bbox is current_bbox else 1.0
    else:
        bbox_delta = max(
            abs(left - right)
            for left, right in zip(previous_bbox, current_bbox, strict=True)
        )
    return area_delta <= 0.05 and bbox_delta <= 1.0 / 64.0


def _normalized_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    points = np.argwhere(mask)
    if not len(points):
        return None
    height, width = mask.shape
    return (
        float(points[:, 0].min() / max(height - 1, 1)),
        float(points[:, 1].min() / max(width - 1, 1)),
        float(points[:, 0].max() / max(height - 1, 1)),
        float(points[:, 1].max() / max(width - 1, 1)),
    )


def _neighbor_ring(cells: tuple[int, ...]) -> tuple[int, ...]:
    source = set(cells)
    output: set[int] = set()
    for cell in source:
        row, column = divmod(cell, 8)
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                candidate_row = row + row_delta
                candidate_column = column + column_delta
                if (
                    0 <= candidate_row < 8
                    and 0 <= candidate_column < 8
                    and (row_delta or column_delta)
                ):
                    output.add(candidate_row * 8 + candidate_column)
    return tuple(sorted(output - source))


__all__ = ["RuleStopController", "StopDecision"]
