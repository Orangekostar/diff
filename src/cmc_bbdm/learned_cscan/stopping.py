"""Visible rule and learned-stop contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from torch import nn

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


class LearnedStopStatus(StrEnum):
    AUTHORIZED = "LEARNED_STOP_AUTHORIZED"
    NOT_AUTHORIZED = "LEARNED_STOP_NOT_AUTHORIZED"


@dataclass(frozen=True, slots=True)
class StopValidationRow:
    specimen_key: str
    task: Task
    probability: float
    label: bool
    mechanically_eligible: bool

    def __post_init__(self) -> None:
        probability = float(self.probability)
        if (
            not self.specimen_key
            or type(self.task) is not Task
            or isinstance(self.probability, bool)
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
            or type(self.label) is not bool
            or type(self.mechanically_eligible) is not bool
        ):
            raise ValueError("stop validation row is invalid")
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True, slots=True)
class StopThresholdDiagnostic:
    threshold: float
    predicted_stop_count: int
    true_stop_count: int
    false_stop_count: int
    negative_support_count: int
    false_stop_rate: float
    authorized: bool


@dataclass(frozen=True, slots=True)
class LearnedStopAuthorization:
    status: LearnedStopStatus
    threshold: float | None
    diagnostics: tuple[StopThresholdDiagnostic, ...]
    validation_specimen_count: int


@dataclass(frozen=True, slots=True)
class LearnedStopDecision:
    should_stop: bool
    authorized: bool
    probability: float
    reason_code: str


@dataclass(frozen=True, slots=True)
class AutonomousOutcome:
    completed: bool
    stop_success: bool
    failure_type: str | None


class LearnedStopHead(nn.Module):
    """Small stop head trained separately from the action actor."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(17 + 10 + 9 + 5, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        cell_features: torch.Tensor,
        subblock_features: torch.Tensor,
        global_features: torch.Tensor,
        history_features: torch.Tensor,
    ) -> torch.Tensor:
        if (
            cell_features.ndim != 3
            or cell_features.shape[1:] != (64, 17)
            or subblock_features.shape != (cell_features.shape[0], 64, 16, 10)
            or global_features.shape != (cell_features.shape[0], 9)
            or history_features.shape != (cell_features.shape[0], 4, 5)
        ):
            raise ValueError("learned stop tensor contract is invalid")
        summary = torch.cat(
            (
                cell_features.mean(dim=1),
                subblock_features.mean(dim=(1, 2)),
                global_features,
                history_features.mean(dim=1),
            ),
            dim=1,
        )
        return self.network(summary).squeeze(-1)


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


def calibrate_learned_stop(
    rows: tuple[StopValidationRow, ...],
    *,
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
    minimum_true_stop_support: int = 3,
    minimum_negative_support: int = 20,
    maximum_false_stop_rate: float = 0.05,
) -> LearnedStopAuthorization:
    if (
        type(rows) is not tuple
        or not rows
        or any(type(row) is not StopValidationRow for row in rows)
        or type(thresholds) is not tuple
        or thresholds != tuple(sorted(set(thresholds)))
        or any(not 0.0 < float(value) < 1.0 for value in thresholds)
        or type(minimum_true_stop_support) is not int
        or minimum_true_stop_support < 1
        or type(minimum_negative_support) is not int
        or minimum_negative_support < 1
        or not 0.0 <= float(maximum_false_stop_rate) < 1.0
    ):
        raise ValueError("learned stop calibration request is invalid")
    negative_support = sum(
        row.mechanically_eligible and not row.label for row in rows
    )
    diagnostics = []
    for threshold in thresholds:
        predicted = tuple(
            row
            for row in rows
            if row.mechanically_eligible and row.probability >= threshold
        )
        true_stops = sum(row.label for row in predicted)
        false_stops = sum(not row.label for row in predicted)
        false_rate = (
            float(false_stops / negative_support) if negative_support else 1.0
        )
        authorized = (
            true_stops >= minimum_true_stop_support
            and negative_support >= minimum_negative_support
            and false_rate <= maximum_false_stop_rate
        )
        diagnostics.append(
            StopThresholdDiagnostic(
                threshold=float(threshold),
                predicted_stop_count=len(predicted),
                true_stop_count=true_stops,
                false_stop_count=false_stops,
                negative_support_count=negative_support,
                false_stop_rate=false_rate,
                authorized=authorized,
            )
        )
    eligible = [row for row in diagnostics if row.authorized]
    selected = (
        min(
            eligible,
            key=lambda row: (
                -row.true_stop_count,
                row.false_stop_rate,
                row.threshold,
            ),
        )
        if eligible
        else None
    )
    return LearnedStopAuthorization(
        status=(
            LearnedStopStatus.AUTHORIZED
            if selected is not None
            else LearnedStopStatus.NOT_AUTHORIZED
        ),
        threshold=None if selected is None else selected.threshold,
        diagnostics=tuple(diagnostics),
        validation_specimen_count=len({row.specimen_key for row in rows}),
    )


def learned_stop_decision(
    *,
    probability: float,
    mechanically_eligible: bool,
    authorization: LearnedStopAuthorization,
) -> LearnedStopDecision:
    value = float(probability)
    if (
        isinstance(probability, bool)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
        or type(mechanically_eligible) is not bool
        or type(authorization) is not LearnedStopAuthorization
    ):
        raise ValueError("learned stop decision request is invalid")
    authorized = (
        authorization.status is LearnedStopStatus.AUTHORIZED
        and authorization.threshold is not None
    )
    should_stop = bool(
        authorized
        and mechanically_eligible
        and value >= float(authorization.threshold)
    )
    if not authorized:
        reason = LearnedStopStatus.NOT_AUTHORIZED.value
    elif not mechanically_eligible:
        reason = "MECHANICAL_REQUIREMENTS_UNMET"
    elif should_stop:
        reason = "LEARNED_STOP_THRESHOLD_MET"
    else:
        reason = "LEARNED_STOP_THRESHOLD_UNMET"
    return LearnedStopDecision(should_stop, authorized, value, reason)


def classify_autonomous_outcome(
    *, stopped: bool, report_success: bool, resource_exhausted: bool
) -> AutonomousOutcome:
    if any(type(value) is not bool for value in (stopped, report_success, resource_exhausted)):
        raise TypeError("autonomous outcome flags must be booleans")
    if stopped:
        return AutonomousOutcome(
            completed=report_success,
            stop_success=report_success,
            failure_type=None if report_success else "FALSE_STOP",
        )
    if resource_exhausted:
        return AutonomousOutcome(False, False, "RESOURCE_EXHAUSTED")
    return AutonomousOutcome(False, False, "NO_STOP")


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


__all__ = [
    "AutonomousOutcome",
    "LearnedStopAuthorization",
    "LearnedStopDecision",
    "LearnedStopHead",
    "LearnedStopStatus",
    "RuleStopController",
    "StopDecision",
    "StopThresholdDiagnostic",
    "StopValidationRow",
    "calibrate_learned_stop",
    "classify_autonomous_outcome",
    "learned_stop_decision",
]
