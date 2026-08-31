"""Exact-checkpoint G0 curve, task-swap, and trajectory metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .contracts import InspectionDecision
from .oracle import OracleTrajectory
from .state import InspectionCellAction


class InspectionEvaluationError(ValueError):
    """Raised when a G0 curve or task-swap comparison is invalid."""


def _readonly(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise InspectionEvaluationError("evaluation array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(shape)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True, eq=False)
class TaskSwapAdvantages:
    field_advantage: np.ndarray
    cai_advantage: np.ndarray


@dataclass(frozen=True, slots=True)
class TrajectoryOverlap:
    action_jaccard: float
    cell_jaccard: float
    high_level_action_overlap: float
    normalized_edit_distance: float


@dataclass(frozen=True, slots=True)
class CheckpointCurve:
    nominal_budgets: np.ndarray
    exact_budgets: np.ndarray
    task_losses: np.ndarray
    state_sha256: tuple[str, ...]


def task_swap_advantages(
    *,
    field_on_field: object,
    cai_on_field: object,
    cai_on_cai: object,
    field_on_cai: object,
) -> TaskSwapAdvantages:
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (field_on_field, cai_on_field, cai_on_cai, field_on_cai)
    )
    if (
        arrays[0].ndim != 1
        or arrays[0].size == 0
        or any(value.shape != arrays[0].shape for value in arrays[1:])
        or any(not np.all(np.isfinite(value)) for value in arrays)
    ):
        raise InspectionEvaluationError("task-swap arrays are invalid")
    return TaskSwapAdvantages(
        field_advantage=_readonly(arrays[1] - arrays[0], arrays[0].shape),
        cai_advantage=_readonly(arrays[3] - arrays[2], arrays[0].shape),
    )


def _jaccard(left: set[object], right: set[object]) -> float:
    union = left | right
    return 1.0 if not union else float(len(left & right) / len(union))


def _edit_distance(left: tuple[object, ...], right: tuple[object, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def trajectory_overlap(
    field_actions: tuple[InspectionCellAction, ...],
    cai_actions: tuple[InspectionCellAction, ...],
    *,
    field_decisions: tuple[InspectionDecision, ...],
    cai_decisions: tuple[InspectionDecision, ...],
) -> TrajectoryOverlap:
    if (
        type(field_actions) is not tuple
        or type(cai_actions) is not tuple
        or not field_actions
        or not cai_actions
        or any(type(value) is not InspectionCellAction for value in (*field_actions, *cai_actions))
        or len(field_decisions) != len(field_actions)
        or len(cai_decisions) != len(cai_actions)
        or any(type(value) is not InspectionDecision for value in (*field_decisions, *cai_decisions))
    ):
        raise InspectionEvaluationError("trajectory-overlap request is invalid")
    field_tokens = tuple(
        (action.cell_index, action.from_level, action.to_level)
        for action in field_actions
    )
    cai_tokens = tuple(
        (action.cell_index, action.from_level, action.to_level)
        for action in cai_actions
    )
    action_jaccard = _jaccard(set(field_tokens), set(cai_tokens))
    cell_jaccard = _jaccard(
        {action.cell_index for action in field_actions},
        {action.cell_index for action in cai_actions},
    )
    common = min(len(field_decisions), len(cai_decisions))
    decision_overlap = sum(
        field_decisions[index] is cai_decisions[index] for index in range(common)
    ) / max(len(field_decisions), len(cai_decisions))
    edit = _edit_distance(field_tokens, cai_tokens) / max(
        len(field_tokens), len(cai_tokens)
    )
    return TrajectoryOverlap(
        action_jaccard=action_jaccard,
        cell_jaccard=cell_jaccard,
        high_level_action_overlap=float(decision_overlap),
        normalized_edit_distance=float(edit),
    )


def project_oracle_checkpoints(
    trajectory: OracleTrajectory,
    checkpoints: tuple[float, ...],
) -> CheckpointCurve:
    if (
        type(trajectory) is not OracleTrajectory
        or not trajectory.steps
        or type(checkpoints) is not tuple
        or not checkpoints
        or checkpoints[0] != 0.0
        or any(not math.isfinite(float(value)) for value in checkpoints)
        or any(second <= first for first, second in pairwise(checkpoints))
    ):
        raise InspectionEvaluationError("checkpoint projection is invalid")
    exact = []
    losses = []
    states = []
    initial = trajectory.steps[0]
    for checkpoint in checkpoints:
        eligible = [
            step for step in trajectory.steps if step.budget_after <= checkpoint + 1.0e-15
        ]
        if eligible:
            selected = eligible[-1]
            exact.append(selected.budget_after)
            losses.append(selected.task_loss_after)
            states.append(selected.state_sha256_after)
        else:
            exact.append(0.0)
            losses.append(initial.task_loss_before)
            states.append(initial.state_sha256_before)
    return CheckpointCurve(
        nominal_budgets=_readonly(checkpoints, (len(checkpoints),)),
        exact_budgets=_readonly(exact, (len(checkpoints),)),
        task_losses=_readonly(losses, (len(checkpoints),)),
        state_sha256=tuple(states),
    )


def zero_inclusive_auebc(budgets: object, losses: object) -> float:
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    if (
        x.ndim != 1
        or x.size < 2
        or y.shape != x.shape
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or x[0] != 0.0
        or x[-1] != 0.25
        or np.any(np.diff(x) <= 0.0)
    ):
        raise InspectionEvaluationError("zero-inclusive AUEBC curve is invalid")
    return float(np.trapezoid(y, x))


__all__ = [
    "CheckpointCurve",
    "InspectionEvaluationError",
    "TaskSwapAdvantages",
    "TrajectoryOverlap",
    "project_oracle_checkpoints",
    "task_swap_advantages",
    "trajectory_overlap",
    "zero_inclusive_auebc",
]
