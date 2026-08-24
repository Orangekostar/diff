"""Score-once rankings and exact-cost frozen MVD refinement plans."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.mva.measurement_state import (
    MeasurementState,
    RefinementAction,
    action_fits_checkpoint,
    apply_action,
    budget_record,
)
from cmc_bbdm.mva.oracle_trajectory import TrajectorySnapshot


class OneShotOracleError(ValueError):
    """Raised when a ranking or fixed one-shot plan violates M0."""


@dataclass(frozen=True, slots=True)
class FrozenInitialRanking:
    method: str
    scores: np.ndarray
    cell_order: tuple[int, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class OneShotPlan:
    method: str
    grid_state_sha256: str
    ranking_state_sha256: str
    checkpoints: tuple[float, ...]
    actions: tuple[RefinementAction, ...]
    action_ranking_positions: tuple[int, ...]
    action_checkpoints: tuple[float, ...]
    snapshots: tuple[TrajectorySnapshot, ...]
    state_sha256: str


def _readonly_scores(value: object) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.shape != (64,) or not np.all(np.isfinite(array)):
        raise OneShotOracleError("initial score vector must contain 64 finite values")
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.float64)
    output.setflags(write=False)
    return output


def _ranking_state(method: str, scores: np.ndarray, order: tuple[int, ...]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"cell_order": order, "method": method},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(scores.tobytes(order="C"))
    return digest.hexdigest()


def score_initial_ranking(
    score_initial: Callable[[], Sequence[float] | np.ndarray], *, method: str
) -> FrozenInitialRanking:
    """Invoke the issued S0 scorer once and freeze its complete ranking."""

    if not callable(score_initial) or type(method) is not str or not method:
        raise OneShotOracleError("one-shot scorer and method are required")
    scores = _readonly_scores(score_initial())
    order = tuple(sorted(range(64), key=lambda cell: (-scores[cell], cell)))
    return FrozenInitialRanking(
        method=method,
        scores=scores,
        cell_order=order,
        state_sha256=_ranking_state(method, scores, order),
    )


def _checkpoints(values: tuple[float, ...]) -> tuple[float, ...]:
    if type(values) is not tuple:
        raise OneShotOracleError("checkpoints must be a tuple")
    output = tuple(float(value) for value in values)
    if (
        not output
        or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in output)
        or any(second <= first for first, second in pairwise(output))
    ):
        raise OneShotOracleError("checkpoints must increase within (0,1]")
    return output


def _plan_state(plan: OneShotPlan) -> str:
    payload = {
        "action_checkpoints": plan.action_checkpoints,
        "action_ranking_positions": plan.action_ranking_positions,
        "actions": [
            (action.cell_index, action.from_level, action.to_level)
            for action in plan.actions
        ],
        "checkpoints": plan.checkpoints,
        "grid_state_sha256": plan.grid_state_sha256,
        "method": plan.method,
        "ranking_state_sha256": plan.ranking_state_sha256,
        "snapshots": [
            (
                snapshot.nominal_checkpoint,
                snapshot.state.levels,
                snapshot.measured_count,
                snapshot.native_count,
                snapshot.effective_budget,
                snapshot.cumulative_actions,
            )
            for snapshot in plan.snapshots
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def plan_frozen_ranking(
    grid: AcquisitionGrid,
    state: MeasurementState,
    *,
    ranking: FrozenInitialRanking,
    checkpoints: tuple[float, ...],
) -> OneShotPlan:
    """Traverse one frozen S0 ranking under exact unique-location caps."""

    caps = _checkpoints(checkpoints)
    if (
        type(grid) is not AcquisitionGrid
        or type(state) is not MeasurementState
        or state.grid_sha256 != grid.state_sha256
        or state.levels != (0,) * 64
        or type(ranking) is not FrozenInitialRanking
        or ranking.cell_order != tuple(
            sorted(range(64), key=lambda cell: (-ranking.scores[cell], cell))
        )
        or _ranking_state(ranking.method, ranking.scores, ranking.cell_order)
        != ranking.state_sha256
    ):
        raise OneShotOracleError("one-shot plan authority changed")
    current = state
    selected: set[int] = set()
    actions: list[RefinementAction] = []
    positions: list[int] = []
    selected_checkpoints: list[float] = []
    snapshots: list[TrajectorySnapshot] = []
    for checkpoint in caps:
        for position, cell_index in enumerate(ranking.cell_order):
            if cell_index in selected:
                continue
            action = RefinementAction(cell_index, 0, 1)
            if not action_fits_checkpoint(grid, current, action, checkpoint):
                continue
            current = apply_action(grid, current, action)
            selected.add(cell_index)
            actions.append(action)
            positions.append(position)
            selected_checkpoints.append(checkpoint)
        budget = budget_record(grid, current)
        snapshots.append(
            TrajectorySnapshot(
                nominal_checkpoint=checkpoint,
                state=current,
                measured_count=budget.measured_count,
                native_count=budget.native_count,
                effective_budget=budget.effective_budget,
                cumulative_actions=len(actions),
            )
        )
    plan = OneShotPlan(
        method=ranking.method,
        grid_state_sha256=grid.state_sha256,
        ranking_state_sha256=ranking.state_sha256,
        checkpoints=caps,
        actions=tuple(actions),
        action_ranking_positions=tuple(positions),
        action_checkpoints=tuple(selected_checkpoints),
        snapshots=tuple(snapshots),
        state_sha256="",
    )
    object.__setattr__(plan, "state_sha256", _plan_state(plan))
    return plan


__all__ = [
    "FrozenInitialRanking",
    "OneShotOracleError",
    "OneShotPlan",
    "plan_frozen_ranking",
    "score_initial_ranking",
]
