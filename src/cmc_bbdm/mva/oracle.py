"""Geometry-only controls and explicit retrospective greedy oracle selection."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import numpy as np

from .acquisition_grid import AcquisitionGrid
from .measurement_state import (
    MeasurementState,
    RefinementAction,
    apply_action,
    fitting_actions,
)


class MVAOracleError(ValueError):
    """Raised when no legal action or an invalid oracle score is supplied."""


@lru_cache(maxsize=1)
def uniform_cell_order() -> tuple[int, ...]:
    """Return a deterministic farthest-point traversal of the 8 x 8 cells."""

    coordinates = {index: divmod(index, 8) for index in range(64)}
    selected = [0]
    remaining = set(range(1, 64))
    while remaining:
        candidate = max(
            remaining,
            key=lambda index: (
                min(
                    (coordinates[index][0] - coordinates[chosen][0]) ** 2
                    + (coordinates[index][1] - coordinates[chosen][1]) ** 2
                    for chosen in selected
                ),
                -index,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return tuple(selected)


def _fitting_actions(
    grid: AcquisitionGrid, state: MeasurementState, checkpoint: float
) -> tuple[RefinementAction, ...]:
    return fitting_actions(grid, state, checkpoint)


def _uniform_from_actions(actions: tuple[RefinementAction, ...]) -> RefinementAction:
    if not actions:
        raise MVAOracleError("no action fits the checkpoint")
    minimum_level = min(action.from_level for action in actions)
    eligible = {
        action.cell_index: action
        for action in actions
        if action.from_level == minimum_level
    }
    for cell_index in uniform_cell_order():
        if cell_index in eligible:
            return eligible[cell_index]
    raise MVAOracleError("uniform order has no eligible action")


def choose_uniform_action(
    grid: AcquisitionGrid,
    state: MeasurementState,
    *,
    checkpoint: float,
) -> RefinementAction:
    """Choose the next geometry-only farthest-spread uniform action."""

    return _uniform_from_actions(_fitting_actions(grid, state, checkpoint))


def choose_random_action(
    grid: AcquisitionGrid,
    state: MeasurementState,
    *,
    checkpoint: float,
    seed: int,
) -> RefinementAction:
    """Choose uniformly from fitting actions using one registered PCG64 seed."""

    if type(seed) is not int:
        raise MVAOracleError("random seed must be an integer")
    actions = _fitting_actions(grid, state, checkpoint)
    if not actions:
        raise MVAOracleError("no action fits the checkpoint")
    generator = np.random.Generator(np.random.PCG64(seed))
    return actions[int(generator.integers(0, len(actions)))]


def choose_greedy_oracle_action(
    grid: AcquisitionGrid,
    state: MeasurementState,
    *,
    checkpoint: float,
    score_candidate: Callable[
        [MeasurementState, RefinementAction, MeasurementState], float
    ],
) -> RefinementAction:
    """Score only legal one-step candidates and apply the registered tie break."""

    if not callable(score_candidate):
        raise MVAOracleError("candidate scorer is required")
    actions = _fitting_actions(grid, state, checkpoint)
    if not actions:
        raise MVAOracleError("no action fits the checkpoint")
    scored: list[tuple[float, RefinementAction]] = []
    for action in actions:
        candidate = apply_action(grid, state, action)
        score = float(score_candidate(state, action, candidate))
        if not math.isfinite(score):
            raise MVAOracleError("oracle score must be finite")
        scored.append((score, action))
    return max(
        scored,
        key=lambda item: (
            item[0],
            -item[1].cell_index,
            -item[1].to_level,
        ),
    )[1]


__all__ = [
    "MVAOracleError",
    "choose_greedy_oracle_action",
    "choose_random_action",
    "choose_uniform_action",
    "uniform_cell_order",
]
