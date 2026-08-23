"""Checkpointed MVA control trajectories with actual-budget caps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

import numpy as np

from .acquisition_grid import AcquisitionGrid
from .measurement_state import (
    MeasurementState,
    RefinementAction,
    apply_action,
    budget_record,
    fitting_actions,
)
from .oracle import MVAOracleError, _uniform_from_actions


@dataclass(frozen=True, slots=True)
class TrajectorySnapshot:
    nominal_checkpoint: float
    state: MeasurementState
    measured_count: int
    native_count: int
    effective_budget: float
    cumulative_actions: int


@dataclass(frozen=True, slots=True)
class ControlTrajectory:
    method: str
    seed: int | None
    actions: tuple[RefinementAction, ...]
    snapshots: tuple[TrajectorySnapshot, ...]


def _checkpoints(values: tuple[float, ...]) -> tuple[float, ...]:
    output = tuple(float(value) for value in values)
    if (
        not output
        or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in output)
        or any(second <= first for first, second in pairwise(output))
    ):
        raise MVAOracleError("checkpoints must be strictly increasing in (0,1]")
    return output


@lru_cache(maxsize=1024)
def run_control_trajectory(
    grid: AcquisitionGrid,
    state: MeasurementState,
    *,
    checkpoints: tuple[float, ...],
    method: str,
    seed: int | None = None,
) -> ControlTrajectory:
    """Run uniform or random acquisition through all registered checkpoints."""

    caps = _checkpoints(checkpoints)
    if method not in {"uniform", "random"}:
        raise MVAOracleError("control method must be uniform or random")
    if method == "uniform" and seed is not None:
        raise MVAOracleError("uniform control does not accept a seed")
    if method == "random" and type(seed) is not int:
        raise MVAOracleError("random control requires an integer seed")
    generator = None if seed is None else np.random.Generator(np.random.PCG64(seed))
    current = state
    selected_actions: list[RefinementAction] = []
    snapshots: list[TrajectorySnapshot] = []
    for checkpoint in caps:
        while True:
            fitting = fitting_actions(grid, current, checkpoint)
            if not fitting:
                break
            if method == "uniform":
                action = _uniform_from_actions(fitting)
            else:
                assert generator is not None
                action = fitting[int(generator.integers(0, len(fitting)))]
            current = apply_action(grid, current, action)
            selected_actions.append(action)
        budget = budget_record(grid, current)
        snapshots.append(
            TrajectorySnapshot(
                nominal_checkpoint=checkpoint,
                state=current,
                measured_count=budget.measured_count,
                native_count=budget.native_count,
                effective_budget=budget.effective_budget,
                cumulative_actions=len(selected_actions),
            )
        )
    return ControlTrajectory(
        method=method,
        seed=seed,
        actions=tuple(selected_actions),
        snapshots=tuple(snapshots),
    )


__all__ = ["ControlTrajectory", "TrajectorySnapshot", "run_control_trajectory"]
