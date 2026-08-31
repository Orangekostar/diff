"""Label-independent state bank for the metadata-free CAI assessor."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

import numpy as np

from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.mva.oracle import uniform_cell_order

from .contracts import InspectionObservation
from .state import (
    GeneralizedMeasurementState,
    InspectionCellAction,
    apply_action,
    candidate_budget_record,
    zero_state,
)
from .surface_hypothesis import SurfaceHypothesis
from .world import CausalInspectionWorld


class StateBankError(ValueError):
    """Raised when a label-independent policy or snapshot roster is invalid."""


class StateBankPolicy(str, Enum):
    UNIFORM_BROADEN = "UNIFORM_BROADEN"
    CENTER_BROADEN = "CENTER_BROADEN"
    RANDOM_BROADEN = "RANDOM_BROADEN"
    SURFACE_FOCUS = "SURFACE_FOCUS"
    UNIFORM_THEN_REFINE = "UNIFORM_THEN_REFINE"
    ALTERNATE_BROADEN_REFINE = "ALTERNATE_BROADEN_REFINE"


@dataclass(frozen=True, slots=True)
class StateBankSnapshot:
    policy: str
    snapshot_index: int
    progress_fraction: float
    observation: InspectionObservation
    state_sha256: str


def _cell_order(
    policy: StateBankPolicy,
    surface_hypothesis: SurfaceHypothesis,
    *,
    surface_sha256: str,
    random_seed: int,
) -> tuple[int, ...]:
    if policy in {
        StateBankPolicy.UNIFORM_BROADEN,
        StateBankPolicy.UNIFORM_THEN_REFINE,
        StateBankPolicy.ALTERNATE_BROADEN_REFINE,
    }:
        return uniform_cell_order()
    if policy is StateBankPolicy.CENTER_BROADEN:
        return tuple(
            sorted(
                range(64),
                key=lambda cell: (
                    (cell // 8 - 3.5) ** 2 + (cell % 8 - 3.5) ** 2,
                    cell,
                ),
            )
        )
    if policy is StateBankPolicy.SURFACE_FOCUS:
        return tuple(
            sorted(
                range(64),
                key=lambda cell: (-float(surface_hypothesis.scores[cell]), cell),
            )
        )
    token = hashlib.sha256(
        f"{random_seed}|{surface_sha256}".encode("ascii")
    ).hexdigest()
    generator = np.random.Generator(np.random.PCG64(int(token[:16], 16)))
    return tuple(int(value) for value in generator.permutation(64))


def _append_if_fitting(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
    action: InspectionCellAction,
    endpoint_budget: float,
    output: list[InspectionCellAction],
) -> GeneralizedMeasurementState:
    if (
        candidate_budget_record(grid, state, action).effective_budget
        > endpoint_budget + 1.0e-15
    ):
        return state
    output.append(action)
    return apply_action(grid, state, action)


def plan_policy_actions(
    grid: AcquisitionGrid,
    policy: StateBankPolicy,
    *,
    surface_hypothesis: SurfaceHypothesis,
    surface_sha256: str,
    random_seed: int,
    endpoint_budget: float,
) -> tuple[InspectionCellAction, ...]:
    endpoint = float(endpoint_budget)
    if (
        type(grid) is not AcquisitionGrid
        or type(policy) is not StateBankPolicy
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(surface_sha256) is not str
        or len(surface_sha256) != 64
        or type(random_seed) is not int
        or isinstance(endpoint_budget, bool)
        or not math.isfinite(endpoint)
        or not 0.0 < endpoint <= 1.0
    ):
        raise StateBankError("state-bank policy request is invalid")
    order = _cell_order(
        policy,
        surface_hypothesis,
        surface_sha256=surface_sha256,
        random_seed=random_seed,
    )
    state = zero_state(grid)
    actions: list[InspectionCellAction] = []
    if policy in {
        StateBankPolicy.UNIFORM_BROADEN,
        StateBankPolicy.CENTER_BROADEN,
        StateBankPolicy.RANDOM_BROADEN,
        StateBankPolicy.SURFACE_FOCUS,
    }:
        for cell in order:
            state = _append_if_fitting(
                grid,
                state,
                InspectionCellAction(cell, -1, 0),
                endpoint,
                actions,
            )
        return tuple(actions)
    if policy is StateBankPolicy.UNIFORM_THEN_REFINE:
        for source, target in ((-1, 0), (0, 1), (1, 2)):
            for cell in order:
                if state.levels[cell] != source:
                    continue
                state = _append_if_fitting(
                    grid,
                    state,
                    InspectionCellAction(cell, source, target),
                    endpoint,
                    actions,
                )
        return tuple(actions)
    for cell in order:
        state = _append_if_fitting(
            grid,
            state,
            InspectionCellAction(cell, -1, 0),
            endpoint,
            actions,
        )
        if state.levels[cell] == 0:
            state = _append_if_fitting(
                grid,
                state,
                InspectionCellAction(cell, 0, 1),
                endpoint,
                actions,
            )
    return tuple(actions)


def _snapshot(
    policy: str,
    index: int,
    fraction: float,
    observation: InspectionObservation,
) -> StateBankSnapshot:
    digest = hashlib.sha256(
        (
            f"inspection-agent-state-bank-v1|{policy}|{index}|"
            f"{fraction:.17g}|{observation.state_sha256}"
        ).encode("ascii")
    ).hexdigest()
    return StateBankSnapshot(
        policy=policy,
        snapshot_index=index,
        progress_fraction=fraction,
        observation=observation,
        state_sha256=digest,
    )


def materialize_state_bank(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    surface_hypothesis: SurfaceHypothesis,
    *,
    random_seed: int,
    snapshot_fractions: tuple[float, ...],
) -> tuple[StateBankSnapshot, ...]:
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(snapshot_fractions) is not tuple
        or len(snapshot_fractions) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= 1.0
            for value in snapshot_fractions
        )
        or any(
            float(second) <= float(first)
            for first, second in pairwise(snapshot_fractions)
        )
        or float(snapshot_fractions[-1]) != 1.0
    ):
        raise StateBankError("state-bank snapshot request is invalid")
    zero = world.reset()
    rows = [_snapshot("ZERO_ANCHOR", -1, 0.0, zero)]
    for policy in StateBankPolicy:
        current = world.reset()
        actions = plan_policy_actions(
            grid,
            policy,
            surface_hypothesis=surface_hypothesis,
            surface_sha256=current.surface_sha256,
            random_seed=random_seed,
            endpoint_budget=current.endpoint_budget,
        )
        if not actions:
            raise StateBankError("state-bank policy produced no action")
        selected = tuple(
            min(len(actions) - 1, math.ceil(float(fraction) * len(actions)) - 1)
            for fraction in snapshot_fractions
        )
        if len(set(selected)) != len(selected):
            raise StateBankError("state-bank snapshot indices are not unique")
        by_action = {action_index: snapshot for snapshot, action_index in enumerate(selected)}
        for action_index, action in enumerate(actions):
            current = world.step(current, action)
            if action_index in by_action:
                snapshot_index = by_action[action_index]
                rows.append(
                    _snapshot(
                        policy.value,
                        snapshot_index,
                        float(snapshot_fractions[snapshot_index]),
                        current,
                    )
                )
    if len(rows) != 19:
        raise StateBankError("state-bank row count changed")
    return tuple(rows)


__all__ = [
    "StateBankError",
    "StateBankPolicy",
    "StateBankSnapshot",
    "materialize_state_bank",
    "plan_policy_actions",
]
