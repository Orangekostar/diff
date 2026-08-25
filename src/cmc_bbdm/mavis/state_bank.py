"""Causal trajectory materialization for the MAVIS source state bank."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import pairwise

from cmc_bbdm.mva.measurement_state import RefinementAction

from .authority import MAVISAuthority
from .contracts import InspectionState
from .reveal import MAVISRevealError, reveal_action_history


class MAVISStateBankError(ValueError):
    """Raised when a source trajectory violates state-bank causality."""


@dataclass(frozen=True, slots=True)
class PlannedAction:
    action: RefinementAction
    nominal_checkpoint: float

    def __post_init__(self) -> None:
        if type(self.action) is not RefinementAction:
            raise MAVISStateBankError("planned action is invalid")
        checkpoint = float(self.nominal_checkpoint)
        if (
            isinstance(self.nominal_checkpoint, bool)
            or not math.isfinite(checkpoint)
            or not 0.0 < checkpoint <= 1.0
        ):
            raise MAVISStateBankError("planned action checkpoint is invalid")
        object.__setattr__(self, "nominal_checkpoint", checkpoint)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    nominal_checkpoint: float
    step: int
    inspection_state: InspectionState


@dataclass(frozen=True, slots=True)
class StateTrajectory:
    specimen_id: str
    method: str
    seed: int | None
    initial_budget: float
    checkpoints: tuple[float, ...]
    actions: tuple[PlannedAction, ...]
    snapshots: tuple[StateSnapshot, ...]
    state_sha256: str


def _checkpoints(values: tuple[float, ...]) -> tuple[float, ...]:
    if type(values) is not tuple or not values:
        raise MAVISStateBankError("trajectory checkpoints are invalid")
    output = tuple(float(value) for value in values)
    if (
        any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in output)
        or any(second <= first for first, second in pairwise(output))
    ):
        raise MAVISStateBankError("trajectory checkpoints are invalid")
    return output


def _trajectory_hash(
    specimen_id: str,
    method: str,
    seed: int | None,
    initial_budget: float,
    checkpoints: tuple[float, ...],
    actions: tuple[PlannedAction, ...],
    snapshots: tuple[StateSnapshot, ...],
) -> str:
    payload = {
        "schema": 1,
        "specimen_id": specimen_id,
        "method": method,
        "seed": seed,
        "initial_budget": initial_budget,
        "checkpoints": checkpoints,
        "actions": tuple(
            (
                item.action.cell_index,
                item.action.from_level,
                item.action.to_level,
                item.nominal_checkpoint,
            )
            for item in actions
        ),
        "snapshots": tuple(
            (item.nominal_checkpoint, item.step, item.inspection_state.state_sha256)
            for item in snapshots
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def materialize_action_plan(
    authority: MAVISAuthority,
    *,
    specimen_id: str,
    method: str,
    seed: int | None,
    initial_budget: float,
    checkpoints: tuple[float, ...],
    actions: tuple[PlannedAction, ...],
) -> StateTrajectory:
    if (
        type(authority) is not MAVISAuthority
        or type(specimen_id) is not str
        or not specimen_id
        or type(method) is not str
        or not method
        or (seed is not None and type(seed) is not int)
        or type(actions) is not tuple
        or any(type(item) is not PlannedAction for item in actions)
    ):
        raise MAVISStateBankError("trajectory inputs are invalid")
    caps = _checkpoints(checkpoints)
    if any(item.nominal_checkpoint not in caps for item in actions) or any(
        second.nominal_checkpoint < first.nominal_checkpoint
        for first, second in pairwise(actions)
    ):
        raise MAVISStateBankError("planned action checkpoint roster changed")
    context = authority.policy_context(specimen_id)
    action_index = 0
    snapshots: list[StateSnapshot] = []
    for checkpoint in caps:
        while (
            action_index < len(actions)
            and actions[action_index].nominal_checkpoint == checkpoint
        ):
            action_index += 1
        try:
            current = reveal_action_history(
                authority,
                context,
                initial_budget=initial_budget,
                checkpoint=caps[-1],
                actions=tuple(item.action for item in actions[:action_index]),
            )
        except MAVISRevealError as error:
            raise MAVISStateBankError(
                "trajectory action or checkpoint is invalid"
            ) from error
        quantized_initial_scout = action_index == 0 and math.isclose(
            checkpoint,
            float(initial_budget),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        if (
            current.effective_budget > checkpoint + 1.0e-15
            and not quantized_initial_scout
        ):
            raise MAVISStateBankError("trajectory state exceeds its checkpoint")
        snapshots.append(
            StateSnapshot(
                nominal_checkpoint=checkpoint,
                step=action_index,
                inspection_state=current,
            )
        )
    if action_index != len(actions):
        raise MAVISStateBankError("trajectory actions are incomplete")
    frozen_snapshots = tuple(snapshots)
    return StateTrajectory(
        specimen_id=specimen_id,
        method=method,
        seed=seed,
        initial_budget=float(initial_budget),
        checkpoints=caps,
        actions=actions,
        snapshots=frozen_snapshots,
        state_sha256=_trajectory_hash(
            specimen_id,
            method,
            seed,
            float(initial_budget),
            caps,
            actions,
            frozen_snapshots,
        ),
    )


__all__ = [
    "MAVISStateBankError",
    "PlannedAction",
    "StateSnapshot",
    "StateTrajectory",
    "materialize_action_plan",
]
