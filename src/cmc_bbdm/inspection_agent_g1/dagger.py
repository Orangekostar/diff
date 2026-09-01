"""Source-only DAgger relabel selection for observable G1 policies."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .teacher import SourceTeacherAuthorization

REGISTERED_DAGGER_ITERATIONS = (0, 1, 2)
DAGGER_MAX_NEW_STATES = 16


class DaggerSourceError(ValueError):
    """Raised when a policy-visited relabel row is not an authorized source row."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class DaggerVisitedState:
    outer_target: str
    source_domain: str
    specimen_sha256: str
    task: InspectionTask
    iteration: int
    trajectory_index: int
    trajectory_length: int
    observation_sha256: str
    policy_state_sha256: str
    teacher_label_sha256: str
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domain) is not str
            or not self.source_domain
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or self.iteration not in REGISTERED_DAGGER_ITERATIONS[1:]
            or type(self.trajectory_index) is not int
            or type(self.trajectory_length) is not int
            or self.trajectory_length < 1
            or not 0 <= self.trajectory_index < self.trajectory_length
            or not all(
                _valid_sha256(value)
                for value in (
                    self.observation_sha256,
                    self.policy_state_sha256,
                    self.teacher_label_sha256,
                )
            )
        ):
            raise DaggerSourceError("DAgger visited-state identity is invalid")
        state = _json_sha(
            {
                "schema": 1,
                "kind": "g1-dagger-visited-state",
                "outer_target": self.outer_target,
                "source_domain": self.source_domain,
                "specimen_sha256": self.specimen_sha256,
                "task": self.task.value,
                "iteration": self.iteration,
                "trajectory_index": self.trajectory_index,
                "trajectory_length": self.trajectory_length,
                "observation_sha256": self.observation_sha256,
                "policy_state_sha256": self.policy_state_sha256,
                "teacher_label_sha256": self.teacher_label_sha256,
            }
        )
        if self.state_sha256 not in ("", state):
            raise DaggerSourceError("DAgger visited-state hash changed")
        object.__setattr__(self, "state_sha256", state)


def trajectory_quantile_indices(
    trajectory_length: int,
    *,
    maximum_states: int = DAGGER_MAX_NEW_STATES,
) -> tuple[int, ...]:
    if (
        type(trajectory_length) is not int
        or trajectory_length < 1
        or type(maximum_states) is not int
        or not 1 <= maximum_states <= DAGGER_MAX_NEW_STATES
    ):
        raise DaggerSourceError("DAgger trajectory quantile request is invalid")
    count = min(trajectory_length, maximum_states)
    if count == trajectory_length:
        return tuple(range(trajectory_length))
    return tuple(
        min(
            trajectory_length - 1,
            ((2 * quantile + 1) * trajectory_length) // (2 * count),
        )
        for quantile in range(count)
    )


def select_source_relabels(
    authorization: SourceTeacherAuthorization,
    trajectory: tuple[DaggerVisitedState, ...],
) -> tuple[DaggerVisitedState, ...]:
    if (
        type(authorization) is not SourceTeacherAuthorization
        or type(trajectory) is not tuple
        or not trajectory
        or any(type(row) is not DaggerVisitedState for row in trajectory)
    ):
        raise DaggerSourceError("DAgger source relabel request is invalid")
    first = trajectory[0]
    identity = (
        first.outer_target,
        first.source_domain,
        first.specimen_sha256,
        first.task,
        first.iteration,
        first.trajectory_length,
    )
    if any(
        (
            row.outer_target,
            row.source_domain,
            row.specimen_sha256,
            row.task,
            row.iteration,
            row.trajectory_length,
        )
        != identity
        for row in trajectory
    ) or tuple(row.trajectory_index for row in trajectory) != tuple(
        range(first.trajectory_length)
    ):
        raise DaggerSourceError("DAgger trajectory is incomplete or mixes identities")
    if first.source_domain == first.outer_target:
        raise DaggerSourceError("outer target is forbidden from DAgger relabeling")
    if (
        first.outer_target != authorization.outer_target
        or first.source_domain != authorization.labeled_domain
        or first.source_domain in authorization.fit_domains
    ):
        raise DaggerSourceError("DAgger trajectory is outside the authorized source world")
    indices = trajectory_quantile_indices(first.trajectory_length)
    return tuple(trajectory[index] for index in indices)


def equal_specimen_task_weights(
    rows: tuple[DaggerVisitedState, ...],
) -> tuple[float, ...]:
    if (
        type(rows) is not tuple
        or not rows
        or any(type(row) is not DaggerVisitedState for row in rows)
        or len({row.state_sha256 for row in rows}) != len(rows)
    ):
        raise DaggerSourceError("DAgger weighting rows are invalid")
    groups = Counter((row.specimen_sha256, row.task) for row in rows)
    group_count = len(groups)
    return tuple(
        1.0 / (group_count * groups[(row.specimen_sha256, row.task)])
        for row in rows
    )


__all__ = [
    "DAGGER_MAX_NEW_STATES",
    "REGISTERED_DAGGER_ITERATIONS",
    "DaggerSourceError",
    "DaggerVisitedState",
    "equal_specimen_task_weights",
    "select_source_relabels",
    "trajectory_quantile_indices",
]
