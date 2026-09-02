"""Fold-safe source-world rollout and privileged DAgger relabel execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .contracts import CAIContextMode, TaskTokenMode
from .dagger import (
    REGISTERED_DAGGER_ITERATIONS,
    DaggerVisitedState,
    trajectory_quantile_indices,
)
from .formal import G1ObservableStateBuilder
from .policy_training import G1PolicyTrainingExample
from .rollout import run_closed_loop
from .teacher import (
    SourceTeacherAuthorization,
    cai_teacher_label,
    field_teacher_label,
    validate_source_teacher_dependencies,
)
from .teacher_bank import G1TeacherBankRecord


class G1DaggerExecutionError(ValueError):
    """Raised when a DAgger rollout or relabel crosses its source fold."""


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
class G1DaggerRelabelBatch:
    outer_target: str
    source_domain: str
    specimen_sha256: str
    task: InspectionTask
    iteration: int
    actor_model_sha256: str
    trajectory_sha256: str
    trajectory_length: int
    records: tuple[G1TeacherBankRecord, ...]
    visited_states: tuple[DaggerVisitedState, ...]
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        indices = trajectory_quantile_indices(self.trajectory_length)
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domain) is not str
            or not self.source_domain
            or self.source_domain == self.outer_target
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or self.iteration not in REGISTERED_DAGGER_ITERATIONS[1:]
            or not _valid_sha256(self.actor_model_sha256)
            or not _valid_sha256(self.trajectory_sha256)
            or type(self.records) is not tuple
            or type(self.visited_states) is not tuple
            or len(self.records) != len(indices)
            or len(self.visited_states) != len(indices)
            or any(type(row) is not G1TeacherBankRecord for row in self.records)
            or any(type(row) is not DaggerVisitedState for row in self.visited_states)
            or tuple(row.trajectory_index for row in self.visited_states) != indices
            or any(
                row.example.outer_target != self.outer_target
                or row.example.source_domain != self.source_domain
                or row.example.specimen_sha256 != self.specimen_sha256
                or row.example.task is not self.task
                or row.example.dagger_iteration != self.iteration
                for row in self.records
            )
            or any(
                row.outer_target != self.outer_target
                or row.source_domain != self.source_domain
                or row.specimen_sha256 != self.specimen_sha256
                or row.task is not self.task
                or row.iteration != self.iteration
                or row.trajectory_length != self.trajectory_length
                for row in self.visited_states
            )
            or any(
                record.source_state_sha256 != visited.state_sha256
                for record, visited in zip(
                    self.records, self.visited_states, strict=True
                )
            )
        ):
            raise G1DaggerExecutionError("DAgger relabel batch is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-dagger-relabel-batch",
                    "outer_target": self.outer_target,
                    "source_domain": self.source_domain,
                    "specimen_sha256": self.specimen_sha256,
                    "task": self.task.value,
                    "iteration": self.iteration,
                    "actor_model_sha256": self.actor_model_sha256,
                    "trajectory_sha256": self.trajectory_sha256,
                    "trajectory_length": self.trajectory_length,
                    "records": tuple(row.state_sha256 for row in self.records),
                    "visited": tuple(
                        row.state_sha256 for row in self.visited_states
                    ),
                }
            ),
        )


def materialize_g1_dagger_relabels_for_world(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    surface_hypothesis: SurfaceHypothesis,
    prior: SourceBackgroundPrior,
    authorization: SourceTeacherAuthorization,
    *,
    assessor: object,
    encoder: object,
    actor: object,
    outer_target: str,
    source_domain: str,
    specimen_sha256: str,
    iteration: int,
    full_scan: np.ndarray,
    true_cai: float,
) -> G1DaggerRelabelBatch:
    audit = getattr(actor, "audit", None)
    hyperparameters = getattr(actor, "hyperparameters", None)
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(prior) is not SourceBackgroundPrior
        or type(authorization) is not SourceTeacherAuthorization
        or not callable(getattr(assessor, "predict", None))
        or not _valid_sha256(getattr(assessor, "model_state_sha256", None))
        or not callable(getattr(encoder, "encode", None))
        or not callable(actor)
        or not _valid_sha256(getattr(actor, "model_state_sha256", None))
        or outer_target != authorization.outer_target
        or source_domain != authorization.labeled_domain
        or source_domain == outer_target
        or not _valid_sha256(specimen_sha256)
        or iteration not in REGISTERED_DAGGER_ITERATIONS[1:]
        or getattr(audit, "outer_target", None) != outer_target
        or getattr(audit, "validation_domain", None) != source_domain
        or tuple(getattr(audit, "fit_domains", ())) != authorization.fit_domains
        or getattr(hyperparameters, "dagger_iterations", None) != iteration - 1
        or getattr(hyperparameters, "cai_context_mode", None)
        not in (
            CAIContextMode.TASK_SPECIFIC_MASKED,
            CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        )
        or getattr(hyperparameters, "task_token_mode", None)
        is not TaskTokenMode.CORRECT
    ):
        raise G1DaggerExecutionError("DAgger world relabel request is invalid")
    validate_source_teacher_dependencies(
        authorization,
        prior,
        assessor=assessor,
    )
    actor_state_builder = G1ObservableStateBuilder(
        grid=grid,
        surface_hypothesis=surface_hypothesis,
        prior=prior,
        assessor=assessor,
        encoder=encoder,
        cai_context_mode=hyperparameters.cai_context_mode,
        task_token_mode=hyperparameters.task_token_mode,
    )
    canonical_state_builder = G1ObservableStateBuilder(
        grid=grid,
        surface_hypothesis=surface_hypothesis,
        prior=prior,
        assessor=assessor,
        encoder=encoder,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
    )
    trajectory = run_closed_loop(
        world,
        grid,
        target_domain=source_domain,
        specimen_sha256=specimen_sha256,
        state_builder=actor_state_builder,
        actor=actor,
        stop_threshold=None,
    )
    if trajectory.stopped or not trajectory.steps:
        raise G1DaggerExecutionError("DAgger actor trajectory is incomplete")
    indices = trajectory_quantile_indices(len(trajectory.steps))
    records = []
    visited = []
    for trajectory_index in indices:
        observation = world.replay(
            trajectory.action_history[: 8 + trajectory_index]
        )
        actor_policy_state = actor_state_builder(observation)
        step = trajectory.steps[trajectory_index]
        if (
            step.observation_sha256 != observation.state_sha256
            or step.policy_state_sha256 != actor_policy_state.state_sha256
        ):
            raise G1DaggerExecutionError("DAgger trajectory replay changed")
        policy_state = canonical_state_builder(observation)
        if observation.task is not trajectory.task:
            raise G1DaggerExecutionError("DAgger replay task changed")
        if trajectory.task is InspectionTask.FIELD:
            label = field_teacher_label(
                observation,
                grid,
                prior,
                surface_hypothesis,
                authorization,
                full_scan=full_scan,
                policy_state_sha256=policy_state.state_sha256,
            )
        elif trajectory.task is InspectionTask.CAI:
            label = cai_teacher_label(
                observation,
                grid,
                prior,
                surface_hypothesis,
                authorization,
                full_scan=full_scan,
                true_cai=true_cai,
                assessor=assessor,
                encoder=encoder,
                policy_state_sha256=policy_state.state_sha256,
            )
        else:
            raise G1DaggerExecutionError("DAgger task is invalid")
        example = G1PolicyTrainingExample(
            outer_target=outer_target,
            source_domain=source_domain,
            specimen_sha256=specimen_sha256,
            task=trajectory.task,
            dagger_iteration=iteration,
            policy_state=policy_state,
            teacher_label=label,
        )
        visited_state = DaggerVisitedState(
            outer_target=outer_target,
            source_domain=source_domain,
            specimen_sha256=specimen_sha256,
            task=trajectory.task,
            iteration=iteration,
            trajectory_index=trajectory_index,
            trajectory_length=len(trajectory.steps),
            observation_sha256=observation.state_sha256,
            policy_state_sha256=policy_state.state_sha256,
            teacher_label_sha256=label.state_sha256,
        )
        records.append(
            G1TeacherBankRecord(
                example=example,
                fit_domains=authorization.fit_domains,
                state_source="DAGGER_ACTOR_VISITED",
                source_state_sha256=visited_state.state_sha256,
                prior_sha256=prior.state_sha256,
                assessor_sha256=assessor.model_state_sha256,
            )
        )
        visited.append(visited_state)
    return G1DaggerRelabelBatch(
        outer_target=outer_target,
        source_domain=source_domain,
        specimen_sha256=specimen_sha256,
        task=trajectory.task,
        iteration=iteration,
        actor_model_sha256=actor.model_state_sha256,
        trajectory_sha256=trajectory.state_sha256,
        trajectory_length=len(trajectory.steps),
        records=tuple(records),
        visited_states=tuple(visited),
    )


__all__ = [
    "G1DaggerExecutionError",
    "G1DaggerRelabelBatch",
    "materialize_g1_dagger_relabels_for_world",
]
