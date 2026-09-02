"""Source-only STOP threshold validation from frozen observable rollouts."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from cmc_bbdm.inspection_agent.cai_assessor import state_scalars
from cmc_bbdm.inspection_agent.contracts import InspectionObservation, InspectionTask
from cmc_bbdm.inspection_agent.field_task import field_loss
from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    SourceBackgroundPrior,
    reconstruct_observation,
)
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .batch_rollout import G1BatchRolloutRequest, run_g1_closed_loop_batch
from .contracts import TaskTokenMode
from .formal import G1ObservableStateBuilder
from .g1 import (
    G1Protocol,
    G1Runtime,
    G1SourceDependencies,
    build_g1_world,
)
from .policy_training import TrainedObservablePolicy
from .rollout import ClosedLoopTrajectory
from .stop_execution import (
    G1FixedEndpointRecord,
    select_g1_source_fixed_reference,
)
from .stop_training import TrainedObservableStopPolicy
from .stopping_policy import SourceStopValidationTrajectory


class G1StopSelectionExecutionError(ValueError):
    """Raised when STOP selection opens truth before a fixed source rollout."""


def _frozen_rollout_observations(
    world: CausalInspectionWorld,
    trajectory: ClosedLoopTrajectory,
) -> tuple[InspectionObservation, ...]:
    current = world.replay(trajectory.action_history[:8])
    observations = [current]
    for step, action in zip(
        trajectory.steps,
        trajectory.action_history[8:],
        strict=True,
    ):
        if step.observation_sha256 != current.state_sha256:
            raise G1StopSelectionExecutionError(
                "source STOP rollout observation hash changed"
            )
        current = world.step(current, action)
        observations.append(current)
    if current.state_sha256 != trajectory.final_observation_sha256:
        raise G1StopSelectionExecutionError(
            "source STOP rollout final observation changed"
        )
    return tuple(observations)


def evaluate_source_stop_validation_trajectory(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    trajectory: ClosedLoopTrajectory,
    prior: SourceBackgroundPrior,
    *,
    assessor: object,
    encoder: object,
    outer_target: str,
    source_domain: str,
    full_scan: np.ndarray,
    true_cai: float,
    reference_true_loss: float,
) -> SourceStopValidationTrajectory:
    image = np.asarray(full_scan)
    target = float(true_cai)
    reference = float(reference_true_loss)
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(trajectory) is not ClosedLoopTrajectory
        or type(prior) is not SourceBackgroundPrior
        or not callable(getattr(assessor, "predict", None))
        or not callable(getattr(encoder, "encode", None))
        or type(outer_target) is not str
        or not outer_target
        or type(source_domain) is not str
        or not source_domain
        or source_domain == outer_target
        or trajectory.target_domain != source_domain
        or trajectory.task is not world.reset().task
        or trajectory.stop_threshold is not None
        or trajectory.stopped
        or len(trajectory.steps) != len(trajectory.action_history) - 8
        or prior.outer_domain != outer_target
        or source_domain in prior.source_domains
        or image.dtype != np.uint8
        or image.shape != (*grid.native_shape, 3)
        or not math.isfinite(target)
        or not math.isfinite(reference)
        or reference < 0.0
    ):
        raise G1StopSelectionExecutionError(
            "source STOP validation trajectory request is invalid"
        )
    observations = _frozen_rollout_observations(world, trajectory)
    reconstructions = tuple(
        reconstruct_observation(observation, grid, prior)
        for observation in observations
    )
    if trajectory.task is InspectionTask.FIELD:
        losses = tuple(field_loss(image, row.image) for row in reconstructions)
    elif trajectory.task is InspectionTask.CAI:
        embeddings = np.asarray(
            encoder.encode(tuple(row.image for row in reconstructions)),
            dtype=np.float64,
        )
        scalars = np.asarray(
            [state_scalars(observation) for observation in observations],
            dtype=np.float64,
        )
        predictions = np.asarray(
            assessor.predict(embeddings, scalars),
            dtype=np.float64,
        )
        if (
            embeddings.shape != (len(observations), 512)
            or predictions.shape != (len(observations),)
            or not np.all(np.isfinite(embeddings))
            or not np.all(np.isfinite(predictions))
        ):
            raise G1StopSelectionExecutionError(
                "source STOP CAI evaluation is invalid"
            )
        losses = tuple(abs(target - float(value)) for value in predictions)
    else:
        raise G1StopSelectionExecutionError("source STOP task changed")
    probabilities = tuple(
        step.scores.stop_probability for step in trajectory.steps
    ) + (0.0,)
    return SourceStopValidationTrajectory(
        outer_target=outer_target,
        source_domain=source_domain,
        specimen_sha256=trajectory.specimen_sha256,
        task=trajectory.task,
        budgets=tuple(value.effective_budget for value in observations),
        stop_probabilities=probabilities,
        true_task_losses=losses,
        reference_true_loss=reference,
        endpoint_budget=0.25,
    )


def materialize_g1_source_stop_validation_trajectories(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    action_policy: TrainedObservablePolicy,
    stop_policy: TrainedObservableStopPolicy,
    fixed_endpoint_records: tuple[G1FixedEndpointRecord, ...],
    progress: Callable[[str], None] | None = None,
) -> tuple[SourceStopValidationTrajectory, ...]:
    roster = getattr(dependencies, "roster", None)
    action_audit = getattr(action_policy, "audit", None)
    stop_audit = getattr(stop_policy, "audit", None)
    hyperparameters = getattr(action_policy, "hyperparameters", None)
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or runtime.domain_order != protocol.domain_order
        or type(dependencies) is not G1SourceDependencies
        or not callable(getattr(encoder, "encode", None))
        or type(action_policy) is not TrainedObservablePolicy
        or type(stop_policy) is not TrainedObservableStopPolicy
        or not callable(getattr(stop_policy, "score_batch", None))
        or type(fixed_endpoint_records) is not tuple
        or not fixed_endpoint_records
        or any(
            type(row) is not G1FixedEndpointRecord
            or row.outer_target != getattr(roster, "outer_target", None)
            for row in fixed_endpoint_records
        )
        or getattr(action_audit, "outer_target", None)
        != getattr(roster, "outer_target", None)
        or getattr(action_audit, "validation_domain", None)
        != getattr(roster, "labeled_domain", None)
        or tuple(getattr(action_audit, "fit_domains", ()))
        != tuple(getattr(roster, "fit_domains", ()))
        or getattr(stop_audit, "outer_target", None)
        != getattr(roster, "outer_target", None)
        or getattr(stop_audit, "validation_domain", None)
        != getattr(roster, "labeled_domain", None)
        or tuple(getattr(stop_audit, "fit_domains", ()))
        != tuple(getattr(roster, "fit_domains", ()))
        or getattr(stop_policy, "base_action_model_sha256", None)
        != action_policy.model_state_sha256
        or getattr(hyperparameters, "task_token_mode", None)
        is not TaskTokenMode.CORRECT
        or (progress is not None and not callable(progress))
    ):
        raise G1StopSelectionExecutionError(
            "source STOP validation materialization request is invalid"
        )
    outer_target = roster.outer_target
    source_domain = roster.labeled_domain
    specimen_ids = tuple(
        specimen
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == source_domain
    )
    if len(specimen_ids) != int(protocol.domain_counts[source_domain]):
        raise G1StopSelectionExecutionError(
            "source STOP validation specimen roster changed"
        )
    references = {
        task: select_g1_source_fixed_reference(
            dependencies.authorization,
            fixed_endpoint_records,
            task=task,
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    }
    expected_sha = {
        runtime.specimen_sha256(source_domain, specimen) for specimen in specimen_ids
    }
    endpoint_map = {
        (row.specimen_sha256, row.task): row
        for row in fixed_endpoint_records
        if row.source_domain == source_domain
        and row.method == references[row.task].method
    }
    if (
        len(endpoint_map) != 2 * len(specimen_ids)
        or {key[0] for key in endpoint_map} != expected_sha
        or any(
            row.dependency_sha256 != dependencies.state_sha256
            or row.fit_domains != roster.fit_domains
            for row in endpoint_map.values()
        )
    ):
        raise G1StopSelectionExecutionError(
            "source STOP fixed endpoint bridge changed"
        )
    requests = []
    contexts = []
    prior = dependencies.prior_fit.prior
    assessor = dependencies.assessor_fit.assessor
    for specimen in specimen_ids:
        specimen_sha = runtime.specimen_sha256(source_domain, specimen)
        truth = runtime.mavis.source_teacher_view(specimen)
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            world, grid, surface = build_g1_world(
                runtime,
                dataset_id=source_domain,
                specimen_id=specimen,
                task=task,
                endpoint_budget=protocol.endpoint_budget,
            )
            builder = G1ObservableStateBuilder(
                grid=grid,
                surface_hypothesis=surface.hypothesis,
                prior=prior,
                assessor=assessor,
                encoder=encoder,
                cai_context_mode=hyperparameters.cai_context_mode,
                task_token_mode=hyperparameters.task_token_mode,
            )
            requests.append(
                G1BatchRolloutRequest(
                    world=world,
                    grid=grid,
                    target_domain=source_domain,
                    specimen_sha256=specimen_sha,
                    state_builder=builder,
                )
            )
            contexts.append(
                (
                    world,
                    grid,
                    truth,
                    endpoint_map[(specimen_sha, task)].task_loss,
                )
            )
    if progress is not None:
        progress(f"G1 source STOP rollout {outer_target}/{source_domain}")
    trajectories = run_g1_closed_loop_batch(
        tuple(requests),
        actor=stop_policy,
        stop_threshold=None,
    )
    if (
        type(trajectories) is not tuple
        or len(trajectories) != len(contexts)
        or any(type(row) is not ClosedLoopTrajectory for row in trajectories)
    ):
        raise G1StopSelectionExecutionError(
            "source STOP rollout result roster changed"
        )
    output = tuple(
        evaluate_source_stop_validation_trajectory(
            world,
            grid,
            trajectory,
            prior,
            assessor=assessor,
            encoder=encoder,
            outer_target=outer_target,
            source_domain=source_domain,
            full_scan=truth.full_scan,
            true_cai=truth.true_cai,
            reference_true_loss=reference_loss,
        )
        for trajectory, (world, grid, truth, reference_loss) in zip(
            trajectories,
            contexts,
            strict=True,
        )
    )
    if progress is not None:
        progress(
            f"G1 source STOP validation complete {outer_target}/{source_domain}: "
            f"{len(output)} trajectories"
        )
    return output


__all__ = [
    "G1StopSelectionExecutionError",
    "evaluate_source_stop_validation_trajectory",
    "materialize_g1_source_stop_validation_trajectories",
]
