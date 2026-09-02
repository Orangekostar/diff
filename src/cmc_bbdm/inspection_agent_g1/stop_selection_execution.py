"""Source-only STOP threshold validation from frozen observable rollouts."""

from __future__ import annotations

import math

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

from .rollout import ClosedLoopTrajectory
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


__all__ = [
    "G1StopSelectionExecutionError",
    "evaluate_source_stop_validation_trajectory",
]
