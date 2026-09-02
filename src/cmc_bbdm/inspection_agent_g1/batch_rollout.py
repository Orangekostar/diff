"""Batched observable-state inference with unchanged causal rollout semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .contracts import ACTION_SLOT_COUNT, G1PolicyState
from .features import canonical_action_from_slot
from .formal import G1ObservableStateBuilder, build_g1_observable_states
from .rollout import ClosedLoopTrajectory, ObservablePolicyScores, RolloutStep
from .stopping_policy import REGISTERED_STOP_THRESHOLDS
from .warm_start import PRIMARY_WARM_START_CELLS


class G1BatchRolloutError(ValueError):
    """Raised when batched inference changes a causal rollout contract."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


@dataclass(frozen=True, slots=True)
class G1BatchRolloutRequest:
    world: CausalInspectionWorld
    grid: AcquisitionGrid
    target_domain: str
    specimen_sha256: str
    state_builder: G1ObservableStateBuilder

    def __post_init__(self) -> None:
        if (
            type(self.world) is not CausalInspectionWorld
            or type(self.grid) is not AcquisitionGrid
            or type(self.target_domain) is not str
            or not self.target_domain
            or not _valid_sha256(self.specimen_sha256)
            or type(self.state_builder) is not G1ObservableStateBuilder
            or self.state_builder.grid.state_sha256 != self.grid.state_sha256
        ):
            raise G1BatchRolloutError("batched rollout request is invalid")


def _validate_scores(
    state: G1PolicyState,
    scores: ObservablePolicyScores,
) -> None:
    mask = state.legal_action_mask
    if (
        type(scores) is not ObservablePolicyScores
        or scores.policy_state_sha256 != state.state_sha256
        or not np.all(np.isfinite(scores.action_logits[mask]))
        or not np.all(np.isneginf(scores.action_logits[~mask]))
    ):
        raise G1BatchRolloutError("batched actor scores changed the legal mask")


def run_g1_closed_loop_batch(
    requests: tuple[G1BatchRolloutRequest, ...],
    *,
    actor: object,
    stop_threshold: float | None,
) -> tuple[ClosedLoopTrajectory, ...]:
    score_batch = getattr(actor, "score_batch", None)
    if (
        type(requests) is not tuple
        or not requests
        or any(type(row) is not G1BatchRolloutRequest for row in requests)
        or len(
            {
                (
                    row.target_domain,
                    row.specimen_sha256,
                    row.world.reset().task.value,
                )
                for row in requests
            }
        )
        != len(requests)
        or not callable(actor)
        or not callable(score_batch)
        or (
            stop_threshold is not None
            and float(stop_threshold) not in REGISTERED_STOP_THRESHOLDS
        )
    ):
        raise G1BatchRolloutError("batched closed-loop request is invalid")
    warm_actions = tuple(
        canonical_action_from_slot(cell) for cell in PRIMARY_WARM_START_CELLS
    )
    observations = [row.world.replay(warm_actions) for row in requests]
    steps: list[list[RolloutStep]] = [[] for _row in requests]
    model_sha256: list[str | None] = [None for _row in requests]
    stopped = [False for _row in requests]
    termination_reasons = ["NO_LEGAL_ACTION" for _row in requests]
    active = list(range(len(requests)))
    for _round in range(ACTION_SLOT_COUNT):
        states = build_g1_observable_states(
            tuple(requests[index].state_builder for index in active),
            tuple(observations[index] for index in active),
        )
        score_indices = []
        score_states = []
        for index, state in zip(active, states, strict=True):
            request = requests[index]
            observation = observations[index]
            if (
                state.observation_sha256 != observation.state_sha256
                or state.grid_sha256 != request.grid.state_sha256
                or state.task is not observation.task
            ):
                raise G1BatchRolloutError(
                    "batched state builder changed the current observation"
                )
            if np.any(state.legal_action_mask):
                score_indices.append(index)
                score_states.append(state)
        if not score_indices:
            active = []
            break
        scores = score_batch(tuple(score_states))
        if type(scores) is not tuple or len(scores) != len(score_states):
            raise G1BatchRolloutError("batched actor output roster changed")
        next_active = []
        for index, state, score in zip(
            score_indices,
            score_states,
            scores,
            strict=True,
        ):
            _validate_scores(state, score)
            if model_sha256[index] is None:
                model_sha256[index] = score.model_sha256
            elif score.model_sha256 != model_sha256[index]:
                raise G1BatchRolloutError(
                    "actor model identity changed during batched rollout"
                )
            step_index = len(steps[index])
            observation = observations[index]
            if (
                stop_threshold is not None
                and score.stop_probability >= float(stop_threshold)
            ):
                steps[index].append(
                    RolloutStep(
                        step_index=step_index,
                        observation_sha256=observation.state_sha256,
                        policy_state_sha256=state.state_sha256,
                        scores=score,
                        selected_slot=None,
                    )
                )
                stopped[index] = True
                termination_reasons[index] = "STOP"
                continue
            selected_slot = int(np.argmax(score.action_logits))
            if not state.legal_action_mask[selected_slot]:
                raise G1BatchRolloutError("batched actor selected an illegal action")
            steps[index].append(
                RolloutStep(
                    step_index=step_index,
                    observation_sha256=observation.state_sha256,
                    policy_state_sha256=state.state_sha256,
                    scores=score,
                    selected_slot=selected_slot,
                )
            )
            observations[index] = requests[index].world.step(
                observation,
                canonical_action_from_slot(selected_slot),
            )
            next_active.append(index)
        active = next_active
        if not active:
            break
    else:
        raise G1BatchRolloutError(
            "batched closed-loop rollout exceeded the finite action roster"
        )
    output = []
    for index, (request, observation) in enumerate(
        zip(requests, observations, strict=True)
    ):
        if model_sha256[index] is None:
            raise G1BatchRolloutError("batched rollout produced no actor score")
        if observation.task not in (InspectionTask.FIELD, InspectionTask.CAI):
            raise G1BatchRolloutError("batched rollout task changed")
        output.append(
            ClosedLoopTrajectory(
                target_domain=request.target_domain,
                specimen_sha256=request.specimen_sha256,
                task=observation.task,
                model_sha256=model_sha256[index],
                stop_threshold=stop_threshold,
                stopped=stopped[index],
                termination_reason=termination_reasons[index],
                action_history=observation.action_history,
                steps=tuple(steps[index]),
                final_observation_sha256=observation.state_sha256,
                acquired_positions=observation.acquired_positions,
                acquired_values=observation.measurement_values,
                native_count=observation.native_count,
                effective_budget=observation.effective_budget,
            )
        )
    return tuple(output)


__all__ = [
    "G1BatchRolloutError",
    "G1BatchRolloutRequest",
    "run_g1_closed_loop_batch",
]
