"""Privileged one-step teachers for G0 opportunity measurement."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .cai_assessor import state_scalars
from .contracts import InspectionDecision, InspectionObservation, InspectionTask
from .field_task import (
    InternalSignalSaliency,
    field_loss,
    internal_signal_saliency,
    signal_capture,
)
from .generalized_reconstruction import (
    SourceBackgroundPrior,
    reconstruct_observation,
)
from .state import (
    GeneralizedMeasurementState,
    InspectionCellAction,
    action_added_positions,
    apply_action,
    fitting_actions,
    measurement_mask,
)
from .world import CausalInspectionWorld


class InspectionOracleError(ValueError):
    """Raised when a privileged teacher query or trajectory is invalid."""


class ReconstructionEncoder(Protocol):
    def encode(self, images: object) -> object: ...


class CAIStatePredictor(Protocol):
    model_state_sha256: str

    def predict(self, embeddings: object, scalars: object) -> object: ...


@dataclass(frozen=True, slots=True)
class OracleCandidateScore:
    action: InspectionCellAction
    exact_added_cost: int
    raw_value: float
    objective_value: float
    task_loss_after: float
    candidate_state_sha256: str


@dataclass(frozen=True, slots=True)
class OracleSelection:
    action: InspectionCellAction
    exact_added_cost: int
    raw_value: float
    objective_value: float
    task_loss_after: float
    selected_index: int
    candidates: tuple[OracleCandidateScore, ...]


@dataclass(frozen=True, slots=True)
class OracleTrajectoryStep:
    step: int
    task: InspectionTask
    decision: InspectionDecision
    action: InspectionCellAction
    exact_cost_before: int
    exact_cost_after: int
    budget_before: float
    budget_after: float
    task_loss_before: float
    task_loss_after: float
    teacher_value: float
    objective_value: float
    candidates: tuple[OracleCandidateScore, ...]
    state_sha256_before: str
    state_sha256_after: str
    stop_status: bool


@dataclass(frozen=True, slots=True)
class OracleTrajectory:
    task: InspectionTask
    method: str
    steps: tuple[OracleTrajectoryStep, ...]
    final_observation: InspectionObservation
    state_sha256: str


def _full_scan(value: object, shape: tuple[int, int]) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.shape != (*shape, 3)
    ):
        raise InspectionOracleError("privileged full scan is invalid")
    return value


def _select(candidates: tuple[OracleCandidateScore, ...]) -> OracleSelection:
    if not candidates:
        raise InspectionOracleError("oracle has no fitting candidate")
    selected_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index].objective_value,
            candidates[index].raw_value,
            -candidates[index].action.cell_index,
            -candidates[index].action.to_level,
        ),
    )
    selected = candidates[selected_index]
    return OracleSelection(
        action=selected.action,
        exact_added_cost=selected.exact_added_cost,
        raw_value=selected.raw_value,
        objective_value=selected.objective_value,
        task_loss_after=selected.task_loss_after,
        selected_index=selected_index,
        candidates=candidates,
    )


def _candidate_observation(
    observation: InspectionObservation,
    grid: AcquisitionGrid,
    action: InspectionCellAction,
    full_scan: np.ndarray,
) -> InspectionObservation:
    candidate = apply_action(grid, observation.measurement_state, action)
    mask = measurement_mask(grid, candidate)
    positions = np.argwhere(mask).astype("<i8", copy=False)
    values = full_scan[positions[:, 0], positions[:, 1]]
    return InspectionObservation(
        surface_rgb=observation.surface_rgb,
        surface_sha256=observation.surface_sha256,
        task=observation.task,
        native_shape=observation.native_shape,
        native_count=observation.native_count,
        grid_sha256=observation.grid_sha256,
        measurement_state=candidate,
        acquired_positions=positions,
        measurement_values=values,
        exact_acquired_count=len(positions),
        endpoint_budget=observation.endpoint_budget,
        action_history=(*observation.action_history, action),
    )


def choose_discovery_action(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
    *,
    full_scan: np.ndarray,
    checkpoint: float,
) -> OracleSelection:
    image = _full_scan(full_scan, grid.native_shape)
    saliency = internal_signal_saliency(image)
    current_mask = measurement_mask(grid, state)
    actions = tuple(
        action
        for action in fitting_actions(grid, state, checkpoint)
        if action.from_level == -1 and action.to_level == 0
    )
    rows = []
    for action in actions:
        added_positions = action_added_positions(grid, state, action)
        added = len(added_positions)
        if added == 0:
            continue
        added_mass = float(
            np.sum(
                saliency.pixel_mass[
                    added_positions[:, 0], added_positions[:, 1]
                ],
                dtype=np.float64,
            )
        )
        raw = 0.0 if saliency.total_mass == 0.0 else added_mass / saliency.total_mass
        candidate = apply_action(grid, state, action)
        if np.any(current_mask & ~measurement_mask(grid, candidate)):
            raise InspectionOracleError("discovery candidate removes evidence")
        rows.append(
            OracleCandidateScore(
                action=action,
                exact_added_cost=added,
                raw_value=raw,
                objective_value=raw / added,
                task_loss_after=float(
                    max(0.0, 1.0 - signal_capture(current_mask, saliency) - raw)
                ),
                candidate_state_sha256=candidate.state_sha256,
            )
        )
    return _select(tuple(rows))


def choose_field_action(
    observation: InspectionObservation,
    grid: AcquisitionGrid,
    prior: SourceBackgroundPrior,
    *,
    full_scan: np.ndarray,
    checkpoint: float,
) -> OracleSelection:
    image = _full_scan(full_scan, grid.native_shape)
    current = reconstruct_observation(observation, grid, prior)
    current_loss = field_loss(image, current.image)
    rows = []
    for action in fitting_actions(grid, observation.measurement_state, checkpoint):
        candidate_observation = _candidate_observation(
            observation,
            grid,
            action,
            image,
        )
        candidate = reconstruct_observation(candidate_observation, grid, prior)
        candidate_loss = field_loss(image, candidate.image)
        added = (
            candidate_observation.exact_acquired_count
            - observation.exact_acquired_count
        )
        if added == 0:
            continue
        raw = current_loss - candidate_loss
        rows.append(
            OracleCandidateScore(
                action=action,
                exact_added_cost=added,
                raw_value=raw,
                objective_value=raw / added,
                task_loss_after=candidate_loss,
                candidate_state_sha256=candidate_observation.state_sha256,
            )
        )
    return _select(tuple(rows))


def _encoded_predictions(
    observations: tuple[InspectionObservation, ...],
    grid: AcquisitionGrid,
    prior: SourceBackgroundPrior,
    assessor: CAIStatePredictor,
    encoder: ReconstructionEncoder,
) -> np.ndarray:
    if (
        not observations
        or not hasattr(encoder, "encode")
        or not hasattr(assessor, "predict")
        or not isinstance(getattr(assessor, "model_state_sha256", None), str)
        or len(assessor.model_state_sha256) != 64
    ):
        raise InspectionOracleError("CAI oracle interfaces are invalid")
    images = tuple(
        reconstruct_observation(observation, grid, prior).image
        for observation in observations
    )
    embeddings = np.asarray(encoder.encode(images), dtype=np.float64)
    scalars = np.asarray(
        [state_scalars(observation) for observation in observations],
        dtype=np.float64,
    )
    predictions = np.asarray(assessor.predict(embeddings, scalars), dtype=np.float64)
    if (
        embeddings.shape != (len(observations), 512)
        or predictions.shape != (len(observations),)
        or not np.all(np.isfinite(embeddings))
        or not np.all(np.isfinite(predictions))
    ):
        raise InspectionOracleError("CAI oracle prediction batch is invalid")
    return predictions


def choose_cai_action(
    observation: InspectionObservation,
    grid: AcquisitionGrid,
    prior: SourceBackgroundPrior,
    *,
    full_scan: np.ndarray,
    true_cai: float,
    assessor: CAIStatePredictor,
    encoder: ReconstructionEncoder,
    checkpoint: float,
) -> OracleSelection:
    _full_scan(full_scan, grid.native_shape)
    target = float(true_cai)
    if (
        type(observation) is not InspectionObservation
        or observation.task is not InspectionTask.CAI
        or not math.isfinite(target)
    ):
        raise InspectionOracleError("CAI oracle request is invalid")
    actions: list[InspectionCellAction] = []
    candidates: list[InspectionObservation] = []
    for action in fitting_actions(grid, observation.measurement_state, checkpoint):
        candidate = _candidate_observation(observation, grid, action, full_scan)
        if candidate.exact_acquired_count == observation.exact_acquired_count:
            continue
        actions.append(action)
        candidates.append(candidate)
    predictions = _encoded_predictions(
        (observation, *candidates),
        grid,
        prior,
        assessor,
        encoder,
    )
    current_error = abs(target - float(predictions[0]))
    rows = []
    for action, candidate, prediction in zip(
        actions,
        candidates,
        predictions[1:],
        strict=True,
    ):
        candidate_error = abs(target - float(prediction))
        added = candidate.exact_acquired_count - observation.exact_acquired_count
        raw = current_error - candidate_error
        rows.append(
            OracleCandidateScore(
                action=action,
                exact_added_cost=added,
                raw_value=raw,
                objective_value=raw / added,
                task_loss_after=candidate_error,
                candidate_state_sha256=candidate.state_sha256,
            )
        )
    return _select(tuple(rows))


def _decision(
    action: InspectionCellAction,
    focus_cells: frozenset[int],
) -> InspectionDecision:
    if action.from_level >= 0:
        return InspectionDecision.REFINE
    return (
        InspectionDecision.FOCUS
        if action.cell_index in focus_cells
        else InspectionDecision.BROADEN
    )


def _trajectory_hash(
    task: InspectionTask,
    method: str,
    steps: tuple[OracleTrajectoryStep, ...],
    final: InspectionObservation,
) -> str:
    payload = {
        "schema": 1,
        "task": task.value,
        "method": method,
        "final": final.state_sha256,
        "steps": [
            {
                "action": (
                    step.action.cell_index,
                    step.action.from_level,
                    step.action.to_level,
                ),
                "before": step.state_sha256_before,
                "after": step.state_sha256_after,
                "teacher_value": step.teacher_value,
                "objective_value": step.objective_value,
                "candidates": [
                    (
                        candidate.action.cell_index,
                        candidate.action.from_level,
                        candidate.action.to_level,
                        candidate.exact_added_cost,
                        candidate.raw_value,
                        candidate.objective_value,
                        candidate.task_loss_after,
                    )
                    for candidate in step.candidates
                ],
            }
            for step in steps
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def run_discovery_oracle(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    *,
    full_scan: np.ndarray,
    surface_hypothesis_cells: tuple[int, ...],
) -> OracleTrajectory:
    image = _full_scan(full_scan, grid.native_shape)
    if type(world) is not CausalInspectionWorld or any(
        type(cell) is not int or not 0 <= cell < 64
        for cell in surface_hypothesis_cells
    ):
        raise InspectionOracleError("discovery trajectory request is invalid")
    saliency: InternalSignalSaliency = internal_signal_saliency(image)
    focus = frozenset(surface_hypothesis_cells)
    current = world.reset()
    target_mask = measurement_mask(
        grid,
        GeneralizedMeasurementState(grid.state_sha256, (0,) * 64),
    )
    steps: list[OracleTrajectoryStep] = []
    while not np.array_equal(
        measurement_mask(grid, current.measurement_state),
        target_mask,
    ):
        before_mask = measurement_mask(grid, current.measurement_state)
        before_capture = signal_capture(before_mask, saliency)
        selection = choose_discovery_action(
            grid,
            current.measurement_state,
            full_scan=image,
            checkpoint=current.endpoint_budget,
        )
        next_observation = world.step(current, selection.action)
        after_capture = signal_capture(
            measurement_mask(grid, next_observation.measurement_state),
            saliency,
        )
        steps.append(
            OracleTrajectoryStep(
                step=len(steps),
                task=InspectionTask.DISCOVERY,
                decision=_decision(selection.action, focus),
                action=selection.action,
                exact_cost_before=current.exact_acquired_count,
                exact_cost_after=next_observation.exact_acquired_count,
                budget_before=current.effective_budget,
                budget_after=next_observation.effective_budget,
                task_loss_before=1.0 - before_capture,
                task_loss_after=1.0 - after_capture,
                teacher_value=selection.raw_value,
                objective_value=selection.objective_value,
                candidates=selection.candidates,
                state_sha256_before=current.state_sha256,
                state_sha256_after=next_observation.state_sha256,
                stop_status=np.array_equal(
                    measurement_mask(grid, next_observation.measurement_state),
                    target_mask,
                ),
            )
        )
        current = next_observation
    frozen_steps = tuple(steps)
    return OracleTrajectory(
        task=InspectionTask.DISCOVERY,
        method="ORACLE_DISCOVERY",
        steps=frozen_steps,
        final_observation=current,
        state_sha256=_trajectory_hash(
            InspectionTask.DISCOVERY,
            "ORACLE_DISCOVERY",
            frozen_steps,
            current,
        ),
    )


def _has_positive_cost_action(
    grid: AcquisitionGrid,
    observation: InspectionObservation,
) -> bool:
    return any(
        len(action_added_positions(grid, observation.measurement_state, action)) > 0
        for action in fitting_actions(
            grid,
            observation.measurement_state,
            observation.endpoint_budget,
        )
    )


def run_field_oracle(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    prior: SourceBackgroundPrior,
    *,
    full_scan: np.ndarray,
    surface_hypothesis_cells: tuple[int, ...],
) -> OracleTrajectory:
    image = _full_scan(full_scan, grid.native_shape)
    focus = frozenset(surface_hypothesis_cells)
    current = world.reset()
    if current.task is not InspectionTask.FIELD:
        raise InspectionOracleError("FIELD oracle requires the FIELD task")
    steps: list[OracleTrajectoryStep] = []
    while _has_positive_cost_action(grid, current):
        before_loss = field_loss(
            image,
            reconstruct_observation(current, grid, prior).image,
        )
        selection = choose_field_action(
            current,
            grid,
            prior,
            full_scan=image,
            checkpoint=current.endpoint_budget,
        )
        next_observation = world.step(current, selection.action)
        after_loss = field_loss(
            image,
            reconstruct_observation(next_observation, grid, prior).image,
        )
        if not math.isclose(
            after_loss,
            selection.task_loss_after,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise InspectionOracleError("FIELD selected loss changed")
        steps.append(
            OracleTrajectoryStep(
                step=len(steps),
                task=InspectionTask.FIELD,
                decision=_decision(selection.action, focus),
                action=selection.action,
                exact_cost_before=current.exact_acquired_count,
                exact_cost_after=next_observation.exact_acquired_count,
                budget_before=current.effective_budget,
                budget_after=next_observation.effective_budget,
                task_loss_before=before_loss,
                task_loss_after=after_loss,
                teacher_value=selection.raw_value,
                objective_value=selection.objective_value,
                candidates=selection.candidates,
                state_sha256_before=current.state_sha256,
                state_sha256_after=next_observation.state_sha256,
                stop_status=not _has_positive_cost_action(grid, next_observation),
            )
        )
        current = next_observation
        if len(steps) > 192:
            raise InspectionOracleError("FIELD oracle exceeded the finite state depth")
    frozen_steps = tuple(steps)
    return OracleTrajectory(
        task=InspectionTask.FIELD,
        method="ORACLE_FIELD",
        steps=frozen_steps,
        final_observation=current,
        state_sha256=_trajectory_hash(
            InspectionTask.FIELD,
            "ORACLE_FIELD",
            frozen_steps,
            current,
        ),
    )


def run_cai_oracle(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    prior: SourceBackgroundPrior,
    *,
    full_scan: np.ndarray,
    true_cai: float,
    assessor: CAIStatePredictor,
    encoder: ReconstructionEncoder,
    surface_hypothesis_cells: tuple[int, ...],
) -> OracleTrajectory:
    image = _full_scan(full_scan, grid.native_shape)
    target = float(true_cai)
    current = world.reset()
    if current.task is not InspectionTask.CAI or not math.isfinite(target):
        raise InspectionOracleError("CAI oracle requires an authorized CAI task")
    focus = frozenset(surface_hypothesis_cells)
    steps: list[OracleTrajectoryStep] = []
    while _has_positive_cost_action(grid, current):
        before_prediction = float(
            _encoded_predictions((current,), grid, prior, assessor, encoder)[0]
        )
        before_loss = abs(target - before_prediction)
        selection = choose_cai_action(
            current,
            grid,
            prior,
            full_scan=image,
            true_cai=target,
            assessor=assessor,
            encoder=encoder,
            checkpoint=current.endpoint_budget,
        )
        next_observation = world.step(current, selection.action)
        steps.append(
            OracleTrajectoryStep(
                step=len(steps),
                task=InspectionTask.CAI,
                decision=_decision(selection.action, focus),
                action=selection.action,
                exact_cost_before=current.exact_acquired_count,
                exact_cost_after=next_observation.exact_acquired_count,
                budget_before=current.effective_budget,
                budget_after=next_observation.effective_budget,
                task_loss_before=before_loss,
                task_loss_after=selection.task_loss_after,
                teacher_value=selection.raw_value,
                objective_value=selection.objective_value,
                candidates=selection.candidates,
                state_sha256_before=current.state_sha256,
                state_sha256_after=next_observation.state_sha256,
                stop_status=not _has_positive_cost_action(grid, next_observation),
            )
        )
        current = next_observation
        if len(steps) > 192:
            raise InspectionOracleError("CAI oracle exceeded the finite state depth")
    frozen_steps = tuple(steps)
    return OracleTrajectory(
        task=InspectionTask.CAI,
        method="ORACLE_CAI",
        steps=frozen_steps,
        final_observation=current,
        state_sha256=_trajectory_hash(
            InspectionTask.CAI,
            "ORACLE_CAI",
            frozen_steps,
            current,
        ),
    )


__all__ = [
    "InspectionOracleError",
    "OracleCandidateScore",
    "OracleSelection",
    "OracleTrajectory",
    "OracleTrajectoryStep",
    "choose_cai_action",
    "choose_discovery_action",
    "choose_field_action",
    "run_cai_oracle",
    "run_discovery_oracle",
    "run_field_oracle",
]
