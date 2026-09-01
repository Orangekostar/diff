"""Observable-only G1 actor feature construction and canonical action slots."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from cmc_bbdm.inspection_agent.contracts import (
    InspectionDecision,
    InspectionObservation,
    InspectionTask,
)
from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    GeneralizedReconstruction,
    SourceBackgroundPrior,
)
from cmc_bbdm.inspection_agent.state import (
    InspectionCellAction,
    candidate_budget_record,
    fitting_actions,
)
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .contracts import (
    ACTION_SLOT_COUNT,
    CANDIDATE_FEATURE_DIMENSION,
    CELL_COUNT,
    CELL_FEATURE_DIMENSION,
    GLOBAL_SCALAR_DIMENSION,
    RECONSTRUCTION_EMBEDDING_DIMENSION,
    CAIContextMode,
    G1PolicyState,
    TaskTokenMode,
)

PRIVILEGED_INPUT_KEYS = frozenset(
    {
        "true_cai",
        "full_scan",
        "dataset_id",
        "specimen_id",
        "oracle_value",
        "oracle_utility",
        "future_measurement",
        "true_task_loss",
    }
)
_TRANSITIONS = ((-1, 0), (0, 1), (1, 2))


class G1FeatureError(ValueError):
    """Raised when actor features are malformed or contain privilege."""


def reject_privileged_columns(columns: Mapping[str, object]) -> None:
    if not isinstance(columns, Mapping):
        raise G1FeatureError("actor supplemental input must be a mapping")
    keys = set(columns)
    if keys & PRIVILEGED_INPUT_KEYS:
        raise G1FeatureError("privileged actor input is forbidden")
    if keys:
        raise G1FeatureError("unexpected actor input is forbidden")


def canonical_action_from_slot(slot: int) -> InspectionCellAction:
    if type(slot) is not int or not 0 <= slot < ACTION_SLOT_COUNT:
        raise G1FeatureError("canonical action slot is invalid")
    transition, cell = divmod(slot, CELL_COUNT)
    source, target = _TRANSITIONS[transition]
    return InspectionCellAction(cell, source, target)


def canonical_slot(action: InspectionCellAction) -> int:
    if type(action) is not InspectionCellAction:
        raise G1FeatureError("inspection action is required")
    try:
        transition = _TRANSITIONS.index((action.from_level, action.to_level))
    except ValueError as error:
        raise G1FeatureError("inspection action is not a canonical transition") from error
    return transition * CELL_COUNT + action.cell_index


def decision_type(
    action: InspectionCellAction,
    surface_hypothesis: SurfaceHypothesis,
) -> InspectionDecision:
    if type(action) is not InspectionCellAction or type(surface_hypothesis) is not SurfaceHypothesis:
        raise G1FeatureError("decision-type request is invalid")
    if action.from_level >= 0:
        return InspectionDecision.REFINE
    return (
        InspectionDecision.FOCUS
        if action.cell_index in surface_hypothesis.top_cells
        else InspectionDecision.BROADEN
    )


def _validate_surface(surface: SurfaceHypothesis) -> None:
    if (
        type(surface) is not SurfaceHypothesis
        or np.asarray(surface.scores).shape != (CELL_COUNT,)
        or not np.all(np.isfinite(surface.scores))
        or type(surface.top_cells) is not tuple
        or not surface.top_cells
        or len(set(surface.top_cells)) != len(surface.top_cells)
        or any(type(cell) is not int or not 0 <= cell < CELL_COUNT for cell in surface.top_cells)
        or type(surface.state_sha256) is not str
        or len(surface.state_sha256) != 64
    ):
        raise G1FeatureError("surface hypothesis is invalid")


def _cell_features(
    observation: InspectionObservation,
    grid: AcquisitionGrid,
    prior: SourceBackgroundPrior,
    surface: SurfaceHypothesis,
) -> np.ndarray:
    observed_mask = np.zeros(grid.native_shape, dtype=np.bool_)
    observed_values = np.zeros((*grid.native_shape, 3), dtype=np.uint8)
    positions = observation.acquired_positions
    observed_mask[positions[:, 0], positions[:, 1]] = True
    observed_values[positions[:, 0], positions[:, 1]] = observation.measurement_values
    output = np.zeros((CELL_COUNT, CELL_FEATURE_DIMENSION), dtype=np.float64)
    previous_cell = (
        observation.action_history[-1].cell_index if observation.action_history else None
    )
    top_cells = set(surface.top_cells)
    for cell, level in zip(grid.cells, observation.measurement_state.levels, strict=True):
        row_start = cell.rows[2][0]
        row_stop = cell.rows[2][-1] + (1 if cell.row == 7 else 0)
        column_start = cell.columns[2][0]
        column_stop = cell.columns[2][-1] + (1 if cell.column == 7 else 0)
        local_mask = observed_mask[row_start:row_stop, column_start:column_stop]
        measured = 0 if level < 0 else int(np.count_nonzero(local_mask))
        native = int(local_mask.size)
        row = output[cell.index]
        row[0] = cell.row / 7.0
        row[1] = cell.column / 7.0
        row[2] = float(surface.scores[cell.index])
        row[3] = float(cell.index in top_cells)
        row[4 + level + 1] = 1.0
        row[8] = measured / native
        if measured:
            values = observed_values[
                row_start:row_stop,
                column_start:column_stop,
            ][local_mask].astype(np.float64)
            row[9:12] = np.mean(values, axis=0, dtype=np.float64) / 255.0
            row[12:15] = np.std(values, axis=0, dtype=np.float64) / 255.0
            row[15] = float(
                np.mean(
                    np.abs(values - prior.background_rgb.astype(np.float64)),
                    dtype=np.float64,
                )
                / 255.0
            )
            row[16] = 1.0
        row[17] = float(cell.index == previous_cell)
    return output


def _global_scalars(
    observation: InspectionObservation,
    *,
    cai_estimate: float | None,
    cai_context_mode: CAIContextMode,
) -> np.ndarray:
    levels = observation.measurement_state.levels
    counts = {level: levels.count(level) for level in (-1, 0, 1, 2)}
    observed = tuple(level for level in levels if level >= 0)
    if cai_context_mode is CAIContextMode.TASK_SPECIFIC_MASKED and observation.task is InspectionTask.FIELD:
        estimate, present = 0.0, 0.0
    else:
        if cai_estimate is None or not math.isfinite(float(cai_estimate)):
            raise G1FeatureError("observable CAI estimate is required")
        estimate, present = float(cai_estimate), 1.0
    height, width = observation.native_shape
    return np.asarray(
        (
            observation.effective_budget,
            observation.remaining_budget,
            len(observed) / CELL_COUNT,
            0.0 if not observed else float(np.mean(observed, dtype=np.float64)),
            float(counts[-1]),
            counts[-1] / CELL_COUNT,
            float(counts[0]),
            counts[0] / CELL_COUNT,
            float(counts[1]),
            counts[1] / CELL_COUNT,
            float(counts[2]),
            counts[2] / CELL_COUNT,
            estimate,
            present,
            height / 1024.0,
            width / 1024.0,
            math.log(height / width),
        ),
        dtype=np.float64,
    )


def _task_token(task: InspectionTask, mode: TaskTokenMode) -> np.ndarray:
    if mode is TaskTokenMode.NO_TASK:
        return np.zeros(2, dtype=np.float64)
    correct = np.asarray(
        (1.0, 0.0) if task is InspectionTask.FIELD else (0.0, 1.0),
        dtype=np.float64,
    )
    return correct if mode is TaskTokenMode.CORRECT else correct[::-1].copy()


def _candidate_features(
    observation: InspectionObservation,
    grid: AcquisitionGrid,
    surface: SurfaceHypothesis,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros(
        (ACTION_SLOT_COUNT, CANDIDATE_FEATURE_DIMENSION), dtype=np.float64
    )
    mask = np.zeros(ACTION_SLOT_COUNT, dtype=np.bool_)
    legal = {
        canonical_slot(action): action
        for action in fitting_actions(
            grid,
            observation.measurement_state,
            observation.endpoint_budget,
        )
    }
    current_budget = observation.effective_budget
    for slot in range(ACTION_SLOT_COUNT):
        action = canonical_action_from_slot(slot)
        row = features[slot]
        row[action.from_level + 1] = 1.0
        row[3 + action.to_level] = 1.0
        row[7] = observation.remaining_budget
        row[8] = current_budget / observation.endpoint_budget
        kind = decision_type(action, surface)
        row[9 + (InspectionDecision.FOCUS, InspectionDecision.BROADEN, InspectionDecision.REFINE).index(kind)] = 1.0
        if slot in legal:
            candidate = candidate_budget_record(
                grid,
                observation.measurement_state,
                action,
            )
            row[6] = candidate.effective_budget - current_budget
            row[8] = candidate.effective_budget / observation.endpoint_budget
            mask[slot] = True
    return features, mask


def build_policy_state(
    observation: InspectionObservation,
    surface_hypothesis: SurfaceHypothesis,
    grid: AcquisitionGrid,
    source_prior: SourceBackgroundPrior,
    reconstruction: GeneralizedReconstruction,
    *,
    reconstruction_embedding: object,
    cai_estimate: float | None,
    cai_context_mode: CAIContextMode,
    task_token_mode: TaskTokenMode,
    supplied_columns: Mapping[str, object] | None = None,
) -> G1PolicyState:
    if supplied_columns is not None:
        reject_privileged_columns(supplied_columns)
    _validate_surface(surface_hypothesis)
    if (
        type(observation) is not InspectionObservation
        or observation.task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or type(grid) is not AcquisitionGrid
        or type(source_prior) is not SourceBackgroundPrior
        or type(reconstruction) is not GeneralizedReconstruction
        or type(cai_context_mode) is not CAIContextMode
        or type(task_token_mode) is not TaskTokenMode
        or observation.grid_sha256 != grid.state_sha256
        or observation.native_shape != grid.native_shape
        or reconstruction.image.shape != (*grid.native_shape, 3)
        or reconstruction.measured_count != observation.exact_acquired_count
        or reconstruction.native_count != observation.native_count
        or reconstruction.effective_budget != observation.effective_budget
        or not reconstruction.observed_values_exact
    ):
        raise G1FeatureError("observable policy-state request is invalid")
    embedding = np.asarray(reconstruction_embedding, dtype=np.float64)
    if (
        embedding.shape != (RECONSTRUCTION_EMBEDDING_DIMENSION,)
        or not np.all(np.isfinite(embedding))
    ):
        raise G1FeatureError("reconstruction embedding is invalid")
    global_scalars = _global_scalars(
        observation,
        cai_estimate=cai_estimate,
        cai_context_mode=cai_context_mode,
    )
    if global_scalars.shape != (GLOBAL_SCALAR_DIMENSION,):
        raise G1FeatureError("global scalar contract changed")
    cells = _cell_features(observation, grid, source_prior, surface_hypothesis)
    candidates, mask = _candidate_features(observation, grid, surface_hypothesis)
    return G1PolicyState(
        task=observation.task,
        task_token_mode=task_token_mode,
        cai_context_mode=cai_context_mode,
        observation_sha256=observation.state_sha256,
        reconstruction_sha256=reconstruction.state_sha256,
        surface_hypothesis_sha256=surface_hypothesis.state_sha256,
        grid_sha256=grid.state_sha256,
        reconstruction_embedding=embedding,
        global_scalars=global_scalars,
        task_token=_task_token(observation.task, task_token_mode),
        cell_features=cells,
        candidate_features=candidates,
        legal_action_mask=mask,
    )


__all__ = [
    "PRIVILEGED_INPUT_KEYS",
    "G1FeatureError",
    "build_policy_state",
    "canonical_action_from_slot",
    "canonical_slot",
    "decision_type",
    "reject_privileged_columns",
]
