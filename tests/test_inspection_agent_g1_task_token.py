from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.contracts import (
    ACTION_SLOT_COUNT,
    CANDIDATE_FEATURE_DIMENSION,
    CELL_COUNT,
    CELL_FEATURE_DIMENSION,
    RECONSTRUCTION_EMBEDDING_DIMENSION,
    CAIContextMode,
    G1PolicyState,
    TaskTokenMode,
)


def _state(task: InspectionTask, mode: TaskTokenMode) -> G1PolicyState:
    legal_action_mask = np.zeros(ACTION_SLOT_COUNT, dtype=np.bool_)
    legal_action_mask[0] = True
    token = {
        (InspectionTask.FIELD, TaskTokenMode.CORRECT): (1.0, 0.0),
        (InspectionTask.FIELD, TaskTokenMode.NO_TASK): (0.0, 0.0),
        (InspectionTask.FIELD, TaskTokenMode.WRONG_TASK): (0.0, 1.0),
        (InspectionTask.CAI, TaskTokenMode.CORRECT): (0.0, 1.0),
        (InspectionTask.CAI, TaskTokenMode.NO_TASK): (0.0, 0.0),
        (InspectionTask.CAI, TaskTokenMode.WRONG_TASK): (1.0, 0.0),
    }[(task, mode)]
    return G1PolicyState(
        task=task,
        task_token_mode=mode,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        observation_sha256="a" * 64,
        reconstruction_sha256="b" * 64,
        surface_hypothesis_sha256="c" * 64,
        grid_sha256="d" * 64,
        reconstruction_embedding=np.zeros(RECONSTRUCTION_EMBEDDING_DIMENSION),
        global_scalars=np.zeros(17),
        task_token=np.asarray(token),
        cell_features=np.zeros((CELL_COUNT, CELL_FEATURE_DIMENSION)),
        candidate_features=np.zeros(
            (ACTION_SLOT_COUNT, CANDIDATE_FEATURE_DIMENSION)
        ),
        legal_action_mask=legal_action_mask,
    )


@pytest.mark.parametrize(
    ("task", "mode", "expected"),
    (
        (InspectionTask.FIELD, TaskTokenMode.CORRECT, (1.0, 0.0)),
        (InspectionTask.FIELD, TaskTokenMode.NO_TASK, (0.0, 0.0)),
        (InspectionTask.FIELD, TaskTokenMode.WRONG_TASK, (0.0, 1.0)),
        (InspectionTask.CAI, TaskTokenMode.CORRECT, (0.0, 1.0)),
        (InspectionTask.CAI, TaskTokenMode.NO_TASK, (0.0, 0.0)),
        (InspectionTask.CAI, TaskTokenMode.WRONG_TASK, (1.0, 0.0)),
    ),
)
def test_task_token_modes_are_exact(
    task: InspectionTask, mode: TaskTokenMode, expected: tuple[float, float]
) -> None:
    state = _state(task, mode)
    np.testing.assert_array_equal(state.task_token, expected)
    assert np.count_nonzero(state.legal_action_mask) == 1


def test_changing_only_task_token_mode_changes_state_hash() -> None:
    correct = _state(InspectionTask.FIELD, TaskTokenMode.CORRECT)
    no_task = _state(InspectionTask.FIELD, TaskTokenMode.NO_TASK)

    assert correct.task == no_task.task
    assert correct.cai_context_mode == no_task.cai_context_mode
    np.testing.assert_array_equal(
        correct.reconstruction_embedding, no_task.reconstruction_embedding
    )
    np.testing.assert_array_equal(correct.global_scalars, no_task.global_scalars)
    np.testing.assert_array_equal(correct.cell_features, no_task.cell_features)
    np.testing.assert_array_equal(correct.candidate_features, no_task.candidate_features)
    np.testing.assert_array_equal(correct.legal_action_mask, no_task.legal_action_mask)
    assert correct.task_token_mode is not no_task.task_token_mode
    assert correct.state_sha256 != no_task.state_sha256
