from __future__ import annotations

import pytest

from cmc_bbdm.inspection_agent.state import (
    GeneralizedMeasurementStateError,
    InspectionCellAction,
    apply_action,
    legal_actions,
    zero_state,
)
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid


def test_only_one_level_forward_transitions_are_legal() -> None:
    grid = build_acquisition_grid(41, 43, initial_budget=0.015625)
    state = zero_state(grid)
    assert legal_actions(grid, state) == tuple(
        InspectionCellAction(cell, -1, 0) for cell in range(64)
    )

    state = apply_action(grid, state, InspectionCellAction(7, -1, 0))
    assert state.levels[7] == 0
    assert InspectionCellAction(7, 0, 1) in legal_actions(grid, state)
    state = apply_action(grid, state, InspectionCellAction(7, 0, 1))
    state = apply_action(grid, state, InspectionCellAction(7, 1, 2))
    assert state.levels[7] == 2
    assert all(action.cell_index != 7 for action in legal_actions(grid, state))


@pytest.mark.parametrize(
    "action",
    [
        InspectionCellAction(0, 0, 1),
        InspectionCellAction(0, -1, 1),
        InspectionCellAction(0, -1, 2),
    ],
)
def test_transition_rejects_wrong_source_or_skipped_level(
    action: InspectionCellAction,
) -> None:
    grid = build_acquisition_grid(41, 43, initial_budget=0.015625)
    with pytest.raises(GeneralizedMeasurementStateError):
        apply_action(grid, zero_state(grid), action)
