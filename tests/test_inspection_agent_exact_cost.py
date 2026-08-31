from __future__ import annotations

import numpy as np

from cmc_bbdm.inspection_agent.state import (
    InspectionCellAction,
    action_added_positions,
    action_added_positions_from_mask,
    apply_action,
    budget_record,
    candidate_budget_record,
    legal_actions,
    measurement_mask,
    zero_state,
)
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.oracle import uniform_cell_order


def test_candidate_cost_is_unique_mask_union_difference() -> None:
    grid = build_acquisition_grid(338, 352, initial_budget=0.015625)
    state = zero_state(grid)
    previous = 0

    for cell in uniform_cell_order():
        action = InspectionCellAction(cell, -1, 0)
        positions = action_added_positions(grid, state, action)
        candidate = candidate_budget_record(grid, state, action)
        assert positions.shape == (candidate.measured_count - previous, 2)
        assert len(np.unique(positions, axis=0)) == len(positions)
        state = apply_action(grid, state, action)
        assert budget_record(grid, state) == candidate
        previous = candidate.measured_count

    assert previous == 1848
    assert np.count_nonzero(measurement_mask(grid, state)) == previous


def test_shared_boundaries_make_action_cost_state_dependent() -> None:
    grid = build_acquisition_grid(41, 43, initial_budget=0.015625)
    zero = zero_state(grid)
    adjacent = InspectionCellAction(1, -1, 0)
    zero_cost = candidate_budget_record(grid, zero, adjacent).measured_count
    after_left = apply_action(grid, zero, InspectionCellAction(0, -1, 0))
    added_after_left = (
        candidate_budget_record(grid, after_left, adjacent).measured_count
        - budget_record(grid, after_left).measured_count
    )
    assert added_after_left < zero_cost


def test_local_candidate_positions_match_exhaustive_mask_difference() -> None:
    grid = build_acquisition_grid(73, 79, initial_budget=0.015625)
    state = zero_state(grid)
    for action in (
        InspectionCellAction(0, -1, 0),
        InspectionCellAction(1, -1, 0),
        InspectionCellAction(0, 0, 1),
        InspectionCellAction(9, -1, 0),
    ):
        state = apply_action(grid, state, action)
    current_mask = measurement_mask(grid, state)
    for action in legal_actions(grid, state):
        candidate = apply_action(grid, state, action)
        expected = np.argwhere(
            measurement_mask(grid, candidate) & ~current_mask
        ).astype("<i8", copy=False)
        actual = action_added_positions_from_mask(grid, state, action, current_mask)
        np.testing.assert_array_equal(actual, expected)
