from __future__ import annotations

import numpy as np

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    apply_action,
    initial_state,
    legal_actions,
    measurement_mask,
)


def test_every_action_advances_one_cell_and_only_adds_measurements() -> None:
    grid = build_acquisition_grid(338, 340, initial_budget=0.03125)
    state = initial_state(grid)

    assert state.levels == (0,) * 64
    assert tuple(action.cell_index for action in legal_actions(grid, state)) == tuple(
        range(64)
    )
    old_mask = measurement_mask(grid, state)
    action = legal_actions(grid, state)[17]
    refined = apply_action(grid, state, action)
    new_mask = measurement_mask(grid, refined)

    assert refined.levels[17] == 1
    assert sum(refined.levels) == 1
    assert np.all(old_mask <= new_mask)
    assert np.count_nonzero(new_mask) > np.count_nonzero(old_mask)

    second = next(
        item for item in legal_actions(grid, refined) if item.cell_index == 17
    )
    full_cell = apply_action(grid, refined, second)
    assert full_cell.levels[17] == 2
    assert np.all(new_mask <= measurement_mask(grid, full_cell))
    assert all(item.cell_index != 17 for item in legal_actions(grid, full_cell))


def test_state_transition_does_not_mutate_prior_state() -> None:
    grid = build_acquisition_grid(674, 675, initial_budget=0.015625)
    state = initial_state(grid)
    before = state.levels
    _refined = apply_action(grid, state, legal_actions(grid, state)[0])

    assert state.levels == before == (0,) * 64
