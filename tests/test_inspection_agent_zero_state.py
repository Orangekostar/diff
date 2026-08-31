from __future__ import annotations

import numpy as np

from cmc_bbdm.inspection_agent.state import (
    budget_record,
    measurement_mask,
    zero_state,
)
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid


def test_zero_state_has_no_measurements_and_all_cells_are_unmeasured() -> None:
    grid = build_acquisition_grid(338, 340, initial_budget=0.015625)

    state = zero_state(grid)
    mask = measurement_mask(grid, state)
    budget = budget_record(grid, state)

    assert state.levels == (-1,) * 64
    assert state.grid_sha256 == grid.state_sha256
    assert len(state.state_sha256) == 64
    assert mask.shape == grid.native_shape
    assert mask.dtype == np.bool_
    assert not np.any(mask)
    assert not mask.flags.writeable
    assert budget.measured_count == 0
    assert budget.native_count == 338 * 340
    assert budget.effective_budget == 0.0
