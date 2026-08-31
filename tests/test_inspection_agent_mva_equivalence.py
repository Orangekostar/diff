from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.state import (
    GeneralizedMeasurementState,
    InspectionCellAction,
    apply_action,
    measurement_mask,
    zero_state,
)
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    MeasurementState,
    initial_state,
)
from cmc_bbdm.mva.measurement_state import (
    measurement_mask as mva_measurement_mask,
)
from cmc_bbdm.mva.oracle import uniform_cell_order

FAMILIES = (
    ((338, 340), 0.015625, 1764),
    ((338, 340), 0.03125, 3600),
    ((338, 352), 0.015625, 1848),
    ((338, 352), 0.03125, 3720),
    ((674, 675), 0.015625, 7056),
    ((674, 675), 0.03125, 14161),
)


@pytest.mark.parametrize(("shape", "initial_budget", "expected_count"), FAMILIES)
def test_individual_zero_start_scout_actions_equal_old_complete_scout(
    shape: tuple[int, int], initial_budget: float, expected_count: int
) -> None:
    grid = build_acquisition_grid(*shape, initial_budget=initial_budget)
    state = zero_state(grid)
    for cell in uniform_cell_order():
        state = apply_action(grid, state, InspectionCellAction(cell, -1, 0))

    generalized = measurement_mask(grid, state)
    legacy = mva_measurement_mask(grid, initial_state(grid))
    np.testing.assert_array_equal(generalized, legacy)
    assert np.count_nonzero(generalized) == expected_count


@pytest.mark.parametrize(("level", "legacy_level"), [(0, 0), (1, 1), (2, 2)])
def test_generalized_nonnegative_uniform_levels_equal_mva_masks(
    level: int, legacy_level: int
) -> None:
    grid = build_acquisition_grid(674, 675, initial_budget=0.015625)
    generalized = GeneralizedMeasurementState(
        grid_sha256=grid.state_sha256,
        levels=(level,) * 64,
    )
    legacy = MeasurementState(
        grid_sha256=grid.state_sha256,
        levels=(legacy_level,) * 64,
    )
    np.testing.assert_array_equal(
        measurement_mask(grid, generalized),
        mva_measurement_mask(grid, legacy),
    )
