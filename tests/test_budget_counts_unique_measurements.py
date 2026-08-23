from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    action_fits_checkpoint,
    apply_action,
    budget_record,
    initial_state,
    legal_actions,
    measurement_mask,
)


@pytest.mark.parametrize("shape", ((674, 675), (338, 352), (338, 340)))
def test_budget_counts_unique_native_locations(shape: tuple[int, int]) -> None:
    grid = build_acquisition_grid(*shape, initial_budget=0.03125)
    state = initial_state(grid)
    for index in (0, 9, 27, 63, 0):
        action = next(
            item for item in legal_actions(grid, state) if item.cell_index == index
        )
        state = apply_action(grid, state, action)

    mask = measurement_mask(grid, state)
    record = budget_record(grid, state)

    assert mask.dtype == np.bool_
    assert record.measured_count == int(np.count_nonzero(mask))
    assert record.native_count == shape[0] * shape[1]
    assert record.effective_budget == pytest.approx(
        record.measured_count / record.native_count, abs=0.0
    )
    assert record.measured_count == len(set(zip(*np.nonzero(mask), strict=True)))


def test_checkpoint_fit_uses_actual_unique_count() -> None:
    grid = build_acquisition_grid(338, 340, initial_budget=0.03125)
    state = initial_state(grid)
    action = legal_actions(grid, state)[0]
    refined = apply_action(grid, state, action)
    exact_fraction = budget_record(grid, refined).effective_budget

    assert action_fits_checkpoint(grid, state, action, exact_fraction)
    assert not action_fits_checkpoint(
        grid, state, action, np.nextafter(exact_fraction, 0.0)
    )
