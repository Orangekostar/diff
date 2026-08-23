from __future__ import annotations

import inspect

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import initial_state
from cmc_bbdm.mva.oracle import choose_random_action, choose_uniform_action


def test_deployable_controls_have_no_image_or_measurement_value_argument() -> None:
    forbidden = {"image", "rgb", "pixels", "target", "cai", "oracle", "value"}
    for function in (choose_uniform_action, choose_random_action):
        parameters = set(inspect.signature(function).parameters)
        assert parameters.isdisjoint(forbidden)


def test_uniform_and_random_controls_run_from_geometry_and_state_only() -> None:
    grid = build_acquisition_grid(91, 93, initial_budget=0.03125)
    state = initial_state(grid)

    assert choose_uniform_action(grid, state, checkpoint=0.25).cell_index == 0
    first = choose_random_action(grid, state, checkpoint=0.25, seed=2026082300)
    second = choose_random_action(grid, state, checkpoint=0.25, seed=2026082300)
    assert first == second
