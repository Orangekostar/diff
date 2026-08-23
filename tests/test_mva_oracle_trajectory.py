from __future__ import annotations

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import initial_state
from cmc_bbdm.mva.oracle import (
    choose_greedy_oracle_action,
    uniform_cell_order,
)
from cmc_bbdm.mva.oracle_trajectory import run_control_trajectory


def test_uniform_order_is_deterministic_farthest_spread() -> None:
    order = uniform_cell_order()

    assert len(order) == 64
    assert len(set(order)) == 64
    assert order[0] == 0
    assert order[1] == 63
    assert uniform_cell_order() == order


def test_greedy_tie_break_is_lower_cell_then_level() -> None:
    grid = build_acquisition_grid(91, 93, initial_budget=0.015625)
    state = initial_state(grid)

    selected = choose_greedy_oracle_action(
        grid,
        state,
        checkpoint=0.25,
        score_candidate=lambda _state, _action, _candidate: 1.0,
    )

    assert selected.cell_index == 0
    assert selected.to_level == 1


def test_random_trajectory_is_seed_replayable() -> None:
    grid = build_acquisition_grid(91, 93, initial_budget=0.03125)
    checkpoints = (0.0625, 0.125, 0.25)

    first = run_control_trajectory(
        grid,
        initial_state(grid),
        checkpoints=checkpoints,
        method="random",
        seed=2026082307,
    )
    second = run_control_trajectory(
        grid,
        initial_state(grid),
        checkpoints=checkpoints,
        method="random",
        seed=2026082307,
    )

    assert first == second
    assert [row.nominal_checkpoint for row in first.snapshots] == list(checkpoints)
