from __future__ import annotations

import inspect

import pytest

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    MeasurementState,
    budget_record,
    initial_state,
)
from cmc_bbdm.mva.oracle import MVAOracleError
from cmc_bbdm.mva.oracle_trajectory import run_static_mask_trajectory

CHECKPOINTS = (0.0625, 0.125, 0.25)
CELL_ORDER = tuple(reversed(range(64)))


def test_static_mask_api_cannot_receive_target_evidence() -> None:
    assert tuple(inspect.signature(run_static_mask_trajectory).parameters) == (
        "grid",
        "state",
        "cell_order",
        "checkpoints",
        "method",
    )


@pytest.mark.parametrize("shape", [(674, 675), (338, 352), (338, 340)])
def test_static_mask_trajectory_follows_frozen_level_one_order(
    shape: tuple[int, int],
) -> None:
    grid = build_acquisition_grid(*shape, initial_budget=0.03125)
    trajectory = run_static_mask_trajectory(
        grid,
        initial_state(grid),
        cell_order=CELL_ORDER,
        checkpoints=CHECKPOINTS,
        method="global_mechanical_mask",
    )

    selected_cells = tuple(action.cell_index for action in trajectory.actions)
    assert selected_cells == CELL_ORDER[: len(selected_cells)]
    assert all(
        (action.from_level, action.to_level) == (0, 1)
        for action in trajectory.actions
    )
    assert tuple(snapshot.nominal_checkpoint for snapshot in trajectory.snapshots) == (
        CHECKPOINTS
    )
    assert tuple(snapshot.cumulative_actions for snapshot in trajectory.snapshots) == (
        tuple(sorted(snapshot.cumulative_actions for snapshot in trajectory.snapshots))
    )
    assert all(
        snapshot.effective_budget <= snapshot.nominal_checkpoint
        for snapshot in trajectory.snapshots
    )
    assert set(trajectory.snapshots[-1].state.levels) <= {0, 1}


@pytest.mark.parametrize(
    "cell_order",
    [
        tuple(range(63)),
        tuple(range(63)) + (62,),
        tuple(range(63)) + (64,),
        tuple(range(63)) + (True,),
    ],
)
def test_static_mask_trajectory_rejects_invalid_ranking(
    cell_order: tuple[int, ...],
) -> None:
    grid = build_acquisition_grid(91, 93, initial_budget=0.03125)

    with pytest.raises(MVAOracleError, match="cell order"):
        run_static_mask_trajectory(
            grid,
            initial_state(grid),
            cell_order=cell_order,
            checkpoints=CHECKPOINTS,
            method="global_mechanical_mask",
        )


def test_static_mask_trajectory_rejects_refined_state_and_unknown_method() -> None:
    grid = build_acquisition_grid(91, 93, initial_budget=0.03125)
    refined_state = MeasurementState(
        grid_sha256=grid.state_sha256,
        levels=(1,) + (0,) * 63,
    )

    with pytest.raises(MVAOracleError, match="initial state"):
        run_static_mask_trajectory(
            grid,
            refined_state,
            cell_order=CELL_ORDER,
            checkpoints=CHECKPOINTS,
            method="global_mechanical_mask",
        )
    with pytest.raises(MVAOracleError, match="global mask"):
        run_static_mask_trajectory(
            grid,
            initial_state(grid),
            cell_order=CELL_ORDER,
            checkpoints=CHECKPOINTS,
            method="uniform",
        )


def test_discrete_initial_state_above_nominal_cap_takes_no_action() -> None:
    grid = build_acquisition_grid(338, 340, initial_budget=0.03125)
    state = initial_state(grid)
    assert budget_record(grid, state).effective_budget > 0.03125

    trajectory = run_static_mask_trajectory(
        grid,
        state,
        cell_order=CELL_ORDER,
        checkpoints=(0.03125, 0.0625),
        method="global_mechanical_mask",
    )

    assert trajectory.snapshots[0].cumulative_actions == 0
    assert trajectory.snapshots[0].effective_budget > 0.03125
    assert trajectory.snapshots[1].cumulative_actions > 0
