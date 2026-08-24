from __future__ import annotations

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import initial_state
from cmc_bbdm.mvd.one_shot_oracle import plan_frozen_ranking, score_initial_ranking


def test_one_shot_plan_reuses_mva_grid_and_state_hash() -> None:
    grid = build_acquisition_grid(338, 340, initial_budget=0.015625)
    state = initial_state(grid)
    ranking = score_initial_ranking(
        lambda: tuple(float(64 - cell) for cell in range(64)),
        method="one_shot_mechanical_oracle",
    )

    plan = plan_frozen_ranking(
        grid,
        state,
        ranking=ranking,
        checkpoints=(0.0625, 0.125, 0.25),
    )

    assert plan.grid_state_sha256 == grid.state_sha256
    assert all(snapshot.state.grid_sha256 == grid.state_sha256 for snapshot in plan.snapshots)
    assert all(action.from_level == 0 and action.to_level == 1 for action in plan.actions)
