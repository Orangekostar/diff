from __future__ import annotations

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import initial_state
from cmc_bbdm.mvd.one_shot_oracle import plan_frozen_ranking, score_initial_ranking


def test_initial_value_scorer_is_invoked_exactly_once() -> None:
    calls = 0

    def scorer() -> tuple[float, ...]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("initial values were recomputed")
        return tuple(float(cell) for cell in range(64))

    ranking = score_initial_ranking(scorer, method="one_shot_mechanical_oracle")
    grid = build_acquisition_grid(674, 675, initial_budget=0.015625)
    plan_frozen_ranking(
        grid,
        initial_state(grid),
        ranking=ranking,
        checkpoints=(0.0625, 0.09375, 0.125, 0.1875, 0.25),
    )

    assert calls == 1
