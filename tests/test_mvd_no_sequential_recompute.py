from __future__ import annotations

import inspect

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import initial_state
from cmc_bbdm.mvd.one_shot_oracle import plan_frozen_ranking, score_initial_ranking


def test_planner_accepts_frozen_ranking_not_scoring_callback() -> None:
    parameters = inspect.signature(plan_frozen_ranking).parameters
    assert "ranking" in parameters
    assert not any("score" in name for name in parameters)

    ranking = score_initial_ranking(
        lambda: tuple(float(cell % 7) for cell in range(64)),
        method="one_shot_reconstruction",
    )
    original_scores = ranking.scores.copy()
    grid = build_acquisition_grid(674, 675, initial_budget=0.03125)
    plan = plan_frozen_ranking(
        grid,
        initial_state(grid),
        ranking=ranking,
        checkpoints=(0.0625, 0.125, 0.25),
    )

    assert not ranking.scores.flags.writeable
    assert tuple(ranking.scores) == tuple(original_scores)
    assert plan.ranking_state_sha256 == ranking.state_sha256
