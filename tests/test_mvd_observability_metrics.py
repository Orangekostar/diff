from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    budget_record,
    initial_state,
)
from cmc_bbdm.mvd.observability_metrics import (
    build_exact_cost_context,
    evaluate_ranking,
)


def test_observability_metrics_reward_exact_ranking_and_exact_cost_sets() -> None:
    truth = np.linspace(-0.5, 1.0, 64)
    grid = build_acquisition_grid(338, 352, initial_budget=0.015625)
    perfect = evaluate_ranking(
        truth,
        truth,
        grid=grid,
        checkpoints=(0.0625, 0.125, 0.25),
    )
    reverse = evaluate_ranking(
        truth,
        -truth,
        grid=grid,
        checkpoints=(0.0625, 0.125, 0.25),
    )

    assert perfect.spearman == pytest.approx(1.0)
    assert perfect.ndcg_10 == pytest.approx(1.0)
    assert perfect.recall_10 == pytest.approx(1.0)
    assert perfect.regret_1 == pytest.approx(0.0)
    assert max(perfect.budgeted_regret) == pytest.approx(0.0)
    assert reverse.spearman == pytest.approx(-1.0)
    assert reverse.regret_1 > 0.0
    assert all(value > 0.0 for value in reverse.budgeted_regret[:2])
    assert reverse.budgeted_regret[-1] == pytest.approx(0.0)


def test_cost_bound_ranking_matches_authoritative_grid_planner() -> None:
    truth = np.random.default_rng(20260824).normal(size=64)
    grid = build_acquisition_grid(338, 352, initial_budget=0.015625)
    state = initial_state(grid)
    initial = budget_record(grid, state)
    checkpoints = (0.0625, 0.09375, 0.125, 0.1875, 0.25)

    authoritative = evaluate_ranking(
        truth, truth[::-1], grid=grid, checkpoints=checkpoints
    )
    cost_bound = evaluate_ranking(
        truth,
        truth[::-1],
        grid=grid,
        checkpoints=checkpoints,
        exact_cost_context=build_exact_cost_context(grid),
    )

    assert initial.measured_count == build_exact_cost_context(grid).initial_measured_count
    assert cost_bound.budgeted_regret == authoritative.budgeted_regret
    assert cost_bound.selected_value == authoritative.selected_value
    assert cost_bound.oracle_value == authoritative.oracle_value
