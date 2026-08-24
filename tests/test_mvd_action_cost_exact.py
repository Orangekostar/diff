from __future__ import annotations

import numpy as np
from test_mvd_config import CONFIG, ROOT

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    RefinementAction,
    budget_record,
    candidate_budget_record,
    initial_state,
)
from cmc_bbdm.mvd.action_cost_audit import build_action_cost_audit
from cmc_bbdm.mvd.authority import load_compact_mvd_authority
from cmc_bbdm.mvd.config import load_mvd_config
from cmc_bbdm.mvd.one_shot_oracle import plan_frozen_ranking, score_initial_ranking


def test_action_cost_audit_matches_candidate_banks_exactly() -> None:
    config = load_mvd_config(CONFIG, project_root=ROOT)
    authority = load_compact_mvd_authority(config, project_root=ROOT)
    audit = build_action_cost_audit(authority)

    assert len(audit.rows) == 276 * 2 * 64
    assert len(audit.summaries) == 2
    for summary in audit.summaries:
        bank = authority.candidate_banks[summary.initial_budget]
        assert summary.minimum == int(np.min(bank.added_measurements))
        assert summary.maximum == int(np.max(bank.added_measurements))
        assert summary.unique_costs == tuple(
            int(value) for value in np.unique(bank.added_measurements)
        )
        assert summary.coefficient_of_variation > 0.0


def test_unequal_cost_selector_skips_nonfitting_prefix_action() -> None:
    grid = build_acquisition_grid(338, 340, initial_budget=0.03125)
    state = initial_state(grid)
    costs = {
        cell: candidate_budget_record(
            grid, state, RefinementAction(cell, 0, 1)
        ).measured_count
        - budget_record(grid, state).measured_count
        for cell in range(64)
    }
    expensive = max(costs, key=lambda cell: (costs[cell], -cell))
    cheap = min(costs, key=lambda cell: (costs[cell], cell))
    cap = (budget_record(grid, state).measured_count + costs[cheap]) / grid.native_shape[0] / grid.native_shape[1]
    scores = np.zeros(64, dtype=np.float64)
    scores[expensive] = 2.0
    scores[cheap] = 1.0
    ranking = score_initial_ranking(
        lambda: scores, method="one_shot_mechanical_oracle"
    )

    plan = plan_frozen_ranking(grid, state, ranking=ranking, checkpoints=(cap,))

    assert plan.actions[0].cell_index == cheap
    assert plan.snapshots[0].measured_count == (
        budget_record(grid, state).measured_count + costs[cheap]
    )
    assert plan.snapshots[0].effective_budget <= cap
