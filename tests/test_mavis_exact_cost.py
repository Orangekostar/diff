from __future__ import annotations

from mavis_test_support import synthetic_authority

from cmc_bbdm.mavis.reveal import reveal_action, reveal_uniform_scout
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    RefinementAction,
    apply_action,
    budget_record,
    initial_state,
)


def test_mavis_exact_action_cost_matches_legacy_mvd() -> None:
    authority = synthetic_authority()
    context = authority.policy_context("sample-001")
    grid = build_acquisition_grid(*context.native_shape, initial_budget=0.015625)
    action = RefinementAction(cell_index=13, from_level=0, to_level=1)
    legacy_after = apply_action(grid, initial_state(grid), action)
    legacy_budget = budget_record(grid, legacy_after)

    state = reveal_uniform_scout(
        authority, context, initial_budget=0.015625, checkpoint=0.25
    )
    state = reveal_action(authority, state, action)

    assert state.exact_acquired_count == legacy_budget.measured_count
    assert state.native_count == legacy_budget.native_count
    assert state.effective_budget == legacy_budget.effective_budget
