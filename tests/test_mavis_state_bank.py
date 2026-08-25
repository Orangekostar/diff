from __future__ import annotations

import numpy as np
import pytest
from mavis_test_support import synthetic_authority

from cmc_bbdm.mavis.state_bank import (
    MAVISStateBankError,
    PlannedAction,
    materialize_action_plan,
)
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    RefinementAction,
    candidate_budget_record,
    initial_state,
)


def test_mavis_state_bank_materializes_monotone_causal_snapshots() -> None:
    authority = synthetic_authority()
    context = authority.policy_context("sample-001")
    grid = build_acquisition_grid(*context.native_shape, initial_budget=0.015625)
    first = RefinementAction(0, 0, 1)
    second = RefinementAction(1, 0, 1)
    first_budget = candidate_budget_record(grid, initial_state(grid), first)
    first_state = initial_state(grid)
    from cmc_bbdm.mva.measurement_state import apply_action

    first_state = apply_action(grid, first_state, first)
    second_budget = candidate_budget_record(grid, first_state, second)
    checkpoints = (
        float(first_budget.effective_budget + 1.0e-12),
        float(second_budget.effective_budget + 1.0e-12),
    )

    trajectory = materialize_action_plan(
        authority,
        specimen_id="sample-001",
        method="uniform",
        seed=None,
        initial_budget=0.015625,
        checkpoints=checkpoints,
        actions=(
            PlannedAction(action=first, nominal_checkpoint=checkpoints[0]),
            PlannedAction(action=second, nominal_checkpoint=checkpoints[1]),
        ),
    )

    assert trajectory.method == "uniform"
    assert len(trajectory.snapshots) == 2
    assert tuple(snapshot.step for snapshot in trajectory.snapshots) == (1, 2)
    counts = tuple(
        snapshot.inspection_state.exact_acquired_count
        for snapshot in trajectory.snapshots
    )
    assert counts == tuple(sorted(counts))
    assert counts[0] < counts[1]
    for snapshot in trajectory.snapshots:
        state = snapshot.inspection_state
        full_scan = authority.evaluation_view("sample-001").full_scan
        np.testing.assert_array_equal(
            state.measurement_values,
            full_scan[state.acquired_positions[:, 0], state.acquired_positions[:, 1]],
        )
        assert state.effective_budget <= snapshot.nominal_checkpoint + 1.0e-15


def test_mavis_state_bank_rejects_action_assigned_to_an_infeasible_checkpoint() -> None:
    authority = synthetic_authority()
    context = authority.policy_context("sample-001")
    grid = build_acquisition_grid(*context.native_shape, initial_budget=0.015625)
    action = RefinementAction(0, 0, 1)
    initial = initial_state(grid)
    candidate = candidate_budget_record(grid, initial, action)
    infeasible = float((candidate.measured_count - 1) / candidate.native_count)

    with pytest.raises(MAVISStateBankError, match="checkpoint"):
        materialize_action_plan(
            authority,
            specimen_id="sample-001",
            method="uniform",
            seed=None,
            initial_budget=0.015625,
            checkpoints=(infeasible,),
            actions=(PlannedAction(action=action, nominal_checkpoint=infeasible),),
        )


def test_mavis_state_bank_preserves_quantized_scout_cost_at_nominal_budget() -> None:
    authority = synthetic_authority()

    trajectory = materialize_action_plan(
        authority,
        specimen_id="sample-001",
        method="uniform",
        seed=None,
        initial_budget=0.015625,
        checkpoints=(0.015625, 0.25),
        actions=(),
    )

    state = trajectory.snapshots[0].inspection_state
    assert state.effective_budget > 0.015625
    assert state.action_history == ()
