from __future__ import annotations

import pytest

from cmc_bbdm.inspection_agent.state import (
    InspectionCellAction,
    apply_action,
    fitting_actions,
    zero_state,
)
from cmc_bbdm.inspection_agent_g1.warm_start import (
    PRIMARY_WARM_START_CELLS,
    apply_warm_start,
    build_deployment_grid,
    warm_start_audit,
)
from cmc_bbdm.mva.oracle import uniform_cell_order

EXPECTED = {
    (338, 340): (294, 114920, 0.0025583014270797078),
    (338, 352): (305, 118976, 0.0025635422270037654),
    (674, 675): (1128, 454950, 0.0024793933399274645),
}


@pytest.mark.parametrize(("native_shape", "expected"), EXPECTED.items())
def test_k8_warm_start_preserves_broaden_and_refine(
    native_shape: tuple[int, int], expected: tuple[int, int, float]
) -> None:
    grid = build_deployment_grid(native_shape)
    state = apply_warm_start(grid, k=8)
    audit = warm_start_audit(grid, state, k=8)
    measured, native, budget = expected
    assert PRIMARY_WARM_START_CELLS == uniform_cell_order()[:8]
    assert audit.cells == PRIMARY_WARM_START_CELLS
    assert audit.measured_count == measured
    assert audit.native_count == native
    assert audit.effective_budget == budget
    assert audit.broaden_action_count == 56
    assert audit.refine_action_count == 8
    assert tuple(index for index, level in enumerate(state.levels) if level == 0) == tuple(
        sorted(PRIMARY_WARM_START_CELLS)
    )
    assert sum(level == -1 for level in state.levels) == 56


def test_complete_level_zero_scout_has_no_broaden_action() -> None:
    grid = build_deployment_grid((338, 340))
    state = zero_state(grid)
    for cell in uniform_cell_order():
        state = apply_action(grid, state, InspectionCellAction(cell, -1, 0))
    actions = fitting_actions(grid, state, 0.25)
    assert all(action.from_level >= 0 for action in actions)
    assert sum(action.from_level == -1 for action in actions) == 0


@pytest.mark.parametrize("k", (4, 8, 16))
def test_registered_warm_start_sizes_are_geometry_only(k: int) -> None:
    grid = build_deployment_grid((41, 43))
    first = apply_warm_start(grid, k=k)
    second = apply_warm_start(grid, k=k)
    assert first == second
    assert sum(level == 0 for level in first.levels) == k
    assert tuple(index for index, level in enumerate(first.levels) if level == 0) == tuple(
        sorted(uniform_cell_order()[:k])
    )
