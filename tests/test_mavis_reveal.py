from __future__ import annotations

import numpy as np
import pytest
from mavis_test_support import synthetic_authority, synthetic_inputs

from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mavis.reveal import (
    MAVISRevealError,
    reveal_action,
    reveal_uniform_scout,
)
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    RefinementAction,
    candidate_budget_record,
    initial_state,
    legal_actions,
    measurement_mask,
)


def test_mavis_reveal_matches_authoritative_scan() -> None:
    authority = synthetic_authority()
    context = authority.policy_context("sample-001")
    state = reveal_uniform_scout(
        authority, context, initial_budget=0.015625, checkpoint=0.25
    )
    action = RefinementAction(cell_index=7, from_level=0, to_level=1)

    revealed = reveal_action(authority, state, action)
    full_scan = authority.evaluation_view("sample-001").full_scan

    np.testing.assert_array_equal(
        revealed.measurement_values,
        full_scan[revealed.acquired_positions[:, 0], revealed.acquired_positions[:, 1]],
    )
    assert not revealed.acquired_positions.flags.writeable
    assert not revealed.measurement_values.flags.writeable
    assert revealed.action_history == (action,)


def test_mavis_future_unacquired_content_is_inaccessible() -> None:
    first_inputs = synthetic_inputs()
    second_inputs = synthetic_inputs()
    image = first_inputs["images"][0]
    grid = build_acquisition_grid(*image.shape[:2], initial_budget=0.015625)
    scout_mask = measurement_mask(grid, initial_state(grid))
    second_inputs["images"][0][~scout_mask] = 255 - image[~scout_mask]
    first = MAVISAuthority.from_arrays(**first_inputs)
    second = MAVISAuthority.from_arrays(**second_inputs)

    first_state = reveal_uniform_scout(
        first, first.policy_context("sample-001"), initial_budget=0.015625, checkpoint=0.25
    )
    second_state = reveal_uniform_scout(
        second,
        second.policy_context("sample-001"),
        initial_budget=0.015625,
        checkpoint=0.25,
    )

    assert first_state == second_state
    np.testing.assert_array_equal(
        first_state.acquired_positions, second_state.acquired_positions
    )
    np.testing.assert_array_equal(
        first_state.measurement_values, second_state.measurement_values
    )
    assert not hasattr(first_state, "full_scan")
    assert not hasattr(first_state, "true_cai")


def test_mavis_no_duplicate_acquisition() -> None:
    authority = synthetic_authority()
    state = reveal_uniform_scout(
        authority,
        authority.policy_context("sample-001"),
        initial_budget=0.015625,
        checkpoint=1.0,
    )
    action = RefinementAction(cell_index=0, from_level=0, to_level=1)
    state = reveal_action(authority, state, action)

    with pytest.raises(MAVISRevealError, match="legal"):
        reveal_action(authority, state, action)


def test_mavis_budget_never_exceeded() -> None:
    authority = synthetic_authority()
    context = authority.policy_context("sample-001")
    grid = build_acquisition_grid(*context.native_shape, initial_budget=0.015625)
    legacy = initial_state(grid)
    first_action = legal_actions(grid, legacy)[0]
    current = int(np.count_nonzero(measurement_mask(grid, legacy)))
    candidate = candidate_budget_record(grid, legacy, first_action).measured_count
    checkpoint = (candidate - 1) / context.native_count
    state = reveal_uniform_scout(
        authority,
        context,
        initial_budget=0.015625,
        checkpoint=checkpoint,
    )

    with pytest.raises(MAVISRevealError, match="budget"):
        reveal_action(authority, state, first_action)
    assert state.exact_acquired_count == current
