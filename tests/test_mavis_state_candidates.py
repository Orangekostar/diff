from __future__ import annotations

import numpy as np
from mavis_test_support import synthetic_authority, synthetic_inputs

from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mavis.reveal import reveal_uniform_scout
from cmc_bbdm.mavis.state_candidates import build_source_candidate_batch
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    apply_action,
    budget_record,
    candidate_budget_record,
    fitting_actions,
    measurement_mask,
)


class _DeterministicEncoder:
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        rows = []
        for image in images:
            channel_means = np.mean(image, axis=(0, 1), dtype=np.float64)
            rows.append(np.resize(channel_means, 512))
        return np.asarray(rows, dtype=np.float64)


def _scouted(authority: MAVISAuthority):
    context = authority.policy_context("sample-001")
    return reveal_uniform_scout(
        authority,
        context,
        initial_budget=0.015625,
        checkpoint=0.25,
    )


def test_mavis_source_candidates_have_exact_incremental_costs() -> None:
    authority = synthetic_authority()
    state = _scouted(authority)
    batch = build_source_candidate_batch(
        authority,
        state,
        dataset_id="domain-a",
        endpoint_budget=0.25,
        encoder=_DeterministicEncoder(),
    )

    grid = build_acquisition_grid(*state.native_shape, initial_budget=state.initial_budget)
    expected_actions = fitting_actions(grid, state.measurement_state, 0.25)
    assert batch.actions == expected_actions
    assert batch.current_embedding.shape == (512,)
    assert batch.candidate_embeddings.shape == (len(expected_actions), 512)
    for action, cost in zip(batch.actions, batch.candidate_costs, strict=True):
        candidate = candidate_budget_record(grid, state.measurement_state, action)
        assert cost == candidate.measured_count - state.exact_acquired_count
        assert candidate.effective_budget <= 0.25


def test_mavis_source_candidates_separate_action_cap_from_global_endpoint() -> None:
    authority = synthetic_authority()
    state = reveal_uniform_scout(
        authority,
        authority.policy_context("sample-001"),
        initial_budget=0.015625,
        checkpoint=0.0625,
    )
    batch = build_source_candidate_batch(
        authority,
        state,
        dataset_id="domain-a",
        endpoint_budget=0.25,
        action_budget=0.0625,
        encoder=_DeterministicEncoder(),
    )

    grid = build_acquisition_grid(*state.native_shape, initial_budget=state.initial_budget)
    assert batch.actions == fitting_actions(grid, state.measurement_state, 0.0625)
    assert batch.endpoint_budget == 0.25


def test_mavis_source_candidates_ignore_content_outside_one_step_reveals() -> None:
    inputs = synthetic_inputs(true_cai=0.4)
    first = MAVISAuthority.from_arrays(**inputs)
    first_state = _scouted(first)
    grid = build_acquisition_grid(
        *first_state.native_shape,
        initial_budget=first_state.initial_budget,
    )
    actions = fitting_actions(grid, first_state.measurement_state, 0.25)
    visible = measurement_mask(grid, first_state.measurement_state)
    for action in actions:
        visible |= measurement_mask(
            grid,
            apply_action(grid, first_state.measurement_state, action),
        )
    altered = np.asarray(inputs["images"][0]).copy()
    altered[~visible] = 255 - altered[~visible]
    changed_inputs = dict(inputs)
    changed_inputs["images"] = (altered,)
    changed_inputs["targets"] = np.asarray([900.0], dtype=np.float64)
    second = MAVISAuthority.from_arrays(**changed_inputs)
    second_state = _scouted(second)

    first_batch = build_source_candidate_batch(
        first,
        first_state,
        dataset_id="domain-a",
        endpoint_budget=0.25,
        encoder=_DeterministicEncoder(),
    )
    second_batch = build_source_candidate_batch(
        second,
        second_state,
        dataset_id="domain-a",
        endpoint_budget=0.25,
        encoder=_DeterministicEncoder(),
    )

    assert first_state.state_sha256 == second_state.state_sha256
    assert first_batch.actions == second_batch.actions
    assert first_batch.candidate_costs == second_batch.candidate_costs
    np.testing.assert_array_equal(
        first_batch.current_embedding,
        second_batch.current_embedding,
    )
    np.testing.assert_array_equal(
        first_batch.candidate_embeddings,
        second_batch.candidate_embeddings,
    )
    assert budget_record(grid, first_state.measurement_state).measured_count == (
        first_state.exact_acquired_count
    )
