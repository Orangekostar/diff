from __future__ import annotations

import numpy as np
import torch
from mavis_test_support import synthetic_authority

from cmc_bbdm.mavis.reveal import reveal_action, reveal_uniform_scout
from cmc_bbdm.mavis.state_encoder import (
    MRISStateEncoder,
    build_mris_input,
    build_mris_shuffle_mapping,
    build_shuffled_mris_input,
    summarize_mris_input,
)
from cmc_bbdm.mva.measurement_state import RefinementAction


def _state():
    authority = synthetic_authority()
    state = reveal_uniform_scout(
        authority,
        authority.policy_context("sample-001"),
        initial_budget=0.015625,
        checkpoint=0.25,
    )
    return reveal_action(authority, state, RefinementAction(0, 0, 1))


def test_mavis_state_is_permutation_invariant() -> None:
    torch.manual_seed(20260825)
    state_input = build_mris_input(_state(), mode="real")
    permutation = np.random.Generator(np.random.PCG64(7)).permutation(
        state_input.acquired_positions.shape[0]
    )
    permuted = state_input.permuted(permutation)
    encoder = MRISStateEncoder(
        context_dimension=34,
        hidden_dimension=64,
        output_dimension=64,
    ).eval()

    with torch.inference_mode():
        first = encoder(state_input)
        second = encoder(permuted)

    torch.testing.assert_close(first, second, rtol=0.0, atol=1.0e-6)
    assert first.shape == (64,)


def test_mavis_batched_summary_matches_single_state_encoder() -> None:
    torch.manual_seed(20260825)
    state_input = build_mris_input(_state(), mode="real")
    summary = summarize_mris_input(state_input)
    encoder = MRISStateEncoder(
        context_dimension=34,
        hidden_dimension=64,
        output_dimension=64,
    ).eval()

    with torch.inference_mode():
        single = encoder(state_input)
        batched = encoder.forward_batch(
            torch.tensor(summary.context_features[None], dtype=torch.float32),
            torch.tensor(summary.token_features[None], dtype=torch.float32),
            torch.tensor(summary.token_mask[None], dtype=torch.bool),
            torch.tensor(summary.cost_features[None], dtype=torch.float32),
        )[0]

    torch.testing.assert_close(single, batched, rtol=0.0, atol=1.0e-6)
    assert summary.token_features.shape == (64, 6)
    assert summary.token_mask.shape == (64,)
    assert int(summary.token_mask.sum()) <= 64


def test_mavis_positions_only_contains_no_measurement_values() -> None:
    state = _state()

    positions_only = build_mris_input(state, mode="positions_only")
    real = build_mris_input(state, mode="real")
    static = build_mris_input(state, mode="static")

    assert positions_only.measurement_values is None
    assert positions_only.acquired_positions.shape == state.acquired_positions.shape
    assert real.measurement_values is not None
    np.testing.assert_array_equal(real.measurement_values, state.measurement_values)
    assert static.measurement_values is None
    assert static.acquired_positions.shape == (0, 2)
    assert static.effective_budget == 0.0


def test_mavis_real_and_positions_modes_are_structurally_distinct() -> None:
    torch.manual_seed(20260825)
    state = _state()
    encoder = MRISStateEncoder(
        context_dimension=34,
        hidden_dimension=64,
        output_dimension=64,
    ).eval()

    with torch.inference_mode():
        real = encoder(build_mris_input(state, mode="real"))
        positions = encoder(build_mris_input(state, mode="positions_only"))

    assert not torch.equal(real, positions)


def _shuffle_authority():
    generator = np.random.Generator(np.random.PCG64(20260825))
    specimen_ids = ("source-a", "source-b", "target-a", "target-b")
    dataset_ids = ("source", "source", "target", "target")
    images = tuple(
        np.full((41 + index, 43 + index, 3), 31 + 47 * index, dtype=np.uint8)
        for index in range(4)
    )
    metadata = np.zeros((4, 13), dtype=np.float64)
    metadata[:, 1] = 16.0 / 24.0
    metadata[:, 2] = 1.0
    metadata[:, 9] = np.log1p(6.0)
    profiles = generator.normal(size=(4, 21))
    from cmc_bbdm.mavis.authority import MAVISAuthority

    return MAVISAuthority.from_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        images=images,
        targets=np.linspace(0.2, 0.5, 4),
        metadata13=metadata,
        profile_stats21=profiles,
    )


def test_mavis_shuffled_content_breaks_specimen_coupling() -> None:
    authority = _shuffle_authority()
    mapping = build_mris_shuffle_mapping(
        authority,
        outer_domain="target",
        seed=20260821,
    )
    repeated = build_mris_shuffle_mapping(
        authority,
        outer_domain="target",
        seed=20260821,
    )

    assert mapping == repeated
    assert {row.recipient_id for row in mapping} == set(authority.specimen_ids)
    assert all(row.recipient_id != row.donor_id for row in mapping)
    assert all(row.recipient_domain == row.donor_domain for row in mapping)
    assert all(row.relaxation == "dataset_exact_layup_energy_bin" for row in mapping)

    recipient_id = "target-a"
    recipient_state = reveal_uniform_scout(
        authority,
        authority.policy_context(recipient_id),
        initial_budget=0.015625,
        checkpoint=0.25,
    )
    donor_id = next(
        row.donor_id for row in mapping if row.recipient_id == recipient_id
    )
    real = build_mris_input(recipient_state, mode="real")
    shuffled = build_shuffled_mris_input(
        recipient_state,
        authority=authority,
        donor_specimen_id=donor_id,
    )

    assert shuffled.specimen_id == recipient_id
    assert shuffled.content_specimen_id == donor_id
    assert shuffled.content_specimen_id != shuffled.specimen_id
    assert shuffled.mode == "shuffled"
    assert shuffled.effective_budget == real.effective_budget
    assert shuffled.remaining_budget == real.remaining_budget
    np.testing.assert_array_equal(
        shuffled.acquired_positions,
        real.acquired_positions,
    )
    assert not np.array_equal(shuffled.measurement_values, real.measurement_values)
