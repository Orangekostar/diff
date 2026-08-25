from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mavis.mris_data import (
    build_mris_feature_bank,
    load_mris_feature_bank,
    save_mris_feature_bank,
)
from cmc_bbdm.mavis.reveal import reveal_uniform_scout


def _inputs(targets: np.ndarray) -> tuple[MAVISAuthority, pl.DataFrame]:
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
    profiles = np.arange(84, dtype=np.float64).reshape(4, 21) / 84.0
    authority = MAVISAuthority.from_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        images=images,
        targets=targets,
        metadata13=metadata,
        profile_stats21=profiles,
    )
    rows: list[dict[str, object]] = []
    for index, (specimen_id, domain_id) in enumerate(
        zip(specimen_ids, dataset_ids, strict=True)
    ):
        state = reveal_uniform_scout(
            authority,
            authority.policy_context(specimen_id),
            initial_budget=0.015625,
            checkpoint=0.25,
        )
        other_domain = "target" if domain_id == "source" else "source"
        rows.append(
            {
                "state_id": hashlib.sha256(specimen_id.encode()).hexdigest(),
                "specimen_id": specimen_id,
                "domain_id": domain_id,
                "trajectory_id": hashlib.sha256(f"t-{specimen_id}".encode()).hexdigest(),
                "method": "uniform",
                "seed": None,
                "nominal_checkpoint": 0.03125,
                "exact_acquired_cost": state.exact_acquired_count,
                "native_count": state.native_count,
                "effective_budget": state.effective_budget,
                "context_features": state.context_features.tolist(),
                "revealed_rows": state.acquired_positions[:, 0].tolist(),
                "revealed_columns": state.acquired_positions[:, 1].tolist(),
                "revealed_red": state.measurement_values[:, 0].tolist(),
                "revealed_green": state.measurement_values[:, 1].tolist(),
                "revealed_blue": state.measurement_values[:, 2].tolist(),
                "teacher_outer_domains": [other_domain],
                "strict_oof_cai_predictions": [0.25 + 0.1 * index],
            }
        )
    return authority, pl.DataFrame(rows, infer_schema_length=None)


def test_mavis_feature_bank_separates_inputs_from_target_labels() -> None:
    authority, states = _inputs(np.asarray([0.2, 0.3, 0.4, 0.5]))
    changed, changed_states = _inputs(np.asarray([0.2, 0.3, 400.0, 500.0]))

    first = build_mris_feature_bank(
        states,
        authority=authority,
        domain_order=("source", "target"),
        shuffle_seed=20260821,
    )
    second = build_mris_feature_bank(
        changed_states,
        authority=changed,
        domain_order=("source", "target"),
        shuffle_seed=20260821,
    )

    assert first.input_state_sha256 == second.input_state_sha256
    assert first.donor_specimen_ids == second.donor_specimen_ids
    np.testing.assert_array_equal(first.context_features, second.context_features)
    np.testing.assert_array_equal(first.real_token_features, second.real_token_features)
    np.testing.assert_array_equal(first.shuffled_token_features, second.shuffled_token_features)
    np.testing.assert_array_equal(
        first.reconstruction_predictions,
        second.reconstruction_predictions,
    )
    assert not np.array_equal(first.targets, second.targets)


def test_mavis_feature_bank_modes_share_cost_and_position_contract() -> None:
    authority, states = _inputs(np.asarray([0.2, 0.3, 0.4, 0.5]))
    bank = build_mris_feature_bank(
        states,
        authority=authority,
        domain_order=("source", "target"),
        shuffle_seed=20260821,
    )

    static = bank.model_inputs("static", outer_domain="target")
    positions = bank.model_inputs("positions_only", outer_domain="target")
    real = bank.model_inputs("real", outer_domain="target")
    shuffled = bank.model_inputs("shuffled", outer_domain="target")

    np.testing.assert_array_equal(positions.token_masks, real.token_masks)
    np.testing.assert_array_equal(shuffled.token_masks, real.token_masks)
    np.testing.assert_array_equal(positions.cost_features, real.cost_features)
    np.testing.assert_array_equal(shuffled.cost_features, real.cost_features)
    np.testing.assert_array_equal(positions.token_features[:, :, :2], real.token_features[:, :, :2])
    np.testing.assert_array_equal(positions.token_features[:, :, 5], real.token_features[:, :, 5])
    assert np.count_nonzero(positions.token_features[:, :, 2:5]) == 0
    assert np.count_nonzero(static.token_masks) == 0
    assert np.count_nonzero(static.cost_features[:, (0, 2)]) == 0
    assert all(
        recipient != donor
        for recipient, donor in zip(
            bank.specimen_ids,
            bank.donor_specimen_ids["target"],
            strict=True,
        )
    )


def test_mavis_feature_bank_cache_round_trip_is_hash_bound(tmp_path) -> None:
    authority, states = _inputs(np.asarray([0.2, 0.3, 0.4, 0.5]))
    bank = build_mris_feature_bank(
        states,
        authority=authority,
        domain_order=("source", "target"),
        shuffle_seed=20260821,
    )
    path = tmp_path / "feature_bank.npz"

    save_mris_feature_bank(bank, path)
    restored = load_mris_feature_bank(path)

    assert restored.input_state_sha256 == bank.input_state_sha256
    assert restored.target_state_sha256 == bank.target_state_sha256
    assert restored.donor_specimen_ids == bank.donor_specimen_ids
    np.testing.assert_array_equal(restored.real_token_features, bank.real_token_features)
    np.testing.assert_array_equal(restored.targets, bank.targets)

    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with np.testing.assert_raises(ValueError):
        load_mris_feature_bank(path)
