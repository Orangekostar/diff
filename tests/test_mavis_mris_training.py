from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mavis.mris_data import build_mris_feature_bank
from cmc_bbdm.mavis.mris_training import (
    fit_inner_mris_fold,
    load_fitted_mris_checkpoint,
    save_fitted_mris_checkpoint,
)
from cmc_bbdm.mavis.reveal import reveal_uniform_scout


def _bank_and_authority(*, outer_target_delta: float):
    domains = ("d0", "d1", "d2", "d3", "d4", "d5")
    specimen_ids = tuple(f"{domain}-{index}" for domain in domains for index in range(2))
    dataset_ids = tuple(domain for domain in domains for _ in range(2))
    images = tuple(
        np.full((41, 43, 3), 20 + 10 * index, dtype=np.uint8)
        for index in range(12)
    )
    metadata = np.zeros((12, 13), dtype=np.float64)
    metadata[:, 1] = 16.0 / 24.0
    metadata[:, 2] = 1.0
    metadata[:, 9] = np.log1p(6.0)
    profiles = np.arange(12 * 21, dtype=np.float64).reshape(12, 21) / 252.0
    targets = 0.2 + 0.01 * np.arange(12, dtype=np.float64)
    targets[:2] += outer_target_delta
    authority = MAVISAuthority.from_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        images=images,
        targets=targets,
        metadata13=metadata,
        profile_stats21=profiles,
    )
    rows: list[dict[str, object]] = []
    for specimen_id, domain_id in zip(specimen_ids, dataset_ids, strict=True):
        state = reveal_uniform_scout(
            authority,
            authority.policy_context(specimen_id),
            initial_budget=0.015625,
            checkpoint=0.25,
        )
        teacher_domains = [domain for domain in domains if domain != domain_id]
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
                "teacher_outer_domains": teacher_domains,
                "strict_oof_cai_predictions": [
                    0.25 + 0.01 * index for index in range(len(teacher_domains))
                ],
            }
        )
    return (
        build_mris_feature_bank(
            pl.DataFrame(rows, infer_schema_length=None),
            authority=authority,
            domain_order=domains,
            shuffle_seed=20260821,
        ),
        authority,
    )


def _bank(*, outer_target_delta: float):
    return _bank_and_authority(outer_target_delta=outer_target_delta)[0]


def test_mavis_outer_target_label_cannot_change_inner_training(tmp_path) -> None:
    first_bank, authority = _bank_and_authority(outer_target_delta=0.0)
    changed_bank = _bank(outer_target_delta=1000.0)

    first = fit_inner_mris_fold(
        first_bank,
        mode="real",
        outer_domain="d0",
        validation_domain="d1",
        hidden_dimension=8,
        mris_dimension=8,
        learning_rate=0.001,
        max_epochs=4,
        patience=2,
        batch_size=4,
        seed=20260825,
        device="cpu",
    )
    changed = fit_inner_mris_fold(
        changed_bank,
        mode="real",
        outer_domain="d0",
        validation_domain="d1",
        hidden_dimension=8,
        mris_dimension=8,
        learning_rate=0.001,
        max_epochs=4,
        patience=2,
        batch_size=4,
        seed=20260825,
        device="cpu",
    )

    assert first.model_state_sha256 == changed.model_state_sha256
    assert first.audit.selected_epoch == changed.audit.selected_epoch
    assert set(first.audit.fit_domains) == {"d2", "d3", "d4", "d5"}
    assert first.audit.validation_domains == ("d1",)
    np.testing.assert_array_equal(
        first.encode(
            first_bank,
            state_ids=first_bank.state_ids,
            batch_size=4,
            device="cpu",
        ),
        changed.encode(
            changed_bank,
            state_ids=changed_bank.state_ids,
            batch_size=4,
            device="cpu",
        ),
    )
    np.testing.assert_array_equal(
        first.predict(first_bank, domain="d0", batch_size=4, device="cpu"),
        changed.predict(changed_bank, domain="d0", batch_size=4, device="cpu"),
    )

    checkpoint = tmp_path / "real-d0.npz"
    save_fitted_mris_checkpoint(first, checkpoint)
    restored = load_fitted_mris_checkpoint(
        checkpoint,
        expected_model_state_sha256=first.model_state_sha256,
    )
    assert restored.model_state_sha256 == first.model_state_sha256
    np.testing.assert_array_equal(
        restored.predict(first_bank, domain="d0", batch_size=4, device="cpu"),
        first.predict(first_bank, domain="d0", batch_size=4, device="cpu"),
    )
    states = tuple(
        reveal_uniform_scout(
            authority,
            authority.policy_context(specimen_id),
            initial_budget=0.015625,
            checkpoint=0.25,
        )
        for specimen_id in authority.specimen_ids[:3]
    )
    np.testing.assert_allclose(
        first.encode_inspection_states(states, batch_size=2, device="cpu"),
        np.stack(
            [first.encode_inspection_state(state, device="cpu") for state in states]
        ),
        rtol=0.0,
        atol=1.0e-7,
    )
