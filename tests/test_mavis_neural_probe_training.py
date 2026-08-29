from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest

from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mavis.mris_data import build_mris_feature_bank
from cmc_bbdm.mavis.neural_probe.training import (
    SpatialMRISTrainingError,
    fit_inner_spatial_mris_fold,
    load_fitted_spatial_mris_checkpoint,
    read_spatial_mris_checkpoint_metadata,
    save_fitted_spatial_mris_checkpoint,
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


def _fit(bank):
    return fit_inner_spatial_mris_fold(
        bank,
        mode="real",
        outer_domain="d0",
        validation_domain="d1",
        learning_rate=0.001,
        max_epochs=4,
        patience=2,
        batch_size=4,
        seed=20260825,
        device="cpu",
        base_commit="9" * 40,
        config_sha256="a" * 64,
    )


def test_spatial_p2_target_domain_cannot_change_training() -> None:
    first_bank, _authority = _bank_and_authority(outer_target_delta=0.0)
    changed_bank, _changed_authority = _bank_and_authority(outer_target_delta=1000.0)

    first = _fit(first_bank)
    changed = _fit(changed_bank)

    assert first.state_dict_sha256 == changed.state_dict_sha256
    assert first.model_state_sha256 == changed.model_state_sha256
    assert first.audit.selected_epoch == changed.audit.selected_epoch
    assert set(first.audit.fit_domains) == {"d2", "d3", "d4", "d5"}
    assert first.audit.validation_domains == ("d1",)
    assert not set(first.audit.fit_specimen_ids) & {"d0-0", "d0-1", "d1-0", "d1-1"}


def test_spatial_p2_model_and_checkpoint_round_trip(tmp_path) -> None:
    bank, authority = _bank_and_authority(outer_target_delta=0.0)
    fitted = _fit(bank)

    assert sum(parameter.numel() for parameter in fitted.model.parameters()) == 27_617
    assert fitted.architecture_name == "spatial_grid_cnn_v1"
    assert len(fitted.model_state_sha256) == 64
    assert len(fitted.state_dict_sha256) == 64

    checkpoint = tmp_path / "spatial-real-d0.npz"
    save_fitted_spatial_mris_checkpoint(fitted, checkpoint)
    metadata = read_spatial_mris_checkpoint_metadata(checkpoint)
    assert metadata["checkpoint_type"] == "mavis_spatial_mris"
    assert metadata["schema_version"] == 1
    assert metadata["architecture_name"] == "spatial_grid_cnn_v1"
    assert metadata["base_commit"] == "9" * 40
    assert metadata["feature_bank_input_sha256"] == bank.input_state_sha256
    assert metadata["feature_bank_target_sha256"] == bank.target_state_sha256
    assert metadata["config_sha256"] == "a" * 64
    assert metadata["hyperparameters"] == {
        "batch_size": 4,
        "learning_rate": 0.001,
        "max_epochs": 4,
        "patience": 2,
        "seed": 20260825,
    }

    restored = load_fitted_spatial_mris_checkpoint(
        checkpoint,
        expected_model_state_sha256=fitted.model_state_sha256,
        expected_base_commit="9" * 40,
        expected_feature_bank_input_sha256=bank.input_state_sha256,
        expected_feature_bank_target_sha256=bank.target_state_sha256,
        expected_config_sha256="a" * 64,
    )
    np.testing.assert_array_equal(
        restored.predict(bank, domain="d0", batch_size=4, device="cpu"),
        fitted.predict(bank, domain="d0", batch_size=4, device="cpu"),
    )
    state = reveal_uniform_scout(
        authority,
        authority.policy_context("d0-0"),
        initial_budget=0.015625,
        checkpoint=0.25,
    )
    np.testing.assert_array_equal(
        restored.encode_inspection_state(state, device="cpu"),
        fitted.encode_inspection_state(state, device="cpu"),
    )
    assert restored.predict_inspection_state(state, device="cpu") == (
        fitted.predict_inspection_state(state, device="cpu")
    )


def test_spatial_checkpoint_rejects_wrong_provenance(tmp_path) -> None:
    bank, _authority = _bank_and_authority(outer_target_delta=0.0)
    fitted = _fit(bank)
    checkpoint = tmp_path / "spatial-real-d0.npz"
    save_fitted_spatial_mris_checkpoint(fitted, checkpoint)

    with pytest.raises(SpatialMRISTrainingError, match="config hash changed"):
        load_fitted_spatial_mris_checkpoint(
            checkpoint,
            expected_config_sha256="b" * 64,
        )
    with pytest.raises(SpatialMRISTrainingError, match="base commit changed"):
        load_fitted_spatial_mris_checkpoint(
            checkpoint,
            expected_base_commit="8" * 40,
        )


def test_spatial_training_is_fixed_seed_deterministic() -> None:
    bank, _authority = _bank_and_authority(outer_target_delta=0.0)

    first = _fit(bank)
    second = _fit(bank)

    assert first.state_dict_sha256 == second.state_dict_sha256
    assert first.model_state_sha256 == second.model_state_sha256
    assert first.audit == second.audit
