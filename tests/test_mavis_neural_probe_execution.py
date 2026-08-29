from __future__ import annotations

import json

import numpy as np
import polars as pl
from test_mavis_mris_execution import _feature_bank

from cmc_bbdm.mavis.neural_probe.execution import run_spatial_mris_outer_domain
from cmc_bbdm.mavis.neural_probe.training import (
    load_fitted_spatial_mris_checkpoint,
)


def test_spatial_outer_worker_preserves_modes_and_nested_lodo(tmp_path) -> None:
    bank = _feature_bank()

    complete_path = run_spatial_mris_outer_domain(
        bank,
        outer_domain="d0",
        output_root=tmp_path,
        trainable_modes=("real",),
        learning_rate=0.001,
        max_epochs=2,
        patience=1,
        batch_size=4,
        seed=20260825,
        device="cpu",
        base_commit="9" * 40,
        config_sha256="a" * 64,
    )

    root = tmp_path / "d0"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    predictions = pl.read_parquet(root / "predictions.parquet")
    audits = pl.read_parquet(root / "model_selection_audit.parquet")

    assert complete_path == root / "complete.json"
    assert complete["architecture_name"] == "spatial_grid_cnn_v1"
    assert complete["feature_bank_input_state_sha256"] == bank.input_state_sha256
    assert complete["feature_bank_target_state_sha256"] == bank.target_state_sha256
    assert complete["inner_fold_count"] == 5
    assert complete["target_data_used_for_selection"] is False
    assert set(predictions.get_column("mode")) == {"real", "reconstruction"}
    assert set(predictions.get_column("outer_domain")) == {"d0"}
    assert set(predictions.get_column("specimen_id")) == {"d0-0", "d0-1"}
    assert audits.filter(pl.col("record_type") == "inner_fold").height == 5
    assert audits.filter(pl.col("record_type") == "final_refit").height == 1
    assert not any(
        "d0" in domains for domains in audits.get_column("fit_domains").to_list()
    )
    assert not audits.get_column("target_data_used_for_selection").any()

    inner_paths = sorted((root / "checkpoints/inner").glob("*.npz"))
    assert [path.stem for path in inner_paths] == [
        f"{domain}__real" for domain in bank.domain_order if domain != "d0"
    ]
    for path in inner_paths:
        fitted = load_fitted_spatial_mris_checkpoint(
            path,
            expected_base_commit="9" * 40,
            expected_feature_bank_input_sha256=bank.input_state_sha256,
            expected_feature_bank_target_sha256=bank.target_state_sha256,
            expected_config_sha256="a" * 64,
        )
        validation_domain = path.stem.split("__", 1)[0]
        assert fitted.audit.validation_domains == (validation_domain,)
        assert validation_domain not in fitted.audit.fit_domains
        assert "d0" not in fitted.audit.fit_domains

    final = load_fitted_spatial_mris_checkpoint(
        root / "checkpoints/real.npz",
        expected_base_commit="9" * 40,
        expected_feature_bank_input_sha256=bank.input_state_sha256,
        expected_feature_bank_target_sha256=bank.target_state_sha256,
        expected_config_sha256="a" * 64,
    )
    assert final.mode == "real"
    assert final.outer_domain == "d0"
    assert set(final.audit.fit_domains) == {"d1", "d2", "d3", "d4", "d5"}


def test_spatial_mode_inputs_preserve_registered_controls() -> None:
    bank = _feature_bank()
    real = bank.model_inputs("real", outer_domain="d0")
    positions = bank.model_inputs("positions_only", outer_domain="d0")
    shuffled = bank.model_inputs("shuffled", outer_domain="d0")
    static = bank.model_inputs("static", outer_domain="d0")

    np.testing.assert_array_equal(real.token_masks, positions.token_masks)
    np.testing.assert_array_equal(real.cost_features, positions.cost_features)
    np.testing.assert_array_equal(real.token_features[:, :, :2], positions.token_features[:, :, :2])
    np.testing.assert_array_equal(positions.token_features[:, :, 2:5], 0.0)
    assert not np.array_equal(real.token_features[:, :, 2:5], shuffled.token_features[:, :, 2:5])
    np.testing.assert_array_equal(shuffled.token_masks, real.token_masks)
    np.testing.assert_array_equal(shuffled.cost_features, real.cost_features)
    np.testing.assert_array_equal(static.token_features, 0.0)
    np.testing.assert_array_equal(static.token_masks, False)
    np.testing.assert_array_equal(static.cost_features[:, 0], 0.0)
    np.testing.assert_array_equal(static.cost_features[:, 1], 1.0)
