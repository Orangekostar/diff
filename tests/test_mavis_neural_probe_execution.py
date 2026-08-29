from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_mavis_mris_execution import _feature_bank

from cmc_bbdm.mavis.neural_probe.artifacts import (
    assign_directional_gate,
    evaluate_n1_comparison,
    verify_artifact_integrity,
    write_artifact_integrity,
)
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


def _comparison_predictions(*, error: float) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_index in range(6):
        domain = f"d{domain_index}"
        for specimen_index in range(2):
            specimen = f"{domain}-{specimen_index}"
            for checkpoint_index, cost in enumerate((10, 20)):
                rows.append(
                    {
                        "outer_domain": domain,
                        "state_id": f"{specimen}-{checkpoint_index}",
                        "specimen_id": specimen,
                        "trajectory_id": f"trajectory-{specimen}",
                        "method": "uniform",
                        "seed": None,
                        "nominal_checkpoint": 0.1 * (checkpoint_index + 1),
                        "exact_acquired_cost": cost,
                        "native_count": 100,
                        "effective_budget": cost / 100.0,
                        "mode": "real",
                        "target": 0.0,
                        "prediction": error,
                        "absolute_error": error,
                        "model_state_sha256": "a" * 64,
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def test_n1_comparison_uses_registered_sign_and_gate() -> None:
    comparison = evaluate_n1_comparison(
        spatial_predictions=_comparison_predictions(error=0.1),
        deepsets_predictions=_comparison_predictions(error=0.2),
        domain_order=tuple(f"d{index}" for index in range(6)),
        replicates=100,
        seed=20260825,
    )

    assert comparison.point_estimate == pytest.approx(0.1)
    assert comparison.ci95_lower == pytest.approx(0.1)
    assert comparison.ci95_upper == pytest.approx(0.1)
    assert comparison.favorable_domain_count == 6
    assert comparison.gate == "REPRESENTATION_STRONG_GO"
    assert comparison.domain_metrics.get_column(
        "deepsets_minus_spatial"
    ).to_list() == pytest.approx([0.1] * 6)

    assert (
        assign_directional_gate(
            prefix="REPRESENTATION",
            point_estimate=0.01,
            ci95_lower=-0.01,
            ci95_upper=0.02,
            favorable_domain_count=4,
        )
        == "REPRESENTATION_PROMISING"
    )
    assert (
        assign_directional_gate(
            prefix="REPRESENTATION",
            point_estimate=-0.01,
            ci95_lower=-0.02,
            ci95_upper=0.01,
            favorable_domain_count=2,
        )
        == "REPRESENTATION_NO_GO"
    )


def test_neural_probe_artifact_integrity_round_trip(tmp_path: Path) -> None:
    (tmp_path / "REPORT.md").write_text("# Test\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")

    write_artifact_integrity(
        tmp_path,
        artifact="mavis_neural_probe_test",
        base_commit="9" * 40,
        config_sha256="a" * 64,
    )

    manifest = verify_artifact_integrity(tmp_path)
    assert manifest["artifact"] == "mavis_neural_probe_test"
    assert set(manifest["files"]) == {"REPORT.md", "summary.json"}
