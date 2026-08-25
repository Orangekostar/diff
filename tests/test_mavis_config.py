from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cmc_bbdm.mavis.config import MAVISConfigError, load_mavis_config

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "paper_v3/configs/mavis_development.yaml"
FINAL = ROOT / "paper_v3/configs/mavis_final.yaml"
DOMAINS = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)


def test_mavis_config_freezes_causal_protocol_and_recommended_defaults() -> None:
    config = load_mavis_config(DEVELOPMENT, project_root=ROOT)

    assert config.mode == "development"
    assert config.seed == 20260825
    assert config.specimen_count == 276
    assert config.domain_order == DOMAINS
    assert config.context_features == ("metadata13", "profile_stats21")
    assert config.initial_budgets == (0.015625, 0.03125, 0.0625)
    assert config.checkpoints == (0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25)
    assert config.budget_unit == "unique_native_raster_locations"
    assert config.scout_policy == "uniform_geometry_neutral"
    assert config.initial_budget_by_domain == {
        "74t7kcdgkr": 0.03125,
        "cgtnjyggtm": 0.015625,
        "w68dtmpfyf": 0.015625,
        "xcmzfsbd9t": 0.015625,
        "yfxyg8jm46": 0.015625,
        "ykhs7s2dck": 0.015625,
    }
    assert config.trajectory_random_seed == 2026082300
    assert config.teacher_interpolation == "bilinear"
    assert config.teacher_pca_dimensions == (8, 16, 32)
    assert config.teacher_ridge_alpha == 10.0
    assert config.teacher_tie_tolerance == 1.0e-12
    assert config.mris_hidden_size == 64
    assert config.mris_dimension == 64
    assert config.learning_rate == 0.001
    assert config.on_policy_rounds == 3
    assert config.outer_split == "leave_one_dataset_out"
    assert config.inner_split == "leave_one_source_dataset_out"
    assert config.selection_criterion == (
        "source_cai_auebc",
        "source_improved_domains",
        "worst_source_domain_auebc",
        "model_simplicity",
    )
    assert set(config.sources) == {
        "mgmr_config",
        "p0_repo_code_map",
        "p0_data_flow",
        "p0_authority_schema",
        "design_spec",
        "a2_oracle_trajectories",
        "mvd_m0_actions",
        "candidate_bank_0p015625",
        "candidate_bank_0p03125",
    }


def test_mavis_config_rejects_unknown_keys_and_source_hash_drift(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(DEVELOPMENT.read_text(encoding="utf-8"))
    payload["unknown"] = True
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(MAVISConfigError, match="schema"):
        load_mavis_config(unknown, project_root=ROOT)

    payload.pop("unknown")
    payload["sources"]["p0_data_flow"]["sha256"] = "0" * 64
    drift = tmp_path / "drift.yaml"
    drift.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(MAVISConfigError, match="hash"):
        load_mavis_config(drift, project_root=ROOT)


def test_mavis_final_config_is_explicitly_locked_pending_source_selection() -> None:
    config = load_mavis_config(FINAL, project_root=ROOT)

    assert config.mode == "final"
    assert config.final_configuration_frozen is False
    assert config.development_package_sha256 is None
    with pytest.raises(MAVISConfigError, match="not frozen"):
        config.require_finalized()
