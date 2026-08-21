from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from cmc_bbdm.cpb_diffusion_marginalization.config import (
    DOMAIN_ORDER,
    D8ConfigError,
    load_d8_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_exploration.yaml"


def _payload() -> dict[str, object]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "d8.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_d8_config_freezes_goal_budget_and_success_gates() -> None:
    config = load_d8_config(CONFIG, project_root=PROJECT_ROOT)
    assert config.schema_version == 1
    assert config.scope == "cpb_d8_morphology_preserving_marginalization"
    assert config.seed == 20260820
    assert config.outer_domains == DOMAIN_ORDER
    assert config.baseline_mae == 0.08963580465761432
    assert config.positive_mae == 0.085154
    assert config.strong_mae == 0.082465
    assert config.stretch_mae == 0.080
    assert config.forced_trials == 12
    assert config.optuna_trials == 60
    assert config.rerank_seeds == (20260820, 20260821, 20260822)
    assert config.p6_draws == 8
    assert config.output_dir == "results/d8_search"
    assert config.escalation == {
        "p6_candidate_minimum_inner_mae_improvement": 0.01,
        "p6_candidate_minimum_inner_domains": 3,
        "p6_candidate_minimum_outer_studies": 3,
        "low_band_energy_fraction": 0.50,
        "low_acceptance_threshold": 0.50,
        "low_acceptance_alpha": 0.10,
        "mismatch_minimum_outer_studies": 3,
        "pilot_freeze_minimum_objective_gain": 0.0001,
        "pilot_freeze_minimum_diffusion_weight": 0.05,
        "pilot_freeze_minimum_outer_studies": 3,
        "decision_priority": (
            "TRAIN_RESIDUAL_DIFFUSION",
            "FREEZE_PILOT_FOR_OUTER_EVALUATION",
            "CLOSE_DIFFUSION_SPECIFIC_ROUTE",
        ),
    }
    assert config.config_sha256 == hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def test_d8_config_preserves_prompt_search_scope_and_ablation_roles() -> None:
    config = load_d8_config(CONFIG, project_root=PROJECT_ROOT)
    decomposition = config.search_space["decomposition"]
    controls = config.search_space["controls"]
    features = config.search_space["features"]
    gate = config.search_space["morphology_gate"]

    assert decomposition["bands"] == ("low", "mid", "mid+high", "high")
    assert decomposition["gaussian_sigma_pixels"] == {
        "minimum": 0.5,
        "maximum": 8.0,
        "distribution": "log_uniform",
    }
    assert decomposition["fourier_cutoff_fraction"] == {
        "minimum": 0.04,
        "maximum": 0.50,
        "distribution": "uniform",
    }
    assert decomposition["fourier_transition_fraction"] == {
        "minimum": 0.01,
        "maximum": 0.10,
        "distribution": "uniform",
    }
    assert decomposition["alpha"] == {
        "minimum": -0.5,
        "maximum": 1.0,
        "distribution": "uniform",
    }
    assert controls["B1"] == "morphology_component_only"
    assert controls["B6"] == "diffusion_plus_consistency"
    assert controls["B7"] == "diffusion_plus_test_time_marginalization"
    assert controls["B8"] == "full_proposed_pipeline"
    assert features["K_train"] == (1, 2, 4, 8)
    assert features["K_test"] == (1, 2, 4, 8, 16)
    assert features["layers"] == ("global", "layer3", "multi_layer")
    assert features["marginalization_stage"] == ("feature", "prediction")
    assert "morphology_weighted" in features["prediction_aggregation"]
    assert features["consistency"] == (
        "none",
        "prediction_variance",
        "feature_variance",
        "pairwise_ranking",
    )
    assert features["consistency_weight"] == {
        "minimum": 0.0,
        "maximum": 1.0,
        "distribution": "uniform",
    }
    assert gate["low_frequency_sigma_pixels"] == 2.0
    assert gate["radial_profile_bins"] == 16


def test_d8_config_binds_all_upstream_authorities() -> None:
    config = load_d8_config(CONFIG, project_root=PROJECT_ROOT)
    assert set(config.sources) == {
        "prompt",
        "design",
        "exploration_plan",
        "implementation_plan",
        "p1_config",
        "p1_manifest",
        "p1_predictions",
        "p1_inner_selection",
        "p5_config",
        "p5_manifest",
        "p6_config",
        "p6_manifest",
        "p6_uncertainty_source",
        "resnet_weights",
        "runtime_requirements",
        "d8_requirements",
        "dockerfile",
        "d8_package_init",
        "d8_config_code",
        "d8_authority_code",
        "d8_baseline_code",
        "d8_residuals_code",
        "d8_decomposition_code",
        "d8_variants_code",
        "d8_features_code",
        "d8_regression_code",
        "d8_search_code",
        "d8_tracking_code",
        "d8_selection_code",
        "d8_pilot_code",
        "d8_artifacts_code",
        "d8_cli_code",
        "d8_cli_wrapper",
    }
    assert config.sources["prompt"].path.endswith("Cross-Domain CAI.md")
    assert config.sources["p1_manifest"].path == (
        "paper_v3/experiments/P1_full_field_oracle/artifact_manifest.json"
    )
    assert config.sources["p6_manifest"].path == (
        "results/cpb_spatial/p6_diffusion_reconstruction/artifact_manifest.json"
    )
    assert config.sources["d8_search_code"].path.endswith("/search.py")
    assert config.sources["d8_tracking_code"].path.endswith("/tracking.py")
    assert config.sources["d8_selection_code"].path.endswith("/selection.py")
    assert config.sources["d8_pilot_code"].path.endswith("/pilot.py")
    assert config.sources["d8_artifacts_code"].path.endswith("/artifacts.py")
    assert config.sources["d8_cli_code"].path == "scripts/run_d8_exploration.py"
    assert config.sources["d8_cli_wrapper"].path == "scripts/run_d8_exploration.sh"


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    (
        ("baseline", "equal_domain_mae", 0.09),
        ("search", "optuna_trials", 59),
        ("search", "forced_trials", True),
        ("posterior", "p6_draws", 4),
        ("outputs", "search_dir", "results/other"),
    ),
)
def test_d8_config_rejects_frozen_value_and_type_drift(
    tmp_path: Path, section: str, key: str, replacement: object
) -> None:
    payload = _payload()
    target = payload[section]
    assert isinstance(target, dict)
    target[key] = replacement
    with pytest.raises(D8ConfigError):
        load_d8_config(_write(tmp_path, payload), project_root=PROJECT_ROOT)


def test_d8_config_rejects_unknown_duplicate_path_and_hash_drift(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["extra"] = True
    with pytest.raises(D8ConfigError, match="keys"):
        load_d8_config(_write(tmp_path, payload), project_root=PROJECT_ROOT)

    payload = _payload()
    sources = payload["sources"]
    assert isinstance(sources, dict)
    prompt = sources["prompt"]
    assert isinstance(prompt, dict)
    prompt["path"] = "../escape"
    with pytest.raises(D8ConfigError, match="path"):
        load_d8_config(_write(tmp_path, payload), project_root=PROJECT_ROOT)

    payload = _payload()
    sources = payload["sources"]
    assert isinstance(sources, dict)
    manifest = sources["p6_manifest"]
    assert isinstance(manifest, dict)
    manifest["sha256"] = "0" * 64
    with pytest.raises(D8ConfigError, match="hash"):
        load_d8_config(_write(tmp_path, payload), project_root=PROJECT_ROOT)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "schema_version: 1\nschema_version: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(D8ConfigError, match="duplicate"):
        load_d8_config(duplicate, project_root=PROJECT_ROOT)


def test_d8_config_rejects_nested_unknown_keys_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    payload = _payload()
    search = payload["search"]
    assert isinstance(search, dict)
    search["extra"] = 1
    with pytest.raises(D8ConfigError, match="search"):
        load_d8_config(_write(tmp_path, payload), project_root=PROJECT_ROOT)

    payload = _payload()
    baseline = payload["baseline"]
    assert isinstance(baseline, dict)
    baseline["equal_domain_mae"] = float("nan")
    with pytest.raises(D8ConfigError):
        load_d8_config(_write(tmp_path, payload), project_root=PROJECT_ROOT)
