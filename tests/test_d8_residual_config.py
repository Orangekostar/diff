from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from cmc_bbdm.cpb_diffusion_marginalization.residual_config import (
    ResidualConfigError,
    load_residual_diffusion_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_residual_diffusion.yaml"


def _payload() -> dict[str, object]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "d8_residual_diffusion.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_registered_residual_config_is_preouter_and_exact() -> None:
    config = load_residual_diffusion_config(CONFIG, project_root=PROJECT_ROOT)
    assert config.schema_version == 1
    assert config.scope == "cpb_d8_residual_diffusion_preouter"
    assert config.outer_evaluation_count == 0
    assert config.output_dir == "results/d8_residual_diffusion_search"
    assert config.replay_output_dir == (
        "results/replay/d8_residual_diffusion_search"
    )
    assert config.candidate_ids == tuple(f"RD{index}" for index in range(8))
    assert config.screening_epochs == 24
    assert config.screening_seed == 20260823
    assert config.finalists_per_outer == 2
    assert config.rerank_epochs == 120
    assert config.training_seeds == (20260823, 20260824, 20260825)
    assert config.objective_weights == (1.0, 0.25, 0.10)
    assert config.minimum_overall_acceptance == 0.80
    assert config.minimum_domain_acceptance == 0.60
    assert config.promotion_margin == 1.0e-4
    assert config.ensemble_margin == 1.0e-4
    assert config.train_timesteps == 1000
    assert config.sample_steps == 25
    assert config.sample_eta == 1.0
    assert config.batch_size == 32
    assert config.learning_rate == 2.0e-4
    assert config.weight_decay == 1.0e-4
    assert config.config_sha256 == hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def test_registered_residual_candidates_match_frozen_design() -> None:
    config = load_residual_diffusion_config(CONFIG, project_root=PROJECT_ROOT)
    expected = {
        "RD0": (32, "epsilon", "squared_cosine", False, 0.00, 0.00),
        "RD1": (32, "epsilon", "squared_cosine", False, 0.05, 0.10),
        "RD2": (32, "v_prediction", "squared_cosine", False, 0.05, 0.10),
        "RD3": (32, "sample", "squared_cosine", False, 0.05, 0.10),
        "RD4": (64, "epsilon", "squared_cosine", True, 0.05, 0.10),
        "RD5": (64, "v_prediction", "squared_cosine", True, 0.05, 0.10),
        "RD6": (32, "epsilon", "linear", False, 0.05, 0.10),
        "RD7": (32, "v_prediction", "linear", False, 0.05, 0.10),
    }
    observed = {
        candidate_id: (
            candidate.base_channels,
            candidate.prediction_type,
            candidate.beta_schedule,
            candidate.bottleneck_attention,
            candidate.spectral_weight,
            candidate.low_pass_weight,
        )
        for candidate_id in config.candidate_ids
        for candidate in (config.candidate(candidate_id),)
    }
    assert observed == expected


def test_registered_residual_config_binds_pilot_decision_and_runtime() -> None:
    config = load_residual_diffusion_config(CONFIG, project_root=PROJECT_ROOT)
    assert set(config.sources) == {
        "prompt",
        "pilot_decision",
        "residual_training_design",
        "residual_training_plan",
        "exploration_config",
        "pilot_manifest",
        "escalation_evidence",
        "d8_requirements",
        "resnet_weights",
    }
    assert config.sources["pilot_decision"].sha256 == (
        "d8a450946a2cbbc1085572900aaa01eb2bab76b3c7356ede8839bbaed012ddcd"
    )
    assert config.sources["pilot_manifest"].sha256 == (
        "3e9a5af42bddb78a9c573265313dd6db6a1b0a6b3bc39d57713ee3b6130070d4"
    )
    assert config.sources["escalation_evidence"].sha256 == (
        "c55e017303297ac2d0f101ce8e517b7092c8d542109d3d65c6f812592a456ba4"
    )
    assert config.pilot_decision == "TRAIN_RESIDUAL_DIFFUSION"
    assert config.pilot_outer_evaluation_count == 0
    assert config.pilot_scientific_digest == (
        "3478d97858236c1873c88d8fc3e910dbe659e05d2c4e472eac15825e999474ca"
    )
    assert config.runtime["diffusers_module"] == "0.39.0"


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    (
        ("pilot", "decision", "FREEZE_PILOT_FOR_OUTER_EVALUATION"),
        ("pilot", "outer_evaluation_count", 1),
        ("search", "screening_epochs", 25),
        ("search", "rerank_epochs", True),
        ("training", "batch_size", 16),
        ("sampling", "steps", 50),
        ("outputs", "output_dir", "results/d8_final"),
    ),
)
def test_residual_config_rejects_frozen_value_and_type_drift(
    tmp_path: Path,
    section: str,
    key: str,
    replacement: object,
) -> None:
    payload = _payload()
    target = payload[section]
    assert isinstance(target, dict)
    target[key] = replacement
    with pytest.raises(ResidualConfigError):
        load_residual_diffusion_config(
            _write(tmp_path, payload), project_root=PROJECT_ROOT
        )


def test_residual_config_rejects_unknown_duplicate_and_source_hash_drift(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["extra"] = True
    with pytest.raises(ResidualConfigError, match="keys"):
        load_residual_diffusion_config(
            _write(tmp_path, payload), project_root=PROJECT_ROOT
        )

    payload = _payload()
    candidates = payload["candidates"]
    assert isinstance(candidates, dict)
    rd0 = candidates["RD0"]
    assert isinstance(rd0, dict)
    rd0["extra"] = 1
    with pytest.raises(ResidualConfigError, match="RD0"):
        load_residual_diffusion_config(
            _write(tmp_path, payload), project_root=PROJECT_ROOT
        )

    payload = _payload()
    sources = payload["sources"]
    assert isinstance(sources, dict)
    manifest = sources["pilot_manifest"]
    assert isinstance(manifest, dict)
    manifest["sha256"] = "0" * 64
    with pytest.raises(ResidualConfigError, match="hash"):
        load_residual_diffusion_config(
            _write(tmp_path, payload), project_root=PROJECT_ROOT
        )

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "schema_version: 1\nschema_version: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ResidualConfigError, match="duplicate"):
        load_residual_diffusion_config(duplicate, project_root=PROJECT_ROOT)


def test_residual_config_is_deeply_immutable() -> None:
    config = load_residual_diffusion_config(CONFIG, project_root=PROJECT_ROOT)
    with pytest.raises(TypeError):
        config.sources["other"] = config.sources["prompt"]  # type: ignore[index]
    with pytest.raises(TypeError):
        config.candidates["RD0"] = config.candidates["RD1"]  # type: ignore[index]
