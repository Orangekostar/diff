from __future__ import annotations

from pathlib import Path

import pytest

from cmc_bbdm.mva.a4_config import A4ConfigError, load_a4_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mva_a4_global_mask.yaml"


def test_a4_config_freezes_outer_safe_static_mask_protocol() -> None:
    config = load_a4_config(CONFIG, project_root=ROOT)

    assert config.schema_version == 1
    assert config.scope == "mva_a4_global_task_mask"
    assert config.seed == 20260823
    assert config.a3_status == "MVA_ORACLE_GO"
    assert config.domain_order == (
        "74t7kcdgkr",
        "cgtnjyggtm",
        "w68dtmpfyf",
        "xcmzfsbd9t",
        "yfxyg8jm46",
        "ykhs7s2dck",
    )
    assert config.specimen_count == 276
    assert config.cell_shape == (8, 8)
    assert config.candidate_from_level == 0
    assert config.candidate_to_level == 1
    assert config.checkpoints == (
        0.03125,
        0.0625,
        0.09375,
        0.125,
        0.1875,
        0.25,
    )
    assert config.auebc_range == (0.0625, 0.25)
    assert config.methods == (
        "global_appearance_mask",
        "global_reconstruction_mask",
        "global_mechanical_mask",
    )
    assert config.rank_aggregation == "equal_domain_mean_normalized_rank"
    assert config.pca_dimensions == (8, 16, 32)
    assert config.ridge_alpha == 10.0
    assert config.primary_prediction_protocol == "P-B"
    assert config.bootstrap_seed == 20260823
    assert config.bootstrap_resamples == 100_000
    assert config.minimum_improved_domains == 4
    assert config.adaptive_gap_threshold == 0.03
    assert config.required_a4_comparisons == (
        "uniform",
        "global_reconstruction_mask",
        "global_appearance_mask",
    )
    assert config.a4_statuses == (
        "MVA_A4_GLOBAL_GO",
        "MVA_A4_GLOBAL_NO_GO",
    )
    assert config.a5_statuses == (
        "MVA_A5_AUTHORIZED",
        "MVA_A5_NOT_AUTHORIZED",
    )
    assert config.output_dir == Path("results/mva/a4_global_task_mask")
    assert config.replay_dir == Path("results/mva/replay/a4_global_task_mask")
    assert set(config.sources) == {
        "a0_a3_config",
        "a1_summary",
        "a2_checksums",
        "a2_manifest",
        "a2_summary",
        "a4_design",
        "a4_protocol",
    }
    for source in config.sources.values():
        assert not source.path.is_absolute()
        assert ".." not in source.path.parts
        assert len(source.sha256) == 64


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("schema_version: 1", "schema_version: 2"),
        ("a3_status: MVA_ORACLE_GO", "a3_status: MVA_ORACLE_NO_GO"),
        ("adaptive_gap_relative: 0.03", "adaptive_gap_relative: 0.02"),
        ("minimum_improved_domains: 4", "minimum_improved_domains: 3"),
        ("resamples: 100000", "resamples: 99999"),
        ("  - global_mechanical_mask", "  - learned_policy"),
    ),
)
def test_a4_config_rejects_frozen_protocol_drift(
    tmp_path: Path, old: str, new: str
) -> None:
    payload = CONFIG.read_text(encoding="utf-8")
    assert payload.count(old) == 1
    changed = tmp_path / "a4.yaml"
    changed.write_text(payload.replace(old, new), encoding="utf-8")

    with pytest.raises(A4ConfigError):
        load_a4_config(changed, project_root=ROOT)


def test_a4_config_rejects_a5_a7_implementation_sections(tmp_path: Path) -> None:
    changed = tmp_path / "a4.yaml"
    changed.write_text(
        CONFIG.read_text(encoding="utf-8")
        + "\nimitation_policy:\n  enabled: false\n",
        encoding="utf-8",
    )

    with pytest.raises(A4ConfigError, match="keys"):
        load_a4_config(changed, project_root=ROOT)


def test_a4_config_rejects_source_hash_drift(tmp_path: Path) -> None:
    payload = CONFIG.read_text(encoding="utf-8")
    source = next(iter(load_a4_config(CONFIG, project_root=ROOT).sources.values()))
    changed = tmp_path / "a4.yaml"
    changed.write_text(payload.replace(source.sha256, "0" * 64, 1), encoding="utf-8")

    with pytest.raises(A4ConfigError, match="hash"):
        load_a4_config(changed, project_root=ROOT)
