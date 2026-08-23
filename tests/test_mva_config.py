from __future__ import annotations

from pathlib import Path

import pytest

from cmc_bbdm.mva.config import MVAConfigError, load_mva_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mva_a0_a3.yaml"


def test_mva_config_freezes_a0_a3_protocol() -> None:
    config = load_mva_config(CONFIG, project_root=ROOT)

    assert config.schema_version == 1
    assert config.scope == "mva_a0_a3_oracle_headroom"
    assert config.seed == 20260823
    assert config.domain_order == (
        "74t7kcdgkr",
        "cgtnjyggtm",
        "w68dtmpfyf",
        "xcmzfsbd9t",
        "yfxyg8jm46",
        "ykhs7s2dck",
    )
    assert config.specimen_count == 276
    assert config.initial_budgets == (0.015625, 0.03125, 0.0625)
    assert config.checkpoints == (
        0.03125,
        0.0625,
        0.09375,
        0.125,
        0.1875,
        0.25,
        0.5,
        1.0,
    )
    assert config.auebc_range == (0.0625, 0.25)
    assert config.pca_dimensions == (8, 16, 32)
    assert config.ridge_alpha == 10.0
    assert config.full_mae == 0.08963580465761432
    assert config.baseline_tolerance == 1.0e-12
    assert config.cell_shape == (8, 8)
    assert config.random_seeds == tuple(range(2026082300, 2026082400))
    assert config.bootstrap_seed == 20260823
    assert config.bootstrap_resamples == 100_000
    assert config.low_budget == 0.125
    assert config.h1_relative_improvement == 0.05
    assert config.h1_minimum_domains == 4
    assert config.h4_relative_auebc == 0.10
    assert config.h4_b5_saving == 0.25
    assert config.methods == (
        "uniform",
        "random",
        "appearance_oracle",
        "reconstruction_oracle",
        "mechanical_oracle",
    )
    assert config.output_dir == Path("results/mva")
    assert set(config.sources) == {
        "mgmr_config",
        "p1_predictions",
        "p5_config",
        "p5_sampling",
        "paired_feature_bank",
        "resnet_weights",
        "protocol",
    }
    for source in config.sources.values():
        assert not source.path.is_absolute()
        assert ".." not in source.path.parts
        assert len(source.sha256) == 64


@pytest.mark.parametrize(
    "old,new",
    (
        ("schema_version: 1", "schema_version: 2"),
        ("ridge_alpha: 10.0", "ridge_alpha: 1.0"),
        ("resamples: 100000", "resamples: 99999"),
        ("minimum_improved_domains: 4", "minimum_improved_domains: 3"),
        ("  - mechanical_oracle", "  - policy_oracle"),
    ),
)
def test_mva_config_rejects_frozen_protocol_drift(
    tmp_path: Path, old: str, new: str
) -> None:
    payload = CONFIG.read_text(encoding="utf-8")
    assert payload.count(old) == 1
    changed = tmp_path / "mva.yaml"
    changed.write_text(payload.replace(old, new), encoding="utf-8")

    with pytest.raises(MVAConfigError):
        load_mva_config(changed, project_root=ROOT)


def test_mva_config_rejects_a4_a7_sections(tmp_path: Path) -> None:
    changed = tmp_path / "mva.yaml"
    changed.write_text(
        CONFIG.read_text(encoding="utf-8") + "\nglobal_mask:\n  enabled: false\n",
        encoding="utf-8",
    )

    with pytest.raises(MVAConfigError, match="keys"):
        load_mva_config(changed, project_root=ROOT)


def test_mva_config_rejects_source_hash_drift(tmp_path: Path) -> None:
    payload = CONFIG.read_text(encoding="utf-8")
    source = next(iter(load_mva_config(CONFIG, project_root=ROOT).sources.values()))
    changed = tmp_path / "mva.yaml"
    changed.write_text(payload.replace(source.sha256, "0" * 64, 1), encoding="utf-8")

    with pytest.raises(MVAConfigError, match="hash"):
        load_mva_config(changed, project_root=ROOT)
