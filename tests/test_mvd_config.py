from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cmc_bbdm.mvd.config import MVDConfigError, load_mvd_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mvd_feasibility.yaml"


def test_mvd_config_freezes_complete_feasibility_protocol() -> None:
    config = load_mvd_config(CONFIG, project_root=ROOT)

    assert config.scope == "mvd_m0_m1_feasibility"
    assert config.domain_order == (
        "74t7kcdgkr",
        "cgtnjyggtm",
        "w68dtmpfyf",
        "xcmzfsbd9t",
        "yfxyg8jm46",
        "ykhs7s2dck",
    )
    assert config.checkpoints == (0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25)
    assert config.initial_budgets == {
        "74t7kcdgkr": 0.03125,
        "cgtnjyggtm": 0.015625,
        "w68dtmpfyf": 0.015625,
        "xcmzfsbd9t": 0.015625,
        "yfxyg8jm46": 0.015625,
        "ykhs7s2dck": 0.015625,
    }
    assert config.m0_minimum_improved_domains == 4
    assert config.m0_minimum_headroom_retention == 0.20
    assert config.m0_strong_headroom_retention == 0.50
    assert config.m1_minimum_improved_domains == 4
    assert config.m1_strong_advantage_capture == 0.35
    assert config.bootstrap_seed == 20260824
    assert config.bootstrap_resamples == 100_000
    assert config.output_m0 == Path("results/mvd/m0_one_shot_oracle")
    assert config.output_m1 == Path("results/mvd/m1_observability")


def test_mvd_config_rejects_unknown_keys(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MVDConfigError, match="top-level keys changed"):
        load_mvd_config(changed, project_root=ROOT)


def test_mvd_config_rejects_source_hash_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["sources"]["m0_protocol"]["sha256"] = "0" * 64
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MVDConfigError, match="source authority changed: m0_protocol"):
        load_mvd_config(changed, project_root=ROOT)
