from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cmc_bbdm.mgmr.protocol import MGMRProtocolError, load_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mgmr_m0.yaml"


def test_registered_protocol_is_exact() -> None:
    protocol = load_protocol(CONFIG, project_root=ROOT)

    assert protocol.specimen_count == 276
    assert protocol.domain_order == (
        "74t7kcdgkr",
        "cgtnjyggtm",
        "w68dtmpfyf",
        "xcmzfsbd9t",
        "yfxyg8jm46",
        "ykhs7s2dck",
    )
    assert protocol.coarse_density == 0.25
    assert protocol.spatial_layer == "layer3"
    assert protocol.wavelet == "db2"
    assert protocol.wavelet_sensitivity == "haar"
    assert protocol.wavelet_mode == "periodization"
    assert protocol.pca_dimensions == (8, 16, 32)
    assert protocol.ridge_alpha == 10.0
    assert protocol.specificity_seeds == (20260831, 20260901, 20260902)
    assert protocol.bootstrap_seed == 20260822
    assert protocol.bootstrap_resamples == 100000
    assert protocol.gate_required == ("A", "B", "D")
    assert protocol.output_paths == {
        "feature_bank": Path("results/mgmr/feature_bank"),
        "formal": Path("results/mgmr/m0_component_gate"),
        "replay": Path("results/mgmr/replay/m0_component_gate"),
    }
    assert protocol.sources["controlling_prompt"].sha256 == (
        "8ac00f73bb171a8c3b7c718f1c0790faba4504c16c9aed1ec114c145aed628c0"
    )


def test_protocol_rejects_unknown_keys(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    changed = tmp_path / "mgmr_m0.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(MGMRProtocolError, match="top-level keys"):
        load_protocol(changed, project_root=ROOT)


def test_protocol_rejects_source_hash_drift(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["sources"]["m0_protocol"]["sha256"] = "0" * 64
    changed = tmp_path / "mgmr_m0.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(MGMRProtocolError, match="source SHA-256 mismatch"):
        load_protocol(changed, project_root=ROOT)


def test_protocol_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    changed = tmp_path / "duplicate.yaml"
    changed.write_bytes(CONFIG.read_bytes() + b"\nscope: mgmr_m0_component_gate\n")

    with pytest.raises(MGMRProtocolError, match="duplicate"):
        load_protocol(changed, project_root=ROOT)


def test_protocol_is_immutable() -> None:
    protocol = load_protocol(CONFIG, project_root=ROOT)

    with pytest.raises((AttributeError, TypeError)):
        protocol.coarse_density = 0.5  # type: ignore[misc]
