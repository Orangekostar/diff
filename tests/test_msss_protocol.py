from __future__ import annotations

from pathlib import Path

import pytest

from cmc_bbdm.msss.protocol import MSSSProtocolError, load_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/msss.yaml"


def test_protocol_freezes_primary_axes_and_gates() -> None:
    protocol = load_protocol(CONFIG, project_root=ROOT)

    assert protocol.sampling_densities == (
        1.0,
        0.75,
        0.625,
        0.5,
        0.375,
        0.25,
        0.1875,
        0.125,
        0.0625,
    )
    assert protocol.gaussian_sigmas == (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
    )
    assert protocol.wavelet_families == ("db2", "haar", "db4")
    assert protocol.wavelet_primary == "db2"
    assert protocol.wavelet_levels == (0, 1, 2, 3)
    assert protocol.noninferiority_margins == (0.025, 0.05, 0.075)
    assert protocol.primary_margin == 0.05
    assert protocol.pca_dimensions == (8, 16, 32)
    assert protocol.ridge_alpha == 10.0
    assert protocol.device == "cuda"
    assert protocol.fourier_enabled is False


def test_protocol_binds_every_source_hash() -> None:
    protocol = load_protocol(CONFIG, project_root=ROOT)

    assert len(protocol.sources) == 20
    assert all(source.path.is_file() for source in protocol.sources)
    assert all(len(source.sha256) == 64 for source in protocol.sources)
    assert protocol.specimen_count == 276
    assert protocol.domain_order == (
        "74t7kcdgkr",
        "cgtnjyggtm",
        "w68dtmpfyf",
        "xcmzfsbd9t",
        "yfxyg8jm46",
        "ykhs7s2dck",
    )


def test_protocol_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")

    with pytest.raises(MSSSProtocolError, match="duplicate YAML key"):
        load_protocol(duplicate, project_root=ROOT)


def test_protocol_rejects_any_nonregistered_config_path(tmp_path: Path) -> None:
    copied = tmp_path / "msss.yaml"
    copied.write_bytes(CONFIG.read_bytes())

    with pytest.raises(MSSSProtocolError, match="exact registered config"):
        load_protocol(copied, project_root=ROOT)
