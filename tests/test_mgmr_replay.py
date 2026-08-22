from __future__ import annotations

from pathlib import Path

from mgmr_test_support import synthetic_formal

from cmc_bbdm.mgmr.artifacts import publish_m0_package
from cmc_bbdm.mgmr.protocol import load_protocol
from cmc_bbdm.mgmr.replay import replay_m0_package

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mgmr_m0.yaml"


def test_replay_is_byte_identical(tmp_path: Path) -> None:
    protocol = load_protocol(CONFIG, project_root=ROOT)
    source = tmp_path / "formal"
    destination = tmp_path / "replay"
    original = publish_m0_package(
        source,
        protocol=protocol,
        formal=synthetic_formal(protocol),
        feature_manifest_sha256="3" * 64,
    )

    replayed = replay_m0_package(
        source,
        destination,
        project_root=ROOT,
        config_path=CONFIG,
    )

    assert replayed == original
    assert sorted(path.name for path in source.iterdir()) == sorted(
        path.name for path in destination.iterdir()
    )
    for source_path in source.iterdir():
        assert source_path.read_bytes() == (destination / source_path.name).read_bytes()
