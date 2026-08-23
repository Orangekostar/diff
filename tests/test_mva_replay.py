from __future__ import annotations

from pathlib import Path

from test_mva_artifacts import CONFIG, ROOT, _package

from cmc_bbdm.mva.artifacts import publish_mva_manifest
from cmc_bbdm.mva.replay import replay_mva_package


def test_mva_replay_is_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "formal"
    destination = tmp_path / "replay"
    _package(source)
    original = publish_mva_manifest(source, project_root=ROOT, config_path=CONFIG)

    replayed = replay_mva_package(
        source,
        destination,
        project_root=ROOT,
        config_path=CONFIG,
    )

    assert replayed == original
    source_files = sorted(
        path.relative_to(source) for path in source.rglob("*") if path.is_file()
    )
    replay_files = sorted(
        path.relative_to(destination)
        for path in destination.rglob("*")
        if path.is_file()
    )
    assert replay_files == source_files
    for relative in source_files:
        assert (destination / relative).read_bytes() == (source / relative).read_bytes()
