from __future__ import annotations

from pathlib import Path

from test_mva_a4_artifacts import CONFIG, ROOT, _package

from cmc_bbdm.mva.a4_artifacts import publish_a4_manifest
from cmc_bbdm.mva.a4_replay import replay_a4_package


def test_a4_replay_is_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "formal"
    destination = tmp_path / "replay"
    _package(source)
    original = publish_a4_manifest(source, project_root=ROOT, config_path=CONFIG)

    replayed = replay_a4_package(
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
    assert all(
        (destination / relative).read_bytes() == (source / relative).read_bytes()
        for relative in source_files
    )
