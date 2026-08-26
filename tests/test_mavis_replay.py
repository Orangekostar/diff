from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cmc_bbdm.mavis.replay import (
    MAVISReplayError,
    compare_package_directories,
    verify_replay_package,
)


def _tree(root: Path) -> None:
    (root / "tables").mkdir(parents=True)
    (root / "summary.json").write_text('{"status":"COMPLETE"}\n', encoding="ascii")
    (root / "tables/result.csv").write_text("x\n1\n", encoding="ascii")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mavis_replay_is_deterministic(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    replay = tmp_path / "replay"
    _tree(formal)
    _tree(replay)

    comparison = compare_package_directories(formal, replay)

    assert comparison["byte_identical"] is True
    assert comparison["file_count"] == 2
    assert len(comparison["tree_state_sha256"]) == 64


def test_mavis_replay_rejects_changed_or_unlisted_file(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    replay = tmp_path / "replay"
    _tree(formal)
    _tree(replay)
    (replay / "tables/result.csv").write_text("x\n2\n", encoding="ascii")
    with pytest.raises(MAVISReplayError, match="content"):
        compare_package_directories(formal, replay)

    (replay / "tables/result.csv").write_text("x\n1\n", encoding="ascii")
    (replay / "extra.csv").write_text("x\n", encoding="ascii")
    with pytest.raises(MAVISReplayError, match="roster"):
        compare_package_directories(formal, replay)


def test_replay_root_manifest_covers_nested_final_manifest(tmp_path: Path) -> None:
    project = tmp_path / "project"
    formal = project / "results/mavis/p7_final_frozen_eval"
    replay_root = project / "results/mavis/replay"
    replay = replay_root / "p7_final_frozen_eval"
    _tree(formal)
    _tree(replay)
    for package in (formal, replay):
        (package / "artifact_manifest.json").write_text("{}\n", encoding="ascii")
    comparison = compare_package_directories(formal, replay)
    summary = replay_root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "formal_package": "results/mavis/p7_final_frozen_eval",
                "replay_package": "results/mavis/replay/p7_final_frozen_eval",
                **comparison,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    scientific = {
        path.relative_to(replay_root).as_posix(): path
        for path in replay_root.rglob("*")
        if path.is_file()
    }
    manifest = replay_root / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "mavis_final_replay",
                "tree_state_sha256": comparison["tree_state_sha256"],
                "byte_identical": True,
                "files": {
                    name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
                    for name, path in scientific.items()
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = {"artifact_manifest.json": manifest, **scientific}
    (replay_root / "CHECKSUMS.sha256").write_text(
        "".join(f"{_sha(path)}  {name}\n" for name, path in sorted(ledger.items())),
        encoding="ascii",
    )

    verified = verify_replay_package(replay_root, project_root=project)

    assert verified["byte_identical"] is True
    assert "p7_final_frozen_eval/artifact_manifest.json" in verified["files"]
