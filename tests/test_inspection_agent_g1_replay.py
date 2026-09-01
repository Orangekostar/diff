from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cmc_bbdm.inspection_agent_g1.artifacts import (
    REQUIRED_G1_OUTPUTS,
    G1ArtifactError,
    compare_g1_packages,
    publish_g1_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/inspection_agent_g1.yaml"


def _package(path: Path) -> None:
    path.mkdir(parents=True)
    for relative in REQUIRED_G1_OUTPUTS:
        target = path / relative
        if relative == "decision_summary.json":
            target.write_text(
                json.dumps(
                    {"status": "G1_POLICY_OBSERVABILITY_NO_GO"},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="ascii",
            )
        else:
            target.write_bytes(f"fixture:{relative}\n".encode("ascii"))
    shutil.copyfile(CONFIG, path / "config.yaml")


def test_g1_formal_and_replay_packages_compare_byte_for_byte(
    tmp_path: Path,
) -> None:
    formal = tmp_path / "formal"
    replay = tmp_path / "replay"
    _package(formal)
    publish_g1_manifest(formal, project_root=ROOT, config_path=CONFIG)
    shutil.copytree(formal, replay)
    comparison = compare_g1_packages(
        formal,
        replay,
        project_root=ROOT,
        config_path=CONFIG,
    )
    assert comparison.byte_identical is True
    assert comparison.package_sha256 == comparison.replay_sha256

    tampered = replay / "REPORT.md"
    payload = tampered.read_bytes()
    tampered.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(G1ArtifactError, match="checksum|roster|output"):
        compare_g1_packages(
            formal,
            replay,
            project_root=ROOT,
            config_path=CONFIG,
        )
