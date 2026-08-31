from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cmc_bbdm.inspection_agent.artifacts import (
    REQUIRED_OUTPUTS,
    InspectionArtifactError,
    compare_g0_packages,
    publish_g0_manifest,
    validate_g0_package,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/inspection_agent_g0.yaml"


def _package(path: Path) -> None:
    path.mkdir(parents=True)
    for relative in REQUIRED_OUTPUTS:
        target = path / relative
        if relative == "decision_summary.json":
            target.write_text(
                json.dumps(
                    {"status": "G0_NO_AGENTIC_HEADROOM_NO_GO"},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="ascii",
            )
        else:
            target.write_bytes(f"fixture:{relative}\n".encode("ascii"))
    shutil.copyfile(CONFIG, path / "config.yaml")


def test_g0_package_manifest_is_complete_and_source_bound(tmp_path: Path) -> None:
    output = tmp_path / "g0"
    _package(output)
    published = publish_g0_manifest(output, project_root=ROOT, config_path=CONFIG)
    validated = validate_g0_package(output, project_root=ROOT, config_path=CONFIG)
    assert published == validated
    assert validated.status == "G0_NO_AGENTIC_HEADROOM_NO_GO"
    assert set(validated.file_sha256) == set(REQUIRED_OUTPUTS) | {"config.yaml"}


def test_g0_package_rejects_tampering_and_unlisted_files(tmp_path: Path) -> None:
    output = tmp_path / "g0"
    _package(output)
    publish_g0_manifest(output, project_root=ROOT, config_path=CONFIG)
    (output / "unlisted.txt").write_text("tampered\n", encoding="ascii")
    with pytest.raises(InspectionArtifactError, match="checksum|roster|output"):
        validate_g0_package(output, project_root=ROOT, config_path=CONFIG)


def test_g0_formal_and_replay_packages_compare_byte_for_byte(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    replay = tmp_path / "replay"
    _package(formal)
    publish_g0_manifest(formal, project_root=ROOT, config_path=CONFIG)
    shutil.copytree(formal, replay)
    comparison = compare_g0_packages(
        formal,
        replay,
        project_root=ROOT,
        config_path=CONFIG,
    )
    assert comparison.byte_identical is True
    assert comparison.package_sha256 == comparison.replay_sha256

