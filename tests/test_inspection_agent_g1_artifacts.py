from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cmc_bbdm.inspection_agent_g1.artifacts import (
    REQUIRED_G1_OUTPUTS,
    G1ArtifactError,
    publish_g1_manifest,
    validate_g1_package,
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


def test_g1_package_manifest_is_complete_and_source_bound(tmp_path: Path) -> None:
    output = tmp_path / "g1"
    _package(output)
    published = publish_g1_manifest(output, project_root=ROOT, config_path=CONFIG)
    validated = validate_g1_package(output, project_root=ROOT, config_path=CONFIG)
    assert published == validated
    assert validated.status == "G1_POLICY_OBSERVABILITY_NO_GO"
    assert set(validated.file_sha256) == set(REQUIRED_G1_OUTPUTS) | {"config.yaml"}
    assert {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    } == set(REQUIRED_G1_OUTPUTS) | {
        "config.yaml",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }


def test_g1_package_rejects_tampering_and_unlisted_files(tmp_path: Path) -> None:
    output = tmp_path / "g1"
    _package(output)
    publish_g1_manifest(output, project_root=ROOT, config_path=CONFIG)
    (output / "unlisted.txt").write_text("tampered\n", encoding="ascii")
    with pytest.raises(G1ArtifactError, match="checksum|roster|output"):
        validate_g1_package(output, project_root=ROOT, config_path=CONFIG)
