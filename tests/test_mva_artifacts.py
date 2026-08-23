from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cmc_bbdm.mva.artifacts import (
    REQUIRED_OUTPUTS,
    MVAArtifactError,
    publish_mva_manifest,
    validate_mva_package,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mva_a0_a3.yaml"


def _package(path: Path) -> None:
    for relative in REQUIRED_OUTPUTS:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == "summary.json":
            target.write_text('{"status":"MVA_ORACLE_NO_GO"}\n', encoding="ascii")
        else:
            target.write_bytes(f"fixture:{relative}\n".encode("ascii"))
    shutil.copyfile(CONFIG, path / "config.yaml")


def test_mva_package_manifest_is_complete_and_validated(tmp_path: Path) -> None:
    output = tmp_path / "a2_oracle_value"
    _package(output)

    published = publish_mva_manifest(output, project_root=ROOT, config_path=CONFIG)
    validated = validate_mva_package(output, project_root=ROOT, config_path=CONFIG)

    assert published == validated
    assert validated.status == "MVA_ORACLE_NO_GO"
    assert (output / "artifact_manifest.json").is_file()
    assert (output / "CHECKSUMS.sha256").is_file()


def test_mva_package_rejects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "a2_oracle_value"
    _package(output)
    publish_mva_manifest(output, project_root=ROOT, config_path=CONFIG)
    with (output / "budget_metrics.csv").open("ab") as handle:
        handle.write(b"tampered\n")

    with pytest.raises(MVAArtifactError, match="checksum|SHA-256"):
        validate_mva_package(output, project_root=ROOT, config_path=CONFIG)


def test_mva_package_rejects_absolute_project_path(tmp_path: Path) -> None:
    output = tmp_path / "a2_oracle_value"
    _package(output)
    (output / "REPORT.md").write_text(str(ROOT), encoding="utf-8")

    with pytest.raises(MVAArtifactError, match="absolute path"):
        publish_mva_manifest(output, project_root=ROOT, config_path=CONFIG)
