from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from mavis_test_support import synthetic_authority

from cmc_bbdm.mavis.authority import MAVISAuthorityError
from cmc_bbdm.mavis.authority_artifacts import (
    verify_mavis_authority_package,
    write_mavis_authority_package,
)


def test_mavis_manifest_hashes_match(tmp_path: Path) -> None:
    package = tmp_path / "mavis_authority"
    authority = synthetic_authority(true_cai=0.4)

    write_mavis_authority_package(
        package,
        authority,
        config_sha256="1" * 64,
    )
    verify_mavis_authority_package(package)

    assert {path.name for path in package.iterdir()} == {
        "CHECKSUMS.sha256",
        "REPORT.md",
        "artifact_manifest.json",
        "scan_manifest.csv",
    }
    manifest = json.loads(
        (package / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["authority_state_sha256"] == authority.state_sha256
    assert manifest["config_sha256"] == "1" * 64
    assert set(manifest["files"]) == {"REPORT.md", "scan_manifest.csv"}
    with (package / "scan_manifest.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert set(rows[0]) == {
        "specimen_id",
        "dataset_id",
        "height",
        "width",
        "native_count",
        "source_image_sha256",
        "decoded_image_sha256",
        "policy_context_state_sha256",
    }
    package_text = "\n".join(
        path.read_text(encoding="utf-8") for path in package.iterdir()
    ).lower()
    assert "true_cai" not in package_text
    assert "target" not in package_text


def test_mavis_authority_checksums_require_the_exact_file_roster(
    tmp_path: Path,
) -> None:
    package = tmp_path / "mavis_authority"
    write_mavis_authority_package(
        package,
        synthetic_authority(),
        config_sha256="1" * 64,
    )
    rows = (package / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    (package / "CHECKSUMS.sha256").write_text(
        "\n".join((rows[0], rows[0], rows[2])) + "\n",
        encoding="ascii",
    )

    with pytest.raises(MAVISAuthorityError, match="roster"):
        verify_mavis_authority_package(package)
