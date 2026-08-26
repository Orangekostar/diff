from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cmc_bbdm.mavis.safety_package import (
    MAVISSafetyPackageError,
    verify_safety_package,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(root: Path) -> None:
    selection = root / "selections.csv"
    selection.write_text("outer_domain,baseline,threshold\nd0,uniform,0.5\n", encoding="ascii")
    report = root / "REPORT.md"
    report.write_text("# Safety\n", encoding="ascii")
    files = {"REPORT.md": report, "selections.csv": selection}
    manifest = root / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "p6_state_sha256": "a" * 64,
                "files": {
                    name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
                    for name, path in files.items()
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = {"artifact_manifest.json": manifest, **files}
    (root / "CHECKSUMS.sha256").write_text(
        "".join(f"{_sha(path)}  {name}\n" for name, path in sorted(ledger.items())),
        encoding="ascii",
    )


def test_safety_package_verifies_recursive_roster_and_hashes(tmp_path: Path) -> None:
    _package(tmp_path)
    manifest = verify_safety_package(tmp_path)
    assert manifest["p6_state_sha256"] == "a" * 64

    (tmp_path / "extra.csv").write_text("x\n", encoding="ascii")
    with pytest.raises(MAVISSafetyPackageError, match="roster"):
        verify_safety_package(tmp_path)


def test_safety_package_rejects_tampered_selection(tmp_path: Path) -> None:
    _package(tmp_path)
    (tmp_path / "selections.csv").write_text("changed\n", encoding="ascii")

    with pytest.raises(MAVISSafetyPackageError, match="checksum"):
        verify_safety_package(tmp_path)
