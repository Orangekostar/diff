from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cmc_bbdm.mavis.mris_package import (
    MAVISMRISPackageError,
    verify_mris_package,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(root: Path) -> None:
    figures = root / "figures"
    figures.mkdir()
    figure = figures / "curve.svg"
    figure.write_text("<svg/>\n", encoding="utf-8")
    summary = root / "summary.json"
    summary.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    files = {
        "figures/curve.svg": figure,
        "summary.json": summary,
    }
    manifest = root / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
                    for name, path in files.items()
                }
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


def test_mavis_mris_package_verifies_recursive_roster_and_hashes(tmp_path: Path) -> None:
    _package(tmp_path)

    verify_mris_package(tmp_path)

    (tmp_path / "figures/extra.svg").write_text("<svg/>", encoding="utf-8")
    with pytest.raises(MAVISMRISPackageError, match="roster"):
        verify_mris_package(tmp_path)


def test_mavis_mris_package_rejects_tampered_nested_file(tmp_path: Path) -> None:
    _package(tmp_path)
    (tmp_path / "figures/curve.svg").write_text("changed", encoding="ascii")

    with pytest.raises(MAVISMRISPackageError, match="checksum"):
        verify_mris_package(tmp_path)
