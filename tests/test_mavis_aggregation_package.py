from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cmc_bbdm.mavis.aggregation_package import (
    MAVISAggregationPackageError,
    verify_aggregation_package,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(root: Path) -> None:
    checkpoint = root / "checkpoints/d0__real.npz"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    report = root / "REPORT.md"
    report.write_text("# P5\n", encoding="ascii")
    files = {
        "REPORT.md": report,
        "checkpoints/d0__real.npz": checkpoint,
    }
    manifest = root / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "p5_state_sha256": "a" * 64,
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


def test_aggregation_package_verifies_recursive_roster_and_hashes(
    tmp_path: Path,
) -> None:
    _package(tmp_path)

    manifest = verify_aggregation_package(tmp_path)

    assert manifest["p5_state_sha256"] == "a" * 64
    (tmp_path / "extra.csv").write_text("x\n", encoding="ascii")
    with pytest.raises(MAVISAggregationPackageError, match="roster"):
        verify_aggregation_package(tmp_path)


def test_aggregation_package_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    _package(tmp_path)
    (tmp_path / "checkpoints/d0__real.npz").write_bytes(b"changed")

    with pytest.raises(MAVISAggregationPackageError, match="checksum"):
        verify_aggregation_package(tmp_path)
