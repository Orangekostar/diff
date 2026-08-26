from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cmc_bbdm.mavis.closed_loop_package import (
    MAVISClosedLoopPackageError,
    verify_closed_loop_package,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(root: Path) -> None:
    nested = root / "tables"
    nested.mkdir()
    table = nested / "aggregate.csv"
    table.write_text("method,cai_auebc\nmavis_full,0.1\n", encoding="ascii")
    summary = root / "summary.json"
    summary.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    files = {"summary.json": summary, "tables/aggregate.csv": table}
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


def test_closed_loop_package_verifies_recursive_roster_and_hashes(
    tmp_path: Path,
) -> None:
    _package(tmp_path)
    verify_closed_loop_package(tmp_path)

    (tmp_path / "extra.csv").write_text("x\n", encoding="ascii")
    with pytest.raises(MAVISClosedLoopPackageError, match="roster"):
        verify_closed_loop_package(tmp_path)


def test_closed_loop_package_rejects_tampered_table(tmp_path: Path) -> None:
    _package(tmp_path)
    (tmp_path / "tables/aggregate.csv").write_text("changed\n", encoding="ascii")

    with pytest.raises(MAVISClosedLoopPackageError, match="checksum"):
        verify_closed_loop_package(tmp_path)
