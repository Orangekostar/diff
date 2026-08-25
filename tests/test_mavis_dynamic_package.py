from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from cmc_bbdm.mavis.dynamic_package import (
    MAVISDynamicPackageError,
    _decision_state_roster,
    verify_dynamic_package,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(root: Path) -> None:
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "d0__real.npz"
    checkpoint.write_bytes(b"checkpoint")
    summary = root / "summary.json"
    summary.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    files = {
        "checkpoints/d0__real.npz": checkpoint,
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


def test_dynamic_package_verifies_recursive_roster_and_hashes(tmp_path: Path) -> None:
    _package(tmp_path)
    verify_dynamic_package(tmp_path)

    (tmp_path / "extra.csv").write_text("x\n", encoding="ascii")
    with pytest.raises(MAVISDynamicPackageError, match="roster"):
        verify_dynamic_package(tmp_path)


def test_dynamic_package_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    _package(tmp_path)
    (tmp_path / "checkpoints/d0__real.npz").write_bytes(b"changed")

    with pytest.raises(MAVISDynamicPackageError, match="checksum"):
        verify_dynamic_package(tmp_path)


def test_dynamic_package_excludes_terminal_states_from_action_roster() -> None:
    states = pl.DataFrame(
        {
            "state_id": ["s0", "s1", "s2", "s3"],
            "domain_id": ["d0", "d0", "d1", "d1"],
            "candidate_cell_indices": [[0], [], [1, 2], []],
        }
    )

    state_ids, counts = _decision_state_roster(
        states,
        domain_order=("d0", "d1"),
        feature_state_ids=("s0", "s1", "s2", "s3"),
    )

    assert state_ids == frozenset({"s0", "s2"})
    assert counts == {"d0": 1, "d1": 1}
