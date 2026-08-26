from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cmc_bbdm.mavis.final_package import (
    MAVISFinalPackageError,
    development_package_sha256,
    verify_final_package,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(root: Path) -> None:
    summary = root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "claim_tier": "B",
                "config_sha256": "c" * 64,
                "configuration_frozen": True,
                "development_package_sha256": "d" * 64,
                "p7_state_sha256": "a" * 64,
                "status": "COMPLETE",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    table = root / "tables/final.csv"
    table.parent.mkdir()
    table.write_text("method,cai_auebc\nmavis_full,0.1\n", encoding="ascii")
    files = {"summary.json": summary, "tables/final.csv": table}
    manifest = root / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "p7_state_sha256": "a" * 64,
                "config_sha256": "c" * 64,
                "development_package_sha256": "d" * 64,
                "claim_tier": "B",
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


def test_final_package_verifies_recursive_roster_and_hashes(tmp_path: Path) -> None:
    _package(tmp_path)

    manifest = verify_final_package(tmp_path)

    assert manifest["p7_state_sha256"] == "a" * 64
    (tmp_path / "extra.csv").write_text("x\n", encoding="ascii")
    with pytest.raises(MAVISFinalPackageError, match="roster"):
        verify_final_package(tmp_path)


def test_final_package_rejects_tampered_scientific_table(tmp_path: Path) -> None:
    _package(tmp_path)
    (tmp_path / "tables/final.csv").write_text("changed\n", encoding="ascii")

    with pytest.raises(MAVISFinalPackageError, match="checksum"):
        verify_final_package(tmp_path)


def test_development_package_hash_binds_all_frozen_stage_states() -> None:
    p4 = {"config_sha256": "c" * 64, "p4_state_sha256": "4" * 64}
    p5 = {"config_sha256": "c" * 64, "p5_state_sha256": "5" * 64}
    p6 = {"config_sha256": "c" * 64, "p6_state_sha256": "6" * 64}

    first = development_package_sha256(p4, p5, p6)
    second = development_package_sha256(dict(reversed(tuple(p4.items()))), p5, p6)

    assert first == second
    assert len(first) == 64
    with pytest.raises(MAVISFinalPackageError, match="config"):
        development_package_sha256(
            p4,
            {**p5, "config_sha256": "e" * 64},
            p6,
        )


def test_final_package_rejects_summary_state_mismatch(tmp_path: Path) -> None:
    _package(tmp_path)
    summary = tmp_path / "summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["p7_state_sha256"] = "b" * 64
    summary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    manifest["files"]["summary.json"] = {
        "bytes": summary.stat().st_size,
        "sha256": _sha(summary),
    }
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ledger = {
        "artifact_manifest.json": tmp_path / "artifact_manifest.json",
        "summary.json": summary,
        "tables/final.csv": tmp_path / "tables/final.csv",
    }
    (tmp_path / "CHECKSUMS.sha256").write_text(
        "".join(f"{_sha(path)}  {name}\n" for name, path in sorted(ledger.items())),
        encoding="ascii",
    )

    with pytest.raises(MAVISFinalPackageError, match="state"):
        verify_final_package(tmp_path)
