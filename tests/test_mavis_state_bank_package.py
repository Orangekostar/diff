from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from cmc_bbdm.mavis.state_bank_package import (
    MAVISStateBankPackageError,
    load_state_action_pairs_package,
    load_state_manifest_package,
    verify_state_bank_package,
    write_compact_state_manifest,
    write_partitioned_action_pairs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(root: Path) -> None:
    summary = root / "summary.json"
    summary.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    manifest = root / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    "summary.json": {
                        "bytes": summary.stat().st_size,
                        "sha256": _sha(summary),
                    }
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "CHECKSUMS.sha256").write_text(
        f"{_sha(manifest)}  artifact_manifest.json\n"
        f"{_sha(summary)}  summary.json\n",
        encoding="ascii",
    )


def test_mavis_state_bank_manifest_hashes_match(tmp_path: Path) -> None:
    _package(tmp_path)

    verify_state_bank_package(tmp_path)

    (tmp_path / "unlisted.txt").write_text("extra", encoding="ascii")
    with pytest.raises(MAVISStateBankPackageError, match="roster"):
        verify_state_bank_package(tmp_path)


def test_mavis_state_bank_manifest_rejects_changed_file(tmp_path: Path) -> None:
    _package(tmp_path)
    (tmp_path / "summary.json").write_text("changed\n", encoding="ascii")

    with pytest.raises(MAVISStateBankPackageError, match="checksum"):
        verify_state_bank_package(tmp_path)


def test_mavis_state_manifest_payloads_round_trip_without_large_monolith(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "state_id": f"state-{index}",
            "domain_id": "domain-a",
            "specimen_id": f"sample-{index // 2}",
            "method": "uniform",
            "nominal_checkpoint": 0.03125,
            "revealed_rows": [0, index + 1],
            "revealed_columns": [2, 3],
            "revealed_red": [4, 5],
            "revealed_green": [6, 7],
            "revealed_blue": [8, 9],
        }
        for index in range(6)
    ]
    states = pl.DataFrame(rows, infer_schema_length=None)

    write_compact_state_manifest(states, tmp_path, specimens_per_part=2)

    compact = pl.read_parquet(tmp_path / "state_manifest.parquet")
    assert "revealed_rows" not in compact.columns
    assert compact.get_column("measurement_payload_file").n_unique() == 2
    assert all(
        path.stat().st_size < 100 * 1024 * 1024
        for path in (tmp_path / "revealed_measurements").glob("*.parquet")
    )
    restored = load_state_manifest_package(tmp_path)
    assert restored.sort("state_id").to_dicts() == states.sort("state_id").to_dicts()


def test_mavis_action_pairs_partition_by_teacher_and_query_domain(tmp_path: Path) -> None:
    actions = pl.DataFrame(
        [
            {
                "outer_domain": outer,
                "domain_id": query,
                "specimen_id": f"{query}-{index}",
                "state_id": f"{outer}-{query}-{index}",
                "candidate_index": index,
            }
            for outer in ("d0", "d1")
            for query in ("d0", "d1")
            if query != outer
            for index in range(3)
        ]
    )

    write_partitioned_action_pairs(actions, tmp_path)

    parts = sorted((tmp_path / "state_action_pairs").glob("*.parquet"))
    assert len(parts) == 2
    assert all(path.stat().st_size < 100 * 1024 * 1024 for path in parts)
    restored = load_state_action_pairs_package(tmp_path)
    assert restored.sort("state_id").to_dicts() == actions.sort("state_id").to_dicts()
