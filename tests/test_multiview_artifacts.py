from __future__ import annotations

import json
from pathlib import Path

import pytest

from cmc_bbdm.aei_multiview_regression.artifacts import (
    ArtifactError,
    e1_oof_csv,
    replay_stage,
    write_stage_package,
)


def _payload() -> dict[str, bytes]:
    return {
        "config.yaml": b"schema_version: 1\n",
        "aggregate_metrics.csv": b"method,equal_domain_mae\nFULL,0.1\n",
        "domain_metrics.csv": b"method,domain,mae\nFULL,d1,0.1\n",
        "summary.json": b'{"gate_status":"GO"}\n',
        "REPORT.md": b"# Report\n",
    }


def test_stage_package_is_atomic_no_overwrite_and_replayable(tmp_path: Path) -> None:
    destination = tmp_path / "e1_audit"
    write_stage_package(destination, stage="E1", files=_payload())

    assert replay_stage(destination).stage == "E1"
    assert (destination / "artifact_manifest.json").is_file()
    assert (destination / "CHECKSUMS.sha256").is_file()
    with pytest.raises(ArtifactError, match="exists"):
        write_stage_package(destination, stage="E1", files=_payload())


def test_replay_rejects_changed_output(tmp_path: Path) -> None:
    destination = tmp_path / "e1_audit"
    write_stage_package(destination, stage="E1", files=_payload())
    (destination / "summary.json").write_text("{}\n", encoding="ascii")

    with pytest.raises(ArtifactError, match="checksum"):
        replay_stage(destination)


def test_e1_oof_schema_has_one_row_per_specimen() -> None:
    payload = e1_oof_csv(
        specimen_ids=("s1", "s2"),
        domain_ids=("d1", "d2"),
        targets=(0.5, 0.6),
        predictions=((0.4, 0.5, 0.6), (0.7, 0.6, 0.5)),
        cai_strength_mpa=(100.0, 120.0),
        intact_strength_mpa=(200.0, 200.0),
    )
    lines = payload.decode("ascii").splitlines()

    assert lines[0].split(",") == [
        "specimen_id",
        "domain_id",
        "y_true",
        "pred_full",
        "pred_50",
        "pred_25",
        "err_full",
        "err_50",
        "err_25",
        "cai_strength_mpa",
        "intact_strength_mpa",
        "pred_full_mpa",
        "pred_50_mpa",
        "pred_25_mpa",
        "abs_err_full_mpa",
        "abs_err_50_mpa",
        "abs_err_25_mpa",
    ]
    assert len(lines) == 3
    assert json.loads('{"rows": 2}')["rows"] == 2
