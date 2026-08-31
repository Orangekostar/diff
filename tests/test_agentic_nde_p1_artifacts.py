from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cmc_bbdm.agentic_nde.p1_artifacts import (
    REQUIRED_P1_FILES,
    P1ArtifactError,
    replay_p1_package,
    write_p1_package,
)


def _tables() -> dict[str, pl.DataFrame]:
    return {
        "authorized_roster": pl.DataFrame(
            {"dataset_id": ["a"], "specimen_id": ["s-1"]}
        ),
        "visual_feature_manifest": pl.DataFrame(
            {
                "array_name": ["global"],
                "sha256": ["a" * 64],
                "shape": ["1x512"],
            }
        ),
        "outer_model_selection": pl.DataFrame(
            {
                "outer_domain": ["a"],
                "stage": ["FINAL_FIT"],
                "candidate_id": ["ridge_alpha_0.1"],
            }
        ),
        "per_state_scores": pl.DataFrame(
            {
                "outer_domain": ["a"],
                "specimen_id": ["s-1"],
                "method": ["proposed"],
                "cell_index": [0],
                "predicted_score": [1.0],
            }
        ),
        "per_specimen_metrics": pl.DataFrame(
            {
                "outer_domain": ["a"],
                "specimen_id": ["s-1"],
                "method": ["proposed"],
                "cai_auebc": [0.1],
            }
        ),
        "domain_metrics": pl.DataFrame(
            {"outer_domain": ["a"], "method": ["proposed"], "cai_auebc": [0.1]}
        ),
        "bootstrap": pl.DataFrame(
            {"effect_key": ["c0_minus_proposed"], "point_estimate": [0.1]}
        ),
        "acquisition_curves": pl.DataFrame(
            {
                "outer_domain": ["a"],
                "specimen_id": ["s-1"],
                "method": ["proposed"],
                "nominal_checkpoint": [0.0625],
            }
        ),
        "control_results": pl.DataFrame(
            {"method": ["proposed"], "cai_auebc": [0.1]}
        ),
    }


def _write(path: Path) -> Path:
    return write_p1_package(
        path,
        config_bytes=b"schema_version: 1\n",
        tables=_tables(),
        summary={
            "schema_version": 1,
            "stage": "P1_VISUAL_OBSERVABILITY",
            "status": "P1_DESCRIPTIVE_SPATIAL_SIGNAL_ONLY",
        },
        report="# P1 Visual Observability\n\nSynthetic package test.\n",
    )


def test_p1_package_is_exact_atomic_and_byte_deterministic(tmp_path: Path) -> None:
    first = _write(tmp_path / "first")
    second = _write(tmp_path / "second")
    assert {path.name for path in first.iterdir()} == REQUIRED_P1_FILES
    assert replay_p1_package(first)["status"] == (
        "P1_DESCRIPTIVE_SPATIAL_SIGNAL_ONLY"
    )
    for name in REQUIRED_P1_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with pytest.raises(P1ArtifactError, match="already exists"):
        _write(first)


def test_p1_package_replay_rejects_payload_tampering(tmp_path: Path) -> None:
    package = _write(tmp_path / "package")
    with (package / "REPORT.md").open("ab") as handle:
        handle.write(b"tampered\n")
    with pytest.raises(P1ArtifactError, match="hash or size"):
        replay_p1_package(package)
