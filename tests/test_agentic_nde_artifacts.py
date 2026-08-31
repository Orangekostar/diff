from __future__ import annotations

from pathlib import Path

import pytest

from cmc_bbdm.agentic_nde.artifacts import (
    REQUIRED_P0_FILES,
    ArtifactError,
    write_p0_package,
)
from cmc_bbdm.agentic_nde.contracts import P0GateFacts, decide_p0


def _no_go_summary() -> dict[str, object]:
    facts = P0GateFacts(
        authorized_by_domain={},
        exact_identity_hashes=True,
        orientation_resolved=False,
        deterministic_transform=False,
        deployable_evidence_only=True,
        replay_verified=False,
    )
    decision = decide_p0(facts)
    return {**decision.as_dict(), "gate_facts": facts.as_dict()}


def _write(output: Path) -> Path:
    return write_p0_package(
        output,
        config_text="schema_version: 1\n",
        surface_manifest=[{"specimen_id": "s1", "dataset_id": "d1"}],
        surface_qc=[{"specimen_id": "s1", "status": "PASS"}],
        registration=[{"specimen_id": "s1", "status": "UNRESOLVED"}],
        registration_qc=[{"specimen_id": "s1", "status": "FAIL"}],
        source_hashes=[{"logical_path": "source.csv", "sha256": "a" * 64}],
        summary=_no_go_summary(),
        report="# P0 Report\n\nDecision: NO-GO.\n",
    )


def test_writer_emits_exact_membership(tmp_path: Path) -> None:
    output = _write(tmp_path / "p0")
    assert {path.name for path in output.iterdir()} == REQUIRED_P0_FILES


def test_writer_refuses_existing_destination(tmp_path: Path) -> None:
    output = tmp_path / "p0"
    output.mkdir()
    with pytest.raises(ArtifactError, match="exists"):
        _write(output)


def test_identical_packages_are_byte_identical(tmp_path: Path) -> None:
    first = _write(tmp_path / "a")
    second = _write(tmp_path / "b")
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_writer_rejects_duplicate_surface_keys(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="duplicate keys"):
        write_p0_package(
            tmp_path / "p0",
            config_text="schema_version: 1\n",
            surface_manifest=[
                {"specimen_id": "s1", "dataset_id": "d1"},
                {"specimen_id": "s1", "dataset_id": "d1"},
            ],
            surface_qc=[{"specimen_id": "s1", "dataset_id": "d1"}],
            registration=[{"specimen_id": "s1", "dataset_id": "d1"}],
            registration_qc=[{"specimen_id": "s1", "dataset_id": "d1"}],
            source_hashes=[{"logical_path": "x", "sha256": "a" * 64}],
            summary=_no_go_summary(),
            report="# Report\n",
        )


def test_writer_rejects_gate_status_inconsistency_without_leaving_output(
    tmp_path: Path,
) -> None:
    facts = P0GateFacts(
        authorized_by_domain={
            "74t7kcdgkr": 45,
            "cgtnjyggtm": 49,
            "w68dtmpfyf": 43,
            "xcmzfsbd9t": 59,
            "yfxyg8jm46": 42,
            "ykhs7s2dck": 38,
        },
        exact_identity_hashes=True,
        orientation_resolved=True,
        deterministic_transform=True,
        deployable_evidence_only=True,
        replay_verified=True,
    )
    output = tmp_path / "p0"
    with pytest.raises(ArtifactError, match="status is inconsistent"):
        write_p0_package(
            output,
            config_text="schema_version: 1\n",
            surface_manifest=[{"specimen_id": "s1"}],
            surface_qc=[{"specimen_id": "s1"}],
            registration=[{"specimen_id": "s1"}],
            registration_qc=[{"specimen_id": "s1"}],
            source_hashes=[{"logical_path": "x", "sha256": "a" * 64}],
            summary={
                "status": "P0_SPATIAL_REGISTRATION_NO_GO",
                "gate_facts": facts.as_dict(),
                "reasons": [],
                "downstream": {},
            },
            report="# Report\n",
        )
    assert not output.exists()


def test_writer_rejects_missing_gate_facts_without_leaving_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "p0"
    with pytest.raises(ArtifactError, match="gate facts"):
        write_p0_package(
            output,
            config_text="schema_version: 1\n",
            surface_manifest=[{"specimen_id": "s1"}],
            surface_qc=[{"specimen_id": "s1"}],
            registration=[{"specimen_id": "s1"}],
            registration_qc=[{"specimen_id": "s1"}],
            source_hashes=[{"logical_path": "x", "sha256": "a" * 64}],
            summary={"status": "P0_SPATIAL_REGISTRATION_NO_GO"},
            report="# Report\n",
        )
    assert not output.exists()
