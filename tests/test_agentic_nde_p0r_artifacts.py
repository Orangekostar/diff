from __future__ import annotations

from pathlib import Path

import pytest

from cmc_bbdm.agentic_nde.author_authority import (
    build_author_registration_authority,
)
from cmc_bbdm.agentic_nde.contracts import PRIMARY_COUNTS, P0RGateFacts, decide_p0r
from cmc_bbdm.agentic_nde.p0r_artifacts import (
    REQUIRED_P0R_FILES,
    P0RArtifactError,
    write_p0r_package,
)


def _summary() -> dict[str, object]:
    facts = P0RGateFacts(
        authorized_by_domain=dict(PRIMARY_COUNTS),
        exact_identity_hashes=True,
        author_statement_bound=True,
        global_orientation_rot90=True,
        all_panels_resolved=True,
        processing_provenance_deterministic=True,
        no_unsupported_rotation_reflection=True,
        composed_transform_replayable=True,
        no_result_driven_orientation=True,
        author_evidence_conflict=False,
        processing_provenance_unresolved=False,
    )
    return {**decide_p0r(facts).as_dict(), "gate_facts": facts.as_dict()}


def _rows() -> dict[str, list[dict[str, object]]]:
    identity = {"dataset_id": "d", "specimen_id": "s"}
    return {
        "surface_manifest": [{**identity, "surface_sha256": "a" * 64}],
        "scan_processing_provenance": [
            {**identity, "panel_index": 0, "decoded_pixel_equal": True}
        ],
        "registration": [
            {**identity, "orientation": "ROT90", "transform_sha256": "b" * 64}
        ],
        "registration_qc": [{**identity, "status": "PASS"}],
        "grid_mapping_qc": [
            {**identity, "cell_id": 0, "round_trip_status": "PASS"}
        ],
    }


def _write(output: Path, *, summary: dict[str, object] | None = None) -> Path:
    rows = _rows()
    return write_p0r_package(
        output,
        config_text="schema_version: 1\nstage: P0R_AUTHOR_SURFACE_CSCAN_REGISTRATION\n",
        author_authority=build_author_registration_authority().as_dict(),
        surface_manifest=rows["surface_manifest"],
        scan_processing_provenance=rows["scan_processing_provenance"],
        registration=rows["registration"],
        registration_qc=rows["registration_qc"],
        grid_mapping_qc=rows["grid_mapping_qc"],
        summary=_summary() if summary is None else summary,
        report="# P0R Report\n\nSynthetic package.\n",
    )


def test_writer_emits_exact_p0r_membership(tmp_path: Path) -> None:
    output = _write(tmp_path / "p0r")

    assert {path.name for path in output.iterdir()} == REQUIRED_P0R_FILES


def test_writer_refuses_existing_destination(tmp_path: Path) -> None:
    output = tmp_path / "p0r"
    output.mkdir()

    with pytest.raises(P0RArtifactError, match="already exists"):
        _write(output)


def test_identical_p0r_packages_are_byte_identical(tmp_path: Path) -> None:
    left = _write(tmp_path / "left")
    right = _write(tmp_path / "right")

    assert {
        path.name: path.read_bytes() for path in left.iterdir()
    } == {path.name: path.read_bytes() for path in right.iterdir()}


def test_writer_rejects_duplicate_specimen_keys(tmp_path: Path) -> None:
    rows = _rows()
    rows["registration"].append(dict(rows["registration"][0]))

    with pytest.raises(P0RArtifactError, match="duplicate"):
        write_p0r_package(
            tmp_path / "p0r",
            config_text="schema_version: 1\n",
            author_authority=build_author_registration_authority().as_dict(),
            surface_manifest=rows["surface_manifest"],
            scan_processing_provenance=rows["scan_processing_provenance"],
            registration=rows["registration"],
            registration_qc=rows["registration_qc"],
            grid_mapping_qc=rows["grid_mapping_qc"],
            summary=_summary(),
            report="# P0R\n",
        )


def test_writer_rejects_absolute_private_paths(tmp_path: Path) -> None:
    rows = _rows()
    rows["surface_manifest"][0]["surface_path"] = "/home/private/surface.png"

    with pytest.raises(P0RArtifactError, match="absolute path"):
        write_p0r_package(
            tmp_path / "p0r",
            config_text="schema_version: 1\n",
            author_authority=build_author_registration_authority().as_dict(),
            surface_manifest=rows["surface_manifest"],
            scan_processing_provenance=rows["scan_processing_provenance"],
            registration=rows["registration"],
            registration_qc=rows["registration_qc"],
            grid_mapping_qc=rows["grid_mapping_qc"],
            summary=_summary(),
            report="# P0R\n",
        )


def test_writer_rejects_gate_status_inconsistency_without_output(
    tmp_path: Path,
) -> None:
    summary = _summary()
    summary["status"] = "P0R_AUTHOR_REGISTRATION_NO_GO"
    output = tmp_path / "p0r"

    with pytest.raises(P0RArtifactError, match="inconsistent"):
        _write(output, summary=summary)
    assert not output.exists()


def test_writer_requires_machine_recomputable_gate_facts(tmp_path: Path) -> None:
    summary = _summary()
    del summary["gate_facts"]

    with pytest.raises(P0RArtifactError, match="gate facts"):
        _write(tmp_path / "p0r", summary=summary)

