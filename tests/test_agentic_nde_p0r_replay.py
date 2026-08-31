from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cmc_bbdm.agentic_nde.author_authority import (
    build_author_registration_authority,
)
from cmc_bbdm.agentic_nde.contracts import PRIMARY_COUNTS, P0RGateFacts, decide_p0r
from cmc_bbdm.agentic_nde.p0r_artifacts import (
    P0RArtifactError,
    replay_p0r_package,
    write_p0r_package,
)


def _package(output: Path) -> Path:
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
    decision = decide_p0r(facts)
    identity = {"dataset_id": "d", "specimen_id": "s"}
    return write_p0r_package(
        output,
        config_text="schema_version: 1\nstage: P0R_AUTHOR_SURFACE_CSCAN_REGISTRATION\n",
        author_authority=build_author_registration_authority().as_dict(),
        surface_manifest=[{**identity, "surface_sha256": "a" * 64}],
        scan_processing_provenance=[
            {**identity, "panel_index": 0, "decoded_pixel_equal": True}
        ],
        registration=[
            {**identity, "orientation": "ROT90", "transform_sha256": "b" * 64}
        ],
        registration_qc=[{**identity, "status": "PASS"}],
        grid_mapping_qc=[
            {**identity, "cell_id": 0, "round_trip_status": "PASS"}
        ],
        summary={**decision.as_dict(), "gate_facts": facts.as_dict()},
        report="# P0R Report\n\nSynthetic package.\n",
    )


def _rebuild_integrity(package: Path) -> None:
    manifest_path = package / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    for name in manifest["files"]:
        value = (package / name).read_bytes()
        manifest["files"][name] = {
            "sha256": hashlib.sha256(value).hexdigest(),
            "size": len(value),
        }
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            ensure_ascii=True,
        )
        + "\n",
        encoding="ascii",
    )
    names = sorted({*manifest["files"], "artifact_manifest.json"})
    (package / "CHECKSUMS.sha256").write_text(
        "".join(
            f"{hashlib.sha256((package / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ),
        encoding="ascii",
    )


def test_replay_verifies_complete_p0r_package(tmp_path: Path) -> None:
    summary = replay_p0r_package(_package(tmp_path / "p0r"))

    assert summary["status"] == "P0R_AUTHOR_REGISTRATION_GO"


def test_replay_rejects_extra_file(tmp_path: Path) -> None:
    package = _package(tmp_path / "p0r")
    (package / "extra.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(P0RArtifactError, match="membership"):
        replay_p0r_package(package)


def test_replay_rejects_missing_file(tmp_path: Path) -> None:
    package = _package(tmp_path / "p0r")
    (package / "registration_qc.csv").unlink()

    with pytest.raises(P0RArtifactError, match="membership"):
        replay_p0r_package(package)


def test_replay_rejects_symlink_member(tmp_path: Path) -> None:
    package = _package(tmp_path / "p0r")
    payload = package / "summary.json"
    replacement = tmp_path / "summary.json"
    payload.replace(replacement)
    payload.symlink_to(replacement)

    with pytest.raises(P0RArtifactError, match="membership"):
        replay_p0r_package(package)


def test_replay_rejects_payload_hash_drift(tmp_path: Path) -> None:
    package = _package(tmp_path / "p0r")
    with (package / "registration.csv").open("a", encoding="utf-8") as stream:
        stream.write("drift\n")

    with pytest.raises(P0RArtifactError, match="hash or size"):
        replay_p0r_package(package)


def test_replay_rejects_author_statement_drift_even_with_rehashed_package(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path / "p0r")
    author = package / "author_authority.json"
    payload = build_author_registration_authority().as_dict()
    payload["orientation"] = "IDENTITY"
    author.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    _rebuild_integrity(package)

    with pytest.raises(P0RArtifactError, match="author authority"):
        replay_p0r_package(package)
