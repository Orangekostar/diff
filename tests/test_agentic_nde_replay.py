from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cmc_bbdm.agentic_nde.artifacts import ArtifactError, replay_p0, write_p0_package
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


def _package(output: Path) -> Path:
    return write_p0_package(
        output,
        config_text="schema_version: 1\n",
        surface_manifest=[{"specimen_id": "s1"}],
        surface_qc=[{"specimen_id": "s1"}],
        registration=[{"specimen_id": "s1", "status": "UNRESOLVED"}],
        registration_qc=[{"specimen_id": "s1", "authorized": "false"}],
        source_hashes=[{"logical_path": "x", "sha256": "a" * 64}],
        summary=_no_go_summary(),
        report="# Report\n",
    )


def test_replay_verifies_complete_package(tmp_path: Path) -> None:
    result = replay_p0(_package(tmp_path / "p0"))
    assert result["status"] == "P0_SPATIAL_REGISTRATION_NO_GO"


def test_replay_rejects_extra_file(tmp_path: Path) -> None:
    package = _package(tmp_path / "p0")
    (package / "extra.txt").write_text("x", encoding="ascii")
    with pytest.raises(ArtifactError, match="membership"):
        replay_p0(package)


def test_replay_rejects_missing_file(tmp_path: Path) -> None:
    package = _package(tmp_path / "p0")
    (package / "REPORT.md").unlink()
    with pytest.raises(ArtifactError, match="membership"):
        replay_p0(package)


def test_replay_rejects_symlink_member(tmp_path: Path) -> None:
    package = _package(tmp_path / "p0")
    report = package / "REPORT.md"
    payload = tmp_path / "report.txt"
    payload.write_bytes(report.read_bytes())
    report.unlink()
    report.symlink_to(payload)
    with pytest.raises(ArtifactError, match="membership"):
        replay_p0(package)


def test_replay_rejects_hash_drift(tmp_path: Path) -> None:
    package = _package(tmp_path / "p0")
    (package / "summary.json").write_text("{}\n", encoding="ascii")
    with pytest.raises(ArtifactError, match="hash"):
        replay_p0(package)


def _source_bound_package(output: Path, source_root: Path) -> Path:
    source = source_root / "images" / "surface.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"surface-authority")
    payload = source.read_bytes()
    return write_p0_package(
        output,
        config_text="schema_version: 1\n",
        surface_manifest=[{"specimen_id": "s1"}],
        surface_qc=[{"specimen_id": "s1"}],
        registration=[{"specimen_id": "s1", "status": "UNRESOLVED"}],
        registration_qc=[{"specimen_id": "s1", "authorized": "false"}],
        source_hashes=[
            {
                "logical_path": "external_hasebe/images/surface.png",
                "logical_root": "external_hasebe",
                "relative_path": "images/surface.png",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
        summary=_no_go_summary(),
        report="# Report\n",
    )


def test_replay_recomputes_external_source_hashes(tmp_path: Path) -> None:
    source_root = tmp_path / "authority"
    package = _source_bound_package(tmp_path / "p0", source_root)
    assert replay_p0(package, surface_root=source_root)["status"].endswith("NO_GO")


def test_replay_rejects_external_source_drift(tmp_path: Path) -> None:
    source_root = tmp_path / "authority"
    package = _source_bound_package(tmp_path / "p0", source_root)
    (source_root / "images" / "surface.png").write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="source authority hash or size"):
        replay_p0(package, surface_root=source_root)
