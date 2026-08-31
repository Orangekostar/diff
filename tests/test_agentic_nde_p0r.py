from __future__ import annotations

import csv
import hashlib
import math
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from cmc_bbdm.agentic_nde.artifacts import write_p0_package
from cmc_bbdm.agentic_nde.author_authority import VERBATIM_STATEMENT
from cmc_bbdm.agentic_nde.contracts import (
    NOT_AUTHORIZED_STAGES,
    PRIMARY_COUNTS,
    P0GateFacts,
    P0RGateFacts,
    P0RStatus,
    StageStatus,
    decide_p0,
    decide_p0r,
)
from cmc_bbdm.agentic_nde.p0r import P0RPipelineError, audit_p0r
from cmc_bbdm.agentic_nde.p0r_artifacts import replay_p0r_package

ROOT = Path(__file__).resolve().parents[1]


def _passing_facts() -> P0RGateFacts:
    return P0RGateFacts(
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


def test_all_p0r_requirements_yield_go() -> None:
    decision = decide_p0r(_passing_facts())

    assert decision.status is P0RStatus.GO
    assert decision.reasons == ()
    assert decision.downstream_registration_status is StageStatus.P0_GO
    assert decision.p1_authorized is True
    assert decision.downstream == ()


def test_author_evidence_conflict_has_highest_precedence() -> None:
    facts = _passing_facts()
    decision = decide_p0r(
        P0RGateFacts(
            **{
                **facts.as_dict(),
                "author_evidence_conflict": True,
                "processing_provenance_unresolved": True,
            }
        )
    )

    assert decision.status is P0RStatus.CONFLICT
    assert decision.reasons == ("author_evidence_conflict",)
    assert decision.p1_authorized is False
    assert decision.downstream == NOT_AUTHORIZED_STAGES


def test_unresolved_processing_provenance_has_distinct_status() -> None:
    facts = _passing_facts()
    decision = decide_p0r(
        P0RGateFacts(
            **{
                **facts.as_dict(),
                "processing_provenance_unresolved": True,
            }
        )
    )

    assert decision.status is P0RStatus.PROVENANCE_UNRESOLVED
    assert decision.reasons == ("processing_provenance_unresolved",)
    assert decision.downstream_registration_status is StageStatus.P0_SPATIAL_REGISTRATION_NO_GO


def test_result_driven_orientation_fails_closed() -> None:
    facts = _passing_facts()
    decision = decide_p0r(
        P0RGateFacts(
            **{
                **facts.as_dict(),
                "no_result_driven_orientation": False,
            }
        )
    )

    assert decision.status is P0RStatus.NO_GO
    assert decision.reasons == ("result_driven_orientation_not_excluded",)
    assert decision.p1_authorized is False


def test_each_boolean_gate_has_a_closed_reason() -> None:
    expected = {
        "exact_identity_hashes": "identity_or_hash_binding_failed",
        "author_statement_bound": "author_statement_not_bound",
        "global_orientation_rot90": "global_rot90_not_fixed",
        "all_panels_resolved": "specimen_panel_unresolved",
        "processing_provenance_deterministic": "processing_provenance_not_deterministic",
        "no_unsupported_rotation_reflection": "unsupported_rotation_or_reflection",
        "composed_transform_replayable": "composed_transform_not_replayable",
    }
    passing = _passing_facts().as_dict()

    for field, reason in expected.items():
        decision = decide_p0r(P0RGateFacts(**{**passing, field: False}))
        assert decision.status is P0RStatus.NO_GO
        assert reason in decision.reasons


def test_p0r_coverage_threshold_applies_per_domain_and_total() -> None:
    facts = _passing_facts()
    counts = dict(PRIMARY_COUNTS)
    domain = "74t7kcdgkr"
    counts[domain] = math.ceil(PRIMARY_COUNTS[domain] * 0.9) - 1
    decision = decide_p0r(
        P0RGateFacts(**{**facts.as_dict(), "authorized_by_domain": counts})
    )

    assert decision.status is P0RStatus.NO_GO
    assert any(reason.startswith(f"coverage_below_90_percent:{domain}:") for reason in decision.reasons)

    low_total = {key: 0 for key in PRIMARY_COUNTS}
    decision = decide_p0r(
        P0RGateFacts(**{**facts.as_dict(), "authorized_by_domain": low_total})
    )
    assert "coverage_below_240_total:0/276" in decision.reasons


def test_p0r_rejects_unexpected_domains() -> None:
    facts = _passing_facts()
    counts = {**dict(PRIMARY_COUNTS), "unexpected": 1}
    decision = decide_p0r(
        P0RGateFacts(**{**facts.as_dict(), "authorized_by_domain": counts})
    )

    assert decision.status is P0RStatus.NO_GO
    assert "unexpected_primary_domains:unexpected" in decision.reasons


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _primary_keys() -> list[tuple[str, str]]:
    return [
        (domain, f"specimen-{index:03d}")
        for domain, count in PRIMARY_COUNTS.items()
        for index in range(count)
    ]


def _make_synthetic_authorities(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "project"
    external = tmp_path / "external"
    data = external / "data"
    preprocessor = external / "src/cmc_bbdm/hasebe.py"
    authority = project / "artifacts/authority.md"
    old_p0 = project / "results/old_p0"
    for path in (data, preprocessor.parent, authority.parent, old_p0.parent):
        path.mkdir(parents=True, exist_ok=True)

    surface = data / "surface.png"
    raw = data / "raw.jpg"
    crop = data / "crop.png"
    Image.new("RGB", (12, 8), (20, 80, 140)).save(surface)
    raw_image = Image.new("RGB", (669, 885), (8, 16, 24))
    raw_image.save(raw, quality=95)
    with Image.open(raw) as decoded:
        decoded.convert("RGB").crop((39, 33, 469, 708)).save(crop)
    preprocessor.write_text(
        "def crop_cscan_panels(image):\n    return image.crop((39, 33, 469, 708))\n",
        encoding="ascii",
    )
    authority.write_text(
        "3560662d4509ea3e059d597cedca15950cce02f706a992330b161381acfba6ba\n"
        f"{VERBATIM_STATEMENT}\n",
        encoding="utf-8",
    )

    relative_surface = surface.relative_to(external).as_posix()
    relative_raw = raw.relative_to(external).as_posix()
    relative_crop = crop.relative_to(external).as_posix()
    surface_hash = _sha256(surface)
    raw_hash = _sha256(raw)
    crop_hash = _sha256(crop)
    keys = _primary_keys()
    surface_manifest = [
        {
            "dataset_id": domain,
            "specimen_id": specimen,
            "impacted_surface_path": relative_surface,
            "surface_sha256": surface_hash,
            "surface_bytes": surface.stat().st_size,
            "cscan_source_path": relative_raw,
            "cscan_source_sha256": raw_hash,
            "registered_cscan_crop_path": relative_crop,
            "registered_cscan_crop_sha256": crop_hash,
            "registered_cscan_width_px": 430,
            "registered_cscan_height_px": 675,
            "cscan_panel_index": 0,
            "identity_status": "PASS_EXACT_SPECIMEN_ID_AND_HASH",
        }
        for domain, specimen in keys
    ]
    surface_qc = [
        {
            "dataset_id": domain,
            "specimen_id": specimen,
            "sha256": surface_hash,
            "width_px": 12,
            "height_px": 8,
            "mode": "RGB",
        }
        for domain, specimen in keys
    ]
    old_registration = [
        {
            "dataset_id": domain,
            "specimen_id": specimen,
            "status": "UNRESOLVED",
        }
        for domain, specimen in keys
    ]
    old_registration_qc = [
        {
            "dataset_id": domain,
            "specimen_id": specimen,
            "authorized": "false",
        }
        for domain, specimen in keys
    ]
    source_hashes = [
        {
            "logical_path": f"external_hasebe/{relative}",
            "logical_root": "external_hasebe",
            "relative_path": relative,
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path, relative, role in (
            (surface, relative_surface, "impacted_surface"),
            (raw, relative_raw, "raw_cscan_screenshot"),
            (crop, relative_crop, "registered_cscan_crop"),
        )
    ]
    old_facts = P0GateFacts(
        authorized_by_domain={domain: 0 for domain in PRIMARY_COUNTS},
        exact_identity_hashes=True,
        orientation_resolved=False,
        deterministic_transform=False,
        deployable_evidence_only=True,
        replay_verified=False,
    )
    old_decision = decide_p0(old_facts)
    write_p0_package(
        old_p0,
        config_text="schema_version: 1\nstage: SYNTHETIC_HISTORICAL_P0\n",
        surface_manifest=surface_manifest,
        surface_qc=surface_qc,
        registration=old_registration,
        registration_qc=old_registration_qc,
        source_hashes=source_hashes,
        summary={**old_decision.as_dict(), "gate_facts": old_facts.as_dict()},
        report="# Synthetic historical P0\n",
    )

    paired = data / "paired.csv"
    paired_rows = [
        {
            "dataset_id": domain,
            "sample_id": specimen,
            "source_path": relative_surface,
            "source_sha256": surface_hash,
            "source_width": 12,
            "source_height": 8,
            "target_screenshot_path": relative_raw,
            "target_screenshot_sha256": raw_hash,
            "target_panel_index": 0,
            "target_path": relative_crop,
            "target_sha256": crop_hash,
            "target_width": 430,
            "target_height": 675,
        }
        for domain, specimen in keys
    ]
    paired_rows.append(
        {
            **paired_rows[0],
            "dataset_id": "6zt73pcnxv",
            "sample_id": "q24-7astm",
        }
    )
    _write_csv(paired, paired_rows)

    template = yaml.safe_load(
        (ROOT / "paper_v3/configs/agentic_nde_p0r_author_registration.yaml").read_text(
            encoding="utf-8"
        )
    )
    template["historical_p0"] = {
        "path": old_p0.relative_to(project).as_posix(),
        "status": "P0_SPATIAL_REGISTRATION_NO_GO",
        "checksums_sha256": _sha256(old_p0 / "CHECKSUMS.sha256"),
        "artifact_manifest_sha256": _sha256(old_p0 / "artifact_manifest.json"),
        "surface_manifest_sha256": _sha256(old_p0 / "surface_manifest.csv"),
        "surface_qc_sha256": _sha256(old_p0 / "surface_qc.csv"),
        "summary_sha256": _sha256(old_p0 / "summary.json"),
    }
    template["author_authority"]["artifact_path"] = authority.relative_to(
        project
    ).as_posix()
    template["author_authority"]["artifact_sha256"] = _sha256(authority)
    template["external_authorities"]["paired_manifest"]["path"] = paired.relative_to(
        external
    ).as_posix()
    template["external_authorities"]["paired_manifest"]["sha256"] = _sha256(
        paired
    )
    template["external_authorities"]["historical_preprocessor"][
        "path"
    ] = preprocessor.relative_to(external).as_posix()
    template["external_authorities"]["historical_preprocessor"][
        "sha256"
    ] = _sha256(preprocessor)
    config = project / "config.yaml"
    config.write_text(
        yaml.safe_dump(template, sort_keys=False, allow_unicode=False),
        encoding="ascii",
    )
    return {
        "project": project,
        "external": external,
        "config": config,
        "authority": authority,
        "preprocessor": preprocessor,
        "old_p0": old_p0,
        "paired": paired,
        "surface": surface,
    }


def test_p0r_pipeline_requires_explicit_surface_root(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="ascii")

    with pytest.raises(P0RPipelineError, match="surface root"):
        audit_p0r(
            config_path=config,
            surface_root=tmp_path / "missing",
            output=tmp_path / "output",
            project_root=tmp_path,
        )


def test_p0r_pipeline_refuses_existing_output_before_work(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="ascii")
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(P0RPipelineError, match="output"):
        audit_p0r(
            config_path=config,
            surface_root=tmp_path,
            output=output,
            project_root=tmp_path,
        )


def test_importing_p0r_does_not_import_model_frameworks() -> None:
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "sklearn" not in sys.modules


def test_synthetic_p0r_audit_and_source_replay_are_complete(tmp_path: Path) -> None:
    fixture = _make_synthetic_authorities(tmp_path)
    output = tmp_path / "p0r"

    audit_p0r(
        config_path=fixture["config"],
        surface_root=fixture["external"],
        output=output,
        project_root=fixture["project"],
    )
    summary = replay_p0r_package(
        output,
        surface_root=fixture["external"],
        project_root=fixture["project"],
    )

    assert summary["status"] == "P0R_AUTHOR_REGISTRATION_GO"
    assert summary["authorized_registration_count"] == 276
    with (output / "grid_mapping_qc.csv").open(encoding="utf-8", newline="") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 276 * 64


@pytest.mark.parametrize("authority_name", ["authority", "preprocessor"])
def test_p0r_audit_rejects_changed_bound_authority(
    tmp_path: Path, authority_name: str
) -> None:
    fixture = _make_synthetic_authorities(tmp_path)
    fixture[authority_name].write_text("changed\n", encoding="ascii")

    with pytest.raises(P0RPipelineError, match="SHA-256"):
        audit_p0r(
            config_path=fixture["config"],
            surface_root=fixture["external"],
            output=tmp_path / "p0r",
            project_root=fixture["project"],
        )


def test_p0r_audit_rejects_changed_historical_package(tmp_path: Path) -> None:
    fixture = _make_synthetic_authorities(tmp_path)
    with (fixture["old_p0"] / "surface_manifest.csv").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write("drift\n")

    with pytest.raises(P0RPipelineError, match="historical P0"):
        audit_p0r(
            config_path=fixture["config"],
            surface_root=fixture["external"],
            output=tmp_path / "p0r",
            project_root=fixture["project"],
        )


def test_p0r_source_replay_rejects_post_audit_surface_drift(tmp_path: Path) -> None:
    fixture = _make_synthetic_authorities(tmp_path)
    output = tmp_path / "p0r"
    audit_p0r(
        config_path=fixture["config"],
        surface_root=fixture["external"],
        output=output,
        project_root=fixture["project"],
    )
    fixture["surface"].write_bytes(b"changed")

    with pytest.raises(P0RPipelineError):
        replay_p0r_package(
            output,
            surface_root=fixture["external"],
            project_root=fixture["project"],
        )


def test_p0r_audit_rejects_panel_index_contradiction(tmp_path: Path) -> None:
    fixture = _make_synthetic_authorities(tmp_path)
    rows: list[dict[str, str]]
    with fixture["paired"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["target_panel_index"] = "1"
    _write_csv(fixture["paired"], rows)
    config = yaml.safe_load(fixture["config"].read_text(encoding="ascii"))
    config["external_authorities"]["paired_manifest"]["sha256"] = _sha256(
        fixture["paired"]
    )
    fixture["config"].write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="ascii"
    )

    with pytest.raises(P0RPipelineError, match="panel"):
        audit_p0r(
            config_path=fixture["config"],
            surface_root=fixture["external"],
            output=tmp_path / "p0r",
            project_root=fixture["project"],
        )
