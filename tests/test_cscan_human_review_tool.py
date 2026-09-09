from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from cmc_bbdm.vlm_cscan.references import reference_from_payload
from cmc_bbdm.vlm_cscan.runtime import InputSpecimen

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/cscan_human_review.py"
CONFIG = ROOT / "paper_v3/configs/cscan_human_review_tool.yaml"
SOURCE_ROOT = Path("/home/ww/paper3/cmc_damage_inference")


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _input_specimen(tmp_path: Path, *, specimen_id: str = "sample-1") -> InputSpecimen:
    pixels = np.zeros((11, 17, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(17, dtype=np.uint8)
    pixels[..., 1] = np.arange(11, dtype=np.uint8)[:, None]
    pixels[3:8, 5:12, 2] = 220
    image_path = tmp_path / f"{specimen_id}.png"
    Image.fromarray(pixels, mode="RGB").save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return InputSpecimen(
        dataset_id="domain-a",
        specimen_id=specimen_id,
        surface_path=image_path,
        surface_sha256=digest,
        cscan_path=image_path,
        cscan_sha256=digest,
        native_shape=(11, 17),
        transform_sha256="b" * 64,
    )


def _blind_layer() -> dict[str, object]:
    def encoded(image: Image.Image) -> tuple[str, str]:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
        return (
            "data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
            hashlib.sha256(payload).hexdigest(),
        )

    measured_data, measured_sha = encoded(Image.new("RGB", (17, 11), (31, 42, 53)))
    report_data, report_sha = encoded(Image.new("RGBA", (17, 11), (0, 0, 0, 0)))
    support_data, support_sha = encoded(Image.new("RGBA", (17, 11), (0, 0, 0, 0)))
    return {
        "native_height": 11,
        "native_width": 17,
        "measured_image_mime": "image/png",
        "measured_image_data": measured_data,
        "measured_image_sha256": measured_sha,
        "report_overlay_mime": "image/png",
        "report_overlay_data": report_data,
        "report_overlay_sha256": report_sha,
        "support_overlay_mime": "image/png",
        "support_overlay_data": support_data,
        "support_overlay_sha256": support_sha,
    }


def test_cli_help_exposes_all_six_human_review_workflows() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0, result.stderr
    for command in (
        "build-ui",
        "prepare-references",
        "prepare-blind",
        "validate-return",
        "export-references",
        "export-blind-reviews",
    ):
        assert command in result.stdout


def test_config_binds_the_frozen_evidence_without_model_calls() -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import (
        load_human_review_config,
    )

    config = load_human_review_config(CONFIG, project_root=ROOT)

    assert config.repository_base_sha == "2102cc4a1726910931dfaaf20e29ad29a20eaf2e"
    assert config.reference_packet_size == 6
    assert config.blind_packet_size == 32
    assert config.blinding_condition == "FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1"
    assert config.resource_limits == {
        "training_updates": 0,
        "vlm_calls": 0,
        "actor_stop_forward_calls": 0,
        "cpu_processes": 1,
    }


def test_reference_packet_preserves_registered_pixels_and_source_identity(
    tmp_path: Path,
) -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import build_reference_packets

    record = _input_specimen(tmp_path)

    result = build_reference_packets(
        (record,), packet_size=6, source_base_sha="2" * 40
    )

    assert len(result["packets"]) == 1
    packet = result["packets"][0]
    item = packet["items"][0]
    assert packet["packet_kind"] == "REFERENCE_ANNOTATION"
    assert item["specimen_key"] == "domain-a:sample-1"
    assert (item["native_height"], item["native_width"]) == (11, 17)
    assert item["source_image_sha256"] == record.cscan_sha256
    assert "source_image_path" not in json.dumps(packet)
    prefix, encoded = item["image_data"].split(",", 1)
    assert prefix == "data:image/png;base64"
    displayed = Image.open(io.BytesIO(base64.b64decode(encoded)))
    expected = Image.open(record.cscan_path)
    assert displayed.size == expected.size
    assert np.array_equal(np.asarray(displayed), np.asarray(expected))
    assert item["display_image_sha256"] == hashlib.sha256(
        base64.b64decode(encoded)
    ).hexdigest()


def test_reference_export_uses_existing_parser_and_keeps_audit_outside_refs(
    tmp_path: Path,
) -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import (
        build_reference_packets,
        export_reference_session,
    )

    record = _input_specimen(tmp_path)
    packet = build_reference_packets(
        (record,), packet_size=6, source_base_sha="2" * 40
    )["packets"][0]
    item_id = packet["items"][0]["item_id"]
    session = {
        "schema_version": 1,
        "session_kind": "CSCAN_REFERENCE_SESSION",
        "packet_id": packet["packet_id"],
        "export_id": "export-001",
        "reviewer": {
            "reviewer_alias": "专家甲",
            "reference_type": "EXPERT_REVIEWED",
            "participated_in_method_development": False,
            "saw_model_outputs": False,
            "annotation_origin": "FROM_SCRATCH",
        },
        "items": {
            item_id: {
                "state": "CONFIRMED",
                "regions": [
                    {
                        "id": "certain-1",
                        "certainty": "certain",
                        "polygon": [[0.1, 0.2], [0.8, 0.2], [0.5, 0.8]],
                    }
                ],
                "uncertain_regions": [
                    {
                        "id": "uncertain-1",
                        "certainty": "uncertain",
                        "polygon": [[0.0, 0.0], [0.3, 0.0], [0.2, 0.4]],
                    }
                ],
                "notes": "包含中文、逗号\n以及换行",
            }
        },
    }

    summary = export_reference_session(
        session=session, packet=packet, output_root=tmp_path / "converted"
    )

    references_root = Path(summary["references_root"])
    reference_files = tuple(references_root.glob("*.json"))
    assert len(reference_files) == 1
    assert tuple(references_root.iterdir()) == reference_files
    payload = json.loads(reference_files[0].read_text(encoding="utf-8"))
    parsed = reference_from_payload(payload, native_shape=(11, 17))
    assert parsed.formal_eligible is True
    assert parsed.reviewer_alias == "专家甲"
    assert summary["exported_reference_count"] == 1
    assert Path(summary["annotation_audit_path"]).parent == references_root.parent
    assert Path(summary["annotation_audit_path"]).parent != references_root


def test_reference_packet_rejects_a_changed_source_image(tmp_path: Path) -> None:
    from dataclasses import replace

    from cmc_bbdm.learned_cscan.human_review_tool import build_reference_packets

    record = replace(_input_specimen(tmp_path), cscan_sha256="0" * 64)

    with pytest.raises(ValueError, match="source image hash"):
        build_reference_packets((record,), packet_size=6, source_base_sha="2" * 40)


def test_return_rejects_reference_packet_dimension_tampering(tmp_path: Path) -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import (
        build_reference_packets,
        validate_return_session,
    )

    packet = build_reference_packets(
        (_input_specimen(tmp_path),), packet_size=6, source_base_sha="2" * 40
    )["packets"][0]
    packet["items"][0]["native_width"] += 1
    session = {
        "schema_version": 1,
        "session_kind": "CSCAN_REFERENCE_SESSION",
        "packet_id": packet["packet_id"],
        "items": {},
    }

    with pytest.raises(ValueError, match="image shape"):
        validate_return_session(session=session, packet=packet)


def test_stored_prefix_replay_does_not_apply_the_stop_step_action() -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import replay_stored_prefix
    from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid

    grid = build_acquisition_grid(19, 23, initial_budget=0.015625)
    rows = (
        {"step": 0, "action_cell": 0, "action_from_level": -1, "action_to_level": 0},
        {"step": 1, "action_cell": 1, "action_from_level": -1, "action_to_level": 0},
        {"step": 2, "action_cell": 2, "action_from_level": -1, "action_to_level": 0},
    )

    endpoint = replay_stored_prefix(grid=grid, trajectory_rows=rows, stop_step=2)

    assert endpoint.action_count == 2
    assert endpoint.state.levels[:4] == (0, 0, -1, -1)
    assert endpoint.measured_count == np.count_nonzero(endpoint.measured_mask)
    assert endpoint.measured_cost == pytest.approx(
        endpoint.measured_count / endpoint.measured_mask.size
    )


def test_blind_packet_serialization_excludes_full_scan_and_private_identity() -> None:
    from cmc_bbdm.learned_cscan.contracts import Task
    from cmc_bbdm.learned_cscan.human_review_tool import (
        build_blind_packets,
        render_blind_layers,
        validate_return_session,
    )
    from cmc_bbdm.learned_cscan.readout import TaskReportV2

    full_scan = np.zeros((11, 17, 3), dtype=np.uint8)
    full_scan[..., 0] = 231
    full_scan[..., 1] = 17
    full_scan[..., 2] = 99
    measured = np.zeros((11, 17), dtype=np.bool_)
    measured[2:5, 3:8] = True
    predicted = np.zeros_like(measured)
    predicted[3:8, 5:12] = True
    report = TaskReportV2(
        task=Task.LOCATE,
        predicted_mask=predicted,
        support_positions=np.asarray([[3, 5], [4, 6]], dtype=np.int64),
        candidate_cells=(0,),
        unverified_boundary_cells=(),
        signal_strength=0.5,
        reason_code="VISIBLE_CANDIDATE",
    )
    layers = render_blind_layers(
        full_scan=full_scan, measured_mask=measured, report=report
    )
    private_row = {
        "report_id": "report-001",
        "specimen_key": "secret-domain:secret-specimen",
        "task": "LOCATE",
        "method": "BC_S3",
        "seed": 3,
        "stop_system": "S_BC_CAL",
        "stop_step": 42,
        "report_sha256": "d" * 64,
        "objective_success": True,
        "reference_version": "PROXY_LEGACY",
    }

    result = build_blind_packets(
        ((private_row, layers),),
        packet_size=32,
        shuffle_seed=20260909,
        source_base_sha="2" * 40,
        blinding_condition="FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1",
    )

    packet_bytes = json.dumps(result["packets"], sort_keys=True).encode()
    assert b"BC_S3" not in packet_bytes
    assert b"secret-domain" not in packet_bytes
    assert b'"seed"' not in packet_bytes
    assert b'"stop_step"' not in packet_bytes
    assert len(result["private_index"]) == 1
    assert result["private_index"]["report-001"]["method"] == "BC_S3"
    measured_png = base64.b64decode(
        layers["measured_image_data"].split(",", 1)[1]
    )
    sanitized = np.asarray(Image.open(io.BytesIO(measured_png)))
    assert np.array_equal(sanitized[3, 5], full_scan[3, 5])
    assert not np.array_equal(sanitized[0, 0], full_scan[0, 0])
    packet = result["packets"][0]
    packet["items"][0]["method"] = "BC_S3"
    session = {
        "schema_version": 1,
        "session_kind": "CSCAN_BLIND_REVIEW_SESSION",
        "packet_id": packet["packet_id"],
        "blinding_condition": "FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1",
        "items": {},
    }
    with pytest.raises(ValueError, match="public fields"):
        validate_return_session(session=session, packet=packet)


def test_blind_export_quotes_notes_and_reuses_existing_summary(tmp_path: Path) -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import (
        build_blind_packets,
        export_blind_review_session,
    )

    layer = _blind_layer()
    row = {
        "report_id": "report-001",
        "specimen_key": "domain-a:sample-1",
        "task": "LOCATE",
        "method": "R_BALANCED_P8",
        "seed": 1,
        "stop_system": "S_BC_CAL",
        "report_sha256": "d" * 64,
        "objective_success": False,
        "reference_version": "PROXY_LEGACY",
    }
    built = build_blind_packets(
        ((row, layer),),
        packet_size=32,
        shuffle_seed=20260909,
        source_base_sha="2" * 40,
        blinding_condition="FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1",
    )
    packet = built["packets"][0]
    item_id = packet["items"][0]["item_id"]
    session = {
        "schema_version": 1,
        "session_kind": "CSCAN_BLIND_REVIEW_SESSION",
        "packet_id": packet["packet_id"],
        "export_id": "blind-export-001",
        "reviewer_id": "reviewer-甲",
        "blinding_condition": "FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1",
        "items": {
            item_id: {
                "state": "CONFIRMED",
                "decision": "UNABLE_TO_JUDGE",
                "problem_type": "边界,模糊",
                "review_basis": "仅首次停止证据",
                "notes": "第一行，含逗号\n第二行\"含引号\"",
            }
        },
        "change_log": [{"item_id": item_id, "from": "DRAFT", "to": "CONFIRMED"}],
    }

    result = export_blind_review_session(
        session=session,
        packet=packet,
        private_index=built["private_index"],
        output_root=tmp_path / "converted",
    )

    csv_text = Path(result["blind_reviews_path"]).read_text(encoding="utf-8")
    assert '"边界,模糊"' in csv_text
    assert '"第一行，含逗号\n第二行""含引号"""' in csv_text
    assert result["confirmed_review_count"] == 1
    assert result["summary"]["report_decisions"][0]["decision"] == "UNABLE_TO_JUDGE"
    assert result["summary"]["method_summary"][0]["consensus"] == "NOT_COMPUTED"
    raw = json.loads(Path(result["raw_session_path"]).read_text(encoding="utf-8"))
    assert raw["change_log"] == session["change_log"]


def test_blind_export_rejects_unmatched_session_items(tmp_path: Path) -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import (
        build_blind_packets,
        export_blind_review_session,
    )

    private = {
        "report_id": "report-001",
        "specimen_key": "domain-a:sample-1",
        "task": "LOCATE",
        "method": "BC_S1",
        "seed": 1,
        "stop_system": "S_BC_CAL",
        "stop_step": 3,
        "report_sha256": "d" * 64,
        "objective_success": True,
        "objective_scope": "PROXY_LEGACY",
        "reference_version": "PROXY_LEGACY",
    }
    built = build_blind_packets(
        ((private, _blind_layer()),),
        packet_size=32,
        shuffle_seed=20260909,
        source_base_sha="2" * 40,
        blinding_condition="FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1",
    )
    packet = built["packets"][0]

    with pytest.raises(ValueError, match="item cannot be matched"):
        export_blind_review_session(
            session={
                "schema_version": 1,
                "session_kind": "CSCAN_BLIND_REVIEW_SESSION",
                "packet_id": packet["packet_id"],
                "export_id": "export",
                "reviewer_id": "reviewer",
                "blinding_condition": "FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1",
                "items": {"unknown": {"state": "DRAFT"}},
            },
            packet=packet,
            private_index=built["private_index"],
            output_root=tmp_path,
        )


def test_real_reference_preparation_uses_the_frozen_24_specimen_test_roster(
    tmp_path: Path,
) -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import (
        load_human_review_config,
        prepare_reference_packets,
    )

    config = load_human_review_config(CONFIG, project_root=ROOT)

    summary = prepare_reference_packets(
        config,
        source_root=SOURCE_ROOT,
        packet_root=tmp_path / "local/reference_packets",
        result_root=tmp_path / "results",
        dist_root=tmp_path / "dist",
    )

    assert summary["reference_packet_status"] == "PREPARED"
    assert summary["physical_specimen_count"] == 24
    assert summary["domain_count"] == 6
    assert summary["packet_count"] == 4
    assert summary["missing_source_count"] == 0
    assert len(tuple((tmp_path / "local/reference_packets").glob("*.json"))) == 4
    manifest_lines = (tmp_path / "results/reference_packet_manifest.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(manifest_lines) == 25
    practice = json.loads(
        (tmp_path / "dist/reference_practice_TEST_ONLY.json").read_text(
            encoding="utf-8"
        )
    )
    assert practice["test_only"] is True
    assert practice["packet_kind"] == "REFERENCE_ANNOTATION"


def test_build_ui_creates_a_self_contained_offline_delivery(tmp_path: Path) -> None:
    output = tmp_path / "index.html"

    result = _run_cli("build-ui", "--output", str(output))

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert Path(summary["output_path"]) == output
    html = output.read_text(encoding="utf-8")
    assert "<!doctype html>" in html.lower()
    assert 'src="https://' not in html
    assert 'src="http://' not in html
    assert 'href="https://' not in html
    assert 'href="http://' not in html
    assert 'id="packet-file"' in html
    assert 'id="session-file"' in html
    assert 'id="review-svg"' in html
    assert "REFERENCE_ANNOTATION" in html
    assert "BLIND_FIRST_STOP_REVIEW" in html


def test_real_p8_and_bc_stop_recovery_matches_frozen_report_and_cost(
    tmp_path: Path,
) -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import (
        load_human_review_config,
        recover_blind_reports,
    )

    config = load_human_review_config(CONFIG, project_root=ROOT)
    stop_table = pl.read_csv(config.first_stop_path, infer_schema_length=None)
    specimen_key = stop_table.filter(
        (pl.col("task") == "LOCATE")
        & (pl.col("method") == "R_BALANCED_P8")
        & pl.col("stopped")
    )["specimen_key"][0]
    selected = stop_table.filter(
        (pl.col("specimen_key") == specimen_key)
        & (pl.col("task") == "LOCATE")
        & pl.col("method").is_in(["R_BALANCED_P8", "BC_S1"])
        & (pl.col("stop_system") == "S_BC_CAL")
        & pl.col("stopped")
    ).sort("method")

    result = recover_blind_reports(
        config,
        source_root=SOURCE_ROOT,
        stop_rows=tuple(selected.to_dicts()),
        trajectories=pl.read_parquet(config.trajectories_path),
        cache_root=tmp_path / "cache",
    )

    assert len(result["reports_and_layers"]) == 2
    assert result["recovery_summary"]["reader_report_count"] == 2
    assert result["recovery_summary"]["report_digest_mismatch_count"] == 0
    assert result["recovery_summary"]["cost_mismatch_count"] == 0
    assert result["recovery_summary"]["stored_action_transition_count"] == sum(
        int(row["stop_step"]) for row in selected.to_dicts()
    )
    for private, layers in result["reports_and_layers"]:
        source = next(row for row in selected.to_dicts() if row["report_id"] == private["report_id"])
        assert private["report_sha256"] == source["stop_report_sha256"]
        assert layers["native_height"] > 0
        assert layers["native_width"] > 0


def test_blind_selection_keeps_all_187_stops_and_all_5_no_stop_runs() -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import (
        load_human_review_config,
        select_blind_stop_rows,
    )

    config = load_human_review_config(CONFIG, project_root=ROOT)

    selected = select_blind_stop_rows(config)

    assert len(selected["stopped_rows"]) == 187
    assert len(selected["no_stop_rows"]) == 5
    assert len(selected["stopped_rows"]) + len(selected["no_stop_rows"]) == 192
    assert {row["method"] for row in selected["stopped_rows"]} == {
        "R_BALANCED_P8",
        "BC_S1",
        "BC_S2",
        "BC_S3",
    }
    assert {row["task"] for row in selected["stopped_rows"]} == {
        "LOCATE",
        "CHARACTERIZE",
    }
    assert all(row["report_id"] for row in selected["stopped_rows"])
    assert all(not row["report_id"] for row in selected["no_stop_rows"])


def test_blind_bundle_writes_public_packets_separately_from_private_index(
    tmp_path: Path,
) -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import (
        build_blind_packets,
        write_blind_packet_bundle,
    )

    layer = _blind_layer()
    private = {
        "report_id": "report-001",
        "specimen_key": "domain-a:sample-1",
        "task": "LOCATE",
        "method": "BC_S1",
        "seed": 1,
        "stop_system": "S_BC_CAL",
        "stop_step": 3,
        "report_sha256": "d" * 64,
        "objective_success": True,
        "objective_scope": "PROXY_LEGACY",
        "reference_version": "PROXY_LEGACY",
    }
    built = build_blind_packets(
        ((private, layer),),
        packet_size=32,
        shuffle_seed=20260909,
        source_base_sha="2" * 40,
        blinding_condition="FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1",
    )

    summary = write_blind_packet_bundle(
        built=built,
        no_stop_rows=({
            "dataset_id": "domain-a",
            "specimen_id": "sample-2",
            "specimen_key": "domain-a:sample-2",
            "task": "CHARACTERIZE",
            "method": "BC_S2",
            "seed": 2,
            "stop_system": "S_BC_CAL",
            "exhausted": True,
        },),
        recovery_summary={
            "reader_report_count": 1,
            "stored_action_transition_count": 3,
            "world_step_count": 0,
        },
        packet_root=tmp_path / "public",
        private_index_root=tmp_path / "private",
        result_root=tmp_path / "results",
    )

    public_text = (tmp_path / "public/blind_packet_001.json").read_text(
        encoding="utf-8"
    )
    assert "BC_S1" not in public_text
    assert "domain-a:sample-1" not in public_text
    private_text = (tmp_path / "private/report_index.json").read_text(
        encoding="utf-8"
    )
    assert "BC_S1" in private_text
    assert summary["blind_packet_status"] == "PREPARED"
    assert summary["eligible_report_count"] == 1
    assert summary["no_stop_count"] == 1
    assert (tmp_path / "results/blind_packet_manifest.csv").is_file()
    assert (tmp_path / "results/no_stop_coverage.csv").is_file()


def test_validate_return_accepts_a_portable_unfinished_reference_backup(
    tmp_path: Path,
) -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import build_reference_packets

    packet = build_reference_packets(
        (_input_specimen(tmp_path),), packet_size=6, source_base_sha="2" * 40
    )["packets"][0]
    item_id = packet["items"][0]["item_id"]
    session = {
        "schema_version": 1,
        "session_kind": "CSCAN_REFERENCE_SESSION",
        "packet_id": packet["packet_id"],
        "export_id": "draft-backup",
        "reviewer": {
            "reviewer_alias": "",
            "reference_type": "",
            "participated_in_method_development": False,
            "saw_model_outputs": False,
            "annotation_origin": "FROM_SCRATCH",
        },
        "items": {
            item_id: {
                "state": "DRAFT",
                "regions": [],
                "uncertain_regions": [],
                "open_polygon": {
                    "certainty": "certain",
                    "polygon": [[0.1, 0.1], [0.4, 0.1]],
                },
                "notes": "尚未完成",
            }
        },
    }
    packet_path = tmp_path / "packet.json"
    session_path = tmp_path / "session.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    session_path.write_text(json.dumps(session), encoding="utf-8")

    result = _run_cli(
        "validate-return",
        "--session",
        str(session_path),
        "--packet",
        str(packet_path),
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "VALID_SESSION_BACKUP"
    assert summary["draft_count"] == 1
    assert summary["canonical_ready_count"] == 0


def test_validate_return_rejects_confirmed_reference_with_open_polygon(
    tmp_path: Path,
) -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import (
        build_reference_packets,
        validate_return_session,
    )

    packet = build_reference_packets(
        (_input_specimen(tmp_path),), packet_size=6, source_base_sha="2" * 40
    )["packets"][0]
    item_id = packet["items"][0]["item_id"]
    session = {
        "schema_version": 1,
        "session_kind": "CSCAN_REFERENCE_SESSION",
        "packet_id": packet["packet_id"],
        "reviewer": {
            "reviewer_alias": "reviewer-a",
            "reference_type": "EXPERT_REVIEWED",
            "participated_in_method_development": False,
            "saw_model_outputs": False,
            "annotation_origin": "FROM_SCRATCH",
        },
        "items": {
            item_id: {
                "state": "CONFIRMED",
                "regions": [
                    {
                        "id": "certain-1",
                        "certainty": "certain",
                        "polygon": [[0.1, 0.1], [0.4, 0.1], [0.2, 0.5]],
                    }
                ],
                "uncertain_regions": [],
                "open_polygon": {
                    "certainty": "uncertain",
                    "polygon": [[0.6, 0.6], [0.8, 0.6]],
                },
            }
        },
    }

    with pytest.raises(ValueError, match="open polygon"):
        validate_return_session(session=session, packet=packet)


def test_blind_export_accepts_the_global_private_index_for_one_packet(
    tmp_path: Path,
) -> None:
    from cmc_bbdm.learned_cscan.human_review_tool import (
        build_blind_packets,
        export_blind_review_session,
    )

    layer = _blind_layer()
    row = {
        "report_id": "report-001",
        "specimen_key": "domain-a:sample-1",
        "task": "LOCATE",
        "method": "BC_S1",
        "seed": 1,
        "stop_system": "S_BC_CAL",
        "report_sha256": "d" * 64,
        "objective_success": True,
        "reference_version": "PROXY_LEGACY",
    }
    built = build_blind_packets(
        ((row, layer),),
        packet_size=1,
        shuffle_seed=20260909,
        source_base_sha="2" * 40,
        blinding_condition="FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1",
    )
    packet = built["packets"][0]
    item_id = packet["items"][0]["item_id"]
    global_index = {
        **built["private_index"],
        "unrelated-report": {**row, "report_id": "unrelated-report"},
    }
    session = {
        "schema_version": 1,
        "session_kind": "CSCAN_BLIND_REVIEW_SESSION",
        "packet_id": packet["packet_id"],
        "export_id": "one-packet",
        "reviewer_id": "reviewer-1",
        "blinding_condition": "FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1",
        "items": {
            item_id: {
                "state": "CONFIRMED",
                "decision": "DELIVERABLE",
            }
        },
    }

    result = export_blind_review_session(
        session=session,
        packet=packet,
        private_index=global_index,
        output_root=tmp_path / "converted",
    )

    assert result["confirmed_review_count"] == 1
