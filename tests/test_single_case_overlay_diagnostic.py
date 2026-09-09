from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(
    os.environ.get(
        "CMC_BBDM_SOURCE_ROOT",
        str(ROOT.parents[2] / "paper3/cmc_damage_inference"),
    )
)
REQUIRED_PANELS = (
    "panel_00_base_registered_cscan.png",
    "panel_01_expert_gt_overlay.png",
    "panel_02_proxy_gt_overlay.png",
    "panel_03_bc_final_mask_overlay.png",
    "panel_04_bc_trajectory_and_mask_overlay.png",
    "panel_05_overlay_compare_expert_vs_bc.png",
    "panel_06_overlay_compare_expert_vs_proxy_vs_bc.png",
    "panel_07_overlay_boundary_only_compare.png",
    "figure_single_case_overlay_summary.png",
)


def _overlay_module():
    try:
        return importlib.import_module("cmc_bbdm.learned_cscan.single_case_overlay")
    except ModuleNotFoundError:
        pytest.fail("single-case overlay diagnostic module is missing")


def _write_selection_fixture(root: Path) -> None:
    tables = root / "tables"
    reviewed = root / "reviewed"
    tables.mkdir(parents=True)
    reviewed.mkdir()
    pl.DataFrame(
        {
            "specimen_key": ["domain:a", "domain:b", "domain:c"],
            "method": ["BC_S1", "BC_S2", "BC_S3"],
            "method_group": ["BC_FIXED_SEED"] * 3,
            "seed": [1, 2, 3],
            "selection_rule": ["PRESELECTED_WITHIN_DOMAIN"] * 3,
        }
    ).write_csv(tables / "fixed_case_process.csv")
    pl.DataFrame(
        {
            "dataset_id": ["domain"] * 3,
            "specimen_id": ["a", "b", "c"],
            "specimen_key": ["domain:a", "domain:b", "domain:c"],
            "task": ["LOCATE"] * 3,
            "method": ["BC_S1", "BC_S2", "BC_S3"],
            "seed": [1, 2, 3],
            "stop_system": ["S_BC_CAL"] * 3,
            "stopped": [True] * 3,
            "stop_step": [10, 20, 30],
            "stop_cost": [0.1, 0.2, 0.3],
            "stop_report_sha256": ["a" * 64, "b" * 64, "c" * 64],
        }
    ).write_csv(reviewed / "per_episode_metrics.csv")
    pl.DataFrame(
        {
            "specimen_key": ["domain:a", "domain:b", "domain:c"],
            "task": ["LOCATE"] * 3,
            "report_sha256": ["a" * 64, "x" * 64, "y" * 64],
            "formal_success": [False] * 3,
            "iou": [0.01, 0.20, 0.05],
        }
    ).write_csv(reviewed / "full_input_readout.csv")


def test_selection_prioritizes_distinct_report_identity_then_lowest_iou(
    tmp_path: Path,
) -> None:
    _write_selection_fixture(tmp_path)

    selected = _overlay_module().select_diagnostic_case(tmp_path)

    assert selected.specimen_key == "domain:c"
    assert selected.method == "BC_S3"
    assert selected.seed == 3
    assert selected.stop_step == 30
    assert selected.proxy_report_sha256 == "y" * 64
    assert "preselected fixed case" in selected.selected_reason
    assert "lowest EXPERT_GT-vs-PROXY_GT" in selected.selected_reason


def test_real_case_recovery_preserves_registered_pixels_and_first_stop() -> None:
    if not SOURCE_ROOT.is_dir():
        pytest.skip("registered source dataset is unavailable")

    bundle = _overlay_module().recover_selected_case(
        project_root=ROOT,
        source_root=SOURCE_ROOT,
    )

    assert bundle.selected.specimen_key == "ykhs7s2dck:q8-4"
    assert bundle.selected.task == "LOCATE"
    assert bundle.selected.method == "BC_S3"
    assert bundle.selected.seed == 3
    assert bundle.selected.stop_step == 153
    assert bundle.selected.stop_cost == pytest.approx(0.7840312122211232)
    assert bundle.coordinate_checks["annotation_reader_pixels_equal"] is True
    assert bundle.coordinate_checks["packet_source_pixels_equal"] is True
    assert bundle.coordinate_checks["orientation_relation"] == "IDENTICAL_NO_TRANSFORM"
    assert bundle.proxy_report_sha256 == (
        "382775ba836a551517567d8ef758b495dea576b7e76e8b6a1bbd42f9bf6dc063"
    )
    assert bundle.bc_report_sha256 == (
        "474027ce54631e59064ce1a1f3c2d6dc66920b487860ca31cf85bd336a711b38"
    )
    assert bundle.trajectory_rows[153]["calibrated_stop_trigger"] is True
    assert not any(
        row["calibrated_stop_trigger"] for row in bundle.trajectory_rows[:153]
    )
    assert bundle.identity_checks["first_stop_trigger_matches"] is True
    assert bundle.endpoint.action_count == 153
    assert bundle.endpoint.measured_cost == pytest.approx(bundle.selected.stop_cost)
    assert bundle.mask_metrics["proxy_bc_pixel_iou"] == pytest.approx(
        0.7337908982115237
    )


def test_export_writes_required_readable_panels_and_manifest(tmp_path: Path) -> None:
    if not SOURCE_ROOT.is_dir():
        pytest.skip("registered source dataset is unavailable")

    result = _overlay_module().export_single_case_overlay_diagnostic(
        project_root=ROOT,
        source_root=SOURCE_ROOT,
        output_root=tmp_path,
    )

    assert tuple(Path(path).name for path in result["overlay_files"]) == REQUIRED_PANELS
    for name in REQUIRED_PANELS:
        path = tmp_path / name
        with Image.open(path) as image:
            image.load()
            assert max(image.size) >= 1800
            assert max(ImageStat.Stat(image.convert("RGB")).var) > 1.0
    manifest_path = tmp_path / "single_case_overlay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "specimen_key",
        "dataset_id",
        "task",
        "method",
        "seed",
        "reference_version",
        "selected_reason",
        "source_cscan_path",
        "source_image_sha256",
        "expert_reference_json",
        "proxy_report_sha256",
        "bc_final_report_sha256",
        "stop_step",
        "image_size",
        "overlay_files",
        "notes",
    }
    assert required <= set(manifest)
    assert manifest["specimen_key"] == "ykhs7s2dck:q8-4"
    assert manifest["overlay_files"] == list(REQUIRED_PANELS)
    assert manifest["coordinate_checks"]["annotation_reader_pixels_equal"] is True
    assert manifest["identity_checks"]["first_stop_report_sha256_match"] is True
    assert manifest["resource_accounting"] == {
        "new_training": 0,
        "new_vlm_calls": 0,
        "actor_forward_calls": 0,
        "stop_forward_calls": 0,
        "world_step_count": 0,
    }


def test_export_cli_exposes_source_and_output_roots() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_single_case_overlay_diagnostic.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--source-root" in result.stdout
    assert "--output-root" in result.stdout
