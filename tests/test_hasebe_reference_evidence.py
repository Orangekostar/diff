from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook

from cmc_bbdm.learned_cscan.hasebe_reference_evidence import (
    AUTHOR_SCALAR_MEASUREMENT,
    AUTHOR_SPATIAL_REFERENCE,
    AUTHOR_ULTRASOUND_MEASUREMENT,
    LOCAL_DERIVED_PROXY,
    MECHANICAL_ENDPOINT_ONLY,
    build_reference_comparison,
    build_test24_crosswalk,
    inspect_workbook,
    provenance_status,
    run_compare,
    run_extract,
    run_inventory,
    summarize_crosswalk,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(
    os.environ.get(
        "CMC_BBDM_SOURCE_ROOT",
        str(ROOT.parents[2] / "paper3/cmc_damage_inference"),
    )
)


@pytest.fixture(scope="module")
def real_extraction(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, object], dict[str, int]]:
    if not SOURCE_ROOT.is_dir():
        pytest.skip("bound Hasebe source root is unavailable")
    output = tmp_path_factory.mktemp("hasebe-reference") / "results"
    inventory = run_inventory(ROOT, SOURCE_ROOT, output)
    extracted = run_extract(ROOT, SOURCE_ROOT, output)
    return output, inventory, extracted


@pytest.fixture(scope="module")
def real_comparison(
    tmp_path_factory: pytest.TempPathFactory,
    real_extraction: tuple[Path, dict[str, object], dict[str, int]],
) -> tuple[Path, Path, dict[str, int]]:
    output, _inventory, _extracted = real_extraction
    artifacts = tmp_path_factory.mktemp("hasebe-reference") / "artifacts"
    compared = run_compare(ROOT, SOURCE_ROOT, output, artifacts)
    return output, artifacts, compared


def test_workbook_inspection_preserves_merged_headers_and_missing_formula_cache(
    tmp_path: Path,
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "measurements"
    sheet.merge_cells("A1:B1")
    sheet["A1"] = "Damage"
    sheet["A2"] = "Area"
    sheet["B2"] = "Length"
    sheet["A3"] = 12.5
    sheet["B3"] = "=A3*2"
    path = tmp_path / "fixture.xlsx"
    workbook.save(path)

    audit = inspect_workbook(path)

    assert audit["sheet_count"] == 1
    inspected = audit["sheets"][0]
    assert inspected["merged_ranges"] == ["A1:B1"]
    assert inspected["comment_count"] == 0
    assert inspected["image_count"] == 0
    assert inspected["chart_count"] == 0
    cells = {row["address"]: row for row in inspected["cells"]}
    assert cells["A1"]["value_raw"] == "Damage"
    assert "B1" not in cells
    assert cells["B3"]["formula"] == "=A3*2"
    assert cells["B3"]["cached_value"] is None
    assert cells["B3"]["cache_status"] == "FORMULA_NO_CACHED_VALUE"


def test_provenance_requires_source_and_derivation_not_friendly_column_name() -> None:
    assert (
        provenance_status(
            source_kind="AUTHOR_WORKBOOK",
            extraction_method="DIRECT_CELL_READ",
            measurement_semantics="PROJECTED_DELAMINATION_AREA",
        )
        == AUTHOR_SCALAR_MEASUREMENT
    )
    assert (
        provenance_status(
            source_kind="AUTHOR_CSCAN_SCREENSHOT",
            extraction_method="FILE_IDENTITY_ONLY",
            measurement_semantics="AMPLITUDE_CODED_CSCAN_SCREENSHOT",
        )
        == AUTHOR_ULTRASOUND_MEASUREMENT
    )
    assert (
        provenance_status(
            source_kind="physical_descriptors.csv",
            extraction_method="RGB_THRESHOLD_MORPHOLOGY",
            measurement_semantics="projected_damage_area",
        )
        == LOCAL_DERIVED_PROXY
    )
    assert (
        provenance_status(
            source_kind="AUTHOR_WORKBOOK",
            extraction_method="DIRECT_CELL_READ",
            measurement_semantics="CAI_STRENGTH",
        )
        == MECHANICAL_ENDPOINT_ONLY
    )


def test_crosswalk_uses_exact_dataset_specimen_and_stage_identity() -> None:
    reviews = [
        {"dataset_id": "d-q8", "specimen_id": "q8-4", "specimen_key": "d-q8:q8-4"},
        {"dataset_id": "d-c8", "specimen_id": "c8-4", "specimen_key": "d-c8:c8-4"},
        {"dataset_id": "d-c8", "specimen_id": "c8-4t", "specimen_key": "d-c8:c8-4t"},
    ]
    measurements = [
        {
            "dataset_id": "d-q8",
            "specimen_key": "d-q8:q8-4",
            "measurement_stage": "POST_LVI_PRE_CAI",
            "measurement_semantics": "PROJECTED_DELAMINATION_AREA",
        },
        {
            "dataset_id": "d-c8",
            "specimen_key": "d-c8:c8-4",
            "measurement_stage": "POST_CAI",
            "measurement_semantics": "PROJECTED_DELAMINATION_AREA",
        },
        {
            "dataset_id": "wrong-domain",
            "specimen_key": "wrong-domain:q8-4",
            "measurement_stage": "POST_LVI_PRE_CAI",
            "measurement_semantics": "PROJECTED_DELAMINATION_AREA",
        },
    ]
    scans = [
        {
            "dataset_id": "d-q8",
            "specimen_id": "q8-4",
            "measurement_stage": "POST_LVI_PRE_CAI",
            "cscan_source_path": "q8-4.jpg",
        },
        {
            "dataset_id": "d-c8",
            "specimen_id": "c8-4t",
            "measurement_stage": "POST_LVI_PRE_CAI",
            "cscan_source_path": "c8-4t.jpg",
        },
    ]

    rows = build_test24_crosswalk(reviews, measurements, scans)

    by_key = {row["specimen_key"]: row for row in rows}
    assert by_key["d-q8:q8-4"]["author_damage_scalar_status"] == "MATCHED"
    assert by_key["d-q8:q8-4"]["author_cscan_status"] == "MATCHED"
    assert by_key["d-c8:c8-4"]["author_damage_scalar_status"] == "STAGE_MISMATCH"
    assert by_key["d-c8:c8-4"]["author_cscan_status"] == "MISSING"
    assert by_key["d-c8:c8-4t"]["author_damage_scalar_status"] == "MISSING"
    assert by_key["d-c8:c8-4t"]["author_cscan_status"] == "MATCHED"


def test_crosswalk_summary_counts_unique_test_specimens_without_inflation() -> None:
    reviews = [
        {"dataset_id": "d", "specimen_id": f"s-{index}", "specimen_key": f"d:s-{index}"}
        for index in range(24)
    ]
    measurements = [
        {
            "dataset_id": "d",
            "specimen_key": f"d:s-{index}",
            "measurement_stage": "POST_LVI_PRE_CAI",
            "measurement_semantics": semantics,
        }
        for index in range(20)
        for semantics in ("PROJECTED_DELAMINATION_AREA", "DENT_DEPTH")
    ]

    rows = build_test24_crosswalk(reviews, measurements, [])
    summary = summarize_crosswalk(rows)

    assert len(rows) == 24
    assert summary == {
        "physical_test_n": 24,
        "spatial_reference_n": 0,
        "scalar_reference_n": 20,
        "author_cscan_screenshot_n": 0,
        "quantitative_ultrasound_n": 0,
        "missing_or_ambiguous_n": 24,
    }
    assert rows[-1]["author_damage_scalar_status"] == "MISSING"
    assert rows[-1]["author_damage_scalar_reason"] == "NO_MATCHING_AUTHOR_DAMAGE_SCALAR"


def test_scalar_comparison_never_invents_physical_values_or_equates_lengths() -> None:
    expert = np.zeros((4, 4), dtype=np.bool_)
    expert[:2, :2] = True
    reader = np.zeros((4, 4), dtype=np.bool_)
    reader[1:3, 1:3] = True

    unresolved = build_reference_comparison(
        specimen_key="d:s-1",
        task_or_target="CHARACTERIZE",
        measurement_stage="POST_LVI_PRE_CAI",
        reference_record_id="r1",
        reference_kind=AUTHOR_SCALAR_MEASUREMENT,
        quantity="PROJECTED_DELAMINATION_AREA",
        unit="mm^2",
        author_value=12.0,
        expert_mask=expert,
        reader_mask=reader,
        scale_evidence=None,
        mask_source="fixture",
    )
    assert unresolved["comparison_status"] == "CALIBRATION_UNRESOLVED"
    assert unresolved["expert_value"] is None
    assert unresolved["reader_value"] is None
    assert unresolved["absolute_difference"] is None

    incomparable = build_reference_comparison(
        specimen_key="d:s-1",
        task_or_target="LOCATE",
        measurement_stage="POST_LVI_PRE_CAI",
        reference_record_id="r2",
        reference_kind=AUTHOR_SCALAR_MEASUREMENT,
        quantity="DELAMINATION_LENGTH",
        unit="mm",
        author_value=4.0,
        expert_mask=expert,
        reader_mask=reader,
        scale_evidence={"width_mm": 4.0, "height_mm": 4.0, "basis": "fixture"},
        mask_source="fixture",
    )
    assert incomparable["comparison_status"] == "INCOMPARABLE_QUANTITY"
    assert incomparable["expert_value"] is None
    assert incomparable["reader_value"] is None


def test_summarize_cli_rebuilds_from_extracted_files_without_source_root(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "results"
    artifact_root = tmp_path / "artifacts"
    result_root.mkdir()
    with (result_root / "test24_reference_crosswalk.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "specimen_key",
                "author_spatial_reference_status",
                "author_damage_scalar_status",
                "author_cscan_status",
                "quantitative_ultrasound_status",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "specimen_key": "d:s-1",
                "author_spatial_reference_status": "MISSING",
                "author_damage_scalar_status": "MATCHED",
                "author_cscan_status": "MATCHED",
                "quantitative_ultrasound_status": "MISSING",
            }
        )
    (result_root / "decision.json").write_text(
        json.dumps(
            {
                "implementation_status": "COMPLETE_WITH_EVIDENCE_LIMITS",
                "recommended_next_step": "REQUEST_AUTHOR_SPATIAL_REFERENCE",
                "not_established": ["UNIQUE_OPTIMAL_SCAN_ROUTE"],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/trace_hasebe_reference_evidence.py"),
            "summarize",
            "--output-root",
            str(result_root),
            "--artifact-root",
            str(artifact_root),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--source-root" not in completed.args
    for name in (
        "SOURCE_EVIDENCE_LEDGER.md",
        "SOURCE_BINDINGS_AND_FIELD_LINEAGE.md",
        "REFERENCE_FEASIBILITY_AND_NEXT_STEP.md",
        "AUTHOR_DATA_REQUEST_DRAFT.md",
        "CODEX_HANDOFF_HASEBE_REFERENCE_EVIDENCE.md",
    ):
        assert (artifact_root / name).is_file()
    payload = json.loads((artifact_root / "summary_counts.json").read_text())
    assert payload["physical_test_n"] == 1
    assert payload["scalar_reference_n"] == 1


def test_real_extraction_traces_workbook_cells_to_exact_test24(
    real_extraction: tuple[Path, dict[str, object], dict[str, int]],
) -> None:
    output, inventory, extracted = real_extraction

    assert inventory["bound_workbook_count"] == 9
    assert inventory["missing_required_source_count"] == 0
    assert extracted["physical_test_n"] == 24
    assert extracted["scalar_reference_n"] == 24
    assert extracted["author_cscan_screenshot_n"] == 24
    assert extracted["spatial_reference_n"] == 0
    assert extracted["quantitative_ultrasound_n"] == 0
    fields = list(csv.DictReader((output / "workbook_fields.csv").open()))
    assert {row["source_sheet"] for row in fields} >= {
        "condition",
        "LVI condition",
        "Specimen size",
        "CAI strength",
    }
    measurements = list(csv.DictReader((output / "author_measurements.csv").open()))
    area = {
        row["specimen_key"]: row
        for row in measurements
        if row["measurement_semantics"] == "PROJECTED_DELAMINATION_AREA"
    }
    assert area["74t7kcdgkr:c8-24"]["value_numeric"] == "49.7"
    assert area["74t7kcdgkr:c8-24"]["source_cell"] == "G64"
    assert area["74t7kcdgkr:c8-24"]["dataset_version"] == "1"
    assert area["74t7kcdgkr:c8-24"]["source_dataset_id"] == "8scdmfdcfb"
    assert area["74t7kcdgkr:c8-24"]["source_dataset_version"] == "3"
    assert area["ykhs7s2dck:q8-4"]["value_numeric"] == "55.2"
    assert area["ykhs7s2dck:q8-4"]["source_cell"] == "G116"
    assert area["cgtnjyggtm:q24-40"]["value_numeric"] == "449.9"
    assert area["cgtnjyggtm:q24-40"]["source_cell"] == "G269"
    assert all(
        row["provenance_status"] != AUTHOR_SPATIAL_REFERENCE for row in measurements
    )
    crosswalk = list(csv.DictReader((output / "test24_reference_crosswalk.csv").open()))
    assert all(
        row["identity_status"] == "PASS_EXACT_SPECIMEN_ID_AND_HASH" for row in crosswalk
    )
    assert all(
        row["pairing_method"] == "exact_published_specimen_id" for row in crosswalk
    )


def test_real_compare_is_bounded_to_fixed_cases_and_saved_reader_identity(
    real_comparison: tuple[Path, Path, dict[str, int]],
) -> None:
    output, artifacts, compared = real_comparison

    assert compared["case_review_n"] == 3
    assert compared["full_input_report_recovery_n"] == 6
    assert compared["new_training"] == 0
    cases = list(csv.DictReader((output / "case_review.csv").open()))
    assert {row["specimen_key"] for row in cases} == {
        "ykhs7s2dck:q8-4",
        "74t7kcdgkr:c8-24",
        "cgtnjyggtm:q24-40",
    }
    assert all(row["crop_verification_status"] == "PIXEL_IDENTICAL" for row in cases)
    comparisons = list(csv.DictReader((output / "reference_comparison.csv").open()))
    assert len(comparisons) == 6
    area = [
        row for row in comparisons if row["quantity"] == "PROJECTED_DELAMINATION_AREA"
    ]
    assert len(area) == 3
    assert all(
        row["comparison_status"] == "COMPARABLE_WITH_SCOPE_LIMIT" for row in area
    )
    assert {row["reader_report_sha256"] for row in area} == {
        "d3d30bb3f23d6806310ed10401fc93beae485e2b8dcabd2398fb9f8909d4dd33",
        "af558c3613e5de168494c0c5f94f4047dabbe96fed1498dd2bc055d6fbce1680",
        "6a24e173e5aa33c33cccbbbc6680648c27b1878eff48cd089d1a98e714b80b27",
    }
    figures = sorted((artifacts / "figures").glob("*.png"))
    assert len(figures) == 3
    for figure in figures:
        from PIL import Image

        with Image.open(figure) as image:
            image.load()
            assert image.size == (891, 891)
