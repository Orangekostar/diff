from __future__ import annotations

import re
from pathlib import Path

from cmc_bbdm.mavis import aei_paper_validation

ROOT = Path(__file__).resolve().parents[1]


def test_aei_paper_results_numbers_have_canonical_provenance() -> None:
    assert aei_paper_validation.unmatched_results_numbers(ROOT) == []


def test_aei_paper_frozen_evidence_paths_are_unchanged() -> None:
    assert aei_paper_validation.changed_frozen_paths(ROOT) == []


def test_aei_paper_validation_closes_claim_figure_table_contract() -> None:
    report = aei_paper_validation.validate_paper(ROOT)
    assert report.passed
    assert report.canonical_claim_count == 39
    assert report.main_visible_claim_count == 27
    assert report.main_mapped_claim_count == 27
    assert report.combined_mapped_claim_count == 39
    assert report.figure_count == 4
    assert report.table_count == 1
    assert report.section_count == 6
    assert report.semantic_errors == ()


def test_aei_supplement_preserves_main_boundary_results() -> None:
    manuscript = re.sub(
        r"\s+",
        " ",
        (ROOT / "paper_aei_information_hierarchy/main.tex").read_text(encoding="utf-8"),
    )
    supplement = re.sub(
        r"\s+",
        " ",
        (
            ROOT / "paper_aei_information_hierarchy/supplementary/supplementary.tex"
        ).read_text(encoding="utf-8"),
    )
    assert re.search(
        r"real minus acquired-\s*position/history", manuscript, re.IGNORECASE
    )
    assert "dynamic real minus shuffled" in manuscript.lower()
    assert "no-feedback reference retained" in supplement
    assert "A3_FEEDBACK_BENEFIT" in supplement
    assert "A3_FEEDBACK_BENEFIT" not in manuscript
    assert "A4_BASELINE_MINUS_MAVIS" not in manuscript


def test_aei_main_and_supplement_follow_visibility_partition() -> None:
    report = aei_paper_validation.validate_paper(ROOT)
    assert report.main_mapped_claim_count == report.main_visible_claim_count
    assert report.combined_mapped_claim_count == report.canonical_claim_count


def test_adverse_numerical_directions_are_preserved() -> None:
    rows = {
        row.claim_id: row
        for row in aei_paper_validation.aei_paper_evidence.build_canonical_metrics(ROOT)
    }
    assert rows["O3_REAL_MINUS_POSITIONS"].estimate > 0
    assert rows["O3_REAL_MINUS_RECONSTRUCTION"].estimate > 0
    assert rows["O4_DYNAMIC_MINUS_SHUFFLED"].estimate > 0
    assert rows["A3_FEEDBACK_BENEFIT"].estimate < 0
    final = rows["A4_BASELINE_MINUS_MAVIS"]
    assert final.reference_value < final.candidate_value
    assert final.estimate < 0
