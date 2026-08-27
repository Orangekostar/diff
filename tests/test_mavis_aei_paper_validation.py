from __future__ import annotations

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
    assert report.mapped_claim_count == 39
    assert report.figure_count == 4
    assert report.table_count == 3
    assert report.section_count == 6
    assert report.semantic_errors == ()


def test_aei_supplement_preserves_main_boundary_results() -> None:
    manuscript = (ROOT / "paper_aei_information_hierarchy/main.tex").read_text(
        encoding="utf-8"
    )
    assert "real minus acquired-position/history" in manuscript
    assert "dynamic real minus shuffled" in manuscript
    assert "no-feedback reference retained" in manuscript
    assert "not performance-superior" in manuscript


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
