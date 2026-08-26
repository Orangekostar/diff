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
    assert report.table_count == 2
    assert report.section_count == 6


def test_aei_supplement_preserves_main_boundary_results() -> None:
    manuscript = (ROOT / "paper_aei_information_hierarchy/main.tex").read_text(
        encoding="utf-8"
    )
    assert "Real minus positions was 0.01740" in manuscript
    assert "Dynamic real minus dynamic shuffled-content regret" in manuscript
    assert "feedback benefit was $-1.496\\times10^{-5}$" in manuscript
    assert "does not support superiority of the learned policy" in manuscript
