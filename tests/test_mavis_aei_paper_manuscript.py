from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_aei_information_hierarchy"
MAIN = PAPER / "main.tex"

SECTIONS = [
    "Introduction",
    "Related Research and Problem Formulation",
    "Task-Relevant Information Hierarchy and Operational Framework",
    "Multi-Domain CFRP Case Study and Experimental Design",
    "Experimental Results and Discussion",
    "Conclusions",
]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_aei_paper_has_exactly_six_top_level_sections() -> None:
    sections = re.findall(r"^\\section\{([^}]+)\}", _text(MAIN), flags=re.MULTILINE)
    assert sections == SECTIONS


def test_aei_paper_outline_matches_fixed_section_contract() -> None:
    outline = _text(PAPER / "MANUSCRIPT_OUTLINE.md")
    headings = re.findall(r"^## [1-6]\. (.+)$", outline, flags=re.MULTILINE)
    assert headings == SECTIONS


def test_aei_paper_citation_keys_are_defined() -> None:
    manuscript = _text(MAIN)
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", manuscript)
        for key in group.split(",")
    }
    defined = set(
        re.findall(
            r"^@[A-Za-z]+\{([^,]+),",
            _text(PAPER / "references.bib"),
            flags=re.MULTILINE,
        )
    )
    assert cited
    assert cited <= defined


def test_aei_paper_claim_sentence_bank_references_canonical_claims() -> None:
    with (ROOT / "artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        canonical = {row["claim_id"] for row in csv.DictReader(stream)}
    referenced = set(
        re.findall(
            r"\b[UOA][1-5]_[A-Z0-9_]+\b",
            _text(PAPER / "CLAIM_SENTENCE_BANK.md"),
        )
    )
    assert referenced
    assert referenced <= canonical


def test_aei_paper_value_is_predictor_conditioned() -> None:
    manuscript = _text(MAIN)
    assert "downstream-predictor-conditioned task value" in manuscript
    assert "U_f(X\\mid\\legal_{i,t})" in manuscript
    assert re.search(
        r"not an intrinsic or\s+universal mechanical property", manuscript
    )


def test_aei_paper_excludes_internal_stage_labels() -> None:
    manuscript = _text(MAIN)
    forbidden = (
        "M0 GO",
        "M1 NO-GO",
        "M0_GO",
        "M1_NO_GO",
        "Tier B",
        "CLAIM_NARROWING_GO",
        "METHOD_EXTENSION_NO_GO",
        "GO_NOGO",
    )
    assert not any(phrase in manuscript for phrase in forbidden)
    assert not re.search(r"\bP(?:[0-9]|1[0-6])\b", manuscript)


def test_aei_paper_no_forbidden_claim_phrases() -> None:
    manuscript = _text(MAIN).lower()
    forbidden = (
        "mavis outperforms the strongest deployable baseline",
        "mavis improves most held-out domains",
        "mris successfully captures specimen-specific mechanical state",
        "real partial ultrasound is more informative than positions",
        "feedback improves acquisition",
        "mechanical value is intrinsic to a location",
        "universal mechanical-value map",
        "first adaptive ultrasonic inspection",
        "first sequential ultrasonic inspection",
        "first ultrasound voi",
        "75% scanner-time reduction",
        "industrial deployment",
        "external generalization",
    )
    assert not any(phrase in manuscript for phrase in forbidden)


def test_aei_paper_keeps_oracles_and_cost_semantics_bounded() -> None:
    manuscript = _text(MAIN)
    assert "retrospective" in manuscript
    assert "non-deployable" in manuscript
    assert "No result is translated into physical scanner time" in manuscript
    assert re.search(
        r"repeated\s+computational records, not independent samples", manuscript
    )
