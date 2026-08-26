from __future__ import annotations

import csv
import re
from pathlib import Path

from cmc_bbdm.mavis import aei_paper_evidence

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
TITLE = (
    "From Useful to Actionable Information: A Task-Relevant Information "
    "Hierarchy for Ultrasonic Inspection of Impacted Composites"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_aei_paper_has_exactly_six_top_level_sections() -> None:
    sections = re.findall(r"^\\section\{([^}]+)\}", _text(MAIN), flags=re.MULTILINE)
    assert sections == SECTIONS


def test_aei_paper_outline_matches_fixed_section_contract() -> None:
    outline = _text(PAPER / "MANUSCRIPT_OUTLINE.md")
    headings = re.findall(r"^## [1-6]\. (.+)$", outline, flags=re.MULTILINE)
    assert headings == SECTIONS


def test_aei_paper_frontmatter_uses_fixed_title_and_review_safe_author() -> None:
    manuscript = _text(MAIN)
    assert f"\\title{{{TITLE}}}" in manuscript
    assert "\\begin{frontmatter}" in manuscript
    assert "\\end{frontmatter}" in manuscript
    assert "Author information withheld for review" in manuscript


def test_aei_paper_abstract_states_hierarchy_evidence_and_boundary() -> None:
    manuscript = _text(MAIN)
    abstract = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        manuscript,
        flags=re.DOTALL,
    )
    assert abstract is not None
    text = re.sub(r"\s+", " ", abstract.group(1))
    assert 150 <= len(text.split()) <= 250
    assert all(
        phrase in text
        for phrase in (
            "useful, observable, and actionable",
            "276 specimens",
            "six experimental domains",
            "32.1\\%",
            "89.9\\%",
            "70.4\\%",
            "did not outperform the strongest deployable baseline",
        )
    )


def test_aei_paper_introduction_and_conclusion_close_all_rqs() -> None:
    manuscript = _text(MAIN)
    introduction = manuscript.split("\\section{Introduction}", 1)[1].split(
        "\\section{Related Research and Problem Formulation}", 1
    )[0]
    conclusion = manuscript.split("\\section{Conclusions}", 1)[1].split(
        "\\bibliographystyle", 1
    )[0]
    assert introduction.lstrip().startswith("\\label{sec:introduction}")
    assert "Ultrasonic inspection produces spatially rich internal observations" in introduction
    assert all(f"RQ{index}" in introduction for index in (1, 2, 3))
    assert all(f"RQ{index}" in conclusion for index in (1, 2, 3))
    assert "Useful information was established" in conclusion
    assert "Actionable improvement was not established" in conclusion


def test_aei_paper_includes_submission_declarations_without_a_seventh_section() -> None:
    manuscript = _text(MAIN)
    assert "\\section*{Data and code availability}" in manuscript
    assert "anonymized reproducibility package" in manuscript
    assert (
        "\\section*{Declaration of generative AI and AI-assisted technologies "
        "in the writing process}" in manuscript
    )
    assert "OpenAI Codex" in manuscript


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


def test_aei_paper_uses_exactly_four_main_figures_and_two_main_tables() -> None:
    manuscript = _text(MAIN)
    figures = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", manuscript)
    tables = re.findall(r"\\input\{(tables/[^}]+)\}", manuscript)
    assert figures == [
        "figure1_information_hierarchy.pdf",
        "figure2_usefulness.pdf",
        "figure3_observability.pdf",
        "figure4_actionability.pdf",
    ]
    assert tables == [
        "tables/table1_case_protocol.tex",
        "tables/table2_hierarchy_evidence.tex",
    ]


def test_aei_paper_results_map_every_canonical_claim() -> None:
    manuscript = _text(MAIN)
    mapped = set(re.findall(r"\b[UOA][1-5]_[A-Z0-9_]+\b", manuscript))
    canonical = {
        row.claim_id for row in aei_paper_evidence.build_canonical_metrics(ROOT)
    }
    assert mapped == canonical


def test_aei_paper_keeps_central_adverse_controls_in_results() -> None:
    manuscript = _text(MAIN)
    required = (
        r"cross-objective support indicator of 0",
        r"Real minus positions was 0\.01740",
        r"real minus\s+reconstruction was 0\.03419",
        r"Dynamic real minus dynamic shuffled-content regret",
        r"feedback benefit was \$-1\.496\\times10\^\{-5\}\$",
        r"does not support superiority of the learned policy",
    )
    assert all(re.search(pattern, manuscript) for pattern in required)


def test_aei_paper_answers_each_research_question_directly() -> None:
    manuscript = _text(MAIN)
    assert manuscript.count("\\paragraph{Answer to RQ1.}") == 1
    assert manuscript.count("\\paragraph{Answer to RQ2.}") == 1
    assert manuscript.count("\\paragraph{Answer to RQ3.}") == 1


def test_aei_paper_headline_result_numbers_match_canonical_metrics() -> None:
    rows = {
        row.claim_id: row for row in aei_paper_evidence.build_canonical_metrics(ROOT)
    }
    manuscript = _text(MAIN)
    expected = (
        f"{rows['U1_MATCHED_FIELD'].estimate:.5f}",
        f"{100 * rows['U1_MATCHED_FIELD'].relative_effect:.1f}\\%",
        f"{rows['U2_SPARSE_FULL_GAP'].estimate:.5f}",
        f"{100 * rows['U2_SPARSE_RETENTION'].estimate:.1f}\\%",
        f"{rows['O1_STATIC_SPEARMAN'].estimate:.4f}",
        f"{100 * rows['O2_TEACHER_TURNOVER'].estimate:.1f}\\%",
        f"{rows['O3_REAL_MINUS_POSITIONS'].estimate:.5f}",
        f"{rows['O3_REAL_MINUS_RECONSTRUCTION'].estimate:.5f}",
        f"{rows['O4_DYNAMIC_MINUS_STATIC'].estimate:.6f}",
        f"{rows['A4_BASELINE_MINUS_MAVIS'].candidate_value:.6f}",
        f"{rows['A4_BASELINE_MINUS_MAVIS'].reference_value:.6f}",
    )
    assert all(number in manuscript for number in expected)
