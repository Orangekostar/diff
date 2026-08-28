from __future__ import annotations

import csv
import re
from pathlib import Path

from cmc_bbdm.mavis import aei_paper_evidence

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_aei_information_hierarchy"
ARTIFACTS = ROOT / "artifacts/aei_information_hierarchy"
MAIN = PAPER / "main.tex"
SUPPLEMENT = PAPER / "supplementary/supplementary.tex"

SECTIONS = [
    "Introduction",
    "Related Work",
    "Task-Relevant Information Acquisition Framework",
    "Multi-Domain CFRP Experimental Design",
    "Experimental Results and Discussion",
    "Conclusions",
]
TITLE = (
    "Task-Relevant Ultrasonic Information Acquisition for Impacted Composites: "
    "From Spatial Information to State-Conditioned Sensing"
)
RELATED_SUBSECTIONS = [
    "Post-impact ultrasonic information and residual-capacity assessment",
    "Sparse and adaptive ultrasonic acquisition",
    "Task-relevant information acquisition formulation",
]
METHOD_SUBSECTIONS = [
    "Task-Relevant Information Characterization",
    "State-Conditioned Measurement Valuation",
    "Cost-Constrained Task-Oriented Acquisition",
]
EXPERIMENT_SUBSECTIONS = [
    "Dataset and Information Representations",
    "Causal Acquisition Protocol",
    "Held-Out-Domain Evaluation and Statistical Analysis",
]
PART1_STAGES = [
    "Spatial information and sparse recoverability",
    "Task-conditioned spatial measurement value",
    "State- and predictor-conditioned measurement value",
]
PART2_STAGES = [
    "State-conditioned valuation improves next-action estimation",
    "Information-source and component decomposition",
    "Cost-constrained set realization",
]
FIGURES = [
    "figure1_task_relevant_acquisition_framework.pdf",
    "figure2_information_characterization.pdf",
    "figure3_state_conditioned_value.pdf",
    "figure4_valuation_planning_realization.pdf",
]
TABLES = [
    "tables/table1_case_protocol.tex",
    "tables/table2_task_relevant_results.tex",
]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _section(manuscript: str, heading: str, next_heading: str | None) -> str:
    section = manuscript.split(f"\\section{{{heading}}}", 1)[1]
    if next_heading is not None:
        section = section.split(f"\\section{{{next_heading}}}", 1)[0]
    return section


def _abstract(manuscript: str) -> str:
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", manuscript, flags=re.DOTALL
    )
    assert match is not None
    return re.sub(r"\s+", " ", match.group(1)).strip()


def test_aei_paper_has_exactly_six_top_level_sections() -> None:
    sections = re.findall(r"^\\section\{([^}]+)\}", _text(MAIN), flags=re.MULTILINE)
    assert sections == SECTIONS


def test_aei_paper_outline_matches_compressed_section_contract() -> None:
    outline = _text(PAPER / "MANUSCRIPT_OUTLINE.md")
    headings = re.findall(r"^## [1-6]\. (.+)$", outline, flags=re.MULTILINE)
    assert headings == SECTIONS


def test_aei_paper_frontmatter_uses_method_identity_title() -> None:
    manuscript = _text(MAIN)
    assert f"\\title{{{TITLE}}}" in manuscript
    assert "\\begin{frontmatter}" in manuscript
    assert "\\end{frontmatter}" in manuscript
    assert "Author information withheld for review" in manuscript


def test_aei_paper_abstract_has_positive_framework_evidence() -> None:
    abstract = _abstract(_text(MAIN))
    assert 150 <= len(abstract.split()) <= 250
    required = (
        "Task-Relevant Information Acquisition",
        "276 specimens",
        "six experimental domains",
        "32.1\\%",
        "89.9\\%",
        "70.4\\%",
        "state-conditioned valuation",
        "five of six held-out domains",
        "task-oriented",
        "cost-constrained sensing",
    )
    forbidden = (
        "MAVIS",
        "0.125053",
        "0.124992",
        "6.114",
        "2/6",
        "residual deployable gap",
        "not performance-superior",
        "deployment readiness",
        "frozen endpoint",
        "post-freeze",
        "hash-bound",
    )
    assert all(phrase in abstract for phrase in required)
    assert not any(phrase in abstract for phrase in forbidden)
    assert not abstract.endswith("comparison.")


def test_aei_paper_introduction_has_six_positive_paragraphs() -> None:
    manuscript = _text(MAIN)
    introduction = _section(manuscript, "Introduction", "Related Work")
    paragraphs = [
        block.strip()
        for block in introduction.split("\n\n")
        if block.strip() and not block.lstrip().startswith("\\label")
    ]
    assert len(paragraphs) == 6
    assert "Existing work does not yet provide an integrated account" in introduction
    assert "RQ-A" in introduction and "RQ-B" in introduction
    assert "276" in introduction and "six" in introduction
    assert "nested leave-one-domain-out" in introduction
    assert "specimen-first" in introduction and "equal-domain" in introduction
    assert "threefold" in introduction
    forbidden = (
        "MAVIS",
        "residual deployable gap",
        "post-freeze",
        "hash-bound",
        "not performance-superior",
        "frozen outer endpoint",
    )
    assert not any(phrase in introduction for phrase in forbidden)


def test_aei_paper_related_work_has_exactly_three_forward_subsections() -> None:
    manuscript = _text(MAIN)
    related = _section(
        manuscript,
        "Related Work",
        "Task-Relevant Information Acquisition Framework",
    )
    headings = re.findall(r"^\\subsection\{([^}]+)\}", related, flags=re.MULTILINE)
    assert headings == RELATED_SUBSECTIONS
    assert "table1_closest_work" not in related
    defensive = (
        "the present contribution is not",
        "neither is a priority claim",
        "this paper does not present",
        "not a new generic",
        "not the novelty",
    )
    assert not any(phrase in related.lower() for phrase in defensive)


def test_aei_paper_method_has_exactly_three_subsections() -> None:
    manuscript = _text(MAIN)
    method = _section(
        manuscript,
        "Task-Relevant Information Acquisition Framework",
        "Multi-Domain CFRP Experimental Design",
    )
    headings = re.findall(r"^\\subsection\{([^}]+)\}", method, flags=re.MULTILINE)
    assert headings == METHOD_SUBSECTIONS
    assert r"U_f(X\mid\legal_{i,t})" in method
    assert r"\widehat U" in method
    assert r"\pi(\legal_{i,t})" in method
    assert "acquired-position/history control" in method
    assert "pure geometry-only" not in method
    assert "MAVIS" not in method


def test_aei_paper_experimental_design_has_exactly_three_subsections() -> None:
    manuscript = _text(MAIN)
    design = _section(
        manuscript,
        "Multi-Domain CFRP Experimental Design",
        "Experimental Results and Discussion",
    )
    headings = re.findall(r"^\\subsection\{([^}]+)\}", design, flags=re.MULTILINE)
    assert headings == EXPERIMENT_SUBSECTIONS
    for phrase in (
        "276",
        "45, 49, 43, 59, 42, and 38",
        "25\\%",
        "8$\\times$8",
        "nested leave-one-domain-out",
        "legal partial state",
        "native-raster",
        "physical specimen",
        "equal-domain",
    ):
        assert phrase in design


def test_aei_paper_results_use_required_three_plus_three_order() -> None:
    manuscript = _text(MAIN)
    results = _section(manuscript, "Experimental Results and Discussion", "Conclusions")
    subsections = re.findall(r"^\\subsection\{([^}]+)\}", results, flags=re.MULTILINE)
    assert subsections == [
        "From Spatial Information to State-Conditioned Task Value",
        "State-Conditioned Task-Oriented Acquisition",
        "Engineering Interpretation",
    ]
    part1, remainder = results.split(
        "\\subsection{State-Conditioned Task-Oriented Acquisition}", 1
    )
    part2 = remainder.split("\\subsection{Engineering Interpretation}", 1)[0]
    assert re.findall(
        r"^\\subsubsection\{([^}]+)\}", part1, flags=re.MULTILINE
    ) == PART1_STAGES
    assert re.findall(
        r"^\\subsubsection\{([^}]+)\}", part2, flags=re.MULTILINE
    ) == PART2_STAGES


def test_part_ii_opens_with_dynamic_over_static_positive_result() -> None:
    manuscript = _text(MAIN)
    part2 = manuscript.split(
        "\\subsection{State-Conditioned Task-Oriented Acquisition}", 1
    )[1].split("\\subsection{Engineering Interpretation}", 1)[0]
    assert part2.index("O4_DYNAMIC_MINUS_STATIC") < part2.index("O1_STATIC_SPEARMAN")
    flat = re.sub(r"\s+", " ", part2)
    assert "dynamic real minus static was $-0.001260$" in flat
    assert "five of six held-out domains" in flat


def test_main_manuscript_has_four_figures_and_two_tables() -> None:
    manuscript = _text(MAIN)
    figures = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", manuscript)
    tables = re.findall(r"\\input\{(tables/[^}]+)\}", manuscript)
    assert figures == FIGURES
    assert tables == TABLES
    assert "table1_closest_work" not in manuscript


def test_main_claim_ids_follow_visibility_authority() -> None:
    manuscript = _text(MAIN)
    mapped = set(re.findall(r"\b[UOA][1-5]_[A-Z0-9_]+\b", manuscript))
    visibility = _csv_rows(ARTIFACTS / "PAPER_CLAIM_VISIBILITY_MAP.csv")
    required = {
        row["claim_id"]
        for row in visibility
        if row["visibility"] in {"MAIN_HEADLINE", "MAIN_SUPPORT"}
    }
    supplement_only = {
        row["claim_id"]
        for row in visibility
        if row["visibility"] == "SUPPLEMENT_ONLY"
    }
    assert mapped == required
    assert not mapped & supplement_only


def test_a4_is_one_main_system_diagnostic_and_a3_is_supplement_only() -> None:
    manuscript = _text(MAIN)
    results = _section(manuscript, "Experimental Results and Discussion", "Conclusions")
    cost_realization = results.split(
        "\\subsubsection{Cost-constrained set realization}", 1
    )[1].split("\\subsection{Engineering Interpretation}", 1)[0]
    assert manuscript.count("A final system-level diagnostic examined") == 1
    assert cost_realization.count("0.125053") == 1
    assert cost_realization.count("0.124992") == 1
    assert "A3_FEEDBACK_BENEFIT" not in manuscript
    assert "1.496\\times10^{-5}" not in manuscript
    assert "no-feedback" not in manuscript.lower()
    supplement = _text(SUPPLEMENT)
    assert "A3_FEEDBACK_BENEFIT" in supplement
    assert "A4_BASELINE_MINUS_MAVIS" in supplement


def test_supplement_uses_s1_to_s6_and_covers_visibility_authority() -> None:
    supplement = _text(SUPPLEMENT)
    sections = re.findall(
        r"^\\section\{(S[1-6]\. [^}]+)\}", supplement, flags=re.MULTILINE
    )
    assert sections == [
        "S1. Evidence authority and statistical units",
        "S2. Spatial information and sparse recoverability",
        "S3. Task-conditioned and predictor-conditioned value",
        "S4. State-conditioned valuation and source controls",
        "S5. Cost-constrained realization and implementation diagnostics",
        "S6. Provenance, chronology, and scope boundaries",
    ]
    visibility = _csv_rows(ARTIFACTS / "PAPER_CLAIM_VISIBILITY_MAP.csv")
    supplement_only = {
        row["claim_id"]
        for row in visibility
        if row["visibility"] == "SUPPLEMENT_ONLY"
    }
    assert len(supplement_only) == 11
    assert all(claim_id in supplement for claim_id in supplement_only)
    for phrase in (
        "-1.496\\times10^{-5}",
        "-6.114\\times10^{-5}",
        "-8.461\\times10^{-5}",
        "-3.777\\times10^{-5}",
        "two of six held-out domains",
    ):
        assert phrase in supplement


def test_main_manuscript_uses_zero_mavis_identity_occurrences() -> None:
    manuscript = _text(MAIN)
    assert "MAVIS" not in manuscript
    assert "mvd_m1_o2" not in manuscript
    supplement = _text(SUPPLEMENT)
    assert (
        "The codebase refers to the supervised state-conditioned closed-loop "
        "implementation"
    ) in re.sub(r"\s+", " ", supplement)
    assert "MAVIS" in supplement
    assert "mvd_m1_o2" in supplement


def test_aei_paper_conclusion_has_three_positive_paragraphs() -> None:
    manuscript = _text(MAIN)
    conclusion = _section(manuscript, "Conclusions", None).split(
        "\\section*{Data and code availability}", 1
    )[0]
    paragraphs = [
        block.strip()
        for block in conclusion.split("\n\n")
        if block.strip() and not block.lstrip().startswith("\\label")
    ]
    assert len(paragraphs) == 3
    forbidden = (
        "MAVIS",
        "0.125053",
        "0.124992",
        "6.114",
        "not performance-superior",
        "residual deployable gap",
        "no-feedback",
        "2/6",
    )
    assert not any(phrase in conclusion for phrase in forbidden)
    assert conclusion.rstrip().endswith(
        "The proposed framework shifts ultrasonic inspection from measuring the\n"
        "complete field by default toward measuring the information that matters for\n"
        "the downstream engineering task."
    )


def test_aei_paper_includes_submission_declarations_without_extra_sections() -> None:
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


def test_aei_paper_claim_sentence_bank_references_only_canonical_claims() -> None:
    canonical = {
        row["claim_id"]
        for row in _csv_rows(ARTIFACTS / "PAPER_CANONICAL_METRICS.csv")
    }
    referenced = set(
        re.findall(
            r"\b[UOA][1-5]_[A-Z0-9_]+\b",
            _text(PAPER / "CLAIM_SENTENCE_BANK.md"),
        )
    )
    assert referenced == canonical


def test_aei_paper_excludes_internal_stage_labels_and_overclaims() -> None:
    manuscript = _text(MAIN).lower()
    forbidden = (
        "m0 go",
        "m1 no-go",
        "m0_go",
        "m1_no_go",
        "tier b",
        "claim_narrowing_go",
        "method_extension_no_go",
        "go_nogo",
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
    assert not re.search(r"\bP(?:[0-9]|1[0-6])\b", _text(MAIN))


def test_aei_paper_keeps_cost_statistics_and_oracle_scope_bounded() -> None:
    manuscript = _text(MAIN)
    assert manuscript.count("retrospective oracle") == 1
    assert manuscript.count("native-raster cost") >= 1
    assert "No result is translated into physical scanner time" in manuscript
    assert re.search(
        r"repeated\s+computational records, not independent samples", manuscript
    )
    assert "registered normalized-RGB-MSE reconstruction objective" in manuscript


def test_result_headings_avoid_defensive_language() -> None:
    manuscript = _text(MAIN)
    results = _section(manuscript, "Experimental Results and Discussion", "Conclusions")
    headings = "\n".join(
        re.findall(r"^\\(?:subsection|subsubsection)\{([^}]+)\}", results, re.MULTILINE)
    ).lower()
    forbidden = (
        "failure",
        "failed",
        "no-go",
        "not established",
        "not performance-superior",
        "residual deployable gap",
        "deployment readiness",
        "adverse",
        "worse",
        "unsupported",
        "post-freeze",
        "hash-bound",
    )
    assert not any(phrase in headings for phrase in forbidden)


def test_main_headline_numbers_match_canonical_metrics() -> None:
    rows = {
        row.claim_id: row for row in aei_paper_evidence.build_canonical_metrics(ROOT)
    }
    manuscript = _text(MAIN)
    expected = (
        f"{rows['U1_MATCHED_FIELD'].reference_value:.5f}",
        f"{rows['U1_MATCHED_FIELD'].candidate_value:.5f}",
        f"{100 * rows['U1_MATCHED_FIELD'].relative_effect:.1f}\\%",
        f"{100 * rows['U2_SPARSE_RETENTION'].estimate:.1f}\\%",
        f"{100 * rows['O2_TEACHER_TURNOVER'].estimate:.1f}\\%",
        f"{rows['O4_DYNAMIC_MINUS_STATIC'].estimate:.6f}",
        f"{rows['A4_BASELINE_MINUS_MAVIS'].candidate_value:.6f}",
        f"{rows['A4_BASELINE_MINUS_MAVIS'].reference_value:.6f}",
    )
    assert all(number in manuscript for number in expected)
