from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/aei_information_hierarchy"
PAPER = ROOT / "paper_aei_information_hierarchy"

IDENTITY_LEDGER = ARTIFACTS / "PAPER_METHOD_IDENTITY_LEDGER.md"
VISIBILITY_CSV = ARTIFACTS / "PAPER_CLAIM_VISIBILITY_MAP.csv"
VISIBILITY_MD = ARTIFACTS / "PAPER_CLAIM_VISIBILITY_MAP.md"

VISIBILITY_COLUMNS = (
    "claim_id",
    "canonical_layer",
    "chronology_class",
    "paper_module",
    "compressed_stage",
    "visibility",
    "main_role",
    "supplement_role",
    "main_figure",
    "main_table",
    "main_section",
    "required_direction",
    "required_scope",
    "source_artifact",
)

EXPECTED_IDENTITY_ROWS = (
    (
        "Task-Relevant Information Acquisition",
        "paper-level proposed framework",
        "PRIMARY METHOD",
        "proposed framework / approach",
    ),
    (
        "Information Characterization",
        "Part-I primary module",
        "PRIMARY MODULE",
        "information characterization",
    ),
    (
        "State-Conditioned Task-Oriented Acquisition",
        "Part-II primary module",
        "PRIMARY MODULE",
        "state-conditioned acquisition",
    ),
    (
        "MAVIS",
        "codebase closed-loop implementation",
        "IMPLEMENTATION ONLY",
        "state-conditioned learned implementation",
    ),
    (
        "mvd_m1_o2",
        "static deployable reference/comparator",
        "REFERENCE",
        "static reference",
    ),
    (
        "mechanical oracle",
        "retrospective task-value opportunity analysis",
        "ORACLE",
        "mechanical oracle",
    ),
    (
        "reconstruction oracle",
        "retrospective objective comparator",
        "ORACLE",
        "reconstruction oracle",
    ),
    (
        "acquired-position/history",
        "source control",
        "CONTROL",
        "acquired-position/history control",
    ),
    (
        "reconstruction",
        "source control",
        "CONTROL",
        "reconstruction control",
    ),
    (
        "shuffled content",
        "source control",
        "CONTROL",
        "shuffled-content control",
    ),
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_method_identity_ledger_declares_exact_scientific_roles() -> None:
    text = IDENTITY_LEDGER.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    for entity, role, status, main_name in EXPECTED_IDENTITY_ROWS:
        row = f"| {entity} | {role} | {status} | {main_name} |"
        assert row in normalized
    assert "MAVIS is the proposed method" not in text
    assert "mvd_m1_o2 is a published competing method" not in text


def test_claim_visibility_map_covers_canonical_authority_once_in_order() -> None:
    canonical = _rows(ARTIFACTS / "PAPER_CANONICAL_METRICS.csv")
    visibility = _rows(VISIBILITY_CSV)
    assert tuple(visibility[0]) == VISIBILITY_COLUMNS
    assert len(canonical) == len(visibility) == 39
    assert [row["claim_id"] for row in visibility] == [
        row["claim_id"] for row in canonical
    ]
    assert len({row["claim_id"] for row in visibility}) == 39


def test_claim_visibility_map_has_required_distribution_and_special_cases() -> None:
    rows = _rows(VISIBILITY_CSV)
    assert Counter(row["visibility"] for row in rows) == {
        "MAIN_HEADLINE": 12,
        "MAIN_SUPPORT": 15,
        "MAIN_SYSTEM_DIAGNOSTIC": 1,
        "SUPPLEMENT_ONLY": 11,
    }
    by_id = {row["claim_id"]: row for row in rows}
    assert by_id["A3_FEEDBACK_BENEFIT"]["visibility"] == "SUPPLEMENT_ONLY"
    assert by_id["A3_FEEDBACK_BENEFIT"]["main_figure"] == "none"
    assert by_id["A3_FEEDBACK_BENEFIT"]["main_table"] == "none"
    assert by_id["A4_BASELINE_MINUS_MAVIS"]["visibility"] == "MAIN_SYSTEM_DIAGNOSTIC"
    assert by_id["A4_BASELINE_MINUS_MAVIS"]["main_figure"] == "none"
    assert by_id["A4_BASELINE_MINUS_MAVIS"]["main_table"] == "none"
    assert by_id["A4_BASELINE_MINUS_MAVIS"]["main_section"] == "5.2.3"


def test_claim_visibility_map_preserves_evidence_identity() -> None:
    canonical = {
        row["claim_id"]: row for row in _rows(ARTIFACTS / "PAPER_CANONICAL_METRICS.csv")
    }
    chronology = {
        row["claim_id"]: row
        for row in _rows(ARTIFACTS / "PAPER_EVIDENCE_CHRONOLOGY.csv")
    }
    for row in _rows(VISIBILITY_CSV):
        claim = canonical[row["claim_id"]]
        timing = chronology[row["claim_id"]]
        assert row["canonical_layer"] == claim["layer"]
        assert row["chronology_class"] == timing["chronology_class"]
        assert row["source_artifact"] == claim["source_artifact"]
        assert row["required_direction"]
        assert row["required_scope"]


def test_claim_visibility_markdown_matches_csv_categories() -> None:
    text = VISIBILITY_MD.read_text(encoding="utf-8")
    assert "39 canonical claims" in text
    for visibility, count in (
        ("MAIN_HEADLINE", 12),
        ("MAIN_SUPPORT", 15),
        ("MAIN_SYSTEM_DIAGNOSTIC", 1),
        ("SUPPLEMENT_ONLY", 11),
    ):
        assert f"| `{visibility}` | {count} |" in text
    for claim_id in (
        "U1_MATCHED_FIELD",
        "O4_DYNAMIC_MINUS_STATIC",
        "A3_FEEDBACK_BENEFIT",
        "A4_BASELINE_MINUS_MAVIS",
    ):
        assert f"`{claim_id}`" in text
