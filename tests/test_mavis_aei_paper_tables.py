from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

from cmc_bbdm.mavis import aei_paper_tables

ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_aei_paper_table_module_exists() -> None:
    assert importlib.util.find_spec("cmc_bbdm.mavis.aei_paper_tables") is not None


def test_aei_paper_builds_exactly_two_main_tables(tmp_path: Path) -> None:
    artifacts = aei_paper_tables.build_paper_tables(ROOT, tmp_path)
    assert set(artifacts) == {"table1", "table2"}
    for artifact in artifacts.values():
        assert artifact.csv.is_file() and artifact.csv.stat().st_size > 500
        assert artifact.tex.is_file() and artifact.tex.stat().st_size > 500
        assert artifact.caption.is_file() and artifact.caption.stat().st_size > 100


def test_aei_paper_table1_preserves_case_study_and_protocol_contract(
    tmp_path: Path,
) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table1"]
    rows = _rows(table.csv)
    assert set(rows[0]) == {"item", "value", "source_artifact", "source_hash"}
    values = {row["item"]: row["value"] for row in rows}
    assert "276" in values["Physical cohort"]
    assert "6" in values["Held-out domains"]
    for count in (45, 49, 43, 59, 42, 38):
        assert str(count) in values["Domain specimen counts"]
    assert "LODO" in values["Validation protocol"]
    assert "native-raster" in values["Acquisition cost"]
    assert "scanner-time" in values["Acquisition cost"]
    assert "non-deployable" in values["Teacher/oracle information"]
    assert "physical specimen" in values["Statistical units"]
    assert "not independent" in values["Computational rows"]


def test_aei_paper_table1_provenance_hashes_are_valid(tmp_path: Path) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table1"]
    for row in _rows(table.csv):
        source = ROOT / row["source_artifact"]
        assert source.is_file()
        assert row["source_hash"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_aei_paper_table2_has_fixed_hierarchy_columns_and_rows(tmp_path: Path) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table2"]
    rows = _rows(table.csv)
    required = {
        "layer",
        "question",
        "key_comparison",
        "effect",
        "ci95",
        "domains",
        "evidence_type",
        "conclusion",
        "source_claim_ids",
    }
    assert required <= set(rows[0])
    assert len(rows) == 13
    assert [row["layer"] for row in rows].count("Useful") == 5
    assert [row["layer"] for row in rows].count("Observable") == 4
    assert [row["layer"] for row in rows].count("Actionable") == 4


def test_aei_paper_table2_preserves_required_claim_groups(tmp_path: Path) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table2"]
    claims = {
        claim
        for row in _rows(table.csv)
        for claim in row["source_claim_ids"].split(";")
    }
    assert {
        "U1_MATCHED_FIELD",
        "U2_SPARSE_RETENTION",
        "U3_UNIFORM_ORACLE",
        "U4_LEARNED_SPECIFICITY_BOUNDARY",
        "U5_RIDGE_MLP_SPEARMAN",
        "O1_STATIC_SPEARMAN",
        "O2_TEACHER_TURNOVER",
        "O3_REAL_MINUS_POSITIONS",
        "O3_REAL_MINUS_RECONSTRUCTION",
        "O4_DYNAMIC_MINUS_SHUFFLED",
        "A1_VALUATION_SUBSTITUTION",
        "A2_GREEDY_PLANNING_REGRET",
        "A3_FEEDBACK_BENEFIT",
        "A4_BASELINE_MINUS_MAVIS",
    } <= claims


def test_aei_paper_table2_keeps_adverse_controls_in_main_table(tmp_path: Path) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table2"]
    text = "\n".join(
        row["effect"] + " " + row["conclusion"] for row in _rows(table.csv)
    ).lower()
    assert "learned global masks" in text
    assert "worse than positions" in text
    assert "shuffled" in text
    assert "feedback is adverse" in text
    assert "did not outperform" in text


def test_aei_paper_tables_are_booktabs_without_internal_stage_labels(
    tmp_path: Path,
) -> None:
    artifacts = aei_paper_tables.build_paper_tables(ROOT, tmp_path)
    for artifact in artifacts.values():
        latex = artifact.tex.read_text(encoding="utf-8")
        assert "\\toprule" in latex
        assert "\\midrule" in latex
        assert "\\bottomrule" in latex
        assert "|" not in latex
        visible = latex + artifact.caption.read_text(encoding="utf-8")
        assert not any(
            phrase in visible for phrase in ("M0_GO", "M1_NO_GO", "Tier B", "GO_NOGO")
        )


def test_aei_paper_table2_latex_groups_repeated_layer_questions(
    tmp_path: Path,
) -> None:
    latex = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table2"].tex.read_text(
        encoding="utf-8"
    )
    assert latex.count("Does it improve the task?") == 1
    assert latex.count("Is value observable from legal state?") == 1
    assert latex.count("Does it improve a bounded decision?") == 1
    assert latex.count("\\addlinespace") == 2


def test_aei_paper_tables_regenerate_deterministically(tmp_path: Path) -> None:
    first = aei_paper_tables.build_paper_tables(ROOT, tmp_path / "first")
    second = aei_paper_tables.build_paper_tables(ROOT, tmp_path / "second")
    for table_id in first:
        for left, right in zip(
            (first[table_id].csv, first[table_id].tex, first[table_id].caption),
            (second[table_id].csv, second[table_id].tex, second[table_id].caption),
            strict=True,
        ):
            assert left.name == right.name
            assert left.read_bytes() == right.read_bytes()
    assert (tmp_path / "first" / "TABLE_CHECKSUMS.csv").read_bytes() == (
        tmp_path / "second" / "TABLE_CHECKSUMS.csv"
    ).read_bytes()


def test_aei_paper_table_manifest_binds_every_deliverable(tmp_path: Path) -> None:
    aei_paper_tables.build_paper_tables(ROOT, tmp_path)
    rows = _rows(tmp_path / "TABLE_CHECKSUMS.csv")
    assert len(rows) == 6
    for row in rows:
        path = tmp_path / row["path"]
        assert int(row["bytes"]) == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
