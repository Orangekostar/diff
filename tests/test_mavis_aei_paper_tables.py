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


def test_aei_paper_table1_is_compact_case_and_protocol_contract(
    tmp_path: Path,
) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table1"]
    rows = _rows(table.csv)
    assert set(rows[0]) == {"item", "value", "source_artifact", "source_hash"}
    assert len(rows) == 6
    values = {row["item"]: row["value"] for row in rows}
    assert "276" in values["Cohort"]
    assert "6" in values["Cohort"]
    for count in (45, 49, 43, 59, 42, 38):
        assert str(count) in values["Cohort"]
    assert "LODO" in values["Evaluation"]
    assert "native-raster" in values["Actions and cost"]
    assert "scanner-time" in values["Actions and cost"]
    assert "non-deployable" in values["Information boundary"]
    assert "physical specimen" in values["Statistics"]
    assert "not independent" in values["Statistics"]


def test_aei_paper_table_provenance_hashes_are_valid(tmp_path: Path) -> None:
    for table in aei_paper_tables.build_paper_tables(ROOT, tmp_path).values():
        for row in _rows(table.csv):
            for source_artifact, source_hash in zip(
                row["source_artifact"].split(";"),
                row["source_hash"].split(";"),
                strict=True,
            ):
                source = ROOT / source_artifact
                assert source.is_file()
                assert source_hash == hashlib.sha256(source.read_bytes()).hexdigest()


def test_aei_paper_table2_has_six_stage_results_and_four_display_columns(
    tmp_path: Path,
) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table2"]
    rows = _rows(table.csv)
    assert set(rows[0]) == {
        "stage",
        "scientific_question",
        "headline_evidence",
        "scope_boundary",
        "source_claim_ids",
        "source_artifact",
        "source_hash",
    }
    assert len(rows) == 6
    assert [row["stage"] for row in rows] == [
        "Spatial information and sparse recoverability",
        "Task-conditioned spatial measurement value",
        "State- and predictor-conditioned measurement value",
        "State-conditioned valuation",
        "Information-source and component decomposition",
        "Cost-constrained set realization",
    ]
    latex = table.tex.read_text(encoding="utf-8")
    for display in ("Stage", "Question", "Headline evidence", "Scope boundary"):
        assert display in latex
    for audit in ("Source claim", "Source artifact", "Source hash"):
        assert audit not in latex


def test_aei_paper_table2_contains_only_main_visible_claims(tmp_path: Path) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table2"]
    claims = {
        claim
        for row in _rows(table.csv)
        for claim in row["source_claim_ids"].split(";")
    }
    visible = {
        row["claim_id"]
        for row in _rows(
            ROOT / "artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.csv"
        )
        if row["visibility"] in {"MAIN_HEADLINE", "MAIN_SUPPORT"}
    }
    assert claims == visible
    assert "A3_FEEDBACK_BENEFIT" not in claims
    assert "A4_BASELINE_MINUS_MAVIS" not in claims


def test_aei_paper_table2_uses_saliency_not_legacy_reconstruction_story(
    tmp_path: Path,
) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table2"]
    visible = table.csv.read_text(encoding="utf-8") + table.tex.read_text(
        encoding="utf-8"
    )
    assert "appearance - CAI AUEBC" in visible
    assert "map Spearman" in visible
    assert "dynamic real - shuffled" in visible
    assert "real - reconstruction" not in visible.lower()
    assert "image task contrast" not in visible.lower()


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
            phrase in visible
            for phrase in (
                "M0_GO",
                "M1_NO_GO",
                "Tier B",
                "GO_NOGO",
                "MAVIS",
                "not performance-superior",
            )
        )
    assert artifacts["table1"].tex.name == "table1_case_protocol.tex"
    assert artifacts["table2"].tex.name == "table2_task_relevant_results.tex"


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
