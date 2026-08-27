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


def test_aei_paper_builds_exactly_three_main_tables(tmp_path: Path) -> None:
    artifacts = aei_paper_tables.build_paper_tables(ROOT, tmp_path)
    assert set(artifacts) == {"table1", "table2", "table3"}
    for artifact in artifacts.values():
        assert artifact.csv.is_file() and artifact.csv.stat().st_size > 500
        assert artifact.tex.is_file() and artifact.tex.stat().st_size > 500
        assert artifact.caption.is_file() and artifact.caption.stat().st_size > 100


def test_aei_paper_table2_preserves_case_study_and_protocol_contract(
    tmp_path: Path,
) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table2"]
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


def test_aei_paper_table2_provenance_hashes_are_valid(tmp_path: Path) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table2"]
    for row in _rows(table.csv):
        source = ROOT / row["source_artifact"]
        assert source.is_file()
        assert row["source_hash"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_aei_paper_table3_has_progressive_columns_and_stage_order(
    tmp_path: Path,
) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table3"]
    rows = _rows(table.csv)
    required = {
        "part",
        "progressive_stage",
        "scientific_question",
        "key_evidence",
        "effect",
        "ci95_domains",
        "narrative_conclusion",
        "control_boundary",
        "source_claim_ids",
        "source_artifacts",
        "source_hashes",
        "canonical_authority_hash",
    }
    assert set(rows[0]) == required
    assert len(rows) == 12
    assert [row["progressive_stage"] for row in rows] == [
        "I1 Spatial enrichment",
        "I2 Sparse recoverability",
        "I3 Spatial heterogeneity",
        "I4 Objective conditioning",
        "I5 State conditioning",
        "I6 Predictor conditioning",
        "II1 Static reference",
        "II2 Dynamic valuation",
        "II3 Information-source attribution",
        "II4 Valuation/planning decomposition",
        "II5 Bounded set realization",
        "II6 Deployment calibration",
    ]


def test_aei_paper_table3_preserves_required_claim_groups(tmp_path: Path) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table3"]
    claims = {
        claim
        for row in _rows(table.csv)
        for claim in row["source_claim_ids"].split(";")
    }
    canonical = {
        row["claim_id"]
        for row in _rows(
            ROOT / "artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv"
        )
    }
    assert claims == canonical


def test_aei_paper_table3_keeps_calibration_boundaries_in_main_table(
    tmp_path: Path,
) -> None:
    table = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table3"]
    text = "\n".join(
        row["effect"] + " " + row["control_boundary"] for row in _rows(table.csv)
    ).lower()
    assert "learned global masks" in text
    assert "acquired-position/history" in text
    assert "shuffled" in text
    assert "no-feedback reference" in text
    assert "not performance-superior" in text


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


def test_aei_paper_table3_latex_groups_two_progressive_parts(
    tmp_path: Path,
) -> None:
    latex = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table3"].tex.read_text(
        encoding="utf-8"
    )
    assert latex.count("Part I") >= 1
    assert latex.count("Part II") >= 1
    assert latex.count("\\addlinespace") == 1


def test_aei_paper_table3_uses_readable_multipage_longtable(tmp_path: Path) -> None:
    latex = aei_paper_tables.build_paper_tables(ROOT, tmp_path)["table3"].tex.read_text(
        encoding="utf-8"
    )
    assert "\\begin{longtable}" in latex
    assert "\\endfirsthead" in latex
    assert "\\endhead" in latex
    assert "Continued on next page" in latex
    assert "\\begin{table" not in latex
    assert table_path_name(tmp_path) == "table3_progressive_evidence_chain.tex"


def table_path_name(tmp_path: Path) -> str:
    return aei_paper_tables.build_paper_tables(ROOT, tmp_path / "name")[
        "table3"
    ].tex.name


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
    assert len(rows) == 9
    for row in rows:
        path = tmp_path / row["path"]
        assert int(row["bytes"]) == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
