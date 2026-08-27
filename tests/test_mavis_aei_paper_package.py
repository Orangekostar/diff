from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

from cmc_bbdm.mavis import aei_paper_package

ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_aei_paper_package_has_complete_working_tree(tmp_path: Path) -> None:
    package = aei_paper_package.build_paper_package(ROOT, tmp_path)
    assert sorted(path.name for path in package.figures.glob("*.pdf")) == [
        "figure1_information_hierarchy.pdf",
        "figure2_usefulness.pdf",
        "figure3_observability.pdf",
        "figure4_actionability.pdf",
    ]
    assert sorted(path.name for path in package.tables.glob("*.tex")) == [
        "table1_closest_work.tex",
        "table2_case_protocol.tex",
        "table3_hierarchy_evidence.tex",
    ]
    assert (package.root / "elsarticle.cls").is_file()
    assert (package.root / "elsarticle-num.bst").is_file()


def test_aei_paper_package_contains_required_supplementary_evidence(
    tmp_path: Path,
) -> None:
    package = aei_paper_package.build_paper_package(ROOT, tmp_path)
    names = {path.name for path in package.supplementary_data.iterdir()}
    assert {
        "S03_p9_checkpoint_metrics.csv",
        "S05_p10_control_matrix.csv",
        "S08_p11_secondary_metrics.csv",
        "S10_p12_substitution_matrix.csv",
        "S12_p13_planning_per_domain.csv",
        "S14_p15_learner_pair_metrics.csv",
        "S15_p16_feedback_strata.csv",
        "S17_provenance_hashes.csv",
        "S18_external_dataset_audit.json",
    } <= names


def test_aei_submission_source_is_flat_and_rewrites_local_inputs(
    tmp_path: Path,
) -> None:
    package = aei_paper_package.build_paper_package(ROOT, tmp_path)
    assert all(path.is_file() for path in package.submission_source.iterdir())
    manuscript = (package.submission_source / "main.tex").read_text(encoding="utf-8")
    assert "figures/" not in manuscript
    assert "tables/" not in manuscript
    assert "\\graphicspath{{./}}" in manuscript
    assert "\\input{table1_closest_work.tex}" in manuscript
    assert "\\input{table2_case_protocol.tex}" in manuscript
    assert "\\input{table3_hierarchy_evidence.tex}" in manuscript


def test_aei_submission_manifest_matches_every_listed_file(tmp_path: Path) -> None:
    package = aei_paper_package.build_paper_package(ROOT, tmp_path)
    with package.manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert [row["path"] for row in rows] == sorted(row["path"] for row in rows)
    for row in rows:
        path = package.submission_source / row["path"]
        assert path.is_file()
        assert int(row["bytes"]) == path.stat().st_size
        assert row["sha256"] == _hash(path)


def test_aei_submission_zip_regenerates_deterministically(tmp_path: Path) -> None:
    first = aei_paper_package.build_paper_package(ROOT, tmp_path / "first")
    second = aei_paper_package.build_paper_package(ROOT, tmp_path / "second")
    assert _hash(first.archive) == _hash(second.archive)
    with zipfile.ZipFile(first.archive) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert all(info.date_time == (2026, 1, 1, 0, 0, 0) for info in archive.infolist())
