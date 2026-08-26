from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cmc_bbdm.mavis import aei_paper_figures

ROOT = Path(__file__).resolve().parents[1]


def test_aei_paper_figure_module_exists() -> None:
    assert importlib.util.find_spec("cmc_bbdm.mavis.aei_paper_figures") is not None


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_aei_paper_builds_four_traceable_figure_sources(tmp_path: Path) -> None:
    sources = aei_paper_figures.build_figure_sources(ROOT, tmp_path)
    assert set(sources) == {"figure1", "figure2", "figure3", "figure4"}
    required = {
        "panel",
        "series",
        "metric",
        "value",
        "ci95_lower",
        "ci95_upper",
        "status",
        "source_claim_id",
        "source_artifact",
        "source_hash",
    }
    for path in sources.values():
        assert path.is_file()
        rows = _rows(path)
        assert rows
        assert required <= set(rows[0])


def test_aei_paper_figure1_contains_no_performance_values(tmp_path: Path) -> None:
    source = aei_paper_figures.build_figure_sources(ROOT, tmp_path)["figure1"]
    assert all(not row["value"] for row in _rows(source))


def test_aei_paper_figure2_uses_registered_b_path_and_learned_boundary(
    tmp_path: Path,
) -> None:
    source = aei_paper_figures.build_figure_sources(ROOT, tmp_path)["figure2"]
    rows = _rows(source)
    claim_ids = {row["source_claim_id"] for row in rows}
    assert "U1_MATCHED_FIELD" in claim_ids
    assert "U1_INDEPENDENT_FIELD_SENSITIVITY" not in claim_ids
    assert "U2_SPARSE_RETENTION" in claim_ids
    assert "U4_LEARNED_SPECIFICITY_BOUNDARY" in claim_ids


def test_aei_paper_figure3_preserves_all_central_observability_controls(
    tmp_path: Path,
) -> None:
    source = aei_paper_figures.build_figure_sources(ROOT, tmp_path)["figure3"]
    rows = _rows(source)
    claim_ids = {row["source_claim_id"] for row in rows}
    assert {
        "O1_STATIC_SPEARMAN",
        "O3_REAL_MINUS_POSITIONS",
        "O3_REAL_MINUS_RECONSTRUCTION",
        "O4_DYNAMIC_MINUS_SHUFFLED",
    } <= claim_ids
    assert any(row["status"] == "ADVERSE_CONTROL" for row in rows)


def test_aei_paper_figure4_preserves_feedback_and_final_boundary(
    tmp_path: Path,
) -> None:
    source = aei_paper_figures.build_figure_sources(ROOT, tmp_path)["figure4"]
    claim_ids = {row["source_claim_id"] for row in _rows(source)}
    assert {"A3_FEEDBACK_BENEFIT", "A4_BASELINE_MINUS_MAVIS"} <= claim_ids


def test_aei_paper_figures_export_vector_and_300_dpi_nonblank_raster(
    tmp_path: Path,
) -> None:
    artifacts = aei_paper_figures.render_paper_figures(ROOT, tmp_path)
    assert set(artifacts) == {"figure1", "figure2", "figure3", "figure4"}
    for artifact in artifacts.values():
        assert {artifact.svg.suffix, artifact.pdf.suffix, artifact.png.suffix} == {
            ".svg",
            ".pdf",
            ".png",
        }
        assert all(
            path.is_file() and path.stat().st_size > 1000 for path in artifact.outputs
        )
        image = Image.open(artifact.png)
        dpi = image.info.get("dpi")
        assert dpi is not None
        assert dpi[0] == pytest.approx(300, abs=0.2)
        pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
        assert pixels.shape[0] > 700
        assert pixels.shape[1] > 1400
        assert pixels.std() > 10.0


def test_aei_paper_visible_figure_text_has_no_internal_stage_labels(
    tmp_path: Path,
) -> None:
    artifacts = aei_paper_figures.render_paper_figures(ROOT, tmp_path)
    forbidden = ("M0_GO", "M1_NO_GO", "Tier B", "CLAIM_NARROWING_GO", "P9 passed")
    for artifact in artifacts.values():
        visible = artifact.svg.read_text(encoding="utf-8") + artifact.caption.read_text(
            encoding="utf-8"
        )
        assert not any(phrase in visible for phrase in forbidden)


def test_aei_paper_figures_regenerate_deterministically(tmp_path: Path) -> None:
    first = aei_paper_figures.render_paper_figures(ROOT, tmp_path / "first")
    second = aei_paper_figures.render_paper_figures(ROOT, tmp_path / "second")
    for figure_id in first:
        first_artifact = first[figure_id]
        second_artifact = second[figure_id]
        for left, right in zip(
            (
                *first_artifact.outputs,
                first_artifact.source_csv,
                first_artifact.caption,
            ),
            (
                *second_artifact.outputs,
                second_artifact.source_csv,
                second_artifact.caption,
            ),
            strict=True,
        ):
            assert left.name == right.name
            assert left.read_bytes() == right.read_bytes()


def test_aei_paper_figure_manifest_binds_every_deliverable(tmp_path: Path) -> None:
    artifacts = aei_paper_figures.render_paper_figures(ROOT, tmp_path)
    manifest = tmp_path / "FIGURE_CHECKSUMS.csv"
    rows = _rows(manifest)
    assert len(rows) == 20
    assert {row["figure_id"] for row in rows} == set(artifacts)
    for row in rows:
        path = tmp_path / row["path"]
        assert path.is_file()
        assert int(row["bytes"]) == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
