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


def _visible_claims(prefix: str) -> set[str]:
    rows = _rows(
        ROOT / "artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.csv"
    )
    return {row["claim_id"] for row in rows if row["main_figure"].startswith(prefix)}


def test_aei_paper_figure2_uses_only_part_i_main_visible_claims(
    tmp_path: Path,
) -> None:
    source = aei_paper_figures.build_figure_sources(ROOT, tmp_path)["figure2"]
    rows = _rows(source)
    claim_ids = {row["source_claim_id"] for row in rows if row["source_claim_id"]}
    assert claim_ids == _visible_claims("figure2")
    assert not claim_ids & {
        "O2_TEACHER_TURNOVER",
        "O2_TEACHER_RANK",
        "O2_TEACHER_TOPK",
        "O2_TEACHER_OPPORTUNITY",
        "U5_RIDGE_HUBER_SPEARMAN",
        "U5_RIDGE_MLP_SPEARMAN",
    }
    assert "U1_INDEPENDENT_FIELD_SENSITIVITY" not in claim_ids
    assert "U4_LEARNED_SPECIFICITY_BOUNDARY" not in claim_ids
    assert {
        "results/mva/a2_oracle_value/oracle_values.parquet",
        "results/mavis/p1_state_bank/state_manifest.parquet",
    } <= {row["source_artifact"] for row in rows}
    assert set("abcdef") <= {panel for row in rows for panel in row["panel"]}


def test_aei_paper_figure3_links_state_evolution_dynamic_value_and_attribution(
    tmp_path: Path,
) -> None:
    source = aei_paper_figures.build_figure_sources(ROOT, tmp_path)["figure3"]
    rows = _rows(source)
    claim_ids = {row["source_claim_id"] for row in rows if row["source_claim_id"]}
    assert claim_ids == _visible_claims("figure3")
    assert {
        "O2_TEACHER_TURNOVER",
        "O2_TEACHER_RANK",
        "O2_TEACHER_TOPK",
        "O2_TEACHER_OPPORTUNITY",
        "U5_RIDGE_HUBER_SPEARMAN",
        "U5_RIDGE_MLP_SPEARMAN",
        "O4_DYNAMIC_MINUS_STATIC",
        "O1_STATIC_SPEARMAN",
    } == claim_ids
    assert {
        "results/mavis/p3_dynamic_voi/action_scores.parquet",
        "results/mavis/p1_state_bank/state_manifest.parquet",
    } <= {row["source_artifact"] for row in rows}
    assert not any(row["source_claim_id"].startswith("O3_") for row in rows)


def test_aei_paper_figure4_contains_controls_realization_and_a4_boundary(
    tmp_path: Path,
) -> None:
    source = aei_paper_figures.build_figure_sources(ROOT, tmp_path)["figure4"]
    rows = _rows(source)
    claim_ids = {row["source_claim_id"] for row in rows if row["source_claim_id"]}
    assert (
        claim_ids
        == _visible_claims("figure4")
        == {
            "O3_REAL_CHANGE",
            "O3_REAL_MINUS_POSITIONS",
            "O3_REAL_MINUS_RECONSTRUCTION",
            "O4_DYNAMIC_MINUS_SHUFFLED",
            "A1_VALUATION_SUBSTITUTION",
            "A1_LEARNED_PLANNING_SUBSTITUTION",
            "A1_TRUE_VALUE_PLANNING_SUBSTITUTION",
            "A2_GREEDY_PLANNING_REGRET",
            "A2_BEAM4_PLANNING_REGRET",
            "A4_BASELINE_MINUS_MAVIS",
        }
    )
    assert "A3_FEEDBACK_BENEFIT" not in claim_ids
    assert any(row["series"] == "Field-content control" for row in rows)


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


def test_aei_paper_figure_outputs_use_progressive_stems(tmp_path: Path) -> None:
    artifacts = aei_paper_figures.render_paper_figures(ROOT, tmp_path)
    assert {key: value.pdf.name for key, value in artifacts.items()} == {
        "figure1": "figure1_task_relevant_acquisition_framework.pdf",
        "figure2": "figure2_information_characterization.pdf",
        "figure3": "figure3_state_conditioned_value.pdf",
        "figure4": "figure4_valuation_planning_realization.pdf",
    }


def test_aei_paper_visible_figure_text_has_no_internal_stage_labels(
    tmp_path: Path,
) -> None:
    artifacts = aei_paper_figures.render_paper_figures(ROOT, tmp_path)
    forbidden = (
        "M0_GO",
        "M1_NO_GO",
        "Tier B",
        "CLAIM_NARROWING_GO",
        "P9 passed",
        "MAVIS",
        "not performance-superior",
    )
    for artifact in artifacts.values():
        visible = artifact.svg.read_text(encoding="utf-8") + artifact.caption.read_text(
            encoding="utf-8"
        )
        assert not any(phrase in visible for phrase in forbidden)


def test_aei_paper_visual_narrative_uses_four_stage_why_flow_and_compact_titles(
    tmp_path: Path,
) -> None:
    artifacts = aei_paper_figures.render_paper_figures(ROOT, tmp_path)
    figure1 = artifacts["figure1"].svg.read_text(encoding="utf-8")
    for phrase in (
        "Complete sensing field",
        "Limited sensing under exact cost",
        "Task-relevant value",
    ):
        assert phrase in figure1
    assert "State-conditioned" in figure1
    assert "acquisition loop" in figure1

    figure2 = artifacts["figure2"].svg.read_text(encoding="utf-8")
    figure3 = artifacts["figure3"].svg.read_text(encoding="utf-8")
    assert "Heterogeneous" in figure2 and "spatial opportunity" in figure2
    assert "Value changes with" in figure3 and "acquired evidence" in figure3


def test_aei_paper_narrative_titles_fit_their_declared_columns(tmp_path: Path) -> None:
    figure1 = aei_paper_figures._render_figure1([])
    figure1.canvas.draw()
    renderer = figure1.canvas.get_renderer()
    axis = figure1.axes[0]
    by_text = {text.get_text(): text for text in axis.texts}
    for label, width_fraction in (
        ("PART II\nSTATE-CONDITIONED ACQUISITION", 0.23),
        ("State-conditioned\nacquisition loop", 0.21),
    ):
        assert by_text[label].get_window_extent(renderer).width <= (
            width_fraction * axis.get_window_extent(renderer).width - 4
        )

    sources = aei_paper_figures.build_figure_sources(ROOT, tmp_path)
    figures = (
        aei_paper_figures._render_figure2_nature(ROOT, _rows(sources["figure2"])),
        aei_paper_figures._render_figure3_nature(ROOT, _rows(sources["figure3"])),
    )
    for figure in figures:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axes = figure._aei_alignment_options["axes"]
        for row in (axes[:3], axes[3:]):
            for index, panel_axis in enumerate(row):
                title_box = panel_axis._left_title.get_window_extent(renderer)
                right_limit = (
                    row[index + 1].get_window_extent(renderer).x0 - 3
                    if index + 1 < len(row)
                    else figure.bbox.x1 - 3
                )
                assert title_box.x1 <= right_limit


def test_aei_paper_image_aspect_render_has_no_fixed_limit_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING", logger="matplotlib.axes._base")
    aei_paper_figures.render_paper_figures(ROOT, tmp_path / "main")
    aei_paper_figures.render_supplementary_figures(ROOT, tmp_path / "supplement")
    assert not any("Ignoring fixed" in record.getMessage() for record in caplog.records)


def test_aei_paper_generated_svgs_have_no_trailing_whitespace(
    tmp_path: Path,
) -> None:
    artifacts = aei_paper_figures.render_paper_figures(ROOT, tmp_path)
    for artifact in artifacts.values():
        lines = artifact.svg.read_text(encoding="utf-8").splitlines()
        assert all(line == line.rstrip() for line in lines)


def test_aei_paper_svgs_keep_text_editable(tmp_path: Path) -> None:
    artifacts = aei_paper_figures.render_paper_figures(ROOT, tmp_path)
    for artifact in artifacts.values():
        svg = artifact.svg.read_text(encoding="utf-8")
        assert "<text" in svg


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
    assert len(rows) == 26
    assert {row["figure_id"] for row in rows} == set(artifacts)
    for row in rows:
        path = tmp_path / row["path"]
        assert path.is_file()
        assert int(row["bytes"]) == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_aei_supplementary_gallery_is_traceable_and_nonblank(tmp_path: Path) -> None:
    artifacts = aei_paper_figures.render_supplementary_figures(ROOT, tmp_path)
    assert set(artifacts) == {"supplementary_figure_s1"}
    artifact = artifacts["supplementary_figure_s1"]
    assert artifact.pdf.name == (
        "supplementary_figure_s1_cross_domain_state_priority_gallery.pdf"
    )
    assert all(
        path.is_file() and path.stat().st_size > 1000 for path in artifact.outputs
    )
    pixels = np.asarray(Image.open(artifact.png).convert("RGB"), dtype=np.float64)
    assert pixels.shape[0] > 1800
    assert pixels.shape[1] > 1400
    assert pixels.std() > 10.0


def test_aei_paper_exports_all_formal_panels_as_unscaled_pngs(
    tmp_path: Path,
) -> None:
    main = aei_paper_figures.render_paper_figures(ROOT, tmp_path / "main")
    supplement = aei_paper_figures.render_supplementary_figures(
        ROOT, tmp_path / "supplement"
    )
    exports = aei_paper_figures.export_panel_pngs(
        {**main, **supplement}, tmp_path / "panels"
    )
    expected = {
        "figure1_full",
        *(f"figure2_{panel}" for panel in "abcdef"),
        *(f"figure3_{panel}" for panel in "abcdef"),
        *(f"figure4_{panel}" for panel in "abcd"),
        *(f"supplementary_figure_s1_{panel}" for panel in "abcdefghijkl"),
    }
    assert set(exports) == expected
    manifest = _rows(tmp_path / "panels/PANEL_PNG_MANIFEST.csv")
    assert len(manifest) == len(expected) == 29
    assert all(not Path(row["source_png"]).is_absolute() for row in manifest)
    for key, path in exports.items():
        assert path.is_file() and path.stat().st_size > 1000, key
        image = Image.open(path).convert("RGB")
        pixels = np.asarray(image, dtype=np.float64)
        assert image.width > 300 and image.height > 200
        assert pixels.std() > 8.0
        if key.startswith("supplementary_figure_s1_"):
            assert 0.65 <= image.width / image.height <= 1.65
