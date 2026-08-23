from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from cmc_bbdm.mva.a4_config import load_a4_config
from cmc_bbdm.mva.a4_figures import A4FigureError, render_a4_figures

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mva_a4_global_mask.yaml"
RANK_METHODS = (
    "global_appearance_mask",
    "global_reconstruction_mask",
    "global_mechanical_mask",
)
CAI_METHODS = (
    "uniform",
    *RANK_METHODS,
    "mechanical_oracle",
    "random_median",
)


def _write_tables(path: Path) -> None:
    config = load_a4_config(CONFIG, project_root=ROOT)
    path.mkdir()
    ranking_rows: list[dict[str, object]] = []
    for fold, outer_domain in enumerate(config.domain_order):
        for method_index, method in enumerate(RANK_METHODS):
            order = sorted(
                range(64),
                key=lambda cell: (
                    -((cell + fold + method_index) % 64 / 63.0),
                    cell,
                ),
            )
            positions = {cell: position for position, cell in enumerate(order)}
            for cell in range(64):
                ranking_rows.append(
                    {
                        "outer_domain": outer_domain,
                        "method": method,
                        "cell_index": cell,
                        "ranking_position": positions[cell],
                        "cell_score": (cell + fold + method_index) % 64 / 63.0,
                        "mean_raw_value": 0.1 + cell / 1000.0,
                        "mean_value_per_measurement": 0.01 + cell / 10000.0,
                        "source_domains": "|".join(
                            value
                            for value in config.domain_order
                            if value != outer_domain
                        ),
                        "source_specimen_count": 200,
                        "source_label_state_sha256": "a" * 63 + str(fold + 1),
                    }
                )
    pl.DataFrame(ranking_rows).write_csv(path / "rankings.csv")

    curve_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    for protocol_index, protocol in enumerate(("P-B", "P-A")):
        for method_index, method in enumerate(CAI_METHODS):
            for checkpoint in config.checkpoints:
                mae = 0.12 + 0.002 * method_index + 0.01 * protocol_index - 0.1 * checkpoint
                random = method == "random_median"
                row = {
                    "method": method,
                    "protocol": protocol,
                    "nominal_checkpoint": checkpoint,
                    "equal_domain_mae": mae,
                    "mae_mean": mae if random else None,
                    "mae_median": mae if random else None,
                    "mae_q05": mae - 0.004 if random else None,
                    "mae_q95": mae + 0.004 if random else None,
                    "effective_mean": checkpoint - 0.0001,
                    "effective_min": checkpoint - 0.0002,
                    "effective_max": checkpoint,
                    "normalized_rgb_mse": (
                        0.30 - checkpoint - 0.01 * method_index
                        if method in RANK_METHODS
                        else None
                    ),
                    "ssim": (
                        0.60 + checkpoint + 0.01 * method_index
                        if method in RANK_METHODS
                        else None
                    ),
                }
                curve_rows.append(row)
                if protocol == "P-B" and method in RANK_METHODS:
                    image_rows.append(row)
    pl.DataFrame(curve_rows, infer_schema_length=None).write_csv(
        path / "cai_curves.csv"
    )
    pl.DataFrame(image_rows, infer_schema_length=None).write_csv(
        path / "image_curves.csv"
    )


def test_render_a4_figures_is_complete_and_nonblank(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _write_tables(evidence)

    output = render_a4_figures(
        evidence,
        config_path=CONFIG,
        project_root=ROOT,
    )

    names = (
        "A4_global_rankings",
        "A4_cai_error_budget",
        "A4_image_task_tradeoff",
    )
    for name in names:
        png = output / f"{name}.png"
        svg = output / f"{name}.svg"
        assert png.is_file() and svg.is_file()
        image = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)
        assert image.shape[0] >= 900
        assert image.shape[1] >= 1800
        assert np.std(image) > 8.0
        text = svg.read_text(encoding="utf-8")
        assert "<svg" in text
        assert "DejaVu Sans" in text

    source = pl.read_csv(output / "source_data.csv")
    assert set(source["figure_id"]) == {
        "A4_global_rankings",
        "A4_cai_error_budget",
        "A4_image_task_tradeoff",
    }
    consensus = source.filter(pl.col("figure_id") == "A4_global_rankings")
    assert consensus.height == 3 * 64
    assert set(consensus["outer_domain"]) == {"consensus_6_outer_folds"}
    assert set(consensus["cell_index"]) == set(range(64))
    assert source.null_count().select(pl.all().sum()).row(0)[0] < source.height * 10


def test_render_a4_figures_rejects_incomplete_outer_roster(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _write_tables(evidence)
    config = load_a4_config(CONFIG, project_root=ROOT)
    rankings = pl.read_csv(evidence / "rankings.csv").filter(
        pl.col("outer_domain") != config.domain_order[-1]
    )
    rankings.write_csv(evidence / "rankings.csv")

    with pytest.raises(A4FigureError, match="ranking roster"):
        render_a4_figures(
            evidence,
            config_path=CONFIG,
            project_root=ROOT,
        )
