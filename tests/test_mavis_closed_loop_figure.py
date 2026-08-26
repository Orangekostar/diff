from __future__ import annotations

import numpy as np
import polars as pl
from PIL import Image

from cmc_bbdm.mavis.closed_loop_figure import render_task_specificity_curve


def test_task_specificity_curve_exports_editable_and_nonblank_formats(tmp_path) -> None:
    methods = (
        "uniform",
        "reconstruction_driven",
        "mavis_no_feedback",
        "mavis_positions_only",
        "mavis_shuffled_content",
        "mavis_full",
    )
    rows = [
        {
            "method": method,
            "nominal_checkpoint": checkpoint,
            "domain_count": 6,
            "mean_exact_acquired_cost": checkpoint * 100000,
            "mean_effective_budget": checkpoint,
            "domain_balanced_cai_mae": 0.25 - 0.01 * index - 0.1 * checkpoint,
            "worst_domain_cai_mae": 0.3,
            "domain_balanced_reconstruction_mse": (
                0.2 - 0.015 * (method == "reconstruction_driven") - 0.1 * checkpoint
            ),
        }
        for index, method in enumerate(methods)
        for checkpoint in (0.03125, 0.0625, 0.125, 0.25)
    ]

    outputs = render_task_specificity_curve(
        pl.DataFrame(rows),
        output_root=tmp_path,
    )

    assert {path.suffix for path in outputs} == {".svg", ".pdf", ".png"}
    assert all(path.stat().st_size > 1000 for path in outputs)
    svg = (tmp_path / "task_specificity_vs_exact_cost.svg").read_text(
        encoding="utf-8"
    )
    assert "CAI MAE" in svg
    assert "Reconstruction MSE" in svg
    assert "<text" in svg
    assert all(line == line.rstrip() for line in svg.splitlines())
    pixels = np.asarray(
        Image.open(tmp_path / "task_specificity_vs_exact_cost.png").convert("RGB")
    )
    assert pixels.std() > 5.0

    replay = tmp_path / "replay"
    replay_outputs = render_task_specificity_curve(
        pl.DataFrame(rows),
        output_root=replay,
    )
    assert [path.read_bytes() for path in outputs] == [
        path.read_bytes() for path in replay_outputs
    ]
