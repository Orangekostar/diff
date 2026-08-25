from __future__ import annotations

import numpy as np
import polars as pl
from PIL import Image

from cmc_bbdm.mavis.mris_figure import render_mris_cost_curve


def test_mavis_mris_cost_curve_exports_editable_and_nonblank_formats(tmp_path) -> None:
    modes = ("static", "positions_only", "real", "shuffled", "reconstruction")
    rows = [
        {
            "mode": mode,
            "nominal_checkpoint": checkpoint,
            "domain_count": 6,
            "mean_exact_acquired_cost": checkpoint * 100000,
            "mean_effective_budget": checkpoint,
            "domain_balanced_mae": 0.2 - 0.01 * mode_index - 0.1 * checkpoint,
            "worst_domain_mae": 0.3,
        }
        for mode_index, mode in enumerate(modes)
        for checkpoint in (0.03125, 0.0625, 0.125, 0.25)
    ]
    metrics = pl.DataFrame(rows)

    outputs = render_mris_cost_curve(metrics, output_root=tmp_path)

    assert {path.suffix for path in outputs} == {".svg", ".pdf", ".png"}
    assert all(path.stat().st_size > 1000 for path in outputs)
    svg = (tmp_path / "mris_cai_mae_vs_exact_cost.svg").read_text(encoding="utf-8")
    assert "CAI MAE" in svg
    assert "Mean exact acquired fraction" in svg
    assert "<text" in svg
    assert all(line == line.rstrip() for line in svg.splitlines())
    pixels = np.asarray(Image.open(tmp_path / "mris_cai_mae_vs_exact_cost.png").convert("RGB"))
    assert pixels.std() > 5.0

    replay = tmp_path / "replay"
    replay_outputs = render_mris_cost_curve(metrics, output_root=replay)
    assert [path.read_bytes() for path in outputs] == [
        path.read_bytes() for path in replay_outputs
    ]
