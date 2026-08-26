from __future__ import annotations

from pathlib import Path

import polars as pl

from cmc_bbdm.mavis.final_figure import render_final_claim_figure


def test_final_claim_figure_exports_deterministic_vector_and_raster_files(
    tmp_path: Path,
) -> None:
    methods = (
        "mvd_m1_o2",
        "mavis_no_aggregation",
        "mavis_full",
        "mavis_safe",
        "sequential_mechanical_oracle",
    )
    curves = pl.DataFrame(
        {
            "method": [method for method in methods for _ in range(2)],
            "nominal_checkpoint": [0.1, 0.2] * len(methods),
            "mean_effective_budget": [0.1, 0.2] * len(methods),
            "domain_balanced_cai_mae": [
                value
                for index in range(len(methods))
                for value in (0.4 - index * 0.03, 0.3 - index * 0.02)
            ],
        }
    )
    effects = pl.DataFrame(
        {
            "outer_domain": ["d0", "d1", "d0", "d1"],
            "contrast": ["baseline_minus_mavis"] * 2
            + ["fallback_minus_safe"] * 2,
            "control_minus_reference_cai_auebc": [0.02, -0.01, 0.01, 0.03],
        }
    )

    outputs = render_final_claim_figure(
        curves,
        effects,
        strongest_baseline="mvd_m1_o2",
        domain_order=("d0", "d1"),
        output_root=tmp_path,
    )

    assert {path.suffix for path in outputs} == {".svg", ".pdf", ".png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    assert "2026-08-26" in outputs[0].read_text(encoding="utf-8")
