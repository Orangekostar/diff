"""Publication-grade E2 MRIS CAI error versus exact-cost figure."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


class MAVISMRISFigureError(ValueError):
    """Raised when the E2 curve cannot be rendered from registered metrics."""


_STYLE = {
    "static": ("Static", "#666666", "-.", "D"),
    "positions_only": ("Positions only", "#E69F00", "--", "s"),
    "shuffled": ("Shuffled content", "#D55E00", ":", "^"),
    "reconstruction": ("Reconstruction", "#009E73", (0, (5, 2)), "X"),
    "real": ("Real measured state", "#0072B2", "-", "o"),
}


def render_mris_cost_curve(
    aggregate_metrics: pl.DataFrame,
    *,
    output_root: str | Path,
) -> tuple[Path, Path, Path]:
    required = {
        "mode",
        "nominal_checkpoint",
        "domain_count",
        "mean_exact_acquired_cost",
        "mean_effective_budget",
        "domain_balanced_mae",
        "worst_domain_mae",
    }
    if (
        not isinstance(aggregate_metrics, pl.DataFrame)
        or aggregate_metrics.height == 0
        or not required <= set(aggregate_metrics.columns)
        or set(aggregate_metrics.get_column("mode").unique()) != set(_STYLE)
        or aggregate_metrics.select(pl.selectors.numeric()).select(
            pl.any_horizontal(pl.all().is_nan().any())
        ).item()
    ):
        raise MAVISMRISFigureError("MRIS aggregate metrics are invalid")
    checkpoints = {
        mode: tuple(
            aggregate_metrics.filter(pl.col("mode") == mode)
            .sort("nominal_checkpoint")
            .get_column("nominal_checkpoint")
            .to_list()
        )
        for mode in _STYLE
    }
    if len(set(checkpoints.values())) != 1 or len(next(iter(checkpoints.values()))) < 2:
        raise MAVISMRISFigureError("MRIS curve checkpoint roster changed")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "svg.fonttype": "none",
            "svg.hashsalt": "mavis-p2-mris",
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.1, 3.65), constrained_layout=True)
    for mode, (label, color, linestyle, marker) in _STYLE.items():
        table = aggregate_metrics.filter(pl.col("mode") == mode).sort(
            "nominal_checkpoint"
        )
        x = table.get_column("mean_effective_budget").to_numpy()
        y = table.get_column("domain_balanced_mae").to_numpy()
        if (
            not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or np.any(np.diff(x) <= 0.0)
            or np.any(y < 0.0)
        ):
            raise MAVISMRISFigureError("MRIS curve values are invalid")
        axis.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.2,
            markerfacecolor="white" if mode != "real" else color,
            markeredgewidth=0.9,
            zorder=5 if mode == "real" else 3,
        )
    axis.set_xlabel("Mean exact acquired fraction")
    axis.set_ylabel("CAI MAE (domain-balanced)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.margins(x=0.025, y=0.1)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.2,
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = root / "mris_cai_mae_vs_exact_cost"
    svg = stem.with_suffix(".svg")
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fixed_time = datetime(2026, 8, 25, tzinfo=timezone.utc)
    figure.savefig(
        svg,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": "2026-08-25T00:00:00Z"},
    )
    figure.savefig(
        pdf,
        format="pdf",
        bbox_inches="tight",
        metadata={"CreationDate": fixed_time, "ModDate": fixed_time},
    )
    figure.savefig(
        png,
        format="png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "MAVIS deterministic figure export"},
    )
    plt.close(figure)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in (svg, pdf, png)):
        raise MAVISMRISFigureError("MRIS figure export failed")
    return svg, pdf, png


__all__ = [
    "MAVISMRISFigureError",
    "render_mris_cost_curve",
]
