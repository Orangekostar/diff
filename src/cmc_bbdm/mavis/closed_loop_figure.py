"""Publication-grade E5 task-specificity figure for closed-loop MAVIS."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


class MAVISClosedLoopFigureError(ValueError):
    """Raised when registered P4 curves cannot be rendered."""


_STYLE = {
    "uniform": ("Uniform", "#666666", "-.", "D"),
    "reconstruction_driven": ("Reconstruction-driven", "#009E73", "--", "s"),
    "mavis_no_feedback": ("MAVIS frozen ranking", "#E69F00", ":", "v"),
    "mavis_positions_only": ("MAVIS positions only", "#CC79A7", (0, (5, 2)), "P"),
    "mavis_shuffled_content": ("MAVIS shuffled content", "#D55E00", "--", "^"),
    "mavis_full": ("MAVIS full", "#0072B2", "-", "o"),
}


def render_task_specificity_curve(
    aggregate_curves: pl.DataFrame,
    *,
    output_root: str | Path,
) -> tuple[Path, Path, Path]:
    required = {
        "method",
        "nominal_checkpoint",
        "domain_count",
        "mean_exact_acquired_cost",
        "mean_effective_budget",
        "domain_balanced_cai_mae",
        "worst_domain_cai_mae",
        "domain_balanced_reconstruction_mse",
    }
    if (
        not isinstance(aggregate_curves, pl.DataFrame)
        or aggregate_curves.height == 0
        or not required <= set(aggregate_curves.columns)
    ):
        raise MAVISClosedLoopFigureError("P4 aggregate curves are invalid")
    selected = aggregate_curves.filter(pl.col("method").is_in(_STYLE))
    if (
        set(selected.get_column("method").unique()) != set(_STYLE)
        or selected.select(pl.any_horizontal(pl.selectors.numeric().is_nan()))
        .to_series()
        .any()
    ):
        raise MAVISClosedLoopFigureError("P4 figure method roster is invalid")
    checkpoints = {
        method: tuple(
            selected.filter(pl.col("method") == method)
            .sort("nominal_checkpoint")
            .get_column("nominal_checkpoint")
            .to_list()
        )
        for method in _STYLE
    }
    if len(set(checkpoints.values())) != 1 or len(next(iter(checkpoints.values()))) < 2:
        raise MAVISClosedLoopFigureError("P4 curve checkpoint roster changed")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.0,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "mavis-p4-closed-loop",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.55),
        sharex=True,
        constrained_layout=True,
    )
    metrics = (
        ("domain_balanced_cai_mae", "CAI MAE (domain-balanced)"),
        (
            "domain_balanced_reconstruction_mse",
            "Reconstruction MSE (domain-balanced)",
        ),
    )
    for method, (label, color, linestyle, marker) in _STYLE.items():
        table = selected.filter(pl.col("method") == method).sort("nominal_checkpoint")
        x = table.get_column("mean_effective_budget").to_numpy()
        if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
            raise MAVISClosedLoopFigureError("P4 exact-cost curve is invalid")
        for axis, (column, _) in zip(axes, metrics, strict=True):
            y = table.get_column(column).to_numpy()
            if not np.all(np.isfinite(y)) or np.any(y < 0.0):
                raise MAVISClosedLoopFigureError("P4 metric curve is invalid")
            axis.plot(
                x,
                y,
                color=color,
                label=label,
                linestyle=linestyle,
                marker=marker,
                markeredgewidth=0.9,
                markerfacecolor=color if method == "mavis_full" else "white",
                markersize=4.0,
                zorder=5 if method == "mavis_full" else 3,
            )
    for index, (axis, (_, ylabel)) in enumerate(zip(axes, metrics, strict=True)):
        axis.set_xlabel("Mean exact acquired fraction")
        axis.set_ylabel(ylabel)
        axis.set_title(f"({'ab'[index]}) {'Mechanical assessment' if index == 0 else 'C-scan reconstruction'}")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.8)
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
        axis.margins(x=0.025, y=0.1)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        loc="lower center",
        ncol=3,
        columnspacing=1.3,
        handlelength=2.5,
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = root / "task_specificity_vs_exact_cost"
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
        raise MAVISClosedLoopFigureError("P4 figure export failed")
    return svg, pdf, png


__all__ = [
    "MAVISClosedLoopFigureError",
    "render_task_specificity_curve",
]
