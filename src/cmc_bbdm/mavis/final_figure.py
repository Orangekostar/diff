"""Deterministic final claim-evidence figure for frozen MAVIS evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


class MAVISFinalFigureError(ValueError):
    """Raised when final claim evidence cannot be rendered."""


_FIXED_STYLE = {
    "mavis_no_aggregation": ("MAVIS before aggregation", "#CC79A7", "--", "s"),
    "mavis_full": ("MAVIS aggregated", "#0072B2", "-", "o"),
    "mavis_safe": ("MAVIS safe", "#009E73", "-", "D"),
    "sequential_mechanical_oracle": (
        "Sequential oracle",
        "#D55E00",
        ":",
        "v",
    ),
}


def render_final_claim_figure(
    aggregate_curves: pl.DataFrame,
    domain_effects: pl.DataFrame,
    *,
    strongest_baseline: str,
    domain_order: tuple[str, ...],
    output_root: str | Path,
) -> tuple[Path, Path, Path]:
    curve_required = {
        "method",
        "nominal_checkpoint",
        "mean_effective_budget",
        "domain_balanced_cai_mae",
    }
    effect_required = {
        "outer_domain",
        "contrast",
        "control_minus_reference_cai_auebc",
    }
    styles = {
        strongest_baseline: ("Strongest deployable baseline", "#666666", "-.", "^"),
        **_FIXED_STYLE,
    }
    if (
        not isinstance(aggregate_curves, pl.DataFrame)
        or not isinstance(domain_effects, pl.DataFrame)
        or not curve_required <= set(aggregate_curves.columns)
        or not effect_required <= set(domain_effects.columns)
        or type(strongest_baseline) is not str
        or not strongest_baseline
        or type(domain_order) is not tuple
        or not domain_order
        or len(set(domain_order)) != len(domain_order)
    ):
        raise MAVISFinalFigureError("final figure input is invalid")
    curves = aggregate_curves.filter(pl.col("method").is_in(styles))
    if (
        set(curves.get_column("method").unique()) != set(styles)
        or set(domain_effects.get_column("outer_domain").unique()) != set(domain_order)
        or set(domain_effects.get_column("contrast").unique())
        != {"baseline_minus_mavis", "fallback_minus_safe"}
        or domain_effects.unique(subset=["outer_domain", "contrast"]).height
        != len(domain_order) * 2
        or curves.select(pl.any_horizontal(pl.selectors.numeric().is_nan()))
        .to_series()
        .any()
        or domain_effects.select(pl.any_horizontal(pl.selectors.numeric().is_nan()))
        .to_series()
        .any()
    ):
        raise MAVISFinalFigureError("final figure roster is invalid")
    checkpoint_rosters = {
        method: tuple(
            curves.filter(pl.col("method") == method)
            .sort("nominal_checkpoint")
            .get_column("nominal_checkpoint")
            .to_list()
        )
        for method in styles
    }
    if len(set(checkpoint_rosters.values())) != 1:
        raise MAVISFinalFigureError("final figure checkpoint roster changed")

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
            "svg.hashsalt": "mavis-p7-final-claim",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.15, 1.0)},
    )
    for method, (label, color, linestyle, marker) in styles.items():
        table = curves.filter(pl.col("method") == method).sort("nominal_checkpoint")
        x = table.get_column("mean_effective_budget").to_numpy()
        y = table.get_column("domain_balanced_cai_mae").to_numpy()
        if (
            not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or np.any(np.diff(x) <= 0.0)
            or np.any(y < 0.0)
        ):
            raise MAVISFinalFigureError("final claim curve is invalid")
        axes[0].plot(
            x,
            y,
            color=color,
            label=label,
            linestyle=linestyle,
            marker=marker,
            markeredgewidth=0.9,
            markerfacecolor=color if method in {"mavis_full", "mavis_safe"} else "white",
            markersize=4.0,
            zorder=5 if method in {"mavis_full", "mavis_safe"} else 3,
        )
    axes[0].set_xlabel("Mean exact acquired fraction")
    axes[0].set_ylabel("CAI MAE (domain-balanced)")
    axes[0].set_title("(a) Frozen closed-loop evaluation")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.8)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["top"].set_visible(False)
    axes[0].margins(x=0.025, y=0.1)

    positions = np.arange(len(domain_order), dtype=np.float64)
    width = 0.34
    contrast_styles = (
        ("baseline_minus_mavis", "Baseline - MAVIS", "#0072B2", -width / 2),
        ("fallback_minus_safe", "Fallback - safe", "#009E73", width / 2),
    )
    for contrast, label, color, offset in contrast_styles:
        table = domain_effects.filter(pl.col("contrast") == contrast)
        lookup = dict(
            table.select(
                "outer_domain", "control_minus_reference_cai_auebc"
            ).iter_rows()
        )
        values = np.asarray([lookup[domain] for domain in domain_order], dtype=np.float64)
        axes[1].bar(
            positions + offset,
            values,
            width=width,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(positions, [f"D{index + 1}" for index in range(len(domain_order))])
    axes[1].set_xlabel("Held-out domain")
    axes[1].set_ylabel("Control - MAVIS CAI AUEBC")
    axes[1].set_title("(b) Paired domain effects")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.8)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    axes[1].margins(y=0.15)

    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    figure.legend(
        [*handles0, *handles1],
        [*labels0, *labels1],
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        loc="lower center",
        ncol=4,
        columnspacing=1.0,
        handlelength=2.2,
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = root / "final_claim_curves_and_domains"
    svg = stem.with_suffix(".svg")
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fixed_time = datetime(2026, 8, 26, tzinfo=timezone.utc)
    figure.savefig(
        svg,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": "2026-08-26T00:00:00Z"},
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
        metadata={"Software": "MAVIS deterministic final figure export"},
    )
    plt.close(figure)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in (svg, pdf, png)):
        raise MAVISFinalFigureError("final figure export failed")
    return svg, pdf, png


__all__ = ["MAVISFinalFigureError", "render_final_claim_figure"]
