"""Publication-grade, evidence-bound figures for formal MVA A5."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .a5_config import load_a5_config
from .a5_evaluation import AGGREGATED_METHODS


class A5FigureError(ValueError):
    """Raised when A5 figure evidence is incomplete or inconsistent."""


STYLE = {
    "center_first": ("#E69F00", "^", "--"),
    "observed_gradient": ("#009E73", "s", "-."),
    "observed_uncertainty": ("#56B4E9", "v", ":"),
    "imitation_policy": ("#0072B2", "o", "-"),
    "uniform": ("#666666", "D", "--"),
    "random_median": ("#CC79A7", "x", ":"),
    "global_mechanical_mask": ("#D55E00", "P", "-"),
    "mechanical_oracle": ("#000000", "*", (0, (6, 2))),
}
LABEL = {
    "center_first": "Center-first",
    "observed_gradient": "Observed gradient",
    "observed_uncertainty": "Observed uncertainty",
    "imitation_policy": "Imitation policy",
    "uniform": "Uniform",
    "random_median": "Random median",
    "global_mechanical_mask": "Global mechanical",
    "mechanical_oracle": "Mechanical oracle",
}
_SOURCE_COLUMNS = (
    "figure_id",
    "panel",
    "method",
    "outer_domain",
    "epoch",
    "nominal_checkpoint",
    "x_value",
    "y_value",
    "y_lower",
    "y_upper",
    "value_name",
)


def _read(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise A5FigureError(f"A5 figure source cannot be read: {path.name}") from error


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(
        path.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(figure)


def _source_row(
    *,
    figure_id: str,
    panel: str,
    method: str | None,
    outer_domain: str | None,
    epoch: int | None,
    checkpoint: float | None,
    x: float,
    y: float,
    lower: float | None,
    upper: float | None,
    value_name: str,
) -> dict[str, object]:
    return dict(
        zip(
            _SOURCE_COLUMNS,
            (
                figure_id,
                panel,
                method,
                outer_domain,
                epoch,
                checkpoint,
                x,
                y,
                lower,
                upper,
                value_name,
            ),
            strict=True,
        )
    )


def render_a5_figures(
    evidence_dir: str | Path,
    *,
    config_path: str | Path,
    project_root: str | Path,
) -> Path:
    """Render the three registered A5 figures and their complete source table."""

    evidence = Path(evidence_dir).resolve(strict=True)
    config = load_a5_config(config_path, project_root=project_root)
    curves = _read(evidence / "cai_curves.csv")
    domains = _read(evidence / "domain_metrics.csv")
    budgets = _read(evidence / "budget_metrics.csv")
    training = _read(evidence / "policy_training.csv")
    primary = curves.filter(pl.col("protocol") == "P-B").sort(
        ["method", "nominal_checkpoint"]
    )
    if (
        primary.height != len(AGGREGATED_METHODS) * len(config.checkpoints)
        or set(primary["method"]) != set(AGGREGATED_METHODS)
        or set(primary["nominal_checkpoint"]) != set(config.checkpoints)
        or training.height != len(config.domain_order) * config.epochs
        or set(training["outer_domain"]) != set(config.domain_order)
    ):
        raise A5FigureError("A5 figure evidence roster changed")
    output = evidence / "figures"
    output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "mva-a5-formal",
        }
    )
    source: list[dict[str, object]] = []

    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for method in AGGREGATED_METHODS:
        selected = primary.filter(pl.col("method") == method).sort(
            "nominal_checkpoint"
        )
        x = selected["nominal_checkpoint"].to_numpy() * 100.0
        y = selected["equal_domain_mae"].to_numpy()
        color, marker, line = STYLE[method]
        width = 2.4 if method == "imitation_policy" else 1.45
        zorder = 5 if method == "imitation_policy" else 3
        axis.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle=line,
            linewidth=width,
            markersize=5.5,
            label=LABEL[method],
            zorder=zorder,
        )
        if method == "random_median":
            lower = selected["mae_q05"].to_numpy()
            upper = selected["mae_q95"].to_numpy()
            axis.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
        else:
            lower = np.full(len(x), np.nan)
            upper = np.full(len(x), np.nan)
        for index, checkpoint in enumerate(selected["nominal_checkpoint"]):
            source.append(
                _source_row(
                    figure_id="A5_error_budget",
                    panel="P-B",
                    method=method,
                    outer_domain=None,
                    epoch=None,
                    checkpoint=float(checkpoint),
                    x=float(x[index]),
                    y=float(y[index]),
                    lower=(float(lower[index]) if np.isfinite(lower[index]) else None),
                    upper=(float(upper[index]) if np.isfinite(upper[index]) else None),
                    value_name="equal_domain_mae",
                )
            )
    axis.set_xlabel("Simulated measured locations (%)")
    axis.set_ylabel("Equal-domain CAI MAE (P-B)")
    axis.set_xticks(np.asarray(config.checkpoints) * 100.0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.legend(ncol=2, frameon=False, fontsize=8, loc="best")
    _save(figure, output / "A5_error_budget")

    global_rows = {
        str(row["dataset_id"]): float(row["auebc"])
        for row in domains.filter(pl.col("method") == "global_mechanical_mask").iter_rows(named=True)
    }
    policy_rows = {
        str(row["dataset_id"]): float(row["auebc"])
        for row in domains.filter(pl.col("method") == "imitation_policy").iter_rows(named=True)
    }
    effects = np.asarray(
        [global_rows[domain] - policy_rows[domain] for domain in config.domain_order]
    )
    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    positions = np.arange(len(config.domain_order))
    colors = np.where(effects > 0.0, "#0072B2", "#D55E00")
    axis.bar(positions, effects, color=colors, width=0.68)
    axis.axhline(0.0, color="#333333", linewidth=0.9)
    axis.set_xticks(positions, [f"D{index + 1}" for index in positions])
    axis.set_ylabel("Global - policy AUEBC")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    for index, domain in enumerate(config.domain_order):
        source.append(
            _source_row(
                figure_id="A5_domain_effects",
                panel="domain",
                method="imitation_policy",
                outer_domain=domain,
                epoch=None,
                checkpoint=None,
                x=float(index + 1),
                y=float(effects[index]),
                lower=None,
                upper=None,
                value_name="global_minus_policy_auebc",
            )
        )
    _save(figure, output / "A5_domain_effects")

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    for index, domain in enumerate(config.domain_order):
        selected = training.filter(pl.col("outer_domain") == domain).sort("epoch")
        epochs = selected["epoch"].to_numpy()
        losses = selected["weighted_pairwise_loss"].to_numpy()
        axes[0].plot(epochs, losses, linewidth=1.35, label=f"D{index + 1}")
        for epoch, loss in zip(epochs, losses, strict=True):
            source.append(
                _source_row(
                    figure_id="A5_training_gap",
                    panel="training",
                    method="imitation_policy",
                    outer_domain=domain,
                    epoch=int(epoch),
                    checkpoint=None,
                    x=float(epoch),
                    y=float(loss),
                    lower=None,
                    upper=None,
                    value_name="weighted_pairwise_loss",
                )
            )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted pairwise loss")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axes[0].legend(ncol=2, frameon=False, fontsize=7)
    area_rows = {
        str(row["method"]): float(row["auebc"])
        for row in budgets.iter_rows(named=True)
    }
    methods = ("global_mechanical_mask", "imitation_policy", "mechanical_oracle")
    values = [area_rows[method] for method in methods]
    axes[1].bar(
        np.arange(3),
        values,
        color=[STYLE[method][0] for method in methods],
        width=0.65,
    )
    axes[1].set_xticks(np.arange(3), ["Global", "Policy", "Oracle"])
    axes[1].set_ylabel("P-B AUEBC")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.7)
    for index, (method, value) in enumerate(zip(methods, values, strict=True)):
        source.append(
            _source_row(
                figure_id="A5_training_gap",
                panel="oracle_gap",
                method=method,
                outer_domain=None,
                epoch=None,
                checkpoint=None,
                x=float(index),
                y=float(value),
                lower=None,
                upper=None,
                value_name="auebc",
            )
        )
    _save(figure, output / "A5_training_gap")
    pl.DataFrame(source, schema=list(_SOURCE_COLUMNS), orient="row").write_csv(
        output / "source_data.csv"
    )
    return output


__all__ = ["A5FigureError", "render_a5_figures"]
