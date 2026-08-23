"""Evidence-bound figures for the formal MVA A4 package."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .a4_config import A4Config, load_a4_config
from .config import load_mva_config


class A4FigureError(ValueError):
    """Raised when A4 figure inputs violate the frozen evidence contract."""


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
RANKING_COLUMNS = {
    "outer_domain",
    "method",
    "cell_index",
    "ranking_position",
    "cell_score",
    "mean_raw_value",
    "mean_value_per_measurement",
    "source_domains",
    "source_specimen_count",
    "source_label_state_sha256",
}
CURVE_COLUMNS = {
    "method",
    "protocol",
    "nominal_checkpoint",
    "equal_domain_mae",
    "mae_mean",
    "mae_median",
    "mae_q05",
    "mae_q95",
    "effective_mean",
    "effective_min",
    "effective_max",
    "normalized_rgb_mse",
    "ssim",
}
METHOD_STYLE = {
    "global_mechanical_mask": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "global_reconstruction_mask": {
        "color": "#009E73",
        "marker": "s",
        "linestyle": "--",
    },
    "global_appearance_mask": {
        "color": "#E69F00",
        "marker": "^",
        "linestyle": "-.",
    },
    "uniform": {"color": "#666666", "marker": "D", "linestyle": "-"},
    "random_median": {
        "color": "#CC79A7",
        "marker": "x",
        "linestyle": ":",
    },
    "mechanical_oracle": {
        "color": "#000000",
        "marker": "*",
        "linestyle": (0, (6, 2)),
    },
}
METHOD_LABEL = {
    "global_mechanical_mask": "Global mechanical",
    "global_reconstruction_mask": "Global reconstruction",
    "global_appearance_mask": "Global appearance",
    "uniform": "Uniform",
    "random_median": "Random median",
    "mechanical_oracle": "Mechanical oracle",
}
_SOURCE_COLUMNS = (
    "figure_id",
    "panel",
    "method",
    "outer_domain",
    "cell_index",
    "cell_row",
    "cell_column",
    "nominal_checkpoint",
    "x_value",
    "y_value",
    "y_lower",
    "y_upper",
    "value_name",
)


def _read_csv(path: Path, label: str) -> pl.DataFrame:
    try:
        return pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise A4FigureError(f"{label} cannot be read") from error


def _finite(table: pl.DataFrame, columns: tuple[str, ...], label: str) -> None:
    try:
        valid = all(
            bool(table.select(pl.col(column).is_finite().all()).item())
            for column in columns
        )
    except pl.exceptions.PolarsError as error:
        raise A4FigureError(f"{label} numeric schema changed") from error
    if not valid:
        raise A4FigureError(f"{label} contains nonfinite values")


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_rankings(table: pl.DataFrame, config: A4Config) -> pl.DataFrame:
    expected_rows = len(config.domain_order) * len(RANK_METHODS) * 64
    if (
        set(table.columns) != RANKING_COLUMNS
        or table.height != expected_rows
        or set(table["outer_domain"]) != set(config.domain_order)
        or set(table["method"]) != set(RANK_METHODS)
        or table.unique(subset=["outer_domain", "method", "cell_index"]).height
        != table.height
    ):
        raise A4FigureError("ranking roster changed")
    _finite(
        table,
        (
            "cell_index",
            "ranking_position",
            "cell_score",
            "mean_raw_value",
            "mean_value_per_measurement",
            "source_specimen_count",
        ),
        "ranking table",
    )
    if table.filter(
        (pl.col("cell_index") < 0)
        | (pl.col("cell_index") >= 64)
        | (pl.col("ranking_position") < 0)
        | (pl.col("ranking_position") >= 64)
        | (pl.col("cell_score") < 0.0)
        | (pl.col("cell_score") > 1.0)
        | (pl.col("source_specimen_count") <= 0)
    ).height:
        raise A4FigureError("ranking values changed")
    for outer_domain in config.domain_order:
        expected_sources = "|".join(
            domain for domain in config.domain_order if domain != outer_domain
        )
        for method in RANK_METHODS:
            selected = table.filter(
                (pl.col("outer_domain") == outer_domain)
                & (pl.col("method") == method)
            )
            cells = tuple(int(value) for value in selected["cell_index"])
            positions = tuple(int(value) for value in selected["ranking_position"])
            scores = {
                int(row["cell_index"]): float(row["cell_score"])
                for row in selected.iter_rows(named=True)
            }
            expected_order = sorted(
                range(64), key=lambda cell: (-scores[cell], cell)
            )
            observed_order = [
                cell
                for _, cell in sorted(zip(positions, cells, strict=True))
            ]
            if (
                set(cells) != set(range(64))
                or set(positions) != set(range(64))
                or observed_order != expected_order
                or set(selected["source_domains"]) != {expected_sources}
                or not all(
                    _is_sha256(value)
                    for value in selected["source_label_state_sha256"]
                )
            ):
                raise A4FigureError("ranking roster changed")
    return table.sort(["outer_domain", "method", "cell_index"])


def _validate_curves(
    table: pl.DataFrame,
    config: A4Config,
    *,
    image_only: bool,
) -> pl.DataFrame:
    methods = RANK_METHODS if image_only else CAI_METHODS
    protocols = ("P-B",) if image_only else ("P-B", "P-A")
    expected_rows = len(methods) * len(protocols) * len(config.checkpoints)
    label = "image curve roster" if image_only else "CAI curve roster"
    if (
        set(table.columns) != CURVE_COLUMNS
        or table.height != expected_rows
        or set(table["method"]) != set(methods)
        or set(table["protocol"]) != set(protocols)
        or {float(value) for value in table["nominal_checkpoint"]}
        != set(config.checkpoints)
        or table.unique(subset=["method", "protocol", "nominal_checkpoint"]).height
        != table.height
    ):
        raise A4FigureError(f"{label} changed")
    _finite(
        table,
        (
            "nominal_checkpoint",
            "equal_domain_mae",
            "effective_mean",
            "effective_min",
            "effective_max",
        ),
        label,
    )
    if table.filter(
        (pl.col("equal_domain_mae") < 0.0)
        | (pl.col("effective_min") <= 0.0)
        | (pl.col("effective_min") > pl.col("effective_mean"))
        | (pl.col("effective_mean") > pl.col("effective_max"))
    ).height:
        raise A4FigureError(f"{label} changed")
    random = table.filter(pl.col("method") == "random_median")
    deterministic = table.filter(pl.col("method") != "random_median")
    if image_only:
        image = table
    else:
        if (
            random.height != len(protocols) * len(config.checkpoints)
            or random.select(
                pl.any_horizontal(
                    pl.col("mae_q05").is_null(),
                    pl.col("mae_q95").is_null(),
                    pl.col("mae_mean").is_null(),
                    pl.col("mae_median").is_null(),
                ).any()
            ).item()
            or random.filter(
                (pl.col("mae_q05") > pl.col("equal_domain_mae"))
                | (pl.col("equal_domain_mae") > pl.col("mae_q95"))
            ).height
            or deterministic.select(
                pl.any_horizontal(
                    pl.col("mae_q05").is_not_null(),
                    pl.col("mae_q95").is_not_null(),
                    pl.col("mae_mean").is_not_null(),
                    pl.col("mae_median").is_not_null(),
                ).any()
            ).item()
        ):
            raise A4FigureError("CAI random interval changed")
        image = table.filter(pl.col("method").is_in(list(RANK_METHODS)))
    if (
        image.select(
            pl.any_horizontal(
                pl.col("normalized_rgb_mse").is_null(),
                pl.col("ssim").is_null(),
            ).any()
        ).item()
        or image.filter(
            (pl.col("normalized_rgb_mse") < 0.0)
            | (pl.col("ssim") < -1.0)
            | (pl.col("ssim") > 1.0)
        ).height
    ):
        raise A4FigureError("image metrics changed")
    _finite(image, ("normalized_rgb_mse", "ssim"), "image metrics")
    return table.sort(["protocol", "method", "nominal_checkpoint"])


def _source_row(**values: object) -> dict[str, object]:
    return {column: values.get(column) for column in _SOURCE_COLUMNS}


def _save(fig: matplotlib.figure.Figure, output: Path, name: str) -> None:
    fig.savefig(
        output / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "cmc_bbdm.mva.a4_figures"},
    )
    fig.savefig(
        output / f"{name}.svg",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None, "Creator": "cmc_bbdm.mva.a4_figures"},
    )
    plt.close(fig)


def _ranking_figure(
    rankings: pl.DataFrame,
    output: Path,
    source_rows: list[dict[str, object]],
) -> None:
    panel_specs = (
        ("global_appearance_mask", "(a) Appearance"),
        ("global_reconstruction_mask", "(b) Reconstruction"),
        ("global_mechanical_mask", "(c) Mechanical"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), constrained_layout=True)
    image = None
    for axis, (method, title) in zip(axes, panel_specs, strict=True):
        selected = rankings.filter(pl.col("method") == method)
        values = np.zeros(64, dtype=np.float64)
        for cell in range(64):
            scores = selected.filter(pl.col("cell_index") == cell)[
                "cell_score"
            ].to_numpy()
            values[cell] = float(np.mean(np.sort(scores), dtype=np.float64))
        matrix = values.reshape(8, 8)
        image = axis.imshow(matrix, cmap="cividis", vmin=0.0, vmax=1.0)
        order = sorted(range(64), key=lambda cell: (-values[cell], cell))
        positions = {cell: position + 1 for position, cell in enumerate(order)}
        for cell in range(64):
            row, column = divmod(cell, 8)
            color = "white" if values[cell] < 0.45 else "#111111"
            axis.text(
                column,
                row,
                str(positions[cell]),
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
            )
            source_rows.append(
                _source_row(
                    figure_id="A4_global_rankings",
                    panel=title[:3],
                    method=method,
                    outer_domain="consensus_6_outer_folds",
                    cell_index=cell,
                    cell_row=row,
                    cell_column=column,
                    y_value=values[cell],
                    value_name="equal_outer_fold_mean_cell_score",
                )
            )
        axis.set_title(title)
        axis.set_xlabel("Cell column")
        axis.set_xticks(range(8))
        axis.set_yticks(range(8))
        axis.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.5, alpha=0.55)
        axis.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Cell row")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Six-fold consensus cell score")
    fig.suptitle("Source-only global ranking consensus", fontsize=10.5)
    _save(fig, output, "A4_global_rankings")


def _cai_figure(
    curves: pl.DataFrame,
    output: Path,
    source_rows: list[dict[str, object]],
    *,
    full_mae: float,
) -> None:
    selected = curves.filter(pl.col("protocol") == "P-B")
    fig, axis = plt.subplots(figsize=(8.2, 4.0), constrained_layout=True)
    for method in CAI_METHODS:
        rows = selected.filter(pl.col("method") == method).sort(
            "nominal_checkpoint"
        )
        x = 100.0 * rows["nominal_checkpoint"].to_numpy()
        y = rows["equal_domain_mae"].to_numpy()
        style = METHOD_STYLE[method]
        if method == "random_median":
            lower = rows["mae_q05"].to_numpy()
            upper = rows["mae_q95"].to_numpy()
            axis.fill_between(x, lower, upper, color=style["color"], alpha=0.15)
        else:
            lower = np.full(len(x), np.nan)
            upper = np.full(len(x), np.nan)
        axis.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.7,
            markersize=4.5,
            label=METHOD_LABEL[method],
        )
        for index, checkpoint in enumerate(rows["nominal_checkpoint"]):
            source_rows.append(
                _source_row(
                    figure_id="A4_cai_error_budget",
                    panel="P-B",
                    method=method,
                    nominal_checkpoint=float(checkpoint),
                    x_value=float(x[index]),
                    y_value=float(y[index]),
                    y_lower=(
                        float(lower[index]) if method == "random_median" else None
                    ),
                    y_upper=(
                        float(upper[index]) if method == "random_median" else None
                    ),
                    value_name="equal_domain_mae",
                )
            )
    checkpoints = np.asarray(sorted(set(selected["nominal_checkpoint"])), dtype=float)
    axis.axhline(full_mae, color="#444444", linewidth=1.0, linestyle=(0, (2, 2)))
    for checkpoint in checkpoints:
        source_rows.append(
            _source_row(
                figure_id="A4_cai_error_budget",
                panel="P-B",
                method="full_measurement_reference",
                nominal_checkpoint=float(checkpoint),
                x_value=100.0 * float(checkpoint),
                y_value=full_mae,
                value_name="full_measurement_mae",
            )
        )
    axis.text(
        100.0 * checkpoints[-1],
        full_mae,
        " FULL",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )
    axis.set_xlabel("Nominal measured budget (%)")
    axis.set_ylabel("Equal-domain CAI MAE")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.legend(
        ncol=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
    )
    axis.set_title("Task error across the registered budget range")
    _save(fig, output, "A4_cai_error_budget")


def _tradeoff_figure(
    image_curves: pl.DataFrame,
    output: Path,
    source_rows: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), constrained_layout=True)
    panel_specs = (
        ("normalized_rgb_mse", "(a) Reconstruction error", "Normalized RGB MSE"),
        ("ssim", "(b) Structural similarity", "SSIM"),
    )
    for method in RANK_METHODS:
        rows = image_curves.filter(pl.col("method") == method).sort(
            "nominal_checkpoint"
        )
        checkpoints = rows["nominal_checkpoint"].to_numpy()
        x_budget = 100.0 * checkpoints
        style = METHOD_STYLE[method]
        for panel_index, (column, title, ylabel) in enumerate(panel_specs):
            y = rows[column].to_numpy()
            axes[panel_index].plot(
                x_budget,
                y,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                markersize=4.5,
                label=METHOD_LABEL[method],
            )
            for index, checkpoint in enumerate(checkpoints):
                source_rows.append(
                    _source_row(
                        figure_id="A4_image_task_tradeoff",
                        panel=title[:3],
                        method=method,
                        nominal_checkpoint=float(checkpoint),
                        x_value=float(x_budget[index]),
                        y_value=float(y[index]),
                        value_name=column,
                    )
                )
            axes[panel_index].set_title(title)
            axes[panel_index].set_xlabel("Nominal budget (%)")
            axes[panel_index].set_ylabel(ylabel)
        mse = rows["normalized_rgb_mse"].to_numpy()
        mae = rows["equal_domain_mae"].to_numpy()
        axes[2].plot(
            mse,
            mae,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.7,
            markersize=4.5,
            label=METHOD_LABEL[method],
        )
        for index, checkpoint in enumerate(checkpoints):
            source_rows.append(
                _source_row(
                    figure_id="A4_image_task_tradeoff",
                    panel="(c)",
                    method=method,
                    nominal_checkpoint=float(checkpoint),
                    x_value=float(mse[index]),
                    y_value=float(mae[index]),
                    value_name="cai_mae_vs_normalized_rgb_mse",
                )
            )
    axes[2].set_title("(c) Task-image tradeoff")
    axes[2].set_xlabel("Normalized RGB MSE")
    axes[2].set_ylabel("Equal-domain CAI MAE")
    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].legend(
        ncol=1,
        frameon=False,
        loc="best",
        handlelength=2.8,
    )
    _save(fig, output, "A4_image_task_tradeoff")


def render_a4_figures(
    evidence_dir: str | Path,
    *,
    config_path: str | Path,
    project_root: str | Path,
) -> Path:
    """Render the frozen A4 visual contract from validated aggregate tables."""

    root = Path(project_root).resolve(strict=True)
    evidence = Path(evidence_dir).resolve(strict=True)
    config = load_a4_config(config_path, project_root=root)
    base_path = root / config.sources["a0_a3_config"].path
    base_config = load_mva_config(base_path, project_root=root)
    full_mae = float(base_config.full_mae)
    if not math.isfinite(full_mae) or full_mae < 0.0:
        raise A4FigureError("FULL reference changed")
    rankings = _validate_rankings(
        _read_csv(evidence / "rankings.csv", "ranking table"), config
    )
    curves = _validate_curves(
        _read_csv(evidence / "cai_curves.csv", "CAI curve table"),
        config,
        image_only=False,
    )
    image_curves = _validate_curves(
        _read_csv(evidence / "image_curves.csv", "image curve table"),
        config,
        image_only=True,
    )
    expected_image = curves.filter(
        (pl.col("protocol") == "P-B")
        & pl.col("method").is_in(list(RANK_METHODS))
    ).sort(["method", "nominal_checkpoint"])
    observed_image = image_curves.sort(["method", "nominal_checkpoint"])
    if expected_image.to_dicts() != observed_image.to_dicts():
        raise A4FigureError("image curve evidence changed")

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "svg.fonttype": "none",
            "svg.hashsalt": "mva-a4-20260823",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    output = evidence / "figures"
    output.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, object]] = []
    _ranking_figure(rankings, output, source_rows)
    _cai_figure(curves, output, source_rows, full_mae=full_mae)
    _tradeoff_figure(image_curves, output, source_rows)
    pl.DataFrame(source_rows, infer_schema_length=None).select(
        list(_SOURCE_COLUMNS)
    ).write_csv(output / "source_data.csv")
    return output


__all__ = ["A4FigureError", "render_a4_figures"]
