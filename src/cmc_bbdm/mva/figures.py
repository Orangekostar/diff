"""Evidence-bound rendering for the formal MVA A2 figure package."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap
from scipy.stats import rankdata

from .acquisition_grid import build_acquisition_grid
from .authority import load_mva_authority
from .config import load_mva_config
from .interpolation import reconstruct_measurement_state
from .measurement_state import initial_state, measurement_mask
from .oracle_execution import _selected_budgets

REPRESENTATIVE_SPECIMEN = "c8-2"
METHOD_STYLE = {
    "uniform": ("#666666", "o", "-"),
    "random": ("#56B4E9", "s", "--"),
    "appearance_oracle": ("#E69F00", "^", "-."),
    "reconstruction_oracle": ("#009E73", "D", ":"),
    "mechanical_oracle": ("#D55E00", "P", "-"),
}
METHOD_LABEL = {
    "uniform": "Uniform",
    "random": "Random",
    "appearance_oracle": "Appearance-first",
    "reconstruction_oracle": "Reconstruction oracle",
    "mechanical_oracle": "CAI oracle",
}
_SOURCE_FIELDS = (
    "artifact",
    "specimen_id",
    "dataset_id",
    "method",
    "protocol",
    "cell_index",
    "cell_row",
    "cell_column",
    "step",
    "nominal_checkpoint",
    "effective_budget",
    "value",
    "value_percentile",
    "mae",
    "q05",
    "q95",
    "native_row",
    "native_column",
    "red",
    "green",
    "blue",
)


def percentile_map(values: Mapping[int, float]) -> np.ndarray:
    """Return an 8x8 within-map percentile image with average tie ranks."""

    if not values or any(type(key) is not int or not 0 <= key < 64 for key in values):
        raise ValueError("cell values must use registered 8x8 indices")
    keys = sorted(values)
    scores = np.asarray([values[key] for key in keys], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("cell values must be finite")
    ranks = rankdata(scores, method="average")
    percentiles = (
        np.ones_like(ranks) if len(keys) == 1 else (ranks - 1.0) / (len(keys) - 1.0)
    )
    output = np.zeros((8, 8), dtype=np.float64)
    for key, percentile in zip(keys, percentiles, strict=True):
        output[key // 8, key % 8] = percentile
    return output


def _save(fig: matplotlib.figure.Figure, output: Path, name: str) -> None:
    fig.savefig(
        output / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "cmc_bbdm.mva.figures"},
    )
    fig.savefig(
        output / f"{name}.svg",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None, "Creator": "cmc_bbdm.mva.figures"},
    )
    plt.close(fig)


def _map_figure(values: np.ndarray, title: str):
    fig, axis = plt.subplots(figsize=(4.3, 3.8), constrained_layout=True)
    image = axis.imshow(values, cmap="cividis", vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.set_xlabel("Cell column")
    axis.set_ylabel("Cell row")
    axis.set_xticks(range(8))
    axis.set_yticks(range(8))
    axis.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.6, alpha=0.75)
    axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Within-map value percentile")
    return fig


def render_mva_figures(config_path: str | Path, *, project_root: str | Path) -> Path:
    """Render all required A2 figures solely from issued formal artifacts."""

    root = Path(project_root).resolve(strict=True)
    config = load_mva_config(config_path, project_root=root)
    authority = load_mva_authority(config, project_root=root)
    output = root / config.output_dir / "a2_oracle_value"
    states_path = output / "state_metrics.parquet"
    values_path = output / "oracle_values.parquet"
    trajectories_path = output / "oracle_trajectories.parquet"
    if not all(
        path.is_file() for path in (states_path, values_path, trajectories_path)
    ):
        raise RuntimeError("formal MVA aggregate artifacts are required for rendering")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "svg.hashsalt": "mva-a2-20260823",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    try:
        specimen_index = authority.specimen_ids.index(REPRESENTATIVE_SPECIMEN)
    except ValueError as error:
        raise RuntimeError("registered representative specimen is absent") from error
    dataset_id = authority.dataset_ids[specimen_index]
    image = authority.images[specimen_index]
    selected_budget = _selected_budgets(root)[dataset_id]
    grid = build_acquisition_grid(
        image.shape[0], image.shape[1], initial_budget=selected_budget
    )
    state = initial_state(grid)
    reconstruction = reconstruct_measurement_state(
        image,
        grid,
        state,
        interpolation="bilinear",
        specimen_id=REPRESENTATIVE_SPECIMEN,
        dataset_id=dataset_id,
    ).image
    mask = measurement_mask(grid, state)
    source_rows: list[dict[str, object]] = []
    measured_rows, measured_columns = np.nonzero(mask)
    for row, column in zip(measured_rows, measured_columns, strict=True):
        red, green, blue = image[row, column]
        source_rows.append(
            {
                "artifact": "O1",
                "specimen_id": REPRESENTATIVE_SPECIMEN,
                "dataset_id": dataset_id,
                "nominal_checkpoint": selected_budget,
                "effective_budget": float(np.mean(mask)),
                "native_row": int(row),
                "native_column": int(column),
                "red": int(red),
                "green": int(green),
                "blue": int(blue),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), constrained_layout=True)
    axes[0].imshow(reconstruction)
    axes[0].set_title("Initial sparse reconstruction")
    axes[1].imshow(mask, cmap=ListedColormap(("#F2F2F2", "#0072B2")), vmin=0, vmax=1)
    axes[1].set_title(f"Measured locations ({100.0 * np.mean(mask):.2f}%)")
    for axis in axes:
        axis.set_xlabel("Native raster column")
        axis.set_ylabel("Native raster row")
    _save(fig, figures, "O1_current_sparse_scan_mask")

    value_table = pl.read_parquet(values_path).filter(
        (pl.col("specimen_id") == REPRESENTATIVE_SPECIMEN) & (pl.col("step") == 0)
    )
    map_specs = (
        (
            "reconstruction_oracle",
            "O2_reconstruction_value_map",
            "Reconstruction-value ranking",
            "O2",
        ),
        (
            "appearance_oracle",
            "O3_appearance_value_map",
            "Appearance-value ranking",
            "O3",
        ),
        (
            "mechanical_oracle",
            "O4_mechanical_value_map",
            "CAI mechanical-value ranking",
            "O4",
        ),
    )
    for method, filename, title, artifact in map_specs:
        rows = value_table.filter(pl.col("method") == method)
        raw = {
            int(row["cell_index"]): float(row["primary_value"])
            for row in rows.iter_rows(named=True)
        }
        if len(raw) != 64:
            raise RuntimeError(f"initial value map is incomplete for {method}")
        ranked = percentile_map(raw)
        _save(_map_figure(ranked, title), figures, filename)
        for cell_index, value in sorted(raw.items()):
            source_rows.append(
                {
                    "artifact": artifact,
                    "specimen_id": REPRESENTATIVE_SPECIMEN,
                    "dataset_id": dataset_id,
                    "method": method,
                    "cell_index": cell_index,
                    "cell_row": cell_index // 8,
                    "cell_column": cell_index % 8,
                    "value": value,
                    "value_percentile": float(ranked[cell_index // 8, cell_index % 8]),
                }
            )

    trajectory_table = pl.read_parquet(trajectories_path).filter(
        (pl.col("specimen_id") == REPRESENTATIVE_SPECIMEN)
        & pl.col("method").is_in(
            ["appearance_oracle", "reconstruction_oracle", "mechanical_oracle"]
        )
    )
    order_maps: dict[str, np.ndarray] = {}
    maximum_step = 1
    for method, _, _, _ in map_specs:
        rows = trajectory_table.filter(pl.col("method") == method).sort("step")
        first_step: dict[int, int] = {}
        for row in rows.iter_rows(named=True):
            first_step.setdefault(int(row["cell_index"]), int(row["step"]) + 1)
            source_rows.append(
                {
                    "artifact": "O5",
                    "specimen_id": REPRESENTATIVE_SPECIMEN,
                    "dataset_id": dataset_id,
                    "method": method,
                    "cell_index": int(row["cell_index"]),
                    "cell_row": int(row["cell_index"]) // 8,
                    "cell_column": int(row["cell_index"]) % 8,
                    "step": int(row["step"]) + 1,
                    "nominal_checkpoint": float(row["nominal_checkpoint"]),
                }
            )
        display = np.full((8, 8), np.nan, dtype=np.float64)
        for cell_index, step in first_step.items():
            display[cell_index // 8, cell_index % 8] = step
            maximum_step = max(maximum_step, step)
        order_maps[method] = display
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), constrained_layout=True)
    last_image = None
    for axis, (method, _, _, _) in zip(axes, map_specs, strict=True):
        last_image = axis.imshow(
            order_maps[method], cmap="cividis", vmin=1, vmax=maximum_step
        )
        axis.set_title(METHOD_LABEL[method])
        axis.set_xlabel("Cell column")
        axis.set_xticks(range(8))
        axis.set_yticks(range(8))
    axes[0].set_ylabel("Cell row")
    assert last_image is not None
    colorbar = fig.colorbar(last_image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("First refinement step")
    _save(fig, figures, "O5_acquisition_trajectories")

    curve_files = {
        "uniform": "uniform_curve.csv",
        "random": "random_curve.csv",
        "appearance_oracle": "appearance_curve.csv",
        "reconstruction_oracle": "reconstruction_oracle_curve.csv",
        "mechanical_oracle": "mechanical_oracle_curve.csv",
    }
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(8.4, 4.2),
        sharey=True,
        gridspec_kw={"width_ratios": (4.2, 1.3), "wspace": 0.08},
    )
    for method, filename in curve_files.items():
        table = (
            pl.read_csv(output / filename)
            .filter(pl.col("protocol") == "P-B")
            .sort("nominal_checkpoint")
        )
        y_column = "mae_mean" if method == "random" else "equal_domain_mae"
        x = 100.0 * table["nominal_checkpoint"].to_numpy()
        y = table[y_column].to_numpy()
        color, marker, linestyle = METHOD_STYLE[method]
        low = x <= 25.0
        left.plot(
            x[low],
            y[low],
            label=METHOD_LABEL[method],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.7,
            markersize=4.5,
        )
        if method == "random":
            left.fill_between(
                x[low],
                table["mae_q05"].to_numpy()[low],
                table["mae_q95"].to_numpy()[low],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        if method == "uniform":
            anchor = x > 25.0
            right.plot(
                x[anchor],
                y[anchor],
                color=color,
                marker=marker,
                linestyle="none",
                markersize=5,
            )
        for row in table.iter_rows(named=True):
            source_rows.append(
                {
                    "artifact": "error_budget_curve",
                    "method": method,
                    "protocol": "P-B",
                    "nominal_checkpoint": float(row["nominal_checkpoint"]),
                    "effective_budget": row.get("effective_mean"),
                    "mae": float(row[y_column]),
                    "q05": row.get("mae_q05"),
                    "q95": row.get("mae_q95"),
                }
            )
    for axis in (left, right):
        axis.axhline(
            config.full_mae,
            color="#222222",
            linestyle=(0, (5, 3)),
            linewidth=1.2,
            label="FULL" if axis is left else None,
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_xlabel("Measurement fraction (%)")
    left.set_xlim(2.0, 26.0)
    left.set_xticks((3.125, 6.25, 9.375, 12.5, 18.75, 25.0))
    left.tick_params(axis="x", rotation=35)
    right.set_xlim(45.0, 102.0)
    right.set_xticks((50.0, 100.0))
    right.plot(
        [100.0],
        [config.full_mae],
        color="#222222",
        marker="*",
        linestyle="none",
        markersize=7,
    )
    left.set_ylabel("Equal-domain CAI MAE")
    left.legend(ncol=2, frameon=False, loc="best")
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    left.tick_params(right=False)
    right.tick_params(left=False, labelleft=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.2, top=0.96)
    _save(fig, figures, "error_budget_curve")
    source_rows.append(
        {
            "artifact": "error_budget_curve",
            "method": "full",
            "protocol": "P-B",
            "nominal_checkpoint": 1.0,
            "effective_budget": 1.0,
            "mae": config.full_mae,
        }
    )

    with (figures / "source_data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(_SOURCE_FIELDS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(source_rows)
    return figures


__all__ = ["REPRESENTATIVE_SPECIMEN", "percentile_map", "render_mva_figures"]
