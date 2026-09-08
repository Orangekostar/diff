"""Evidence-bound figures for the BC C-scan Path-B supplement."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from PIL import Image

from .bc_supplement import load_supplement_config

matplotlib.use("Agg")
from matplotlib import pyplot as plt

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
MAGENTA = "#CC79A7"
GRAY = "#666666"
BLACK = "#222222"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.0,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(figure: object, root: Path, stem: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    png = root / f"{stem}.png"
    pdf = root / f"{stem}.pdf"
    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "cmc_bbdm BC supplement"},
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "cmc_bbdm BC supplement",
            "Producer": "matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return png, pdf


def _planner_success_cost(episodes: pl.DataFrame, root: Path) -> list[Path]:
    costs = np.asarray([0.025, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00])
    fields = (
        "success_c0025",
        "success_c005",
        "success_c010",
        "success_c020",
        "success_c040",
        "success_c060",
        "success_c080",
        "success_c100",
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for axis, task in zip(axes, ("LOCATE", "CHARACTERIZE"), strict=True):
        task_rows = episodes.filter(pl.col("task") == task)
        bc_seed_curves = []
        for method in ("BC_S1", "BC_S2", "BC_S3"):
            selected = task_rows.filter(pl.col("method") == method)
            bc_seed_curves.append(
                np.asarray([selected[field].mean() for field in fields], dtype=float)
            )
        bc_seed = np.vstack(bc_seed_curves)
        axis.fill_between(
            costs,
            bc_seed.min(axis=0),
            bc_seed.max(axis=0),
            color=BLUE,
            alpha=0.16,
            linewidth=0,
            label="BC seed range",
        )
        axis.plot(
            costs,
            bc_seed.mean(axis=0),
            color=BLUE,
            marker="o",
            label="BC (3-seed mean)",
        )
        for method, label, color, linestyle, marker in (
            ("R_BALANCED_P4", "P4 demonstrator", ORANGE, "--", "s"),
            ("R_BALANCED_P8", "P8 comparator", GRAY, ":", "^"),
        ):
            selected = task_rows.filter(pl.col("method") == method)
            axis.plot(
                costs,
                [selected[field].mean() for field in fields],
                color=color,
                linestyle=linestyle,
                marker=marker,
                label=label,
            )
        axis.set_title(task.title())
        axis.set_xlabel("Exact acquisition cost")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].set_ylabel("Current-report proxy success rate")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.subplots_adjust(top=0.77, bottom=0.18, left=0.09, right=0.98, wspace=0.10)
    return list(_save(figure, root, "figure_1_planner_success_cost"))


def _risk_value(
    risk: pl.DataFrame, task: str, planner: str, stop: str, field: str
) -> float:
    rows = risk.filter(
        (pl.col("task") == task)
        & (pl.col("planner") == planner)
        & (pl.col("stop_system") == stop)
    )
    if rows.height != 1:
        raise RuntimeError(f"risk figure row is not unique: {task}/{planner}/{stop}")
    return float(rows[field][0])


def _autonomous_path_b(risk: pl.DataFrame, root: Path) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), sharex=True)
    positions = np.asarray([0.0, 1.0])
    width = 0.34
    for row_index, task in enumerate(("LOCATE", "CHARACTERIZE")):
        for column_index, (field, title) in enumerate(
            (
                ("completion_rate", "Completion rate"),
                ("failure_penalized_cost", "Failure-penalized cost"),
            )
        ):
            axis = axes[row_index, column_index]
            for offset, stop, hatch in (
                (-width / 2, "S_RULE", "///"),
                (width / 2, "S_BC_CAL", "..."),
            ):
                p8 = _risk_value(risk, task, "R_BALANCED_P8", stop, field)
                bc = float(
                    np.mean(
                        [
                            _risk_value(risk, task, method, stop, field)
                            for method in ("BC_S1", "BC_S2", "BC_S3")
                        ]
                    )
                )
                bars = axis.bar(
                    positions + offset,
                    [p8, bc],
                    width,
                    color=[GRAY, BLUE],
                    edgecolor=BLACK,
                    linewidth=0.6,
                    hatch=hatch,
                    label=stop,
                )
                axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=6.8)
            axis.axhline(0.0, color=BLACK, linewidth=0.7)
            axis.set_ylim(0.0, 1.12)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
            axis.set_title(f"{task.title()}: {title}")
            axis.set_xticks(positions, ["P8", "BC 3-seed mean"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.subplots_adjust(top=0.91, bottom=0.09, left=0.08, right=0.98, hspace=0.38, wspace=0.20)
    return list(_save(figure, root, "figure_2_autonomous_path_b"))


def _paired_stability(
    paired: pl.DataFrame, domains: pl.DataFrame, root: Path
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for column, task in enumerate(("LOCATE", "CHARACTERIZE")):
        top = axes[0, column]
        main = paired.filter(
            (pl.col("analysis") == "H_PLAN_BC_3SEED_VS_REGISTERED_RULE")
            & (pl.col("task") == task)
            & (pl.col("metric") == "planner_ausc_any")
            & (pl.col("comparator").is_in(["R_BALANCED_P4", "R_BALANCED_P8"]))
            & (pl.col("confidence_level") == 0.95)
        ).sort("comparator")
        for y, row in enumerate(main.iter_rows(named=True)):
            top.errorbar(
                float(row["estimate"]),
                y,
                xerr=np.asarray(
                    [
                        [float(row["estimate"]) - float(row["ci_lower"])],
                        [float(row["ci_upper"]) - float(row["estimate"])],
                    ]
                ),
                fmt="o",
                color=BLUE,
                capsize=2.5,
            )
        seed_rows = paired.filter(
            (pl.col("analysis") == "H_PLAN_SEED_SPECIFIC")
            & (pl.col("task") == task)
            & (pl.col("metric") == "planner_ausc_any")
            & (pl.col("comparator") == "R_BALANCED_P8")
            & (pl.col("confidence_level") == 0.95)
        ).sort("treatment")
        top.scatter(
            seed_rows["estimate"].to_numpy(),
            np.full(seed_rows.height, 2.4),
            marker="x",
            color=ORANGE,
            label="BC seeds vs P8",
        )
        top.axvline(0.0, color=BLACK, linewidth=0.8)
        top.set_yticks([0, 1, 2.4], ["P4", "P8", "P8 seed range"])
        top.set_title(f"{task.title()}: paired BC effect")
        top.set_xlabel("BC minus rule AUSC")
        top.grid(axis="x", color="#D9D9D9", linewidth=0.6)

        bottom = axes[1, column]
        domain = domains.filter(
            (pl.col("analysis") == "H_PLAN_BC_3SEED_VS_REGISTERED_RULE")
            & (pl.col("task") == task)
            & (pl.col("metric") == "planner_ausc_any")
            & (pl.col("comparator") == "R_BALANCED_P8")
        ).sort("dataset_id")
        y = np.arange(domain.height)
        bottom.axvline(0.0, color=BLACK, linewidth=0.8)
        bottom.scatter(domain["estimate"].to_numpy(), y, color=BLUE, marker="o")
        bottom.hlines(
            y,
            domain["seed_effect_min"].to_numpy(),
            domain["seed_effect_max"].to_numpy(),
            color=GRAY,
            linewidth=1.4,
        )
        bottom.set_yticks(y, domain["dataset_id"].to_list())
        bottom.set_title(f"{task.title()}: six-domain BC minus P8")
        bottom.set_xlabel("Domain mean AUSC effect; line = seed range")
        bottom.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    figure.subplots_adjust(top=0.94, bottom=0.10, left=0.13, right=0.98, hspace=0.42, wspace=0.32)
    return list(_save(figure, root, "figure_3_paired_domain_seed_effects"))


def _ablation_figure(ablations: pl.DataFrame, root: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    for axis, task in zip(axes, ("LOCATE", "CHARACTERIZE"), strict=True):
        rows = ablations.filter(
            (pl.col("task") == task) & (pl.col("confidence_level") == 0.95)
        ).sort("ablated_actor")
        labels = [
            "Surface cues" if value == "ACTOR_SURFACE_CUES" else "US feedback"
            for value in rows["isolated_input"].to_list()
        ]
        colors = [MAGENTA, GREEN]
        for y, (row, color) in enumerate(
            zip(rows.iter_rows(named=True), colors, strict=True)
        ):
            axis.errorbar(
                float(row["estimate"]),
                y,
                xerr=np.asarray(
                    [
                        [float(row["estimate"]) - float(row["ci_lower"])],
                        [float(row["ci_upper"]) - float(row["estimate"])],
                    ]
                ),
                fmt="o",
                color=color,
                capsize=3,
            )
        axis.axvline(0.0, color=BLACK, linewidth=0.8)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(task.title())
        axis.set_xlabel("Full BC minus ablation AUSC")
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    figure.suptitle("Seed-1 Actor-input increments (proxy-scoped)", y=0.98)
    figure.subplots_adjust(top=0.82, bottom=0.20, left=0.15, right=0.98, wspace=0.22)
    return list(_save(figure, root, "figure_4_actor_input_ablations"))


def _representative_traces(
    trajectories: pl.DataFrame, manifest: dict[str, object], root: Path
) -> list[Path]:
    true_break = manifest["true_break"]["episodes"]
    selected = []
    for row in true_break:
        key = str(row["specimen_key"])
        if key not in selected:
            selected.append(key)
    if len(selected) != 6:
        raise RuntimeError("representative trace selector changed")
    figure, axes = plt.subplots(6, 2, figsize=(7.2, 10.5), sharex=True, sharey=True)
    for row_index, key in enumerate(selected):
        for column, task in enumerate(("LOCATE", "CHARACTERIZE")):
            axis = axes[row_index, column]
            for method, label, color, linestyle in (
                ("BC_S1", "BC S1", BLUE, "-"),
                ("R_BALANCED_P8", "P8", GRAY, "--"),
            ):
                rows = trajectories.filter(
                    (pl.col("specimen_key") == key)
                    & (pl.col("task") == task)
                    & (pl.col("method") == method)
                ).sort("step")
                axis.step(
                    rows["cost"].to_numpy(),
                    rows["success"].cast(pl.Int8).to_numpy(),
                    where="post",
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
                stop = rows.filter(pl.col("calibrated_stop_trigger"))
                if stop.height == 1:
                    axis.axvline(
                        float(stop["cost"][0]), color=color, linestyle=":", alpha=0.75
                    )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(-0.08, 1.08)
            axis.set_yticks([0, 1])
            axis.grid(axis="x", color="#D9D9D9", linewidth=0.5)
            if row_index == 0:
                axis.set_title(task.title())
            if column == 0:
                domain, specimen = key.split(":", maxsplit=1)
                axis.set_ylabel(f"{domain}\n{specimen}", fontsize=6.8)
            if row_index == len(selected) - 1:
                axis.set_xlabel("Exact acquisition cost")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.subplots_adjust(top=0.96, bottom=0.06, left=0.18, right=0.98, hspace=0.22, wspace=0.08)
    return list(_save(figure, root, "figure_5_six_domain_true_break_traces"))


def render_supplement_figures(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    """Render and minimally verify all result-driven supplement figures."""

    del source_root
    config = load_supplement_config(config_path, project_root=project_root)
    _style()
    episodes = pl.read_csv(config.output_root / "per_episode_metrics.csv")
    paired = pl.read_csv(config.output_root / "paired_effects.csv")
    domains = pl.read_csv(config.output_root / "per_domain_effects.csv")
    ablations = pl.read_csv(config.output_root / "ablation_effects.csv")
    risk = pl.read_csv(config.output_root / "risk_coverage.csv")
    trajectories = pl.read_parquet(config.output_root / "trajectories.parquet")
    manifest = json.loads(
        (config.output_root / "report_manifest.json").read_text(encoding="utf-8")
    )
    root = config.output_root / "figures"
    outputs = []
    outputs.extend(_planner_success_cost(episodes, root))
    outputs.extend(_autonomous_path_b(risk, root))
    outputs.extend(_paired_stability(paired, domains, root))
    outputs.extend(_ablation_figure(ablations, root))
    outputs.extend(_representative_traces(trajectories, manifest, root))
    png_checks = []
    for path in outputs:
        if path.suffix != ".png":
            continue
        with Image.open(path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
            png_checks.append(
                {
                    "path": path.relative_to(config.output_root).as_posix(),
                    "width": image.width,
                    "height": image.height,
                    "grayscale_standard_deviation": float(grayscale.std()),
                    "nonblank": bool(grayscale.std() > 2.0),
                }
            )
    if len(outputs) != 10 or not all(row["nonblank"] for row in png_checks):
        raise RuntimeError("supplement figure render QA failed")
    return {
        "stage": "SUPPLEMENT_FIGURES_RENDERED",
        "statistical_figures": 4,
        "representative_figures": 1,
        "files": [path.relative_to(config.output_root).as_posix() for path in outputs],
        "png_checks": png_checks,
    }


__all__ = ["render_supplement_figures"]
