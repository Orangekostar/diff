"""Preselected-case visual diagnostics for the CAI active-image v2 study."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from cmc_bbdm.learned_cscan.runtime import (
    load_study_config,
    load_study_context,
    render_registered_surface,
)

from .contracts import Method
from .environment import NativeCellGrid
from .features import FeatureBank
from .protocol import CAIActiveImageProtocol


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _draw_grid(axis: object, shape: tuple[int, int], *, color: str = "white") -> None:
    height, width = shape
    for fraction in np.linspace(0.0, 1.0, 9):
        axis.axhline(fraction * height - 0.5, color=color, linewidth=0.45, alpha=0.65)
        axis.axvline(fraction * width - 0.5, color=color, linewidth=0.45, alpha=0.65)


def _draw_cells(
    axis: object,
    grid: NativeCellGrid,
    cells: list[int],
    *,
    color: str,
    fill: bool,
    linewidth: float = 2.0,
) -> None:
    for index in cells:
        cell = grid.cells[index]
        axis.add_patch(
            Rectangle(
                (cell.col_start - 0.5, cell.row_start - 0.5),
                cell.col_stop - cell.col_start,
                cell.row_stop - cell.row_start,
                facecolor=color if fill else "none",
                edgecolor=color,
                alpha=0.2 if fill else 1.0,
                linewidth=linewidth,
            )
        )


def _measured_image(
    full_scan: np.ndarray, grid: NativeCellGrid, cells: list[int]
) -> np.ndarray:
    result = np.full_like(full_scan, 238)
    for index in cells:
        cell = grid.cells[index]
        result[cell.row_start : cell.row_stop, cell.col_start : cell.col_stop] = (
            full_scan[cell.row_start : cell.row_stop, cell.col_start : cell.col_stop]
        )
    return result


def generate_preselected_figures(
    protocol: CAIActiveImageProtocol,
    *,
    project_root: str | Path,
    source_root: str | Path,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    bank = FeatureBank.load(protocol.results_dir / "feature_bank.npz")
    trajectories = _rows(protocol.results_dir / "trajectories.csv")
    episode_rows = _rows(protocol.results_dir / "test_episodes.csv")
    summary = json.loads((protocol.results_dir / "summary.json").read_text(encoding="utf-8"))
    study_config = load_study_config(
        protocol.source("learned_cscan_config"), project_root=root
    )
    context = load_study_context(
        study_config,
        source_root=source_root,
        verify_pilot_hashes=False,
    )
    records = {
        assignment.record.specimen_key: assignment.record
        for assignment in context.roster.assignments
    }
    figure_dir = protocol.results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for specimen_key in protocol.preselected_cases:
        if specimen_key not in bank.specimen_keys:
            raise ValueError("preselected figure case left the frozen cohort")
        index = bank.specimen_keys.index(specimen_key)
        if bank.splits[index] != "TEST":
            raise ValueError("preselected figure case is not TEST")
        record = records[specimen_key]
        registered = render_registered_surface(
            record,
            max_edge=int(context.config.legacy_config.values["surface"]["max_edge"]),
        )
        surface = np.asarray(registered.render.clean, dtype=np.uint8)
        full_scan = np.asarray(
            context.authority.source_teacher_view(record.specimen_id).full_scan,
            dtype=np.uint8,
        )
        grid = NativeCellGrid.from_shape(tuple(int(value) for value in full_scan.shape[:2]))
        main_rows = sorted(
            [
                row
                for row in trajectories
                if row["specimen_key"] == specimen_key
                and row["method"] == Method.VLM_CAI_FEEDBACK_AGENT.value
                and int(row["seed"]) == protocol.seeds[0]
            ],
            key=lambda row: int(row["actor_call_index"]),
        )
        if not main_rows:
            raise ValueError("preselected figure trajectory is missing")
        actions = [int(row["next_cell"]) for row in main_rows]
        candidates = [
            int(value) for value in str(main_rows[0]["initial_candidates"]).split(";") if value
        ]

        figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        axes[0, 0].imshow(surface)
        surface_grid = NativeCellGrid.from_shape(surface.shape[:2])
        _draw_grid(axes[0, 0], surface.shape[:2])
        _draw_cells(axes[0, 0], surface_grid, candidates, color="#00a6a6", fill=True)
        axes[0, 0].set_title("Registered surface + frozen VLM prior")

        axes[0, 1].imshow(surface)
        _draw_grid(axes[0, 1], surface.shape[:2])
        _draw_cells(axes[0, 1], surface_grid, [actions[0]], color="#d62728", fill=True)
        axes[0, 1].set_title(f"First acquired cell: {actions[0]}")

        state_counts = [min(4, len(actions)), min(8, len(actions)), len(actions)]
        for axis, count in zip((axes[0, 2], axes[1, 0], axes[1, 1]), state_counts, strict=True):
            acquired = actions[:count]
            axis.imshow(_measured_image(full_scan, grid, acquired))
            _draw_grid(axis, full_scan.shape[:2], color="#555555")
            _draw_cells(axis, grid, acquired, color="#2ca02c", fill=False, linewidth=1.0)
            if count < len(actions):
                _draw_cells(axis, grid, [actions[count]], color="#d62728", fill=False, linewidth=2.5)
                suffix = f"; next={actions[count]}"
            else:
                suffix = "; endpoint"
            cost = float(main_rows[count - 1]["after_cost"])
            axis.set_title(f"Measured C-scan after {count} cells, cost={cost:.3f}{suffix}")

        curve_axis = axes[1, 2]
        comparison_methods = (
            Method.VLM_CAI_FEEDBACK_AGENT.value,
            Method.NO_VLM_FEEDBACK.value,
            Method.VLM_OPEN_LOOP.value,
            str(summary["best_nonadaptive"]),
        )
        colors = ("#d62728", "#1f77b4", "#9467bd", "#2ca02c")
        for method, color in zip(comparison_methods, colors, strict=True):
            rows = sorted(
                [
                    row
                    for row in trajectories
                    if row["specimen_key"] == specimen_key
                    and row["method"] == method
                    and int(row["seed"]) in {0, protocol.seeds[0]}
                ],
                key=lambda row: int(row["actor_call_index"]),
            )
            if not rows:
                continue
            costs = [0.0] + [float(row["after_cost"]) for row in rows]
            predictions = [float(rows[0]["prediction_before_mpa"])] + [
                float(row["prediction_mpa"]) for row in rows
            ]
            curve_axis.plot(costs, predictions, color=color, linewidth=1.8, label=method)
        target = next(
            float(row["target_mpa"])
            for row in episode_rows
            if row["specimen_key"] == specimen_key
        )
        curve_axis.axhline(target, color="black", linestyle="--", linewidth=1.2, label="true CAI MPa")
        curve_axis.set_xlabel("Exact native-raster acquisition cost")
        curve_axis.set_ylabel("Predicted CAI (MPa)")
        curve_axis.set_title("Prediction trajectory")
        curve_axis.legend(fontsize=6, loc="best")
        curve_axis.grid(alpha=0.2)
        for axis in axes.flat[:5]:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(f"CAI active-image diagnostic: {specimen_key}", fontsize=15)
        output = figure_dir / f"case_{specimen_key.replace(':', '_')}.png"
        figure.savefig(output, dpi=180, facecolor="white")
        plt.close(figure)
        with Image.open(output) as exported:
            pixel_size = list(exported.size)
        manifest_rows.append(
            {
                "specimen_key": specimen_key,
                "split": "TEST",
                "selection_rule": "preselected before TEST evaluation in frozen v2 config",
                "file": str(output.relative_to(root)),
                "sha256": _sha256_file(output),
                "pixel_size": pixel_size,
                "full_cscan_displayed": False,
                "vlm_prior_cells": candidates,
                "main_seed": protocol.seeds[0],
            }
        )
    manifest = {
        "schema_version": 2,
        "case_count": len(manifest_rows),
        "selection_frozen_before_test": True,
        "maximum_allowed_cases": 3,
        "records": manifest_rows,
    }
    path = protocol.results_dir / "figure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = ["generate_preselected_figures"]
