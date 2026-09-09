"""Single reviewed-case overlay export from frozen C-scan evidence."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.vlm_cscan.references import reference_from_payload

from .benchmark import _report_digest
from .contracts import Split, Task
from .frozen_process_analysis import load_frozen_process_config
from .frozen_process_recovery import FrozenVisibleReportReader, _full_reports
from .human_review_tool import ReplayedEndpoint, replay_stored_prefix
from .runtime import load_study_config, load_study_context, open_study_specimen

EXPERT_COLOR = "#009E73"
PROXY_COLOR = "#0072B2"
BC_COLOR = "#D55E00"
MEASUREMENT_COLOR = "#E69F00"
TRAJECTORY_COLOR = "#CC79A7"
REQUIRED_OVERLAY_FILES = (
    "panel_00_base_registered_cscan.png",
    "panel_01_expert_gt_overlay.png",
    "panel_02_proxy_gt_overlay.png",
    "panel_03_bc_final_mask_overlay.png",
    "panel_04_bc_trajectory_and_mask_overlay.png",
    "panel_05_overlay_compare_expert_vs_bc.png",
    "panel_06_overlay_compare_expert_vs_proxy_vs_bc.png",
    "panel_07_overlay_boundary_only_compare.png",
    "figure_single_case_overlay_summary.png",
)


@dataclass(frozen=True, slots=True)
class SelectedCase:
    dataset_id: str
    specimen_id: str
    specimen_key: str
    task: str
    method: str
    seed: int
    stop_step: int
    stop_cost: float
    bc_report_sha256: str
    proxy_report_sha256: str
    expert_proxy_iou: float
    selection_rule: str
    selected_reason: str


@dataclass(frozen=True, slots=True)
class RecoveredCase:
    selected: SelectedCase
    reference_version: str
    registered_cscan: np.ndarray
    expert_mask: np.ndarray
    uncertain_mask: np.ndarray
    proxy_mask: np.ndarray
    bc_mask: np.ndarray
    grid: AcquisitionGrid
    endpoint: ReplayedEndpoint
    trajectory_rows: tuple[Mapping[str, object], ...]
    source_cscan_path: str
    source_image_sha256: str
    expert_reference_json: str
    proxy_report_sha256: str
    bc_report_sha256: str
    coordinate_checks: Mapping[str, object]
    identity_checks: Mapping[str, object]
    mask_metrics: Mapping[str, object]


def select_diagnostic_case(reviewed_result_root: str | Path) -> SelectedCase:
    """Select one predeclared fixed case by a reproducible diagnostic rule."""

    root = Path(reviewed_result_root)
    fixed = (
        pl.read_csv(root / "tables/fixed_case_process.csv", infer_schema_length=None)
        .filter(pl.col("method_group") == "BC_FIXED_SEED")
        .select("specimen_key", "method", "seed", "selection_rule")
        .unique()
    )
    episodes = (
        pl.read_csv(root / "reviewed/per_episode_metrics.csv", infer_schema_length=None)
        .filter(
            (pl.col("task") == "LOCATE")
            & (pl.col("stop_system") == "S_BC_CAL")
            & pl.col("stopped")
        )
        .select(
            "dataset_id",
            "specimen_id",
            "specimen_key",
            "task",
            "method",
            "seed",
            "stop_step",
            "stop_cost",
            pl.col("stop_report_sha256").alias("bc_report_sha256"),
        )
    )
    full_input = (
        pl.read_csv(root / "reviewed/full_input_readout.csv", infer_schema_length=None)
        .filter((pl.col("task") == "LOCATE") & ~pl.col("formal_success"))
        .select(
            "specimen_key",
            pl.col("report_sha256").alias("proxy_report_sha256"),
            pl.col("iou").alias("expert_proxy_iou"),
        )
    )
    eligible = (
        fixed.join(
            episodes,
            on=("specimen_key", "method", "seed"),
            how="inner",
            validate="1:1",
        )
        .join(full_input, on="specimen_key", how="inner", validate="1:1")
        .filter(pl.col("bc_report_sha256") != pl.col("proxy_report_sha256"))
        .sort("expert_proxy_iou", "specimen_key")
    )
    if eligible.height < 1:
        raise ValueError("no preselected fixed case satisfies the diagnostic rule")
    row = eligible.row(0, named=True)
    reason = (
        "Among reviewed TEST LOCATE cases in the preselected fixed case set with "
        "an actual calibrated first STOP and distinct BC/proxy report identities, "
        "this case has the lowest EXPERT_GT-vs-PROXY_GT full-input Reader IoU. "
        "It was selected for diagnostic contrast, not favorable performance."
    )
    return SelectedCase(
        dataset_id=str(row["dataset_id"]),
        specimen_id=str(row["specimen_id"]),
        specimen_key=str(row["specimen_key"]),
        task=str(row["task"]),
        method=str(row["method"]),
        seed=int(row["seed"]),
        stop_step=int(row["stop_step"]),
        stop_cost=float(row["stop_cost"]),
        bc_report_sha256=str(row["bc_report_sha256"]),
        proxy_report_sha256=str(row["proxy_report_sha256"]),
        expert_proxy_iou=float(row["expert_proxy_iou"]),
        selection_rule=str(row["selection_rule"]),
        selected_reason=reason,
    )


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pixel_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return float(intersection / union) if union else 1.0


def _find_reference(
    reference_root: Path, specimen_key: str
) -> tuple[Path, dict[str, object]]:
    matches = []
    for path in sorted(reference_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("specimen_key") == specimen_key:
            matches.append((path, payload))
    if len(matches) != 1:
        raise ValueError("selected expert reference identity is ambiguous")
    return matches[0]


def _manifest_row(path: Path, specimen_key: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["specimen_key"] == specimen_key
        ]
    if len(rows) != 1:
        raise ValueError("selected reference packet manifest identity is ambiguous")
    return rows[0]


def _html_display_pixels(
    source_path: Path, manifest: Mapping[str, str], specimen_key: str
) -> tuple[np.ndarray, dict[str, object]]:
    with Image.open(source_path) as source:
        source.load()
        image = source.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=9)
        reconstructed = buffer.getvalue()
    reconstructed_sha = hashlib.sha256(reconstructed).hexdigest()
    if reconstructed_sha != manifest["display_image_sha256"]:
        raise ValueError("reconstructed HTML display image hash changed")
    display_pixels = np.asarray(Image.open(io.BytesIO(reconstructed)).convert("RGB"))
    packet_path = Path(manifest["packet_path"])
    packet_available = packet_path.is_file()
    if packet_available:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        items = [
            item for item in packet["items"] if item["specimen_key"] == specimen_key
        ]
        if len(items) != 1:
            raise ValueError("selected HTML packet item identity is ambiguous")
        item = items[0]
        payload = base64.b64decode(str(item["image_data"]).split(",", 1)[1])
        if (
            hashlib.sha256(payload).hexdigest() != manifest["display_image_sha256"]
            or item["display_image_sha256"] != manifest["display_image_sha256"]
            or item["source_image_sha256"] != manifest["source_image_sha256"]
        ):
            raise ValueError("selected HTML packet image identity changed")
        packet_pixels = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
        if not np.array_equal(packet_pixels, display_pixels):
            raise ValueError("HTML packet pixels differ from registered source")
    return display_pixels, {
        "packet_payload_available": packet_available,
        "reconstructed_html_display_sha256_matches_manifest": True,
        "packet_display_sha256": manifest["display_image_sha256"],
    }


def recover_selected_case(
    *, project_root: str | Path, source_root: str | Path
) -> RecoveredCase:
    """Recover the selected case from stored actions and verify every identity."""

    root = Path(project_root).resolve(strict=True)
    external = Path(source_root).resolve(strict=True)
    reviewed_root = root / "results/bc_cscan_expert_pooled_rescore/v1"
    selected = select_diagnostic_case(reviewed_root)
    frozen = load_frozen_process_config(
        root / "paper_v3/configs/bc_cscan_expert_pooled_rescore_v1.yaml",
        project_root=root,
    )
    parent = load_study_config(frozen.parent_config_path, project_root=root)
    context = load_study_context(parent, source_root=external)
    assignments = [
        assignment
        for assignment in context.roster.assignments
        if assignment.record.specimen_key == selected.specimen_key
    ]
    if len(assignments) != 1 or assignments[0].split is not Split.TEST:
        raise ValueError("selected specimen is not uniquely in frozen TEST")
    record = assignments[0].record
    runtime = open_study_specimen(context, record)
    authority_view = context.authority.source_teacher_view(record.specimen_id)
    full_scan = np.asarray(authority_view.full_scan, dtype=np.uint8)

    manifest = _manifest_row(
        root / "results/cscan_human_review_html/reference_packet_manifest.csv",
        selected.specimen_key,
    )
    source_path = Path(manifest["source_image_path"]).resolve(strict=True)
    with Image.open(source_path) as source_image:
        source_image.load()
        source_pixels = np.asarray(source_image.convert("RGB"))
    html_pixels, html_checks = _html_display_pixels(
        source_path, manifest, selected.specimen_key
    )
    source_sha256 = _sha256_path(source_path)
    expected_shape = (int(manifest["native_height"]), int(manifest["native_width"]), 3)
    coordinate_checks = {
        "manifest_source_path_matches_reader_record": (
            source_path == record.cscan_path.resolve(strict=True)
        ),
        "manifest_dimensions_match": source_pixels.shape == expected_shape,
        "manifest_source_sha256_matches_file": (
            source_sha256 == manifest["source_image_sha256"]
        ),
        "reader_record_sha256_matches_file": source_sha256 == record.cscan_sha256,
        "authority_source_sha256_matches_file": (
            source_sha256 == authority_view.source_image_sha256
        ),
        "packet_source_pixels_equal": bool(np.array_equal(html_pixels, source_pixels)),
        "annotation_reader_pixels_equal": bool(np.array_equal(html_pixels, full_scan)),
        "orientation_relation": (
            "IDENTICAL_NO_TRANSFORM"
            if np.array_equal(html_pixels, full_scan)
            else "MISMATCH"
        ),
        **html_checks,
    }
    if not all(
        bool(coordinate_checks[field])
        for field in (
            "manifest_source_path_matches_reader_record",
            "manifest_dimensions_match",
            "manifest_source_sha256_matches_file",
            "reader_record_sha256_matches_file",
            "authority_source_sha256_matches_file",
            "packet_source_pixels_equal",
            "annotation_reader_pixels_equal",
        )
    ):
        raise ValueError("annotation and Reader coordinate identities differ")

    reference_root = reviewed_root / "inputs/references"
    reference_path, reference_payload = _find_reference(
        reference_root, selected.specimen_key
    )
    if reference_payload["source_image_sha256"] != source_sha256:
        raise ValueError("expert reference source image identity changed")
    reference = reference_from_payload(
        reference_payload, native_shape=record.native_shape
    )
    reference_manifest = json.loads(
        (reviewed_root / "inputs/reference_set_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    reference_version = str(reference_manifest["reference_version"])

    proxy_report = _full_reports(context, runtime, full_scan)[selected.task]
    proxy_digest = _report_digest(proxy_report)
    if proxy_digest != selected.proxy_report_sha256:
        raise ValueError("full-input proxy report identity changed")

    trajectory = (
        pl.read_parquet(frozen.source_result_root / "trajectories.parquet")
        .filter(
            (pl.col("specimen_key") == selected.specimen_key)
            & (pl.col("task") == selected.task)
            & (pl.col("method") == selected.method)
            & (pl.col("seed") == selected.seed)
        )
        .sort("step")
    )
    trajectory_rows = tuple(trajectory.to_dicts())
    if tuple(row["step"] for row in trajectory_rows) != tuple(
        range(len(trajectory_rows))
    ):
        raise ValueError("stored trajectory step order changed")
    stop_row = trajectory_rows[selected.stop_step]
    earlier_stop_triggered = any(
        bool(row["calibrated_stop_trigger"])
        for row in trajectory_rows[: selected.stop_step]
    )
    if earlier_stop_triggered or not bool(stop_row["calibrated_stop_trigger"]):
        raise ValueError("selected row is not the calibrated first STOP")
    endpoint = replay_stored_prefix(
        grid=runtime.grid,
        trajectory_rows=trajectory_rows,
        stop_step=selected.stop_step,
    )
    reader = FrozenVisibleReportReader(
        grid=runtime.grid,
        full_scan=full_scan,
        prior=context.background_prior,
        distance_threshold=float(context.config.values["reader"]["distance_threshold"]),
    )
    bc_report = reader.report(
        state=endpoint.state,
        measured_mask=endpoint.measured_mask,
        task=Task(selected.task),
    )
    bc_digest = _report_digest(bc_report)
    identity_checks = {
        "selected_split": assignments[0].split.value,
        "first_stop_trigger_matches": not earlier_stop_triggered,
        "first_stop_cost_matches": math.isclose(
            endpoint.measured_cost, selected.stop_cost, abs_tol=1e-12
        )
        and math.isclose(
            endpoint.measured_cost, float(stop_row["cost"]), abs_tol=1e-12
        ),
        "first_stop_report_sha256_match": (
            bc_digest == selected.bc_report_sha256 == stop_row["report_sha256"]
        ),
        "stored_action_count": endpoint.action_count,
        "training_updates": 0,
        "vlm_calls": 0,
        "actor_forward_calls": 0,
        "stop_forward_calls": 0,
        "world_step_count": 0,
    }
    if (
        not identity_checks["first_stop_cost_matches"]
        or not identity_checks["first_stop_report_sha256_match"]
    ):
        raise ValueError("BC autonomous first STOP identity changed")

    expert_mask = np.asarray(reference.certain_mask, dtype=np.bool_)
    proxy_mask = np.asarray(proxy_report.predicted_mask, dtype=np.bool_)
    bc_mask = np.asarray(bc_report.predicted_mask, dtype=np.bool_)
    mask_metrics = {
        "expert_pixel_count": int(np.count_nonzero(expert_mask)),
        "proxy_pixel_count": int(np.count_nonzero(proxy_mask)),
        "bc_pixel_count": int(np.count_nonzero(bc_mask)),
        "uncertain_pixel_count": int(np.count_nonzero(reference.uncertain_mask)),
        "expert_proxy_pixel_iou": _pixel_iou(expert_mask, proxy_mask),
        "expert_bc_pixel_iou": _pixel_iou(expert_mask, bc_mask),
        "proxy_bc_pixel_iou": _pixel_iou(proxy_mask, bc_mask),
        "reviewed_full_input_bbox_iou": selected.expert_proxy_iou,
    }
    return RecoveredCase(
        selected=selected,
        reference_version=reference_version,
        registered_cscan=full_scan,
        expert_mask=expert_mask,
        uncertain_mask=np.asarray(reference.uncertain_mask, dtype=np.bool_),
        proxy_mask=proxy_mask,
        bc_mask=bc_mask,
        grid=runtime.grid,
        endpoint=endpoint,
        trajectory_rows=trajectory_rows,
        source_cscan_path=("source_root:" + str(source_path.relative_to(external))),
        source_image_sha256=source_sha256,
        expert_reference_json=str(reference_path.relative_to(root)),
        proxy_report_sha256=proxy_digest,
        bc_report_sha256=bc_digest,
        coordinate_checks=coordinate_checks,
        identity_checks=identity_checks,
        mask_metrics=mask_metrics,
    )


def _plot_base(axis: object, bundle: RecoveredCase, title: str) -> None:
    height, width = bundle.registered_cscan.shape[:2]
    axis.imshow(bundle.registered_cscan, interpolation="nearest", origin="upper")
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_xlabel("Registered x (pixels)")
    axis.set_ylabel("Registered y (pixels)")
    axis.set_title(title, fontsize=11, pad=10)


def _overlay_mask(
    axis: object,
    mask: np.ndarray,
    *,
    color: str,
    alpha: float,
    linestyle: str,
    linewidth: float = 2.0,
) -> None:
    from matplotlib.colors import to_rgb

    values = np.asarray(mask, dtype=np.bool_)
    if not np.any(values):
        return
    rgba = np.zeros((*values.shape, 4), dtype=np.float32)
    rgba[values, :3] = to_rgb(color)
    rgba[values, 3] = alpha
    axis.imshow(rgba, interpolation="nearest", origin="upper")
    if linewidth > 0.0:
        axis.contour(
            values.astype(np.uint8),
            levels=(0.5,),
            colors=(color,),
            linewidths=(linewidth,),
            linestyles=(linestyle,),
            origin="upper",
        )


def _mask_handle(label: str, color: str, linestyle: str) -> object:
    from matplotlib.lines import Line2D

    return Line2D(
        (0,),
        (0,),
        color=color,
        linestyle=linestyle,
        linewidth=2.4,
        label=label,
    )


def _legend_below(axis: object, handles: tuple[object, ...], *, columns: int) -> None:
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        frameon=True,
        framealpha=0.96,
        fontsize=8.5,
        ncol=columns,
    )


def _save_panel(
    output: Path,
    filename: str,
    bundle: RecoveredCase,
    *,
    title: str,
    description: str,
    draw: object | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.4, 8.4), constrained_layout=True)
    _plot_base(axis, bundle, title)
    if draw is not None:
        draw(axis)
    figure.savefig(
        output / filename,
        dpi=240,
        facecolor="white",
        metadata={
            "Title": title,
            "Description": description,
            "Software": "cmc_bbdm single-case overlay diagnostic",
        },
    )
    plt.close(figure)


def _cell_center(bundle: RecoveredCase, cell_index: int) -> tuple[float, float]:
    from .readout import owned_cell_slices

    row_slice, column_slice = owned_cell_slices(
        bundle.grid, bundle.grid.cells[cell_index]
    )
    return (
        float((column_slice.start + column_slice.stop - 1) / 2),
        float((row_slice.start + row_slice.stop - 1) / 2),
    )


def _draw_trajectory(axis: object, bundle: RecoveredCase) -> None:
    from collections import Counter

    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Rectangle

    measured = np.asarray(bundle.endpoint.measured_mask, dtype=np.bool_)
    _overlay_mask(
        axis,
        measured,
        color="#F0E442",
        alpha=0.06,
        linestyle="solid",
        linewidth=0.0,
    )
    height, width = measured.shape
    row_lines = [float(value) - 0.5 for value in bundle.grid.row_boundaries[:-1]]
    row_lines.append(height - 0.5)
    column_lines = [float(value) - 0.5 for value in bundle.grid.column_boundaries[:-1]]
    column_lines.append(width - 0.5)
    for value in row_lines:
        axis.axhline(value, color="#222222", linewidth=0.5, alpha=0.32, zorder=5)
    for value in column_lines:
        axis.axvline(value, color="#222222", linewidth=0.5, alpha=0.32, zorder=5)
    for cell, level in zip(
        bundle.grid.cells, bundle.endpoint.state.levels, strict=True
    ):
        if level < 0:
            continue
        x0, y0 = _cell_center(bundle, cell.index)
        column_width = column_lines[cell.column + 1] - column_lines[cell.column]
        row_height = row_lines[cell.row + 1] - row_lines[cell.row]
        axis.add_patch(
            Rectangle(
                (x0 - column_width / 2, y0 - row_height / 2),
                column_width,
                row_height,
                fill=False,
                edgecolor=MEASUREMENT_COLOR,
                linewidth=0.55,
                alpha=0.30,
                zorder=6,
            )
        )

    executed = [
        row
        for row in bundle.trajectory_rows[: bundle.selected.stop_step]
        if int(row["action_cell"]) >= 0
    ]
    if len(executed) != bundle.endpoint.action_count:
        raise ValueError("stored trajectory action count changed during rendering")
    centers = [_cell_center(bundle, int(row["action_cell"])) for row in executed]
    xs = [point[0] for point in centers]
    ys = [point[1] for point in centers]
    axis.plot(
        xs,
        ys,
        color=TRAJECTORY_COLOR,
        linewidth=0.75,
        alpha=0.30,
        zorder=7,
    )
    arrow_stride = max(1, len(centers) // 18)
    for index in range(arrow_stride, len(centers), arrow_stride):
        start = centers[index - 1]
        end = centers[index]
        if start == end:
            continue
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8,
                color=TRAJECTORY_COLOR,
                linewidth=0.9,
                alpha=0.72,
                zorder=8,
            )
        )
    milestones = tuple(
        sorted(
            {0, len(centers) - 1, *(value - 1 for value in range(25, len(centers), 25))}
        )
    )
    for offset, index in enumerate(milestones):
        x_value, y_value = centers[index]
        axis.annotate(
            str(index + 1),
            xy=(x_value, y_value),
            xytext=(4 if offset % 2 == 0 else -4, 5 if offset % 2 == 0 else -5),
            textcoords="offset points",
            ha="left" if offset % 2 == 0 else "right",
            va="bottom" if offset % 2 == 0 else "top",
            fontsize=7,
            color="#111111",
            bbox={
                "boxstyle": "circle,pad=0.18",
                "facecolor": "white",
                "edgecolor": TRAJECTORY_COLOR,
                "alpha": 0.92,
            },
            zorder=10,
        )
    axis.scatter(
        (centers[0][0],),
        (centers[0][1],),
        marker="D",
        s=42,
        facecolor="white",
        edgecolor="#222222",
        linewidth=1.0,
        zorder=11,
    )
    axis.scatter(
        (centers[-1][0],),
        (centers[-1][1],),
        marker="*",
        s=110,
        facecolor=TRAJECTORY_COLOR,
        edgecolor="#222222",
        linewidth=0.8,
        zorder=11,
    )
    _overlay_mask(
        axis,
        bundle.bc_mask,
        color=BC_COLOR,
        alpha=0.08,
        linestyle="dotted",
        linewidth=1.7,
    )
    visited = Counter(int(row["action_cell"]) for row in executed)
    _legend_below(
        axis,
        (
            _mask_handle("BC_AUTONOMOUS_FINAL_MASK", BC_COLOR, "dotted"),
            _mask_handle("Measured native positions", "#B49B00", "solid"),
            _mask_handle("Measured 8x8 cells", MEASUREMENT_COLOR, "solid"),
            Line2D(
                (0,),
                (0,),
                color=TRAJECTORY_COLOR,
                marker=">",
                linewidth=1.1,
                label=f"Stored macro-actions ({sum(visited.values())})",
            ),
        ),
        columns=2,
    )


def _trajectory_summary(bundle: RecoveredCase) -> dict[str, object]:
    from collections import Counter

    executed = tuple(
        int(row["action_cell"])
        for row in bundle.trajectory_rows[: bundle.selected.stop_step]
        if int(row["action_cell"]) >= 0
    )
    quadrants = Counter()
    for cell_index in executed:
        row, column = divmod(cell_index, 8)
        vertical = "top" if row < 4 else "bottom"
        horizontal = "left" if column < 4 else "right"
        quadrants[f"{vertical}_{horizontal}"] += 1
    visits = Counter(executed)
    levels = Counter(int(level) for level in bundle.endpoint.state.levels)
    return {
        "executed_action_count": len(executed),
        "action_cells_in_order": list(executed),
        "unique_action_cells": len(visits),
        "most_visited_cells": [
            {"cell": cell, "action_count": count}
            for cell, count in sorted(
                visits.items(), key=lambda item: (-item[1], item[0])
            )[:10]
        ],
        "actions_by_grid_quadrant": {
            name: quadrants[name]
            for name in ("top_left", "top_right", "bottom_left", "bottom_right")
        },
        "final_cell_level_counts": {
            "unmeasured": levels[-1],
            "coarse": levels[0],
            "intermediate": levels[1],
            "full": levels[2],
        },
        "measured_native_pixels": bundle.endpoint.measured_count,
        "native_pixel_count": int(bundle.endpoint.measured_mask.size),
        "measured_cost": bundle.endpoint.measured_cost,
        "first_executed_action_cell": executed[0],
        "last_executed_action_cell": executed[-1],
        "stop_row_action_not_executed": int(
            bundle.trajectory_rows[bundle.selected.stop_step]["action_cell"]
        ),
    }


def _save_summary(output: Path, bundle: RecoveredCase) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), constrained_layout=True)
    panels = axes.ravel()
    titles = (
        "Registered C-scan",
        "EXPERT_GT",
        "PROXY_GT (full-input Reader target)",
        "BC_AUTONOMOUS_FINAL_MASK",
        "Stored BC trajectory + final mask",
        "Boundary-only comparison",
    )
    for axis, title in zip(panels, titles, strict=True):
        _plot_base(axis, bundle, title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(labelsize=6)
    _overlay_mask(
        panels[1], bundle.expert_mask, color=EXPERT_COLOR, alpha=0.24, linestyle="solid"
    )
    _overlay_mask(
        panels[2], bundle.proxy_mask, color=PROXY_COLOR, alpha=0.20, linestyle="dashed"
    )
    _overlay_mask(
        panels[3], bundle.bc_mask, color=BC_COLOR, alpha=0.20, linestyle="dotted"
    )
    _draw_trajectory(panels[4], bundle)
    _overlay_mask(
        panels[5], bundle.expert_mask, color=EXPERT_COLOR, alpha=0.0, linestyle="solid"
    )
    _overlay_mask(
        panels[5], bundle.proxy_mask, color=PROXY_COLOR, alpha=0.0, linestyle="dashed"
    )
    _overlay_mask(
        panels[5], bundle.bc_mask, color=BC_COLOR, alpha=0.0, linestyle="dotted"
    )
    figure.suptitle(
        f"Single reviewed TEST case | {bundle.selected.specimen_key} | "
        f"{bundle.selected.method} seed {bundle.selected.seed} | {bundle.selected.task}",
        fontsize=12,
    )
    figure.savefig(
        output / "figure_single_case_overlay_summary.png",
        dpi=180,
        facecolor="white",
        metadata={
            "Title": "Single reviewed TEST case overlay summary",
            "Description": "EXPERT_GT; PROXY_GT (full-input Reader target); BC_AUTONOMOUS_FINAL_MASK; stored macro-action trajectory",
            "Software": "cmc_bbdm single-case overlay diagnostic",
        },
    )
    plt.close(figure)


def export_single_case_overlay_diagnostic(
    *, project_root: str | Path, source_root: str | Path, output_root: str | Path
) -> dict[str, object]:
    """Export paper-ready panels for one frozen reviewed TEST specimen."""

    root = Path(project_root).resolve(strict=True)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = recover_selected_case(project_root=root, source_root=source_root)
    identity = (
        f"{bundle.selected.specimen_key} | {bundle.selected.method} seed "
        f"{bundle.selected.seed} | {bundle.selected.task}"
    )

    def draw_base_legend(axis: object) -> None:
        from matplotlib.patches import Patch

        _legend_below(
            axis,
            (
                Patch(
                    facecolor="#BDBDBD",
                    edgecolor="#333333",
                    label="Registered C-scan (no overlay)",
                ),
            ),
            columns=1,
        )

    _save_panel(
        output,
        "panel_00_base_registered_cscan.png",
        bundle,
        title=f"Registered C-scan | {bundle.selected.specimen_key}",
        description="Registered C-scan background only; no mask overlay.",
        draw=draw_base_legend,
    )

    def draw_expert(axis: object) -> None:
        _overlay_mask(
            axis,
            bundle.expert_mask,
            color=EXPERT_COLOR,
            alpha=0.24,
            linestyle="solid",
        )
        if np.any(bundle.uncertain_mask):
            _overlay_mask(
                axis,
                bundle.uncertain_mask,
                color=EXPERT_COLOR,
                alpha=0.09,
                linestyle="dashed",
            )
            uncertain = _mask_handle("EXPERT_GT uncertain", EXPERT_COLOR, "dashed")
        else:
            uncertain = _mask_handle("uncertain: none", "#666666", "dotted")
        _legend_below(
            axis,
            (_mask_handle("EXPERT_GT", EXPERT_COLOR, "solid"), uncertain),
            columns=2,
        )

    _save_panel(
        output,
        "panel_01_expert_gt_overlay.png",
        bundle,
        title=f"EXPERT_GT | {bundle.selected.specimen_key}",
        description="EXPERT_GT; uncertain: none",
        draw=draw_expert,
    )

    def draw_proxy(axis: object) -> None:
        _overlay_mask(
            axis,
            bundle.proxy_mask,
            color=PROXY_COLOR,
            alpha=0.20,
            linestyle="dashed",
        )
        _legend_below(
            axis,
            (
                _mask_handle(
                    "PROXY_GT (full-input Reader target)", PROXY_COLOR, "dashed"
                ),
            ),
            columns=1,
        )

    _save_panel(
        output,
        "panel_02_proxy_gt_overlay.png",
        bundle,
        title=f"PROXY_GT | full-input Reader | {bundle.selected.task} | {bundle.selected.specimen_key}",
        description="PROXY_GT (full-input Reader target); not expert ground truth.",
        draw=draw_proxy,
    )

    def draw_bc(axis: object) -> None:
        _overlay_mask(
            axis,
            bundle.bc_mask,
            color=BC_COLOR,
            alpha=0.20,
            linestyle="dotted",
        )
        _legend_below(
            axis,
            (_mask_handle("BC_AUTONOMOUS_FINAL_MASK", BC_COLOR, "dotted"),),
            columns=1,
        )

    _save_panel(
        output,
        "panel_03_bc_final_mask_overlay.png",
        bundle,
        title=f"BC autonomous first STOP | {identity} | stop_step {bundle.selected.stop_step}",
        description="BC_AUTONOMOUS_FINAL_MASK recovered at the actual calibrated first STOP.",
        draw=draw_bc,
    )
    _save_panel(
        output,
        "panel_04_bc_trajectory_and_mask_overlay.png",
        bundle,
        title=(
            f"Stored BC trajectory + first-STOP mask | {identity}\n"
            f"stop_step {bundle.selected.stop_step} | final cost {bundle.selected.stop_cost:.4f}"
        ),
        description="BC_AUTONOMOUS_FINAL_MASK; measured native positions; 8x8 acquisition grid; ordered stored macro-actions.",
        draw=lambda axis: _draw_trajectory(axis, bundle),
    )

    def draw_expert_bc(axis: object) -> None:
        _overlay_mask(
            axis, bundle.expert_mask, color=EXPERT_COLOR, alpha=0.18, linestyle="solid"
        )
        _overlay_mask(
            axis, bundle.bc_mask, color=BC_COLOR, alpha=0.14, linestyle="dotted"
        )
        _legend_below(
            axis,
            (
                _mask_handle("EXPERT_GT", EXPERT_COLOR, "solid"),
                _mask_handle("BC_AUTONOMOUS_FINAL_MASK", BC_COLOR, "dotted"),
            ),
            columns=2,
        )

    _save_panel(
        output,
        "panel_05_overlay_compare_expert_vs_bc.png",
        bundle,
        title=f"EXPERT_GT vs BC first-STOP mask | {identity}",
        description="EXPERT_GT; BC_AUTONOMOUS_FINAL_MASK.",
        draw=draw_expert_bc,
    )

    def draw_all(axis: object, *, fill: bool) -> None:
        _overlay_mask(
            axis,
            bundle.expert_mask,
            color=EXPERT_COLOR,
            alpha=0.15 if fill else 0.0,
            linestyle="solid",
        )
        _overlay_mask(
            axis,
            bundle.proxy_mask,
            color=PROXY_COLOR,
            alpha=0.10 if fill else 0.0,
            linestyle="dashed",
        )
        _overlay_mask(
            axis,
            bundle.bc_mask,
            color=BC_COLOR,
            alpha=0.10 if fill else 0.0,
            linestyle="dotted",
        )
        _legend_below(
            axis,
            (
                _mask_handle("EXPERT_GT", EXPERT_COLOR, "solid"),
                _mask_handle(
                    "PROXY_GT (full-input Reader target)", PROXY_COLOR, "dashed"
                ),
                _mask_handle("BC_AUTONOMOUS_FINAL_MASK", BC_COLOR, "dotted"),
            ),
            columns=3,
        )

    _save_panel(
        output,
        "panel_06_overlay_compare_expert_vs_proxy_vs_bc.png",
        bundle,
        title=f"EXPERT_GT vs PROXY_GT vs BC first-STOP mask | {identity}",
        description="EXPERT_GT; PROXY_GT (full-input Reader target); BC_AUTONOMOUS_FINAL_MASK.",
        draw=lambda axis: draw_all(axis, fill=True),
    )
    _save_panel(
        output,
        "panel_07_overlay_boundary_only_compare.png",
        bundle,
        title=f"Boundary-only mask comparison | {identity}",
        description="Boundary only: EXPERT_GT; PROXY_GT (full-input Reader target); BC_AUTONOMOUS_FINAL_MASK.",
        draw=lambda axis: draw_all(axis, fill=False),
    )
    _save_summary(output, bundle)

    trajectory = _trajectory_summary(bundle)
    manifest = {
        "schema_version": 1,
        "repository_base_sha": "6954e37c8c6aa5fc8530d2668d689eae85604691",
        "specimen_key": bundle.selected.specimen_key,
        "dataset_id": bundle.selected.dataset_id,
        "task": bundle.selected.task,
        "method": bundle.selected.method,
        "seed": bundle.selected.seed,
        "reference_version": bundle.reference_version,
        "selected_reason": bundle.selected.selected_reason,
        "selection_rule": bundle.selected.selection_rule,
        "source_cscan_path": bundle.source_cscan_path,
        "source_image_sha256": bundle.source_image_sha256,
        "expert_reference_json": bundle.expert_reference_json,
        "proxy_report_sha256": bundle.proxy_report_sha256,
        "bc_final_report_sha256": bundle.bc_report_sha256,
        "stop_step": bundle.selected.stop_step,
        "final_cost": bundle.selected.stop_cost,
        "image_size": {
            "width": int(bundle.registered_cscan.shape[1]),
            "height": int(bundle.registered_cscan.shape[0]),
        },
        "overlay_files": list(REQUIRED_OVERLAY_FILES),
        "coordinate_checks": dict(bundle.coordinate_checks),
        "identity_checks": dict(bundle.identity_checks),
        "mask_metrics": dict(bundle.mask_metrics),
        "trajectory": trajectory,
        "visual_encoding": {
            "EXPERT_GT": {"color": EXPERT_COLOR, "line": "solid"},
            "PROXY_GT": {"color": PROXY_COLOR, "line": "dashed"},
            "BC_FINAL": {"color": BC_COLOR, "line": "dotted"},
        },
        "resource_accounting": {
            "new_training": 0,
            "new_vlm_calls": 0,
            "actor_forward_calls": 0,
            "stop_forward_calls": 0,
            "world_step_count": 0,
        },
        "notes": [
            "PROXY_GT means the full-input Reader target; it is not expert ground truth.",
            "BC final mask is recovered at the actual calibrated first STOP from stored actions.",
            "The stop-row action is not executed; stored actions 0 through stop_step-1 define the trajectory.",
            "All overlays use the unchanged registered C-scan pixel coordinate frame.",
            "No training, replanning, model selection, or endpoint rescoring was performed.",
        ],
    }
    manifest_path = output / "single_case_overlay_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "specimen_key": bundle.selected.specimen_key,
        "task": bundle.selected.task,
        "method": bundle.selected.method,
        "seed": bundle.selected.seed,
        "output_root": str(output),
        "overlay_files": tuple(str(output / name) for name in REQUIRED_OVERLAY_FILES),
        "manifest": str(manifest_path),
    }


__all__ = [
    "REQUIRED_OVERLAY_FILES",
    "RecoveredCase",
    "SelectedCase",
    "export_single_case_overlay_diagnostic",
    "recover_selected_case",
    "select_diagnostic_case",
]
