"""Stored-action Reader recovery for frozen C-scan process diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl
from scipy import ndimage
from scipy.interpolate import interpn

from cmc_bbdm.inspection_agent.state import (
    GeneralizedMeasurementState,
    InspectionCellAction,
    action_added_positions_from_mask,
    apply_action,
    zero_state,
)
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.vlm_cscan.contracts import CScanReference
from cmc_bbdm.vlm_cscan.references import evaluate_task_report

from .benchmark import _load_percepts, _proxy_score, _report_digest
from .contracts import Split, Task
from .frozen_process_analysis import (
    FrozenProcessConfig,
    overlap_metrics,
    surface_cell_partition,
)
from .readout import (
    BackgroundPrior,
    TaskReportV2,
    build_task_report_v2,
    owned_cell_slices,
    read_visible_evidence,
)
from .runtime import (
    load_study_config,
    load_study_context,
    open_study_specimen,
)
from .supplement_adapters import adapt_task_report_v2


class FrozenVisibleReportReader:
    """Exact Reader fast path for many frozen states of one specimen."""

    def __init__(
        self,
        *,
        grid: AcquisitionGrid,
        full_scan: np.ndarray,
        prior: BackgroundPrior,
        distance_threshold: float,
    ) -> None:
        scan = np.asarray(full_scan)
        if (
            type(grid) is not AcquisitionGrid
            or scan.dtype != np.uint8
            or scan.shape != (*grid.native_shape, 3)
            or type(prior) is not BackgroundPrior
            or isinstance(distance_threshold, bool)
            or not 0.0 < float(distance_threshold) < 1.0
        ):
            raise ValueError("frozen visible Reader request is invalid")
        self.grid = grid
        self.full_scan = scan
        self.prior = prior
        self.distance_threshold = float(distance_threshold)
        self.observed_scores = np.linalg.norm(
            scan.astype(np.float64) - prior.rgb.astype(np.float64), axis=-1
        ) / math.sqrt(3.0 * 255.0**2)
        self._cell_cache: dict[
            tuple[int, tuple[int, ...], tuple[int, ...]],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}

    def _interpolated_cell(
        self,
        *,
        cell_index: int,
        measured_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        cell = self.grid.cells[cell_index]
        row_lower, row_upper = self.grid.row_boundaries[cell.row : cell.row + 2]
        column_lower, column_upper = self.grid.column_boundaries[
            cell.column : cell.column + 2
        ]
        local_support = measured_mask[
            row_lower : row_upper + 1, column_lower : column_upper + 1
        ]
        support_rows = (
            np.flatnonzero(np.any(local_support, axis=1)).astype(np.int64)
            + row_lower
        )
        support_columns = (
            np.flatnonzero(np.any(local_support, axis=0)).astype(np.int64)
            + column_lower
        )
        if (
            not len(support_rows)
            or not len(support_columns)
            or np.count_nonzero(local_support)
            != len(support_rows) * len(support_columns)
        ):
            return None
        key = (
            cell_index,
            tuple(int(value) for value in support_rows),
            tuple(int(value) for value in support_columns),
        )
        cached = self._cell_cache.get(key)
        if cached is not None:
            return cached
        row_slice, column_slice = owned_cell_slices(self.grid, cell)
        target_rows = np.arange(row_slice.start, row_slice.stop, dtype=np.int64)
        target_columns = np.arange(
            column_slice.start, column_slice.stop, dtype=np.int64
        )
        target_rows = target_rows[
            (target_rows >= support_rows[0]) & (target_rows <= support_rows[-1])
        ]
        target_columns = target_columns[
            (target_columns >= support_columns[0])
            & (target_columns <= support_columns[-1])
        ]
        if not len(target_rows) or not len(target_columns):
            return None
        lattice = self.full_scan[np.ix_(support_rows, support_columns)].astype(
            np.float64
        )
        result: np.ndarray | None = None
        if len(support_rows) >= 2 and len(support_columns) >= 2:
            target_row_grid, target_column_grid = np.meshgrid(
                target_rows, target_columns, indexing="ij"
            )
            points = np.column_stack(
                (target_row_grid.ravel(), target_column_grid.ravel())
            )
            result = interpn(
                (support_rows, support_columns), lattice, points, method="linear"
            ).reshape(len(target_rows), len(target_columns), 3)
        elif len(support_rows) == 1 and len(support_columns) >= 2:
            result = np.column_stack(
                [
                    np.interp(
                        target_columns, support_columns, lattice[0, :, channel]
                    )
                    for channel in range(3)
                ]
            )[None, :, :]
        elif len(support_rows) >= 2 and len(support_columns) == 1:
            result = np.column_stack(
                [
                    np.interp(target_rows, support_rows, lattice[:, 0, channel])
                    for channel in range(3)
                ]
            )[:, None, :]
        if result is None:
            return None
        scores = np.linalg.norm(
            result - self.prior.rgb.astype(np.float64), axis=-1
        ) / math.sqrt(3.0 * 255.0**2)
        output = (
            target_rows,
            target_columns,
            scores >= self.distance_threshold,
        )
        self._cell_cache[key] = output
        return output

    def report(
        self,
        *,
        state: GeneralizedMeasurementState,
        measured_mask: np.ndarray,
        task: Task,
    ) -> TaskReportV2:
        """Return the same public report as ``read_visible_task_report``."""

        measured = np.asarray(measured_mask)
        if (
            type(state) is not GeneralizedMeasurementState
            or state.grid_sha256 != self.grid.state_sha256
            or measured.dtype != np.bool_
            or measured.shape != self.grid.native_shape
            or type(task) is not Task
        ):
            raise ValueError("frozen visible Reader state is invalid")
        candidate = np.zeros(self.grid.native_shape, dtype=np.bool_)
        estimate_valid = np.array(measured, copy=True)
        candidate[measured] = (
            self.observed_scores[measured] >= self.distance_threshold
        )
        for cell, level in zip(self.grid.cells, state.levels, strict=True):
            if not 0 <= level < 2:
                continue
            interpolation = self._interpolated_cell(
                cell_index=cell.index, measured_mask=measured
            )
            if interpolation is None:
                continue
            target_rows, target_columns, local_candidate = interpolation
            estimate_valid[np.ix_(target_rows, target_columns)] = True
            candidate[np.ix_(target_rows, target_columns)] = local_candidate
        candidate[measured] = (
            self.observed_scores[measured] >= self.distance_threshold
        )
        candidate_cells = []
        unverified = []
        for cell, level in zip(self.grid.cells, state.levels, strict=True):
            row_slice, column_slice = owned_cell_slices(self.grid, cell)
            is_candidate = bool(
                level >= 0 and np.any(candidate[row_slice, column_slice])
            )
            if is_candidate:
                candidate_cells.append(cell.index)
                if np.any(~estimate_valid[row_slice, column_slice]):
                    unverified.append(cell.index)
        prediction = np.array(candidate, copy=True)
        if task is Task.LOCATE and np.any(prediction):
            labels, count = ndimage.label(prediction)
            sizes = np.bincount(labels.ravel(), minlength=count + 1)
            sizes[0] = 0
            prediction = labels == int(np.argmax(sizes))
        support = np.argwhere(prediction & measured).astype(np.int64, copy=False)
        signal = (
            float(np.mean(self.observed_scores[support[:, 0], support[:, 1]]))
            if len(support)
            else 0.0
        )
        return TaskReportV2(
            task=task,
            predicted_mask=prediction,
            support_positions=support,
            candidate_cells=tuple(candidate_cells),
            unverified_boundary_cells=tuple(unverified),
            signal_strength=float(np.clip(signal, 0.0, 1.0)),
            reason_code=(
                "VISIBLE_CANDIDATE" if candidate_cells else "NO_VISIBLE_CANDIDATE"
            ),
        )


def cell_partition_masks(
    grid: AcquisitionGrid, partition: Mapping[str, tuple[int, ...]]
) -> dict[str, np.ndarray]:
    """Rasterize the mutually exclusive owned-cell V/R/O partition."""

    if type(grid) is not AcquisitionGrid or set(partition) != {"V", "R", "O"}:
        raise ValueError("cell partition request is invalid")
    cells = []
    for name in ("V", "R", "O"):
        group = partition[name]
        if (
            type(group) is not tuple
            or any(type(cell) is not int or not 0 <= cell < 64 for cell in group)
        ):
            raise ValueError("cell partition request is invalid")
        cells.extend(group)
    if len(cells) != 64 or set(cells) != set(range(64)):
        raise ValueError("cell partition is not exhaustive and exclusive")
    masks = {}
    for name in ("V", "R", "O"):
        mask = np.zeros(grid.native_shape, dtype=np.bool_)
        for cell_index in partition[name]:
            row_slice, column_slice = owned_cell_slices(
                grid, grid.cells[cell_index]
            )
            mask[row_slice, column_slice] = True
        mask.setflags(write=False)
        masks[name] = mask
    total = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(total == 1):
        raise ValueError("cell partition rasterization is invalid")
    return masks


def _checked_spatial_masks(
    target_mask: np.ndarray, region_masks: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target = np.asarray(target_mask)
    if target.dtype != np.bool_ or target.ndim != 2 or set(region_masks) != {
        "V",
        "R",
        "O",
    }:
        raise ValueError("spatial masks are invalid")
    regions = {name: np.asarray(region_masks[name]) for name in ("V", "R", "O")}
    if any(mask.dtype != np.bool_ or mask.shape != target.shape for mask in regions.values()):
        raise ValueError("spatial masks are invalid")
    total = sum(mask.astype(np.int8) for mask in regions.values())
    if not np.all(total == 1):
        raise ValueError("spatial regions are not exhaustive and exclusive")
    return target, regions


def action_spatial_allocation(
    grid: AcquisitionGrid,
    *,
    state: GeneralizedMeasurementState,
    action: InspectionCellAction,
    current_mask: np.ndarray,
    target_mask: np.ndarray,
    region_masks: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Separate action-cell membership from the positions it newly measures."""

    target, regions = _checked_spatial_masks(target_mask, region_masks)
    current = np.asarray(current_mask)
    if (
        type(grid) is not AcquisitionGrid
        or type(state) is not GeneralizedMeasurementState
        or type(action) is not InspectionCellAction
        or current.dtype != np.bool_
        or current.shape != grid.native_shape
        or target.shape != grid.native_shape
    ):
        raise ValueError("spatial action request is invalid")
    added = action_added_positions_from_mask(grid, state, action, current)
    rows = added[:, 0]
    columns = added[:, 1]
    row_slice, column_slice = owned_cell_slices(grid, grid.cells[action.cell_index])
    representative = (row_slice.start, column_slice.start)
    action_region = next(
        name for name in ("V", "R", "O") if regions[name][representative]
    )
    counts = {
        name: int(np.count_nonzero(regions[name][rows, columns]))
        for name in ("V", "R", "O")
    }
    added_count = len(added)
    if sum(counts.values()) != added_count:
        raise ValueError("added-position region allocation changed")
    return {
        "action_cell_region": action_region,
        "added_pixel_count": added_count,
        "added_target_pixel_count": int(
            np.count_nonzero(target[rows, columns])
        ),
        "added_v_pixel_count": counts["V"],
        "added_r_pixel_count": counts["R"],
        "added_o_pixel_count": counts["O"],
        "added_positions": added,
    }


def support_spatial_allocation(
    support_positions: np.ndarray,
    *,
    target_mask: np.ndarray,
    region_masks: Mapping[str, np.ndarray],
) -> dict[str, int]:
    """Count report support independently of action and acquisition locations."""

    target, regions = _checked_spatial_masks(target_mask, region_masks)
    positions = np.asarray(support_positions)
    if (
        positions.dtype.kind not in "iu"
        or positions.ndim != 2
        or positions.shape[1:] != (2,)
        or (
            positions.size
            and (
                np.any(positions < 0)
                or np.any(positions[:, 0] >= target.shape[0])
                or np.any(positions[:, 1] >= target.shape[1])
            )
        )
    ):
        raise ValueError("report support positions are invalid")
    rows = positions[:, 0]
    columns = positions[:, 1]
    return {
        "support_pixel_count": len(positions),
        "support_target_pixel_count": int(
            np.count_nonzero(target[rows, columns])
        ),
        "support_v_pixel_count": int(np.count_nonzero(regions["V"][rows, columns])),
        "support_r_pixel_count": int(np.count_nonzero(regions["R"][rows, columns])),
        "support_o_pixel_count": int(np.count_nonzero(regions["O"][rows, columns])),
    }


def _full_reports(
    context: object, runtime: object, full_scan: np.ndarray | None = None
) -> dict[str, object]:
    if full_scan is None:
        full_scan = context.authority.source_teacher_view(  # type: ignore[attr-defined]
            runtime.record.specimen_id  # type: ignore[attr-defined]
        ).full_scan
    shape = runtime.grid.native_shape  # type: ignore[attr-defined]
    positions = np.argwhere(np.ones(shape, dtype=np.bool_))
    readout = read_visible_evidence(
        grid=runtime.grid,  # type: ignore[attr-defined]
        positions=positions,
        values=full_scan[positions[:, 0], positions[:, 1]],
        cell_levels=(2,) * 64,
        prior=context.background_prior,  # type: ignore[attr-defined]
        distance_threshold=float(  # type: ignore[attr-defined]
            context.config.values["reader"]["distance_threshold"]
        ),
    )
    return {task.value: build_task_report_v2(readout, task=task) for task in Task}


def _stop_steps(rows: tuple[Mapping[str, object], ...]) -> dict[str, int | None]:
    calibrated = next(
        (
            int(row["step"])
            for row in rows
            if bool(row["calibrated_stop_trigger"])
        ),
        None,
    )
    rule = next(
        (int(row["step"]) for row in rows if bool(row["rule_stop"])), None
    )
    return {"S_BC_CAL": calibrated, "S_RULE": rule}


def _locate_support_ready(support: np.ndarray, target: np.ndarray) -> bool:
    if not len(support):
        return False
    inside = target[support[:, 0], support[:, 1]]
    accepted = support[inside]
    return bool(
        len(accepted) >= 3
        and len({int(value) for value in accepted[:, 0]}) >= 2
        and len({int(value) for value in accepted[:, 1]}) >= 2
    )


def _reviewed_target(reference: CScanReference, task: Task) -> np.ndarray:
    target = np.asarray(reference.certain_mask, dtype=np.bool_)
    if task is Task.LOCATE and np.any(target):
        labels, count = ndimage.label(target)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        target = labels == int(np.argmax(sizes))
    return target


def _first_state(
    states: list[dict[str, object]], field: str, *, end_step: int
) -> dict[str, object] | None:
    return next(
        (
            row
            for row in states
            if int(row["step"]) <= end_step and bool(row[field])
        ),
        None,
    )


def _spatial_summary_rows(
    identity: Mapping[str, object],
    *,
    states: list[dict[str, object]],
    actions: list[dict[str, object]],
    stop_steps: Mapping[str, int | None],
    terminal_step: int,
) -> tuple[dict[str, object], ...]:
    scopes = (
        ("FULL_PLANNER_DIAGNOSTIC", "NONE_FULL_TRAJECTORY", terminal_step),
        (
            "AUTONOMOUS_PREFIX",
            "S_BC_CAL",
            terminal_step
            if stop_steps["S_BC_CAL"] is None
            else int(stop_steps["S_BC_CAL"]),
        ),
        (
            "AUTONOMOUS_PREFIX",
            "S_RULE",
            terminal_step
            if stop_steps["S_RULE"] is None
            else int(stop_steps["S_RULE"]),
        ),
    )
    output = []
    for scope, stop_system, end_step in scopes:
        selected = [row for row in actions if int(row["step"]) < end_step]
        measured = _first_state(states, "has_target_measurement", end_step=end_step)
        support = _first_state(states, "has_target_support", end_step=end_step)
        locate_ready = _first_state(states, "locate_support_ready", end_step=end_step)
        first_v_index = next(
            (
                index
                for index, row in enumerate(selected)
                if row["action_cell_region"] == "V"
            ),
            None,
        )
        outside = (
            None
            if first_v_index is None
            else next(
                (
                    row
                    for row in selected[first_v_index + 1 :]
                    if row["action_cell_region"] != "V"
                ),
                None,
            )
        )
        output.append(
            {
                **identity,
                "analysis_scope": scope,
                "stop_system": stop_system,
                "execution_end_step": end_step,
                "action_count": len(selected),
                "first_target_measurement_step": (
                    None if measured is None else measured["step"]
                ),
                "first_target_measurement_cost": (
                    None if measured is None else measured["cost"]
                ),
                "first_target_support_step": (
                    None if support is None else support["step"]
                ),
                "first_target_support_cost": (
                    None if support is None else support["cost"]
                ),
                "first_locate_support_criterion_step": (
                    None if locate_ready is None else locate_ready["step"]
                ),
                "first_locate_support_criterion_cost": (
                    None if locate_ready is None else locate_ready["cost"]
                ),
                "added_v_measurement_fraction": float(
                    sum(float(row["added_v_measurement_fraction"]) for row in selected)
                ),
                "added_r_measurement_fraction": float(
                    sum(float(row["added_r_measurement_fraction"]) for row in selected)
                ),
                "added_o_measurement_fraction": float(
                    sum(float(row["added_o_measurement_fraction"]) for row in selected)
                ),
                "entered_v": first_v_index is not None,
                "first_v_action_step": (
                    None if first_v_index is None else selected[first_v_index]["step"]
                ),
                "post_v_outside_action_status": (
                    "NOT_APPLICABLE"
                    if first_v_index is None
                    else (
                        "NONE_RECORDED_AFTER_ENTRY"
                        if outside is None
                        else "OBSERVED"
                    )
                ),
                "first_post_v_outside_action_step": (
                    None if outside is None else outside["step"]
                ),
            }
        )
    return tuple(output)


def recover_spatial_analysis(
    config: FrozenProcessConfig,
    *,
    source_root: str | Path,
    trajectories: pl.DataFrame,
) -> dict[str, object]:
    """Replay only frozen P8/BC actions and recover compact W4 summaries."""

    if type(config) is not FrozenProcessConfig or type(trajectories) is not pl.DataFrame:
        raise TypeError("typed frozen config and trajectory table are required")
    external = Path(source_root).resolve(strict=True)
    parent = load_study_config(config.parent_config_path, project_root=config.project_root)
    context = load_study_context(parent, source_root=external)
    percepts = _load_percepts(
        parent, context, config.surface_percept_cache.parent
    )
    test_records = {
        assignment.record.specimen_key: assignment.record
        for assignment in context.roster.assignments
        if assignment.split is Split.TEST
    }
    selected = trajectories.filter(pl.col("method").is_in(config.spatial_methods))
    episode_frames = selected.partition_by(
        ["specimen_key", "task", "method", "seed"], maintain_order=True
    )
    if len(episode_frames) != 192:
        raise ValueError("spatial recovery episode matrix changed")
    runtimes = {}
    readers = {}
    targets = {}
    region_masks_by_specimen = {}
    agreement_rows = []
    for specimen_key in sorted(test_records):
        runtime = open_study_specimen(context, test_records[specimen_key])
        runtimes[specimen_key] = runtime
        full_scan = context.authority.source_teacher_view(
            runtime.record.specimen_id
        ).full_scan
        reports = _full_reports(context, runtime, full_scan)
        targets[specimen_key] = reports
        readers[specimen_key] = FrozenVisibleReportReader(
            grid=runtime.grid,
            full_scan=full_scan,
            prior=context.background_prior,
            distance_threshold=float(
                context.config.values["reader"]["distance_threshold"]
            ),
        )
        percept = percepts[specimen_key]
        cue_cells = tuple(
            sorted({cell for region in percept.regions for cell in region.cells})
        )
        partition = surface_cell_partition(cue_cells)
        masks = cell_partition_masks(runtime.grid, partition)
        region_masks_by_specimen[specimen_key] = (partition, masks)
        for task in config.tasks:
            target = reports[task].predicted_mask
            metrics = overlap_metrics(target, masks["V"])
            agreement_rows.append(
                {
                    "schema_version": 1,
                    "dataset_id": test_records[specimen_key].dataset_id,
                    "specimen_id": test_records[specimen_key].specimen_id,
                    "specimen_key": specimen_key,
                    "task": task,
                    "analysis_scope": "PROXY_SPATIAL_RELATION",
                    "reference_version": "PROXY_LEGACY",
                    "no_reliable_cue": bool(percept.no_reliable_cue),
                    "surface_cue_cells": json_tuple(partition["V"]),
                    "surface_ring_cells": json_tuple(partition["R"]),
                    **metrics,
                }
            )
    spatial_actions = []
    spatial_summaries = []
    transition_steps = 0
    report_states = 0
    for frame in episode_frames:
        rows = tuple(frame.sort("step").to_dicts())
        first = rows[0]
        specimen_key = str(first["specimen_key"])
        task = Task(str(first["task"]))
        runtime = runtimes[specimen_key]
        reader = readers[specimen_key]
        target_report = targets[specimen_key][task.value]
        target = target_report.predicted_mask
        partition, region_masks = region_masks_by_specimen[specimen_key]
        state = zero_state(runtime.grid)
        measured_mask = np.zeros(runtime.grid.native_shape, dtype=np.bool_)
        states = []
        actions = []
        stop_steps = _stop_steps(rows)
        identity = {
            "schema_version": 1,
            "dataset_id": str(first["dataset_id"]),
            "specimen_id": str(first["specimen_id"]),
            "specimen_key": specimen_key,
            "task": task.value,
            "method": str(first["method"]),
            "seed": int(first["seed"]),
            "reference_version": "PROXY_LEGACY",
        }
        for index, row in enumerate(rows):
            if int(row["step"]) != index or not math.isclose(
                float(row["cost"]),
                float(np.count_nonzero(measured_mask) / measured_mask.size),
                abs_tol=1e-12,
            ):
                raise ValueError("stored spatial replay order changed")
            report = reader.report(
                state=state,
                measured_mask=measured_mask,
                task=task,
            )
            if _report_digest(report) != row["report_sha256"]:
                raise ValueError("stored report replay identity changed")
            score = _proxy_score(report, target_report)
            if (
                bool(score["success"]) != bool(row["success"])
                or not math.isclose(float(score["iou"]), float(row["iou"]), abs_tol=1e-12)
                or not math.isclose(
                    float(score["recall"]), float(row["recall"]), abs_tol=1e-12
                )
                or not math.isclose(
                    float(score["relative_area_error"]),
                    float(row["relative_area_error"]),
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("stored proxy score replay changed")
            support = support_spatial_allocation(
                report.support_positions,
                target_mask=target,
                region_masks=region_masks,
            )
            measured_target = int(np.count_nonzero(measured_mask & target))
            states.append(
                {
                    "step": index,
                    "cost": float(row["cost"]),
                    "has_target_measurement": measured_target > 0,
                    "has_target_support": support["support_target_pixel_count"] > 0,
                    "locate_support_ready": (
                        task is Task.LOCATE
                        and _locate_support_ready(report.support_positions, target)
                    ),
                }
            )
            report_states += 1
            action_cell = int(row["action_cell"])
            if action_cell < 0:
                if index != len(rows) - 1:
                    raise ValueError("stored terminal action is misplaced")
                continue
            action = InspectionCellAction(
                action_cell,
                int(row["action_from_level"]),
                int(row["action_to_level"]),
            )
            allocation = action_spatial_allocation(
                runtime.grid,
                state=state,
                action=action,
                current_mask=measured_mask,
                target_mask=target,
                region_masks=region_masks,
            )
            native_count = int(np.prod(runtime.grid.native_shape))
            added_fraction = allocation["added_pixel_count"] / native_count
            source_delta = float(rows[index + 1]["cost"]) - float(row["cost"])
            if not math.isclose(added_fraction, source_delta, abs_tol=1e-12):
                raise ValueError("spatial added-position cost changed")
            action_row = {
                **identity,
                "analysis_scope": "FULL_PLANNER_DIAGNOSTIC",
                "step": index,
                "cost_before_action": float(row["cost"]),
                "action_cell": action.cell_index,
                "action_from_level": action.from_level,
                "action_to_level": action.to_level,
                "action_cell_region": allocation["action_cell_region"],
                "added_pixel_count": allocation["added_pixel_count"],
                "added_target_pixel_count": allocation["added_target_pixel_count"],
                "added_v_pixel_count": allocation["added_v_pixel_count"],
                "added_r_pixel_count": allocation["added_r_pixel_count"],
                "added_o_pixel_count": allocation["added_o_pixel_count"],
                "added_measurement_fraction": added_fraction,
                "added_target_measurement_fraction": allocation[
                    "added_target_pixel_count"
                ]
                / native_count,
                "added_v_measurement_fraction": allocation["added_v_pixel_count"]
                / native_count,
                "added_r_measurement_fraction": allocation["added_r_pixel_count"]
                / native_count,
                "added_o_measurement_fraction": allocation["added_o_pixel_count"]
                / native_count,
                "pre_action_success": bool(row["success"]),
                "post_action_success": bool(rows[index + 1]["success"]),
                **support,
                "in_s_bc_cal_prefix": (
                    stop_steps["S_BC_CAL"] is None
                    or index < int(stop_steps["S_BC_CAL"])
                ),
                "in_s_rule_prefix": (
                    stop_steps["S_RULE"] is None
                    or index < int(stop_steps["S_RULE"])
                ),
            }
            action_row.pop("added_positions", None)
            actions.append(action_row)
            spatial_actions.append(action_row)
            added_positions = allocation["added_positions"]
            measured_mask[
                added_positions[:, 0], added_positions[:, 1]
            ] = True
            state = apply_action(runtime.grid, state, action)
            transition_steps += 1
        spatial_summaries.extend(
            _spatial_summary_rows(
                identity,
                states=states,
                actions=actions,
                stop_steps=stop_steps,
                terminal_step=int(rows[-1]["step"]),
            )
        )
    if (
        transition_steps != 36864
        or transition_steps > config.base_recovery_step_cap
    ):
        raise ValueError("spatial recovery step cap exceeded")
    return {
        "surface_proxy_agreement": tuple(agreement_rows),
        "spatial_action_statistics": tuple(spatial_actions),
        "spatial_episode_summary": tuple(spatial_summaries),
        "recovery_manifest": {
            "schema_version": 1,
            "stage": "FROZEN_SPATIAL_RECOVERY_COMPLETE",
            "analysis_scope": "PROXY_SPATIAL_RELATION",
            "reference_version": "PROXY_LEGACY",
            "source_root": str(external),
            "episode_count": len(episode_frames),
            "report_state_count": report_states,
            "reader_recovery_episode_count": len(episode_frames),
            "stored_action_transition_count": transition_steps,
            "stored_action_transition_cap": config.base_recovery_step_cap,
            "world_step_count": 0,
            "full_input_specimen_reads": len(test_records),
            "full_input_task_reports": len(test_records) * len(config.tasks),
            "report_digest_mismatches": 0,
            "proxy_score_mismatches": 0,
            "actor_forward_calls": 0,
            "stop_forward_calls": 0,
            "vlm_calls": 0,
            "training_updates": 0,
            "packet_hash_scope": "REPORT_DIGEST_ONLY_EXACT_CACHED_READER_PATH",
            "cache_reuse": (
                "SURFACE_PERCEPT_CACHE_STORED_ACTIONS_AND_CELL_INTERPOLATION"
            ),
            "report_cache_scope": "NONE_W4_COMPACT_SUMMARIES_ONLY",
            "report_cache_file_count": 0,
            "report_cache_entries": [],
        },
    }


def recover_reviewed_report_scores(
    config: FrozenProcessConfig,
    *,
    source_root: str | Path,
    trajectories: pl.DataFrame,
    references: Mapping[str, CScanReference],
    reference_version: str,
) -> dict[str, object]:
    """Score frozen reports against real reviewed references without model calls."""

    if (
        type(config) is not FrozenProcessConfig
        or type(trajectories) is not pl.DataFrame
        or not references
        or any(type(value) is not CScanReference for value in references.values())
        or not reference_version.startswith("REVIEWED_")
    ):
        raise ValueError("reviewed report recovery request is invalid")
    external = Path(source_root).resolve(strict=True)
    parent = load_study_config(config.parent_config_path, project_root=config.project_root)
    context = load_study_context(parent, source_root=external)
    records = {
        assignment.record.specimen_key: assignment.record
        for assignment in context.roster.assignments
        if assignment.split is Split.TEST
    }
    methods = (
        "R_BALANCED_P8",
        "BC_S1",
        "BC_S2",
        "BC_S3",
        "BC_NO_VLM_S1",
        "BC_NO_US_FEEDBACK_S1",
    )
    selected = trajectories.filter(
        pl.col("specimen_key").is_in(tuple(references))
        & pl.col("method").is_in(methods)
    )
    expected_episodes = len(references) * len(config.tasks) * len(methods)
    frames = selected.partition_by(
        ["specimen_key", "task", "method", "seed"], maintain_order=True
    )
    if len(frames) != expected_episodes:
        raise ValueError("reviewed frozen episode matrix is incomplete")
    runtimes = {}
    readers = {}
    full_reports = {}
    region_masks_by_specimen = {}
    target_masks = {}
    agreement_rows = []
    percepts = _load_percepts(
        parent, context, config.surface_percept_cache.parent
    )
    for specimen_key in sorted(references):
        runtime = open_study_specimen(context, records[specimen_key])
        runtimes[specimen_key] = runtime
        full_scan = context.authority.source_teacher_view(
            runtime.record.specimen_id
        ).full_scan
        readers[specimen_key] = FrozenVisibleReportReader(
            grid=runtime.grid,
            full_scan=full_scan,
            prior=context.background_prior,
            distance_threshold=float(
                context.config.values["reader"]["distance_threshold"]
            ),
        )
        full_reports[specimen_key] = _full_reports(
            context, runtime, full_scan
        )
        percept = percepts[specimen_key]
        cue_cells = tuple(
            sorted({cell for region in percept.regions for cell in region.cells})
        )
        partition = surface_cell_partition(cue_cells)
        region_masks = cell_partition_masks(runtime.grid, partition)
        region_masks_by_specimen[specimen_key] = region_masks
        for task in Task:
            target = _reviewed_target(references[specimen_key], task)
            target_masks[(specimen_key, task.value)] = target
            agreement_rows.append(
                {
                    "schema_version": 1,
                    "dataset_id": records[specimen_key].dataset_id,
                    "specimen_id": records[specimen_key].specimen_id,
                    "specimen_key": specimen_key,
                    "task": task.value,
                    "analysis_scope": "REVIEWED_SPATIAL_RELATION",
                    "reference_version": reference_version,
                    "no_reliable_cue": bool(percept.no_reliable_cue),
                    "surface_cue_cells": json_tuple(partition["V"]),
                    "surface_ring_cells": json_tuple(partition["R"]),
                    **overlap_metrics(target, region_masks["V"]),
                }
            )
    report_scores = []
    full_input = []
    for specimen_key in sorted(references):
        reports = full_reports[specimen_key]
        for task_name in config.tasks:
            report = reports[task_name]
            score = evaluate_task_report(
                adapt_task_report_v2(report), references[specimen_key]
            )
            if score.formal_success is None:
                raise ValueError("reviewed full-input score lost formal eligibility")
            full_input.append(
                {
                    "schema_version": 1,
                    "dataset_id": records[specimen_key].dataset_id,
                    "specimen_id": records[specimen_key].specimen_id,
                    "specimen_key": specimen_key,
                    "task": task_name,
                    "analysis_scope": "FULL_INPUT_READER_DIAGNOSTIC",
                    "reference_version": reference_version,
                    "report_sha256": _report_digest(report),
                    "formal_success": bool(score.formal_success),
                    "iou": score.iou,
                    "recall": score.recall,
                    "relative_area_error": score.relative_area_error,
                    "failure_types": "|".join(score.failure_types),
                }
            )
    transition_steps = 0
    spatial_actions = []
    spatial_summaries = []
    for frame in frames:
        source_rows = tuple(frame.sort("step").to_dicts())
        first = source_rows[0]
        specimen_key = str(first["specimen_key"])
        task = Task(str(first["task"]))
        method = str(first["method"])
        seed = int(first["seed"])
        runtime = runtimes[specimen_key]
        reader = readers[specimen_key]
        state = zero_state(runtime.grid)
        measured_mask = np.zeros(runtime.grid.native_shape, dtype=np.bool_)
        reference = references[specimen_key]
        spatial = method in config.spatial_methods
        target = target_masks[(specimen_key, task.value)]
        region_masks = region_masks_by_specimen[specimen_key]
        state_rows = []
        action_rows = []
        stop_steps = _stop_steps(source_rows)
        identity = {
            "schema_version": 1,
            "dataset_id": str(first["dataset_id"]),
            "specimen_id": str(first["specimen_id"]),
            "specimen_key": specimen_key,
            "task": task.value,
            "method": method,
            "seed": seed,
            "reference_version": reference_version,
        }
        for index, source in enumerate(source_rows):
            report = reader.report(
                state=state,
                measured_mask=measured_mask,
                task=task,
            )
            if (
                int(source["step"]) != index
                or str(source["report_sha256"]) != _report_digest(report)
                or not math.isclose(
                    float(source["cost"]),
                    float(np.count_nonzero(measured_mask) / measured_mask.size),
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("reviewed report recovery identity changed")
            score = evaluate_task_report(adapt_task_report_v2(report), reference)
            if score.formal_success is None:
                raise ValueError("reviewed report score lost formal eligibility")
            if spatial and action_rows:
                action_rows[-1]["post_action_success"] = bool(
                    score.formal_success
                )
            if spatial:
                support = support_spatial_allocation(
                    report.support_positions,
                    target_mask=target,
                    region_masks=region_masks,
                )
                state_rows.append(
                    {
                        "step": index,
                        "cost": float(source["cost"]),
                        "has_target_measurement": bool(
                            np.any(measured_mask & target)
                        ),
                        "has_target_support": (
                            support["support_target_pixel_count"] > 0
                        ),
                        "locate_support_ready": (
                            task is Task.LOCATE
                            and _locate_support_ready(
                                report.support_positions, target
                            )
                        ),
                    }
                )
            report_scores.append(
                {
                    "schema_version": 1,
                    "dataset_id": str(source["dataset_id"]),
                    "specimen_id": str(source["specimen_id"]),
                    "specimen_key": specimen_key,
                    "task": task.value,
                    "method": method,
                    "seed": seed,
                    "step": int(source["step"]),
                    "cost": float(source["cost"]),
                    "report_sha256": str(source["report_sha256"]),
                    "rule_stop": bool(source["rule_stop"]),
                    "calibrated_stop_trigger": bool(
                        source["calibrated_stop_trigger"]
                    ),
                    "analysis_scope": "FROZEN_REPORT_REVIEWED_RESCORE",
                    "reference_version": reference_version,
                    "formal_success": bool(score.formal_success),
                    "iou": score.iou,
                    "recall": score.recall,
                    "relative_area_error": score.relative_area_error,
                    "failure_types": "|".join(score.failure_types),
                }
            )
            action_cell = int(source["action_cell"])
            if action_cell >= 0:
                action = InspectionCellAction(
                    action_cell,
                    int(source["action_from_level"]),
                    int(source["action_to_level"]),
                )
                if spatial:
                    allocation = action_spatial_allocation(
                        runtime.grid,
                        state=state,
                        action=action,
                        current_mask=measured_mask,
                        target_mask=target,
                        region_masks=region_masks,
                    )
                    added_positions = allocation["added_positions"]
                    native_count = measured_mask.size
                    action_row = {
                        **identity,
                        "analysis_scope": "REVIEWED_SPATIAL_RELATION",
                        "step": index,
                        "cost_before_action": float(source["cost"]),
                        "action_cell": action.cell_index,
                        "action_from_level": action.from_level,
                        "action_to_level": action.to_level,
                        "action_cell_region": allocation["action_cell_region"],
                        "added_pixel_count": allocation["added_pixel_count"],
                        "added_target_pixel_count": allocation[
                            "added_target_pixel_count"
                        ],
                        "added_v_pixel_count": allocation["added_v_pixel_count"],
                        "added_r_pixel_count": allocation["added_r_pixel_count"],
                        "added_o_pixel_count": allocation["added_o_pixel_count"],
                        "added_measurement_fraction": (
                            allocation["added_pixel_count"] / native_count
                        ),
                        "added_target_measurement_fraction": (
                            allocation["added_target_pixel_count"] / native_count
                        ),
                        "added_v_measurement_fraction": (
                            allocation["added_v_pixel_count"] / native_count
                        ),
                        "added_r_measurement_fraction": (
                            allocation["added_r_pixel_count"] / native_count
                        ),
                        "added_o_measurement_fraction": (
                            allocation["added_o_pixel_count"] / native_count
                        ),
                        "pre_action_success": bool(score.formal_success),
                        "post_action_success": None,
                        **support,
                        "in_s_bc_cal_prefix": (
                            stop_steps["S_BC_CAL"] is None
                            or index < int(stop_steps["S_BC_CAL"])
                        ),
                        "in_s_rule_prefix": (
                            stop_steps["S_RULE"] is None
                            or index < int(stop_steps["S_RULE"])
                        ),
                    }
                    action_row.pop("added_positions", None)
                    action_rows.append(action_row)
                    spatial_actions.append(action_row)
                else:
                    added_positions = action_added_positions_from_mask(
                        runtime.grid, state, action, measured_mask
                    )
                measured_mask[
                    added_positions[:, 0], added_positions[:, 1]
                ] = True
                state = apply_action(runtime.grid, state, action)
                transition_steps += 1
        if spatial:
            spatial_summaries.extend(
                _spatial_summary_rows(
                    identity,
                    states=state_rows,
                    actions=action_rows,
                    stop_steps=stop_steps,
                    terminal_step=int(source_rows[-1]["step"]),
                )
            )
    expected_steps = len(frames) * config.actions_per_episode
    if (
        transition_steps != expected_steps
        or transition_steps > config.reviewed_recovery_step_cap
    ):
        raise ValueError("reviewed recovery step cap exceeded")
    return {
        "report_scores": tuple(report_scores),
        "full_input_readout": tuple(full_input),
        "surface_reference_agreement": tuple(agreement_rows),
        "spatial_action_statistics": tuple(spatial_actions),
        "spatial_episode_summary": tuple(spatial_summaries),
        "recovery": {
            "reference_version": reference_version,
            "reviewed_specimen_count": len(references),
            "episode_count": len(frames),
            "cache_hit_episode_count": 0,
            "stored_action_transition_count": transition_steps,
            "conditional_transition_cap": config.reviewed_recovery_step_cap,
            "world_step_count": 0,
            "cache_reuse": "STORED_ACTIONS_AND_CELL_INTERPOLATION",
            "actor_forward_calls": 0,
            "stop_forward_calls": 0,
            "vlm_calls": 0,
            "training_updates": 0,
        },
    }


def json_tuple(values: tuple[int, ...]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


__all__ = [
    "FrozenVisibleReportReader",
    "action_spatial_allocation",
    "cell_partition_masks",
    "recover_reviewed_report_scores",
    "recover_spatial_analysis",
    "support_spatial_allocation",
]
