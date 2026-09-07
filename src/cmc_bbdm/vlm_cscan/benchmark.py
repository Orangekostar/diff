"""Causal method-matrix runner for the bounded VLM C-scan pilot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cmc_bbdm.inspection_agent.contracts import InspectionObservation
from cmc_bbdm.inspection_agent.state import action_added_positions

from .contracts import (
    BenchmarkTask,
    CScanReference,
    MethodId,
    SurfacePlan,
    TaskReport,
    TaskScore,
)
from .planning import (
    MacroMenuOption,
    MacroPlan,
    build_macro_menu,
    cell_evidence_statistics,
    initial_cell_order,
    plan_feedback_macro,
    plan_open_macro,
    visible_replan_event,
)
from .reader import read_sparse_evidence
from .references import evaluate_task_report
from .reporting import PublicStopTracker, build_task_report
from .route import compile_route, reference_full_raster_length
from .runtime import BenchmarkConfig, SpecimenRuntime
from .vlm import (
    ReplanChoiceCache,
    ReplanInference,
    ReplanRequest,
    SurfacePlanInference,
    VLMRawResponse,
)

_OPEN_METHODS = frozenset((MethodId.B0, MethodId.B1, MethodId.B2, MethodId.B3))
_VLM_METHODS = frozenset((MethodId.B3, MethodId.B5, MethodId.B6))


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    episode_rows: tuple[dict[str, object], ...]
    report_rows: tuple[dict[str, object], ...]
    action_count: int
    replan_count: int
    replan_actual_calls: int


def run_method_trajectory(
    runtime: SpecimenRuntime,
    *,
    method: MethodId,
    tasks: tuple[BenchmarkTask, ...],
    surface_inference: SurfacePlanInference,
    proxy_reference: CScanReference,
    config: BenchmarkConfig,
    cache_path: str | Path,
    replan_infer: Callable[[tuple[Image.Image, ...], str], VLMRawResponse] | None = None,
) -> TrajectoryResult:
    """Run one clean trajectory; fixed methods may share it across both tasks."""

    _validate_run_request(
        runtime,
        method=method,
        tasks=tasks,
        surface_inference=surface_inference,
        proxy_reference=proxy_reference,
        config=config,
    )
    values = config.values
    reader = values["reader"]
    planning = values["planning"]
    max_actions = int(planning["macro_action_limit"])
    exploration_period = int(planning["exploration_period"])
    max_replans = int(values["model"]["max_replans_per_episode"])
    surface_plan = surface_inference.plan
    cell_order = initial_cell_order(
        method,
        surface_plan=surface_plan,
        saliency_scores=runtime.saliency_scores,
        exploration_period=exploration_period,
    )
    observation = runtime.world.reset()
    trackers = {task: PublicStopTracker(task) for task in tasks}
    autonomous_reports: dict[BenchmarkTask, TaskReport | None] = {
        task: None for task in tasks
    }
    autonomous_steps: dict[BenchmarkTask, int | None] = {task: None for task in tasks}
    autonomous_costs: dict[BenchmarkTask, float | None] = {
        task: None for task in tasks
    }
    episode_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    route_position = tuple(float(value) for value in values["acquisition"]["start_position"])
    full_route_length = _cached_full_route_length(
        runtime.record.native_shape,
        route_position,
    )
    cumulative_transit = 0.0
    cumulative_scan = 0.0
    cumulative_turns = 0
    primitive_count = 0
    macro_index = 0
    replan_count = 0
    replan_actual_calls = 0
    replan_deployment_calls = 0
    replan_latency = 0.0
    replan_input_tokens = 0
    replan_output_tokens = 0
    previous_indication_cells: tuple[int, ...] = ()
    handled_events: set[str] = set()
    initial_vlm_calls = (
        surface_inference.original_call_count if method in _VLM_METHODS else 0
    )
    initial_vlm_latency = (
        surface_inference.original_latency_seconds if method in _VLM_METHODS else 0.0
    )
    initial_vlm_input_tokens = (
        surface_inference.original_input_tokens if method in _VLM_METHODS else 0
    )
    initial_vlm_output_tokens = (
        surface_inference.original_output_tokens if method in _VLM_METHODS else 0
    )

    evidence = _read_evidence(runtime, observation, reader)
    for task in tasks:
        report = _build_report(
            runtime,
            evidence,
            task,
            reader,
            cell_levels=observation.measurement_state.levels,
        )
        tracked = trackers[task].update(
            report,
            measured_count=0,
            cell_levels=observation.measurement_state.levels,
        )
        report_rows.append(
            _report_row(
                runtime,
                method=method,
                task=task,
                step=0,
                report=report,
                current_score=evaluate_task_report(report, proxy_reference),
                autonomous_report=tracked,
                autonomous_score=evaluate_task_report(tracked, proxy_reference),
                acquisition_cost=0.0,
                normalized_route_cost=0.0,
                public_stop_triggered=False,
                autonomous_stop_step=None,
                autonomous_stop_cost=None,
            )
        )

    while primitive_count < 192:
        cell_levels = observation.measurement_state.levels
        if method in _OPEN_METHODS:
            macro = plan_open_macro(
                cell_levels,
                cell_order=cell_order,
                max_actions=max_actions,
            )
        else:
            active_task = tasks[0]
            macro = plan_feedback_macro(
                active_task,
                evidence,
                runtime.grid,
                cell_levels=cell_levels,
                cell_order=cell_order,
                primitive_count=primitive_count,
                exploration_period=exploration_period,
                max_actions=max_actions,
                adaptive_rbf=method is MethodId.B7,
            )
            if method is MethodId.B6 and replan_count < max_replans:
                _measured, indication, _scores = cell_evidence_statistics(
                    evidence,
                    runtime.grid,
                )
                current_indication_cells = tuple(
                    int(cell) for cell in np.flatnonzero(indication)
                )
                initial_roi_surveyed = bool(surface_plan.priority_cells) and all(
                    cell_levels[cell] >= 0 for cell in surface_plan.priority_cells
                )
                event = visible_replan_event(
                    previous_indication_cells=previous_indication_cells,
                    current_indication_cells=current_indication_cells,
                    initial_priority_cells=surface_plan.priority_cells,
                    initial_roi_surveyed=initial_roi_surveyed,
                )
                if event is not None and event not in handled_events:
                    menu = build_macro_menu(
                        active_task,
                        evidence,
                        runtime.grid,
                        cell_levels=cell_levels,
                        cell_order=cell_order,
                        primitive_count=primitive_count,
                        exploration_period=exploration_period,
                        max_actions=max_actions,
                        report_allowed=False,
                    )
                    if menu:
                        replan = _resolve_replan(
                            runtime,
                            config=config,
                            task=active_task,
                            event=event,
                            menu=menu,
                            indication_cells=current_indication_cells,
                            observation=observation,
                            cache_path=cache_path,
                            infer=replan_infer,
                        )
                        macro = next(
                            option.plan
                            for option in menu
                            if option.menu_id == replan.choice.menu_id
                        )
                        replan_count += 1
                        replan_actual_calls += replan.actual_call_count
                        replan_deployment_calls += replan.original_call_count
                        replan_latency += replan.original_latency_seconds
                        replan_input_tokens += replan.original_input_tokens
                        replan_output_tokens += replan.original_output_tokens
                    handled_events.add(event)
                previous_indication_cells = current_indication_cells
        if not macro.actions:
            break
        macro_index += 1
        for action_index, action in enumerate(macro.actions, start=1):
            legal = runtime.world.legal_actions(observation)
            if action not in legal:
                raise RuntimeError("planner emitted an action outside the causal world")
            added = action_added_positions(
                runtime.grid,
                observation.measurement_state,
                action,
            )
            route = compile_route(
                added,
                native_shape=runtime.record.native_shape,
                start_position=route_position,
            )
            observation = runtime.world.step(observation, action)
            route_position = route.end_position
            cumulative_transit += route.transit_length
            cumulative_scan += route.scan_length
            cumulative_turns += route.turn_count
            primitive_count += 1
            evidence = _read_evidence(runtime, observation, reader)
            normalized_route = (
                (cumulative_transit + cumulative_scan) / full_route_length
                if full_route_length
                else 0.0
            )
            for task in tasks:
                current_report = _build_report(
                    runtime,
                    evidence,
                    task,
                    reader,
                    cell_levels=observation.measurement_state.levels,
                )
                tracked = trackers[task].update(
                    current_report,
                    measured_count=observation.exact_acquired_count,
                    cell_levels=observation.measurement_state.levels,
                )
                just_stopped = (
                    tracked.public_complete and autonomous_reports[task] is None
                )
                if just_stopped:
                    autonomous_reports[task] = tracked
                    autonomous_steps[task] = primitive_count
                    autonomous_costs[task] = observation.effective_budget
                frozen = autonomous_reports[task] or tracked
                current_score = evaluate_task_report(current_report, proxy_reference)
                autonomous_score = evaluate_task_report(frozen, proxy_reference)
                common = _episode_common(
                    runtime,
                    method=method,
                    task=task,
                    step=primitive_count,
                    macro_index=macro_index,
                    macro=macro,
                    action_index=action_index,
                    acquisition_cost=observation.effective_budget,
                    initial_vlm_calls=initial_vlm_calls,
                    initial_vlm_latency=initial_vlm_latency,
                    initial_vlm_input_tokens=initial_vlm_input_tokens,
                    initial_vlm_output_tokens=initial_vlm_output_tokens,
                    replan_count=replan_count,
                    replan_actual_calls=replan_actual_calls,
                    replan_deployment_calls=replan_deployment_calls,
                    replan_latency=replan_latency,
                    replan_input_tokens=replan_input_tokens,
                    replan_output_tokens=replan_output_tokens,
                )
                episode_rows.append(
                    {
                        **common,
                        "cell_index": action.cell_index,
                        "from_level": action.from_level,
                        "to_level": action.to_level,
                        "newly_revealed_count": len(added),
                        "exact_acquired_count": observation.exact_acquired_count,
                        "transit_length": route.transit_length,
                        "scan_length": route.scan_length,
                        "route_length": route.total_length,
                        "cumulative_transit_length": cumulative_transit,
                        "cumulative_scan_length": cumulative_scan,
                        "cumulative_route_length": cumulative_transit
                        + cumulative_scan,
                        "normalized_route_cost": normalized_route,
                        "turn_count": route.turn_count,
                        "cumulative_turn_count": cumulative_turns,
                        "revisit_count": route.revisit_count,
                        "route_end_x": route_position[0],
                        "route_end_y": route_position[1],
                        "public_stop_triggered": just_stopped,
                        "post_autonomous_stop": autonomous_reports[task] is not None
                        and not just_stopped,
                        "autonomous_stop_step": autonomous_steps[task],
                        "autonomous_stop_cost": autonomous_costs[task],
                        "reason_code": macro.reason_code,
                    }
                )
                report_rows.append(
                    _report_row(
                        runtime,
                        method=method,
                        task=task,
                        step=primitive_count,
                        report=current_report,
                        current_score=current_score,
                        autonomous_report=frozen,
                        autonomous_score=autonomous_score,
                        acquisition_cost=observation.effective_budget,
                        normalized_route_cost=normalized_route,
                        public_stop_triggered=just_stopped,
                        autonomous_stop_step=autonomous_steps[task],
                        autonomous_stop_cost=autonomous_costs[task],
                    )
                )
        if observation.effective_budget >= 1.0 - 1.0e-15:
            break
    if primitive_count != 192 or observation.effective_budget < 1.0 - 1.0e-15:
        raise RuntimeError("trajectory did not reach the registered full-raster endpoint")
    for task in tasks:
        terminal = (
            autonomous_reports[task].reason_code
            if autonomous_reports[task] is not None
            else "RESOURCE_EXHAUSTED"
        )
        task_reports = [row for row in report_rows if row["task"] == task.value]
        first_hindsight = next(
            (
                float(row["acquisition_cost"])
                for row in task_reports
                if row["proxy_diagnostic_success"]
            ),
            None,
        )
        for row in episode_rows:
            if row["task"] == task.value:
                row["terminal_reason"] = terminal
                row["first_hindsight_sufficient_cost"] = first_hindsight
        for row in report_rows:
            if row["task"] == task.value:
                row["terminal_reason"] = terminal
                row["first_hindsight_sufficient_cost"] = first_hindsight
    return TrajectoryResult(
        episode_rows=tuple(episode_rows),
        report_rows=tuple(report_rows),
        action_count=primitive_count,
        replan_count=replan_count,
        replan_actual_calls=replan_actual_calls,
    )


def render_visible_evidence(observation: InspectionObservation) -> tuple[Image.Image, str]:
    """Render only measured RGB samples; neutral gray denotes unknown pixels."""

    if type(observation) is not InspectionObservation:
        raise TypeError("issued observation is required")
    canvas = np.full((*observation.native_shape, 3), 127, dtype=np.uint8)
    if observation.exact_acquired_count:
        rows, columns = observation.acquired_positions.T
        canvas[rows, columns] = observation.measurement_values
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for index in range(1, 8):
        x = round(index * image.width / 8)
        y = round(index * image.height / 8)
        draw.line((x, 0, x, image.height - 1), fill=(0, 255, 255, 90), width=1)
        draw.line((0, y, image.width - 1, y), fill=(0, 255, 255, 90), width=1)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return image, hashlib.sha256(buffer.getvalue()).hexdigest()


def _resolve_replan(
    runtime: SpecimenRuntime,
    *,
    config: BenchmarkConfig,
    task: BenchmarkTask,
    event: str,
    menu: tuple[MacroMenuOption, ...],
    indication_cells: tuple[int, ...],
    observation: InspectionObservation,
    cache_path: str | Path,
    infer: Callable[[tuple[Image.Image, ...], str], VLMRawResponse] | None,
) -> ReplanInference:
    menu_rows = [
        {
            "menu_id": option.menu_id,
            "skill": option.plan.skill.value,
            "actions": [action.cell_index for action in option.plan.actions],
            "reason_code": option.plan.reason_code,
        }
        for option in menu
    ]
    measured_cells = tuple(
        index
        for index, level in enumerate(observation.measurement_state.levels)
        if level >= 0
    )
    prompt = (
        "你是无损检测工具规划器。灰色像素是未知区，不是正常测量；只能依据已测超声和表面图。"
        "不得猜测未测区域，不得输出自由坐标。\n"
        f"task={task.value}; event={event}; acquisition_cost={observation.effective_budget:.9f}; "
        f"measured_cells={list(measured_cells)}; indication_cells={list(indication_cells)}.\n"
        f"legal_menu={json.dumps(menu_rows, ensure_ascii=False, separators=(',', ':'))}\n"
        "只输出 JSON：{\"menu_id\":\"m1\",\"reason_code\":\"简短可见证据理由\"}。"
    )
    evidence_image, evidence_image_sha256 = render_visible_evidence(observation)
    model = config.values["model"]
    menu_sha256 = hashlib.sha256(
        json.dumps(menu_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    request = ReplanRequest(
        model_repository=model["repository"],
        model_revision=model["revision"],
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        clean_image_sha256=runtime.render.clean_sha256,
        evidence_image_sha256=evidence_image_sha256,
        evidence_state_sha256=observation.state_sha256,
        menu_sha256=menu_sha256,
        task=task.value,
        event_code=event,
    )

    def invoke(current_prompt: str) -> VLMRawResponse:
        if infer is None:
            raise RuntimeError("uncached VLM replan requires the configured backend")
        return infer((runtime.render.clean, evidence_image), current_prompt)

    return ReplanChoiceCache(cache_path).resolve(
        request,
        prompt=prompt,
        legal_menu_ids=tuple(option.menu_id for option in menu),
        infer=invoke,
    )


def _read_evidence(
    runtime: SpecimenRuntime,
    observation: InspectionObservation,
    reader: dict[str, object],
):
    return read_sparse_evidence(
        native_shape=runtime.record.native_shape,
        positions=observation.acquired_positions,
        values=observation.measurement_values,
        background_rgb=runtime.background_rgb,
        distance_threshold=float(reader["distance_threshold"]),
    )


def _build_report(
    runtime: SpecimenRuntime,
    evidence,
    task: BenchmarkTask,
    reader: dict[str, object],
    *,
    cell_levels: tuple[int, ...],
) -> TaskReport:
    return build_task_report(
        evidence,
        runtime.grid,
        cell_levels=cell_levels,
        task=task,
        cell_indication_fraction=float(reader["cell_indication_fraction"]),
        minimum_component_pixels=int(reader["minimum_component_pixels"]),
    )


def _episode_common(
    runtime: SpecimenRuntime,
    *,
    method: MethodId,
    task: BenchmarkTask,
    step: int,
    macro_index: int,
    macro: MacroPlan,
    action_index: int,
    acquisition_cost: float,
    initial_vlm_calls: int,
    initial_vlm_latency: float,
    initial_vlm_input_tokens: int,
    initial_vlm_output_tokens: int,
    replan_count: int,
    replan_actual_calls: int,
    replan_deployment_calls: int,
    replan_latency: float,
    replan_input_tokens: int,
    replan_output_tokens: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": runtime.record.dataset_id,
        "specimen_id": runtime.record.specimen_id,
        "specimen_key": runtime.record.specimen_key,
        "method": method.value,
        "task": task.value,
        "step": step,
        "macro_index": macro_index,
        "macro_action_index": action_index,
        "macro_skill": macro.skill.value,
        "acquisition_cost": acquisition_cost,
        "initial_vlm_deployment_calls": initial_vlm_calls,
        "initial_vlm_latency_seconds": initial_vlm_latency,
        "initial_vlm_input_tokens": initial_vlm_input_tokens,
        "initial_vlm_output_tokens": initial_vlm_output_tokens,
        "replan_count": replan_count,
        "replan_actual_calls": replan_actual_calls,
        "replan_deployment_calls": replan_deployment_calls,
        "replan_latency_seconds": replan_latency,
        "replan_input_tokens": replan_input_tokens,
        "replan_output_tokens": replan_output_tokens,
    }


def _report_row(
    runtime: SpecimenRuntime,
    *,
    method: MethodId,
    task: BenchmarkTask,
    step: int,
    report: TaskReport,
    current_score: TaskScore,
    autonomous_report: TaskReport,
    autonomous_score: TaskScore,
    acquisition_cost: float,
    normalized_route_cost: float,
    public_stop_triggered: bool,
    autonomous_stop_step: int | None,
    autonomous_stop_cost: float | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": runtime.record.dataset_id,
        "specimen_id": runtime.record.specimen_id,
        "specimen_key": runtime.record.specimen_key,
        "method": method.value,
        "task": task.value,
        "step": step,
        "acquisition_cost": acquisition_cost,
        "normalized_route_cost": normalized_route_cost,
        "report_confidence": report.confidence,
        "report_reason_code": report.reason_code,
        "prediction_area_pixels": int(np.count_nonzero(report.predicted_mask)),
        "prediction_mask_sha256": _mask_sha256(report.predicted_mask),
        "prediction_bbox_json": _bbox_json(report.predicted_mask),
        "support_count": len(report.support_positions),
        "support_row_count": len(set(report.support_positions[:, 0])),
        "support_column_count": len(set(report.support_positions[:, 1])),
        "reference_eligible": current_score.reference_eligible,
        "formal_success": current_score.formal_success,
        "proxy_diagnostic_success": current_score.proxy_success,
        "proxy_iou": current_score.iou,
        "proxy_recall": current_score.recall,
        "proxy_relative_area_error": current_score.relative_area_error,
        "proxy_failure_types_json": json.dumps(current_score.failure_types),
        "public_stop_triggered": public_stop_triggered,
        "autonomous_stop_step": autonomous_stop_step,
        "autonomous_stop_cost": autonomous_stop_cost,
        "autonomous_report_frozen": autonomous_stop_step is not None,
        "autonomous_prediction_mask_sha256": _mask_sha256(
            autonomous_report.predicted_mask
        ),
        "autonomous_formal_success": autonomous_score.formal_success,
        "autonomous_proxy_diagnostic_success": autonomous_score.proxy_success,
        "autonomous_proxy_failure_types_json": json.dumps(
            autonomous_score.failure_types
        ),
        "reference_status": "REFERENCE_PENDING",
    }


def _mask_sha256(mask: np.ndarray) -> str:
    values = np.ascontiguousarray(mask, dtype=np.bool_)
    return hashlib.sha256(
        json.dumps(values.shape, separators=(",", ":")).encode("ascii")
        + values.tobytes(order="C")
    ).hexdigest()


def _bbox_json(mask: np.ndarray) -> str:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return "null"
    height, width = mask.shape
    payload = {
        "x0": float(columns.min() / max(width - 1, 1)),
        "y0": float(rows.min() / max(height - 1, 1)),
        "x1": float(columns.max() / max(width - 1, 1)),
        "y1": float(rows.max() / max(height - 1, 1)),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _validate_run_request(
    runtime: SpecimenRuntime,
    *,
    method: MethodId,
    tasks: tuple[BenchmarkTask, ...],
    surface_inference: SurfacePlanInference,
    proxy_reference: CScanReference,
    config: BenchmarkConfig,
) -> None:
    if (
        type(runtime) is not SpecimenRuntime
        or type(method) is not MethodId
        or type(tasks) is not tuple
        or not tasks
        or len(set(tasks)) != len(tasks)
        or any(type(task) is not BenchmarkTask for task in tasks)
        or type(surface_inference) is not SurfacePlanInference
        or type(surface_inference.plan) is not SurfacePlan
        or type(proxy_reference) is not CScanReference
        or proxy_reference.specimen_key != runtime.record.specimen_key
        or proxy_reference.certain_mask.shape != runtime.record.native_shape
        or type(config) is not BenchmarkConfig
        or (method not in _OPEN_METHODS and len(tasks) != 1)
    ):
        raise ValueError("trajectory request is invalid")


@lru_cache(maxsize=16)
def _cached_full_route_length(
    native_shape: tuple[int, int], start_position: tuple[float, float]
) -> float:
    return reference_full_raster_length(native_shape, start_position)


__all__ = [
    "TrajectoryResult",
    "render_visible_evidence",
    "run_method_trajectory",
]
