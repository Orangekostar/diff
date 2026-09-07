"""Initial and feedback planning over the legal 8x8 action space."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.mva.oracle import uniform_cell_order

from .contracts import BenchmarkTask, EvidenceMap, MethodId, SurfacePlan


class MacroSkill(str, Enum):
    SURVEY_ROI = "SURVEY_ROI"
    EXPAND_BOUNDARY = "EXPAND_BOUNDARY"
    REFINE_REGION = "REFINE_REGION"
    BROADEN_SEARCH = "BROADEN_SEARCH"
    REPORT = "REPORT"


@dataclass(frozen=True, slots=True)
class MacroPlan:
    skill: MacroSkill
    actions: tuple[InspectionCellAction, ...]
    reason_code: str


@dataclass(frozen=True, slots=True)
class MacroMenuOption:
    menu_id: str
    plan: MacroPlan


def geometry_cell_order() -> tuple[int, ...]:
    return tuple(int(cell) for cell in uniform_cell_order())


def initial_cell_order(
    method: MethodId,
    *,
    surface_plan: SurfacePlan,
    saliency_scores: np.ndarray,
) -> tuple[int, ...]:
    if type(method) is not MethodId or type(surface_plan) is not SurfacePlan:
        raise TypeError("typed method and surface plan are required")
    scores = np.asarray(saliency_scores, dtype=np.float64)
    if scores.shape != (64,) or not np.all(np.isfinite(scores)):
        raise ValueError("saliency scores are invalid")
    geometry = geometry_cell_order()
    if method in {MethodId.B0, MethodId.B4, MethodId.B7}:
        return geometry
    if method is MethodId.B1:
        return tuple(
            sorted(
                range(64),
                key=lambda cell: (
                    (cell // 8 - 3.5) ** 2 + (cell % 8 - 3.5) ** 2,
                    cell,
                ),
            )
        )
    if method is MethodId.B2:
        return tuple(sorted(range(64), key=lambda cell: (-scores[cell], cell)))
    priority = surface_plan.priority_cells
    return (*priority, *(cell for cell in geometry if cell not in priority))


def exploration_interleaved_order(
    preferred_order: tuple[int, ...], *, period: int
) -> tuple[int, ...]:
    if (
        type(preferred_order) is not tuple
        or set(preferred_order) != set(range(64))
        or len(preferred_order) != 64
        or type(period) is not int
        or period < 2
    ):
        raise ValueError("exploration schedule is invalid")
    geometry = geometry_cell_order()
    output: list[int] = []
    preferred_index = 0
    geometry_index = 0
    while len(output) < 64:
        use_geometry = (len(output) + 1) % period == 0
        source = geometry if use_geometry else preferred_order
        index = geometry_index if use_geometry else preferred_index
        while index < len(source) and source[index] in output:
            index += 1
        if index == len(source):
            source = preferred_order if use_geometry else geometry
            index = 0
            while source[index] in output:
                index += 1
        selected = source[index]
        output.append(selected)
        if source is geometry:
            geometry_index = index + 1
        else:
            preferred_index = index + 1
    return tuple(output)


def plan_open_macro(
    cell_levels: tuple[int, ...],
    *,
    cell_order: tuple[int, ...],
    max_actions: int,
) -> MacroPlan:
    _validate_planning_inputs(cell_levels, cell_order, max_actions)
    active_level = min(cell_levels)
    if active_level == 2:
        return MacroPlan(MacroSkill.REPORT, (), "FULL_RASTER_REACHED")
    actions = tuple(
        InspectionCellAction(cell, active_level, active_level + 1)
        for cell in cell_order
        if cell_levels[cell] == active_level
    )[:max_actions]
    skill = MacroSkill.BROADEN_SEARCH if active_level == -1 else MacroSkill.REFINE_REGION
    return MacroPlan(skill, actions, f"OPEN_LEVEL_{active_level}_TO_{active_level + 1}")


def plan_feedback_macro(
    task: BenchmarkTask,
    evidence: EvidenceMap,
    grid: AcquisitionGrid,
    *,
    cell_levels: tuple[int, ...],
    cell_order: tuple[int, ...],
    primitive_count: int,
    exploration_period: int,
    max_actions: int,
    forced_skill: MacroSkill | None = None,
    adaptive_rbf: bool = False,
) -> MacroPlan:
    _validate_planning_inputs(cell_levels, cell_order, max_actions)
    if (
        type(task) is not BenchmarkTask
        or type(evidence) is not EvidenceMap
        or type(grid) is not AcquisitionGrid
        or evidence.states.shape != grid.native_shape
        or type(primitive_count) is not int
        or primitive_count < 0
        or type(exploration_period) is not int
        or exploration_period < 2
        or (forced_skill is not None and type(forced_skill) is not MacroSkill)
        or type(adaptive_rbf) is not bool
    ):
        raise ValueError("feedback planning request is invalid")
    measured, indication, scores = cell_evidence_statistics(evidence, grid)
    positive = [
        cell
        for cell in range(64)
        if indication[cell] > 0 and scores[cell] >= 0.18
    ]
    boundary = _unmeasured_neighbors(positive, cell_levels)
    refine = [cell for cell in positive if cell_levels[cell] < 2]
    broaden = [cell for cell in cell_order if cell_levels[cell] == -1]
    other_refine = [
        cell for cell in cell_order if 0 <= cell_levels[cell] < 2 and cell not in refine
    ]
    if adaptive_rbf and broaden:
        broaden = _rbf_candidate_order(
            broaden,
            measured=measured,
            scores=scores,
        )
    priorities: list[tuple[MacroSkill, list[int]]]
    if forced_skill is MacroSkill.EXPAND_BOUNDARY:
        priorities = [(MacroSkill.EXPAND_BOUNDARY, boundary)]
    elif forced_skill is MacroSkill.REFINE_REGION:
        priorities = [(MacroSkill.REFINE_REGION, refine)]
    elif forced_skill in {MacroSkill.BROADEN_SEARCH, MacroSkill.SURVEY_ROI}:
        priorities = [(forced_skill, broaden)]
    elif task is BenchmarkTask.CHARACTERIZE:
        priorities = [
            (MacroSkill.EXPAND_BOUNDARY, boundary),
            (MacroSkill.REFINE_REGION, refine),
            (MacroSkill.BROADEN_SEARCH, broaden),
            (MacroSkill.REFINE_REGION, other_refine),
        ]
    else:
        priorities = [
            (MacroSkill.REFINE_REGION, refine),
            (MacroSkill.EXPAND_BOUNDARY, boundary),
            (MacroSkill.BROADEN_SEARCH, broaden),
            (MacroSkill.REFINE_REGION, other_refine),
        ]
    selected: list[int] = []
    selected_skill = MacroSkill.BROADEN_SEARCH
    for skill, candidates in priorities:
        for cell in candidates:
            if cell not in selected and cell_levels[cell] < 2:
                if not selected:
                    selected_skill = skill
                selected.append(cell)
            if len(selected) == max_actions:
                break
        if len(selected) == max_actions:
            break
    if forced_skill is not None and not selected:
        return plan_feedback_macro(
            task,
            evidence,
            grid,
            cell_levels=cell_levels,
            cell_order=cell_order,
            primitive_count=primitive_count,
            exploration_period=exploration_period,
            max_actions=max_actions,
            forced_skill=None,
            adaptive_rbf=adaptive_rbf,
        )
    geometry = geometry_cell_order()
    if selected and (primitive_count + len(selected)) % exploration_period == 0:
        exploratory = next(
            (cell for cell in geometry if cell_levels[cell] == -1 and cell not in selected),
            None,
        )
        if exploratory is not None:
            selected[-1] = exploratory
    actions = tuple(
        InspectionCellAction(cell, cell_levels[cell], cell_levels[cell] + 1)
        for cell in selected
    )
    if not actions:
        return MacroPlan(MacroSkill.REPORT, (), "FULL_RASTER_REACHED")
    reason = (
        "OBSERVED_RBF_ADAPTIVE"
        if adaptive_rbf
        else f"VISIBLE_{selected_skill.value}"
    )
    return MacroPlan(selected_skill, actions, reason)


def build_macro_menu(
    task: BenchmarkTask,
    evidence: EvidenceMap,
    grid: AcquisitionGrid,
    *,
    cell_levels: tuple[int, ...],
    cell_order: tuple[int, ...],
    primitive_count: int,
    exploration_period: int,
    max_actions: int,
    report_allowed: bool,
) -> tuple[MacroMenuOption, ...]:
    options = []
    for skill in (
        MacroSkill.EXPAND_BOUNDARY,
        MacroSkill.REFINE_REGION,
        MacroSkill.BROADEN_SEARCH,
    ):
        plan = plan_feedback_macro(
            task,
            evidence,
            grid,
            cell_levels=cell_levels,
            cell_order=cell_order,
            primitive_count=primitive_count,
            exploration_period=exploration_period,
            max_actions=max_actions,
            forced_skill=skill,
        )
        if plan.actions and all(option.plan.actions != plan.actions for option in options):
            options.append(MacroMenuOption(f"m{len(options) + 1}", plan))
    if report_allowed:
        options.append(
            MacroMenuOption(
                f"m{len(options) + 1}",
                MacroPlan(MacroSkill.REPORT, (), "PUBLIC_STOP_ALLOWED"),
            )
        )
    return tuple(options[:6])


def cell_evidence_statistics(
    evidence: EvidenceMap, grid: AcquisitionGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(evidence) is not EvidenceMap or type(grid) is not AcquisitionGrid:
        raise TypeError("typed evidence and grid are required")
    measured = np.zeros(64, dtype=np.int64)
    indication = np.zeros(64, dtype=np.int64)
    scores = np.zeros(64, dtype=np.float64)
    for cell in grid.cells:
        row_start = grid.row_boundaries[cell.row]
        row_stop = grid.row_boundaries[cell.row + 1] + (1 if cell.row == 7 else 0)
        column_start = grid.column_boundaries[cell.column]
        column_stop = grid.column_boundaries[cell.column + 1] + (
            1 if cell.column == 7 else 0
        )
        owned = (slice(row_start, row_stop), slice(column_start, column_stop))
        mask = evidence.measured_mask[owned]
        measured[cell.index] = np.count_nonzero(mask)
        indication[cell.index] = np.count_nonzero(evidence.indication_mask[owned])
        if measured[cell.index]:
            scores[cell.index] = float(evidence.scores[owned][mask].mean())
    return measured, indication, scores


def visible_replan_event(
    *,
    previous_indication_cells: tuple[int, ...],
    current_indication_cells: tuple[int, ...],
    initial_priority_cells: tuple[int, ...],
    initial_roi_surveyed: bool,
) -> str | None:
    previous = set(previous_indication_cells)
    current = set(current_indication_cells)
    initial = set(initial_priority_cells)
    if initial_roi_surveyed and initial and not (current & initial):
        return "INITIAL_ROI_UNSUPPORTED"
    if (current - previous) - initial:
        return "NEW_INDICATION_OUTSIDE_INITIAL_ROI"
    return None


def _validate_planning_inputs(
    cell_levels: tuple[int, ...], cell_order: tuple[int, ...], max_actions: int
) -> None:
    if (
        type(cell_levels) is not tuple
        or len(cell_levels) != 64
        or any(level not in (-1, 0, 1, 2) for level in cell_levels)
        or type(cell_order) is not tuple
        or set(cell_order) != set(range(64))
        or len(cell_order) != 64
        or type(max_actions) is not int
        or not 1 <= max_actions <= 4
    ):
        raise ValueError("planning state is invalid")


def _unmeasured_neighbors(
    cells: list[int], cell_levels: tuple[int, ...]
) -> list[int]:
    output = []
    for cell in cells:
        row, column = divmod(cell, 8)
        for neighbor_row, neighbor_column in (
            (row - 1, column),
            (row, column - 1),
            (row, column + 1),
            (row + 1, column),
        ):
            if 0 <= neighbor_row < 8 and 0 <= neighbor_column < 8:
                neighbor = neighbor_row * 8 + neighbor_column
                if cell_levels[neighbor] == -1 and neighbor not in output:
                    output.append(neighbor)
    return output


def _rbf_candidate_order(
    candidates: list[int], *, measured: np.ndarray, scores: np.ndarray
) -> list[int]:
    observed = np.flatnonzero(measured)
    if not len(observed):
        geometry = geometry_cell_order()
        return sorted(candidates, key=geometry.index)
    observed_xy = np.asarray([divmod(int(cell), 8) for cell in observed], dtype=np.float64)
    ranked = []
    for cell in candidates:
        xy = np.asarray(divmod(cell, 8), dtype=np.float64)
        distances = np.linalg.norm(observed_xy - xy, axis=1) / np.sqrt(98.0)
        weights = np.exp(-8.0 * distances**2)
        prediction = float(np.sum(weights * scores[observed]) / max(weights.sum(), 1e-12))
        uncertainty = float(distances.min())
        ranked.append((-(prediction + 0.25 * uncertainty), cell))
    return [cell for _score, cell in sorted(ranked)]


__all__ = [
    "MacroMenuOption",
    "MacroPlan",
    "MacroSkill",
    "build_macro_menu",
    "cell_evidence_statistics",
    "exploration_interleaved_order",
    "geometry_cell_order",
    "initial_cell_order",
    "plan_feedback_macro",
    "plan_open_macro",
    "visible_replan_event",
]
