"""Read-only diagnostics for frozen learned C-scan trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl
import yaml
from PIL import Image

from .artifacts import (
    _atomic_write,
    write_checksums,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)
from .contracts import Task
from .metrics import MetricRecord, StepSnapshot, exact_step_integral
from .supplement_analysis import make_domain_bootstrap_draws, paired_domain_bootstrap

STOP_SYSTEMS = ("S_BC_CAL", "S_RULE")
FROZEN_PROCESS_BASE_SHA = "fe58de39298c412d0829580cfda40b7c7c4c53e9"


@dataclass(frozen=True, slots=True)
class FrozenProcessConfig:
    path: Path
    project_root: Path
    config_sha256: str
    repository_base_sha: str
    source_result_root: Path
    surface_percept_cache: Path
    parent_config_path: Path
    output_root: Path
    artifact_root: Path
    methods: tuple[str, ...]
    tasks: tuple[str, ...]
    spatial_methods: tuple[str, ...]
    stop_thresholds: dict[str, float]
    expected_input_hashes: MappingProxyType[str, str]
    expected_external_hashes: MappingProxyType[str, str]
    physical_specimens: int
    domains: int
    episode_count: int
    states_per_episode: int
    actions_per_episode: int
    trajectory_rows: int
    bootstrap_seed: int
    bootstrap_replicates: int
    training_updates: int
    vlm_calls: int
    actor_stop_forward_calls: int
    base_recovery_step_cap: int
    reviewed_recovery_step_cap: int
    values: MappingProxyType[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_input(
    path: str | Path, expected_sha256: str, *, label: str
) -> str:
    source = Path(path)
    if (
        source.is_symlink()
        or not source.is_file()
        or type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or not label
    ):
        raise ValueError(f"{label} frozen input is invalid")
    actual = _file_sha256(source)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} hash changed: expected {expected_sha256}, got {actual}"
        )
    return actual


def _project_path(root: Path, value: object) -> Path:
    if type(value) is not str or not value:
        raise ValueError("frozen process path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("frozen process path is invalid")
    resolved = (root / relative).resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError("frozen process path leaves the project root")
    return resolved


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or not value
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} is invalid")
    return tuple(value)


def load_frozen_process_config(
    path: str | Path, *, project_root: str | Path
) -> FrozenProcessConfig:
    """Load and verify the immutable source bindings for the analysis."""

    root = Path(project_root).resolve(strict=True)
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError("frozen process config is invalid YAML") from error
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("stage") != "BC_CSCAN_FROZEN_PROCESS_ANALYSIS"
        or payload.get("repository_base_sha") != FROZEN_PROCESS_BASE_SHA
    ):
        raise ValueError("frozen process config identity is invalid")
    paths = payload.get("paths")
    matrix = payload.get("matrix")
    stop = payload.get("stop")
    statistics = payload.get("statistics")
    limits = payload.get("resource_limits")
    frozen = payload.get("frozen_inputs")
    external = payload.get("external_frozen_inputs")
    if any(
        type(value) is not dict
        for value in (paths, matrix, stop, statistics, limits, frozen, external)
    ):
        raise ValueError("frozen process config sections are invalid")
    assert isinstance(paths, dict)
    assert isinstance(matrix, dict)
    assert isinstance(stop, dict)
    assert isinstance(statistics, dict)
    assert isinstance(limits, dict)
    assert isinstance(frozen, dict)
    assert isinstance(external, dict)
    methods = _string_tuple(matrix.get("methods"), label="frozen methods")
    tasks = _string_tuple(matrix.get("tasks"), label="frozen tasks")
    spatial_methods = _string_tuple(
        matrix.get("spatial_methods"), label="spatial methods"
    )
    if (
        methods
        != (
            "R_BALANCED_P4",
            "R_BALANCED_P8",
            "BC_S1",
            "BC_S2",
            "BC_S3",
            "BC_NO_VLM_S1",
            "BC_NO_US_FEEDBACK_S1",
        )
        or tasks != ("LOCATE", "CHARACTERIZE")
        or spatial_methods != ("R_BALANCED_P8", "BC_S1", "BC_S2", "BC_S3")
    ):
        raise ValueError("frozen process matrix identity is invalid")
    thresholds = stop.get("thresholds")
    if (
        stop.get("model") != "S_BC_CAL"
        or type(thresholds) is not dict
        or set(thresholds) != set(tasks)
        or any(float(thresholds[task]) != 0.99 for task in tasks)
    ):
        raise ValueError("frozen STOP identity is invalid")
    fixed_counts = {
        "physical_specimens": 24,
        "domains": 6,
        "episode_count": 336,
        "states_per_episode": 193,
        "actions_per_episode": 192,
        "trajectory_rows": 64848,
    }
    if any(matrix.get(name) != value for name, value in fixed_counts.items()):
        raise ValueError("frozen process matrix counts changed")
    fixed_limits = {
        "training_updates": 0,
        "vlm_calls": 0,
        "actor_stop_forward_calls": 0,
        "base_recovery_step_cap": 40000,
        "reviewed_recovery_step_cap": 60000,
        "cpu_processes": 4,
        "figure_groups": 3,
        "case_specimens": 6,
    }
    if any(limits.get(name) != value for name, value in fixed_limits.items()):
        raise ValueError("frozen process resource limits changed")
    if (
        statistics.get("bootstrap_seed") != 2026090801
        or statistics.get("bootstrap_replicates") != 5000
        or float(statistics.get("exploratory_confidence", -1)) != 0.95
        or float(statistics.get("path_b_confidence", -1)) != 0.975
        or float(statistics.get("completion_noninferiority_margin", 0)) != -0.05
    ):
        raise ValueError("frozen process statistical contract changed")
    source_result_root = _project_path(root, paths.get("source_result_root"))
    output_root = _project_path(root, paths.get("output_root"))
    artifact_root = _project_path(root, paths.get("artifact_root"))
    surface_cache = _project_path(root, paths.get("surface_percept_cache"))
    parent_config = _project_path(root, paths.get("parent_config"))
    if source_result_root in output_root.parents or output_root in source_result_root.parents:
        raise ValueError("source and output roots overlap")
    expected_hashes = {}
    for relative, expected in frozen.items():
        if type(relative) is not str or type(expected) is not str:
            raise ValueError("frozen input hash binding is invalid")
        verify_frozen_input(source_result_root / relative, expected, label=relative)
        expected_hashes[relative] = expected
    expected_external = {}
    for relative, expected in external.items():
        if type(relative) is not str or type(expected) is not str:
            raise ValueError("external frozen input hash binding is invalid")
        verify_frozen_input(_project_path(root, relative), expected, label=relative)
        expected_external[relative] = expected
    return FrozenProcessConfig(
        path=source,
        project_root=root,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        repository_base_sha=FROZEN_PROCESS_BASE_SHA,
        source_result_root=source_result_root,
        surface_percept_cache=surface_cache,
        parent_config_path=parent_config,
        output_root=output_root,
        artifact_root=artifact_root,
        methods=methods,
        tasks=tasks,
        spatial_methods=spatial_methods,
        stop_thresholds={task: float(thresholds[task]) for task in tasks},
        expected_input_hashes=MappingProxyType(expected_hashes),
        expected_external_hashes=MappingProxyType(expected_external),
        physical_specimens=fixed_counts["physical_specimens"],
        domains=fixed_counts["domains"],
        episode_count=fixed_counts["episode_count"],
        states_per_episode=fixed_counts["states_per_episode"],
        actions_per_episode=fixed_counts["actions_per_episode"],
        trajectory_rows=fixed_counts["trajectory_rows"],
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        bootstrap_replicates=int(statistics["bootstrap_replicates"]),
        training_updates=fixed_limits["training_updates"],
        vlm_calls=fixed_limits["vlm_calls"],
        actor_stop_forward_calls=fixed_limits["actor_stop_forward_calls"],
        base_recovery_step_cap=fixed_limits["base_recovery_step_cap"],
        reviewed_recovery_step_cap=fixed_limits["reviewed_recovery_step_cap"],
        values=MappingProxyType(payload),
    )


def _ordered_rows(
    rows: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, object], ...]:
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("episode trajectory rows are invalid")
    steps = [row.get("step") for row in rows]
    costs = [row.get("cost") for row in rows]
    if (
        any(type(step) is not int or step < 0 for step in steps)
        or len(set(steps)) != len(steps)
        or any(right <= left for left, right in pairwise(steps))
        or any(
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or not 0.0 <= float(cost) <= 1.0
            for cost in costs
        )
        or any(float(right) < float(left) - 1e-15 for left, right in pairwise(costs))
        or any(type(row.get("success")) is not bool for row in rows)
    ):
        raise ValueError("episode trajectory rows are not ordered")
    return rows


def _stop_index(
    rows: tuple[Mapping[str, object], ...], stop_system: str
) -> int | None:
    if stop_system not in STOP_SYSTEMS:
        raise ValueError("stop system is invalid")
    field = "calibrated_stop_trigger" if stop_system == "S_BC_CAL" else "rule_stop"
    for index, row in enumerate(rows):
        value = row.get(field)
        if type(value) is not bool:
            raise ValueError(f"trajectory {field} is invalid")
        if value:
            return index
    return None


def _snapshots(rows: tuple[Mapping[str, object], ...]) -> tuple[StepSnapshot, ...]:
    snapshots = []
    for row in rows:
        loss = row.get("task_loss", float(not bool(row["success"])))
        digest = row.get("report_sha256", f"step-{row['step']}")
        snapshots.append(
            StepSnapshot(
                cost=float(row["cost"]),
                success=bool(row["success"]),
                task_loss=float(loss),
                report_digest=str(digest),
            )
        )
    return tuple(snapshots)


def _integral(
    snapshots: tuple[StepSnapshot, ...],
    *,
    field: str,
    start: float,
    end: float,
) -> float:
    if math.isclose(start, end, abs_tol=1e-15):
        return 0.0
    return exact_step_integral(
        snapshots,
        field=field,  # type: ignore[arg-type]
        start_cost=start,
        end_cost=end,
    )


def stop_decomposition(
    rows: tuple[Mapping[str, object], ...], stop_system: str
) -> dict[str, object]:
    """Describe the first frozen STOP without executing its row's action."""

    checked = _ordered_rows(rows)
    index = _stop_index(checked, stop_system)
    stopped = index is not None
    terminal_index = len(checked) - 1 if index is None else index
    terminal = checked[terminal_index]
    success_at_stop = bool(terminal["success"]) if stopped else None
    completion = bool(stopped and success_at_stop)
    false_stop = bool(stopped and not success_at_stop)
    executable = checked[:terminal_index] if stopped else checked[:-1]
    prefix_measurement_cost = sum(
        float(checked[position + 1]["cost"]) - float(checked[position]["cost"])
        for position in range(len(executable))
    )
    stop_cost = float(terminal["cost"]) if stopped else None
    return {
        "stop_system": stop_system,
        "stopped": stopped,
        "stop_step": int(terminal["step"]) if stopped else None,
        "stop_cost": stop_cost,
        "stop_report_sha256": (
            str(terminal.get("report_sha256", "")) if stopped else None
        ),
        "success_at_stop": success_at_stop,
        "completion": completion,
        "false_stop": false_stop,
        "exhausted": not stopped,
        "autonomous_ausc": 1.0 - stop_cost if completion and stop_cost is not None else 0.0,
        "failure_penalized_cost": stop_cost if completion else 1.0,
        "measured_cost_at_execution_end": float(terminal["cost"]),
        "prefix_action_count": len(executable),
        "prefix_measurement_cost": float(prefix_measurement_cost),
        "prefix_route_cost": float(terminal.get("cumulative_route_cost", 0.0)),
        "prefix_route_turns": int(terminal.get("cumulative_route_turns", 0)),
    }


def report_stability(
    rows: tuple[Mapping[str, object], ...], *, stop_step: int | None
) -> dict[str, object]:
    """Find first and sustained success without making success absorbing."""

    checked = _ordered_rows(rows)
    successes = [bool(row["success"]) for row in checked]
    first = next((index for index, value in enumerate(successes) if value), None)
    suffix_all = [False] * len(successes)
    running = True
    for index in range(len(successes) - 1, -1, -1):
        running = running and successes[index]
        suffix_all[index] = running
    sustained = next((index for index, value in enumerate(suffix_all) if value), None)
    success_to_failure = sum(
        left and not right for left, right in pairwise(successes)
    )
    failure_to_success = sum(
        not left and right for left, right in pairwise(successes)
    )
    index_by_step = {int(row["step"]): index for index, row in enumerate(checked)}
    stop_index = None if stop_step is None else index_by_step.get(stop_step)
    if stop_step is not None and stop_index is None:
        raise ValueError("stop step is absent from the trajectory")
    if stop_index is None:
        category = (
            "NO_STOP_DESPITE_SUSTAINED_SUCCESS"
            if sustained is not None
            else "NO_STOP_NO_SUSTAINED_SUCCESS"
        )
    elif successes[stop_index]:
        if any(not value for value in successes[stop_index + 1 :]):
            category = "SUCCESS_BEFORE_LATER_REGRESSION"
        elif (
            sustained is not None
            and float(checked[stop_index]["cost"])
            > float(checked[sustained]["cost"]) + 1e-15
        ):
            category = "SUCCESS_WITH_POSTSUSTAINED_DELAY"
        else:
            category = "SUCCESS_AT_STOP_NO_POSTSUSTAINED_DELAY"
    elif any(successes[stop_index + 1 :]):
        category = "FALSE_STOP_WITH_LATER_SUCCESS"
    else:
        category = "FALSE_STOP_NEVER_SUCCESS"
    delay = None
    if (
        stop_index is not None
        and sustained is not None
        and float(checked[stop_index]["cost"])
        >= float(checked[sustained]["cost"]) - 1e-15
    ):
        delay = float(checked[stop_index]["cost"]) - float(
            checked[sustained]["cost"]
        )
    uncompleted_wait = None
    if stop_index is None and sustained is not None:
        uncompleted_wait = float(checked[-1]["cost"]) - float(
            checked[sustained]["cost"]
        )
    return {
        "first_success_step": None if first is None else int(checked[first]["step"]),
        "first_success_cost": None if first is None else float(checked[first]["cost"]),
        "sustained_success_step": (
            None if sustained is None else int(checked[sustained]["step"])
        ),
        "sustained_success_cost": (
            None if sustained is None else float(checked[sustained]["cost"])
        ),
        "sustained_until_recorded_end_only": sustained is not None,
        "full_input_only": sustained == len(checked) - 1,
        "success_to_failure_count": success_to_failure,
        "failure_to_success_count": failure_to_success,
        "regressed_after_first_success": bool(
            first is not None and any(not value for value in successes[first + 1 :])
        ),
        "post_sustained_stop_delay": delay,
        "uncompleted_wait_after_sustained": uncompleted_wait,
        "outcome_category": category,
    }


def planning_stop_accounting(
    rows: tuple[Mapping[str, object], ...], *, stop_step: int | None
) -> dict[str, object]:
    """Compute the signed planner-to-autonomous accounting identity."""

    checked = _ordered_rows(rows)
    snapshots = _snapshots(checked)
    planner_ausc = _integral(
        snapshots, field="success", start=0.0, end=1.0
    )
    index_by_step = {int(row["step"]): index for index, row in enumerate(checked)}
    stop_index = None if stop_step is None else index_by_step.get(stop_step)
    if stop_step is not None and stop_index is None:
        raise ValueError("stop step is absent from the trajectory")
    completion = bool(stop_index is not None and checked[stop_index]["success"])
    stop_cost = None if stop_index is None else float(checked[stop_index]["cost"])
    autonomous = 1.0 - stop_cost if completion and stop_cost is not None else 0.0
    gap = planner_ausc - autonomous
    boundary = 1.0 if stop_cost is None else stop_cost
    pre_success = _integral(
        snapshots, field="success", start=0.0, end=boundary
    )
    post_success = _integral(
        snapshots, field="success", start=boundary, end=1.0
    )
    post_failure = _integral(
        snapshots, field="failure", start=boundary, end=1.0
    )
    return {
        "stopped": stop_index is not None,
        "completion": completion,
        "false_stop": bool(stop_index is not None and not completion),
        "exhausted": stop_index is None,
        "stop_step": stop_step,
        "stop_cost": stop_cost,
        "planner_ausc": float(planner_ausc),
        "autonomous_ausc": float(autonomous),
        "gap": float(gap),
        "pre_stop_success_area": float(pre_success),
        "post_stop_success_area": float(post_success),
        "post_stop_failure_area": float(post_failure),
        "identity_residual": float(
            gap - (pre_success - post_failure) if completion else 0.0
        ),
    }


def derive_action_events(
    rows: tuple[Mapping[str, object], ...],
    stop_steps: Mapping[str, int | None],
) -> tuple[dict[str, object], ...]:
    """Align each stored action with its successor state and cost delta."""

    checked = _ordered_rows(rows)
    if set(stop_steps) != set(STOP_SYSTEMS):
        raise ValueError("action event stop steps are incomplete")
    levels = [-1] * 64
    events = []
    for index, (row, successor) in enumerate(pairwise(checked)):
        cell = row.get("action_cell")
        from_level = row.get("action_from_level")
        to_level = row.get("action_to_level")
        if (
            type(cell) is not int
            or not 0 <= cell < 64
            or type(from_level) is not int
            or type(to_level) is not int
            or levels[cell] != from_level
            or to_level != from_level + 1
        ):
            raise ValueError("stored action sequence is invalid")
        added_cost = float(successor["cost"]) - float(row["cost"])
        added_route = float(successor.get("cumulative_route_cost", 0.0)) - float(
            row.get("cumulative_route_cost", 0.0)
        )
        added_turns = int(successor.get("cumulative_route_turns", 0)) - int(
            row.get("cumulative_route_turns", 0)
        )
        if added_cost < -1e-15 or added_route < -1e-15 or added_turns < 0:
            raise ValueError("stored action deltas are invalid")
        action_type = {
            (-1, 0): "SURVEY_NEW_CELL",
            (0, 1): "REFINE_MEDIUM",
            (1, 2): "REFINE_DENSE",
        }.get((from_level, to_level))
        if action_type is None:
            raise ValueError("stored action transition is invalid")
        events.append(
            {
                "step": int(row["step"]),
                "cost_before_action": float(row["cost"]),
                "action_cell": cell,
                "action_from_level": from_level,
                "action_to_level": to_level,
                "action_type": action_type,
                "added_measurement_fraction": float(added_cost),
                "added_route_cost": float(added_route),
                "added_route_turns": added_turns,
                "pre_action_success": bool(row["success"]),
                "post_action_success": bool(successor["success"]),
                "added_support_count": int(successor.get("support_count", 0))
                - int(row.get("support_count", 0)),
                "support_event": (
                    "NEW_SUPPORT"
                    if int(successor.get("support_count", 0))
                    > int(row.get("support_count", 0))
                    else "NO_IMMEDIATE_NEW_SUPPORT"
                ),
                "unmeasured_cell_count_before": sum(level < 0 for level in levels),
                "in_s_bc_cal_prefix": (
                    stop_steps["S_BC_CAL"] is None
                    or int(row["step"]) < int(stop_steps["S_BC_CAL"])
                ),
                "in_s_rule_prefix": (
                    stop_steps["S_RULE"] is None
                    or int(row["step"]) < int(stop_steps["S_RULE"])
                ),
            }
        )
        levels[cell] = to_level
    terminal = checked[-1]
    if (
        terminal.get("action_cell") != -1
        or terminal.get("action_from_level") != -2
        or terminal.get("action_to_level") != -2
    ):
        raise ValueError("stored terminal action is invalid")
    return tuple(events)


def _matrix_contract(
    trajectories: pl.DataFrame, config: FrozenProcessConfig
) -> dict[str, int]:
    required = {
        "dataset_id",
        "specimen_id",
        "specimen_key",
        "task",
        "method",
        "seed",
        "step",
        "cost",
        "success",
        "task_loss",
        "iou",
        "recall",
        "relative_area_error",
        "report_sha256",
        "support_count",
        "action_cell",
        "action_from_level",
        "action_to_level",
        "rule_stop",
        "calibrated_stop_trigger",
        "learned_stop_probability",
        "learned_stop_eligible",
        "cumulative_route_cost",
        "cumulative_route_turns",
        "action_inference_seconds",
        "reference_version",
        "stop_model",
        "calibrated_threshold",
    }
    if not required <= set(trajectories.columns):
        raise ValueError(
            f"trajectory columns are incomplete: {sorted(required - set(trajectories.columns))}"
        )
    keys = ["specimen_key", "task", "method", "seed"]
    groups = trajectories.group_by(keys).agg(
        pl.len().alias("row_count"),
        pl.col("step").n_unique().alias("unique_steps"),
        pl.col("step").min().alias("first_step"),
        pl.col("step").max().alias("last_step"),
    )
    counts = {
        "physical_specimens": trajectories["specimen_key"].n_unique(),
        "domains": trajectories["dataset_id"].n_unique(),
        "methods": trajectories["method"].n_unique(),
        "tasks": trajectories["task"].n_unique(),
        "episodes": groups.height,
        "trajectory_rows": trajectories.height,
        "states_per_episode": config.states_per_episode,
        "actions_per_episode": config.actions_per_episode,
    }
    expected = {
        "physical_specimens": config.physical_specimens,
        "domains": config.domains,
        "methods": len(config.methods),
        "tasks": len(config.tasks),
        "episodes": config.episode_count,
        "trajectory_rows": config.trajectory_rows,
    }
    if any(counts[name] != value for name, value in expected.items()):
        raise ValueError(f"frozen trajectory matrix changed: {counts}")
    if (
        set(trajectories["method"].unique()) != set(config.methods)
        or set(trajectories["task"].unique()) != set(config.tasks)
        or groups.filter(
            (pl.col("row_count") != config.states_per_episode)
            | (pl.col("unique_steps") != config.states_per_episode)
            | (pl.col("first_step") != 0)
            | (pl.col("last_step") != config.actions_per_episode)
        ).height
        or trajectories.filter(
            (pl.col("step") == config.actions_per_episode)
            & (
                (pl.col("action_cell") != -1)
                | (pl.col("action_from_level") != -2)
                | (pl.col("action_to_level") != -2)
            )
        ).height
        or trajectories.filter(pl.col("reference_version") != "PROXY_LEGACY").height
        or trajectories.filter(pl.col("stop_model") != "S_BC_CAL").height
        or trajectories.filter(pl.col("calibrated_threshold") != 0.99).height
    ):
        raise ValueError("frozen trajectory identity changed")
    return counts


def _identity(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": str(row["dataset_id"]),
        "specimen_id": str(row["specimen_id"]),
        "specimen_key": str(row["specimen_key"]),
        "task": str(row["task"]),
        "method": str(row["method"]),
        "seed": int(row["seed"]),
        "reference_version": str(row["reference_version"]),
    }


def _episode_key(row: Mapping[str, object]) -> tuple[str, str, str, int]:
    return (
        str(row["specimen_key"]),
        str(row["task"]),
        str(row["method"]),
        int(row["seed"]),
    )


def _nullable_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _report_id(identity: Mapping[str, object], stop: Mapping[str, object]) -> str | None:
    if not stop["stopped"]:
        return None
    payload = {
        "specimen_key": identity["specimen_key"],
        "task": identity["task"],
        "method": identity["method"],
        "seed": identity["seed"],
        "stop_system": stop["stop_system"],
        "stop_step": stop["stop_step"],
        "report_sha256": stop["stop_report_sha256"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_stop_anchor(
    stop: Mapping[str, object], episode: Mapping[str, object], stop_system: str
) -> None:
    prefix = "calibrated" if stop_system == "S_BC_CAL" else "rule"
    comparisons = {
        "stopped": bool(episode[f"{prefix}_stopped"]),
        "completion": bool(episode[f"{prefix}_stop_success"]),
        "false_stop": bool(episode[f"{prefix}_false_stop"]),
        "exhausted": bool(episode[f"{prefix}_resource_exhausted"]),
        "prefix_action_count": int(episode[f"{prefix}_action_count"]),
    }
    if any(stop[field] != expected for field, expected in comparisons.items()):
        raise ValueError(f"{stop_system} frozen episode anchor changed")
    expected_cost = _nullable_float(episode[f"{prefix}_stop_cost"])
    actual_cost = stop["stop_cost"]
    if (expected_cost is None) != (actual_cost is None) or (
        expected_cost is not None
        and actual_cost is not None
        and not math.isclose(expected_cost, float(actual_cost), abs_tol=1e-12)
    ):
        raise ValueError(f"{stop_system} frozen stop cost changed")
    for field in ("autonomous_ausc", "failure_penalized_cost"):
        if not math.isclose(
            float(stop[field]), float(episode[f"{prefix}_{field}"]), abs_tol=1e-12
        ):
            raise ValueError(f"{stop_system} frozen {field} changed")


def _wait_components(
    rows: tuple[Mapping[str, object], ...],
    *,
    stop_system: str,
    stop: Mapping[str, object],
    stability: Mapping[str, object],
    threshold: float,
) -> dict[str, object]:
    sustained_step = stability["sustained_success_step"]
    if sustained_step is None:
        return {
            "wait_definition": "NO_SUSTAINED_SUCCESS",
            "defined": False,
            "total_wait_cost": None,
            "ineligible_wait_cost": None,
            "eligible_below_threshold_wait_cost": None,
            "eligible_crossed_unstopped_cost": None,
            "missing_reason": "NO_SUSTAINED_SUCCESS",
        }
    index_by_step = {int(row["step"]): index for index, row in enumerate(rows)}
    sustained_index = index_by_step[int(sustained_step)]
    stop_index = (
        None
        if stop["stop_step"] is None
        else index_by_step[int(stop["stop_step"])]
    )
    if stop_index is not None and stop_index < sustained_index:
        return {
            "wait_definition": "STOP_PRECEDES_SUSTAINED_SUCCESS",
            "defined": False,
            "total_wait_cost": None,
            "ineligible_wait_cost": None,
            "eligible_below_threshold_wait_cost": None,
            "eligible_crossed_unstopped_cost": None,
            "missing_reason": "STOP_PRECEDES_SUSTAINED_SUCCESS",
        }
    end_index = len(rows) - 1 if stop_index is None else stop_index
    total = float(rows[end_index]["cost"]) - float(rows[sustained_index]["cost"])
    if stop_system == "S_RULE":
        return {
            "wait_definition": (
                "UNCOMPLETED_WAIT_AFTER_SUSTAINED"
                if stop_index is None
                else "POST_SUSTAINED_STOP_DELAY"
            ),
            "defined": True,
            "total_wait_cost": total,
            "ineligible_wait_cost": None,
            "eligible_below_threshold_wait_cost": None,
            "eligible_crossed_unstopped_cost": None,
            "missing_reason": "S_RULE_COMPONENTS_NOT_RECORDED",
        }
    ineligible = 0.0
    below = 0.0
    crossed = 0.0
    for index in range(sustained_index, end_index):
        delta = float(rows[index + 1]["cost"]) - float(rows[index]["cost"])
        if not bool(rows[index]["learned_stop_eligible"]):
            ineligible += delta
        elif float(rows[index]["learned_stop_probability"]) < threshold:
            below += delta
        else:
            crossed += delta
    if crossed > 1e-12:
        raise ValueError("eligible threshold crossing continued without frozen STOP")
    return {
        "wait_definition": (
            "UNCOMPLETED_WAIT_AFTER_SUSTAINED"
            if stop_index is None
            else "POST_SUSTAINED_STOP_DELAY"
        ),
        "defined": True,
        "total_wait_cost": total,
        "ineligible_wait_cost": float(ineligible),
        "eligible_below_threshold_wait_cost": float(below),
        "eligible_crossed_unstopped_cost": float(crossed),
        "missing_reason": "",
    }


def _allocation_rows(
    identity: Mapping[str, object],
    events: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    scopes = (
        ("FULL_PLANNER_DIAGNOSTIC", None),
        ("AUTONOMOUS_PREFIX", "in_s_bc_cal_prefix"),
        ("AUTONOMOUS_PREFIX", "in_s_rule_prefix"),
    )
    output = []
    for scope, flag in scopes:
        selected = events if flag is None else tuple(row for row in events if row[flag])
        stop_system = (
            "NONE_FULL_TRAJECTORY"
            if flag is None
            else ("S_BC_CAL" if flag == "in_s_bc_cal_prefix" else "S_RULE")
        )
        total_cost = sum(float(row["added_measurement_fraction"]) for row in selected)
        by_type = {}
        for action_type in ("SURVEY_NEW_CELL", "REFINE_MEDIUM", "REFINE_DENSE"):
            typed = [row for row in selected if row["action_type"] == action_type]
            typed_cost = sum(float(row["added_measurement_fraction"]) for row in typed)
            by_type[action_type] = (len(typed), typed_cost)
        dense = next(
            (row for row in selected if row["action_type"] == "REFINE_DENSE"), None
        )
        output.append(
            {
                **identity,
                "analysis_scope": scope,
                "stop_system": stop_system,
                "action_count": len(selected),
                "measurement_cost": float(total_cost),
                "route_cost": float(
                    sum(float(row["added_route_cost"]) for row in selected)
                ),
                "route_turns": sum(int(row["added_route_turns"]) for row in selected),
                "survey_action_count": by_type["SURVEY_NEW_CELL"][0],
                "medium_action_count": by_type["REFINE_MEDIUM"][0],
                "dense_action_count": by_type["REFINE_DENSE"][0],
                "survey_measurement_cost": by_type["SURVEY_NEW_CELL"][1],
                "medium_measurement_cost": by_type["REFINE_MEDIUM"][1],
                "dense_measurement_cost": by_type["REFINE_DENSE"][1],
                "survey_cost_fraction": (
                    None if total_cost <= 0 else by_type["SURVEY_NEW_CELL"][1] / total_cost
                ),
                "medium_cost_fraction": (
                    None if total_cost <= 0 else by_type["REFINE_MEDIUM"][1] / total_cost
                ),
                "dense_cost_fraction": (
                    None if total_cost <= 0 else by_type["REFINE_DENSE"][1] / total_cost
                ),
                "first_dense_step": None if dense is None else dense["step"],
                "first_dense_cost": None if dense is None else dense["cost_before_action"],
                "first_dense_pre_action_success": (
                    None if dense is None else dense["pre_action_success"]
                ),
                "first_dense_unmeasured_cell_count": (
                    None if dense is None else dense["unmeasured_cell_count_before"]
                ),
            }
        )
    return tuple(output)


def _prefix_divergence(
    episodes: Mapping[tuple[str, str, str, int], tuple[Mapping[str, object], ...]],
    stops: Mapping[tuple[str, str, str, int], Mapping[str, int | None]],
) -> tuple[dict[str, object], ...]:
    output = []
    specimen_tasks = sorted({(key[0], key[1]) for key in episodes})
    for specimen_key, task in specimen_tasks:
        p8_key = (specimen_key, task, "R_BALANCED_P8", 1)
        p8 = episodes[p8_key]
        p8_actions = [
            (row["action_cell"], row["action_from_level"], row["action_to_level"])
            for row in p8[:-1]
        ]
        for method, seed in (("BC_S1", 1), ("BC_S2", 2), ("BC_S3", 3)):
            bc_key = (specimen_key, task, method, seed)
            bc = episodes[bc_key]
            bc_actions = [
                (row["action_cell"], row["action_from_level"], row["action_to_level"])
                for row in bc[:-1]
            ]
            divergence = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(bc_actions, p8_actions, strict=True)
                    )
                    if left != right
                ),
                None,
            )
            identity = _identity(bc[0])
            bc_stop = stops[bc_key]["S_BC_CAL"]
            p8_stop = stops[p8_key]["S_BC_CAL"]
            before_either = divergence is not None and all(
                stop is None or divergence < stop for stop in (bc_stop, p8_stop)
            )
            output.append(
                {
                    **identity,
                    "analysis_scope": "COMMON_HISTORY_DIVERGENCE",
                    "comparator": "R_BALANCED_P8",
                    "diverged": divergence is not None,
                    "common_prefix_action_count": (
                        len(bc_actions) if divergence is None else divergence
                    ),
                    "first_divergence_step": divergence,
                    "bc_action": None if divergence is None else json.dumps(bc_actions[divergence]),
                    "p8_action": None if divergence is None else json.dumps(p8_actions[divergence]),
                    "bc_pre_divergence_cost": (
                        None if divergence is None else float(bc[divergence]["cost"])
                    ),
                    "p8_pre_divergence_cost": (
                        None if divergence is None else float(p8[divergence]["cost"])
                    ),
                    "bc_pre_divergence_success": (
                        None if divergence is None else bool(bc[divergence]["success"])
                    ),
                    "p8_pre_divergence_success": (
                        None if divergence is None else bool(p8[divergence]["success"])
                    ),
                    "divergence_before_either_s_bc_stop": before_either,
                }
            )
    return tuple(output)


def _paired_effects(
    accounting_rows: tuple[dict[str, object], ...], config: FrozenProcessConfig
) -> tuple[dict[str, object], ...]:
    specimen_domains = {
        str(row["specimen_key"]): str(row["dataset_id"])
        for row in accounting_rows
    }
    draws = make_domain_bootstrap_draws(
        specimen_domains,
        replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    metric_fields = (
        "planner_ausc",
        "autonomous_ausc",
        "gap",
        "gap_completion_contribution",
        "gap_false_stop_contribution",
        "gap_exhaustion_contribution",
    )
    output = []
    for task in config.tasks:
        for stop_system in STOP_SYSTEMS:
            selected = [
                row
                for row in accounting_rows
                if row["task"] == task
                and row["stop_system"] == stop_system
                and row["method"] in config.spatial_methods
            ]
            for field in metric_fields:
                records = tuple(
                    MetricRecord(
                        specimen_key=str(row["specimen_key"]),
                        domain=str(row["dataset_id"]),
                        method=(
                            "BC_3SEED"
                            if str(row["method"]).startswith("BC_S")
                            else str(row["method"])
                        ),
                        task=Task(task),
                        seed=int(row["seed"]),
                        value=float(row[field]),
                    )
                    for row in selected
                )
                effect = paired_domain_bootstrap(
                    records,
                    treatment="BC_3SEED",
                    comparator="R_BALANCED_P8",
                    confidence_level=0.95,
                    draws=draws,
                )
                output.append(
                    {
                        "schema_version": 1,
                        "task": task,
                        "stop_system": stop_system,
                        "analysis_scope": "EXPLORATORY_FROZEN_COHORT_DIAGNOSTIC",
                        "metric": field,
                        "treatment": "BC_3SEED",
                        "comparator": "R_BALANCED_P8",
                        "effect_direction": "BC_MINUS_P8",
                        "estimate": effect.estimate,
                        "ci_lower": effect.ci_lower,
                        "ci_upper": effect.ci_upper,
                        "confidence_level": effect.confidence_level,
                        "physical_specimen_count": effect.physical_specimen_count,
                        "domain_count": effect.domain_count,
                        "bootstrap_replicates": effect.replicates,
                        "bootstrap_draws_sha256": effect.draws_sha256,
                        "reference_version": "PROXY_LEGACY",
                    }
                )
    return tuple(output)


def _planner_anchors(
    accounting_rows: tuple[dict[str, object], ...], config: FrozenProcessConfig
) -> dict[str, dict[str, float]]:
    output = {}
    for task in config.tasks:
        rows = [
            row
            for row in accounting_rows
            if row["task"] == task and row["stop_system"] == "S_BC_CAL"
        ]
        bc = [
            float(row["planner_ausc"])
            for row in rows
            if row["method"] in {"BC_S1", "BC_S2", "BC_S3"}
        ]
        p8 = [
            float(row["planner_ausc"])
            for row in rows
            if row["method"] == "R_BALANCED_P8"
        ]
        output[task] = {
            "BC_3SEED_MEAN": float(np.mean(bc)),
            "R_BALANCED_P8": float(np.mean(p8)),
        }
    summary = json.loads((config.source_result_root / "summary.json").read_text())
    expected = summary["planner_means_proxy"]
    for task in config.tasks:
        for method in ("BC_3SEED_MEAN", "R_BALANCED_P8"):
            if not math.isclose(
                output[task][method], float(expected[task][method]), abs_tol=1e-12
            ):
                raise ValueError("planner AUsC anchor changed")
    return output


def analyze_trajectory_tables(
    config: FrozenProcessConfig, *, trajectories: pl.DataFrame
) -> dict[str, object]:
    """Compute W0-W3 from immutable tables without model or Reader calls."""

    if type(config) is not FrozenProcessConfig or type(trajectories) is not pl.DataFrame:
        raise TypeError("typed frozen config and trajectory table are required")
    matrix = _matrix_contract(trajectories, config)
    episode_table = pl.read_csv(
        config.source_result_root / "per_episode_metrics.csv",
        infer_schema_length=None,
    )
    episode_metrics = {
        _episode_key(row): row for row in episode_table.to_dicts()
    }
    if len(episode_metrics) != config.episode_count:
        raise ValueError("frozen episode summary matrix changed")
    episodes = {}
    first_stop_rows = []
    stability_rows = []
    accounting_rows = []
    wait_rows = []
    action_rows = []
    allocation_rows = []
    stop_steps = {}
    for frame in trajectories.partition_by(
        ["specimen_key", "task", "method", "seed"], maintain_order=True
    ):
        ordered = frame.sort("step").to_dicts()
        rows = tuple(ordered)
        key = _episode_key(rows[0])
        episodes[key] = rows
        episode = episode_metrics[key]
        snapshots = _snapshots(rows)
        computed_ausc = _integral(
            snapshots, field="success", start=0.0, end=1.0
        )
        if not math.isclose(computed_ausc, float(episode["ausc_any"]), abs_tol=1e-12):
            raise ValueError("frozen episode AUsC anchor changed")
        identity = _identity(rows[0])
        episode_stop_steps = {}
        for stop_system in STOP_SYSTEMS:
            stop = stop_decomposition(rows, stop_system)
            _validate_stop_anchor(stop, episode, stop_system)
            episode_stop_steps[stop_system] = stop["stop_step"]
            stop_row = (
                None
                if stop["stop_step"] is None
                else rows[int(stop["stop_step"])]
            )
            true_break = bool(episode["true_break_executed"])
            if true_break and not bool(episode["true_break_prefix_match"]):
                raise ValueError("true-break prefix identity changed")
            enriched_stop = {
                **identity,
                "analysis_scope": "AUTONOMOUS_PREFIX",
                **stop,
                "report_id": _report_id(identity, stop),
                "false_stop_iou": None if stop_row is None or stop["completion"] else float(stop_row["iou"]),
                "false_stop_recall": None if stop_row is None or stop["completion"] else float(stop_row["recall"]),
                "false_stop_relative_area_error": None if stop_row is None or stop["completion"] else float(stop_row["relative_area_error"]),
                "false_stop_task_loss": None if stop_row is None or stop["completion"] else float(stop_row["task_loss"]),
                "execution_evidence": (
                    "EXISTING_TRUE_BREAK"
                    if stop_system == "S_BC_CAL" and true_break
                    else "CACHED_PREFIX"
                ),
                "planner_inference_seconds": float(
                    episode[
                        "calibrated_planner_inference_seconds"
                        if stop_system == "S_BC_CAL"
                        else "rule_planner_inference_seconds"
                    ]
                ),
                "stop_inference_seconds": (
                    _nullable_float(episode["true_break_stop_inference_seconds"])
                    if stop_system == "S_BC_CAL" and true_break
                    else None
                ),
                "timing_scope": (
                    "EXISTING_TRUE_BREAK_MEASURED"
                    if stop_system == "S_BC_CAL" and true_break
                    else "PLANNER_ONLY_STOP_LATENCY_NOT_MEASURED"
                ),
            }
            first_stop_rows.append(enriched_stop)
            stability = report_stability(rows, stop_step=stop["stop_step"])
            stability_rows.append(
                {
                    **identity,
                    "analysis_scope": "POST_STOP_FROZEN_SUFFIX_DIAGNOSTIC",
                    "stop_system": stop_system,
                    **stability,
                }
            )
            accounting = planning_stop_accounting(
                rows, stop_step=stop["stop_step"]
            )
            accounting_rows.append(
                {
                    **identity,
                    "analysis_scope": "EXPLORATORY_FROZEN_COHORT_DIAGNOSTIC",
                    "stop_system": stop_system,
                    **accounting,
                    "gap_completion_contribution": (
                        accounting["gap"] if accounting["completion"] else 0.0
                    ),
                    "gap_false_stop_contribution": (
                        accounting["gap"] if accounting["false_stop"] else 0.0
                    ),
                    "gap_exhaustion_contribution": (
                        accounting["gap"] if accounting["exhausted"] else 0.0
                    ),
                }
            )
            wait = _wait_components(
                rows,
                stop_system=stop_system,
                stop=stop,
                stability=stability,
                threshold=config.stop_thresholds[str(rows[0]["task"])],
            )
            wait_rows.append(
                {
                    **identity,
                    "analysis_scope": "AUTONOMOUS_PREFIX",
                    "stop_system": stop_system,
                    **wait,
                }
            )
        stop_steps[key] = episode_stop_steps
        events = derive_action_events(rows, episode_stop_steps)
        for event in events:
            action_rows.append(
                {
                    **identity,
                    "analysis_scope": "FULL_PLANNER_DIAGNOSTIC",
                    **event,
                }
            )
        allocation_rows.extend(_allocation_rows(identity, events))
    first_stop_tuple = tuple(first_stop_rows)
    stability_tuple = tuple(stability_rows)
    accounting_tuple = tuple(accounting_rows)
    wait_tuple = tuple(wait_rows)
    action_tuple = tuple(action_rows)
    allocation_tuple = tuple(allocation_rows)
    divergence = _prefix_divergence(episodes, stop_steps)
    paired = _paired_effects(accounting_tuple, config)
    anchors = _planner_anchors(accounting_tuple, config)
    return {
        "input_manifest": {
            "schema_version": 1,
            "stage": "FROZEN_INPUTS_VALIDATED",
            "repository_base_sha": config.repository_base_sha,
            "config_sha256": config.config_sha256,
            "matrix": matrix,
            "planner_ausc_anchors": anchors,
            "source_hashes": dict(config.expected_input_hashes),
            "external_source_hashes": dict(config.expected_external_hashes),
            "frozen_call_counts": {
                "training_updates": 0,
                "vlm_calls": 0,
                "actor_forward_calls": 0,
                "stop_forward_calls": 0,
            },
        },
        "first_stop_decomposition": first_stop_tuple,
        "report_stability_and_delay": stability_tuple,
        "planning_to_stop_accounting": accounting_tuple,
        "stop_wait_components": wait_tuple,
        "action_events": action_tuple,
        "action_cost_allocation": allocation_tuple,
        "prefix_divergence": divergence,
        "paired_diagnostic_effects": paired,
    }


def _write_text(path: Path, text: str) -> None:
    payload = text if text.endswith("\n") else text + "\n"

    def writer(temporary: Path) -> None:
        temporary.write_text(payload, encoding="utf-8")

    _atomic_write(path, writer)


def _write_csv(path: Path, rows: tuple[dict[str, object], ...], fields: tuple[str, ...] | None = None) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"empty CSV {path} requires an explicit schema")
        fields = tuple(rows[0])
    write_csv_atomic(path, rows, fields)


def _analysis_summary(
    tables: Mapping[str, object], spatial: Mapping[str, object]
) -> dict[str, object]:
    stops = tables["first_stop_decomposition"]
    stability = tables["report_stability_and_delay"]
    allocation = tables["action_cost_allocation"]
    agreements = spatial["surface_proxy_agreement"]
    effects = tables["paired_diagnostic_effects"]
    stop_summary = {}
    for task in ("LOCATE", "CHARACTERIZE"):
        stop_summary[task] = {}
        for system in STOP_SYSTEMS:
            rows = [
                row
                for row in stops  # type: ignore[union-attr]
                if row["task"] == task
                and row["stop_system"] == system
                and row["method"] in {"R_BALANCED_P8", "BC_S1", "BC_S2", "BC_S3"}
            ]
            stop_summary[task][system] = {
                "episode_count": len(rows),
                "completion_rate": sum(bool(row["completion"]) for row in rows)
                / len(rows),
                "false_stop_episode_rate": sum(bool(row["false_stop"]) for row in rows)
                / len(rows),
                "exhaustion_rate": sum(bool(row["exhausted"]) for row in rows)
                / len(rows),
            }
    category_counts = Counter(
        (str(row["task"]), str(row["stop_system"]), str(row["outcome_category"]))
        for row in stability  # type: ignore[union-attr]
    )
    cost_means = {}
    for task in ("LOCATE", "CHARACTERIZE"):
        cost_means[task] = {}
        for method_group in ("BC_3SEED", "R_BALANCED_P8"):
            rows = [
                row
                for row in allocation  # type: ignore[union-attr]
                if row["task"] == task
                and row["analysis_scope"] == "AUTONOMOUS_PREFIX"
                and row["stop_system"] == "S_BC_CAL"
                and (
                    (method_group == "BC_3SEED" and str(row["method"]).startswith("BC_S"))
                    or row["method"] == method_group
                )
            ]
            cost_means[task][method_group] = {
                name: float(np.mean([float(row[name]) for row in rows]))
                for name in (
                    "survey_measurement_cost",
                    "medium_measurement_cost",
                    "dense_measurement_cost",
                )
            }
    return {
        "schema_version": 1,
        "stage": "W0_W5_FROZEN_ANALYSIS_COMPLETE",
        "scientific_status_unchanged": "BC_PLANNING_SUPPORTED_STOP_NOT_SUPPORTED_PROXY_ONLY",
        "matrix": tables["input_manifest"]["matrix"],  # type: ignore[index]
        "stop_summary_main_methods": stop_summary,
        "report_outcome_category_counts": [
            {
                "task": key[0],
                "stop_system": key[1],
                "category": key[2],
                "count": value,
            }
            for key, value in sorted(category_counts.items())
        ],
        "action_cost_means": cost_means,
        "surface_proxy_group_counts": dict(
            Counter(str(row["overlap_group"]) for row in agreements)  # type: ignore[union-attr]
        ),
        "paired_diagnostic_effects": [
            row
            for row in effects  # type: ignore[union-attr]
            if row["stop_system"] == "S_BC_CAL"
            and row["metric"] in {"planner_ausc", "autonomous_ausc", "gap"}
        ],
        "observed_fact": "Frozen trajectories permit exact signed planning-to-stop and action-cost accounting.",
        "supported_association": "Report stability, STOP waiting, and spatial proxy groups are descriptive frozen-path associations.",
        "not_established": "No causal mechanism, hardware-time benefit, independent correctness, or human superiority is established.",
        "resource_use": spatial["recovery_manifest"],
    }


def _case_manifest(
    config: FrozenProcessConfig, agreements: tuple[dict[str, object], ...]
) -> tuple[dict[str, object], ...]:
    cohort = pl.read_csv(
        config.source_result_root / "cohort_and_reference_coverage.csv",
        infer_schema_length=None,
    ).filter(pl.col("split") == "TEST")
    agreement_by_key = {
        (str(row["specimen_key"]), str(row["task"])): row for row in agreements
    }
    output = []
    for domain in sorted(cohort["dataset_id"].unique()):
        rows = cohort.filter(pl.col("dataset_id") == domain).to_dicts()
        selected = min(
            rows,
            key=lambda row: hashlib.sha256(
                f"bc-frozen-case-v1|{row['specimen_key']}".encode()
            ).hexdigest(),
        )
        domain_rank = int(selected["domain_rank"])
        seed = domain_rank % 3 + 1
        locate = agreement_by_key[(str(selected["specimen_key"]), "LOCATE")]
        characterize = agreement_by_key[
            (str(selected["specimen_key"]), "CHARACTERIZE")
        ]
        output.append(
            {
                "schema_version": 1,
                "dataset_id": domain,
                "specimen_id": selected["specimen_id"],
                "specimen_key": selected["specimen_key"],
                "domain_rank": domain_rank,
                "bc_seed": seed,
                "bc_method": f"BC_S{seed}",
                "selection_rule": "MIN_SHA256_BC_FROZEN_CASE_V1_WITHIN_DOMAIN",
                "locate_overlap_group": locate["overlap_group"],
                "characterize_overlap_group": characterize["overlap_group"],
                "reference_version": "PROXY_LEGACY",
            }
        )
    return tuple(output)


def _render_figures(
    output_root: Path,
    *,
    tables: Mapping[str, object],
    spatial: Mapping[str, object],
    cases: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    palette = {"planner_ausc": "#2C7BB6", "autonomous_ausc": "#1A9850", "gap": "#D73027"}
    effects = [
        row
        for row in tables["paired_diagnostic_effects"]  # type: ignore[index]
        if row["stop_system"] == "S_BC_CAL"
        and row["metric"] in palette
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    tasks = ("LOCATE", "CHARACTERIZE")
    x = np.arange(len(tasks))
    width = 0.24
    for offset, metric in zip((-1, 0, 1), palette, strict=True):
        rows = [next(row for row in effects if row["task"] == task and row["metric"] == metric) for task in tasks]
        values = [float(row["estimate"]) for row in rows]
        lower = [value - float(row["ci_lower"]) for value, row in zip(values, rows, strict=True)]
        upper = [float(row["ci_upper"]) - value for value, row in zip(values, rows, strict=True)]
        ax.bar(x + offset * width, values, width, color=palette[metric], label=metric.replace("_", " "))
        ax.errorbar(x + offset * width, values, yerr=np.asarray([lower, upper]), fmt="none", color="black", capsize=3, linewidth=0.8)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, tasks)
    ax.set_ylabel("BC minus P8 effect")
    ax.legend(frameon=False, ncols=3, fontsize=8)
    path1 = figure_root / "figure_1_planning_to_stop_accounting.png"
    fig.savefig(path1, dpi=220)
    plt.close(fig)

    allocation = [
        row
        for row in tables["action_cost_allocation"]  # type: ignore[index]
        if row["analysis_scope"] == "AUTONOMOUS_PREFIX"
        and row["stop_system"] == "S_BC_CAL"
        and row["method"] in {"R_BALANCED_P8", "BC_S1", "BC_S2", "BC_S3"}
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True, sharey=True)
    colors = ("#4575B4", "#FDAE61", "#1A9850")
    fields = ("survey_measurement_cost", "medium_measurement_cost", "dense_measurement_cost")
    for ax, task in zip(axes, tasks, strict=True):
        bottom = np.zeros(2)
        for field, color in zip(fields, colors, strict=True):
            values = []
            for group in ("P8", "BC"):
                selected = [
                    row
                    for row in allocation
                    if row["task"] == task
                    and ((group == "P8" and row["method"] == "R_BALANCED_P8") or (group == "BC" and str(row["method"]).startswith("BC_S")))
                ]
                values.append(float(np.mean([float(row[field]) for row in selected])))
            ax.bar((0, 1), values, bottom=bottom, color=color, label=field.split("_")[0])
            bottom += values
        ax.set_title(task)
        ax.set_xticks((0, 1), ("P8", "BC"))
        ax.set_ylabel("Native-raster measurement fraction")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=3,
        frameon=False,
        fontsize=8,
    )
    path2 = figure_root / "figure_2_action_cost_allocation.png"
    fig.savefig(path2, dpi=220)
    plt.close(fig)

    summaries = spatial["spatial_episode_summary"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    labels = []
    y = np.arange(len(cases))
    for index, case in enumerate(cases):
        labels.append(str(case["dataset_id"]))
        for method, marker, color in (("R_BALANCED_P8", "o", "#555555"), (case["bc_method"], "s", "#2C7BB6")):
            row = next(
                row
                for row in summaries  # type: ignore[union-attr]
                if row["specimen_key"] == case["specimen_key"]
                and row["task"] == "LOCATE"
                and row["method"] == method
                and row["analysis_scope"] == "AUTONOMOUS_PREFIX"
                and row["stop_system"] == "S_BC_CAL"
            )
            ax.scatter(
                float(row["added_v_measurement_fraction"]),
                index + (-0.12 if method == "R_BALANCED_P8" else 0.12),
                marker=marker,
                color=color,
                label=(method if index == 0 else None),
            )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Autonomous-prefix measurement fraction in surface-cue cells")
    ax.set_ylabel("Frozen domain case")
    ax.legend(frameon=False, fontsize=8)
    path3 = figure_root / "figure_3_surface_proxy_process_cases.png"
    fig.savefig(path3, dpi=220)
    plt.close(fig)
    checks = []
    for path in (path1, path2, path3):
        with Image.open(path) as image:
            values = np.asarray(image.convert("L"), dtype=np.float64)
            checks.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "width": image.width,
                    "height": image.height,
                    "grayscale_standard_deviation": float(values.std()),
                    "nonblank": bool(values.std() > 1.0),
                }
            )
    if any(not row["nonblank"] for row in checks):
        raise ValueError("rendered frozen-process figure is blank")
    return tuple(checks)


def _write_analysis_documents(
    config: FrozenProcessConfig,
    *,
    summary: Mapping[str, object],
    figure_checks: tuple[dict[str, object], ...],
) -> None:
    trajectory_schema = pl.read_parquet_schema(
        config.source_result_root / "trajectories.parquet"
    )
    schema_lines = [
        f"- `{column}`: `{dtype}`" for column, dtype in trajectory_schema.items()
    ]
    frozen_hash_lines = [
        f"- `{config.source_result_root.relative_to(config.project_root) / name}`: "
        f"`{sha256}`"
        for name, sha256 in sorted(config.expected_input_hashes.items())
    ]
    external_hash_lines = [
        f"- `{name}`: `{sha256}`"
        for name, sha256 in sorted(config.expected_external_hashes.items())
    ]
    bindings = f"""# Input and Code Bindings

## Repository and Matrix

- Base commit: `{config.repository_base_sha}`
- Config: `{config.path.relative_to(config.project_root)}` (`{config.config_sha256}`)
- Matrix: 24 physical specimens, 6 domains, 7 methods, 2 tasks, 336 episodes, 64,848 state rows.
- State/action binding: row `j` is state/report `j`; its action produces row `j+1`; a STOP at `j` excludes that action.
- Recovery scope: Reader-only replay of stored actions; report digest is checked, packet hash is not claimed for the fast path.

## Frozen Input SHA-256

{chr(10).join(frozen_hash_lines)}
{chr(10).join(external_hash_lines)}

## Trajectory Columns and Dtypes

{chr(10).join(schema_lines)}

## Function Bindings

- `verify_frozen_input`: `hashlib.sha256` and the predeclared config identities.
- `analyze_trajectory_tables`: `exact_step_integral`, `make_domain_bootstrap_draws`, and `paired_domain_bootstrap`.
- `recover_spatial_analysis`: `_load_percepts`, `FrozenVisibleReportReader`, `_report_digest`, `_proxy_score`, `action_added_positions_from_mask`, and `apply_action` over stored actions only.
- `recover_reviewed_report_scores`: the same stored-action Reader path plus `adapt_task_report_v2` and `evaluate_task_report`; no Actor, STOP, or VLM inference.
- `finalize_frozen_evidence`: reviewed-score aggregation, physical-specimen/domain bootstrap, external file identity checks, blind-review pairing, and human-session comparability gates.
- `execute_frozen_analysis` / `execute_frozen_finalize`: existing atomic CSV, Parquet, JSON, text, and checksum writers under the new output roots only.
"""
    _write_text(config.artifact_root / "INPUT_AND_CODE_BINDINGS.md", bindings)
    result_text = f"""# Results and Limitations

## Completion

W0-W5 frozen computation is complete. Historical scientific status remains `BC_PLANNING_SUPPORTED_STOP_NOT_SUPPORTED_PROXY_ONLY`.

## Evidence Layers

- Observed fact: {summary['observed_fact']}
- Supported association: {summary['supported_association']}
- Not established: {summary['not_established']}

## Limits

These diagnostics retain `PROXY_LEGACY`, use the original 24 TEST specimens and six domains, and do not change any success label, action, STOP point, model, threshold, or historical Path B decision. Process accounting is explanatory and does not substitute for independent reviewed correctness or human comparison.

## Figure QA

{json.dumps(figure_checks, indent=2)}
"""
    _write_text(config.artifact_root / "RESULTS_AND_LIMITATIONS.md", result_text)
    user_inputs = """# Inputs from User

The same `finalize` command accepts any subset of the following real inputs. Omitted tracks remain `PENDING_USER_INPUT`.

## Reviewed References

A directory of one JSON file per specimen using the existing `registered_cscan` reference schema: `specimen_key`, `source_image_sha256`, `reference_type`, `review_state=reviewed`, `reviewer_alias`, `frame=registered_cscan`, `regions`, and `uncertain_regions`.

## Blind Report Reviews

A UTF-8 CSV with `report_id`, `reviewer_id`, and `decision`. Decision is one of `DELIVERABLE`, `NEEDS_FURTHER_INSPECTION`, or `UNABLE_TO_JUDGE`. Optional columns are `problem_type`, `review_basis`, and `blinding_condition`.

## Human Sessions

A UTF-8 CSV or JSON list containing `session_id`, `specimen_key`, `task`, `operator_id`, `geometry_version`, `reference_version`, `completion`, `false_stop`, `measurement_cost`, `route_cost`, `cost_unit`, and `route_cost_unit`. For process-level interpretation also provide `information_permission`, `report_production`, `action_sequence`, `first_stop_or_handoff`, and `final_report_id` or `final_report_sha256`. Use `FROZEN_NATIVE_8X8_THREE_LEVEL`, `NATIVE_RASTER_FRACTION`, `FROZEN_GRID_ROUTE_COST`, `MATCHED_TO_FROZEN_MODEL`, and `COMMON_READER_UNEDITED` only when those contracts are actually satisfied. Missing process fields remain explicit and restrict the result to a full-system endpoint comparison.

## Command

```bash
PYTHONPATH=src python scripts/analyze_bc_cscan_frozen_process.py finalize \\
  --config paper_v3/configs/bc_cscan_frozen_process_analysis.yaml \\
  --source-root /home/ww/paper3/cmc_damage_inference \\
  --references <reference-directory> \\
  --blind-reviews <blind-review.csv> \\
  --human-sessions <human-sessions.csv-or-json>
```
"""
    _write_text(config.artifact_root / "INPUTS_FROM_USER.md", user_inputs)


def execute_frozen_analysis(
    *,
    config_path: str | Path,
    project_root: str | Path,
    source_root: str | Path,
) -> dict[str, object]:
    """Execute W0-W5 and write deterministic compact artifacts."""

    config = load_frozen_process_config(config_path, project_root=project_root)
    trajectories = pl.read_parquet(config.source_result_root / "trajectories.parquet")
    tables = analyze_trajectory_tables(config, trajectories=trajectories)
    output = config.output_root
    output.mkdir(parents=True, exist_ok=True)
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "input_manifest.json", tables["input_manifest"])
    _write_csv(output / "first_stop_decomposition.csv", tables["first_stop_decomposition"])
    _write_csv(output / "report_stability_and_delay.csv", tables["report_stability_and_delay"])
    _write_csv(output / "stop_wait_components.csv", tables["stop_wait_components"])
    _write_csv(output / "planning_to_stop_accounting.csv", tables["planning_to_stop_accounting"])
    write_parquet_atomic(output / "action_events.parquet", tables["action_events"])
    _write_csv(output / "action_cost_allocation.csv", tables["action_cost_allocation"])
    _write_csv(output / "prefix_divergence.csv", tables["prefix_divergence"])
    _write_csv(output / "paired_diagnostic_effects.csv", tables["paired_diagnostic_effects"])
    from .frozen_process_recovery import recover_spatial_analysis

    spatial = recover_spatial_analysis(
        config, source_root=source_root, trajectories=trajectories
    )
    _write_csv(output / "surface_proxy_agreement.csv", spatial["surface_proxy_agreement"])
    write_parquet_atomic(
        output / "spatial_action_statistics.parquet",
        spatial["spatial_action_statistics"],
    )
    _write_csv(output / "spatial_episode_summary.csv", spatial["spatial_episode_summary"])
    write_json_atomic(output / "recovery_manifest.json", spatial["recovery_manifest"])
    cases = _case_manifest(config, spatial["surface_proxy_agreement"])
    _write_csv(output / "case_manifest.csv", cases)
    figure_checks = _render_figures(output, tables=tables, spatial=spatial, cases=cases)
    summary = _analysis_summary(tables, spatial)
    summary["figure_checks"] = figure_checks
    write_json_atomic(output / "analysis_summary.json", summary)
    reproduce = """# Reproduce

```bash
PYTHONPATH=src python scripts/analyze_bc_cscan_frozen_process.py analyze \\
  --config paper_v3/configs/bc_cscan_frozen_process_analysis.yaml \\
  --source-root /home/ww/paper3/cmc_damage_inference

PYTHONPATH=src python scripts/analyze_bc_cscan_frozen_process.py finalize \\
  --config paper_v3/configs/bc_cscan_frozen_process_analysis.yaml \\
  --source-root /home/ww/paper3/cmc_damage_inference
```

Both commands are CPU-only. They perform no training, VLM call, Actor forward call, STOP forward call, threshold scan, or action selection.
"""
    _write_text(output / "reproduce.md", reproduce)
    _write_analysis_documents(config, summary=summary, figure_checks=figure_checks)
    write_checksums(output, output / "CHECKSUMS.sha256")
    return {
        "stage": "W0_W5_FROZEN_ANALYSIS_COMPLETE",
        "output_root": str(output),
        "episodes": 336,
        "spatial_recovery_episodes": spatial["recovery_manifest"]["episode_count"],
        "reader_recovery_steps": spatial["recovery_manifest"][
            "stored_action_transition_count"
        ],
        "training_updates": 0,
        "vlm_calls": 0,
        "actor_forward_calls": 0,
        "stop_forward_calls": 0,
    }


def _write_optional_csv(
    path: Path,
    rows: tuple[dict[str, object], ...],
    empty_fields: tuple[str, ...],
) -> None:
    _write_csv(path, rows, None if rows else empty_fields)


def _claim_matrix_markdown(rows: tuple[dict[str, object], ...]) -> str:
    lines = [
        "# Submission Evidence Matrix",
        "",
        "| Claim | Task | Reference | N | Status | Source | Permitted wording | Prohibited overstatement |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        values = (
            row["claim_id"],
            row["task"],
            row["reference_version"],
            row["physical_n"],
            row["status"],
            row["source_table"],
            row["permitted_wording"],
            row["prohibited_overstatement"],
        )
        escaped = [str(value).replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend(("", "## Complete Claim Records", ""))
    fields = (
        ("Task", "task"),
        ("Exact claim", "exact_claim"),
        ("Planned evidence", "planned_evidence"),
        ("Existing evidence", "existing_evidence"),
        ("New evidence", "new_evidence"),
        ("Reference version", "reference_version"),
        ("Physical n", "physical_n"),
        ("Seed count", "seed_count"),
        ("Human coverage", "human_coverage"),
        ("Estimate", "estimate"),
        ("Interval", "interval"),
        ("Source table", "source_table"),
        ("Status", "status"),
        ("Permitted wording", "permitted_wording"),
        ("Prohibited overstatement", "prohibited_overstatement"),
    )
    for row in rows:
        lines.append(f"### {row['claim_id']}")
        lines.append("")
        lines.extend(
            f"- {label}: {row[field]}" if row[field] != "" else f"- {label}:"
            for label, field in fields
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_finalize_documents(
    config: FrozenProcessConfig, finalized: Mapping[str, object]
) -> None:
    claims = finalized["claim_rows"]
    _write_text(
        config.artifact_root / "SUBMISSION_EVIDENCE_MATRIX.md",
        _claim_matrix_markdown(claims),  # type: ignore[arg-type]
    )
    decisions = finalized["reviewed_task_decisions"]
    manifest = finalized["final_evidence_manifest"]
    boundaries = f"""# Final Results and Claim Boundaries

## Computational Status

`{manifest['stage']}`. W0-W5 frozen analysis has been executed and W6-W8 import/statistical paths are implemented.

## Reviewed Path B

- LOCATE: `{decisions['LOCATE']['status']}`
- CHARACTERIZE: `{decisions['CHARACTERIZE']['status']}`

## External Evidence

- Reviewed references: `{manifest['external_evidence']['reviewed_references']}`
- Blind report reviews: `{manifest['external_evidence']['blind_reviews']}`
- Human sessions: `{manifest['external_evidence']['human_sessions']}`

## Boundaries

No scientific result, frozen action, STOP point, model, threshold, or historical proxy decision was changed. The process analysis supports descriptive associations only. It does not establish an independent-correctness result, causal failure mechanism, material generalization, hardware-time benefit, natural-language reasoning, or superiority over human operators.
"""
    _write_text(
        config.artifact_root / "FINAL_RESULTS_AND_CLAIM_BOUNDARIES.md",
        boundaries,
    )
    recovery = finalized["reviewed_recovery"]
    analysis_path = config.output_root / "analysis_summary.json"
    main_recovery_path = config.output_root / "recovery_manifest.json"
    analysis = json.loads(analysis_path.read_text())
    main_recovery = json.loads(main_recovery_path.read_text())
    input_identities = finalized["external_input_identities"]
    coverage = finalized["reference_coverage"]
    matched_references = sum(
        row["status"] == "REVIEWED_REFERENCE_MATCHED" for row in coverage
    )
    missing_references = sum(
        row["status"] == "MISSING_REVIEWED_REFERENCE" for row in coverage
    )
    unevaluable_references = sum(
        row["status"] == "REVIEWED_NO_CERTAIN_REGION" for row in coverage
    )
    pending_references = sum(
        row["status"] == "PENDING_USER_INPUT" for row in coverage
    )
    unmatched_references = sum(
        row["status"]
        in {"UNMATCHED_SPECIMEN_KEY", "UNMATCHED_FROZEN_EPISODES"}
        for row in coverage
    )

    diagnostic_lines = []
    for task in config.tasks:
        effects = {
            str(row["metric"]): row
            for row in analysis["paired_diagnostic_effects"]
            if row["task"] == task and row["stop_system"] == "S_BC_CAL"
        }
        stop = analysis["stop_summary_main_methods"][task]["S_BC_CAL"]
        diagnostic_lines.extend(
            (
                (
                    f"- {task} planner AUsC, BC minus P8: "
                    f"{effects['planner_ausc']['estimate']:+.6f} "
                    f"(95% CI [{effects['planner_ausc']['ci_lower']:+.6f}, "
                    f"{effects['planner_ausc']['ci_upper']:+.6f}])."
                ),
                (
                    f"- {task} autonomous AUsC, BC minus P8: "
                    f"{effects['autonomous_ausc']['estimate']:+.6f} "
                    f"(95% CI "
                    f"[{effects['autonomous_ausc']['ci_lower']:+.6f}, "
                    f"{effects['autonomous_ausc']['ci_upper']:+.6f}])."
                ),
                (
                    f"- {task} S_BC_CAL completion "
                    f"{stop['completion_rate']:.6f}; false-stop "
                    f"{stop['false_stop_episode_rate']:.6f}; exhaustion "
                    f"{stop['exhaustion_rate']:.6f} across "
                    f"{stop['episode_count']} main-method episodes."
                ),
            )
        )
    stability_rows = pl.read_csv(
        config.output_root / "report_stability_and_delay.csv",
        infer_schema_length=None,
    ).to_dicts()
    delay_lines = []
    main_methods = {"R_BALANCED_P8", "BC_S1", "BC_S2", "BC_S3"}
    for task in config.tasks:
        selected = [
            row
            for row in stability_rows
            if row["task"] == task
            and row["stop_system"] == "S_BC_CAL"
            and row["method"] in main_methods
        ]
        delay_values = [
            float(row["post_sustained_stop_delay"])
            for row in selected
            if row["post_sustained_stop_delay"] not in (None, "")
        ]
        delay_lines.append(
            f"- {task}: post-sustained STOP delay is defined in "
            f"{len(delay_values)}/{len(selected)} main-method episodes, with conditional "
            f"mean native-raster cost {np.mean(delay_values):.6f}; "
            f"{sum(bool(row['regressed_after_first_success']) for row in selected)} "
            "episodes regress after first success."
        )
    cost_lines = []
    for task in config.tasks:
        for group in ("R_BALANCED_P8", "BC_3SEED"):
            values = analysis["action_cost_means"][task][group]
            total = sum(float(value) for value in values.values())
            cost_lines.append(
                f"- {task} {group}: survey {values['survey_measurement_cost']:.6f}, "
                f"medium {values['medium_measurement_cost']:.6f}, dense "
                f"{values['dense_measurement_cost']:.6f}, total {total:.6f}."
            )
    identity_lines = []
    for label in ("reviewed_references", "blind_reviews", "human_sessions"):
        identity = input_identities[label]
        identity_lines.append(
            f"- {label}: status `{identity['status']}`; path "
            f"`{identity['path'] or 'NOT_PROVIDED'}`; SHA-256 "
            f"`{identity['sha256'] or 'NOT_AVAILABLE'}`."
        )
    claim_lines = [
        f"- {row['claim_id']} ({row['task']}): `{row['status']}`; "
        f"reference `{row['reference_version']}`; N={row['physical_n']}; "
        f"source `{row['source_table']}`."
        for row in claims
    ]
    result_inventory = (
        "input_manifest.json",
        "first_stop_decomposition.csv",
        "report_stability_and_delay.csv",
        "stop_wait_components.csv",
        "planning_to_stop_accounting.csv",
        "action_events.parquet",
        "action_cost_allocation.csv",
        "prefix_divergence.csv",
        "surface_proxy_agreement.csv",
        "spatial_action_statistics.parquet",
        "spatial_episode_summary.csv",
        "recovery_manifest.json",
        "paired_diagnostic_effects.csv",
        "case_manifest.csv",
        "analysis_summary.json",
        "reviewed/",
        "human_review/",
        "human_planning/",
        "final_evidence_manifest.json",
        "figures/",
        "reproduce.md",
        "CHECKSUMS.sha256",
    )
    artifact_inventory = (
        "INPUT_AND_CODE_BINDINGS.md",
        "RESULTS_AND_LIMITATIONS.md",
        "INPUTS_FROM_USER.md",
        "SUBMISSION_EVIDENCE_MATRIX.md",
        "FINAL_RESULTS_AND_CLAIM_BOUNDARIES.md",
        "CODEX_HANDOFF_FROZEN_PROCESS_ANALYSIS.md",
    )
    handoff = f"""# Codex Handoff: Frozen Process Analysis

## Repository

- Base: `{config.repository_base_sha}`
- Branch: `research/bc-cscan-frozen-process-analysis`
- Final commit: resolve with `git rev-parse HEAD`; the document does not self-reference its commit.
- Frozen source root: `{main_recovery['source_root']}`
- Config: `{config.path.relative_to(config.project_root)}`; SHA-256 `{config.config_sha256}`
- Frozen source identities: `results/bc_cscan_frozen_process_analysis/input_manifest.json`

## External Input Identities

{chr(10).join(identity_lines)}

Reference coverage: {matched_references} matched TEST references, {unevaluable_references} reviewed references without a certain region, {missing_references} missing TEST references, {pending_references} pending TEST references, and {unmatched_references} unmatched supplied references. Missing external inputs are not synthesized.

## W0-W8

- W0-W5: executed from immutable source tables and stored-action Reader recovery.
- W6: implemented; reference status `{finalized['reference_input_status']}`.
- W7A: `{finalized['human_review']['status']}`.
- W7B: `{finalized['human_planning']['status']}`.
- W8: claim matrix generated with per-claim status and wording boundaries.

## Main Frozen Recovery

- Physical cohort: {analysis['matrix']['physical_specimens']} specimens across {analysis['matrix']['domains']} domains.
- Scalar analysis: {analysis['matrix']['episodes']} episodes and {analysis['matrix']['trajectory_rows']:,} recorded states.
- Main stored-action recovery: {main_recovery['episode_count']} episodes, {main_recovery['report_state_count']:,} report states, and {main_recovery['stored_action_transition_count']:,} action transitions.
- Main recovery cap: {main_recovery['stored_action_transition_cap']:,}; cache reuse: `{main_recovery['cache_reuse']}`.
- Report digest mismatches: {main_recovery['report_digest_mismatches']}; proxy score mismatches: {main_recovery['proxy_score_mismatches']}.

## Reviewed Recovery

- Status: `{recovery.get('status', finalized['reference_input_status'])}`.
- Episodes: {recovery.get('episode_count', 0)}; report states: {recovery.get('report_state_count', 0):,}; source action transitions represented: {recovery.get('source_action_transition_count', 0):,}.
- Transitions recovered in this run: {recovery.get('stored_action_transition_count', 0):,}; transitions reused from identity-bound cache: {recovery.get('cached_action_transition_count', 0):,}.
- Cache-hit specimens: {recovery.get('cache_hit_specimen_count', 0)}; cache-hit episodes: {recovery.get('cache_hit_episode_count', 0)}; recovery cap: {config.reviewed_recovery_step_cap:,}.

## Planning-to-Stop Findings

{chr(10).join(diagnostic_lines)}

These are exploratory frozen-cohort diagnostics under `PROXY_LEGACY`. Historical 97.5% Path B results remain the authoritative proxy decision; reviewed Path B is reported separately below.

## Report Stability and STOP Delay

{chr(10).join(delay_lines)}

Delay means are conditional on a sustained-success point existing before the frozen stop. They are descriptive and do not replace completion or failure-penalized cost.

## Action-Cost Allocation

{chr(10).join(cost_lines)}

Values are autonomous-prefix native-raster measurement fractions under `S_BC_CAL`; they are not wall-clock or hardware travel costs.

## Process Associations and Boundaries

- Observed fact: {analysis['observed_fact']}
- Supported association: {analysis['supported_association']}
- Boundary: {analysis['not_established']}
- Surface/proxy groups: `{json.dumps(analysis['surface_proxy_group_counts'], sort_keys=True)}`.

## Reviewed Path B and Claim Status

- Reviewed LOCATE Path B: `{decisions['LOCATE']['status']}`.
- Reviewed CHARACTERIZE Path B: `{decisions['CHARACTERIZE']['status']}`.
{chr(10).join(claim_lines)}

## Human Evidence and Comparability

- Blind-review status: `{finalized['human_review']['status']}`; raw decisions: {len(finalized['human_review']['report_decisions'])}; paired specimen/reviewer contrasts: {len(finalized['human_review']['paired_specimen_differences'])}.
- Human-session status: `{finalized['human_planning']['status']}`; supplied sessions: {len(finalized['human_planning']['per_session_results'])}; matched model comparisons: {len(finalized['human_planning']['matched_method_comparison'])}.
- Measurement cost is comparable only for `NATIVE_RASTER_FRACTION`; route cost is comparable only for `FROZEN_GRID_ROUTE_COST`.

## Resource Integrity

- Training updates: 0
- New VLM calls: 0
- Actor forward calls: 0
- STOP forward calls: 0
- Reviewed recovery world steps: `{recovery.get('world_step_count', 0)}`
- Main report cache hits: `{recovery.get('cache_hit_episode_count', 0)}`

## Scientific Status

Historical proxy status remains `BC_PLANNING_SUPPORTED_STOP_NOT_SUPPORTED_PROXY_ONLY`. Computation completion and scientific support are reported separately in `final_evidence_manifest.json` and `SUBMISSION_EVIDENCE_MATRIX.md`.

## Output Inventory

Result root: `results/bc_cscan_frozen_process_analysis/`

{chr(10).join(f'- `{path}`' for path in result_inventory)}

Artifact root: `artifacts/bc_cscan_frozen_process_analysis/`

{chr(10).join(f'- `{path}`' for path in artifact_inventory)}

## Later Finalize Command

```bash
PYTHONPATH=src python scripts/analyze_bc_cscan_frozen_process.py finalize \\
  --config paper_v3/configs/bc_cscan_frozen_process_analysis.yaml \\
  --source-root /home/ww/paper3/cmc_damage_inference \\
  --references <reference-directory> \\
  --blind-reviews <blind-review.csv> \\
  --human-sessions <human-sessions.csv-or-json>
```

The same command imports any subset of the three external tracks; omitted tracks remain `PENDING_INPUT`. It performs no training and no Actor, STOP, or VLM inference.
"""
    _write_text(
        config.artifact_root / "CODEX_HANDOFF_FROZEN_PROCESS_ANALYSIS.md",
        handoff,
    )


def execute_frozen_finalize(
    *,
    config_path: str | Path,
    project_root: str | Path,
    source_root: str | Path,
    references_path: str | Path | None = None,
    blind_reviews_path: str | Path | None = None,
    human_sessions_path: str | Path | None = None,
) -> dict[str, object]:
    """Execute the W6-W8 external-evidence assembly path."""

    config = load_frozen_process_config(config_path, project_root=project_root)
    if not (config.output_root / "analysis_summary.json").is_file():
        raise ValueError("run frozen analyze before finalize")
    trajectories = pl.read_parquet(config.source_result_root / "trajectories.parquet")
    tables = analyze_trajectory_tables(config, trajectories=trajectories)
    from .frozen_evidence_finalize import (
        _reviewed_cache_identity,
        finalize_frozen_evidence,
    )

    finalized = finalize_frozen_evidence(
        config,
        source_root=source_root,
        trajectories=trajectories,
        proxy_stop_rows=tables["first_stop_decomposition"],
        references_path=references_path,
        blind_reviews_path=blind_reviews_path,
        human_sessions_path=human_sessions_path,
    )
    output = config.output_root
    reviewed = output / "reviewed"
    human_review = output / "human_review"
    human_planning = output / "human_planning"
    for root in (reviewed, human_review, human_planning, config.artifact_root):
        root.mkdir(parents=True, exist_ok=True)
    _write_csv(reviewed / "reference_coverage.csv", finalized["reference_coverage"])
    report_scores = finalized["reviewed_report_scores"]
    if report_scores:
        write_parquet_atomic(reviewed / "report_scores.parquet", report_scores)
    else:
        write_parquet_atomic(
            reviewed / "report_scores.parquet",
            (),
            {
                "schema_version": pl.Int64,
                "dataset_id": pl.String,
                "specimen_id": pl.String,
                "specimen_key": pl.String,
                "task": pl.String,
                "method": pl.String,
                "seed": pl.Int64,
                "step": pl.Int64,
                "cost": pl.Float64,
                "report_sha256": pl.String,
                "rule_stop": pl.Boolean,
                "calibrated_stop_trigger": pl.Boolean,
                "analysis_scope": pl.String,
                "reference_version": pl.String,
                "formal_success": pl.Boolean,
                "iou": pl.Float64,
                "recall": pl.Float64,
                "relative_area_error": pl.Float64,
                "failure_types": pl.String,
            },
        )
    _write_optional_csv(
        reviewed / "full_input_readout.csv",
        finalized["reviewed_full_input_readout"],
        (
            "schema_version", "dataset_id", "specimen_id", "specimen_key", "task",
            "analysis_scope", "reference_version", "report_sha256", "formal_success",
            "iou", "recall", "relative_area_error", "failure_types",
        ),
    )
    _write_optional_csv(
        reviewed / "surface_reference_agreement.csv",
        finalized["reviewed_surface_reference_agreement"],
        (
            "schema_version", "dataset_id", "specimen_id", "specimen_key", "task",
            "analysis_scope", "reference_version", "no_reliable_cue", "surface_cue_cells",
            "surface_ring_cells", "target_pixel_count", "surface_pixel_count",
            "intersection_pixel_count", "union_pixel_count", "target_coverage_by_surface",
            "surface_precision_proxy", "surface_target_iou", "overlap_group",
        ),
    )
    reviewed_spatial_actions = finalized["reviewed_spatial_action_statistics"]
    if reviewed_spatial_actions:
        write_parquet_atomic(
            reviewed / "spatial_action_statistics.parquet",
            reviewed_spatial_actions,
        )
    else:
        write_parquet_atomic(
            reviewed / "spatial_action_statistics.parquet",
            (),
            {
                "schema_version": pl.Int64,
                "dataset_id": pl.String,
                "specimen_id": pl.String,
                "specimen_key": pl.String,
                "task": pl.String,
                "method": pl.String,
                "seed": pl.Int64,
                "reference_version": pl.String,
                "analysis_scope": pl.String,
                "step": pl.Int64,
                "cost_before_action": pl.Float64,
                "action_cell": pl.Int64,
                "action_from_level": pl.Int64,
                "action_to_level": pl.Int64,
                "action_cell_region": pl.String,
                "added_pixel_count": pl.Int64,
                "added_target_pixel_count": pl.Int64,
                "added_v_pixel_count": pl.Int64,
                "added_r_pixel_count": pl.Int64,
                "added_o_pixel_count": pl.Int64,
                "added_measurement_fraction": pl.Float64,
                "added_target_measurement_fraction": pl.Float64,
                "added_v_measurement_fraction": pl.Float64,
                "added_r_measurement_fraction": pl.Float64,
                "added_o_measurement_fraction": pl.Float64,
                "pre_action_success": pl.Boolean,
                "post_action_success": pl.Boolean,
                "support_pixel_count": pl.Int64,
                "support_target_pixel_count": pl.Int64,
                "support_v_pixel_count": pl.Int64,
                "support_r_pixel_count": pl.Int64,
                "support_o_pixel_count": pl.Int64,
                "in_s_bc_cal_prefix": pl.Boolean,
                "in_s_rule_prefix": pl.Boolean,
            },
        )
    _write_optional_csv(
        reviewed / "spatial_episode_summary.csv",
        finalized["reviewed_spatial_episode_summary"],
        (
            "schema_version", "dataset_id", "specimen_id", "specimen_key", "task",
            "method", "seed", "reference_version", "analysis_scope", "stop_system",
            "execution_end_step", "action_count", "first_target_measurement_step",
            "first_target_measurement_cost", "first_target_support_step",
            "first_target_support_cost", "first_locate_support_criterion_step",
            "first_locate_support_criterion_cost", "added_v_measurement_fraction",
            "added_r_measurement_fraction", "added_o_measurement_fraction", "entered_v",
            "first_v_action_step", "post_v_outside_action_status",
            "first_post_v_outside_action_step",
        ),
    )
    _write_optional_csv(
        reviewed / "per_episode_metrics.csv",
        finalized["reviewed_per_episode_metrics"],
        (
            "schema_version", "dataset_id", "specimen_id", "specimen_key", "task",
            "method", "seed", "stop_system", "analysis_scope", "reference_version",
            "report_id", "stopped", "stop_step", "stop_cost", "stop_report_sha256",
            "success_at_stop", "completion", "false_stop", "exhausted", "autonomous_ausc",
            "failure_penalized_cost", "measured_cost_at_execution_end", "prefix_action_count",
            "prefix_measurement_cost", "prefix_route_cost", "prefix_route_turns", "planner_ausc",
            "gap", "pre_stop_success_area", "post_stop_success_area", "post_stop_failure_area",
            "identity_residual",
        ),
    )
    _write_optional_csv(
        reviewed / "report_stability_and_delay.csv",
        finalized["reviewed_report_stability_and_delay"],
        (
            "schema_version", "dataset_id", "specimen_id", "specimen_key", "task",
            "method", "seed", "reference_version", "analysis_scope", "stop_system",
            "first_success_step", "first_success_cost", "sustained_success_step",
            "sustained_success_cost", "sustained_until_recorded_end_only",
            "full_input_only", "success_to_failure_count", "failure_to_success_count",
            "regressed_after_first_success", "post_sustained_stop_delay",
            "uncompleted_wait_after_sustained", "outcome_category",
        ),
    )
    _write_optional_csv(
        reviewed / "stop_wait_components.csv",
        finalized["reviewed_stop_wait_components"],
        (
            "schema_version", "dataset_id", "specimen_id", "specimen_key", "task",
            "method", "seed", "reference_version", "analysis_scope", "stop_system",
            "wait_definition", "defined", "total_wait_cost", "ineligible_wait_cost",
            "eligible_below_threshold_wait_cost", "eligible_crossed_unstopped_cost",
            "missing_reason",
        ),
    )
    effect_fields = (
        "schema_version", "task", "analysis_scope", "reference_version", "metric",
        "treatment", "comparator", "estimate", "ci_lower", "ci_upper", "confidence_level",
        "physical_specimen_count", "domain_count", "bootstrap_replicates", "bootstrap_draws_sha256",
    )
    _write_optional_csv(reviewed / "planner_effects.csv", finalized["reviewed_planner_effects"], effect_fields)
    _write_optional_csv(
        reviewed / "autonomous_effects.csv",
        finalized["reviewed_autonomous_effects"],
        (
            "schema_version", "task", "stop_system", "analysis_scope", "reference_version",
            "metric", "effect_direction", "treatment", "comparator", "estimate", "ci_lower",
            "ci_upper", "confidence_level", "physical_specimen_count", "domain_count",
            "bootstrap_replicates", "bootstrap_draws_sha256",
        ),
    )
    _write_optional_csv(
        reviewed / "risk_and_cost.csv",
        finalized["reviewed_risk_and_cost"],
        (
            "schema_version", "task", "method", "seed", "stop_system", "analysis_scope",
            "reference_version", "physical_specimen_count", "completion_rate",
            "false_stop_episode_rate", "exhaustion_rate", "autonomous_ausc",
            "failure_penalized_cost",
        ),
    )
    _write_optional_csv(
        reviewed / "actor_input_ablations.csv",
        finalized["reviewed_actor_input_ablations"],
        (
            "schema_version", "task", "analysis_scope", "reference_version", "isolated_input",
            "full_actor", "ablated_actor", "metric", "estimate", "ci_lower", "ci_upper",
            "confidence_level", "physical_specimen_count", "domain_count", "bootstrap_replicates",
            "bootstrap_draws_sha256",
        ),
    )
    write_json_atomic(reviewed / "task_decisions.json", finalized["reviewed_task_decisions"])
    reference_files = {
        str(row["specimen_key"]): str(row["reference_sha256"])
        for row in finalized["reference_coverage"]
        if row.get("status") == "REVIEWED_REFERENCE_MATCHED"
    }
    write_json_atomic(
        reviewed / "recovery_manifest.json",
        {
            **_reviewed_cache_identity(config),
            "reference_version": finalized["reference_version"],
            "reference_files": reference_files,
            "recovery": finalized["reviewed_recovery"],
        },
    )
    blind = finalized["human_review"]
    _write_csv(human_review / "input_coverage.csv", blind["input_coverage"])
    _write_optional_csv(
        human_review / "report_decisions.csv",
        blind["report_decisions"],
        ("report_id", "specimen_key", "task", "method", "seed", "stop_system", "report_sha256", "objective_success", "reference_version", "analysis_scope", "reviewer_id", "decision", "problem_type", "review_basis", "blinding_condition", "objective_human_disagreement"),
    )
    _write_optional_csv(
        human_review / "method_summary.csv",
        blind["method_summary"],
        ("task", "method", "stop_system", "reference_version", "analysis_scope", "review_count", "report_count", "physical_specimen_count", "deliverable_count", "needs_further_inspection_count", "unable_to_judge_count", "consensus"),
    )
    _write_optional_csv(
        human_review / "paired_specimen_differences.csv",
        blind["paired_specimen_differences"],
        (
            "specimen_key", "task", "stop_system", "reviewer_id",
            "reference_version", "bc_report_count", "p8_report_count",
            "bc_deliverable_fraction", "p8_deliverable_fraction",
            "bc_minus_p8_deliverable_fraction", "analysis_scope",
        ),
    )
    _write_optional_csv(
        human_review / "objective_review_disagreements.csv",
        blind["objective_review_disagreements"],
        ("report_id", "specimen_key", "task", "method", "seed", "stop_system", "report_sha256", "objective_success", "reference_version", "analysis_scope", "reviewer_id", "decision", "problem_type", "review_basis", "blinding_condition", "objective_human_disagreement"),
    )
    human = finalized["human_planning"]
    _write_csv(human_planning / "comparability_manifest.csv", human["comparability_manifest"])
    _write_optional_csv(
        human_planning / "per_session_results.csv",
        human["per_session_results"],
        (
            "session_id", "specimen_key", "task", "operator_id", "geometry_version",
            "reference_version", "completion", "false_stop", "measurement_cost",
            "route_cost", "cost_unit", "route_cost_unit", "information_permission",
            "report_production", "action_sequence", "first_stop_or_handoff",
            "final_report_id", "final_report_sha256", "analysis_scope", "status",
            "matched_frozen_method_count", "action_sequence_recorded",
            "first_stop_or_handoff_recorded", "final_report_identity",
            "process_trace_comparable", "pure_planner_comparable",
        ),
    )
    _write_optional_csv(
        human_planning / "matched_method_comparison.csv",
        human["matched_method_comparison"],
        ("session_id", "specimen_key", "task", "operator_id", "reference_version", "model_method", "model_seed", "model_stop_system", "analysis_scope", "information_permission", "report_production", "process_trace_comparable", "pure_planner_comparable", "final_report_identity", "human_completion", "model_completion", "human_false_stop", "model_false_stop", "human_measurement_cost", "model_measurement_cost", "model_failure_penalized_cost", "human_route_cost", "model_route_cost", "comparison_scope"),
    )
    _write_text(
        human_planning / "limitations.md",
        "# Human Planning Limitations\n\nOnly sessions matched on specimen, task, frozen geometry, and scoring reference are paired. Measurement and route costs require their respective registered units; unknown units remain unreported. Missing action sequences or first-stop/handoff records preclude process-level comparison. Missing endpoint-report identity limits provenance. Information-permission or report-production differences are full-system differences, not pure planner effects. Multiple operators do not increase the physical specimen count.\n",
    )
    write_json_atomic(output / "final_evidence_manifest.json", finalized["final_evidence_manifest"])
    _write_finalize_documents(config, finalized)
    write_checksums(output, output / "CHECKSUMS.sha256")
    manifest = finalized["final_evidence_manifest"]
    return {
        "stage": manifest["stage"],
        "output_root": str(output),
        "reference_status": finalized["reference_input_status"],
        "blind_review_status": finalized["human_review"]["status"],
        "human_session_status": finalized["human_planning"]["status"],
        "reviewed_recovery_steps": finalized["reviewed_recovery"].get(
            "stored_action_transition_count", 0
        ),
        "training_updates": 0,
        "vlm_calls": 0,
        "actor_forward_calls": 0,
        "stop_forward_calls": 0,
    }


def surface_cell_partition(cue_cells: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    """Partition the 8x8 grid into cue cells, their 8-neighbor ring, and other."""

    if (
        type(cue_cells) is not tuple
        or any(type(cell) is not int or not 0 <= cell < 64 for cell in cue_cells)
        or len(set(cue_cells)) != len(cue_cells)
    ):
        raise ValueError("surface cue cells are invalid")
    visible = set(cue_cells)
    ring = set()
    for cell in visible:
        row, column = divmod(cell, 8)
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                neighbor_row = row + row_delta
                neighbor_column = column + column_delta
                if 0 <= neighbor_row < 8 and 0 <= neighbor_column < 8:
                    ring.add(neighbor_row * 8 + neighbor_column)
    ring -= visible
    other = set(range(64)) - visible - ring
    return {
        "V": tuple(sorted(visible)),
        "R": tuple(sorted(ring)),
        "O": tuple(sorted(other)),
    }


def overlap_metrics(target: np.ndarray, cue: np.ndarray) -> dict[str, object]:
    """Compute fixed continuous overlap measures without threshold search."""

    target_mask = np.asarray(target)
    cue_mask = np.asarray(cue)
    if (
        target_mask.dtype != np.bool_
        or cue_mask.dtype != np.bool_
        or target_mask.ndim != 2
        or target_mask.shape != cue_mask.shape
    ):
        raise ValueError("spatial overlap masks are invalid")
    target_count = int(np.count_nonzero(target_mask))
    cue_count = int(np.count_nonzero(cue_mask))
    intersection = int(np.count_nonzero(target_mask & cue_mask))
    union = int(np.count_nonzero(target_mask | cue_mask))
    if not target_count:
        group = "EMPTY_TARGET"
    elif not cue_count:
        group = "NO_CUE"
    elif intersection:
        group = "CUE_OVERLAP"
    else:
        group = "CUE_DISJOINT"
    return {
        "target_pixel_count": target_count,
        "surface_pixel_count": cue_count,
        "intersection_pixel_count": intersection,
        "union_pixel_count": union,
        "target_coverage_by_surface": (
            None if not target_count else intersection / target_count
        ),
        "surface_precision_proxy": None if not cue_count else intersection / cue_count,
        "surface_target_iou": None if not union else intersection / union,
        "overlap_group": group,
    }


__all__ = [
    "STOP_SYSTEMS",
    "FrozenProcessConfig",
    "analyze_trajectory_tables",
    "derive_action_events",
    "execute_frozen_analysis",
    "execute_frozen_finalize",
    "load_frozen_process_config",
    "overlap_metrics",
    "planning_stop_accounting",
    "report_stability",
    "stop_decomposition",
    "surface_cell_partition",
    "verify_frozen_input",
]
