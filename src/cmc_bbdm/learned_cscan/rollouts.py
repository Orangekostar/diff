"""Counterfactual suffix labels and true-break visible episode execution."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import StepSnapshot, exact_step_integral


@dataclass(frozen=True, slots=True, eq=False)
class CostToGoTargets:
    queried_actions: tuple[Hashable, ...]
    main_costs: np.ndarray
    auxiliary_costs: np.ndarray
    total_costs: np.ndarray
    normalized_costs: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.queried_actions)
        arrays = tuple(
            _readonly(value, (count,))
            for value in (
                self.main_costs,
                self.auxiliary_costs,
                self.total_costs,
                self.normalized_costs,
                self.probabilities,
            )
        )
        if (
            not count
            or len(set(self.queried_actions)) != count
            or any(not np.all(np.isfinite(value)) for value in arrays)
            or any(np.any(value < 0.0) for value in arrays)
            or not math.isclose(float(arrays[-1].sum()), 1.0, abs_tol=1e-12)
        ):
            raise ValueError("cost-to-go targets are invalid")
        (
            main,
            auxiliary,
            total,
            normalized,
            probabilities,
        ) = arrays
        object.__setattr__(self, "main_costs", main)
        object.__setattr__(self, "auxiliary_costs", auxiliary)
        object.__setattr__(self, "total_costs", total)
        object.__setattr__(self, "normalized_costs", normalized)
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    snapshots: tuple[StepSnapshot, ...]
    actions: tuple[object, ...]
    stopped: bool
    exhausted: bool
    terminal_snapshot: StepSnapshot


def compile_route_cost(
    positions: np.ndarray,
    *,
    native_shape: tuple[int, int],
    start_position: tuple[float, float],
) -> tuple[float, tuple[float, float]]:
    """Return the exact route cost and endpoint without computing turn count."""
    if (
        type(native_shape) is not tuple
        or len(native_shape) != 2
        or any(type(value) is not int or value < 2 for value in native_shape)
        or type(start_position) is not tuple
        or len(start_position) != 2
    ):
        raise ValueError("route request is invalid")
    start = tuple(float(value) for value in start_position)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in start):
        raise ValueError("route request is invalid")
    points = np.asarray(positions)
    if (
        points.dtype.kind not in "iu"
        or points.ndim != 2
        or points.shape[1:] != (2,)
        or (
            points.size
            and (
                np.any(points < 0)
                or np.any(points[:, 0] >= native_shape[0])
                or np.any(points[:, 1] >= native_shape[1])
            )
        )
    ):
        raise ValueError("route request is invalid")
    linear = points[:, 0] * native_shape[1] + points[:, 1]
    unique = (
        np.ascontiguousarray(points, dtype=np.int64)
        if len(points) < 2 or np.all(np.diff(linear) > 0)
        else np.unique(np.ascontiguousarray(points, dtype=np.int64), axis=0)
    )
    if not len(unique):
        return 0.0, start
    row_forward = _snake(unique, primary_axis=0)
    column_forward = _snake(unique, primary_axis=1)
    ranked = []
    for candidate_index, candidate in enumerate(
        (row_forward, row_forward[::-1], column_forward, column_forward[::-1])
    ):
        normalized = _normalized(candidate, native_shape)
        transit = float(np.linalg.norm(normalized[0] - np.asarray(start)))
        scan = float(
            np.linalg.norm(np.diff(normalized, axis=0), axis=1).sum()
            if len(normalized) > 1
            else 0.0
        )
        ranked.append(
            (transit + scan, transit, candidate_index, normalized)
        )
    total, _transit, _index, normalized = min(
        ranked, key=lambda item: (item[0], item[1], item[2])
    )
    return float(total), (float(normalized[-1, 0]), float(normalized[-1, 1]))


def cost_to_go_targets(
    *,
    current_cost: float,
    candidate_snapshots: dict[Hashable, tuple[StepSnapshot, ...]],
    temperature: float,
    auxiliary_weight: float,
) -> CostToGoTargets:
    current = float(current_cost)
    tau = float(temperature)
    weight = float(auxiliary_weight)
    if (
        isinstance(current_cost, bool)
        or not math.isfinite(current)
        or not 0.0 <= current < 1.0
        or type(candidate_snapshots) is not dict
        or not candidate_snapshots
        or not math.isfinite(tau)
        or tau <= 0.0
        or not math.isfinite(weight)
        or not 0.0 <= weight <= 1.0
    ):
        raise ValueError("cost-to-go request is invalid")
    actions = tuple(candidate_snapshots)
    main = []
    auxiliary = []
    for snapshots in candidate_snapshots.values():
        if type(snapshots) is not tuple or not snapshots:
            raise ValueError("candidate suffix is empty")
        if not math.isclose(snapshots[0].cost, current, abs_tol=1e-12):
            raise ValueError("candidate suffix does not start at current cost")
        main.append(
            exact_step_integral(
                snapshots, field="failure", start_cost=current, end_cost=1.0
            )
        )
        auxiliary.append(
            exact_step_integral(
                snapshots, field="task_loss", start_cost=current, end_cost=1.0
            )
        )
    main_array = np.asarray(main, dtype=np.float64)
    auxiliary_array = np.asarray(auxiliary, dtype=np.float64)
    total = main_array + weight * auxiliary_array
    normalized = total / max(1.0 - current, 1e-6)
    shifted = normalized - normalized.min()
    weights = np.exp(-shifted / tau)
    probabilities = weights / weights.sum()
    return CostToGoTargets(
        queried_actions=actions,
        main_costs=main_array,
        auxiliary_costs=auxiliary_array,
        total_costs=total,
        normalized_costs=normalized,
        probabilities=probabilities,
    )


def run_candidate_branches(
    *,
    world_factory: Callable[[], Any],
    base_history: tuple[object, ...],
    candidates: tuple[Hashable, ...],
    continue_branch: Callable[[Any, Any], Any],
) -> dict[Hashable, Any]:
    if (
        not callable(world_factory)
        or type(base_history) is not tuple
        or type(candidates) is not tuple
        or not candidates
        or len(set(candidates)) != len(candidates)
        or not callable(continue_branch)
    ):
        raise ValueError("candidate branch request is invalid")
    output = {}
    world_ids = set()
    for candidate in candidates:
        world = world_factory()
        if id(world) in world_ids:
            raise ValueError("counterfactual branches must use independent worlds")
        world_ids.add(id(world))
        world.replay(base_history)
        observation = world.step(candidate)
        output[candidate] = continue_branch(world, observation)
    return output


def run_visible_episode(
    world: Any,
    *,
    select_action: Callable[[Any], object],
    make_snapshot: Callable[[Any], StepSnapshot],
    should_stop: Callable[[StepSnapshot], bool],
    max_actions: int,
) -> EpisodeTrace:
    if (
        not hasattr(world, "reset")
        or not hasattr(world, "step")
        or not callable(select_action)
        or not callable(make_snapshot)
        or not callable(should_stop)
        or type(max_actions) is not int
        or max_actions < 1
    ):
        raise ValueError("visible episode request is invalid")
    observation = world.reset()
    snapshots = [make_snapshot(observation)]
    if type(snapshots[0]) is not StepSnapshot:
        raise TypeError("snapshot builder returned an invalid value")
    actions: list[object] = []
    stopped = bool(should_stop(snapshots[0]))
    while not stopped and len(actions) < max_actions:
        action = select_action(observation)
        observation = world.step(action)
        actions.append(action)
        snapshot = make_snapshot(observation)
        if type(snapshot) is not StepSnapshot:
            raise TypeError("snapshot builder returned an invalid value")
        if snapshot.cost < snapshots[-1].cost - 1e-15:
            raise ValueError("episode acquisition cost regressed")
        snapshots.append(snapshot)
        stopped = bool(should_stop(snapshot))
    return EpisodeTrace(
        snapshots=tuple(snapshots),
        actions=tuple(actions),
        stopped=stopped,
        exhausted=not stopped and len(actions) >= max_actions,
        terminal_snapshot=snapshots[-1],
    )


def _readonly(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError("cost-to-go array shape is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(shape)
    output.setflags(write=False)
    return output


def _snake(points: np.ndarray, *, primary_axis: int) -> np.ndarray:
    secondary_axis = 1 - primary_axis
    output: list[np.ndarray] = []
    for group_index, primary in enumerate(sorted(set(points[:, primary_axis]))):
        group = points[points[:, primary_axis] == primary]
        order = np.argsort(group[:, secondary_axis], kind="stable")
        if group_index % 2:
            order = order[::-1]
        output.extend(group[order])
    return np.ascontiguousarray(output, dtype=np.int64)


def _normalized(points: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    output = np.empty(points.shape, dtype=np.float64)
    output[:, 0] = points[:, 1] / (native_shape[1] - 1)
    output[:, 1] = points[:, 0] / (native_shape[0] - 1)
    return output


__all__ = [
    "CostToGoTargets",
    "EpisodeTrace",
    "compile_route_cost",
    "cost_to_go_targets",
    "run_candidate_branches",
    "run_visible_episode",
]
