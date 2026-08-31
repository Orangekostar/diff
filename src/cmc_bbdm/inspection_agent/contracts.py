"""Typed policy-visible contracts for zero-start inspection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .state import GeneralizedMeasurementState, InspectionCellAction


class InspectionContractError(ValueError):
    """Raised when a policy-visible inspection contract is invalid."""


class InspectionTask(str, Enum):
    DISCOVERY = "DISCOVERY"
    FIELD = "FIELD"
    CAI = "CAI"


class InspectionDecision(str, Enum):
    FOCUS = "FOCUS"
    BROADEN = "BROADEN"
    REFINE = "REFINE"
    STOP = "STOP"


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.ascontiguousarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise InspectionContractError("observation array cannot be snapshotted") from error
    if array.shape != shape:
        raise InspectionContractError("observation array shape is invalid")
    base: object = array
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    if (
        not array.flags.writeable
        and array.flags.c_contiguous
        and isinstance(base, bytes)
    ):
        return array
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _hash(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
            digest.update(value.tobytes(order="C"))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class InspectionObservation:
    surface_rgb: np.ndarray
    surface_sha256: str
    task: InspectionTask
    native_shape: tuple[int, int]
    native_count: int
    grid_sha256: str
    measurement_state: GeneralizedMeasurementState
    acquired_positions: np.ndarray
    measurement_values: np.ndarray
    exact_acquired_count: int
    endpoint_budget: float
    action_history: tuple[InspectionCellAction, ...]
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.task) is not InspectionTask
            or type(self.native_shape) is not tuple
            or len(self.native_shape) != 2
            or any(type(value) is not int or value < 9 for value in self.native_shape)
            or type(self.native_count) is not int
            or self.native_count != self.native_shape[0] * self.native_shape[1]
            or not _sha256(self.surface_sha256)
            or not _sha256(self.grid_sha256)
            or type(self.measurement_state) is not GeneralizedMeasurementState
            or self.measurement_state.grid_sha256 != self.grid_sha256
            or type(self.exact_acquired_count) is not int
            or self.exact_acquired_count < 0
            or type(self.action_history) is not tuple
            or any(type(action) is not InspectionCellAction for action in self.action_history)
        ):
            raise InspectionContractError("inspection observation identity is invalid")
        endpoint = float(self.endpoint_budget)
        if (
            isinstance(self.endpoint_budget, bool)
            or not math.isfinite(endpoint)
            or not 0.0 < endpoint <= 1.0
            or self.exact_acquired_count / self.native_count > endpoint + 1.0e-15
        ):
            raise InspectionContractError("inspection observation budget is invalid")
        surface = np.asarray(self.surface_rgb)
        if surface.dtype != np.uint8 or surface.ndim != 3 or surface.shape[2] != 3:
            raise InspectionContractError("surface observation must be RGB uint8")
        frozen_surface = _readonly(surface, dtype=np.uint8, shape=surface.shape)
        positions = _readonly(
            self.acquired_positions,
            dtype="<i8",
            shape=(self.exact_acquired_count, 2),
        )
        values = _readonly(
            self.measurement_values,
            dtype=np.uint8,
            shape=(self.exact_acquired_count, 3),
        )
        if self.exact_acquired_count:
            linear = positions[:, 0] * self.native_shape[1] + positions[:, 1]
            if (
                np.any(positions < 0)
                or np.any(positions[:, 0] >= self.native_shape[0])
                or np.any(positions[:, 1] >= self.native_shape[1])
                or (linear.size > 1 and np.any(np.diff(linear) <= 0))
            ):
                raise InspectionContractError("acquired positions are invalid")
        history = tuple(
            (action.cell_index, action.from_level, action.to_level)
            for action in self.action_history
        )
        state = _hash(
            "inspection-agent-observation-v1",
            self.surface_sha256,
            self.task.value,
            self.native_shape,
            self.grid_sha256,
            self.measurement_state.state_sha256,
            positions,
            values,
            endpoint,
            history,
        )
        if self.state_sha256 not in ("", state):
            raise InspectionContractError("inspection observation hash changed")
        object.__setattr__(self, "surface_rgb", frozen_surface)
        object.__setattr__(self, "acquired_positions", positions)
        object.__setattr__(self, "measurement_values", values)
        object.__setattr__(self, "endpoint_budget", endpoint)
        object.__setattr__(self, "state_sha256", state)

    @property
    def effective_budget(self) -> float:
        return float(self.exact_acquired_count / self.native_count)

    @property
    def remaining_budget(self) -> float:
        return float(max(0.0, self.endpoint_budget - self.effective_budget))

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is InspectionObservation
            and self.state_sha256 == other.state_sha256
        )


@dataclass(frozen=True, slots=True)
class InspectionBeliefRecord:
    task_id: InspectionTask
    surface_hypothesis_cells: tuple[int, ...]
    surface_hypothesis_scores: np.ndarray
    observed_cells: tuple[int, ...]
    cell_levels: tuple[int, ...]
    internal_evidence_scores: np.ndarray
    coverage_fraction: float
    current_task_estimate: float
    current_task_uncertainty: float
    unexplored_fraction: float
    decision: InspectionDecision
    confidence: float
    reason_code: str
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.task_id) is not InspectionTask
            or type(self.decision) is not InspectionDecision
            or type(self.surface_hypothesis_cells) is not tuple
            or type(self.observed_cells) is not tuple
            or type(self.cell_levels) is not tuple
            or len(self.cell_levels) != 64
            or any(level not in (-1, 0, 1, 2) for level in self.cell_levels)
            or any(type(cell) is not int or not 0 <= cell < 64 for cell in self.surface_hypothesis_cells)
            or any(type(cell) is not int or not 0 <= cell < 64 for cell in self.observed_cells)
            or len(set(self.surface_hypothesis_cells)) != len(self.surface_hypothesis_cells)
            or len(set(self.observed_cells)) != len(self.observed_cells)
            or type(self.reason_code) is not str
            or not self.reason_code
        ):
            raise InspectionContractError("inspection belief identity is invalid")
        surface = _readonly(self.surface_hypothesis_scores, dtype="<f8", shape=(64,))
        internal = _readonly(self.internal_evidence_scores, dtype="<f8", shape=(64,))
        numbers = (
            float(self.coverage_fraction),
            float(self.current_task_estimate),
            float(self.current_task_uncertainty),
            float(self.unexplored_fraction),
            float(self.confidence),
        )
        if (
            not np.all(np.isfinite(surface))
            or not np.all(np.isfinite(internal))
            or not all(math.isfinite(value) for value in numbers)
            or not 0.0 <= numbers[0] <= 1.0
            or numbers[2] < 0.0
            or not 0.0 <= numbers[3] <= 1.0
            or not 0.0 <= numbers[4] <= 1.0
        ):
            raise InspectionContractError("inspection belief values are invalid")
        state = _hash(
            "inspection-agent-belief-v1",
            self.task_id.value,
            self.surface_hypothesis_cells,
            surface,
            self.observed_cells,
            self.cell_levels,
            internal,
            numbers,
            self.decision.value,
            self.reason_code,
        )
        if self.state_sha256 not in ("", state):
            raise InspectionContractError("inspection belief hash changed")
        object.__setattr__(self, "surface_hypothesis_scores", surface)
        object.__setattr__(self, "internal_evidence_scores", internal)
        object.__setattr__(self, "coverage_fraction", numbers[0])
        object.__setattr__(self, "current_task_estimate", numbers[1])
        object.__setattr__(self, "current_task_uncertainty", numbers[2])
        object.__setattr__(self, "unexplored_fraction", numbers[3])
        object.__setattr__(self, "confidence", numbers[4])
        object.__setattr__(self, "state_sha256", state)


__all__ = [
    "InspectionBeliefRecord",
    "InspectionContractError",
    "InspectionDecision",
    "InspectionObservation",
    "InspectionTask",
]
