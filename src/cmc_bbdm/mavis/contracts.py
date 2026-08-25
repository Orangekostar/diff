"""Typed visibility boundaries for causal MAVIS inspection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.mva.measurement_state import MeasurementState, RefinementAction


class MAVISContractError(ValueError):
    """Raised when a MAVIS visibility or state contract is invalid."""


def _readonly(
    value: object,
    *,
    dtype: object,
    shape: tuple[int, ...] | None,
    label: str,
    finite: bool = True,
) -> np.ndarray:
    try:
        array = np.ascontiguousarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise MAVISContractError(f"{label} cannot be snapshotted") from error
    if shape is not None and array.shape != shape:
        raise MAVISContractError(f"{label} shape is invalid")
    if finite and not np.all(np.isfinite(array)):
        raise MAVISContractError(f"{label} contains non-finite values")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    output.setflags(write=False)
    return output


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


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise MAVISContractError(f"{label} is invalid")
    return value


def _native_shape(value: object) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int or item < 9 for item in value)
    ):
        raise MAVISContractError("native shape is invalid")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class PolicyContext:
    specimen_id: str
    context_features: np.ndarray
    native_shape: tuple[int, int]
    native_count: int
    state_sha256: str = ""

    def __post_init__(self) -> None:
        specimen_id = _identifier(self.specimen_id, "specimen ID")
        shape = _native_shape(self.native_shape)
        if type(self.native_count) is not int or self.native_count != shape[0] * shape[1]:
            raise MAVISContractError("native count is invalid")
        features = _readonly(
            self.context_features,
            dtype="<f8",
            shape=(34,),
            label="policy context features",
        )
        state = _hash("mavis-policy-context-v1", specimen_id, features, shape)
        if self.state_sha256 not in ("", state):
            raise MAVISContractError("policy context state hash changed")
        object.__setattr__(self, "specimen_id", specimen_id)
        object.__setattr__(self, "context_features", features)
        object.__setattr__(self, "native_shape", shape)
        object.__setattr__(self, "state_sha256", state)

    def __eq__(self, other: object) -> bool:
        return type(other) is PolicyContext and self.state_sha256 == other.state_sha256


@dataclass(frozen=True, slots=True)
class SourceTeacherView:
    specimen_id: str
    dataset_id: str
    policy_context: PolicyContext
    full_scan: np.ndarray
    true_cai: float
    source_image_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.specimen_id, "specimen ID")
        _identifier(self.dataset_id, "dataset ID")
        if type(self.policy_context) is not PolicyContext:
            raise MAVISContractError("issued policy context is required")
        scan = _readonly(
            self.full_scan,
            dtype=np.uint8,
            shape=(*self.policy_context.native_shape, 3),
            label="teacher full scan",
            finite=False,
        )
        cai = float(self.true_cai)
        if isinstance(self.true_cai, bool) or not math.isfinite(cai):
            raise MAVISContractError("teacher true CAI is invalid")
        _identifier(self.source_image_sha256, "source image SHA256")
        if len(self.source_image_sha256) != 64:
            raise MAVISContractError("source image SHA256 is invalid")
        object.__setattr__(self, "full_scan", scan)
        object.__setattr__(self, "true_cai", cai)


@dataclass(frozen=True, slots=True)
class EvaluationView:
    specimen_id: str
    dataset_id: str
    full_scan: np.ndarray
    true_cai: float
    source_image_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.specimen_id, "specimen ID")
        _identifier(self.dataset_id, "dataset ID")
        scan = np.asarray(self.full_scan)
        if scan.ndim != 3 or scan.shape[2] != 3:
            raise MAVISContractError("evaluation full scan shape is invalid")
        frozen = _readonly(
            scan,
            dtype=np.uint8,
            shape=scan.shape,
            label="evaluation full scan",
            finite=False,
        )
        cai = float(self.true_cai)
        if isinstance(self.true_cai, bool) or not math.isfinite(cai):
            raise MAVISContractError("evaluation true CAI is invalid")
        _identifier(self.source_image_sha256, "source image SHA256")
        if len(self.source_image_sha256) != 64:
            raise MAVISContractError("source image SHA256 is invalid")
        object.__setattr__(self, "full_scan", frozen)
        object.__setattr__(self, "true_cai", cai)


@dataclass(frozen=True, slots=True, eq=False)
class InspectionState:
    specimen_id: str
    context_features: np.ndarray
    native_shape: tuple[int, int]
    native_count: int
    initial_budget: float
    checkpoint: float
    grid_state_sha256: str
    measurement_levels: tuple[int, ...]
    acquired_positions: np.ndarray
    measurement_values: np.ndarray
    exact_acquired_count: int
    action_history: tuple[RefinementAction, ...]
    state_sha256: str = ""

    def __post_init__(self) -> None:
        specimen_id = _identifier(self.specimen_id, "specimen ID")
        shape = _native_shape(self.native_shape)
        native_count = shape[0] * shape[1]
        if type(self.native_count) is not int or self.native_count != native_count:
            raise MAVISContractError("native count is invalid")
        initial_budget = float(self.initial_budget)
        checkpoint = float(self.checkpoint)
        if (
            isinstance(self.initial_budget, bool)
            or isinstance(self.checkpoint, bool)
            or not math.isfinite(initial_budget)
            or not math.isfinite(checkpoint)
            or not 0.0 < initial_budget <= checkpoint <= 1.0
        ):
            raise MAVISContractError("inspection budget is invalid")
        if type(self.grid_state_sha256) is not str or len(self.grid_state_sha256) != 64:
            raise MAVISContractError("grid state hash is invalid")
        if (
            type(self.measurement_levels) is not tuple
            or len(self.measurement_levels) != 64
            or any(type(level) is not int or level not in (0, 1, 2) for level in self.measurement_levels)
        ):
            raise MAVISContractError("measurement levels are invalid")
        if type(self.exact_acquired_count) is not int or self.exact_acquired_count <= 0:
            raise MAVISContractError("exact acquired count is invalid")
        positions = _readonly(
            self.acquired_positions,
            dtype="<i8",
            shape=(self.exact_acquired_count, 2),
            label="acquired positions",
        )
        values = _readonly(
            self.measurement_values,
            dtype=np.uint8,
            shape=(self.exact_acquired_count, 3),
            label="measurement values",
            finite=False,
        )
        linear_positions = positions[:, 0] * shape[1] + positions[:, 1]
        if (
            np.any(positions < 0)
            or np.any(positions[:, 0] >= shape[0])
            or np.any(positions[:, 1] >= shape[1])
            or np.any(np.diff(linear_positions) <= 0)
            or self.exact_acquired_count / native_count > checkpoint + 1.0e-15
        ):
            raise MAVISContractError("acquired positions or budget are invalid")
        if type(self.action_history) is not tuple or any(
            type(action) is not RefinementAction for action in self.action_history
        ):
            raise MAVISContractError("action history is invalid")
        features = _readonly(
            self.context_features,
            dtype="<f8",
            shape=(34,),
            label="inspection context features",
        )
        state = _hash(
            "mavis-inspection-state-v1",
            specimen_id,
            features,
            shape,
            initial_budget,
            checkpoint,
            self.grid_state_sha256,
            self.measurement_levels,
            positions,
            values,
            tuple(
                (action.cell_index, action.from_level, action.to_level)
                for action in self.action_history
            ),
        )
        if self.state_sha256 not in ("", state):
            raise MAVISContractError("inspection state hash changed")
        object.__setattr__(self, "specimen_id", specimen_id)
        object.__setattr__(self, "context_features", features)
        object.__setattr__(self, "native_shape", shape)
        object.__setattr__(self, "initial_budget", initial_budget)
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(self, "acquired_positions", positions)
        object.__setattr__(self, "measurement_values", values)
        object.__setattr__(self, "state_sha256", state)

    @property
    def effective_budget(self) -> float:
        return float(self.exact_acquired_count / self.native_count)

    @property
    def remaining_count(self) -> int:
        return int(self.native_count - self.exact_acquired_count)

    @property
    def measurement_state(self) -> MeasurementState:
        return MeasurementState(
            grid_sha256=self.grid_state_sha256,
            levels=self.measurement_levels,
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is InspectionState and self.state_sha256 == other.state_sha256


__all__ = [
    "EvaluationView",
    "InspectionState",
    "MAVISContractError",
    "PolicyContext",
    "SourceTeacherView",
]
