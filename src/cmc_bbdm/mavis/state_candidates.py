"""Causal one-step candidate materialization for source state-bank rows."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.interpolation import (
    RefinementPatchCache,
    reconstruct_measurement_state,
    refine_reconstruction,
)
from cmc_bbdm.mva.measurement_state import (
    RefinementAction,
    fitting_actions,
    measurement_mask,
)

from .authority import MAVISAuthority
from .contracts import InspectionState, SourceTeacherView


class MAVISStateCandidateError(ValueError):
    """Raised when a causal source candidate batch is invalid."""


class CandidateEncoder(Protocol):
    def encode(self, images: list[np.ndarray]) -> np.ndarray: ...


def _readonly(value: object, *, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.ascontiguousarray(value, dtype="<f8")
    except (TypeError, ValueError, OverflowError) as error:
        raise MAVISStateCandidateError("candidate embeddings are invalid") from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise MAVISStateCandidateError("candidate embeddings are invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class StateCandidateBatch:
    specimen_id: str
    dataset_id: str
    inspection_state_sha256: str
    endpoint_budget: float
    action_budget: float
    actions: tuple[RefinementAction, ...]
    candidate_costs: tuple[int, ...]
    current_embedding: np.ndarray
    candidate_embeddings: np.ndarray
    current_reconstruction_sha256: str
    candidate_reconstruction_sha256: tuple[str, ...]
    state_sha256: str = ""

    def __post_init__(self) -> None:
        count = len(self.actions)
        if (
            type(self.specimen_id) is not str
            or not self.specimen_id
            or type(self.dataset_id) is not str
            or not self.dataset_id
            or type(self.inspection_state_sha256) is not str
            or len(self.inspection_state_sha256) != 64
            or type(self.actions) is not tuple
            or any(type(action) is not RefinementAction for action in self.actions)
            or type(self.candidate_costs) is not tuple
            or len(self.candidate_costs) != count
            or any(type(cost) is not int or cost <= 0 for cost in self.candidate_costs)
            or type(self.candidate_reconstruction_sha256) is not tuple
            or len(self.candidate_reconstruction_sha256) != count
            or any(len(value) != 64 for value in self.candidate_reconstruction_sha256)
            or type(self.current_reconstruction_sha256) is not str
            or len(self.current_reconstruction_sha256) != 64
        ):
            raise MAVISStateCandidateError("candidate batch metadata is invalid")
        endpoint = float(self.endpoint_budget)
        action_cap = float(self.action_budget)
        if (
            isinstance(self.endpoint_budget, bool)
            or isinstance(self.action_budget, bool)
            or not math.isfinite(endpoint)
            or not math.isfinite(action_cap)
            or not 0.0 < action_cap <= endpoint <= 1.0
        ):
            raise MAVISStateCandidateError("candidate endpoint is invalid")
        current = _readonly(self.current_embedding, shape=(512,))
        candidates = _readonly(self.candidate_embeddings, shape=(count, 512))
        payload = {
            "schema": 1,
            "specimen_id": self.specimen_id,
            "dataset_id": self.dataset_id,
            "inspection_state_sha256": self.inspection_state_sha256,
            "endpoint_budget": endpoint,
            "actions": tuple(
                (action.cell_index, action.from_level, action.to_level)
                for action in self.actions
            ),
            "candidate_costs": self.candidate_costs,
            "current_embedding_sha256": hashlib.sha256(
                current.tobytes(order="C")
            ).hexdigest(),
            "candidate_embeddings_sha256": hashlib.sha256(
                candidates.tobytes(order="C")
            ).hexdigest(),
            "current_reconstruction_sha256": self.current_reconstruction_sha256,
            "candidate_reconstruction_sha256": self.candidate_reconstruction_sha256,
        }
        if action_cap != endpoint:
            payload["action_budget"] = action_cap
        state = _hash(payload)
        if self.state_sha256 not in ("", state):
            raise MAVISStateCandidateError("candidate batch state hash changed")
        object.__setattr__(self, "endpoint_budget", endpoint)
        object.__setattr__(self, "action_budget", action_cap)
        object.__setattr__(self, "current_embedding", current)
        object.__setattr__(self, "candidate_embeddings", candidates)
        object.__setattr__(self, "state_sha256", state)

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is StateCandidateBatch
            and self.state_sha256 == other.state_sha256
        )


def build_source_candidate_batch(
    authority: MAVISAuthority,
    state: InspectionState,
    *,
    dataset_id: str,
    endpoint_budget: float,
    action_budget: float | None = None,
    encoder: CandidateEncoder,
    interpolation: str = "bilinear",
) -> StateCandidateBatch:
    if (
        type(authority) is not MAVISAuthority
        or type(state) is not InspectionState
        or type(dataset_id) is not str
        or not dataset_id
        or not hasattr(encoder, "encode")
    ):
        raise MAVISStateCandidateError("source candidate inputs are invalid")
    teacher = authority.source_teacher_view(state.specimen_id)
    return build_source_candidate_batch_from_view(
        teacher,
        state,
        dataset_id=dataset_id,
        endpoint_budget=endpoint_budget,
        action_budget=action_budget,
        encoder=encoder,
        interpolation=interpolation,
    )


def build_source_candidate_batch_from_view(
    teacher: SourceTeacherView,
    state: InspectionState,
    *,
    dataset_id: str,
    endpoint_budget: float,
    action_budget: float | None = None,
    encoder: CandidateEncoder,
    interpolation: str = "bilinear",
) -> StateCandidateBatch:
    if (
        type(teacher) is not SourceTeacherView
        or type(state) is not InspectionState
        or type(dataset_id) is not str
        or not dataset_id
        or not hasattr(encoder, "encode")
    ):
        raise MAVISStateCandidateError("source candidate inputs are invalid")
    endpoint = float(endpoint_budget)
    action_cap = endpoint if action_budget is None else float(action_budget)
    if (
        isinstance(endpoint_budget, bool)
        or isinstance(action_budget, bool)
        or not math.isfinite(endpoint)
        or not math.isfinite(action_cap)
        or not state.effective_budget <= action_cap <= state.checkpoint
        or not action_cap <= endpoint <= 1.0
    ):
        raise MAVISStateCandidateError("source candidate endpoint is invalid")
    context = teacher.policy_context
    expected_values = teacher.full_scan[
        state.acquired_positions[:, 0], state.acquired_positions[:, 1]
    ]
    if (
        teacher.dataset_id != dataset_id
        or teacher.specimen_id != state.specimen_id
        or context.native_shape != state.native_shape
        or not np.array_equal(context.context_features, state.context_features)
        or not np.array_equal(expected_values, state.measurement_values)
    ):
        raise MAVISStateCandidateError("inspection state is not authority-issued")
    grid = build_acquisition_grid(
        *state.native_shape,
        initial_budget=state.initial_budget,
    )
    measurement_state = state.measurement_state
    actions = fitting_actions(grid, measurement_state, action_cap)
    current_result = reconstruct_measurement_state(
        teacher.full_scan,
        grid,
        measurement_state,
        interpolation=interpolation,
        specimen_id=state.specimen_id,
        dataset_id=dataset_id,
    )
    current_mask = measurement_mask(grid, measurement_state)
    patch_cache = RefinementPatchCache(image=teacher.full_scan, grid=grid)
    candidates: list[np.ndarray] = []
    costs: list[int] = []
    candidate_hashes: list[str] = []
    for action in actions:
        cell = grid.cells[action.cell_index]
        rows = np.asarray(cell.rows[action.to_level], dtype=np.int64)
        columns = np.asarray(cell.columns[action.to_level], dtype=np.int64)
        cost = int(np.count_nonzero(~current_mask[np.ix_(rows, columns)]))
        effective_budget = (state.exact_acquired_count + cost) / state.native_count
        if cost <= 0 or effective_budget > action_cap + 1.0e-15:
            raise MAVISStateCandidateError("candidate exact cost is invalid")
        image = refine_reconstruction(
            teacher.full_scan,
            grid,
            measurement_state,
            current_result.image,
            action,
            interpolation=interpolation,
            current_mask=current_mask,
            patch_cache=patch_cache,
        )
        candidates.append(image)
        costs.append(cost)
        candidate_hashes.append(
            hashlib.sha256(image.tobytes(order="C")).hexdigest()
        )
    try:
        encoded = np.asarray(
            encoder.encode([current_result.image, *candidates]),
            dtype="<f8",
        )
    except Exception as error:
        raise MAVISStateCandidateError("candidate encoding failed") from error
    if encoded.shape != (len(actions) + 1, 512) or not np.all(np.isfinite(encoded)):
        raise MAVISStateCandidateError("candidate encoder output is invalid")
    return StateCandidateBatch(
        specimen_id=state.specimen_id,
        dataset_id=dataset_id,
        inspection_state_sha256=state.state_sha256,
        endpoint_budget=endpoint,
        action_budget=action_cap,
        actions=actions,
        candidate_costs=tuple(costs),
        current_embedding=encoded[0],
        candidate_embeddings=encoded[1:],
        current_reconstruction_sha256=current_result.output_sha256,
        candidate_reconstruction_sha256=tuple(candidate_hashes),
    )


__all__ = [
    "CandidateEncoder",
    "MAVISStateCandidateError",
    "StateCandidateBatch",
    "build_source_candidate_batch",
    "build_source_candidate_batch_from_view",
]
