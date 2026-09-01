"""Immutable actor-visible contracts for the G1 inspection policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask

RECONSTRUCTION_EMBEDDING_DIMENSION = 512
GLOBAL_SCALAR_DIMENSION = 17
CELL_COUNT = 64
CELL_FEATURE_DIMENSION = 18
ACTION_SLOT_COUNT = 192
CANDIDATE_FEATURE_DIMENSION = 12


class G1ContractError(ValueError):
    """Raised when an observable G1 actor contract is invalid."""


class TaskTokenMode(str, Enum):
    CORRECT = "CORRECT"
    NO_TASK = "NO_TASK"
    WRONG_TASK = "WRONG_TASK"


class CAIContextMode(str, Enum):
    TASK_SPECIFIC_MASKED = "TASK_SPECIFIC_MASKED"
    SHARED_OBSERVABLE_STATE_CONTEXT = "SHARED_OBSERVABLE_STATE_CONTEXT"


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.ascontiguousarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise G1ContractError("G1 actor tensor cannot be snapshotted") from error
    if array.shape != shape or (
        array.dtype != np.bool_ and not np.all(np.isfinite(array))
    ):
        raise G1ContractError("G1 actor tensor shape or values are invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _task_token(task: InspectionTask, mode: TaskTokenMode) -> tuple[float, float]:
    if mode is TaskTokenMode.NO_TASK:
        return (0.0, 0.0)
    field = (1.0, 0.0) if task is InspectionTask.FIELD else (0.0, 1.0)
    return field if mode is TaskTokenMode.CORRECT else field[::-1]


@dataclass(frozen=True, slots=True, eq=False)
class G1PolicyState:
    task: InspectionTask
    task_token_mode: TaskTokenMode
    cai_context_mode: CAIContextMode
    observation_sha256: str
    reconstruction_sha256: str
    surface_hypothesis_sha256: str
    grid_sha256: str
    reconstruction_embedding: np.ndarray
    global_scalars: np.ndarray
    task_token: np.ndarray
    cell_features: np.ndarray
    candidate_features: np.ndarray
    legal_action_mask: np.ndarray
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or type(self.task_token_mode) is not TaskTokenMode
            or type(self.cai_context_mode) is not CAIContextMode
            or not all(
                _valid_sha256(value)
                for value in (
                    self.observation_sha256,
                    self.reconstruction_sha256,
                    self.surface_hypothesis_sha256,
                    self.grid_sha256,
                )
            )
        ):
            raise G1ContractError("G1 policy-state identity is invalid")
        embedding = _readonly(
            self.reconstruction_embedding,
            dtype="<f8",
            shape=(RECONSTRUCTION_EMBEDDING_DIMENSION,),
        )
        global_scalars = _readonly(
            self.global_scalars,
            dtype="<f8",
            shape=(GLOBAL_SCALAR_DIMENSION,),
        )
        task_token = _readonly(self.task_token, dtype="<f8", shape=(2,))
        cells = _readonly(
            self.cell_features,
            dtype="<f8",
            shape=(CELL_COUNT, CELL_FEATURE_DIMENSION),
        )
        candidates = _readonly(
            self.candidate_features,
            dtype="<f8",
            shape=(ACTION_SLOT_COUNT, CANDIDATE_FEATURE_DIMENSION),
        )
        mask = _readonly(
            self.legal_action_mask,
            dtype=np.bool_,
            shape=(ACTION_SLOT_COUNT,),
        )
        if not np.array_equal(
            task_token,
            np.asarray(_task_token(self.task, self.task_token_mode), dtype=np.float64),
        ):
            raise G1ContractError("G1 task token contradicts its registered mode")
        cai_estimate, cai_present = global_scalars[12:14]
        if (
            cai_present not in (0.0, 1.0)
            or (cai_present == 0.0 and cai_estimate != 0.0)
            or (
                self.cai_context_mode is CAIContextMode.TASK_SPECIFIC_MASKED
                and self.task is InspectionTask.FIELD
                and cai_present != 0.0
            )
        ):
            raise G1ContractError("G1 CAI context contradicts its registered mode")
        digest = hashlib.sha256()
        digest.update(b"inspection-agent-g1-policy-state-v1")
        digest.update(
            json.dumps(
                {
                    "task": self.task.value,
                    "task_token_mode": self.task_token_mode.value,
                    "cai_context_mode": self.cai_context_mode.value,
                    "observation_sha256": self.observation_sha256,
                    "reconstruction_sha256": self.reconstruction_sha256,
                    "surface_hypothesis_sha256": self.surface_hypothesis_sha256,
                    "grid_sha256": self.grid_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        for value in (embedding, global_scalars, task_token, cells, candidates, mask):
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
            digest.update(value.tobytes(order="C"))
        state = digest.hexdigest()
        if self.state_sha256 not in ("", state):
            raise G1ContractError("G1 policy-state hash changed")
        object.__setattr__(self, "reconstruction_embedding", embedding)
        object.__setattr__(self, "global_scalars", global_scalars)
        object.__setattr__(self, "task_token", task_token)
        object.__setattr__(self, "cell_features", cells)
        object.__setattr__(self, "candidate_features", candidates)
        object.__setattr__(self, "legal_action_mask", mask)
        object.__setattr__(self, "state_sha256", state)

    def __eq__(self, other: object) -> bool:
        return type(other) is G1PolicyState and self.state_sha256 == other.state_sha256


__all__ = [
    "ACTION_SLOT_COUNT",
    "CANDIDATE_FEATURE_DIMENSION",
    "CELL_COUNT",
    "CELL_FEATURE_DIMENSION",
    "GLOBAL_SCALAR_DIMENSION",
    "RECONSTRUCTION_EMBEDDING_DIMENSION",
    "CAIContextMode",
    "G1ContractError",
    "G1PolicyState",
    "TaskTokenMode",
]
