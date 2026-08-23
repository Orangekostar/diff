"""Observed-only state assembly for the registered MVA A5 policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .acquisition_grid import AcquisitionGrid
from .candidate_features import build_candidate_features
from .measurement_state import MeasurementState, RefinementAction, budget_record


class PolicyStateError(ValueError):
    """Raised when an A5 policy observation violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    actions: tuple[RefinementAction, ...]
    global_features: np.ndarray
    candidate_features: np.ndarray
    checkpoint: float
    used_budget: float
    remaining_budget: float


def _finite_fraction(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyStateError(f"{label} must be finite")
    output = float(value)
    if not math.isfinite(output) or not 0.0 < output <= 1.0:
        raise PolicyStateError(f"{label} must be in (0,1]")
    return output


def _readonly(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise PolicyStateError("global policy features are invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(shape)
    output.setflags(write=False)
    return output


def build_policy_observation(
    grid: AcquisitionGrid,
    state: MeasurementState,
    *,
    current_reconstruction: np.ndarray,
    current_embedding: np.ndarray,
    current_prediction: float,
    checkpoint: float,
    maximum_budget: float,
) -> PolicyObservation:
    """Build the frozen 579/8 A5 policy tensors from current observations."""

    cap = _finite_fraction(checkpoint, "checkpoint")
    maximum = _finite_fraction(maximum_budget, "maximum budget")
    if cap > maximum:
        raise PolicyStateError("checkpoint exceeds the maximum budget")
    embedding = np.asarray(current_embedding, dtype=np.float64)
    if embedding.shape != (512,) or not np.all(np.isfinite(embedding)):
        raise PolicyStateError("current embedding must contain 512 finite values")
    if isinstance(current_prediction, bool) or not isinstance(
        current_prediction, (int, float)
    ):
        raise PolicyStateError("current prediction must be finite")
    prediction = float(current_prediction)
    if not math.isfinite(prediction):
        raise PolicyStateError("current prediction must be finite")

    budget = budget_record(grid, state)
    used = budget.effective_budget
    if used > cap:
        raise PolicyStateError("current state already exceeds the checkpoint")
    remaining = maximum - used
    if remaining < 0.0:
        raise PolicyStateError("current state exceeds the maximum budget")
    levels = np.asarray(state.levels, dtype=np.float64) / 2.0
    global_features = _readonly(
        np.concatenate(
            (
                embedding,
                levels,
                np.asarray((prediction, used, remaining), dtype=np.float64),
            )
        ),
        (579,),
    )
    actions, candidate_features = build_candidate_features(
        grid,
        state,
        current_reconstruction=current_reconstruction,
        checkpoint=cap,
    )
    return PolicyObservation(
        actions=actions,
        global_features=global_features,
        candidate_features=candidate_features,
        checkpoint=cap,
        used_budget=used,
        remaining_budget=remaining,
    )


__all__ = [
    "PolicyObservation",
    "PolicyStateError",
    "build_policy_observation",
]
