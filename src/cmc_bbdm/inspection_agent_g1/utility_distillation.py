"""Robust full-candidate teacher distributions and soft actor objectives."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import torch

from .contracts import ACTION_SLOT_COUNT
from .teacher import PrivilegedTeacherLabel

REGISTERED_TEMPERATURES = (0.25, 0.5, 1.0, 2.0)


class G1UtilityDistillationError(ValueError):
    """Raised when teacher utilities or masked soft targets are invalid."""


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape or (
        array.dtype != np.bool_ and not np.all(np.isfinite(array))
    ):
        raise G1UtilityDistillationError("teacher-distribution array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class TeacherDistribution:
    teacher_label_sha256: str
    probabilities: np.ndarray
    utilities: np.ndarray
    legal_action_mask: np.ndarray
    tau: float
    scale: float
    all_tied: bool
    state_sha256: str


def teacher_distribution(
    label: PrivilegedTeacherLabel,
    *,
    tau: float,
    epsilon: float = 1.0e-12,
    tie_tolerance: float = 1.0e-12,
) -> TeacherDistribution:
    temperature = float(tau)
    scale_floor = float(epsilon)
    tolerance = float(tie_tolerance)
    if (
        type(label) is not PrivilegedTeacherLabel
        or temperature not in REGISTERED_TEMPERATURES
        or isinstance(epsilon, bool)
        or isinstance(tie_tolerance, bool)
        or not math.isfinite(scale_floor)
        or scale_floor <= 0.0
        or not math.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise G1UtilityDistillationError("teacher-distribution request is invalid")
    utilities = np.zeros(ACTION_SLOT_COUNT, dtype=np.float64)
    legal = np.zeros(ACTION_SLOT_COUNT, dtype=np.bool_)
    slots = np.asarray([candidate.slot for candidate in label.candidates], dtype=np.int64)
    values = np.asarray(
        [candidate.objective_value for candidate in label.candidates], dtype=np.float64
    )
    utilities[slots] = values
    legal[slots] = True
    spread = float(np.max(values) - np.min(values))
    quartiles = np.percentile(values, (25.0, 75.0))
    robust_scale = max(float(quartiles[1] - quartiles[0]), scale_floor)
    all_tied = spread <= tolerance
    if all_tied:
        legal_probabilities = np.full(len(values), 1.0 / len(values), dtype=np.float64)
    else:
        regrets = float(np.max(values)) - values
        weights = np.exp(-regrets / (temperature * robust_scale))
        legal_probabilities = weights / np.sum(weights, dtype=np.float64)
    probabilities = np.zeros(ACTION_SLOT_COUNT, dtype=np.float64)
    probabilities[slots] = legal_probabilities
    frozen_probabilities = _readonly(
        probabilities, dtype="<f8", shape=(ACTION_SLOT_COUNT,)
    )
    frozen_utilities = _readonly(utilities, dtype="<f8", shape=(ACTION_SLOT_COUNT,))
    frozen_legal = _readonly(legal, dtype=np.bool_, shape=(ACTION_SLOT_COUNT,))
    digest = hashlib.sha256()
    digest.update(b"inspection-agent-g1-teacher-distribution-v1")
    digest.update(
        json.dumps(
            {
                "teacher_label_sha256": label.state_sha256,
                "tau": temperature,
                "scale": robust_scale,
                "all_tied": all_tied,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for array in (frozen_probabilities, frozen_utilities, frozen_legal):
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return TeacherDistribution(
        teacher_label_sha256=label.state_sha256,
        probabilities=frozen_probabilities,
        utilities=frozen_utilities,
        legal_action_mask=frozen_legal,
        tau=temperature,
        scale=robust_scale,
        all_tied=all_tied,
        state_sha256=digest.hexdigest(),
    )


def _validate_torch_rows(
    action_logits: torch.Tensor,
    values: torch.Tensor,
    legal_action_mask: torch.Tensor,
) -> None:
    if (
        not isinstance(action_logits, torch.Tensor)
        or not isinstance(values, torch.Tensor)
        or not isinstance(legal_action_mask, torch.Tensor)
        or action_logits.ndim != 2
        or action_logits.shape != values.shape
        or action_logits.shape != legal_action_mask.shape
        or action_logits.shape[1] != ACTION_SLOT_COUNT
        or not action_logits.is_floating_point()
        or not values.is_floating_point()
        or legal_action_mask.dtype is not torch.bool
        or not torch.all(legal_action_mask.any(dim=1))
        or not torch.isfinite(action_logits[legal_action_mask]).all()
        or not torch.isfinite(values).all()
    ):
        raise G1UtilityDistillationError("masked distillation tensors are invalid")


def soft_utility_distillation_loss(
    action_logits: torch.Tensor,
    target_probabilities: torch.Tensor,
    legal_action_mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_torch_rows(action_logits, target_probabilities, legal_action_mask)
    if (
        torch.any(target_probabilities < 0)
        or not torch.equal(
            target_probabilities[~legal_action_mask],
            torch.zeros_like(target_probabilities[~legal_action_mask]),
        )
        or not torch.allclose(
            target_probabilities.sum(dim=1),
            torch.ones(action_logits.shape[0], device=action_logits.device),
            atol=1.0e-6,
            rtol=0.0,
        )
    ):
        raise G1UtilityDistillationError("teacher probability mass is invalid")
    log_probabilities = torch.log_softmax(
        action_logits.masked_fill(~legal_action_mask, -torch.inf), dim=1
    )
    per_state = -(
        target_probabilities
        * torch.where(legal_action_mask, log_probabilities, 0.0)
    ).sum(dim=1)
    if sample_weights is None:
        return per_state.mean()
    if (
        not isinstance(sample_weights, torch.Tensor)
        or sample_weights.shape != per_state.shape
        or not torch.isfinite(sample_weights).all()
        or torch.any(sample_weights <= 0)
    ):
        raise G1UtilityDistillationError("sample weights are invalid")
    return torch.sum(per_state * sample_weights) / torch.sum(sample_weights)


def expected_privileged_regret(
    action_logits: torch.Tensor,
    utilities: torch.Tensor,
    legal_action_mask: torch.Tensor,
) -> torch.Tensor:
    _validate_torch_rows(action_logits, utilities, legal_action_mask)
    probabilities = torch.softmax(
        action_logits.masked_fill(~legal_action_mask, -torch.inf), dim=1
    )
    best = utilities.masked_fill(~legal_action_mask, -torch.inf).max(dim=1).values
    regrets = best.unsqueeze(1) - utilities
    per_state = (
        probabilities * torch.where(legal_action_mask, regrets, 0.0)
    ).sum(dim=1)
    return per_state.mean()


__all__ = [
    "REGISTERED_TEMPERATURES",
    "G1UtilityDistillationError",
    "TeacherDistribution",
    "expected_privileged_regret",
    "soft_utility_distillation_loss",
    "teacher_distribution",
]
