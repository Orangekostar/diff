"""Deployable state-conditioned exact-cost action scoring for MAVIS."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class MAVISDynamicVoIError(ValueError):
    """Raised when deployable action scoring or teacher utility is invalid."""


@dataclass(frozen=True, slots=True)
class CandidateDescriptor:
    cell_index: int
    from_level: int
    to_level: int
    exact_added_cost: int
    native_count: int
    remaining_cost: int

    def __post_init__(self) -> None:
        if (
            type(self.cell_index) is not int
            or not 0 <= self.cell_index < 64
            or type(self.from_level) is not int
            or type(self.to_level) is not int
            or self.from_level not in (0, 1)
            or self.to_level != self.from_level + 1
            or type(self.exact_added_cost) is not int
            or type(self.native_count) is not int
            or type(self.remaining_cost) is not int
            or self.exact_added_cost <= 0
            or self.native_count <= 0
            or self.remaining_cost < self.exact_added_cost
            or self.remaining_cost > self.native_count
        ):
            raise MAVISDynamicVoIError("candidate descriptor is invalid")

    def features(self) -> np.ndarray:
        row = self.cell_index // 8
        column = self.cell_index % 8
        values = np.asarray(
            [
                row / 7.0,
                column / 7.0,
                self.from_level / 2.0,
                self.to_level / 2.0,
                self.exact_added_cost / self.native_count,
                self.remaining_cost / self.native_count,
                self.exact_added_cost / self.remaining_cost,
                (self.remaining_cost - self.exact_added_cost) / self.native_count,
            ],
            dtype=np.float64,
        )
        values.setflags(write=False)
        return values


@dataclass(frozen=True, slots=True)
class ActionScoreBatch:
    scores: torch.Tensor
    value_predictions: torch.Tensor


class DynamicActionScorer(nn.Module):
    """Shared action scorer with no target-label or future-content interface."""

    def __init__(self, *, mris_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        if (
            type(mris_dimension) is not int
            or type(hidden_dimension) is not int
            or min(mris_dimension, hidden_dimension) <= 0
        ):
            raise MAVISDynamicVoIError("dynamic scorer dimensions are invalid")
        self.mris_dimension = mris_dimension
        self.state_mlp = nn.Sequential(
            nn.Linear(mris_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(8, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, 2),
        )

    def forward(
        self,
        mris: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> ActionScoreBatch:
        if (
            not isinstance(mris, torch.Tensor)
            or not isinstance(candidate_features, torch.Tensor)
            or mris.shape != (self.mris_dimension,)
            or candidate_features.ndim != 2
            or candidate_features.shape[0] == 0
            or candidate_features.shape[1] != 8
        ):
            raise MAVISDynamicVoIError("dynamic scorer input is invalid")
        grouped = self.forward_grouped(
            mris.unsqueeze(0),
            candidate_features,
            torch.zeros(
                candidate_features.shape[0],
                dtype=torch.long,
                device=candidate_features.device,
            ),
        )
        return grouped

    def forward_grouped(
        self,
        mris: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_group_indices: torch.Tensor,
    ) -> ActionScoreBatch:
        if (
            not isinstance(mris, torch.Tensor)
            or not isinstance(candidate_features, torch.Tensor)
            or not isinstance(candidate_group_indices, torch.Tensor)
            or mris.ndim != 2
            or mris.shape[0] == 0
            or mris.shape[1] != self.mris_dimension
            or candidate_features.ndim != 2
            or candidate_features.shape[0] == 0
            or candidate_features.shape[1] != 8
            or candidate_group_indices.shape != (candidate_features.shape[0],)
            or candidate_group_indices.dtype != torch.long
            or int(candidate_group_indices.min()) < 0
            or int(candidate_group_indices.max()) >= mris.shape[0]
        ):
            raise MAVISDynamicVoIError("grouped dynamic scorer input is invalid")
        state = self.state_mlp(mris)
        actions = self.action_mlp(candidate_features)
        state_rows = state[candidate_group_indices]
        output = self.fusion(torch.cat((state_rows, actions), dim=1))
        if output.shape != (actions.shape[0], 2) or not torch.isfinite(output).all():
            raise MAVISDynamicVoIError("dynamic scorer output is invalid")
        return ActionScoreBatch(scores=output[:, 0], value_predictions=output[:, 1])

    def score_actions(
        self,
        mris: torch.Tensor,
        candidates: tuple[CandidateDescriptor, ...],
    ) -> ActionScoreBatch:
        if (
            type(candidates) is not tuple
            or not candidates
            or any(type(value) is not CandidateDescriptor for value in candidates)
        ):
            raise MAVISDynamicVoIError("candidate roster is invalid")
        parameter = next(self.parameters())
        features = torch.tensor(
            np.stack([value.features() for value in candidates]),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        return self.forward(mris.to(device=parameter.device, dtype=parameter.dtype), features)


def conditional_teacher_value(
    *,
    true_cai: float,
    current_prediction: float,
    candidate_predictions: object,
) -> np.ndarray:
    target = float(true_cai)
    current = float(current_prediction)
    candidates = np.asarray(candidate_predictions, dtype=np.float64)
    if (
        isinstance(true_cai, bool)
        or isinstance(current_prediction, bool)
        or not math.isfinite(target)
        or not math.isfinite(current)
        or candidates.ndim != 1
        or candidates.size == 0
        or not np.all(np.isfinite(candidates))
    ):
        raise MAVISDynamicVoIError("conditional teacher inputs are invalid")
    values = abs(target - current) - np.abs(target - candidates)
    output = np.frombuffer(
        np.ascontiguousarray(values, dtype="<f8").tobytes(order="C"),
        dtype="<f8",
    )
    output.setflags(write=False)
    return output


__all__ = [
    "ActionScoreBatch",
    "CandidateDescriptor",
    "DynamicActionScorer",
    "MAVISDynamicVoIError",
    "conditional_teacher_value",
]
