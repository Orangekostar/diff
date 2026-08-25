"""Preference-first multi-task losses for dynamic mechanical VoI."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


class MAVISLossError(ValueError):
    """Raised when dynamic VoI loss tensors or weights are invalid."""


@dataclass(frozen=True, slots=True)
class DynamicVoILoss:
    total: torch.Tensor
    cai: torch.Tensor
    pair: torch.Tensor
    listwise: torch.Tensor
    value: torch.Tensor


def _action_tensors(
    scores: torch.Tensor,
    teacher_values: torch.Tensor,
) -> None:
    if (
        not isinstance(scores, torch.Tensor)
        or not isinstance(teacher_values, torch.Tensor)
        or scores.ndim != 1
        or teacher_values.shape != scores.shape
        or scores.numel() == 0
        or not torch.isfinite(scores).all()
        or not torch.isfinite(teacher_values).all()
    ):
        raise MAVISLossError("dynamic VoI action tensors are invalid")


def pairwise_preference_loss(
    action_scores: torch.Tensor,
    teacher_values: torch.Tensor,
    *,
    tie_tolerance: float = 1.0e-12,
) -> torch.Tensor:
    _action_tensors(action_scores, teacher_values)
    tolerance = float(tie_tolerance)
    if (
        isinstance(tie_tolerance, bool)
        or not math.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise MAVISLossError("pairwise tie tolerance is invalid")
    count = action_scores.numel()
    rows, columns = torch.triu_indices(count, count, offset=1, device=action_scores.device)
    teacher_delta = teacher_values[rows] - teacher_values[columns]
    active = torch.abs(teacher_delta) > tolerance
    if not torch.any(active):
        return action_scores.sum() * 0.0
    score_delta = action_scores[rows[active]] - action_scores[columns[active]]
    direction = torch.sign(teacher_delta[active])
    return torch.mean(F.softplus(-direction * score_delta))


def listwise_preference_loss(
    action_scores: torch.Tensor,
    teacher_values: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    _action_tensors(action_scores, teacher_values)
    value = float(temperature)
    if isinstance(temperature, bool) or not math.isfinite(value) or value <= 0.0:
        raise MAVISLossError("listwise temperature is invalid")
    target = torch.softmax(teacher_values / value, dim=0)
    return -torch.sum(target * torch.log_softmax(action_scores, dim=0))


def dynamic_voi_loss(
    *,
    cai_prediction: torch.Tensor,
    cai_target: torch.Tensor,
    action_scores: torch.Tensor,
    value_predictions: torch.Tensor,
    teacher_values: torch.Tensor,
    cai_weight: float,
    pair_weight: float,
    list_weight: float,
    value_weight: float,
) -> DynamicVoILoss:
    _action_tensors(action_scores, teacher_values)
    if (
        not isinstance(cai_prediction, torch.Tensor)
        or not isinstance(cai_target, torch.Tensor)
        or cai_prediction.numel() != 1
        or cai_target.numel() != 1
        or value_predictions.shape != teacher_values.shape
        or not torch.isfinite(cai_prediction).all()
        or not torch.isfinite(cai_target).all()
        or not torch.isfinite(value_predictions).all()
    ):
        raise MAVISLossError("dynamic VoI auxiliary tensors are invalid")
    weights = tuple(float(value) for value in (cai_weight, pair_weight, list_weight, value_weight))
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise MAVISLossError("dynamic VoI loss weight is invalid")
    cai = torch.mean(torch.abs(cai_prediction.reshape(-1) - cai_target.reshape(-1)))
    pair = pairwise_preference_loss(action_scores, teacher_values)
    listwise = listwise_preference_loss(action_scores, teacher_values)
    value = torch.mean(torch.abs(value_predictions - teacher_values))
    total = weights[0] * cai + weights[1] * pair + weights[2] * listwise + weights[3] * value
    if not torch.isfinite(total):
        raise MAVISLossError("dynamic VoI total loss is invalid")
    return DynamicVoILoss(
        total=total,
        cai=cai,
        pair=pair,
        listwise=listwise,
        value=value,
    )


def grouped_dynamic_voi_loss(
    *,
    cai_predictions: torch.Tensor,
    cai_targets: torch.Tensor,
    action_scores: torch.Tensor,
    value_predictions: torch.Tensor,
    teacher_values: torch.Tensor,
    candidate_group_indices: torch.Tensor,
    cai_weight: float,
    pair_weight: float,
    list_weight: float,
    value_weight: float,
    tie_tolerance: float = 1.0e-12,
) -> DynamicVoILoss:
    _action_tensors(action_scores, teacher_values)
    group_count = int(cai_predictions.numel())
    if (
        not isinstance(cai_predictions, torch.Tensor)
        or not isinstance(cai_targets, torch.Tensor)
        or not isinstance(value_predictions, torch.Tensor)
        or not isinstance(candidate_group_indices, torch.Tensor)
        or cai_predictions.ndim != 1
        or cai_targets.shape != cai_predictions.shape
        or value_predictions.shape != teacher_values.shape
        or candidate_group_indices.shape != teacher_values.shape
        or candidate_group_indices.dtype != torch.long
        or group_count == 0
        or int(candidate_group_indices.min()) != 0
        or int(candidate_group_indices.max()) != group_count - 1
        or not torch.isfinite(cai_predictions).all()
        or not torch.isfinite(cai_targets).all()
        or not torch.isfinite(value_predictions).all()
    ):
        raise MAVISLossError("grouped dynamic VoI tensors are invalid")
    counts = torch.bincount(candidate_group_indices, minlength=group_count)
    expected_groups = torch.repeat_interleave(
        torch.arange(group_count, device=action_scores.device),
        counts,
    )
    if torch.any(counts <= 0) or not torch.equal(
        candidate_group_indices,
        expected_groups,
    ):
        raise MAVISLossError("grouped dynamic candidate ordering is invalid")
    tolerance = float(tie_tolerance)
    weights = tuple(
        float(value)
        for value in (cai_weight, pair_weight, list_weight, value_weight)
    )
    if (
        isinstance(tie_tolerance, bool)
        or not math.isfinite(tolerance)
        or tolerance < 0.0
        or any(not math.isfinite(value) or value < 0.0 for value in weights)
    ):
        raise MAVISLossError("grouped dynamic VoI parameters are invalid")

    maximum = int(counts.max())
    starts = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
    positions = torch.arange(action_scores.numel(), device=action_scores.device) - starts
    flat_indices = candidate_group_indices * maximum + positions
    shape = (group_count, maximum)
    mask = torch.zeros(
        group_count * maximum,
        dtype=torch.bool,
        device=action_scores.device,
    ).scatter(0, flat_indices, True).reshape(shape)

    def pad(values: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        return torch.full(
            (group_count * maximum,),
            fill,
            dtype=values.dtype,
            device=values.device,
        ).scatter(0, flat_indices, values).reshape(shape)

    score_pad = pad(action_scores)
    value_pad = pad(value_predictions)
    teacher_pad = pad(teacher_values)
    cai_per_group = torch.abs(cai_predictions - cai_targets)
    value_per_group = (
        torch.where(mask, torch.abs(value_pad - teacher_pad), 0.0).sum(dim=1)
        / counts
    )
    target_distribution = torch.softmax(
        teacher_pad.masked_fill(~mask, float("-inf")),
        dim=1,
    )
    score_log_probabilities = torch.log_softmax(
        score_pad.masked_fill(~mask, float("-inf")),
        dim=1,
    )
    list_per_group = -torch.where(
        mask,
        target_distribution * score_log_probabilities,
        0.0,
    ).sum(dim=1)

    upper = torch.triu(
        torch.ones((maximum, maximum), dtype=torch.bool, device=action_scores.device),
        diagonal=1,
    )
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & upper.unsqueeze(0)
    teacher_delta = teacher_pad.unsqueeze(2) - teacher_pad.unsqueeze(1)
    active = pair_mask & (torch.abs(teacher_delta) > tolerance)
    score_delta = score_pad.unsqueeze(2) - score_pad.unsqueeze(1)
    pair_values = F.softplus(-torch.sign(teacher_delta) * score_delta)
    active_count = active.sum(dim=(1, 2))
    pair_per_group = torch.where(active, pair_values, 0.0).sum(dim=(1, 2)) / torch.clamp(
        active_count,
        min=1,
    )
    total_per_group = (
        weights[0] * cai_per_group
        + weights[1] * pair_per_group
        + weights[2] * list_per_group
        + weights[3] * value_per_group
    )
    total = total_per_group.mean()
    if not torch.isfinite(total):
        raise MAVISLossError("grouped dynamic VoI total loss is invalid")
    return DynamicVoILoss(
        total=total,
        cai=cai_per_group.mean(),
        pair=pair_per_group.mean(),
        listwise=list_per_group.mean(),
        value=value_per_group.mean(),
    )


__all__ = [
    "DynamicVoILoss",
    "MAVISLossError",
    "dynamic_voi_loss",
    "grouped_dynamic_voi_loss",
    "listwise_preference_loss",
    "pairwise_preference_loss",
]
