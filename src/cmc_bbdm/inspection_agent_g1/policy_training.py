"""Masked supervised objectives for observable G1 policies."""

from __future__ import annotations

import torch

from .contracts import ACTION_SLOT_COUNT


class G1PolicyTrainingError(ValueError):
    """Raised when supervised policy targets violate legal-action contracts."""


def hard_behavior_cloning_loss(
    action_logits: torch.Tensor,
    selected_slots: torch.Tensor,
    legal_action_mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        not isinstance(action_logits, torch.Tensor)
        or not isinstance(selected_slots, torch.Tensor)
        or not isinstance(legal_action_mask, torch.Tensor)
        or action_logits.ndim != 2
        or action_logits.shape[1] != ACTION_SLOT_COUNT
        or selected_slots.shape != (action_logits.shape[0],)
        or selected_slots.dtype is not torch.long
        or legal_action_mask.shape != action_logits.shape
        or legal_action_mask.dtype is not torch.bool
        or not action_logits.is_floating_point()
        or torch.any(selected_slots < 0)
        or torch.any(selected_slots >= ACTION_SLOT_COUNT)
        or not torch.isfinite(action_logits[legal_action_mask]).all()
    ):
        raise G1PolicyTrainingError("hard behavior-cloning tensors are invalid")
    rows = torch.arange(action_logits.shape[0], device=action_logits.device)
    if not torch.all(legal_action_mask[rows, selected_slots]):
        raise G1PolicyTrainingError("selected action must be legal")
    log_probabilities = torch.log_softmax(
        action_logits.masked_fill(~legal_action_mask, -torch.inf), dim=1
    )
    per_state = -log_probabilities[rows, selected_slots]
    if sample_weights is None:
        return per_state.mean()
    if (
        not isinstance(sample_weights, torch.Tensor)
        or sample_weights.shape != per_state.shape
        or not torch.isfinite(sample_weights).all()
        or torch.any(sample_weights <= 0)
    ):
        raise G1PolicyTrainingError("sample weights are invalid")
    return torch.sum(per_state * sample_weights) / torch.sum(sample_weights)


__all__ = ["G1PolicyTrainingError", "hard_behavior_cloning_loss"]
