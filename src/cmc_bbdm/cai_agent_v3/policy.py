"""Visible legal-action and one-step VLM C0 contracts."""

from __future__ import annotations

import torch


def vlm_first_action_mask(
    *,
    legal: torch.Tensor,
    use_vlm: bool,
    action_count: torch.Tensor,
    indicator: torch.Tensor,
    confidence: torch.Tensor,
    available: torch.Tensor,
    no_reliable: torch.Tensor,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Restrict only action zero to the highest reliable medium/high VLM cells."""

    batch = legal.shape[0] if legal.ndim == 2 else -1
    if (
        batch < 1
        or legal.shape != (batch, 64)
        or legal.dtype is not torch.bool
        or action_count.shape != (batch,)
        or indicator.shape != (batch, 64)
        or confidence.shape != (batch, 64)
        or available.shape != (batch,)
        or no_reliable.shape != (batch,)
        or available.dtype is not torch.bool
        or no_reliable.dtype is not torch.bool
        or torch.any(action_count < 0)
    ):
        raise ValueError("VLM proposal state is invalid")
    output = legal.clone()
    reasons: list[str] = []
    for row in range(batch):
        if int(action_count[row]) > 0:
            reasons.append("C0_RELEASED_AFTER_FIRST_ACTION")
            continue
        if not use_vlm:
            reasons.append("METHOD_DOES_NOT_USE_VLM_C0")
            continue
        if not bool(available[row]):
            reasons.append("VLM_UNAVAILABLE")
            continue
        if bool(no_reliable[row]):
            reasons.append("VLM_NO_RELIABLE_CUE")
            continue
        eligible = (indicator[row] > 0.0) & (confidence[row] >= 2.0 / 3.0)
        if not bool(eligible.any()):
            reasons.append(
                "NO_MEDIUM_OR_HIGH_CONFIDENCE"
                if bool((indicator[row] > 0.0).any())
                else "NO_VLM_REGION_CANDIDATES"
            )
            continue
        highest = confidence[row][eligible].max()
        restricted = legal[row] & eligible & (confidence[row] == highest)
        if not bool(restricted.any()):
            reasons.append("C0_NO_AFFORDABLE_LEGAL_CELL")
            continue
        output[row] = restricted
        reasons.append("HIGHEST_RELIABLE_CONFIDENCE_C0")
    return output, tuple(reasons)


__all__ = ["vlm_first_action_mask"]
