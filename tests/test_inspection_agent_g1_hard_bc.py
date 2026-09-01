from __future__ import annotations

import pytest
import torch

from cmc_bbdm.inspection_agent_g1.policy_training import (
    G1PolicyTrainingError,
    hard_behavior_cloning_loss,
)


def test_hard_bc_optimizes_only_the_selected_legal_slot() -> None:
    logits = torch.nn.Parameter(torch.zeros(2, 192))
    mask = torch.zeros(2, 192, dtype=torch.bool)
    mask[0, (3, 8, 19)] = True
    mask[1, (2, 7)] = True
    selected = torch.tensor((8, 2))
    optimizer = torch.optim.SGD((logits,), lr=1.0)
    before = hard_behavior_cloning_loss(logits, selected, mask).item()
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = hard_behavior_cloning_loss(logits, selected, mask)
        loss.backward()
        optimizer.step()
    after = hard_behavior_cloning_loss(logits, selected, mask).item()
    assert after < before
    assert logits[0, 8] > logits[0, 3]
    assert logits[1, 2] > logits[1, 7]


def test_hard_bc_rejects_an_illegal_selected_slot() -> None:
    logits = torch.zeros(1, 192)
    mask = torch.zeros(1, 192, dtype=torch.bool)
    mask[0, 0] = True
    with pytest.raises(G1PolicyTrainingError, match="selected action must be legal"):
        hard_behavior_cloning_loss(logits, torch.tensor((1,)), mask)
