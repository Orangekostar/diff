from __future__ import annotations

import torch

from cmc_bbdm.inspection_agent_g1.utility_distillation import (
    expected_privileged_regret,
    soft_utility_distillation_loss,
)


def test_soft_distillation_prefers_logits_matching_full_teacher_mass() -> None:
    mask = torch.zeros(1, 192, dtype=torch.bool)
    mask[0, :3] = True
    target = torch.zeros(1, 192)
    target[0, :3] = torch.tensor((0.6, 0.3, 0.1))
    matching = torch.full((1, 192), -10.0)
    matching[0, :3] = torch.log(target[0, :3])
    reversed_logits = matching.clone()
    reversed_logits[0, :3] = torch.log(torch.tensor((0.1, 0.3, 0.6)))
    assert soft_utility_distillation_loss(matching, target, mask) < (
        soft_utility_distillation_loss(reversed_logits, target, mask)
    )


def test_expected_regret_uses_only_legal_actor_probability() -> None:
    mask = torch.zeros(1, 192, dtype=torch.bool)
    mask[0, :3] = True
    logits = torch.full((1, 192), -torch.inf)
    logits[0, :3] = torch.tensor((0.0, 0.0, 0.0))
    utilities = torch.zeros(1, 192)
    utilities[0, :3] = torch.tensor((1.0, 0.5, 0.0))
    regret = expected_privileged_regret(logits, utilities, mask)
    torch.testing.assert_close(regret, torch.tensor(0.5))
