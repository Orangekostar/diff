from __future__ import annotations

import inspect

import torch

from cmc_bbdm.inspection_agent_g1.policy_model import (
    SharedActionMLP,
    StructuredInspectionPolicy,
    masked_action_probabilities,
)

FORBIDDEN = {
    "true_cai",
    "full_scan",
    "dataset_id",
    "specimen_id",
    "oracle_value",
    "future_measurement",
    "true_task_loss",
}


def _batch(batch_size: int = 2):
    generator = torch.Generator().manual_seed(20260901)
    embedding = torch.randn(batch_size, 512, generator=generator)
    scalars = torch.randn(batch_size, 17, generator=generator)
    task = torch.zeros(batch_size, 2)
    task[:, 0] = 1.0
    cells = torch.randn(batch_size, 64, 18, generator=generator)
    candidates = torch.randn(batch_size, 192, 12, generator=generator)
    mask = torch.zeros(batch_size, 192, dtype=torch.bool)
    mask[:, (0, 65, 130)] = True
    return embedding, scalars, task, cells, candidates, mask


def test_structured_policy_is_small_and_masks_illegal_actions_exactly() -> None:
    torch.manual_seed(20260901)
    model = StructuredInspectionPolicy()
    assert model.parameter_count < 1_000_000
    output = model(*_batch())
    mask = _batch()[-1]
    assert output.action_logits.shape == (2, 192)
    assert output.stop_logits.shape == (2,)
    assert torch.isneginf(output.action_logits[~mask]).all()
    probabilities = masked_action_probabilities(output.action_logits, mask)
    assert torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask]))
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))


def test_shared_mlp_obeys_the_same_action_contract_without_attention() -> None:
    torch.manual_seed(20260901)
    model = SharedActionMLP()
    assert model.parameter_count < 1_000_000
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())
    inputs = _batch(batch_size=1)
    output = model(*inputs)
    assert output.action_logits.shape == (1, 192)
    assert output.stop_logits.shape == (1,)
    assert torch.isneginf(output.action_logits[~inputs[-1]]).all()


def test_actor_model_forward_signatures_exclude_privilege() -> None:
    for model_type in (SharedActionMLP, StructuredInspectionPolicy):
        assert FORBIDDEN.isdisjoint(inspect.signature(model_type.forward).parameters)
