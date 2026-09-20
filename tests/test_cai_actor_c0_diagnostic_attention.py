import torch

from cmc_bbdm.cai_agent_v3.models import SpatialCAIActor
from scripts.cai_actor_c0_diagnostic.diagnose import instrumented_actor_forward


def test_instrumented_attention_is_per_head_65_token_and_preserves_logits():
    torch.manual_seed(7)
    actor = SpatialCAIActor(use_vlm=True, use_feedback=True).eval()
    surface = torch.randn(1, 64, 512)
    cscan = torch.randn(1, 64, 512)
    measured = torch.zeros(1, 64, dtype=torch.bool)
    measured[0, [1, 8]] = True
    history = torch.zeros(1, 64)
    history[0, 1] = 1 / 64
    history[0, 8] = 2 / 64
    indicator = torch.zeros(1, 64)
    indicator[0, 3] = 1
    confidence = torch.zeros(1, 64)
    confidence[0, 3] = 1
    args = (
        surface,
        cscan,
        measured,
        history,
        indicator,
        confidence,
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
        torch.tensor([250.0]),
        torch.tensor([0.02]),
        torch.tensor([0.23]),
    )
    with torch.inference_mode():
        expected, _ = actor(*args)
        actual, _, attention, rollout = instrumented_actor_forward(actor, *args)
    assert attention.shape == (1, 2, 4, 65, 65)
    assert rollout.shape == (1, 65, 65)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(1, 2, 4, 65), atol=1e-6)
    assert torch.allclose(rollout.sum(dim=-1), torch.ones(1, 65), atol=1e-6)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
