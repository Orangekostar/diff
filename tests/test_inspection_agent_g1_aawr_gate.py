from __future__ import annotations

import inspect

import pytest
import torch

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.privileged_awr import (
    AAWRSourceEvidence,
    PrivilegedAWRError,
    advantage_weighted_actor_loss,
    advantage_weights,
    authorize_conditional_aawr,
    expectile_value_loss,
)

SOURCES = ("d1", "d2", "d3", "d4", "d5")


def _evidence(policy: float, *, task: InspectionTask = InspectionTask.CAI):
    return AAWRSourceEvidence(
        outer_target="d6",
        task=task,
        source_domains=SOURCES,
        fixed_auebc=(1.0,) * 5,
        policy_auebc=(policy,) * 5,
        oracle_auebc=(0.0,) * 5,
    )


def test_aawr_is_authorized_only_for_positive_low_gap_source_signal() -> None:
    decision = authorize_conditional_aawr((_evidence(0.9),))
    assert decision.status == "AUTHORIZED_SOURCE_ONLY"
    assert decision.authorized_tasks == (InspectionTask.CAI,)
    assert decision.evidence[0].improved_domains == 5
    assert decision.evidence[0].oracle_gap_closure == pytest.approx(0.1)

    high_closure = authorize_conditional_aawr((_evidence(0.7),))
    no_signal = authorize_conditional_aawr((_evidence(1.1),))
    assert high_closure.status == "NOT_RUN_NOT_AUTHORIZED"
    assert no_signal.status == "NOT_RUN_NOT_AUTHORIZED"


def test_aawr_rejects_outer_target_in_source_validation_evidence() -> None:
    with pytest.raises(PrivilegedAWRError, match="outer target"):
        AAWRSourceEvidence(
            outer_target="d6",
            task=InspectionTask.FIELD,
            source_domains=("d1", "d2", "d3", "d4", "d6"),
            fixed_auebc=(1.0,) * 5,
            policy_auebc=(0.9,) * 5,
            oracle_auebc=(0.0,) * 5,
        )


def test_aawr_weights_and_expectile_match_registered_equations() -> None:
    advantages = torch.tensor([-10.0, 0.0, 10.0])
    weights = advantage_weights(advantages, beta=1.0, maximum_weight=100.0)
    assert weights.tolist() == pytest.approx([torch.exp(torch.tensor(-10.0)).item(), 1.0, 100.0])

    predictions = torch.tensor([0.0, 0.0])
    targets = torch.tensor([-2.0, 2.0])
    loss = expectile_value_loss(predictions, targets, expectile=0.8)
    assert loss.item() == pytest.approx(2.0)


def test_aawr_actor_extraction_accepts_no_privileged_critic_tensor() -> None:
    parameters = set(inspect.signature(advantage_weighted_actor_loss).parameters)
    assert parameters == {
        "action_logits",
        "behavior_slots",
        "legal_action_mask",
        "advantages",
        "beta",
        "maximum_weight",
    }
    logits = torch.zeros((2, 192), dtype=torch.float32)
    slots = torch.tensor([0, 1], dtype=torch.long)
    legal = torch.zeros((2, 192), dtype=torch.bool)
    legal[:, :2] = True
    advantages = torch.tensor([0.0, 1.0])
    loss = advantage_weighted_actor_loss(
        logits,
        slots,
        legal,
        advantages,
        beta=1.0,
        maximum_weight=100.0,
    )
    assert torch.isfinite(loss)
