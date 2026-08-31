from __future__ import annotations

from cmc_bbdm.inspection_agent.g0 import (
    FinalG0Evidence,
    G0Status,
    GateEvidence,
    assessor_authorization_gate,
    decide_g0_status,
    hierarchical_headroom_gate,
    initialization_headroom_gate,
    stopping_headroom_gate,
)


def _positive() -> GateEvidence:
    return GateEvidence(0.2, 0.05, 0.3, 5)


def test_registered_component_gates_require_effect_ci_domains_and_magnitude() -> None:
    effect = _positive()
    assert initialization_headroom_gate(
        effect,
        relative_auc_improvement=0.11,
        capture_budget_reduction=0.0,
    )
    assert not initialization_headroom_gate(
        effect,
        relative_auc_improvement=0.09,
        capture_budget_reduction=0.09,
    )
    assert hierarchical_headroom_gate(
        effect,
        relative_auebc_improvement=0.1,
        sufficiency_budget_reduction=0.0,
    )
    assert stopping_headroom_gate(
        effect,
        mean_budget_saving=0.12,
        task_loss_ratio=1.04,
    )
    assert not stopping_headroom_gate(
        effect,
        mean_budget_saving=0.12,
        task_loss_ratio=1.06,
    )


def test_assessor_gate_includes_replay_and_outer_exclusion() -> None:
    assert assessor_authorization_gate(
        zero_mae=0.4,
        endpoint_mae=0.3,
        improvement=_positive(),
        replay_valid=True,
        outer_exclusion_valid=True,
    )
    assert not assessor_authorization_gate(
        zero_mae=0.4,
        endpoint_mae=0.3,
        improvement=_positive(),
        replay_valid=False,
        outer_exclusion_valid=True,
    )


def test_final_status_uses_registered_nuanced_vocabulary() -> None:
    strongest = FinalG0Evidence(
        initialization_headroom=True,
        field_hierarchical_headroom=True,
        cai_assessor_authorized=True,
        cai_hierarchical_headroom=True,
        task_conditioning_headroom=True,
        field_stopping_headroom=True,
        cai_stopping_headroom=False,
    )
    assert decide_g0_status(strongest) is G0Status.TASK_CONDITIONED
    assert decide_g0_status(
        FinalG0Evidence(
            initialization_headroom=True,
            field_hierarchical_headroom=True,
            cai_assessor_authorized=True,
            cai_hierarchical_headroom=True,
            task_conditioning_headroom=False,
            field_stopping_headroom=True,
            cai_stopping_headroom=False,
        )
    ) is G0Status.ACTIVE_INSPECTION
    assert decide_g0_status(
        FinalG0Evidence(
            initialization_headroom=False,
            field_hierarchical_headroom=True,
            cai_assessor_authorized=False,
            cai_hierarchical_headroom=False,
            task_conditioning_headroom=False,
            field_stopping_headroom=True,
            cai_stopping_headroom=False,
        )
    ) is G0Status.FIELD_ONLY
    assert decide_g0_status(
        FinalG0Evidence(
            initialization_headroom=True,
            field_hierarchical_headroom=False,
            cai_assessor_authorized=False,
            cai_hierarchical_headroom=False,
            task_conditioning_headroom=False,
            field_stopping_headroom=False,
            cai_stopping_headroom=False,
        )
    ) is G0Status.NO_AGENTIC_HEADROOM
