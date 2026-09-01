from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.g1 import (
    FINAL_G1_STATUSES,
    PolicyGateEvidence,
    StopGateEvidence,
    evaluate_final_g1_decision,
    evaluate_policy_gate,
    evaluate_stop_gate,
    evaluate_task_conditioning_gate,
)
from cmc_bbdm.inspection_agent_g1.statistics import G1PairedBootstrap


def _bootstrap(
    *, point: float, lower: float, improved: int, upper: float | None = None
) -> G1PairedBootstrap:
    distribution = np.asarray([point], dtype=np.float64)
    distribution.setflags(write=False)
    return G1PairedBootstrap(
        point_estimate=point,
        ci_lower=lower,
        ci_upper=point + 0.1 if upper is None else upper,
        improved_domains=improved,
        domain_effects=tuple((f"d{index}", point) for index in range(6)),
        replicates=100_000,
        seed=2026090104,
        distribution=distribution,
        distribution_sha256=hashlib.sha256(distribution.tobytes()).hexdigest(),
    )


def test_task_policy_gate_requires_every_registered_condition() -> None:
    passing = PolicyGateEvidence(
        task=InspectionTask.FIELD,
        fixed_auebc=1.0,
        learned_auebc=0.7,
        oracle_auebc=0.0,
        baseline_minus_learned=_bootstrap(point=0.3, lower=0.1, improved=5),
        no_target_leakage=True,
        deterministic_replay=True,
    )
    result = evaluate_policy_gate(passing)
    assert result.status == "G1_FIELD_POLICY_GO"
    assert result.oracle_gap_closure == pytest.approx(0.3)

    descriptive = evaluate_policy_gate(
        PolicyGateEvidence(
            task=InspectionTask.FIELD,
            fixed_auebc=1.0,
            learned_auebc=0.9,
            oracle_auebc=0.0,
            baseline_minus_learned=_bootstrap(point=0.1, lower=0.01, improved=4),
            no_target_leakage=True,
            deterministic_replay=True,
        )
    )
    assert descriptive.status == "G1_FIELD_DESCRIPTIVE_POLICY_SIGNAL_ONLY"
    assert descriptive.passed is False


def test_task_conditioning_and_stop_have_independent_gates() -> None:
    positive = _bootstrap(point=0.2, lower=0.01, improved=4)
    assert evaluate_task_conditioning_gate(
        field_wrong_minus_correct=positive,
        field_no_task_minus_correct=positive,
        cai_wrong_minus_correct=positive,
        cai_no_task_minus_correct=positive,
    ).status == "G1_TASK_CONDITIONING_GO"

    stop = evaluate_stop_gate(
        StopGateEvidence(
            task=InspectionTask.CAI,
            normalized_measurement_saving=0.2,
            task_loss_ratio=1.02,
            premature_stop_rate=0.04,
            saving_bootstrap=_bootstrap(point=0.2, lower=0.01, improved=5),
        )
    )
    assert stop.status == "G1_CAI_STOPPING_GO"


@pytest.mark.parametrize(
    ("field", "cai", "task_conditioned", "expected"),
    (
        (True, True, True, "G1_TASK_CONDITIONED_POLICY_GO"),
        (True, True, False, "G1_ACTIVE_POLICY_GO"),
        (True, False, False, "G1_FIELD_ONLY_POLICY_GO"),
        (False, True, False, "G1_CAI_ONLY_POLICY_GO"),
        (False, False, False, "G1_POLICY_OBSERVABILITY_NO_GO"),
    ),
)
def test_final_decision_uses_only_the_registered_vocabulary(
    field: bool,
    cai: bool,
    task_conditioned: bool,
    expected: str,
) -> None:
    decision = evaluate_final_g1_decision(
        deployment_bridge_valid=True,
        field_policy_go=field,
        cai_policy_go=cai,
        task_conditioning_go=task_conditioned,
    )
    assert decision.status == expected
    assert decision.status in FINAL_G1_STATUSES
    assert decision.g2_authorized is ("NO_GO" not in expected)


def test_invalid_deployment_bridge_has_precedence() -> None:
    decision = evaluate_final_g1_decision(
        deployment_bridge_valid=False,
        field_policy_go=True,
        cai_policy_go=True,
        task_conditioning_go=True,
    )
    assert decision.status == "G1_DEPLOYMENT_BRIDGE_NO_GO"
    assert decision.g2_authorized is False
