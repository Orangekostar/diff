from __future__ import annotations

import hashlib

import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.formal import FIXED_BASELINE_METHODS
from cmc_bbdm.inspection_agent_g1.metrics import build_engineering_curve
from cmc_bbdm.inspection_agent_g1.source_bridge import OuterFixedBridgeSelection
from cmc_bbdm.inspection_agent_g1.target_analysis import (
    G1TargetAnalysisError,
    analyze_g1_target_curves,
)
from cmc_bbdm.inspection_agent_g1.target_evaluation import G1TargetCurveRecord
from cmc_bbdm.inspection_agent_g1.target_execution import TargetPolicyVariant
from cmc_bbdm.inspection_agent_g1.target_reference import (
    TARGET_ORACLE_METHODS,
    G1TargetReferenceCurveRecord,
)

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _curve(
    domain: str,
    specimen: str,
    task: InspectionTask,
    method: str,
    loss: float,
):
    return build_engineering_curve(
        method=method,
        target_domain=domain,
        specimen_sha256=_sha(specimen),
        task=task,
        grid_sha256=_sha(f"grid-{domain}-{specimen}-{task.value}"),
        evaluator_sha256=_sha(f"evaluator-{domain}-{task.value}"),
        warm_start_sha256=_sha(f"warm-{domain}-{specimen}-{task.value}"),
        state_budgets=(0.0, 0.05, 0.1, 0.18, 0.24),
        state_losses=(loss,) * 5,
        state_sha256=tuple(
            _sha(f"state-{domain}-{specimen}-{task.value}-{method}-{index}")
            for index in range(5)
        ),
    )


def _formal_inputs():
    learned = []
    references = []
    selections = []
    variant_losses = {
        TargetPolicyVariant.PROPOSED: 0.6,
        TargetPolicyVariant.NO_TASK: 0.9,
        TargetPolicyVariant.WRONG_TASK: 0.8,
        TargetPolicyVariant.NO_SURFACE: 0.7,
        TargetPolicyVariant.SHUFFLED_SURFACE: 0.75,
    }
    for domain in DOMAINS:
        specimen = f"{domain}-specimen"
        bank_seal = _sha(f"bank-seal-{domain}")
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            for variant, loss in variant_losses.items():
                curve = _curve(domain, specimen, task, variant.value, loss)
                learned.append(
                    G1TargetCurveRecord(
                        outer_target=domain,
                        specimen_id=specimen,
                        specimen_sha256=_sha(specimen),
                        task=task,
                        variant=variant,
                        target_record_sha256=_sha(
                            f"target-{domain}-{task.value}-{variant.value}"
                        ),
                        trajectory_sha256=_sha(
                            f"trajectory-{domain}-{task.value}-{variant.value}"
                        ),
                        bank_seal_sha256=bank_seal,
                        trajectory_seal_sha256=_sha(
                            f"trajectory-seal-{domain}-{task.value}-{variant.value}"
                        ),
                        truth_view_sha256=_sha(f"truth-{domain}-{specimen}"),
                        curve=curve,
                    )
                )
            for method in (*FIXED_BASELINE_METHODS, TARGET_ORACLE_METHODS[task]):
                loss = 0.0 if method == TARGET_ORACLE_METHODS[task] else 1.0
                curve = _curve(domain, specimen, task, method, loss)
                references.append(
                    G1TargetReferenceCurveRecord(
                        outer_target=domain,
                        specimen_id=specimen,
                        specimen_sha256=_sha(specimen),
                        task=task,
                        method=method,
                        final_dependency_sha256=_sha(f"dependencies-{domain}"),
                        prior_sha256=_sha(f"prior-{domain}"),
                        assessor_sha256=_sha(f"assessor-{domain}"),
                        bank_seal_sha256=bank_seal,
                        authorization_trajectory_sha256=_sha(
                            f"authorization-{domain}-{specimen}"
                        ),
                        truth_view_sha256=_sha(f"truth-{domain}-{specimen}"),
                        action_history_sha256=_sha(
                            f"actions-{domain}-{task.value}-{method}"
                        ),
                        curve=curve,
                    )
                )
            sources = tuple(value for value in DOMAINS if value != domain)
            selections.append(
                OuterFixedBridgeSelection(
                    outer_target=domain,
                    task=task,
                    method="ZERO_UNIFORM",
                    source_domains=sources,
                    equal_domain_auebc=0.25,
                    domain_auebc=tuple((source, 0.25) for source in sources),
                    evidence_sha256=_sha(f"fixed-selection-{domain}-{task.value}"),
                )
            )
    return tuple(learned), tuple(references), tuple(selections)


def test_target_curve_analysis_issues_registered_gates_from_six_domains() -> None:
    learned, references, selections = _formal_inputs()

    result = analyze_g1_target_curves(
        learned,
        references,
        selections,
        no_target_leakage=True,
        deterministic_replay=True,
        deployment_bridge_valid=True,
    )

    assert result.field.policy_gate.status == "G1_FIELD_POLICY_GO"
    assert result.cai.policy_gate.status == "G1_CAI_POLICY_GO"
    assert result.field.oracle_gap_closure == pytest.approx(0.4)
    assert result.cai.oracle_gap_closure == pytest.approx(0.4)
    assert result.task_conditioning.gate.status == "G1_TASK_CONDITIONING_GO"
    assert result.task_conditioning.field_wrong_minus_correct.point_estimate == (
        pytest.approx(0.05)
    )
    assert result.surface_robustness.cai_no_surface_minus_correct.point_estimate == (
        pytest.approx(0.025)
    )
    assert result.final_decision.status == "G1_TASK_CONDITIONED_POLICY_GO"


def test_target_curve_analysis_rejects_an_incomplete_control_roster() -> None:
    learned, references, selections = _formal_inputs()
    incomplete = tuple(
        row
        for row in learned
        if not (
            row.outer_target == "d1"
            and row.task is InspectionTask.FIELD
            and row.variant is TargetPolicyVariant.WRONG_TASK
        )
    )

    with pytest.raises(G1TargetAnalysisError, match="roster"):
        analyze_g1_target_curves(
            incomplete,
            references,
            selections,
            no_target_leakage=True,
            deterministic_replay=True,
            deployment_bridge_valid=True,
        )
