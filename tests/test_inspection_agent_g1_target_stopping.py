from __future__ import annotations

import hashlib

import pytest
from test_inspection_agent_g1_target_execution import _record

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.contracts import TaskTokenMode
from cmc_bbdm.inspection_agent_g1.metrics import build_engineering_curve
from cmc_bbdm.inspection_agent_g1.rollout import ClosedLoopTrajectory, SurfaceVariant
from cmc_bbdm.inspection_agent_g1.source_bridge import OuterFixedBridgeSelection
from cmc_bbdm.inspection_agent_g1.target_evaluation import G1TargetCurveRecord
from cmc_bbdm.inspection_agent_g1.target_execution import (
    G1TargetTrajectoryRecord,
    TargetPolicyVariant,
)
from cmc_bbdm.inspection_agent_g1.target_reference import (
    G1TargetReferenceCurveRecord,
)
from cmc_bbdm.inspection_agent_g1.target_stopping import (
    G1TargetStopOutcome,
    analyze_g1_target_stopping,
    materialize_g1_target_stop_outcomes,
)

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _outcomes(*, premature_domain: str | None = None):
    output = []
    for domain_index, domain in enumerate(DOMAINS):
        authorized = domain_index < 4
        premature = authorized and domain == premature_domain
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            output.append(
                G1TargetStopOutcome(
                    outer_target=domain,
                    specimen_id=f"{domain}-specimen",
                    specimen_sha256=_sha(f"{domain}-specimen"),
                    task=task,
                    stop_authorized=authorized,
                    threshold=0.9 if authorized else None,
                    stopped=authorized,
                    budget=0.2 if authorized else 0.25,
                    normalized_measurement_saving=0.2 if authorized else 0.0,
                    task_loss=1.02 if not premature else 1.2,
                    reference_task_loss=1.0,
                    task_loss_ratio=1.02 if not premature else 1.2,
                    premature_stop=premature,
                    false_continue=False if authorized else None,
                    proposed_trajectory_sha256=_sha(
                        f"proposed-{domain}-{task.value}"
                    ),
                    deployed_trajectory_sha256=_sha(
                        f"deployed-{domain}-{task.value}"
                    ),
                    deployed_curve_sha256=_sha(f"curve-{domain}-{task.value}"),
                    reference_curve_sha256=_sha(
                        f"reference-{domain}-{task.value}"
                    ),
                )
            )
    return tuple(output)


def test_target_stopping_analysis_requires_four_improved_domains() -> None:
    result = analyze_g1_target_stopping(_outcomes())

    assert result.field.gate is not None
    assert result.field.gate.status == "G1_FIELD_STOPPING_GO"
    assert result.cai.gate is not None
    assert result.cai.gate.status == "G1_CAI_STOPPING_GO"
    assert result.field.normalized_measurement_saving == pytest.approx(2.0 / 15.0)
    assert result.field.saving_bootstrap.improved_domains == 4
    assert result.field.authorized_domains == 4
    assert result.field.fraction_stopped == pytest.approx(2.0 / 3.0)


def test_target_stopping_analysis_rejects_unsafe_premature_stops() -> None:
    result = analyze_g1_target_stopping(_outcomes(premature_domain="d1"))

    assert result.field.gate is not None
    assert result.field.gate.status == "G1_FIELD_STOPPING_NO_GO"
    assert result.field.premature_stop_rate == pytest.approx(1.0 / 6.0)


def _issued_curve(method: str, loss: float, *, budget: float):
    return build_engineering_curve(
        method=method,
        target_domain="d6",
        specimen_sha256=_sha("sample"),
        task=InspectionTask.CAI,
        grid_sha256=_sha("grid"),
        evaluator_sha256=_sha("evaluator"),
        warm_start_sha256=_sha("warm"),
        state_budgets=(0.0, budget),
        state_losses=(loss, loss),
        state_sha256=(_sha(f"{method}-zero"), _sha(f"{method}-final")),
    )


def test_stop_outcomes_are_materialized_from_exact_trajectory_curve_pairs() -> None:
    stop_record = _record()
    stop_trajectory = stop_record.trajectory
    proposed_trajectory = ClosedLoopTrajectory(
        target_domain=stop_trajectory.target_domain,
        specimen_sha256=stop_trajectory.specimen_sha256,
        task=stop_trajectory.task,
        model_sha256=stop_trajectory.model_sha256,
        stop_threshold=None,
        stopped=False,
        termination_reason="NO_LEGAL_ACTION",
        action_history=stop_trajectory.action_history,
        steps=stop_trajectory.steps[:-1],
        final_observation_sha256=stop_trajectory.final_observation_sha256,
        acquired_positions=stop_trajectory.acquired_positions,
        acquired_values=stop_trajectory.acquired_values,
        native_count=stop_trajectory.native_count,
        effective_budget=stop_trajectory.effective_budget,
    )
    proposed_record = G1TargetTrajectoryRecord(
        outer_target="d6",
        specimen_id="sample",
        specimen_sha256=_sha("sample"),
        task=InspectionTask.CAI,
        variant=TargetPolicyVariant.PROPOSED,
        task_token_mode=TaskTokenMode.CORRECT,
        surface_variant=SurfaceVariant.CORRECT_SURFACE,
        donor_surface_sha256=None,
        final_dependency_sha256=stop_record.final_dependency_sha256,
        action_selection_sha256=stop_record.action_selection_sha256,
        action_model_sha256=stop_record.action_model_sha256,
        stop_model_sha256=stop_record.stop_model_sha256,
        trajectory=proposed_trajectory,
    )
    bank_seal = _sha("bank-seal")
    learned = tuple(
        G1TargetCurveRecord(
            outer_target="d6",
            specimen_id="sample",
            specimen_sha256=_sha("sample"),
            task=InspectionTask.CAI,
            variant=record.variant,
            target_record_sha256=record.state_sha256,
            trajectory_sha256=record.trajectory.state_sha256,
            bank_seal_sha256=bank_seal,
            trajectory_seal_sha256=_sha(f"seal-{record.variant.value}"),
            truth_view_sha256=_sha("truth"),
            curve=_issued_curve(
                record.variant.value,
                0.5,
                budget=record.trajectory.effective_budget,
            ),
        )
        for record in (proposed_record, stop_record)
    )
    reference_curve = _issued_curve(
        "ZERO_UNIFORM",
        1.0,
        budget=stop_record.trajectory.effective_budget,
    )
    reference = G1TargetReferenceCurveRecord(
        outer_target="d6",
        specimen_id="sample",
        specimen_sha256=_sha("sample"),
        task=InspectionTask.CAI,
        method="ZERO_UNIFORM",
        final_dependency_sha256=stop_record.final_dependency_sha256,
        prior_sha256=_sha("prior"),
        assessor_sha256=_sha("assessor"),
        bank_seal_sha256=bank_seal,
        authorization_trajectory_sha256=stop_record.trajectory.state_sha256,
        truth_view_sha256=_sha("truth"),
        action_history_sha256=_sha("reference-actions"),
        curve=reference_curve,
    )
    selection = OuterFixedBridgeSelection(
        outer_target="d6",
        task=InspectionTask.CAI,
        method="ZERO_UNIFORM",
        source_domains=("d1", "d2", "d3", "d4", "d5"),
        equal_domain_auebc=reference_curve.auebc,
        domain_auebc=tuple(
            (domain, reference_curve.auebc)
            for domain in ("d1", "d2", "d3", "d4", "d5")
        ),
        evidence_sha256=_sha("selection"),
    )

    outcomes = materialize_g1_target_stop_outcomes(
        (proposed_record, stop_record),
        learned,
        (reference,),
        (selection,),
    )

    assert len(outcomes) == 1
    assert outcomes[0].stop_authorized
    assert outcomes[0].stopped
    assert outcomes[0].task_loss_ratio == pytest.approx(0.5)
    assert not outcomes[0].premature_stop
