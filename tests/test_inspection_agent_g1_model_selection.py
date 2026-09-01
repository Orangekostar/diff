from __future__ import annotations

import hashlib

import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.policy_selection import (
    G1PolicySelectionError,
    InnerPolicyEngineeringMetric,
    PolicyCandidateEvaluation,
    select_outer_policy,
)
from cmc_bbdm.inspection_agent_g1.policy_training import (
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _hyperparameters(*, structured: bool) -> PolicyTrainingHyperparameters:
    return PolicyTrainingHyperparameters(
        model_name=(
            PolicyModelName.STRUCTURED_INSPECTION_POLICY
            if structured
            else PolicyModelName.SHARED_ACTION_MLP
        ),
        route=(
            TrainingRoute.SOFT_UTILITY_DISTILL
            if structured
            else TrainingRoute.HARD_BC
        ),
        cai_context_mode=CAIContextMode.TASK_SPECIFIC_MASKED,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=0.5 if structured else None,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=1 if structured else 0,
    )


def _candidate(
    hyperparameters: PolicyTrainingHyperparameters,
    *,
    relative_auebc: float,
    selected_epochs: tuple[int, ...] = (5, 7, 9, 11, 13),
) -> PolicyCandidateEvaluation:
    rows: list[InnerPolicyEngineeringMetric] = []
    for domain_index, domain in enumerate(("d1", "d2", "d3", "d4", "d5")):
        for task, fixed in (
            (InspectionTask.FIELD, 0.001),
            (InspectionTask.CAI, 0.020),
        ):
            rows.append(
                InnerPolicyEngineeringMetric(
                    outer_target="d6",
                    validation_domain=domain,
                    task=task,
                    hyperparameters_sha256=hyperparameters.state_sha256,
                    model_state_sha256=_sha(
                        f"{hyperparameters.state_sha256}-{domain}"
                    ),
                    learned_auebc=fixed * relative_auebc,
                    fixed_auebc=fixed,
                    oracle_auebc=fixed * 0.5,
                    selected_epoch=selected_epochs[domain_index],
                )
            )
    return PolicyCandidateEvaluation(
        hyperparameters=hyperparameters,
        inner_metrics=tuple(rows),
    )


def test_outer_selection_uses_source_only_scale_free_joint_engineering_metric() -> None:
    simple = _candidate(_hyperparameters(structured=False), relative_auebc=0.90)
    structured = _candidate(_hyperparameters(structured=True), relative_auebc=0.80)
    selection = select_outer_policy((simple, structured))
    assert selection.outer_target == "d6"
    assert selection.source_validation_domains == ("d1", "d2", "d3", "d4", "d5")
    assert selection.selected_hyperparameters_sha256 == (
        structured.hyperparameters.state_sha256
    )
    assert selection.final_refit_epochs == 9
    assert selection.target_outcomes_opened is False


def test_outer_selection_tie_prefers_simpler_model_and_fewer_stages() -> None:
    simple = _candidate(_hyperparameters(structured=False), relative_auebc=0.80)
    structured = _candidate(_hyperparameters(structured=True), relative_auebc=0.80)
    selection = select_outer_policy((structured, simple))
    assert selection.selected_hyperparameters_sha256 == simple.hyperparameters.state_sha256


def test_inner_selection_metric_rejects_the_outer_target() -> None:
    hyperparameters = _hyperparameters(structured=False)
    with pytest.raises(G1PolicySelectionError, match="outer target"):
        InnerPolicyEngineeringMetric(
            outer_target="d6",
            validation_domain="d6",
            task=InspectionTask.FIELD,
            hyperparameters_sha256=hyperparameters.state_sha256,
            model_state_sha256=_sha("model"),
            learned_auebc=0.8,
            fixed_auebc=1.0,
            oracle_auebc=0.5,
            selected_epoch=5,
        )


def test_outer_selection_rejects_inconsistent_fixed_oracle_bridges() -> None:
    simple = _candidate(_hyperparameters(structured=False), relative_auebc=0.90)
    structured = _candidate(_hyperparameters(structured=True), relative_auebc=0.80)
    first = structured.inner_metrics[0]
    changed = InnerPolicyEngineeringMetric(
        outer_target=first.outer_target,
        validation_domain=first.validation_domain,
        task=first.task,
        hyperparameters_sha256=first.hyperparameters_sha256,
        model_state_sha256=first.model_state_sha256,
        learned_auebc=first.learned_auebc,
        fixed_auebc=first.fixed_auebc * 2.0,
        oracle_auebc=first.oracle_auebc,
        selected_epoch=first.selected_epoch,
    )
    inconsistent = PolicyCandidateEvaluation(
        hyperparameters=structured.hyperparameters,
        inner_metrics=(changed, *structured.inner_metrics[1:]),
    )
    with pytest.raises(G1PolicySelectionError, match="bridge"):
        select_outer_policy((simple, inconsistent))
