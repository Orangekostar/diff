from __future__ import annotations

import hashlib
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionDecision, InspectionTask
from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.inspection_agent_g1.aawr_selection_execution import (
    materialize_g1_aawr_training_records,
)
from cmc_bbdm.inspection_agent_g1.contracts import (
    CAIContextMode,
    G1PolicyState,
    TaskTokenMode,
)
from cmc_bbdm.inspection_agent_g1.g1 import G1Protocol, G1Runtime
from cmc_bbdm.inspection_agent_g1.policy_training import (
    G1PolicyTrainingExample,
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
    fit_inner_observable_policy,
)
from cmc_bbdm.inspection_agent_g1.privileged_awr import (
    AAWRSourceEvidence,
    G1AAWRTrainingRecord,
    PrivilegedAWRError,
    aawr_policy_hyperparameters,
    authorize_conditional_aawr,
    fit_inner_aawr_policy,
    read_aawr_fit_evidence,
    write_aawr_fit_evidence,
)
from cmc_bbdm.inspection_agent_g1.teacher import (
    PrivilegedTeacherLabel,
    TeacherCandidateRecord,
)
from cmc_bbdm.inspection_agent_g1.teacher_bank import G1TeacherBankRecord


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _example(
    domain: str,
    task: InspectionTask,
    value: float,
    *,
    suffix: str = "base",
    dagger_iteration: int = 0,
) -> G1PolicyTrainingExample:
    identity = f"{domain}-{task.value}-{suffix}"
    mask = np.zeros(192, dtype=np.bool_)
    mask[:2] = True
    state = G1PolicyState(
        task=task,
        task_token_mode=TaskTokenMode.CORRECT,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        observation_sha256=_sha(f"observation-{identity}"),
        reconstruction_sha256=_sha(f"reconstruction-{identity}"),
        surface_hypothesis_sha256=_sha(f"surface-{identity}"),
        grid_sha256=_sha("grid"),
        reconstruction_embedding=np.full(512, value, dtype=np.float64),
        global_scalars=np.asarray(
            [value] * 12 + [value, 1.0] + [value] * 3,
            dtype=np.float64,
        ),
        task_token=np.asarray(
            (1.0, 0.0) if task is InspectionTask.FIELD else (0.0, 1.0),
            dtype=np.float64,
        ),
        cell_features=np.full((64, 18), value, dtype=np.float64),
        candidate_features=np.full((192, 12), value, dtype=np.float64),
        legal_action_mask=mask,
    )
    candidates = tuple(
        TeacherCandidateRecord(
            slot=slot,
            action=InspectionCellAction(slot, -1, 0),
            decision=InspectionDecision.BROADEN,
            exact_added_cost=slot + 1,
            raw_value=float(2 - slot),
            objective_value=float(2 - slot) / float(slot + 1),
            task_loss_after=float(slot + 1),
            candidate_state_sha256=_sha(f"candidate-{identity}-{slot}"),
            selected=slot == 0,
        )
        for slot in range(2)
    )
    label = PrivilegedTeacherLabel(
        task=task,
        authorization_sha256=_sha("teacher-authorization"),
        observation_sha256=state.observation_sha256,
        policy_state_sha256=state.state_sha256,
        selected_slot=0,
        candidates=candidates,
    )
    return G1PolicyTrainingExample(
        outer_target="d6",
        source_domain=domain,
        specimen_sha256=_sha(f"specimen-{domain}"),
        task=task,
        dagger_iteration=dagger_iteration,
        policy_state=state,
        teacher_label=label,
    )


def _base_hyperparameters() -> PolicyTrainingHyperparameters:
    return PolicyTrainingHyperparameters(
        model_name=PolicyModelName.SHARED_ACTION_MLP,
        route=TrainingRoute.SOFT_UTILITY_DISTILL,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=0.5,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=1,
    )


def _roster() -> tuple[G1PolicyTrainingExample, ...]:
    return tuple(
        _example(domain, task, float(index))
        for index, domain in enumerate(("d1", "d2", "d3", "d4", "d5"), start=1)
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    )


def test_aawr_hyperparameters_bind_base_authorization_and_registered_grid() -> None:
    base = _base_hyperparameters()
    value = aawr_policy_hyperparameters(
        base,
        authorization_sha256=_sha("authorization"),
        authorized_tasks=(InspectionTask.CAI,),
        expectile=0.8,
        beta=3.0,
    )
    assert value.route is TrainingRoute.PRIVILEGED_AAWR
    assert value.base_hyperparameters_sha256 == base.state_sha256
    assert value.aawr_authorized_tasks == (InspectionTask.CAI,)
    assert value.aawr_expectile == 0.8
    assert value.aawr_beta == 3.0

    invalid_base = PolicyTrainingHyperparameters(
        model_name=PolicyModelName.SHARED_ACTION_MLP,
        route=TrainingRoute.HARD_BC,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=None,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=0,
    )
    with pytest.raises(PrivilegedAWRError, match="soft-distillation.*DAgger"):
        aawr_policy_hyperparameters(
            invalid_base,
            authorization_sha256=_sha("authorization"),
            authorized_tasks=(InspectionTask.CAI,),
            expectile=0.8,
            beta=3.0,
        )


def test_aawr_training_is_source_only_deterministic_and_actor_visible_only(
    tmp_path,
) -> None:
    examples = _roster()
    base_hyperparameters = _base_hyperparameters()
    base_actor = fit_inner_observable_policy(
        examples,
        validation_domain="d5",
        hyperparameters=base_hyperparameters,
        max_epochs=1,
        patience=1,
        device="cpu",
    )
    authorization_sha256 = _sha("authorization")
    hyperparameters = aawr_policy_hyperparameters(
        base_hyperparameters,
        authorization_sha256=authorization_sha256,
        authorized_tasks=(InspectionTask.CAI,),
        expectile=0.7,
        beta=1.0,
    )
    records = tuple(
        G1AAWRTrainingRecord(
            example=example,
            state_source="DAGGER_ACTOR_VISITED",
            source_record_sha256=_sha(f"record-{example.state_sha256}"),
            full_scan_embedding=np.full(512, 0.1, dtype=np.float64),
            true_cai=0.2 if example.task is InspectionTask.CAI else None,
        )
        for example in examples
        if example.task is InspectionTask.CAI
    )
    first, first_evidence = fit_inner_aawr_policy(
        records,
        base_actor=base_actor,
        hyperparameters=hyperparameters,
        validation_domain="d5",
        max_epochs=1,
        patience=1,
        device="cpu",
    )
    second, second_evidence = fit_inner_aawr_policy(
        records,
        base_actor=base_actor,
        hyperparameters=hyperparameters,
        validation_domain="d5",
        max_epochs=1,
        patience=1,
        device="cpu",
    )

    assert first.audit.fit_domains == ("d1", "d2", "d3", "d4")
    assert first.audit.validation_domain == "d5"
    assert first.hyperparameters == hyperparameters
    assert first.model_state_sha256 == second.model_state_sha256
    assert first_evidence.state_sha256 == second_evidence.state_sha256
    assert first_evidence.target_outcomes_opened is False
    evidence_path = write_aawr_fit_evidence(
        tmp_path / "fit.json",
        first_evidence,
    )
    assert read_aawr_fit_evidence(evidence_path) == first_evidence
    scores = first(examples[-1].policy_state)
    assert scores.model_sha256 == first.model_state_sha256
    assert np.all(np.isfinite(scores.action_logits[:2]))


def test_aawr_transition_materialization_never_opens_outer_target(
    monkeypatch,
) -> None:
    domains = ("d1", "d2", "d3", "d4", "d5", "d6")
    source_domains = domains[:-1]
    records = tuple(
        G1TeacherBankRecord(
            example=_example(
                domain,
                InspectionTask.CAI,
                float(index),
                suffix=state_source,
                dagger_iteration=(
                    1 if state_source == "DAGGER_ACTOR_VISITED" else 0
                ),
            ),
            fit_domains=tuple(value for value in source_domains if value != domain),
            state_source=state_source,
            source_state_sha256=_sha(f"state-{domain}-{state_source}"),
            prior_sha256=_sha(f"prior-{domain}"),
            assessor_sha256=_sha(f"assessor-{domain}"),
        )
        for index, domain in enumerate(source_domains, start=1)
        for state_source in (
            "UNIFORM_CONTINUE",
            "ORACLE_CHECKPOINT",
            "DAGGER_ACTOR_VISITED",
        )
    )
    specimen_ids = tuple(f"specimen-{domain}" for domain in domains)
    opened = []

    def source_teacher_view(specimen_id: str):
        opened.append(specimen_id)
        if specimen_id == "specimen-d6":
            raise AssertionError("outer target truth was opened")
        return SimpleNamespace(
            full_scan=np.zeros((8, 8, 3), dtype=np.uint8),
            true_cai=0.2,
        )

    runtime = object.__new__(G1Runtime)
    mavis = SimpleNamespace(
        dataset_ids=domains,
        specimen_ids=specimen_ids,
        source_image_sha256=tuple(_sha(f"source-{domain}") for domain in domains),
        decoded_image_sha256=tuple(_sha(f"decoded-{domain}") for domain in domains),
        source_teacher_view=source_teacher_view,
    )
    object.__setattr__(runtime, "mavis", mavis)
    object.__setattr__(
        runtime,
        "_identity_index",
        MappingProxyType(
            {
                (domain, specimen): index
                for index, (domain, specimen) in enumerate(
                    zip(domains, specimen_ids, strict=True)
                )
            }
        ),
    )
    object.__setattr__(
        runtime,
        "surfaces",
        MappingProxyType(
            {
                (domain, specimen): SimpleNamespace(
                    surface_sha256=_sha(f"surface-{domain}")
                )
                for domain, specimen in zip(domains, specimen_ids, strict=True)
            }
        ),
    )
    protocol = object.__new__(G1Protocol)
    object.__setattr__(protocol, "domain_order", domains)
    object.__setattr__(protocol, "encoder_batch_size", 8)
    monkeypatch.setattr(
        G1Runtime,
        "specimen_sha256",
        lambda _self, _dataset_id, specimen_id: _sha(specimen_id),
    )
    authorization = authorize_conditional_aawr(
        (
            AAWRSourceEvidence(
                outer_target="d6",
                task=InspectionTask.CAI,
                source_domains=source_domains,
                fixed_auebc=(1.0,) * 5,
                policy_auebc=(0.9,) * 5,
                oracle_auebc=(0.0,) * 5,
            ),
        )
    )
    encoder = SimpleNamespace(
        encode=lambda images: np.zeros((len(tuple(images)), 512), dtype=np.float64)
    )

    output = materialize_g1_aawr_training_records(
        runtime,
        protocol,
        records,
        encoder=encoder,
        base_hyperparameters=_base_hyperparameters(),
        authorization=authorization,
    )

    assert len(output) == 15
    assert set(opened) == set(specimen_ids[:-1])
    assert "specimen-d6" not in opened
    assert {row.example.source_domain for row in output} == set(source_domains)
