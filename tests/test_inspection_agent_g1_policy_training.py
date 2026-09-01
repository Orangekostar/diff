from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionDecision, InspectionTask
from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.inspection_agent_g1.contracts import (
    CAIContextMode,
    G1PolicyState,
    TaskTokenMode,
)
from cmc_bbdm.inspection_agent_g1.policy_training import (
    G1PolicyTrainingError,
    G1PolicyTrainingExample,
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
    equal_policy_training_weights,
    fit_inner_observable_policy,
    fit_policy_normalizer,
    rebind_training_example_modes,
)
from cmc_bbdm.inspection_agent_g1.teacher import (
    PrivilegedTeacherLabel,
    TeacherCandidateRecord,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _state(task: InspectionTask, identity: str, value: float) -> G1PolicyState:
    global_scalars = np.full(17, value, dtype=np.float64)
    global_scalars[12:14] = (value, 1.0)
    mask = np.zeros(192, dtype=np.bool_)
    mask[:2] = True
    return G1PolicyState(
        task=task,
        task_token_mode=TaskTokenMode.CORRECT,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        observation_sha256=_sha(f"observation-{identity}"),
        reconstruction_sha256=_sha(f"reconstruction-{identity}"),
        surface_hypothesis_sha256=_sha(f"surface-{identity}"),
        grid_sha256=_sha("grid"),
        reconstruction_embedding=np.full(512, value, dtype=np.float64),
        global_scalars=global_scalars,
        task_token=np.asarray(
            (1.0, 0.0) if task is InspectionTask.FIELD else (0.0, 1.0),
            dtype=np.float64,
        ),
        cell_features=np.full((64, 18), value, dtype=np.float64),
        candidate_features=np.full((192, 12), value, dtype=np.float64),
        legal_action_mask=mask,
    )


def _label(state: G1PolicyState) -> PrivilegedTeacherLabel:
    selected = 0 if state.task is InspectionTask.FIELD else 1
    candidates = tuple(
        TeacherCandidateRecord(
            slot=slot,
            action=InspectionCellAction(slot, -1, 0),
            decision=InspectionDecision.BROADEN,
            exact_added_cost=1,
            raw_value=float(2 - slot),
            objective_value=float(2 - slot) if selected == 0 else float(slot + 1),
            task_loss_after=float(slot + 1),
            candidate_state_sha256=_sha(f"candidate-{state.state_sha256}-{slot}"),
            selected=slot == selected,
        )
        for slot in range(2)
    )
    return PrivilegedTeacherLabel(
        task=state.task,
        authorization_sha256=_sha("authorization"),
        observation_sha256=state.observation_sha256,
        policy_state_sha256=state.state_sha256,
        selected_slot=selected,
        candidates=candidates,
    )


def _example(
    domain: str,
    specimen: str,
    task: InspectionTask,
    index: int,
    *,
    value: float,
) -> G1PolicyTrainingExample:
    state = _state(task, f"{domain}-{specimen}-{task.value}-{index}", value)
    return G1PolicyTrainingExample(
        outer_target="d6",
        source_domain=domain,
        specimen_sha256=_sha(specimen),
        task=task,
        dagger_iteration=0,
        policy_state=state,
        teacher_label=_label(state),
    )


def _roster(*, validation_value: float = 5.0) -> tuple[G1PolicyTrainingExample, ...]:
    rows: list[G1PolicyTrainingExample] = []
    for domain_index, domain in enumerate(("d1", "d2", "d3", "d4", "d5"), start=1):
        value = validation_value if domain == "d5" else float(domain_index)
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            rows.append(_example(domain, f"{domain}-s1", task, 0, value=value))
    return tuple(rows)


def test_policy_normalizer_is_fit_only_on_explicit_inner_training_domains() -> None:
    fit_domains = ("d1", "d2", "d3", "d4")
    first = fit_policy_normalizer(_roster(validation_value=5.0), fit_domains=fit_domains)
    sentinel = fit_policy_normalizer(
        _roster(validation_value=1.0e9),
        fit_domains=fit_domains,
    )
    assert first.fit_domains == fit_domains
    assert first.state_sha256 == sentinel.state_sha256
    np.testing.assert_array_equal(first.embedding_mean, sentinel.embedding_mean)
    assert "d5" not in first.fit_domains
    assert "d6" not in first.fit_domains


def test_policy_training_weights_equalize_specimens_tasks_and_states() -> None:
    rows = (
        _example("d1", "s1", InspectionTask.FIELD, 0, value=1.0),
        _example("d1", "s1", InspectionTask.FIELD, 1, value=2.0),
        _example("d1", "s1", InspectionTask.CAI, 0, value=3.0),
        _example("d2", "s2", InspectionTask.FIELD, 0, value=4.0),
        _example("d2", "s2", InspectionTask.CAI, 0, value=5.0),
        _example("d2", "s2", InspectionTask.CAI, 1, value=6.0),
        _example("d2", "s2", InspectionTask.CAI, 2, value=7.0),
    )
    weights = equal_policy_training_weights(rows)
    assert float(np.sum(weights)) == pytest.approx(1.0)
    specimen_task_mass: dict[tuple[str, InspectionTask], float] = {}
    for row, weight in zip(rows, weights, strict=True):
        key = (row.specimen_sha256, row.task)
        specimen_task_mass[key] = specimen_task_mass.get(key, 0.0) + float(weight)
    assert all(value == pytest.approx(0.25) for value in specimen_task_mass.values())


def test_inner_policy_fit_excludes_validation_and_outer_target_and_replays() -> None:
    hyperparameters = PolicyTrainingHyperparameters(
        model_name=PolicyModelName.SHARED_ACTION_MLP,
        route=TrainingRoute.HARD_BC,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=None,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=0,
    )
    first = fit_inner_observable_policy(
        _roster(),
        validation_domain="d5",
        hyperparameters=hyperparameters,
        max_epochs=2,
        patience=2,
        device="cpu",
    )
    second = fit_inner_observable_policy(
        _roster(),
        validation_domain="d5",
        hyperparameters=hyperparameters,
        max_epochs=2,
        patience=2,
        device="cpu",
    )
    assert first.audit.fit_domains == ("d1", "d2", "d3", "d4")
    assert first.audit.validation_domain == "d5"
    assert first.audit.outer_target == "d6"
    assert first.model_state_sha256 == second.model_state_sha256
    target_state = _state(InspectionTask.FIELD, "unseen-target", 42.0)
    scores = first(target_state)
    assert scores.policy_state_sha256 == target_state.state_sha256
    assert scores.model_sha256 == first.model_state_sha256
    assert np.all(np.isfinite(scores.action_logits[target_state.legal_action_mask]))
    assert np.all(np.isneginf(scores.action_logits[~target_state.legal_action_mask]))


def test_policy_training_rejects_an_outer_target_row() -> None:
    row = _example("d1", "s1", InspectionTask.FIELD, 0, value=1.0)
    with pytest.raises(G1PolicyTrainingError, match="outer target"):
        G1PolicyTrainingExample(
            outer_target="d1",
            source_domain="d1",
            specimen_sha256=row.specimen_sha256,
            task=row.task,
            dagger_iteration=0,
            policy_state=row.policy_state,
            teacher_label=row.teacher_label,
        )


def test_training_example_modes_rebind_only_observable_arrays_and_hashes() -> None:
    original = _example("d1", "s1", InspectionTask.FIELD, 0, value=3.0)
    rebound = rebind_training_example_modes(
        original,
        cai_context_mode=CAIContextMode.TASK_SPECIFIC_MASKED,
        task_token_mode=TaskTokenMode.NO_TASK,
    )
    assert rebound.policy_state.cai_context_mode is CAIContextMode.TASK_SPECIFIC_MASKED
    assert rebound.policy_state.task_token_mode is TaskTokenMode.NO_TASK
    np.testing.assert_array_equal(rebound.policy_state.global_scalars[12:14], (0.0, 0.0))
    np.testing.assert_array_equal(rebound.policy_state.task_token, (0.0, 0.0))
    np.testing.assert_array_equal(
        rebound.policy_state.reconstruction_embedding,
        original.policy_state.reconstruction_embedding,
    )
    assert rebound.policy_state.state_sha256 != original.policy_state.state_sha256
    assert rebound.teacher_label.policy_state_sha256 == rebound.policy_state.state_sha256
    assert tuple(
        candidate.objective_value for candidate in rebound.teacher_label.candidates
    ) == tuple(candidate.objective_value for candidate in original.teacher_label.candidates)
