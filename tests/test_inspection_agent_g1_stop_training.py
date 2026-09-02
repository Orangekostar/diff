from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionDecision, InspectionTask
from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.inspection_agent_g1.contracts import (
    CAIContextMode,
    G1PolicyState,
    TaskTokenMode,
)
from cmc_bbdm.inspection_agent_g1.policy_training import (
    G1PolicyTrainingExample,
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
    fit_inner_observable_policy,
)
from cmc_bbdm.inspection_agent_g1.stop_training import (
    G1StopTrainingExample,
    equal_stop_training_weights,
    fit_inner_observable_stop_head,
)
from cmc_bbdm.inspection_agent_g1.stopping_policy import SourceStopLabel
from cmc_bbdm.inspection_agent_g1.teacher import (
    PrivilegedTeacherLabel,
    TeacherCandidateRecord,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _state(task: InspectionTask, identity: str, value: float) -> G1PolicyState:
    scalars = np.full(17, value, dtype=np.float64)
    scalars[12:14] = (value, 1.0)
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
        global_scalars=scalars,
        task_token=np.asarray(
            (1.0, 0.0) if task is InspectionTask.FIELD else (0.0, 1.0),
            dtype=np.float64,
        ),
        cell_features=np.full((64, 18), value, dtype=np.float64),
        candidate_features=np.full((192, 12), value, dtype=np.float64),
        legal_action_mask=mask,
    )


def _action_example(
    domain: str,
    specimen: str,
    task: InspectionTask,
    value: float,
) -> G1PolicyTrainingExample:
    state = _state(task, f"{domain}-{specimen}-{task.value}", value)
    candidates = tuple(
        TeacherCandidateRecord(
            slot=slot,
            action=InspectionCellAction(slot, -1, 0),
            decision=InspectionDecision.BROADEN,
            exact_added_cost=1,
            raw_value=float(2 - slot),
            objective_value=float(2 - slot),
            task_loss_after=float(slot + 1),
            candidate_state_sha256=_sha(f"candidate-{state.state_sha256}-{slot}"),
            selected=slot == 0,
        )
        for slot in range(2)
    )
    label = PrivilegedTeacherLabel(
        task=task,
        authorization_sha256=_sha("authorization"),
        observation_sha256=state.observation_sha256,
        policy_state_sha256=state.state_sha256,
        selected_slot=0,
        candidates=candidates,
    )
    return G1PolicyTrainingExample(
        outer_target="d6",
        source_domain=domain,
        specimen_sha256=_sha(specimen),
        task=task,
        dagger_iteration=0,
        policy_state=state,
        teacher_label=label,
    )


def _roster() -> tuple[
    tuple[G1PolicyTrainingExample, ...], tuple[G1StopTrainingExample, ...]
]:
    actions = []
    stops = []
    for index, domain in enumerate(("d1", "d2", "d3", "d4", "d5"), start=1):
        specimen = f"{domain}-s1"
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            action = _action_example(domain, specimen, task, float(index))
            current = 1.0 if index <= 3 else 2.0
            label = SourceStopLabel(
                authorization_sha256=_sha(f"authorization-{domain}"),
                fixed_reference_sha256=_sha(f"reference-{domain}-{task.value}"),
                source_domain=domain,
                specimen_sha256=action.specimen_sha256,
                task=task,
                policy_state_sha256=action.policy_state.state_sha256,
                reference_method="RANDOM",
                current_true_loss=current,
                reference_true_loss=1.0,
                tolerance=0.05,
                is_sufficient=current <= 1.05,
            )
            actions.append(action)
            stops.append(
                G1StopTrainingExample(
                    outer_target="d6",
                    source_domain=domain,
                    specimen_sha256=action.specimen_sha256,
                    task=task,
                    source_policy_state_sha256=action.policy_state.state_sha256,
                    action_example_sha256=action.state_sha256,
                    policy_state=action.policy_state,
                    label=label,
                )
            )
    return tuple(actions), tuple(stops)


def _hyperparameters() -> PolicyTrainingHyperparameters:
    return PolicyTrainingHyperparameters(
        model_name=PolicyModelName.SHARED_ACTION_MLP,
        route=TrainingRoute.HARD_BC,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=None,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=0,
    )


def test_stop_weights_equalize_physical_specimens_and_tasks() -> None:
    _actions, stops = _roster()
    weights = equal_stop_training_weights(stops)
    assert np.sum(weights) == 1.0
    assert np.unique(weights).tolist() == [0.1]


def test_stop_head_fit_is_deterministic_source_only_and_action_invariant() -> None:
    actions, stops = _roster()
    action_policy = fit_inner_observable_policy(
        actions,
        validation_domain="d5",
        hyperparameters=_hyperparameters(),
        max_epochs=2,
        patience=2,
        device="cpu",
    )
    first = fit_inner_observable_stop_head(
        action_policy,
        stops,
        validation_domain="d5",
        max_epochs=2,
        patience=2,
        device="cpu",
    )
    second = fit_inner_observable_stop_head(
        action_policy,
        stops,
        validation_domain="d5",
        max_epochs=2,
        patience=2,
        device="cpu",
    )

    state = stops[0].policy_state
    before = action_policy(state)
    after = first(state)
    np.testing.assert_array_equal(after.action_logits, before.action_logits)
    assert first.action_backbone_sha256 == second.action_backbone_sha256
    assert first.model_state_sha256 == second.model_state_sha256
    assert first.audit.fit_domains == ("d1", "d2", "d3", "d4")
    assert first.audit.validation_domain == "d5"
    assert "d6" not in first.audit.fit_domains
