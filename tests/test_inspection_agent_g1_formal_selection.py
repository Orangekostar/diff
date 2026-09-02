from __future__ import annotations

import json

import pytest
from test_inspection_agent_g1_source_bridge import DOMAINS, _record, _sha

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.formal import FIXED_BASELINE_METHODS
from cmc_bbdm.inspection_agent_g1.formal_selection import (
    G1FormalSelectionError,
    freeze_g1_outer_formal_selection,
    read_g1_outer_formal_selection,
)
from cmc_bbdm.inspection_agent_g1.policy_training import (
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
)
from cmc_bbdm.inspection_agent_g1.stopping_policy import StopThresholdSelection


def _bridges():
    return tuple(
        _record(
            source,
            method,
            1.0 + method_index,
            task=task,
        )
        for source in DOMAINS[:-1]
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
        for method_index, method in enumerate(FIXED_BASELINE_METHODS)
    )


def _thresholds():
    return tuple(
        StopThresholdSelection(
            outer_target="d6",
            task=task,
            status="STOP_NOT_AUTHORIZED",
            threshold=None,
            candidates=(),
            trajectory_sha256=(),
            state_sha256=_sha(f"threshold-{task.value}"),
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    )


def _hyperparameters() -> PolicyTrainingHyperparameters:
    return PolicyTrainingHyperparameters(
        model_name=PolicyModelName.STRUCTURED_INSPECTION_POLICY,
        route=TrainingRoute.HARD_BC,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=None,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=0,
    )


def _aawr_hyperparameters() -> PolicyTrainingHyperparameters:
    return PolicyTrainingHyperparameters(
        model_name=PolicyModelName.STRUCTURED_INSPECTION_POLICY,
        route=TrainingRoute.PRIVILEGED_AAWR,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=0.5,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=1,
        aawr_expectile=0.8,
        aawr_beta=3.0,
        aawr_authorized_tasks=(InspectionTask.FIELD, InspectionTask.CAI),
        aawr_authorization_sha256=_sha("aawr-authorization"),
        base_hyperparameters_sha256=_sha("base-hyperparameters"),
    )


def test_outer_formal_selection_freezes_only_source_evidence(tmp_path) -> None:
    path = tmp_path / "selection.json"

    first = freeze_g1_outer_formal_selection(
        _bridges(),
        _thresholds(),
        outer_target="d6",
        action_hyperparameters=_hyperparameters(),
        action_selection_sha256=_sha("action-selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=_sha("stop-model"),
        decision_diagnostic_manifest_sha256=_sha("decision-diagnostics"),
        path=path,
    )
    replay = freeze_g1_outer_formal_selection(
        _bridges(),
        _thresholds(),
        outer_target="d6",
        action_hyperparameters=_hyperparameters(),
        action_selection_sha256=_sha("action-selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=_sha("stop-model"),
        decision_diagnostic_manifest_sha256=_sha("decision-diagnostics"),
        path=path,
    )

    assert replay == first == read_g1_outer_formal_selection(path)
    assert tuple(row.method for row in first.fixed_selections) == (
        "RANDOM",
        "RANDOM",
    )
    assert first.target_outcomes_opened is False
    assert first.action_hyperparameters == _hyperparameters()
    assert first.decision_diagnostic_manifest_sha256 == _sha(
        "decision-diagnostics"
    )


def test_outer_formal_selection_rejects_tampering(tmp_path) -> None:
    path = tmp_path / "selection.json"
    freeze_g1_outer_formal_selection(
        _bridges(),
        _thresholds(),
        outer_target="d6",
        action_hyperparameters=_hyperparameters(),
        action_selection_sha256=_sha("action-selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=_sha("stop-model"),
        decision_diagnostic_manifest_sha256=_sha("decision-diagnostics"),
        path=path,
    )
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["target_outcomes_opened"] = True
    path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(G1FormalSelectionError, match="changed|invalid"):
        read_g1_outer_formal_selection(path)


def test_outer_formal_selection_rejects_training_route_tampering(tmp_path) -> None:
    path = tmp_path / "selection.json"
    freeze_g1_outer_formal_selection(
        _bridges(),
        _thresholds(),
        outer_target="d6",
        action_hyperparameters=_hyperparameters(),
        action_selection_sha256=_sha("action-selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=_sha("stop-model"),
        decision_diagnostic_manifest_sha256=_sha("decision-diagnostics"),
        path=path,
    )
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["action_hyperparameters"]["route"] = "SOFT_UTILITY_DISTILL"
    path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(G1FormalSelectionError, match="changed|invalid"):
        read_g1_outer_formal_selection(path)


def test_outer_formal_selection_round_trips_conditional_aawr_route(tmp_path) -> None:
    path = tmp_path / "selection.json"
    expected = _aawr_hyperparameters()

    frozen = freeze_g1_outer_formal_selection(
        _bridges(),
        _thresholds(),
        outer_target="d6",
        action_hyperparameters=expected,
        action_selection_sha256=_sha("action-selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=_sha("stop-model"),
        decision_diagnostic_manifest_sha256=_sha("decision-diagnostics"),
        path=path,
    )

    assert frozen.action_hyperparameters == expected
    payload = json.loads(path.read_text(encoding="ascii"))
    assert payload["action_hyperparameters"]["route"] == "PRIVILEGED_AAWR"
    assert payload["action_hyperparameters"]["aawr_authorized_tasks"] == [
        "FIELD",
        "CAI",
    ]
