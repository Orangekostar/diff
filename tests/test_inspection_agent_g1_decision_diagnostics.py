from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionDecision, InspectionTask
from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.inspection_agent_g1.contracts import (
    ACTION_SLOT_COUNT,
    CAIContextMode,
    G1PolicyState,
    TaskTokenMode,
)
from cmc_bbdm.inspection_agent_g1.decision_diagnostics import (
    G1DecisionDiagnosticError,
    materialize_g1_source_decision_diagnostics,
    read_g1_source_decision_diagnostic_bank,
    source_decision_diagnostic_bank_path,
    summarize_g1_source_decision_diagnostics,
    write_g1_source_decision_diagnostic_bank,
)
from cmc_bbdm.inspection_agent_g1.features import canonical_slot
from cmc_bbdm.inspection_agent_g1.policy_training import G1PolicyTrainingExample
from cmc_bbdm.inspection_agent_g1.rollout import ObservablePolicyScores
from cmc_bbdm.inspection_agent_g1.teacher import (
    PrivilegedTeacherLabel,
    TeacherCandidateRecord,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _example(source: str, specimen: str, task: InspectionTask) -> G1PolicyTrainingExample:
    actions = (
        InspectionCellAction(0, -1, 0),
        InspectionCellAction(1, -1, 0),
        InspectionCellAction(0, 0, 1),
    )
    slots = tuple(canonical_slot(action) for action in actions)
    legal = np.zeros(ACTION_SLOT_COUNT, dtype=np.bool_)
    legal[list(slots)] = True
    global_scalars = np.zeros(17, dtype=np.float64)
    global_scalars[12:14] = (0.5, 1.0)
    observation_sha = _sha(f"observation-{source}-{specimen}-{task.value}")
    state = G1PolicyState(
        task=task,
        task_token_mode=TaskTokenMode.CORRECT,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        observation_sha256=observation_sha,
        reconstruction_sha256=_sha(f"reconstruction-{source}-{specimen}"),
        surface_hypothesis_sha256=_sha(f"surface-{source}-{specimen}"),
        grid_sha256=_sha("grid"),
        reconstruction_embedding=np.zeros(512, dtype=np.float64),
        global_scalars=global_scalars,
        task_token=np.asarray(
            (1.0, 0.0) if task is InspectionTask.FIELD else (0.0, 1.0),
            dtype=np.float64,
        ),
        cell_features=np.zeros((64, 18), dtype=np.float64),
        candidate_features=np.zeros((ACTION_SLOT_COUNT, 12), dtype=np.float64),
        legal_action_mask=legal,
    )
    decisions = (
        InspectionDecision.FOCUS,
        InspectionDecision.BROADEN,
        InspectionDecision.REFINE,
    )
    candidates = tuple(
        TeacherCandidateRecord(
            slot=slot,
            action=action,
            decision=decision,
            exact_added_cost=index + 1,
            raw_value=float(3 - index),
            objective_value=float(3 - index),
            task_loss_after=float(index),
            candidate_state_sha256=_sha(
                f"candidate-{source}-{specimen}-{task.value}-{slot}"
            ),
            selected=index == 0,
        )
        for index, (slot, action, decision) in enumerate(
            zip(slots, actions, decisions, strict=True)
        )
    )
    label = PrivilegedTeacherLabel(
        task=task,
        authorization_sha256=_sha(f"authorization-{source}"),
        observation_sha256=observation_sha,
        policy_state_sha256=state.state_sha256,
        selected_slot=slots[0],
        candidates=candidates,
    )
    return G1PolicyTrainingExample(
        outer_target="d6",
        source_domain=source,
        specimen_sha256=_sha(specimen),
        task=task,
        dagger_iteration=0,
        policy_state=state,
        teacher_label=label,
    )


class _Actor:
    model_state_sha256 = _sha("action-model")

    def score_batch(self, states):
        rows = []
        for state in states:
            legal = np.flatnonzero(state.legal_action_mask)
            logits = np.full(ACTION_SLOT_COUNT, -np.inf, dtype=np.float64)
            logits[legal] = (2.0, 3.0, 1.0)
            rows.append(
                ObservablePolicyScores(
                    policy_state_sha256=state.state_sha256,
                    model_sha256=self.model_state_sha256,
                    action_logits=logits,
                    stop_probability=0.0,
                )
            )
        return tuple(rows)


def test_source_decision_diagnostics_use_full_legal_utility_distribution() -> None:
    records = materialize_g1_source_decision_diagnostics(
        tuple(
            _example(source, f"{source}-s1", task)
            for source in ("d1", "d2", "d3", "d4", "d5")
            for task in (InspectionTask.FIELD, InspectionTask.CAI)
        ),
        _Actor(),
    )

    assert len(records) == 10
    assert all(row.high_level_decision_accuracy == 0.0 for row in records)
    assert all(row.primitive_top1_match == 0.0 for row in records)
    assert all(row.top5_utility_recall == 1.0 for row in records)
    assert all(row.expected_teacher_regret > 0.0 for row in records)
    assert all(0.0 < row.candidate_utility_ndcg < 1.0 for row in records)
    assert all(row.teacher_decision is InspectionDecision.FOCUS for row in records)
    assert all(row.predicted_decision is InspectionDecision.BROADEN for row in records)
    summary = summarize_g1_source_decision_diagnostics(records)
    assert summary.high_level_decision_accuracy == 0.0
    assert summary.primitive_top1_match == 0.0
    assert summary.top5_utility_recall == 1.0
    assert summary.predicted_decision_proportions == (
        ("FOCUS", 0.0),
        ("BROADEN", 1.0),
        ("REFINE", 0.0),
    )
    assert summary.transition_proportions == (
        ("FOCUS", "BROADEN", 1.0),
    )


def test_source_decision_diagnostic_bank_replays_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    rows = materialize_g1_source_decision_diagnostics(
        tuple(
            _example(source, f"{source}-s1", InspectionTask.FIELD)
            for source in ("d1", "d2", "d3", "d4", "d5")
        ),
        _Actor(),
    )
    path = source_decision_diagnostic_bank_path(tmp_path, "d6")
    identity = write_g1_source_decision_diagnostic_bank(path, rows)
    replay_identity, replay = read_g1_source_decision_diagnostic_bank(path)

    assert replay_identity == identity
    assert replay == rows
    assert identity.outer_target == "d6"
    assert identity.source_domains == ("d1", "d2", "d3", "d4", "d5")

    manifest_path = path.with_suffix(".parquet.manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    payload["action_model_sha256"] = _sha("tampered")
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(G1DecisionDiagnosticError, match="manifest"):
        read_g1_source_decision_diagnostic_bank(path)


def test_source_decision_diagnostic_path_rejects_unsafe_identity(tmp_path: Path) -> None:
    with pytest.raises(G1DecisionDiagnosticError, match="path"):
        source_decision_diagnostic_bank_path(tmp_path, "../d6")
