from __future__ import annotations

import hashlib

import numpy as np
import pytest
from mavis_test_support import synthetic_authority

from cmc_bbdm.mavis import aggregation_execution
from cmc_bbdm.mavis.aggregation import (
    AggregationModel,
    MAVISAggregationError,
    build_on_policy_group,
    run_source_only_aggregation,
)
from cmc_bbdm.mavis.aggregation_execution import checkpoint_decision_states
from cmc_bbdm.mavis.dynamic_data import DynamicStateGroup
from cmc_bbdm.mavis.dynamic_voi import CandidateDescriptor
from cmc_bbdm.mavis.reveal import reveal_uniform_scout
from cmc_bbdm.mavis.rollout import rollout_scout_and_focus_curve
from cmc_bbdm.mavis.state_candidates import build_source_candidate_batch
from cmc_bbdm.mavis.teacher import FoldStateLabels, TeacherActionValue


def test_aggregation_encoder_uses_registered_source_project_root(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    def fake_encoder(root, device):
        captured["root"] = root
        captured["device"] = device
        return "encoder"

    monkeypatch.setattr(aggregation_execution, "_encoder", fake_encoder)
    monkeypatch.setattr(
        aggregation_execution,
        "MVAEncoderSession",
        lambda encoder: ("session", encoder),
    )

    session = aggregation_execution.build_registered_encoder_session(
        tmp_path,
        "cuda:2",
    )

    assert captured == {"root": tmp_path.resolve(), "device": "cuda:2"}
    assert session == ("session", "encoder")


def _group(domain: str, specimen: str, state: str) -> DynamicStateGroup:
    teacher = np.asarray([0.2, -0.1], dtype=np.float64)
    teacher.setflags(write=False)
    predictions = np.asarray([0.4, 0.7], dtype=np.float64)
    predictions.setflags(write=False)
    return DynamicStateGroup(
        state_id=state,
        specimen_id=specimen,
        domain_id=domain,
        outer_domain="d0",
        candidates=(
            CandidateDescriptor(0, 0, 1, 5, 100, 20),
            CandidateDescriptor(1, 0, 1, 5, 100, 20),
        ),
        true_cai=0.5,
        current_prediction=0.6,
        candidate_predictions=predictions,
        teacher_values=teacher,
        teacher_outer_domains=("d0",),
        teacher_fold_count=1,
        state_sha256=hashlib.sha256(state.encode()).hexdigest(),
    )


def _train(groups, round_index):
    return AggregationModel(
        model={"round": round_index},
        model_state_sha256=hashlib.sha256(
            f"model-{round_index}-{len(groups)}".encode()
        ).hexdigest(),
    )


def test_source_only_aggregation_deduplicates_and_records_rounds() -> None:
    initial = (_group("d1", "s1", "base-1"), _group("d2", "s2", "base-2"))

    def collect(_model, _specimens, round_index):
        return (
            _group("d1", "s1", "base-1"),
            _group("d1", "s1", f"visited-{round_index}"),
        )

    result = run_source_only_aggregation(
        initial,
        outer_domain="d0",
        rounds=2,
        train_model=_train,
        collect_source_groups=collect,
    )

    assert len(result.groups) == 4
    assert [audit.appended_state_count for audit in result.audits] == [1, 1]
    assert all(audit.target_state_count == 0 for audit in result.audits)
    assert result.final_model.model == {"round": 2}


def test_source_only_aggregation_rejects_target_visited_state() -> None:
    initial = (_group("d1", "s1", "base-1"), _group("d2", "s2", "base-2"))

    def collect(_model, _specimens, _round_index):
        return (_group("d0", "s1", "target-state"),)

    with pytest.raises(MAVISAggregationError, match="target"):
        run_source_only_aggregation(
            initial,
            outer_domain="d0",
            rounds=1,
            train_model=_train,
            collect_source_groups=collect,
        )


class _Encoder:
    def encode(self, images):
        return np.stack(
            [np.full(512, np.asarray(image).mean(), dtype=np.float64) for image in images]
        )


def test_on_policy_group_binds_visited_state_to_strict_oof_labels() -> None:
    authority = synthetic_authority()
    state = reveal_uniform_scout(
        authority,
        authority.policy_context("sample-001"),
        initial_budget=0.015625,
        checkpoint=0.25,
    )
    batch = build_source_candidate_batch(
        authority,
        state,
        dataset_id="domain-a",
        endpoint_budget=0.25,
        encoder=_Encoder(),
    )
    true_cai = authority.source_teacher_view("sample-001").true_cai
    current = 0.3
    values = tuple(
        TeacherActionValue(
            specimen_id="sample-001",
            dataset_id="domain-a",
            action=action,
            exact_added_cost=cost,
            true_cai=true_cai,
            current_prediction=current,
            candidate_prediction=current + 0.001 * (index + 1),
            error_before=abs(true_cai - current),
            error_after=abs(true_cai - (current + 0.001 * (index + 1))),
            primary_value=(
                abs(true_cai - current)
                - abs(true_cai - (current + 0.001 * (index + 1)))
            ),
            secondary_value=0.0,
            predictor_state_sha256="a" * 64,
        )
        for index, (action, cost) in enumerate(
            zip(batch.actions, batch.candidate_costs, strict=True)
        )
    )
    labels = FoldStateLabels(
        outer_domain="domain-b",
        query_domain="domain-a",
        current_prediction=current,
        teacher_state_sha256="b" * 64,
        predictor_state_sha256="a" * 64,
        action_values=values,
    )

    group = build_on_policy_group(
        state,
        batch,
        labels,
        outer_domain="domain-b",
    )

    assert group.outer_domain == "domain-b"
    assert group.domain_id == "domain-a"
    assert group.teacher_fold_count == 1
    assert group.state_id.startswith("on_policy::domain-b::sample-001::")
    np.testing.assert_allclose(
        group.teacher_values,
        [value.primary_value for value in values],
    )


class _Scorer:
    def score_actions(self, _state, candidates):
        return np.asarray(
            [-candidate.cell_index for candidate in candidates],
            dtype=np.float64,
        )


def test_aggregation_uses_one_on_policy_decision_state_per_checkpoint() -> None:
    authority = synthetic_authority()
    checkpoints = (0.0625, 0.125, 0.25)
    curve = rollout_scout_and_focus_curve(
        authority,
        specimen_id="sample-001",
        initial_budget=0.015625,
        checkpoints=checkpoints,
        scorer=_Scorer(),
        objective="direct_cost_aware",
        feedback=True,
    )

    selected = checkpoint_decision_states(curve)

    assert tuple(state.checkpoint for state in selected) == checkpoints
    assert tuple(state.state_sha256 for state in selected) == tuple(
        curve.scoring_states[
            next(
                index
                for index, step in enumerate(curve.steps)
                if step.nominal_checkpoint == checkpoint
            )
        ].state_sha256
        for checkpoint in checkpoints
    )


def test_aggregation_skips_initial_checkpoint_without_a_decision() -> None:
    authority = synthetic_authority(native_shape=(338, 340))
    curve = rollout_scout_and_focus_curve(
        authority,
        specimen_id="sample-001",
        initial_budget=0.03125,
        checkpoints=(0.03125, 0.0625),
        scorer=_Scorer(),
        objective="direct_cost_aware",
        feedback=True,
    )

    selected = checkpoint_decision_states(curve)

    assert tuple(state.checkpoint for state in selected) == (0.0625,)
