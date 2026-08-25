from __future__ import annotations

import math

import numpy as np
from mavis_test_support import synthetic_authority

from cmc_bbdm.mavis.state_bank import (
    PlannedAction,
    materialize_action_plan,
)
from cmc_bbdm.mavis.state_bank_artifacts import (
    StateBankSample,
    state_bank_rows,
)
from cmc_bbdm.mavis.state_candidates import build_source_candidate_batch
from cmc_bbdm.mavis.teacher import FoldStateLabels, TeacherActionValue
from cmc_bbdm.mva.measurement_state import RefinementAction


class _Encoder:
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        return np.asarray(
            [np.resize(np.mean(image, axis=(0, 1)), 512) for image in images],
            dtype=np.float64,
        )


def _sample() -> tuple[StateBankSample, str]:
    authority = synthetic_authority()
    action = RefinementAction(0, 0, 1)
    trajectory = materialize_action_plan(
        authority,
        specimen_id="sample-001",
        method="uniform",
        seed=None,
        initial_budget=0.015625,
        checkpoints=(0.25,),
        actions=(PlannedAction(action=action, nominal_checkpoint=0.25),),
    )
    snapshot = trajectory.snapshots[0]
    candidates = build_source_candidate_batch(
        authority,
        snapshot.inspection_state,
        dataset_id="domain-a",
        endpoint_budget=0.25,
        encoder=_Encoder(),
    )
    folds = []
    for fold_index, outer_domain in enumerate(("domain-c", "domain-b")):
        predictor_hash = str(fold_index + 1) * 64
        values = tuple(
            TeacherActionValue(
                specimen_id="sample-001",
                dataset_id="domain-a",
                action=candidate,
                exact_added_cost=cost,
                true_cai=0.4,
                current_prediction=0.3 + 0.01 * fold_index,
                candidate_prediction=0.35 + 0.01 * fold_index,
                error_before=0.1 - 0.01 * fold_index,
                error_after=0.05 - 0.01 * fold_index,
                primary_value=0.05,
                secondary_value=0.0075,
                predictor_state_sha256=predictor_hash,
            )
            for candidate, cost in zip(
                candidates.actions,
                candidates.candidate_costs,
                strict=True,
            )
        )
        folds.append(
            FoldStateLabels(
                outer_domain=outer_domain,
                query_domain="domain-a",
                current_prediction=0.3 + 0.01 * fold_index,
                teacher_state_sha256=chr(ord("a") + fold_index) * 64,
                predictor_state_sha256=predictor_hash,
                action_values=values,
            )
        )
    return (
        StateBankSample(
            dataset_id="domain-a",
            trajectory=trajectory,
            snapshot=snapshot,
            candidates=candidates,
            fold_labels=tuple(folds),
        ),
        authority.state_sha256,
    )


def test_mavis_state_bank_rows_preserve_causal_state_and_privileged_labels() -> None:
    sample, authority_sha256 = _sample()

    state_row, action_rows = state_bank_rows(
        sample,
        authority_state_sha256=authority_sha256,
        endpoint_budget=0.25,
    )

    state = sample.snapshot.inspection_state
    assert state_row["specimen_id"] == "sample-001"
    assert state_row["domain_id"] == "domain-a"
    assert state_row["state_id"]
    assert state_row["step"] == 1
    assert state_row["exact_acquired_cost"] == state.exact_acquired_count
    assert len(state_row["revealed_rows"]) == state.exact_acquired_count
    assert len(state_row["revealed_red"]) == state.exact_acquired_count
    assert state_row["acquired_action_cell_indices"] == [0]
    assert state_row["candidate_cell_indices"] == [
        action.cell_index for action in sample.candidates.actions
    ]
    assert state_row["teacher_outer_domains"] == ["domain-b", "domain-c"]
    assert "teacher_true_cai" not in state_row
    expected_remaining = math.floor(0.25 * state.native_count) - (
        state.exact_acquired_count
    )
    assert state_row["remaining_cost_to_endpoint"] == expected_remaining

    assert len(action_rows) == 2 * len(sample.candidates.actions)
    assert {row["outer_domain"] for row in action_rows} == {
        "domain-b",
        "domain-c",
    }
    assert {row["teacher_true_cai"] for row in action_rows} == {0.4}
    for row in action_rows:
        assert row["candidate_exact_cost_after"] == (
            state.exact_acquired_count + row["exact_added_cost"]
        )
        assert row["primary_value_per_cost"] == (
            row["primary_value"] / row["exact_added_cost"]
        )
