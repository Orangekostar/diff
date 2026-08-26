from __future__ import annotations

import numpy as np
import polars as pl
from mavis_test_support import synthetic_authority

from cmc_bbdm.mavis.closed_loop_execution import (
    _target_donor_lookup,
    evaluate_inspection_curve,
)
from cmc_bbdm.mavis.reveal import reveal_uniform_scout
from cmc_bbdm.mavis.rollout import rollout_scout_and_focus_curve


class _Scorer:
    def score_actions(self, _state, candidates):
        return np.asarray(
            [-candidate.cell_index for candidate in candidates],
            dtype=np.float64,
        )


class _Evaluator:
    model_state_sha256 = "e" * 64

    def predict_inspection_state(self, _state, *, device):
        assert device == "cpu"
        return 0.3


def test_closed_loop_curve_uses_one_cai_endpoint_and_exact_cost() -> None:
    authority = synthetic_authority(true_cai=0.4)
    curve = rollout_scout_and_focus_curve(
        authority,
        specimen_id="sample-001",
        initial_budget=0.015625,
        checkpoints=(0.0625, 0.125),
        scorer=_Scorer(),
        objective="direct_cost_aware",
        feedback=True,
    )

    rows = evaluate_inspection_curve(
        authority,
        outer_domain="domain-a",
        method="mavis_full",
        checkpoints=curve.checkpoints,
        states=curve.checkpoint_states,
        cai_evaluator=_Evaluator(),
        device="cpu",
    )

    assert len(rows) == 2
    np.testing.assert_allclose([row["absolute_error"] for row in rows], 0.1)
    assert all(
        row["effective_budget"]
        == row["exact_acquired_cost"] / row["native_count"]
        for row in rows
    )
    assert all(row["reconstruction_mse"] >= 0.0 for row in rows)


def test_closed_loop_target_donors_exclude_source_recipient_rows() -> None:
    mapping = pl.DataFrame(
        {
            "outer_domain": ["d0", "d0", "d0"],
            "recipient_id": ["target-a", "target-b", "source-a"],
            "recipient_domain": ["d0", "d0", "d1"],
            "recipient_pool": ["target", "target", "source"],
            "donor_id": ["donor-a", "donor-b", "donor-c"],
        }
    )

    assert _target_donor_lookup(
        mapping,
        outer_domain="d0",
        target_ids=("target-a", "target-b"),
    ) == {"target-a": "donor-a", "target-b": "donor-b"}


def test_closed_loop_curve_accepts_quantized_initial_scout() -> None:
    authority = synthetic_authority(native_shape=(338, 340))
    state = reveal_uniform_scout(
        authority,
        authority.policy_context("sample-001"),
        initial_budget=0.03125,
        checkpoint=0.25,
    )
    assert state.effective_budget > 0.03125

    rows = evaluate_inspection_curve(
        authority,
        outer_domain="domain-a",
        method="uniform",
        checkpoints=(0.03125,),
        states=(state,),
        cai_evaluator=_Evaluator(),
        device="cpu",
    )

    assert rows[0]["effective_budget"] == state.effective_budget
