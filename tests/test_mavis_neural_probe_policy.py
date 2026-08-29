from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch
from test_mavis_neural_probe_training import _bank_and_authority, _fit

from cmc_bbdm.mavis.dynamic_training import (
    DynamicTrainingAudit,
    DynamicVoITrainingModel,
    FittedDynamicVoI,
)
from cmc_bbdm.mavis.dynamic_voi import CandidateDescriptor
from cmc_bbdm.mavis.neural_probe.closed_loop import evaluate_n4_comparison
from cmc_bbdm.mavis.neural_probe.policy import (
    SpatialProbeDeployedScorer,
    SpatialProbePolicyError,
)
from cmc_bbdm.mavis.reveal import reveal_uniform_scout
from cmc_bbdm.mavis.rollout import rollout_scout_and_focus_curve


def _scorer():
    bank, authority = _bank_and_authority(outer_target_delta=0.0)
    spatial = _fit(bank)
    torch.manual_seed(20260825)
    dynamic = FittedDynamicVoI(
        outer_domain="d0",
        mris_dimension=64,
        hidden_dimension=8,
        model=DynamicVoITrainingModel(
            mris_dimension=64,
            hidden_dimension=8,
        ).eval(),
        audit=DynamicTrainingAudit(
            outer_domain="d0",
            validation_domain=None,
            fit_domains=("d2", "d3", "d4", "d5"),
            fit_specimen_ids=("source",),
            validation_specimen_ids=(),
            epochs_run=1,
            selected_epoch=1,
            best_validation_regret=None,
        ),
        loss_weights=(
            ("cai", 1.0),
            ("list", 1.0),
            ("pair", 1.0),
            ("value", 0.25),
        ),
        model_state_sha256="d" * 64,
    )
    return (
        SpatialProbeDeployedScorer(
            mris_model=spatial,
            dynamic_model=dynamic,
            device="cpu",
        ),
        spatial,
        authority,
    )


def test_spatial_probe_scorer_is_deterministic_and_preserves_candidates() -> None:
    scorer, _spatial, authority = _scorer()
    state = reveal_uniform_scout(
        authority,
        authority.policy_context("d0-0"),
        initial_budget=0.015625,
        checkpoint=0.25,
    )
    candidates = (
        CandidateDescriptor(4, 0, 1, 20, state.native_count, 100),
        CandidateDescriptor(2, 0, 1, 10, state.native_count, 100),
    )
    roster = tuple(
        (
            item.cell_index,
            item.from_level,
            item.to_level,
            item.exact_added_cost,
            item.remaining_cost,
        )
        for item in candidates
    )

    first = scorer.score_actions(state, candidates)
    second = scorer.score_actions(state, candidates)

    torch.testing.assert_close(first.scores, second.scores)
    torch.testing.assert_close(first.value_predictions, second.value_predictions)
    assert tuple(
        (
            item.cell_index,
            item.from_level,
            item.to_level,
            item.exact_added_cost,
            item.remaining_cost,
        )
        for item in candidates
    ) == roster
    assert scorer.mode == "real"
    assert scorer.outer_domain == "d0"


def test_spatial_probe_scorer_has_no_target_label_input() -> None:
    scorer, _spatial, authority = _scorer()
    _changed_bank, changed_authority = _bank_and_authority(outer_target_delta=1000.0)
    state = reveal_uniform_scout(
        authority,
        authority.policy_context("d0-0"),
        initial_budget=0.015625,
        checkpoint=0.25,
    )
    changed_state = reveal_uniform_scout(
        changed_authority,
        changed_authority.policy_context("d0-0"),
        initial_budget=0.015625,
        checkpoint=0.25,
    )
    candidates = (
        CandidateDescriptor(4, 0, 1, 20, state.native_count, 100),
        CandidateDescriptor(2, 0, 1, 10, state.native_count, 100),
    )

    torch.testing.assert_close(
        scorer.score_actions(state, candidates).scores,
        scorer.score_actions(changed_state, candidates).scores,
    )


def test_spatial_probe_scorer_runs_unchanged_rollout_and_cai_adapter() -> None:
    scorer, spatial, authority = _scorer()
    curve = rollout_scout_and_focus_curve(
        authority,
        specimen_id="d0-0",
        initial_budget=0.015625,
        checkpoints=(0.0625, 0.125),
        scorer=scorer,
        objective="direct_cost_aware",
        feedback=True,
    )

    assert curve.steps
    assert all(
        step.exact_cost_after - step.exact_cost_before
        == step.candidate.exact_added_cost
        for step in curve.steps
    )
    assert len(spatial.model_state_sha256) == 64
    assert np.isfinite(
        spatial.predict_inspection_state(curve.checkpoint_states[-1], device="cpu")
    )


def test_spatial_probe_scorer_rejects_wrong_native_cost_roster() -> None:
    scorer, _spatial, authority = _scorer()
    state = reveal_uniform_scout(
        authority,
        authority.policy_context("d0-0"),
        initial_budget=0.015625,
        checkpoint=0.25,
    )
    candidates = (CandidateDescriptor(4, 0, 1, 20, state.native_count + 1, 100),)

    with pytest.raises(SpatialProbePolicyError, match="request is invalid"):
        scorer.score_actions(state, candidates)


def _closed_loop_predictions(method: str, *, error: float):
    rows: list[dict[str, object]] = []
    for domain_index in range(6):
        domain = f"d{domain_index}"
        for specimen_index in range(2):
            specimen = f"{domain}-{specimen_index}"
            for checkpoint_index, checkpoint in enumerate((0.1, 0.2)):
                cost = 10 * (checkpoint_index + 1)
                rows.append(
                    {
                        "outer_domain": domain,
                        "specimen_id": specimen,
                        "method": method,
                        "nominal_checkpoint": checkpoint,
                        "initial_budget": 0.05,
                        "action_count": checkpoint_index + 1,
                        "exact_acquired_cost": cost,
                        "native_count": 100,
                        "effective_budget": cost / 100.0,
                        "target": 0.0,
                        "prediction": error,
                        "absolute_error": error,
                        "reconstruction_mse": error,
                    }
                )
    return rows


def test_n4_comparison_uses_static_minus_candidate_auebc() -> None:
    predictions = pl.DataFrame(
        [
            *_closed_loop_predictions("spatial_probe", error=0.1),
            *_closed_loop_predictions("mvd_m1_o2", error=0.2),
            *_closed_loop_predictions("mavis_full", error=0.15),
        ],
        infer_schema_length=None,
    )

    result = evaluate_n4_comparison(
        predictions,
        domain_order=tuple(f"d{index}" for index in range(6)),
        checkpoints=(0.1, 0.2),
        bootstrap_replicates=100,
        seed=20260825,
    )

    assert result.point_estimate == pytest.approx(0.1)
    assert result.ci95_lower == pytest.approx(0.1)
    assert result.ci95_upper == pytest.approx(0.1)
    assert result.favorable_domain_count == 6
    assert result.gate == "END_TO_END_STRONG_GO"
    assert result.domain_metrics.get_column(
        "static_minus_spatial_probe_auebc"
    ).to_list() == pytest.approx([0.1] * 6)
