from __future__ import annotations

import numpy as np
import torch
from mavis_test_support import synthetic_authority

from cmc_bbdm.mavis.dynamic_training import (
    DynamicTrainingAudit,
    DynamicVoITrainingModel,
    FittedDynamicVoI,
)
from cmc_bbdm.mavis.dynamic_voi import CandidateDescriptor
from cmc_bbdm.mavis.mechanics_head import FoldNormalizer, MRISMechanicsModel
from cmc_bbdm.mavis.mris_training import FittedMRISModel, MRISTrainingAudit
from cmc_bbdm.mavis.policy import (
    DeployedDynamicScorer,
    FrozenCellScorer,
    select_cost_aware_action,
)
from cmc_bbdm.mavis.reveal import reveal_uniform_scout
from cmc_bbdm.mavis.state_encoder import MRISStateEncoder


def _candidates() -> tuple[CandidateDescriptor, ...]:
    return (
        CandidateDescriptor(4, 0, 1, 20, 1000, 100),
        CandidateDescriptor(2, 0, 1, 10, 1000, 100),
        CandidateDescriptor(1, 0, 1, 10, 1000, 100),
    )


def test_mavis_policy_ties_are_cost_then_geometry_deterministic() -> None:
    selected = select_cost_aware_action(
        _candidates(),
        np.asarray([1.0, 1.0, 1.0]),
        objective="raw_score",
    )

    assert selected.candidate.cell_index == 1


def test_mavis_policy_value_per_cost_uses_exact_native_location_cost() -> None:
    selected = select_cost_aware_action(
        _candidates(),
        np.asarray([2.0, 1.5, 1.0]),
        objective="value_per_exact_cost",
    )

    assert selected.candidate.cell_index == 2
    assert selected.objective_score == 0.15


def test_frozen_cell_scorer_reuses_initial_cell_value_after_refinement() -> None:
    scores = np.arange(64, dtype=np.float64)
    scorer = FrozenCellScorer(scores)
    candidates = (
        CandidateDescriptor(4, 0, 1, 20, 1000, 100),
        CandidateDescriptor(4, 1, 2, 40, 1000, 100),
    )

    np.testing.assert_array_equal(
        scorer.score_actions(None, candidates),
        np.asarray([4.0, 4.0]),
    )


def _deployed_scorer() -> DeployedDynamicScorer:
    torch.manual_seed(20260825)
    encoder = MRISStateEncoder(
        context_dimension=34,
        hidden_dimension=8,
        output_dimension=8,
    )
    normalizer = FoldNormalizer(
        outer_domain="domain-a",
        excluded_domains=("domain-a",),
        context_mean=np.zeros(34),
        context_scale=np.ones(34),
        target_mean=0.0,
        target_scale=1.0,
        fit_specimen_ids=("source",),
        fit_dataset_ids=("source-domain",),
        fit_domains=("source-domain",),
        state_sha256="normalizer",
    )
    mris = FittedMRISModel(
        mode="real",
        outer_domain="domain-a",
        hidden_dimension=8,
        mris_dimension=8,
        model=MRISMechanicsModel(encoder).eval(),
        normalizer=normalizer,
        audit=MRISTrainingAudit(
            mode="real",
            outer_domain="domain-a",
            validation_domains=(),
            fit_domains=("source-domain",),
            fit_specimen_ids=("source",),
            validation_specimen_ids=(),
            epochs_run=1,
            selected_epoch=1,
            best_validation_mae=None,
            normalizer_state_sha256="normalizer",
        ),
        model_state_sha256="mris",
    )
    dynamic = FittedDynamicVoI(
        outer_domain="domain-a",
        mris_dimension=8,
        hidden_dimension=8,
        model=DynamicVoITrainingModel(
            mris_dimension=8,
            hidden_dimension=8,
        ).eval(),
        audit=DynamicTrainingAudit(
            outer_domain="domain-a",
            validation_domain=None,
            fit_domains=("source-domain",),
            fit_specimen_ids=("source",),
            validation_specimen_ids=(),
            epochs_run=1,
            selected_epoch=1,
            best_validation_regret=None,
        ),
        loss_weights=(("cai", 1.0), ("list", 1.0), ("pair", 1.0), ("value", 0.25)),
        model_state_sha256="dynamic",
    )
    return DeployedDynamicScorer(mris_model=mris, dynamic_model=dynamic)


def test_deployed_dynamic_scorer_has_no_target_label_input() -> None:
    first = synthetic_authority(true_cai=0.1)
    changed = synthetic_authority(true_cai=1000.0)
    state = reveal_uniform_scout(
        first,
        first.policy_context("sample-001"),
        initial_budget=0.015625,
        checkpoint=0.0625,
    )
    changed_state = reveal_uniform_scout(
        changed,
        changed.policy_context("sample-001"),
        initial_budget=0.015625,
        checkpoint=0.0625,
    )
    scorer = _deployed_scorer()
    candidates = (
        CandidateDescriptor(4, 0, 1, 20, state.native_count, 100),
        CandidateDescriptor(2, 0, 1, 10, state.native_count, 100),
    )

    torch.testing.assert_close(
        scorer.score_actions(state, candidates).scores,
        scorer.score_actions(changed_state, candidates).scores,
    )
