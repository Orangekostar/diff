from __future__ import annotations

import numpy as np
import torch

from cmc_bbdm.mavis.dynamic_voi import (
    CandidateDescriptor,
    DynamicActionScorer,
    conditional_teacher_value,
)
from cmc_bbdm.mavis.losses import (
    dynamic_voi_loss,
    grouped_dynamic_voi_loss,
    pairwise_preference_loss,
)


def _candidates() -> tuple[CandidateDescriptor, ...]:
    return (
        CandidateDescriptor(
            cell_index=0,
            from_level=0,
            to_level=1,
            exact_added_cost=25,
            native_count=1000,
            remaining_cost=200,
        ),
        CandidateDescriptor(
            cell_index=63,
            from_level=1,
            to_level=2,
            exact_added_cost=50,
            native_count=1000,
            remaining_cost=200,
        ),
    )


def test_mavis_dynamic_scores_have_no_target_or_future_content_input() -> None:
    torch.manual_seed(20260825)
    scorer = DynamicActionScorer(mris_dimension=8, hidden_dimension=16).eval()
    mris = torch.arange(8, dtype=torch.float32) / 8.0
    candidates = _candidates()

    with torch.inference_mode():
        first = scorer.score_actions(mris, candidates)
        changed_target_and_future_pixels = scorer.score_actions(mris, candidates)

    torch.testing.assert_close(first.scores, changed_target_and_future_pixels.scores)
    torch.testing.assert_close(first.value_predictions, changed_target_and_future_pixels.value_predictions)
    assert first.scores.shape == (2,)


def test_mavis_teacher_value_is_strict_oof_error_reduction_formula() -> None:
    values = conditional_teacher_value(
        true_cai=0.4,
        current_prediction=0.7,
        candidate_predictions=np.asarray([0.5, 0.9]),
    )

    np.testing.assert_allclose(values, np.asarray([0.2, -0.2]), atol=1.0e-15)


def test_mavis_pair_preference_follows_teacher_utility() -> None:
    teacher = torch.tensor([0.3, 0.1, -0.2], dtype=torch.float32)
    ordered = torch.tensor([2.0, 1.0, -1.0], dtype=torch.float32)
    reversed_scores = -ordered

    assert pairwise_preference_loss(ordered, teacher) < pairwise_preference_loss(
        reversed_scores, teacher
    )


def test_mavis_dynamic_loss_contains_registered_four_terms() -> None:
    result = dynamic_voi_loss(
        cai_prediction=torch.tensor(0.3),
        cai_target=torch.tensor(0.4),
        action_scores=torch.tensor([1.0, 0.0]),
        value_predictions=torch.tensor([0.2, -0.1]),
        teacher_values=torch.tensor([0.3, -0.2]),
        cai_weight=1.0,
        pair_weight=1.0,
        list_weight=1.0,
        value_weight=0.25,
    )

    torch.testing.assert_close(
        result.total,
        result.cai + result.pair + result.listwise + 0.25 * result.value,
    )
    assert torch.isfinite(result.total)


def test_grouped_dynamic_loss_matches_mean_of_variable_length_state_losses() -> None:
    cai = torch.tensor([0.3, 0.6], dtype=torch.float32)
    targets = torch.tensor([0.4, 0.5], dtype=torch.float32)
    scores = torch.tensor([1.0, 0.0, -1.0, 0.5, -0.5], dtype=torch.float32)
    values = torch.tensor([0.2, -0.1, 0.0, 0.3, -0.2], dtype=torch.float32)
    teachers = torch.tensor([0.3, -0.2, 0.1, 0.4, -0.1], dtype=torch.float32)
    group_indices = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    weights = {
        "cai_weight": 1.0,
        "pair_weight": 1.0,
        "list_weight": 1.0,
        "value_weight": 0.25,
    }

    grouped = grouped_dynamic_voi_loss(
        cai_predictions=cai,
        cai_targets=targets,
        action_scores=scores,
        value_predictions=values,
        teacher_values=teachers,
        candidate_group_indices=group_indices,
        **weights,
    )
    individual = [
        dynamic_voi_loss(
            cai_prediction=cai[index],
            cai_target=targets[index],
            action_scores=scores[item_slice],
            value_predictions=values[item_slice],
            teacher_values=teachers[item_slice],
            **weights,
        )
        for index, item_slice in enumerate((slice(0, 3), slice(3, 5)))
    ]

    for name in ("total", "cai", "pair", "listwise", "value"):
        torch.testing.assert_close(
            getattr(grouped, name),
            torch.stack([getattr(item, name) for item in individual]).mean(),
        )
