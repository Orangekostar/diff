from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.aei_multiview_regression.search import (
    CooperativeCandidate,
    select_cooperative_oof,
)


def test_search_ranks_equal_domain_then_worst_sd_and_registered_order() -> None:
    targets = np.asarray([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    domains = ("a", "a", "b", "b", "c", "c")
    base = np.column_stack((targets + 0.2, targets + 0.2, targets + 0.2))
    better = np.column_stack((targets + 0.1, targets + 0.1, targets + 0.1))
    candidates = {
        CooperativeCandidate("mse", 0.0): base,
        CooperativeCandidate("mse", 0.001): better,
        CooperativeCandidate("huber", 0.001): better,
    }

    result = select_cooperative_oof(candidates, targets=targets, domains=domains)

    assert result.selected == CooperativeCandidate("mse", 0.001)
    assert result.scores[0].candidate == CooperativeCandidate("mse", 0.0)
    assert result.scores[1].equal_domain_mae == pytest.approx(0.1)


def test_search_reports_collapse_without_rewarding_agreement() -> None:
    targets = np.asarray([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    domains = ("a", "a", "b", "b", "c", "c")
    diverse = np.column_stack((targets + 0.1, targets - 0.1, targets + 0.2))
    collapsed = np.column_stack((targets + 0.3, targets + 0.3, targets + 0.3))

    result = select_cooperative_oof(
        {
            CooperativeCandidate("mse", 0.0): diverse,
            CooperativeCandidate("mse", 1.0): collapsed,
        },
        targets=targets,
        domains=domains,
    )

    assert result.selected.lambda_consistency == 0.0
    collapsed_score = result.scores[1]
    assert collapsed_score.mean_absolute_disagreement == 0.0
    assert collapsed_score.collapsed is True


def test_search_monitors_all_pairwise_residual_correlations() -> None:
    targets = np.asarray([0.0, 0.2, 0.6, 1.1, 1.7, 2.4])
    domains = ("a", "a", "b", "b", "c", "c")
    predictions = np.column_stack(
        (
            targets - np.asarray([0.1, 0.3, 0.2, 0.5, 0.4, 0.8]),
            targets - np.asarray([0.2, 0.1, 0.4, 0.3, 0.7, 0.6]),
            targets - np.asarray([0.4, 0.2, 0.1, 0.6, 0.3, 0.9]),
        )
    )

    result = select_cooperative_oof(
        {CooperativeCandidate("mse", 0.1): predictions},
        targets=targets,
        domains=domains,
    )

    residuals = targets[:, None] - predictions
    expected = tuple(
        float(np.corrcoef(residuals[:, left], residuals[:, right])[0, 1])
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    assert result.scores[0].residual_correlations == pytest.approx(expected)
