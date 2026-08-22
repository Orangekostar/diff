from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.aei_multiview_regression.formal_outer import (
    authorize_e4,
    authorize_e5,
    performance_metrics,
    stress_group_splits,
)


def test_e4_requires_below_full_below_best_single_and_four_domains() -> None:
    assert authorize_e4(
        fusion_mae=0.085,
        baseline_mae=0.08963580465761432,
        best_single_mae=0.088,
        improved_domain_count=4,
    )
    assert not authorize_e4(
        fusion_mae=0.089,
        baseline_mae=0.08963580465761432,
        best_single_mae=0.088,
        improved_domain_count=6,
    )
    assert not authorize_e4(
        fusion_mae=0.085,
        baseline_mae=0.08963580465761432,
        best_single_mae=0.088,
        improved_domain_count=3,
    )


def test_e5_requires_all_three_registered_conditions() -> None:
    assert authorize_e5(
        e1_nontrivial=True,
        complementarity_confirmed=True,
        oracle_gap_fraction=0.06,
    )
    assert not authorize_e5(
        e1_nontrivial=True,
        complementarity_confirmed=False,
        oracle_gap_fraction=0.20,
    )
    assert not authorize_e5(
        e1_nontrivial=True,
        complementarity_confirmed=True,
        oracle_gap_fraction=0.01,
    )


def test_stress_group_splits_are_exhaustive_and_disjoint() -> None:
    groups = ("a", "a", "b", "b", "c", "c")
    splits = stress_group_splits(groups)

    assert tuple(item[0] for item in splits) == ("a", "b", "c")
    covered = []
    for group, fit, query in splits:
        assert set(fit).isdisjoint(query)
        assert all(groups[index] != group for index in fit)
        assert all(groups[index] == group for index in query)
        covered.extend(query)
    assert sorted(covered) == list(range(6))


def test_performance_metrics_report_registered_mpa_scale() -> None:
    targets = np.asarray([0.5, 0.6, 0.7, 0.8])
    predictions = np.asarray([0.4, 0.5, 0.9, 0.7])
    intact_strength_mpa = np.asarray([100.0, 200.0, 150.0, 250.0])

    result = performance_metrics(
        "method",
        targets,
        predictions,
        ("a", "a", "b", "b"),
        intact_strength_mpa=intact_strength_mpa,
    )

    absolute_mpa = np.abs(targets - predictions) * intact_strength_mpa
    assert result.equal_domain_mae_mpa == pytest.approx(
        np.mean((np.mean(absolute_mpa[:2]), np.mean(absolute_mpa[2:])))
    )
    assert result.rmse_mpa == pytest.approx(
        np.sqrt(np.mean(((targets - predictions) * intact_strength_mpa) ** 2))
    )
