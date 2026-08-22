from __future__ import annotations

import pytest

from cmc_bbdm.msss.msss_selector import axis_gate, s1_gate, selection_stability
from cmc_bbdm.msss.noninferiority import NoninferiorityError, select_noninferior


def test_noninferiority_selects_coarsest_eligible_candidate() -> None:
    result = select_noninferior(
        (1.0, 0.5, 0.25, 0.125),
        (1.0, 1.02, 1.05, 1.08),
        margin=0.05,
    )

    assert result.sufficient_candidates == (1.0, 0.5, 0.25)
    assert result.selected == 0.25
    assert result.over_coarse == 0.125
    assert result.boundary_confirmed
    assert result.plateau


def test_noninferiority_uses_relative_full_margin_and_reports_missing_boundary() -> None:
    result = select_noninferior(
        (0.0, 1.0, 2.0, 4.0),
        (0.2, 0.205, 0.208, 0.209),
        margin=0.05,
    )

    assert result.threshold == pytest.approx(0.21)
    assert result.selected == 4.0
    assert result.over_coarse is None
    assert not result.boundary_confirmed


def test_noninferiority_rejects_nonfinite_or_misaligned_inputs() -> None:
    with pytest.raises(NoninferiorityError):
        select_noninferior((1.0, 0.5), (0.1,), margin=0.05)
    with pytest.raises(NoninferiorityError):
        select_noninferior((1.0, 0.5), (0.1, float("nan")), margin=0.05)


def test_selection_stability_requires_four_selections_in_adjacent_window() -> None:
    stable = selection_stability(
        ("a", "b", "b", "c", "b", "c"),
        candidate_order=("a", "b", "c", "d"),
        minimum_outer_folds=4,
        window_steps=1,
    )
    unstable = selection_stability(
        ("a", "a", "c", "c", "e", "e"),
        candidate_order=("a", "b", "c", "d", "e"),
        minimum_outer_folds=4,
        window_steps=1,
    )

    assert stable.passed and stable.maximum_in_window == 5
    assert not unstable.passed and unstable.maximum_in_window == 2


def test_axis_and_s1_gates_require_all_frozen_conditions() -> None:
    passing = axis_gate(
        plateau=True,
        boundary_confirmed=True,
        stable=True,
        mechanically_sufficient=True,
        spatially_specific=True,
    )
    failing = axis_gate(
        plateau=True,
        boundary_confirmed=False,
        stable=True,
        mechanically_sufficient=True,
        spatially_specific=True,
    )

    assert passing.status == "PASS"
    assert failing.status == "FAIL"
    assert s1_gate((passing, passing, failing)).status == "GO"
    assert s1_gate((passing, passing, passing)).status == "STRONG_GO"
    assert s1_gate((passing, failing, failing)).status == "NO_GO"
