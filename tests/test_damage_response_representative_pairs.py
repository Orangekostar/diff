from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cmc_bbdm.damage_response.representative_pairs import (
    CurveRecord,
    RepresentativePairError,
    select_representative_pairs,
)


def _curve(value: float) -> np.ndarray:
    return np.linspace(0.0, value, 101, dtype=np.float64)


def test_nearest_strength_pairing_resolves_ties_by_canonical_id() -> None:
    records = (
        CurveRecord("a", "d1", 100.0, _curve(1.0)),
        CurveRecord("b", "d2", 90.0, _curve(2.0)),
        CurveRecord("c", "d3", 110.0, _curve(4.0)),
        CurveRecord("d", "d4", 89.0, _curve(8.0)),
        CurveRecord("e", "d5", 111.0, _curve(16.0)),
    )

    pairs = select_representative_pairs(records)

    pair_ids = {(pair.left_specimen_id, pair.right_specimen_id) for pair in pairs}
    assert ("a", "b") in pair_ids
    assert ("a", "c") not in pair_ids


def test_unordered_pairs_are_deduplicated_and_ranked_by_curve_rms() -> None:
    records = (
        CurveRecord("a", "d1", 100.0, _curve(0.0)),
        CurveRecord("b", "d1", 101.0, _curve(1.0)),
        CurveRecord("c", "d2", 200.0, _curve(2.0)),
        CurveRecord("d", "d2", 201.0, _curve(6.0)),
        CurveRecord("e", "d3", 300.0, _curve(3.0)),
        CurveRecord("f", "d3", 301.0, _curve(5.0)),
    )

    pairs = select_representative_pairs(records)

    assert [(pair.left_specimen_id, pair.right_specimen_id) for pair in pairs] == [
        ("c", "d"),
        ("e", "f"),
        ("a", "b"),
    ]
    assert len(pairs) == 3
    assert pairs[0].curve_rms > pairs[1].curve_rms > pairs[2].curve_rms
    assert pairs[0].strength_abs_difference_mpa == 1.0


def test_pair_selection_retains_only_top_twelve_and_is_input_order_invariant() -> None:
    records = tuple(
        CurveRecord(
            specimen_id=f"s{index:02d}",
            domain_id=f"d{index % 6}",
            published_cai_strength_mpa=float(100 + index // 2 * 10 + index % 2),
            normalized_curve=_curve(float(index)),
        )
        for index in range(30)
    )

    forward = select_representative_pairs(records)
    reverse = select_representative_pairs(tuple(reversed(records)))

    assert forward == reverse
    assert len(forward) == 12
    assert [pair.curve_rms for pair in forward] == sorted(
        (pair.curve_rms for pair in forward), reverse=True
    )


def test_pair_selection_uses_exact_101_point_finite_curves() -> None:
    records = (
        CurveRecord("a", "d1", 100.0, _curve(1.0)),
        CurveRecord("b", "d2", 101.0, _curve(2.0)),
    )

    with pytest.raises(RepresentativePairError, match="101"):
        select_representative_pairs(
            (replace(records[0], normalized_curve=np.ones(100)), records[1])
        )
    bad = _curve(1.0)
    bad[50] = np.nan
    with pytest.raises(RepresentativePairError, match="nonfinite"):
        select_representative_pairs(
            (replace(records[0], normalized_curve=bad), records[1])
        )
    with pytest.raises(RepresentativePairError, match="identity"):
        select_representative_pairs(
            (records[0], replace(records[1], specimen_id=records[0].specimen_id))
        )
