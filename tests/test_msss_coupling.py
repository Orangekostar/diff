from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.msss.coupling import (
    CouplingError,
    align_factor_directions,
    classify_ordered_direction,
    diagnose_coupling,
    evaluate_group_curve,
    stable_rank_tertiles,
)
from cmc_bbdm.msss.scale_features import ScaleCondition


def _conditions() -> tuple[ScaleCondition, ...]:
    return tuple(
        ScaleCondition(
            condition_id=f"sampling:density={value}",
            axis="sampling",
            value=value,
            coarse_rank=rank,
            primary_eligible=True,
            is_full_identity=rank == 0,
        )
        for rank, value in enumerate((1.0, 0.5, 0.25, 0.125))
    )


def test_stable_rank_tertiles_use_value_then_specimen_identity() -> None:
    specimen_ids = ("s3", "s1", "s2", "s6", "s4", "s5")
    values = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    labels = stable_rank_tertiles(values, specimen_ids=specimen_ids)

    assert labels == ("middle", "low", "low", "high", "middle", "high")
    assert tuple(labels.count(name) for name in ("low", "middle", "high")) == (
        2,
        2,
        2,
    )


def test_group_curve_uses_equal_domain_mae_and_coarsest_sufficient_scale() -> None:
    conditions = _conditions()
    errors = {
        conditions[0].condition_id: np.asarray([0.0, 1.0, 1.0, 1.0]),
        conditions[1].condition_id: np.asarray([0.0, 1.04, 1.04, 1.04]),
        conditions[2].condition_id: np.asarray([0.0, 1.05, 1.05, 1.05]),
        conditions[3].condition_id: np.asarray([0.0, 1.20, 1.20, 1.20]),
    }

    rows, selection = evaluate_group_curve(
        conditions,
        absolute_errors=errors,
        dataset_ids=("d1", "d2", "d2", "d2"),
        selected_indices=(0, 1, 2, 3),
        grouping="layup_family",
        group_value="cross_ply",
        margin=0.05,
    )

    assert rows[0].equal_domain_mae == pytest.approx(0.5)
    assert rows[2].equal_domain_mae == pytest.approx(0.525)
    assert selection.selected_condition_id == conditions[2].condition_id
    assert selection.selected_coarse_rank == 2
    assert selection.over_coarse_condition_id == conditions[3].condition_id
    assert selection.boundary_confirmed is True


def test_group_curve_rejects_an_incomplete_candidate_roster() -> None:
    conditions = _conditions()
    errors = {
        condition.condition_id: np.zeros(4, dtype=np.float64)
        for condition in conditions[:-1]
    }

    with pytest.raises(CouplingError, match="roster"):
        evaluate_group_curve(
            conditions,
            absolute_errors=errors,
            dataset_ids=("d1", "d1", "d2", "d2"),
            selected_indices=(0, 1, 2, 3),
            grouping="ply_count",
            group_value="8",
            margin=0.05,
        )


@pytest.mark.parametrize(
    ("ranks", "expected"),
    [
        ((1, 2, 2), "COARSER"),
        ((3, 2, 1), "FINER"),
        ((2, 2, 2), "SAME"),
        ((1, 3, 2), "NON_MONOTONIC"),
    ],
)
def test_ordered_direction_is_conservative(
    ranks: tuple[int, int, int], expected: str
) -> None:
    assert classify_ordered_direction(ranks) == expected


def test_cross_axis_alignment_requires_two_matching_non_neutral_axes() -> None:
    aligned = align_factor_directions(("COARSER", "COARSER", "NON_MONOTONIC"))
    split = align_factor_directions(("COARSER", "FINER", "SAME"))

    assert aligned.status == "CROSS_AXIS_ALIGNED"
    assert aligned.direction == "COARSER"
    assert aligned.axis_count == 2
    assert split.status == "NO_CROSS_AXIS_ALIGNMENT"
    assert split.direction is None
    assert split.axis_count == 1


def test_complete_diagnostic_covers_all_requested_groups_without_promotion() -> None:
    axes = ("sampling", "gaussian", "wavelet")
    conditions = tuple(
        ScaleCondition(
            condition_id=f"{axis}:candidate={rank}",
            axis=axis,
            value=float(rank),
            coarse_rank=rank,
            primary_eligible=True,
            is_full_identity=rank == 0,
            wavelet="db2" if axis == "wavelet" else None,
            level=rank if axis == "wavelet" else None,
            mode="low_only" if axis == "wavelet" else None,
        )
        for axis in axes
        for rank in range(3)
    )
    specimen_ids = tuple(f"s{index:02d}" for index in range(12))
    dataset_ids = tuple(
        domain for domain in ("d1", "d2", "d3", "d4", "d5", "d6") for _ in range(2)
    )
    ply_count = tuple(value for value in (8, 8, 16, 16, 24, 24) for _ in range(2))
    layup_family = tuple(
        value
        for value in (
            "cross_ply",
            "quasi_isotropic",
            "cross_ply",
            "quasi_isotropic",
            "cross_ply",
            "quasi_isotropic",
        )
        for _ in range(2)
    )
    damage = np.arange(1.0, 13.0)
    absolute_errors = {
        condition.condition_id: np.full(
            12, (1.0, 1.02, 1.20)[condition.coarse_rank], dtype=np.float64
        )
        for condition in conditions
    }

    result = diagnose_coupling(
        conditions=conditions,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        ply_count=ply_count,
        layup_family=np.asarray(layup_family, dtype=str),
        damage_sizes={
            "damage_area": damage,
            "damage_height": damage[::-1],
            "damage_width": np.roll(damage, 2),
        },
        absolute_errors=absolute_errors,
        margin=0.05,
    )

    assert len(result.selections) == 3 * (6 + 3 + 2 + 3 + 3 + 3)
    assert len(result.curves) == len(result.selections) * 3
    assert len(result.damage_bins) == 12 * 3
    assert len(result.trends) == 3 * 5
    assert len(result.alignments) == 5
    assert result.coupling_status == "NO_CONSISTENT_SIGNAL"
    assert result.validation_status == "NOT_VALIDATED_POST_HOC"
    assert result.s2_status == "NOT_RUN_NOT_AUTHORIZED"
    assert len(result.state_sha256) == 64
