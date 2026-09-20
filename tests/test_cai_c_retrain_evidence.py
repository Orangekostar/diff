from __future__ import annotations

import numpy as np
import pytest


def _episode(prediction: float, run: int = 0) -> dict[str, str]:
    return {
        "specimen_key": "domain:item",
        "dataset_id": "domain",
        "capture_group_id": "group",
        "method": "RANDOM",
        "run": str(run),
        "target_mpa": "0",
        "cells": "1",
        "costs": "0;0.25",
        "predictions_mpa": f"{prediction};{prediction}",
    }


def test_random_repeats_average_losses_not_predictions():
    from scripts.cai_c_retrain.evidence import curve_metrics

    result = curve_metrics([_episode(-10, 0), _episode(10, 1)], [0.25])

    assert result["mae"].tolist() == [10.0]
    assert result["physical_abs"].tolist() == [[10.0]]


def test_reused_bootstrap_weights_are_reindexed_by_key():
    from scripts.cai_c_retrain.evidence import reindex_bootstrap_weights

    old_keys = np.array(["a", "b", "c"])
    weights = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    reordered = reindex_bootstrap_weights(
        weights,
        old_keys,
        np.array(["c", "a", "b"]),
        np.array(["d2", "d1", "d1"]),
        np.array(["g3", "g1", "g2"]),
        old_domains=np.array(["d1", "d1", "d2"]),
        old_groups=np.array(["g1", "g2", "g3"]),
    )

    assert reordered.tolist() == [[3, 1, 2], [6, 4, 5]]


def test_equal_quality_keeps_unreached_negative_and_recrossing():
    from scripts.cai_c_retrain.evidence import equal_quality_comparison

    costs = np.array([0.0, 0.1, 0.2, 0.25])
    result = equal_quality_comparison(
        costs,
        main_mae=np.array([10.0, 7.0, 9.0, 6.0]),
        control_mae=np.array([10.0, 9.0, 7.0, 7.0]),
        target=8.0,
    )
    assert result["main_status"] == result["control_status"] == "REACHED"
    assert result["main_later_recrosses_target"] is True
    assert result["absolute_saving"] == pytest.approx(0.1)

    negative = equal_quality_comparison(
        costs,
        main_mae=np.array([10.0, 9.0, 8.0, 7.0]),
        control_mae=np.array([10.0, 6.0, 6.0, 6.0]),
        target=7.5,
    )
    assert negative["absolute_saving"] == pytest.approx(-0.15)

    missing = equal_quality_comparison(
        costs,
        main_mae=np.array([10.0, 9.0, 8.0, 7.0]),
        control_mae=np.array([10.0, 9.0, 9.0, 9.0]),
        target=7.5,
    )
    assert missing["control_status"] == "NOT_REACHED_WITHIN_OBSERVED_RANGE"
    assert missing["absolute_saving"] is None


def test_timing_keeps_negative_terms_and_satisfies_identity():
    from scripts.cai_c_retrain.evidence import timing_decomposition

    result = timing_decomposition(
        [0.0, 0.05, 0.13, 0.20, 0.25],
        [10.0, 8.0, 9.0, 5.0, 6.0],
    )

    assert any(value < 0 for value in result["terms"])
    assert result["area"] == pytest.approx(
        result["initial_error"] - sum(result["terms"]), abs=1e-12
    )
    assert sum(result["stage_contributions"]) == pytest.approx(sum(result["terms"]))


def test_event_grid_is_derived_not_fixed_length():
    from scripts.cai_c_retrain.evidence import derive_event_grid

    rows = [
        {"costs": "0;0.03;0.25"},
        {"costs": "0;0.125;0.20;0.25"},
    ]

    assert derive_event_grid(rows).tolist() == [0.0, 0.03, 0.125, 0.2, 0.25]


def test_invalid_cost_trajectory_is_rejected():
    from scripts.cai_c_retrain.evidence import curve_metrics

    row = _episode(1)
    row["costs"] = "0;0.2;0.1"
    row["predictions_mpa"] = "1;1;1"
    with pytest.raises(ValueError, match="cost"):
        curve_metrics([row], [0.1])
