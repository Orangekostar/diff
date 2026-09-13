"""Independent synthetic examples for frozen evidence; no model execution."""

import numpy as np
import pytest

from cmc_bbdm.cai_agent_v3 import paper_evidence_math as m


def test_left_step_and_horizon():
    assert m.held([0, 0.125, 0.25], [190, 194, 198], [0.0625, 0.125]).tolist() == [
        190,
        194,
    ]
    assert m.area([0, 0.125, 0.25], [190, 194, 198], 200) == 8
    assert m.area([0, 0.125], [190, 198], 200) == 6
    assert m.held([0, 0.125], [190, 198], [0.25])[0] == 198
    with pytest.raises(ValueError):
        m.held([0, 0.125], [190, 198], [0.5])


def test_repeats_and_domain_estimands():
    residual = np.array([[[-10.0], [-3.0]], [[10.0], [3.0]]])
    out = m.aggregate(residual, np.array([200.0, 210.0]))
    assert out["mae"][0] == 6.5
    assert out["mse"][0] == 54.5
    assert out["rmse"][0] == np.sqrt(54.5)
    assert m.weighted(np.array([0.0, 10.0, 100.0]), np.array([[2, 2, 1]]))[0] == 24
    assert (
        m.weighted(np.array([0.0, 10.0, 100.0]), np.ones((1, 3)), ["a", "a", "b"])[0]
        == 52.5
    )


def test_group_draws_preserve_members_and_domains():
    w = m.bootstrap_weights(
        ["a", "a", "a", "b"], ["g1", "g1", "g2", "g3"], replicates=100, seed=3
    )
    assert np.array_equal(w[:, 0], w[:, 1])
    assert np.all(w[:, 0] + w[:, 2] == 2) and np.all(w[:, 3] == 1)
    assert np.array_equal(
        w,
        m.bootstrap_weights(
            ["a", "a", "a", "b"], ["g1", "g1", "g2", "g3"], replicates=100, seed=3
        ),
    )


def test_quality_both_earliest_and_nonmonotonic():
    grid = [0, 0.0625, 0.125, 0.1875, 0.25]
    q = 46.9099169921875
    a = m.first_quality(grid, [58.55, 45.5174, 44.1136, 44.6845, 44.2858], q)
    b = m.first_quality(grid, [58.55, 50.2209, 46.189, 46.5551, q], q)
    s = m.first_quality(grid, [58.55, 46.465, 48.0642, 46.784, 46.9931], q)
    assert (a["cost"], b["cost"], s["cost"]) == (0.0625, 0.125, 0.0625)
    assert m.saving(a["cost"], b["cost"])["relative_saving"] == 0.5
    assert m.saving(a["cost"], s["cost"])["relative_saving"] == 0
    values = [10, 4, 6, 3]
    r = m.first_quality([0, 0.0625, 0.125, 0.25], values, 5)
    assert r["later_recrosses_target"] and values == [10, 4, 6, 3]


def test_quality_no_oracle_stop_missing_full_and_zero():
    assert (
        m.first_quality([0.0625, 0.125], np.mean([[1, 9], [9, 1]], axis=0), 2)["cost"]
        is None
    )
    assert m.first_quality([1], [41.69], 40)["cost"] is None
    assert m.first_quality([1], [41.69], 42)["cost"] == 1
    assert m.saving(None, 0.25)["relative_saving"] is None
    assert m.saving(0, 0)["relative_saving"] is None
    assert m.saving(0.25, 0.125)["relative_saving"] == -1


def test_full_original_order_and_labels_reject_permutation():
    rows = [
        {"specimen_key": "train", "split": "TRAIN"},
        {"specimen_key": "vB", "split": "VALID"},
        {"specimen_key": "vA", "split": "VALID"},
    ]
    expected = {"vA": 200.0, "vB": 300.0}
    mapped = m.map_full(rows, [2, 1], [30.0, 40.0], [200.0, 300.0], expected)
    assert [r["specimen_key"] for r in mapped] == ["vA", "vB"]
    assert [r["full_prediction_mpa"] for r in mapped] == [30.0, 40.0]
    with pytest.raises(ValueError):
        m.map_full(
            [rows[0], rows[2], rows[1]], [2, 1], [30.0, 40.0], [200.0, 300.0], expected
        )


def test_saved_trace_is_verified_not_reconstructed():
    import json

    from cmc_bbdm.cai_agent_v3.paper_evidence import validate_episode

    t = {
        "actor_call_index": 1,
        "action_index": 1,
        "cell": 0,
        "visible_cells_before": [],
        "environment_legal": "1" * 64,
        "proposal_legal": "1" * 64,
        "c0_reason": "fixture",
        "before_cost": 0.0,
        "after_cost": 0.125,
        "new_pixels": 1,
        "cumulative_pixels": 1,
        "before_prediction_mpa": 190.0,
        "after_prediction_mpa": 198.0,
    }
    row = {
        "costs": "0;.125",
        "predictions_mpa": "190;198",
        "cells": "0",
        "execution_trace": json.dumps([t]),
        "actor_state_dict_sha256": "synthetic",
        "native_pixels": "8",
    }
    assert validate_episode(row)["trace"][0]["actor_call_index"] == 1
    t["actor_call_index"] = 2
    row["execution_trace"] = json.dumps([t])
    with pytest.raises(ValueError):
        validate_episode(row)


def test_table_fragments_preserve_null_and_negative(tmp_path):
    from cmc_bbdm.cai_agent_v3.paper_evidence import table_fragments

    table_fragments(
        tmp_path,
        "table",
        [{"method": "NO_VLM", "gain": -1.5, "cost": None}],
        ["method", "gain", "cost"],
    )
    assert "| NO_VLM | -1.500000 | NA |" in (tmp_path / "table.md").read_text()
    assert r"NO\_VLM & -1.500000 & NA" in (tmp_path / "table.tex").read_text()
