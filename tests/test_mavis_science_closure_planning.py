from __future__ import annotations

import pytest

from cmc_bbdm.mavis.science_closure_planning import (
    PlanningCandidate,
    build_registered_substitutions,
    select_joint_utility_set,
)


def _candidate(name: str, cost: int, value: float) -> PlanningCandidate:
    return PlanningCandidate(key=name, exact_added_cost=cost, point_value=value)


def test_oracle_substitution_is_marked_non_deployable() -> None:
    rows = build_registered_substitutions(
        current_auebc=0.20,
        learned_lookahead_auebc=0.19,
        true_greedy_auebc=0.18,
        true_set_auebc=0.17,
        policy_checkpoint_sha256="a" * 64,
    )

    by_row = {row.row_id: row for row in rows}
    assert by_row["A"].deployable
    assert by_row["C"].deployable
    assert not by_row["B"].deployable
    assert not by_row["D"].deployable
    assert not by_row["E"].deployable


def test_oracle_substitution_does_not_modify_policy_checkpoint() -> None:
    checkpoint = "b" * 64
    rows = build_registered_substitutions(
        current_auebc=0.20,
        learned_lookahead_auebc=0.19,
        true_greedy_auebc=0.18,
        true_set_auebc=0.17,
        policy_checkpoint_sha256=checkpoint,
    )

    assert {row.policy_checkpoint_sha256 for row in rows} == {checkpoint}


def test_true_value_planner_respects_exact_budget() -> None:
    candidates = (
        _candidate("a", 4, 4.0),
        _candidate("b", 3, 3.0),
        _candidate("c", 2, 2.0),
    )

    selected = select_joint_utility_set(
        candidates,
        exact_budget=5,
        set_size=2,
        joint_utility=lambda keys: sum({"a": 4.0, "b": 3.0, "c": 2.0}[key] for key in keys),
    )

    assert selected.keys == ("b", "c")
    assert selected.exact_cost == 5


def test_set_planner_uses_joint_utility_not_sum_of_point_values() -> None:
    candidates = (
        _candidate("a", 1, 10.0),
        _candidate("b", 1, 9.0),
        _candidate("c", 1, 8.0),
    )
    utilities = {
        ("a", "b"): 1.0,
        ("a", "c"): 2.0,
        ("b", "c"): 5.0,
    }

    selected = select_joint_utility_set(
        candidates,
        exact_budget=2,
        set_size=2,
        joint_utility=lambda keys: utilities[tuple(sorted(keys))],
    )

    assert selected.keys == ("b", "c")
    assert selected.joint_utility == 5.0
    assert selected.point_value_sum == 17.0


def test_set_planner_breaks_ties_deterministically_without_duplicates() -> None:
    candidates = (
        _candidate("c", 1, 1.0),
        _candidate("a", 1, 1.0),
        _candidate("b", 1, 1.0),
    )

    first = select_joint_utility_set(
        candidates,
        exact_budget=2,
        set_size=2,
        joint_utility=lambda _keys: 1.0,
    )
    second = select_joint_utility_set(
        tuple(reversed(candidates)),
        exact_budget=2,
        set_size=2,
        joint_utility=lambda _keys: 1.0,
    )

    assert first == second
    assert first.keys == ("a", "b")
    assert len(set(first.keys)) == len(first.keys)


def test_set_planner_rejects_duplicate_candidate_keys() -> None:
    with pytest.raises(ValueError, match="duplicated"):
        select_joint_utility_set(
            (_candidate("a", 1, 1.0), _candidate("a", 1, 2.0)),
            exact_budget=2,
            set_size=2,
            joint_utility=lambda _keys: 1.0,
        )
