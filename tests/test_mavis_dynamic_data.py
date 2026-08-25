from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

from cmc_bbdm.mavis.dynamic_data import (
    build_dynamic_training_groups,
    build_target_evaluation_groups,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    states: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    domains = ("d0", "d1", "d2")
    for domain_index, domain in enumerate(domains):
        state_id = _sha(f"state-{domain}")
        states.append(
            {
                "state_id": state_id,
                "specimen_id": f"s-{domain}",
                "domain_id": domain,
                "exact_acquired_cost": 10,
                "native_count": 100,
                "remaining_cost_to_endpoint": 30,
                "candidate_cell_indices": [2, 7],
                "candidate_from_levels": [0, 1],
                "candidate_to_levels": [1, 2],
                "candidate_exact_added_costs": [3, 5],
            }
        )
        for outer in domains:
            if outer == domain:
                continue
            for candidate_index, cell_index in enumerate((2, 7)):
                current = 0.2 + 0.01 * domain_index + 0.02 * domains.index(outer)
                candidate = current + (-0.03 if candidate_index == 0 else 0.01)
                target = 0.3 + 0.01 * domain_index
                actions.append(
                    {
                        "state_id": state_id,
                        "specimen_id": f"s-{domain}",
                        "domain_id": domain,
                        "outer_domain": outer,
                        "candidate_index": candidate_index,
                        "cell_index": cell_index,
                        "from_level": candidate_index,
                        "to_level": candidate_index + 1,
                        "exact_added_cost": (3, 5)[candidate_index],
                        "teacher_true_cai": target,
                        "current_prediction": current,
                        "candidate_prediction": candidate,
                        "primary_value": abs(target - current)
                        - abs(target - candidate),
                        "teacher_state_sha256": _sha(
                            f"teacher-{domain}-{outer}"
                        ),
                    }
                )
    return pl.DataFrame(states), pl.DataFrame(actions)


def test_dynamic_training_groups_are_outer_target_isolated() -> None:
    states, actions = _tables()
    original = build_dynamic_training_groups(
        states,
        actions,
        outer_domain="d0",
    )
    mutated = actions.with_columns(
        pl.when(pl.col("domain_id") == "d0")
        .then(pl.lit(99.0))
        .otherwise(pl.col("teacher_true_cai"))
        .alias("teacher_true_cai")
    )
    changed = build_dynamic_training_groups(
        states,
        mutated,
        outer_domain="d0",
    )

    assert {group.domain_id for group in original} == {"d1", "d2"}
    assert all(group.outer_domain == "d0" for group in original)
    assert [group.state_sha256 for group in changed] == [
        group.state_sha256 for group in original
    ]


def test_target_evaluation_aggregates_only_strict_oof_predictions() -> None:
    states, actions = _tables()
    groups = build_target_evaluation_groups(
        states,
        actions,
        target_domain="d0",
    )

    assert len(groups) == 1
    group = groups[0]
    assert group.domain_id == "d0"
    assert group.outer_domain == "d0"
    assert group.teacher_fold_count == 2
    np.testing.assert_allclose(
        group.teacher_values,
        abs(group.true_cai - group.current_prediction)
        - np.abs(group.true_cai - group.candidate_predictions),
    )
    assert group.teacher_outer_domains == ("d1", "d2")
