from __future__ import annotations

import hashlib

import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.crossfit import build_crossfit_roster
from cmc_bbdm.inspection_agent_g1.dagger import (
    DaggerSourceError,
    DaggerVisitedState,
    equal_specimen_task_weights,
    select_source_relabels,
    trajectory_quantile_indices,
)
from cmc_bbdm.inspection_agent_g1.teacher import authorize_source_teacher

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _visited(
    *,
    source_domain: str,
    specimen: str,
    task: InspectionTask,
    iteration: int,
    count: int,
) -> tuple[DaggerVisitedState, ...]:
    return tuple(
        DaggerVisitedState(
            outer_target="d6",
            source_domain=source_domain,
            specimen_sha256=_sha(specimen),
            task=task,
            iteration=iteration,
            trajectory_index=index,
            trajectory_length=count,
            observation_sha256=_sha(f"observation-{specimen}-{task.value}-{index}"),
            policy_state_sha256=_sha(f"policy-{specimen}-{task.value}-{index}"),
            teacher_label_sha256=_sha(f"teacher-{specimen}-{task.value}-{index}"),
        )
        for index in range(count)
    )


def _authorization():
    roster = build_crossfit_roster(
        DOMAINS,
        outer_target="d6",
        labeled_domain="d1",
    )
    return authorize_source_teacher(roster, query_domain="d1")


def test_dagger_quantiles_are_deterministic_and_capped_at_sixteen() -> None:
    first = trajectory_quantile_indices(101)
    second = trajectory_quantile_indices(101)
    assert first == second
    assert len(first) == 16
    assert len(set(first)) == 16
    assert first == tuple(sorted(first))
    assert all(0 <= index < 101 for index in first)
    assert trajectory_quantile_indices(7) == tuple(range(7))


def test_dagger_relabel_bank_rejects_the_outer_target() -> None:
    rows = _visited(
        source_domain="d6",
        specimen="target-specimen",
        task=InspectionTask.FIELD,
        iteration=1,
        count=20,
    )
    with pytest.raises(DaggerSourceError, match="outer target"):
        select_source_relabels(_authorization(), rows)


def test_dagger_selects_only_authorized_source_quantiles() -> None:
    rows = _visited(
        source_domain="d1",
        specimen="source-specimen",
        task=InspectionTask.CAI,
        iteration=2,
        count=37,
    )
    selected = select_source_relabels(_authorization(), rows)
    assert len(selected) == 16
    assert tuple(row.trajectory_index for row in selected) == trajectory_quantile_indices(37)
    assert all(row.source_domain == "d1" for row in selected)
    assert all(row.outer_target == "d6" for row in selected)
    assert all(row.iteration == 2 for row in selected)


def test_dagger_weights_each_physical_specimen_and_task_equally() -> None:
    rows = (
        *select_source_relabels(
            _authorization(),
            _visited(
                source_domain="d1",
                specimen="s1",
                task=InspectionTask.FIELD,
                iteration=1,
                count=35,
            ),
        ),
        *select_source_relabels(
            _authorization(),
            _visited(
                source_domain="d1",
                specimen="s2",
                task=InspectionTask.CAI,
                iteration=1,
                count=4,
            ),
        ),
    )
    weights = equal_specimen_task_weights(rows)
    first_total = sum(
        weight
        for row, weight in zip(rows, weights, strict=True)
        if row.specimen_sha256 == _sha("s1")
    )
    second_total = sum(
        weight
        for row, weight in zip(rows, weights, strict=True)
        if row.specimen_sha256 == _sha("s2")
    )
    assert sum(weights) == pytest.approx(1.0)
    assert first_total == pytest.approx(0.5)
    assert second_total == pytest.approx(0.5)
