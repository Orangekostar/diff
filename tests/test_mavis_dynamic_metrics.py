from __future__ import annotations

import numpy as np
import polars as pl

from cmc_bbdm.mavis.dynamic_data import DynamicStateGroup
from cmc_bbdm.mavis.dynamic_metrics import (
    aggregate_dynamic_metrics,
    bootstrap_dynamic_contrasts,
    evaluate_dynamic_scores,
)
from cmc_bbdm.mavis.dynamic_voi import CandidateDescriptor


def _group(specimen: str, values: list[float]) -> DynamicStateGroup:
    candidates = tuple(
        CandidateDescriptor(
            cell_index=index,
            from_level=0,
            to_level=1,
            exact_added_cost=5,
            native_count=100,
            remaining_cost=20,
        )
        for index in range(len(values))
    )
    teacher = np.asarray(values, dtype=np.float64)
    teacher.setflags(write=False)
    predictions = np.asarray(values, dtype=np.float64)
    predictions.setflags(write=False)
    return DynamicStateGroup(
        state_id=f"state-{specimen}",
        specimen_id=specimen,
        domain_id="d0",
        outer_domain="d0",
        candidates=candidates,
        true_cai=0.5,
        current_prediction=0.6,
        candidate_predictions=predictions,
        teacher_values=teacher,
        teacher_outer_domains=("d1", "d2"),
        teacher_fold_count=2,
        state_sha256=f"hash-{specimen}",
    )


def test_dynamic_metrics_use_state_then_specimen_then_domain_units() -> None:
    groups = (_group("s0", [0.4, 0.2, -0.1]), _group("s1", [0.3, 0.1, 0.0]))
    scores = (np.asarray([3.0, 2.0, 1.0]), np.asarray([0.0, 1.0, 2.0]))

    per_state = evaluate_dynamic_scores(groups, scores, mode="real", recall_k=2)
    tables = aggregate_dynamic_metrics(per_state)

    first = per_state.filter(per_state["specimen_id"] == "s0").row(0, named=True)
    assert first["next_action_regret"] == 0.0
    assert first["one_step_cai_utility"] == 0.4
    np.testing.assert_allclose(first["spearman"], 1.0, atol=1.0e-15)
    assert first["recall_at_k"] == 1.0
    assert tables.per_specimen.height == 2
    assert tables.per_domain.height == 1
    assert tables.aggregate.height == 1
    assert tables.aggregate.row(0, named=True)["statistical_unit"] == "equal_domain"


def test_dynamic_bootstrap_resamples_specimens_within_domains() -> None:
    groups = (_group("s0", [0.4, 0.2, -0.1]), _group("s1", [0.3, 0.1, 0.0]))
    real = evaluate_dynamic_scores(
        groups,
        (np.asarray([3.0, 2.0, 1.0]), np.asarray([3.0, 2.0, 1.0])),
        mode="real",
        recall_k=2,
    )
    shuffled = evaluate_dynamic_scores(
        groups,
        (np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])),
        mode="shuffled",
        recall_k=2,
    )
    per_specimen = aggregate_dynamic_metrics(
        pl.concat([real, shuffled])
    ).per_specimen
    bootstrap = bootstrap_dynamic_contrasts(
        per_specimen,
        reference_mode="real",
        control_modes=("shuffled",),
        domain_order=("d0",),
        replicates=10,
        seed=20260825,
    )

    assert bootstrap.height == 10
    assert bootstrap.get_column("control_minus_reference_regret").min() >= 0.0
