from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cmc_bbdm.mavis.value_stability import (
    ValueStabilityError,
    fit_outer_value_learners,
    validate_shared_value_bank,
)


def test_value_stability_each_learner_is_strict_oof() -> None:
    domains = ("d0", "d1", "d2", "d3")
    specimen_ids = tuple(f"s{index}" for index in range(16))
    dataset_ids = tuple(domains[index // 4] for index in range(16))
    rng = np.random.default_rng(20260826)
    metadata = rng.normal(size=(16, 3))
    embeddings = rng.normal(size=(16, 10))
    targets = 0.4 * metadata[:, 0] - 0.2 * embeddings[:, 0]

    learners = fit_outer_value_learners(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domains,
        targets=targets,
        metadata=metadata,
        embeddings=embeddings,
        outer_domain="d3",
        pca_dimensions=(2,),
        seed=20260826,
    )

    assert set(learners) == {"ridge", "huber", "shallow_mlp"}
    query_ids = set(specimen_ids[-4:])
    for learner in learners.values():
        assert "d3" not in learner.fit_domains
        assert query_ids.isdisjoint(learner.fit_specimen_ids)


def test_value_stability_same_split_same_state_bank() -> None:
    rows = []
    for learner in ("ridge", "huber", "shallow_mlp"):
        for cell_index in range(3):
            rows.append(
                {
                    "specimen_id": "s0",
                    "dataset_id": "d0",
                    "learner": learner,
                    "cell_index": cell_index,
                    "initial_budget": 0.03125,
                    "added_measurements": 10 + cell_index,
                    "native_count": 1000,
                    "candidate_bank_state_sha256": "a" * 64,
                }
            )
    frame = pl.DataFrame(rows)

    validate_shared_value_bank(frame)

    drifted = frame.with_columns(
        pl.when((pl.col("learner") == "huber") & (pl.col("cell_index") == 1))
        .then(pl.lit(99))
        .otherwise(pl.col("added_measurements"))
        .alias("added_measurements")
    )
    with pytest.raises(ValueStabilityError, match="shared state/action bank"):
        validate_shared_value_bank(drifted)
