from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mvd.observability_dataset import ObservedValueExamples
from cmc_bbdm.mvd.observability_models import fit_ridge_scorer


def _examples(domains: tuple[str, ...]) -> ObservedValueExamples:
    count = len(domains)
    return ObservedValueExamples(
        outer_domain="outer",
        role="source_train",
        specimen_ids=tuple(f"s{index}" for index in range(count)),
        dataset_ids=domains,
        initial_embeddings=np.zeros((count, 512)),
        current_predictions=np.zeros(count),
        candidate_features=np.zeros((count, 64, 8)),
        initial_used_budgets=np.full(count, 0.02),
        mechanical_values=np.tile(np.linspace(-1.0, 1.0, 64), (count, 1)),
        candidate_costs=np.ones((count, 64), dtype=np.int64),
        teacher_predictor_state_sha256=("a" * 64,) * count,
        candidate_bank_state_sha256="b" * 64,
        observed_feature_state_sha256="c" * 64,
        state_sha256="d" * 64,
    )


def test_mvd_outer_domain_never_trains_observability_model() -> None:
    clean = _examples(("d0", "d1", "d2", "d3", "d4"))
    model = fit_ridge_scorer(
        clean, outer_domain="outer", mode="global_candidate", alpha=1.0
    )
    assert "outer" not in model.fit_domains

    leaked = _examples(("d0", "d1", "d2", "d3", "outer"))
    with pytest.raises(ValueError, match="outer domain"):
        fit_ridge_scorer(
            leaked, outer_domain="outer", mode="global_candidate", alpha=1.0
        )
