from __future__ import annotations

import numpy as np

from cmc_bbdm.aei_multiview_regression.view_experts import fit_view_expert


def test_view_expert_is_fold_local_deterministic_and_immutable() -> None:
    rng = np.random.default_rng(20260821)
    embeddings = rng.normal(size=(18, 12))
    metadata = rng.normal(size=(18, 3))
    targets = 0.4 * metadata[:, 0] - 0.2 * embeddings[:, 0]
    fit = np.arange(12, dtype=np.int64)
    query = np.arange(12, 18, dtype=np.int64)

    first = fit_view_expert(
        embeddings,
        metadata,
        targets,
        fit,
        pca_dimension=4,
        alpha=10.0,
    )
    second = fit_view_expert(
        embeddings,
        metadata,
        targets,
        fit,
        pca_dimension=4,
        alpha=10.0,
    )

    np.testing.assert_array_equal(
        first.predict(embeddings[query], metadata[query]),
        second.predict(embeddings[query], metadata[query]),
    )
    assert first.components.flags.writeable is False
    assert first.coef.flags.writeable is False
    assert first.fit_indices == tuple(range(12))


def test_view_expert_ignores_query_targets() -> None:
    rng = np.random.default_rng(17)
    embeddings = rng.normal(size=(20, 10))
    metadata = rng.normal(size=(20, 2))
    targets = rng.normal(size=20)
    fit = np.arange(15, dtype=np.int64)
    query = np.arange(15, 20, dtype=np.int64)

    first = fit_view_expert(
        embeddings, metadata, targets, fit, pca_dimension=4, alpha=10.0
    )
    changed = targets.copy()
    changed[query] += 10_000.0
    second = fit_view_expert(
        embeddings, metadata, changed, fit, pca_dimension=4, alpha=10.0
    )

    np.testing.assert_array_equal(
        first.predict(embeddings[query], metadata[query]),
        second.predict(embeddings[query], metadata[query]),
    )
