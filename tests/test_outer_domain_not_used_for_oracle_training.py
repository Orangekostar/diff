from __future__ import annotations

import numpy as np

from cmc_bbdm.mva.crossfit import fit_outer_source_predictor

DOMAINS = tuple(f"d{index}" for index in range(6))


def _arrays() -> tuple[
    tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(13)
    ids = tuple(f"{domain}-{row}" for domain in DOMAINS for row in range(9))
    domains = tuple(domain for domain in DOMAINS for _ in range(9))
    metadata = rng.normal(size=(54, 3))
    embeddings = rng.normal(size=(54, 10))
    targets = rng.normal(size=54)
    return ids, domains, targets, metadata, embeddings


def test_outer_data_perturbation_cannot_change_oracle_predictor_fit() -> None:
    ids, domains, targets, metadata, embeddings = _arrays()
    target_domain = DOMAINS[2]
    mask = np.asarray(domains, dtype=object) == target_domain
    changed_targets = targets.copy()
    changed_metadata = metadata.copy()
    changed_embeddings = embeddings.copy()
    changed_targets[mask] += 10_000.0
    changed_metadata[mask] -= 20_000.0
    changed_embeddings[mask] *= -30_000.0

    first = fit_outer_source_predictor(
        method="P-A",
        outer_domain=target_domain,
        specimen_ids=ids,
        dataset_ids=domains,
        domain_order=DOMAINS,
        targets=targets,
        metadata=metadata,
        embeddings=embeddings,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )
    second = fit_outer_source_predictor(
        method="P-A",
        outer_domain=target_domain,
        specimen_ids=ids,
        dataset_ids=domains,
        domain_order=DOMAINS,
        targets=changed_targets,
        metadata=changed_metadata,
        embeddings=changed_embeddings,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    assert first.model.state_sha256 == second.model.state_sha256
    assert first.selected_pca_dimension == second.selected_pca_dimension
    assert first.inner_dimension_mae == second.inner_dimension_mae
