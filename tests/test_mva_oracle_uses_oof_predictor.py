from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mva.crossfit import fit_cross_fitted_evaluator

DOMAINS = tuple(f"d{index}" for index in range(6))


def _fixture() -> tuple[
    tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(20260823)
    specimen_ids = tuple(f"{domain}-{row}" for domain in DOMAINS for row in range(10))
    dataset_ids = tuple(domain for domain in DOMAINS for _ in range(10))
    metadata = rng.normal(size=(60, 2))
    embeddings = rng.normal(size=(60, 12))
    targets = 0.6 + 0.05 * metadata[:, 0] + 0.03 * embeddings[:, 0]
    return specimen_ids, dataset_ids, targets, metadata, embeddings


def test_every_oracle_label_uses_a_predictor_that_excludes_its_domain() -> None:
    specimen_ids, dataset_ids, targets, metadata, embeddings = _fixture()
    evaluator = fit_cross_fitted_evaluator(
        method="P-A",
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=DOMAINS,
        targets=targets,
        metadata=metadata,
        embeddings=embeddings,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    assert evaluator.predictions.shape == (60,)
    assert np.all(np.isfinite(evaluator.predictions))
    for domain in DOMAINS:
        model = evaluator.models[domain]
        assert domain not in model.fit_domains
        assert set(model.fit_domains) == set(DOMAINS) - {domain}
    for specimen_id, domain, prediction in zip(
        specimen_ids, dataset_ids, evaluator.predictions, strict=True
    ):
        index = specimen_ids.index(specimen_id)
        model = evaluator.models[domain]
        replay = model.predict(
            metadata[index : index + 1], embeddings[index : index + 1]
        )
        assert replay[0] == pytest.approx(prediction, abs=1.0e-15)


def test_fit_audit_records_all_inner_and_outer_information_barriers() -> None:
    specimen_ids, dataset_ids, targets, metadata, embeddings = _fixture()
    evaluator = fit_cross_fitted_evaluator(
        method="P-A",
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=DOMAINS,
        targets=targets,
        metadata=metadata,
        embeddings=embeddings,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    assert len(evaluator.fit_audits) == 66
    for row in evaluator.fit_audits:
        assert set(row.query_domains).isdisjoint(row.fit_domains)
        assert set(row.query_specimen_ids).isdisjoint(row.fit_specimen_ids)
        assert row.outer_domain not in row.fit_domains
