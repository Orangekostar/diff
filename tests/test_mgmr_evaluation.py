from __future__ import annotations

import numpy as np

from cmc_bbdm.mgmr.evaluation import nested_lodo_predictions


def _problem() -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray
]:
    domains = tuple(f"d{index}" for index in range(6))
    dataset_ids = tuple(domain for domain in domains for _ in range(6))
    specimen_ids = tuple(f"s{index:02d}" for index in range(len(dataset_ids)))
    rng = np.random.Generator(np.random.PCG64(20260822))
    metadata = rng.normal(size=(36, 3))
    block = rng.normal(size=(36, 12))
    targets = 0.4 * metadata[:, 0] - 0.2 * block[:, 0] + 0.05 * block[:, 3]
    return domains, dataset_ids, specimen_ids, metadata, block, targets


def test_nested_lodo_is_deterministic_and_selects_dimensions() -> None:
    domains, dataset_ids, specimen_ids, metadata, block, targets = _problem()

    first = nested_lodo_predictions(
        method="B1",
        metadata=metadata,
        blocks={"coarse": block},
        targets=targets,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domains,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )
    second = nested_lodo_predictions(
        method="B1",
        metadata=metadata,
        blocks={"coarse": block},
        targets=targets,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domains,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    np.testing.assert_array_equal(first.predictions, second.predictions)
    assert len(first.selections) == 6
    assert len(first.inner_scores) == 6 * 2 * 5
    assert all(selection.dimensions in {(2,), (4,)} for selection in first.selections)
    assert all(np.isfinite(record.prediction) for record in first.records)


def test_dimension_ties_prefer_lower_total_then_lexicographic() -> None:
    domains, dataset_ids, specimen_ids, metadata, _block, targets = _problem()
    zero = np.zeros((36, 8), dtype=np.float64)

    run = nested_lodo_predictions(
        method="B3",
        metadata=metadata,
        blocks={"coarse": zero, "boundary": zero.copy()},
        targets=targets,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domains,
        pca_dimensions=(1, 2),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    assert all(selection.dimensions == (1, 1) for selection in run.selections)


def test_pca_reuses_the_maximum_fit_for_identical_training_rows(monkeypatch) -> None:
    domains, dataset_ids, specimen_ids, metadata, block, targets = _problem()
    original = np.linalg.svd
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted)
    nested_lodo_predictions(
        method="B1",
        metadata=metadata,
        blocks={"coarse": block},
        targets=targets,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domains,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    # Six five-domain fits plus the 15 unique four-domain combinations.
    assert calls == 21
