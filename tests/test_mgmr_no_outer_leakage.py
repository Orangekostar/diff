from __future__ import annotations

import numpy as np

from cmc_bbdm.mgmr.evaluation import nested_lodo_predictions


def _run(target_shift: float):
    domains = tuple(f"d{index}" for index in range(6))
    dataset_ids = np.asarray([domain for domain in domains for _ in range(6)])
    specimen_ids = tuple(f"s{index:02d}" for index in range(len(dataset_ids)))
    rng = np.random.Generator(np.random.PCG64(20260823))
    metadata = rng.normal(size=(36, 2))
    block = rng.normal(size=(36, 10))
    targets = metadata[:, 0] + 0.1 * block[:, 0]
    targets[dataset_ids == "d5"] += target_shift
    return nested_lodo_predictions(
        method="B1",
        metadata=metadata,
        blocks={"coarse": block},
        targets=targets,
        specimen_ids=specimen_ids,
        dataset_ids=tuple(dataset_ids.tolist()),
        domain_order=domains,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )


def test_outer_labels_cannot_change_their_own_predictions() -> None:
    original = _run(0.0)
    shifted = _run(10000.0)
    mask = np.asarray(original.dataset_ids) == "d5"

    np.testing.assert_array_equal(original.predictions[mask], shifted.predictions[mask])
    assert original.selection_by_domain["d5"] == shifted.selection_by_domain["d5"]


def test_every_fit_excludes_its_query_domain() -> None:
    run = _run(0.0)

    for record in run.fit_records:
        assert set(record.fit_domains).isdisjoint(record.query_domains)
        assert set(record.fit_specimen_ids).isdisjoint(record.query_specimen_ids)
