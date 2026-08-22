from __future__ import annotations

import numpy as np

from cmc_bbdm.mgmr.evaluation import nested_lodo_predictions


def test_each_specimen_has_exactly_one_outer_prediction() -> None:
    domains = ("a", "b", "c", "d", "e", "f")
    dataset_ids = tuple(domain for domain in domains for _ in range(6))
    specimen_ids = tuple(f"{domain}-{row}" for domain in domains for row in range(6))
    rng = np.random.Generator(np.random.PCG64(20260824))
    metadata = rng.normal(size=(36, 2))
    block = rng.normal(size=(36, 8))
    targets = rng.normal(size=36)

    run = nested_lodo_predictions(
        method="B2",
        metadata=metadata,
        blocks={"boundary": block},
        targets=targets,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domains,
        pca_dimensions=(2,),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    assert tuple(record.specimen_id for record in run.records) == specimen_ids
    assert tuple(record.dataset_id for record in run.records) == dataset_ids
    assert len({record.specimen_id for record in run.records}) == len(specimen_ids)
    np.testing.assert_array_equal(
        run.predictions,
        np.asarray([record.prediction for record in run.records]),
    )
