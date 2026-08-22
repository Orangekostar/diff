from __future__ import annotations

import numpy as np

from cmc_bbdm.mgmr.m0_components import evaluate_component_arrays


def test_component_evaluation_has_the_frozen_b0_to_b4_roster() -> None:
    domains = tuple(f"d{index}" for index in range(6))
    dataset_ids = tuple(domain for domain in domains for _ in range(6))
    specimen_ids = tuple(f"s{index:02d}" for index in range(36))
    rng = np.random.Generator(np.random.PCG64(20260825))
    metadata = rng.normal(size=(36, 3))
    full = rng.normal(size=(36, 10))
    coarse = rng.normal(size=(36, 8))
    boundary = rng.normal(size=(36, 12))
    targets = 0.5 * metadata[:, 0] + 0.2 * coarse[:, 0]
    baseline = targets + rng.normal(scale=0.1, size=36)

    result = evaluate_component_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domains,
        targets=targets,
        metadata=metadata,
        full=full,
        coarse=coarse,
        boundary=boundary,
        baseline_predictions=baseline,
        baseline_dimensions=(2, 2, 2, 2, 2, 2),
        pca_dimensions=(2,),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    assert result.methods == ("B0", "B1", "B2", "B3", "B4")
    assert tuple(result.runs) == ("B1", "B2", "B3", "B4")
    np.testing.assert_array_equal(result.predictions["B0"], baseline)
    for method in result.methods:
        assert result.predictions[method].shape == (36,)
        assert result.predictions[method].flags.writeable is False
    for run in result.runs.values():
        assert all(record.outer_domain not in record.fit_domains for record in run.fit_records)


def test_metadata_is_concatenated_once_for_multiblock_models() -> None:
    domains = tuple(f"d{index}" for index in range(6))
    dataset_ids = tuple(domain for domain in domains for _ in range(5))
    specimen_ids = tuple(f"s{index:02d}" for index in range(30))
    rng = np.random.Generator(np.random.PCG64(20260826))
    metadata = rng.normal(size=(30, 2))
    full = rng.normal(size=(30, 4))
    coarse = rng.normal(size=(30, 4))
    boundary = rng.normal(size=(30, 4))
    targets = metadata[:, 0]

    result = evaluate_component_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domains,
        targets=targets,
        metadata=metadata,
        full=full,
        coarse=coarse,
        boundary=boundary,
        baseline_predictions=targets,
        baseline_dimensions=(1,) * 6,
        pca_dimensions=(1,),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    outer_b3 = next(row for row in result.runs["B3"].fit_records if row.stage == "outer")
    outer_b4 = next(row for row in result.runs["B4"].fit_records if row.stage == "outer")
    # Ridge sees metadata width plus one component per independent block.
    assert outer_b3.ridge_feature_count == 4
    assert outer_b4.ridge_feature_count == 4
