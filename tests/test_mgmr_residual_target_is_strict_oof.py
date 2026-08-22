from __future__ import annotations

import numpy as np

from cmc_bbdm.mgmr.m0_residual_audit import audit_residual_arrays


def _problem():
    domains = tuple(f"d{index}" for index in range(6))
    dataset_ids = tuple(domain for domain in domains for _ in range(6))
    specimen_ids = tuple(f"s{index:02d}" for index in range(36))
    rng = np.random.Generator(np.random.PCG64(20260827))
    metadata = rng.normal(size=(36, 2))
    full = rng.normal(size=(36, 8))
    coarse = rng.normal(size=(36, 8))
    boundary = rng.normal(size=(36, 10))
    targets = 0.5 * metadata[:, 0] + 0.3 * coarse[:, 0] + 0.2 * boundary[:, 0]
    coarse_outer = targets + rng.normal(scale=0.2, size=36)
    full_outer = targets + rng.normal(scale=0.1, size=36)
    shuffles = {
        seed: rng.normal(size=(36, 10))
        for seed in (20260831, 20260901, 20260902)
    }
    return (
        domains,
        dataset_ids,
        specimen_ids,
        metadata,
        full,
        coarse,
        boundary,
        targets,
        coarse_outer,
        full_outer,
        shuffles,
    )


def test_source_residual_targets_are_strict_domain_oof() -> None:
    (
        domains,
        dataset_ids,
        specimen_ids,
        metadata,
        full,
        coarse,
        boundary,
        targets,
        coarse_outer,
        full_outer,
        shuffles,
    ) = _problem()

    audit = audit_residual_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domains,
        targets=targets,
        metadata=metadata,
        full=full,
        coarse=coarse,
        boundary=boundary,
        coarse_outer_predictions=coarse_outer,
        full_outer_predictions=full_outer,
        shuffled_boundary=shuffles,
        pca_dimensions=(2,),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    assert {row.branch for row in audit.source_residuals} == {"coarse", "full"}
    for row in audit.source_residuals:
        assert row.dataset_id not in row.baseline_fit_domains
        assert row.specimen_id not in row.baseline_fit_specimen_ids
        assert row.residual == row.target - row.baseline_prediction
    for branch in (audit.coarse, audit.full, *audit.shuffles.values()):
        for row in branch.outer_records:
            assert row.outer_domain not in row.residual_fit_domains
            assert set(row.specimen_ids).isdisjoint(row.residual_fit_specimen_ids)
            assert row.residual_ridge_feature_count == row.selected_dimension


def test_full_correction_uses_exact_supplied_b0_outer_predictions() -> None:
    values = list(_problem())
    audit = audit_residual_arrays(
        specimen_ids=values[2],
        dataset_ids=values[1],
        domain_order=values[0],
        targets=values[7],
        metadata=values[3],
        full=values[4],
        coarse=values[5],
        boundary=values[6],
        coarse_outer_predictions=values[8],
        full_outer_predictions=values[9],
        shuffled_boundary=values[10],
        pca_dimensions=(2,),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    np.testing.assert_array_equal(audit.full.baseline_predictions, values[9])
    np.testing.assert_array_equal(
        audit.full.corrected_predictions,
        audit.full.baseline_predictions + audit.full.corrections,
    )
