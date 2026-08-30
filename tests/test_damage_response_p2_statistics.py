from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS
from cmc_bbdm.damage_response.p2_evaluation import P2OOFPrediction
from cmc_bbdm.damage_response.p2_statistics import (
    FAMILYWISE_LOWER_QUANTILE,
    FAMILYWISE_UPPER_QUANTILE,
    P2_BOOTSTRAP_REPLICATES,
    P2_BOOTSTRAP_SEED,
    P2StatisticsError,
    analyze_p2_contrasts,
    synchronized_within_domain_bootstrap,
)

DOMAIN_ORDER = tuple(PRIMARY_COUNTS)
ENDPOINTS = (
    "extension_peak_mm",
    "slope_u20_u60_mpa_per_mm",
    "normalized_prepeak_auc",
)


def _prediction(
    specimen_id: str,
    domain_id: str,
    endpoint: str,
    view_name: str,
    truth: float,
    absolute_error: float,
) -> P2OOFPrediction:
    prediction = truth + absolute_error
    return P2OOFPrediction(
        specimen_id=specimen_id,
        domain_id=domain_id,
        held_out_domain=domain_id,
        endpoint=endpoint,
        view_name=view_name,
        truth=truth,
        prediction=prediction,
        absolute_error=absolute_error,
        standardized_absolute_error=absolute_error / 2.0,
        source_target_std=2.0,
        selected_ridge_alpha=1.0,
        selected_pca_dimension=(None if view_name == "F2" else 8),
        preprocessor_state_sha256="a" * 64,
        fold_state_sha256="b" * 64,
    )


def _predictions() -> tuple[P2OOFPrediction, ...]:
    rows: list[P2OOFPrediction] = []
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            for specimen_index in range(3):
                specimen_id = f"{domain[:2]}-{specimen_index}"
                truth = 10.0 + endpoint_index + specimen_index
                reference_error = 2.0 + 0.1 * specimen_index
                # F3 improves the first four domains by 20%, but worsens the last two.
                f3_factor = 0.8 if domain_index < 4 else 1.1
                # F4 is a uniform 5% improvement and therefore below the point gate.
                f4_factor = 0.95
                rows.extend(
                    (
                        _prediction(
                            specimen_id,
                            domain,
                            endpoint,
                            "F2",
                            truth,
                            reference_error,
                        ),
                        _prediction(
                            specimen_id,
                            domain,
                            endpoint,
                            "F3",
                            truth,
                            reference_error * f3_factor,
                        ),
                        _prediction(
                            specimen_id,
                            domain,
                            endpoint,
                            "F4",
                            truth,
                            reference_error * f4_factor,
                        ),
                    )
                )
    return tuple(rows)


def test_registered_bootstrap_constants_and_familywise_quantiles() -> None:
    assert P2_BOOTSTRAP_SEED == 20260830
    assert P2_BOOTSTRAP_REPLICATES == 100000
    assert FAMILYWISE_LOWER_QUANTILE == pytest.approx(0.025 / 6)
    assert FAMILYWISE_UPPER_QUANTILE == pytest.approx(1.0 - 0.025 / 6)


def test_bootstrap_reuses_exact_indices_across_all_contrasts() -> None:
    contrasts = {
        domain: np.asarray(
            [[1.0 + index, 2.0 * (1.0 + index)] for index in range(4)]
        )
        for domain in DOMAIN_ORDER
    }

    samples = synchronized_within_domain_bootstrap(
        contrasts, seed=17, replicates=37
    )
    expected = np.zeros((37, 2), dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(17))
    for domain in DOMAIN_ORDER:
        draws = generator.integers(0, 4, size=(37, 4), dtype=np.int64)
        expected += np.mean(contrasts[domain][draws], axis=1) / 6.0

    assert samples.shape == (37, 2)
    assert samples.flags.writeable is False
    np.testing.assert_allclose(samples, expected, rtol=1e-15, atol=0.0)
    np.testing.assert_array_equal(samples[:, 1], 2.0 * samples[:, 0])


def test_contrast_analysis_uses_specimen_bootstrap_then_equal_domain_aggregation() -> None:
    analysis = analyze_p2_contrasts(_predictions(), seed=19, replicates=5000)

    assert len(analysis.contrasts) == 9
    assert len([row for row in analysis.contrasts if row.primary_family]) == 6
    f3 = next(
        row
        for row in analysis.contrasts
        if row.endpoint == ENDPOINTS[0]
        and row.reference_view == "F2"
        and row.candidate_view == "F3"
    )
    f4 = next(
        row
        for row in analysis.contrasts
        if row.endpoint == ENDPOINTS[0]
        and row.reference_view == "F2"
        and row.candidate_view == "F4"
    )

    assert f3.improved_domain_count == 4
    assert f3.observed_reference_equal_domain_mae == pytest.approx(2.1)
    assert f3.observed_candidate_equal_domain_mae == pytest.approx(1.89)
    assert f3.observed_improvement == pytest.approx(0.21)
    assert f3.relative_improvement == pytest.approx(0.10)
    assert f3.familywise_interval is not None
    assert f4.relative_improvement == pytest.approx(0.05)
    secondary = next(
        row
        for row in analysis.contrasts
        if row.reference_view == "F3" and row.candidate_view == "F4"
    )
    assert secondary.primary_family is False
    assert secondary.familywise_interval is None
    expected = np.quantile(
        analysis.bootstrap_samples[:, f3.bootstrap_column],
        [FAMILYWISE_LOWER_QUANTILE, FAMILYWISE_UPPER_QUANTILE],
        method="linear",
    )
    np.testing.assert_array_equal(f3.familywise_interval, expected)


def test_contrast_analysis_is_byte_deterministic() -> None:
    first = analyze_p2_contrasts(_predictions(), seed=23, replicates=300)
    second = analyze_p2_contrasts(_predictions(), seed=23, replicates=300)

    assert first.synchronized_replicate_sha256 == second.synchronized_replicate_sha256
    assert first.contrasts == second.contrasts
    np.testing.assert_array_equal(first.bootstrap_samples, second.bootstrap_samples)


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "truth", "error"))
def test_contrast_prediction_membership_fails_closed(mutation: str) -> None:
    rows = list(_predictions())
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[-1])
    elif mutation == "truth":
        row = rows[-1]
        rows[-1] = replace(row, truth=row.truth + 1.0)
    else:
        row = rows[-1]
        rows[-1] = replace(row, absolute_error=row.absolute_error + 1.0)

    with pytest.raises(P2StatisticsError):
        analyze_p2_contrasts(tuple(rows), seed=29, replicates=30)


def test_bootstrap_domain_membership_fails_closed() -> None:
    contrasts = {
        domain: np.ones((2, 1), dtype=np.float64) for domain in DOMAIN_ORDER[:-1]
    }

    with pytest.raises(P2StatisticsError, match="six canonical domains"):
        synchronized_within_domain_bootstrap(contrasts, seed=1, replicates=10)
