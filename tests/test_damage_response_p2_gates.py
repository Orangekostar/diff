from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS, StageStatus
from cmc_bbdm.damage_response.p2_gate import (
    P2GateError,
    evaluate_p2_gate,
)
from cmc_bbdm.damage_response.p2_statistics import (
    P2ContrastAnalysis,
    P2ContrastResult,
)

ENDPOINTS = (
    "extension_peak_mm",
    "slope_u20_u60_mpa_per_mm",
    "normalized_prepeak_auc",
)
DOMAIN_ORDER = tuple(PRIMARY_COUNTS)


def _contrast(
    endpoint: str,
    candidate: str,
    *,
    relative: float,
    improved_domains: int,
    lower: float,
    primary: bool = True,
) -> P2ContrastResult:
    reference = "F2" if primary else "F3"
    if improved_domains == 6:
        domain_values = (relative,) * 6
    else:
        positive = 6.0 * relative / improved_domains + 0.1
        negative = -0.1 * improved_domains / (6 - improved_domains)
        domain_values = (positive,) * improved_domains + (negative,) * (
            6 - improved_domains
        )
    return P2ContrastResult(
        name=f"{endpoint}__{candidate}_vs_{reference}",
        endpoint=endpoint,
        reference_view=reference,
        candidate_view=candidate,
        primary_family=primary,
        observed_reference_equal_domain_mae=1.0,
        observed_candidate_equal_domain_mae=1.0 - relative,
        observed_improvement=relative,
        relative_improvement=relative,
        improved_domain_count=improved_domains,
        domain_improvements=tuple(zip(DOMAIN_ORDER, domain_values, strict=True)),
        bootstrap_mean=relative,
        ordinary_interval=(lower, relative + 0.01),
        familywise_interval=((lower, relative + 0.02) if primary else None),
        probability_positive=1.0,
        bootstrap_column=0,
        replicate_sha256="a" * 64,
    )


def _analysis(
    *, passing_endpoint: str | None = None, passing_candidate: str = "F3"
) -> P2ContrastAnalysis:
    rows = []
    for endpoint in ENDPOINTS:
        for candidate in ("F3", "F4"):
            passed = endpoint == passing_endpoint and candidate == passing_candidate
            rows.append(
                replace(
                    _contrast(
                    endpoint,
                    candidate,
                    relative=0.10 if passed else 0.099,
                    improved_domains=4 if passed else 3,
                    lower=1e-12 if passed else 0.0,
                    ),
                    bootstrap_column=len(rows),
                )
            )
    for endpoint in ENDPOINTS:
        rows.append(
            replace(
                _contrast(
                    endpoint,
                    "F4",
                    relative=0.01,
                    improved_domains=3,
                    lower=-0.01,
                    primary=False,
                ),
                bootstrap_column=len(rows),
            )
        )
    return P2ContrastAnalysis(
        seed=20260830,
        replicates=100000,
        contrasts=tuple(rows),
        synchronized_replicate_sha256="b" * 64,
        bootstrap_samples=np.zeros((1, 9), dtype=np.float64),
    )


def test_any_primary_endpoint_contrast_can_authorize_p2() -> None:
    decision = evaluate_p2_gate(
        _analysis(passing_endpoint=ENDPOINTS[1], passing_candidate="F3")
    )

    assert decision.status is StageStatus.P2_GO
    assert decision.passing_contrasts == ((ENDPOINTS[1], "F3", "F2"),)
    passed = next(row for row in decision.contrast_decisions if row.passed)
    assert passed.point_threshold_passed is True
    assert passed.domain_count_passed is True
    assert passed.familywise_interval_passed is True


def test_all_primary_contrasts_fail_with_registered_negative_status() -> None:
    decision = evaluate_p2_gate(_analysis())

    assert decision.status is StageStatus.MACK_EXTENSION_NO_GO
    assert decision.passing_contrasts == ()
    assert all(not row.passed for row in decision.contrast_decisions)


@pytest.mark.parametrize(
    ("relative", "domains", "lower", "failed_field"),
    (
        (0.10 - 1e-12, 4, 1e-9, "point_threshold_passed"),
        (0.10, 3, 1e-9, "domain_count_passed"),
        (0.10, 4, 0.0, "familywise_interval_passed"),
    ),
)
def test_each_gate_boundary_is_independently_required(
    relative: float, domains: int, lower: float, failed_field: str
) -> None:
    analysis = _analysis(passing_endpoint=ENDPOINTS[0])
    rows = list(analysis.contrasts)
    index = next(
        i
        for i, row in enumerate(rows)
        if row.endpoint == ENDPOINTS[0]
        and row.candidate_view == "F3"
        and row.primary_family
    )
    rows[index] = replace(
        rows[index],
        observed_candidate_equal_domain_mae=1.0 - relative,
        observed_improvement=relative,
        relative_improvement=relative,
        improved_domain_count=domains,
        domain_improvements=_contrast(
            ENDPOINTS[0],
            "F3",
            relative=relative,
            improved_domains=domains,
            lower=lower,
        ).domain_improvements,
        familywise_interval=(lower, 0.2),
    )

    decision = evaluate_p2_gate(replace(analysis, contrasts=tuple(rows)))
    row = next(
        value
        for value in decision.contrast_decisions
        if value.endpoint == ENDPOINTS[0] and value.candidate_view == "F3"
    )

    assert decision.status is StageStatus.MACK_EXTENSION_NO_GO
    assert getattr(row, failed_field) is False
    assert row.passed is False


def test_secondary_contrasts_cannot_authorize_p2() -> None:
    analysis = _analysis()
    rows = tuple(
        replace(
            row,
            observed_candidate_equal_domain_mae=0.5,
            observed_improvement=0.5,
            relative_improvement=0.50,
            improved_domain_count=6,
            domain_improvements=tuple((domain, 0.5) for domain in DOMAIN_ORDER),
            ordinary_interval=(0.1, 0.9),
        )
        if not row.primary_family
        else row
        for row in analysis.contrasts
    )

    decision = evaluate_p2_gate(replace(analysis, contrasts=rows))

    assert decision.status is StageStatus.MACK_EXTENSION_NO_GO


def test_gate_requires_exact_six_primary_contrasts() -> None:
    analysis = _analysis()

    with pytest.raises(P2GateError):
        evaluate_p2_gate(replace(analysis, contrasts=analysis.contrasts[1:]))
