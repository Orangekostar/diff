from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS, StageStatus
from cmc_bbdm.damage_response.feature_views import PRIMARY_TARGET_FIELDS
from cmc_bbdm.damage_response.p1_gate import (
    ENDPOINT_RANGE_TOLERANCES,
    EndpointGateFacts,
    GateError,
    evaluate_p1_gate,
)

DOMAINS = tuple(PRIMARY_COUNTS)


def _passing_endpoint(endpoint: str) -> EndpointGateFacts:
    return EndpointGateFacts(
        endpoint=endpoint,
        valid_count=276,
        valid_domain_counts=MappingProxyType(dict(PRIMARY_COUNTS)),
        strength_only_pooled_r2=0.89,
        strength_only_domain_spearman=MappingProxyType(
            {domain: 0.99 for domain in DOMAINS}
        ),
        descriptor_range=20.0 * ENDPOINT_RANGE_TOLERANCES[endpoint],
        range_tolerance=ENDPOINT_RANGE_TOLERANCES[endpoint],
        replay_byte_identical=True,
    )


def _facts() -> tuple[EndpointGateFacts, ...]:
    return tuple(_passing_endpoint(endpoint) for endpoint in PRIMARY_TARGET_FIELDS)


def test_p1_gate_is_go_when_any_endpoint_passes() -> None:
    failing = replace(
        _passing_endpoint(PRIMARY_TARGET_FIELDS[0]),
        valid_count=247,
        valid_domain_counts=MappingProxyType(
            {**dict(PRIMARY_COUNTS), DOMAINS[0]: PRIMARY_COUNTS[DOMAINS[0]] - 29}
        ),
    )
    facts = (failing, *_facts()[1:])

    decision = evaluate_p1_gate(facts)

    assert decision.status is StageStatus.P1_GO
    assert decision.passing_endpoints == PRIMARY_TARGET_FIELDS[1:]
    assert decision.endpoint_decisions[0].passed is False
    assert "coverage" in decision.endpoint_decisions[0].reasons[0]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        (
            {
                "valid_count": 247,
                "valid_domain_counts": MappingProxyType(
                    {
                        **dict(PRIMARY_COUNTS),
                        DOMAINS[0]: PRIMARY_COUNTS[DOMAINS[0]] - 29,
                    }
                ),
            },
            "coverage",
        ),
        (
            {
                "valid_count": 231,
                "valid_domain_counts": MappingProxyType(
                    {**dict(PRIMARY_COUNTS), DOMAINS[0]: 0}
                ),
            },
            "six domains",
        ),
        (
            {
                "strength_only_pooled_r2": 0.90,
                "strength_only_domain_spearman": MappingProxyType(
                    {
                        domain: 0.94 if index < 3 else 0.95
                        for index, domain in enumerate(DOMAINS)
                    }
                ),
            },
            "near-determined",
        ),
        (
            {"descriptor_range": 10.0 * 0.001},
            "range",
        ),
        (
            {"replay_byte_identical": False},
            "replay",
        ),
    ),
)
def test_each_endpoint_gate_reason_fails_independently(
    mutation: dict[str, object], reason: str
) -> None:
    endpoint = PRIMARY_TARGET_FIELDS[0]
    facts = list(_facts())
    facts[0] = replace(facts[0], **mutation)
    facts[1] = replace(facts[1], replay_byte_identical=False)
    facts[2] = replace(facts[2], replay_byte_identical=False)

    decision = evaluate_p1_gate(tuple(facts))

    assert decision.status is StageStatus.RESPONSE_BEYOND_STRENGTH_NO_GO
    endpoint_decision = next(
        item for item in decision.endpoint_decisions if item.endpoint == endpoint
    )
    assert endpoint_decision.passed is False
    assert any(reason in item for item in endpoint_decision.reasons)


def test_redundancy_condition_passes_by_four_low_domain_correlations() -> None:
    endpoint = PRIMARY_TARGET_FIELDS[0]
    facts = list(_facts())
    facts[0] = replace(
        facts[0],
        strength_only_pooled_r2=0.95,
        strength_only_domain_spearman=MappingProxyType(
            {
                domain: 0.94 if index < 4 else 0.99
                for index, domain in enumerate(DOMAINS)
            }
        ),
    )

    decision = evaluate_p1_gate(tuple(facts))

    endpoint_decision = next(
        item for item in decision.endpoint_decisions if item.endpoint == endpoint
    )
    assert endpoint_decision.passed is True
    assert endpoint in decision.passing_endpoints


def test_p1_gate_rejects_missing_endpoint_or_domain_metric() -> None:
    with pytest.raises(GateError, match="three primary endpoints"):
        evaluate_p1_gate(_facts()[:-1])

    facts = list(_facts())
    facts[0] = replace(
        facts[0],
        strength_only_domain_spearman=MappingProxyType(
            {
                domain: value
                for domain, value in facts[0].strength_only_domain_spearman.items()
                if domain != DOMAINS[-1]
            }
        ),
    )
    with pytest.raises(GateError, match="domain Spearman"):
        evaluate_p1_gate(tuple(facts))


def test_p1_gate_thresholds_are_frozen() -> None:
    assert ENDPOINT_RANGE_TOLERANCES == {
        "extension_peak_mm": 0.001,
        "slope_u20_u60_mpa_per_mm": 1.0,
        "normalized_prepeak_auc": 0.0001,
    }
    facts = list(_facts())
    facts[0] = replace(facts[0], range_tolerance=0.002)
    with pytest.raises(GateError, match="tolerance"):
        evaluate_p1_gate(tuple(facts))
