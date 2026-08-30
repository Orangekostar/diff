"""Preregistered P1 response-richness gate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS, StageStatus
from cmc_bbdm.damage_response.feature_views import PRIMARY_TARGET_FIELDS

PRIMARY_COHORT_SIZE = sum(PRIMARY_COUNTS.values())
MINIMUM_COVERAGE_FRACTION = 0.90
NEAR_DETERMINISTIC_POOLED_R2 = 0.90
NEAR_DETERMINISTIC_ABSOLUTE_DOMAIN_SPEARMAN = 0.95
NONREDUNDANT_DOMAIN_COUNT = 4
MINIMUM_RANGE_TOLERANCE_MULTIPLIER = 10.0
ENDPOINT_RANGE_TOLERANCES: Mapping[str, float] = MappingProxyType(
    {
        "extension_peak_mm": 0.001,
        "slope_u20_u60_mpa_per_mm": 1.0,
        "normalized_prepeak_auc": 0.0001,
    }
)


class GateError(ValueError):
    """Raised when P1 gate facts do not match the frozen contract."""


@dataclass(frozen=True, slots=True)
class EndpointGateFacts:
    endpoint: str
    valid_count: int
    valid_domain_counts: Mapping[str, int]
    strength_only_pooled_r2: float
    strength_only_domain_spearman: Mapping[str, float]
    descriptor_range: float
    range_tolerance: float
    replay_byte_identical: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "valid_domain_counts", MappingProxyType(dict(self.valid_domain_counts))
        )
        object.__setattr__(
            self,
            "strength_only_domain_spearman",
            MappingProxyType(dict(self.strength_only_domain_spearman)),
        )


@dataclass(frozen=True, slots=True)
class EndpointGateDecision:
    endpoint: str
    passed: bool
    reasons: tuple[str, ...]
    coverage_fraction: float
    nonredundant_domain_count: int


@dataclass(frozen=True, slots=True)
class P1GateDecision:
    status: StageStatus
    passing_endpoints: tuple[str, ...]
    endpoint_decisions: tuple[EndpointGateDecision, ...]


def _validate_facts(facts: EndpointGateFacts) -> None:
    if facts.endpoint not in PRIMARY_TARGET_FIELDS:
        raise GateError(f"unknown P1 endpoint: {facts.endpoint!r}")
    expected_tolerance = ENDPOINT_RANGE_TOLERANCES[facts.endpoint]
    if facts.range_tolerance != expected_tolerance:
        raise GateError(f"P1 range tolerance changed: {facts.endpoint}")
    if (
        not isinstance(facts.valid_count, int)
        or isinstance(facts.valid_count, bool)
        or not 0 <= facts.valid_count <= PRIMARY_COHORT_SIZE
    ):
        raise GateError(f"invalid P1 valid count: {facts.endpoint}")
    if set(facts.valid_domain_counts) != set(PRIMARY_COUNTS):
        raise GateError(f"P1 valid domain counts changed: {facts.endpoint}")
    for domain, count in facts.valid_domain_counts.items():
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 <= count <= PRIMARY_COUNTS[domain]
        ):
            raise GateError(f"invalid P1 domain count: {facts.endpoint}/{domain}")
    if sum(facts.valid_domain_counts.values()) != facts.valid_count:
        raise GateError(f"P1 valid counts do not reconcile: {facts.endpoint}")
    if set(facts.strength_only_domain_spearman) != set(PRIMARY_COUNTS):
        raise GateError(f"P1 domain Spearman facts changed: {facts.endpoint}")
    correlations = tuple(facts.strength_only_domain_spearman.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or abs(float(value)) > 1.0
        for value in correlations
    ):
        raise GateError(f"invalid P1 domain Spearman: {facts.endpoint}")
    numeric = (facts.strength_only_pooled_r2, facts.descriptor_range)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric
    ) or facts.descriptor_range < 0.0:
        raise GateError(f"invalid P1 endpoint metric: {facts.endpoint}")
    if type(facts.replay_byte_identical) is not bool:
        raise GateError(f"invalid P1 replay fact: {facts.endpoint}")


def _evaluate_endpoint(facts: EndpointGateFacts) -> EndpointGateDecision:
    _validate_facts(facts)
    reasons: list[str] = []
    coverage = facts.valid_count / PRIMARY_COHORT_SIZE
    if coverage < MINIMUM_COVERAGE_FRACTION:
        reasons.append(
            f"coverage {coverage:.6f} is below {MINIMUM_COVERAGE_FRACTION:.2f}"
        )
    if any(facts.valid_domain_counts[domain] == 0 for domain in PRIMARY_COUNTS):
        reasons.append("valid extraction does not cover all six domains")

    nonredundant_domains = sum(
        abs(float(facts.strength_only_domain_spearman[domain]))
        < NEAR_DETERMINISTIC_ABSOLUTE_DOMAIN_SPEARMAN
        for domain in PRIMARY_COUNTS
    )
    if not (
        facts.strength_only_pooled_r2 < NEAR_DETERMINISTIC_POOLED_R2
        or nonredundant_domains >= NONREDUNDANT_DOMAIN_COUNT
    ):
        reasons.append("response is near-determined by the strength-only reference")
    required_range = (
        MINIMUM_RANGE_TOLERANCE_MULTIPLIER * facts.range_tolerance
    )
    if not facts.descriptor_range > required_range:
        reasons.append(
            f"descriptor range does not exceed {MINIMUM_RANGE_TOLERANCE_MULTIPLIER:g}x tolerance"
        )
    if not facts.replay_byte_identical:
        reasons.append("artifact replay is not byte-identical")
    return EndpointGateDecision(
        endpoint=facts.endpoint,
        passed=not reasons,
        reasons=tuple(reasons),
        coverage_fraction=coverage,
        nonredundant_domain_count=nonredundant_domains,
    )


def evaluate_p1_gate(facts: tuple[EndpointGateFacts, ...]) -> P1GateDecision:
    """Apply all five endpoint criteria and the any-endpoint P1 decision."""

    values = tuple(facts)
    endpoints = tuple(item.endpoint for item in values)
    if len(values) != len(PRIMARY_TARGET_FIELDS) or set(endpoints) != set(
        PRIMARY_TARGET_FIELDS
    ):
        raise GateError("P1 gate requires exactly the three primary endpoints")
    by_endpoint = {item.endpoint: item for item in values}
    decisions = tuple(
        _evaluate_endpoint(by_endpoint[endpoint]) for endpoint in PRIMARY_TARGET_FIELDS
    )
    passing = tuple(item.endpoint for item in decisions if item.passed)
    status = (
        StageStatus.P1_GO
        if passing
        else StageStatus.RESPONSE_BEYOND_STRENGTH_NO_GO
    )
    return P1GateDecision(
        status=status,
        passing_endpoints=passing,
        endpoint_decisions=decisions,
    )
