from __future__ import annotations

import math
import re
from dataclasses import dataclass

from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS, StageStatus
from cmc_bbdm.damage_response.feature_views import PRIMARY_TARGET_FIELDS
from cmc_bbdm.damage_response.p2_statistics import (
    P2_BOOTSTRAP_REPLICATES,
    P2_BOOTSTRAP_SEED,
    P2ContrastAnalysis,
    P2ContrastResult,
)

MINIMUM_RELATIVE_EQUAL_DOMAIN_MAE_IMPROVEMENT = 0.10
MINIMUM_IMPROVED_DOMAIN_COUNT = 4
PRIMARY_REFERENCE_VIEW = "F2"
PRIMARY_CANDIDATE_VIEWS = ("F3", "F4")

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class P2GateError(ValueError):
    """Raised when P2 gate evidence does not match the frozen family."""


@dataclass(frozen=True, slots=True)
class P2ContrastGateDecision:
    endpoint: str
    candidate_view: str
    reference_view: str
    passed: bool
    point_threshold_passed: bool
    domain_count_passed: bool
    familywise_interval_passed: bool
    relative_improvement: float
    improved_domain_count: int
    familywise_lower_bound: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class P2GateDecision:
    status: StageStatus
    passing_contrasts: tuple[tuple[str, str, str], ...]
    contrast_decisions: tuple[P2ContrastGateDecision, ...]


def _validate_result(result: P2ContrastResult) -> None:
    expected_name = (
        f"{result.endpoint}__{result.candidate_view}_vs_{result.reference_view}"
    )
    if (
        result.endpoint not in PRIMARY_TARGET_FIELDS
        or result.name != expected_name
        or result.reference_view not in {"F2", "F3"}
        or result.candidate_view not in {"F3", "F4"}
        or result.reference_view == result.candidate_view
        or _SHA256_RE.fullmatch(result.replicate_sha256) is None
    ):
        raise P2GateError("P2 contrast identity changed")
    numeric = (
        result.observed_reference_equal_domain_mae,
        result.observed_candidate_equal_domain_mae,
        result.observed_improvement,
        result.relative_improvement,
        result.bootstrap_mean,
        *result.ordinary_interval,
        result.probability_positive,
    )
    if any(not math.isfinite(float(value)) for value in numeric):
        raise P2GateError("P2 contrast contains nonfinite values")
    if (
        result.observed_reference_equal_domain_mae <= 0.0
        or result.observed_candidate_equal_domain_mae < 0.0
        or not math.isclose(
            result.observed_improvement,
            result.observed_reference_equal_domain_mae
            - result.observed_candidate_equal_domain_mae,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            result.relative_improvement,
            result.observed_improvement
            / result.observed_reference_equal_domain_mae,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not 0.0 <= result.probability_positive <= 1.0
        or result.ordinary_interval[0] > result.ordinary_interval[1]
    ):
        raise P2GateError("P2 contrast metrics do not reconcile")
    if tuple(domain for domain, _value in result.domain_improvements) != tuple(
        PRIMARY_COUNTS
    ):
        raise P2GateError("P2 contrast domain registry changed")
    domain_values = tuple(value for _domain, value in result.domain_improvements)
    if any(not math.isfinite(float(value)) for value in domain_values):
        raise P2GateError("P2 contrast domain values are nonfinite")
    if (
        result.improved_domain_count != sum(value > 0.0 for value in domain_values)
        or not math.isclose(
            result.observed_improvement,
            sum(domain_values) / len(domain_values),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise P2GateError("P2 contrast domain values do not reconcile")
    if result.primary_family:
        if result.familywise_interval is None:
            raise P2GateError("primary P2 contrast lacks familywise interval")
        if (
            any(not math.isfinite(float(value)) for value in result.familywise_interval)
            or result.familywise_interval[0] > result.familywise_interval[1]
        ):
            raise P2GateError("P2 familywise interval is invalid")
    elif result.familywise_interval is not None:
        raise P2GateError("secondary P2 contrast has a familywise interval")


def evaluate_p2_gate(analysis: P2ContrastAnalysis) -> P2GateDecision:
    """Apply the preregistered any-endpoint F3/F4-versus-F2 P2 gate."""

    if not isinstance(analysis, P2ContrastAnalysis):
        raise P2GateError("P2 contrast analysis type changed")
    if (
        analysis.seed != P2_BOOTSTRAP_SEED
        or analysis.replicates != P2_BOOTSTRAP_REPLICATES
        or _SHA256_RE.fullmatch(analysis.synchronized_replicate_sha256) is None
    ):
        raise P2GateError("P2 bootstrap identity changed")
    results = tuple(analysis.contrasts)
    if len(results) != 9:
        raise P2GateError("P2 gate requires six primary and three secondary contrasts")
    for result in results:
        _validate_result(result)
    expected_primary = {
        (endpoint, candidate, PRIMARY_REFERENCE_VIEW)
        for endpoint in PRIMARY_TARGET_FIELDS
        for candidate in PRIMARY_CANDIDATE_VIEWS
    }
    observed_primary = {
        (result.endpoint, result.candidate_view, result.reference_view)
        for result in results
        if result.primary_family
    }
    expected_secondary = {
        (endpoint, "F4", "F3") for endpoint in PRIMARY_TARGET_FIELDS
    }
    observed_secondary = {
        (result.endpoint, result.candidate_view, result.reference_view)
        for result in results
        if not result.primary_family
    }
    if observed_primary != expected_primary or observed_secondary != expected_secondary:
        raise P2GateError("P2 contrast family membership changed")
    if {result.bootstrap_column for result in results} != set(range(9)):
        raise P2GateError("P2 bootstrap column registry changed")

    by_identity = {
        (result.endpoint, result.candidate_view, result.reference_view): result
        for result in results
    }
    decisions: list[P2ContrastGateDecision] = []
    for endpoint in PRIMARY_TARGET_FIELDS:
        for candidate in PRIMARY_CANDIDATE_VIEWS:
            result = by_identity[(endpoint, candidate, PRIMARY_REFERENCE_VIEW)]
            familywise = result.familywise_interval
            if familywise is None:
                raise P2GateError("primary P2 contrast lacks familywise evidence")
            point_passed = (
                result.relative_improvement
                >= MINIMUM_RELATIVE_EQUAL_DOMAIN_MAE_IMPROVEMENT
            )
            domains_passed = (
                result.improved_domain_count >= MINIMUM_IMPROVED_DOMAIN_COUNT
            )
            interval_passed = familywise[0] > 0.0
            reasons: list[str] = []
            if not point_passed:
                reasons.append("relative equal-domain MAE improvement is below 10%")
            if not domains_passed:
                reasons.append("fewer than four held-out domains improve")
            if not interval_passed:
                reasons.append("familywise bootstrap lower bound is not positive")
            decisions.append(
                P2ContrastGateDecision(
                    endpoint=endpoint,
                    candidate_view=candidate,
                    reference_view=PRIMARY_REFERENCE_VIEW,
                    passed=not reasons,
                    point_threshold_passed=point_passed,
                    domain_count_passed=domains_passed,
                    familywise_interval_passed=interval_passed,
                    relative_improvement=result.relative_improvement,
                    improved_domain_count=result.improved_domain_count,
                    familywise_lower_bound=familywise[0],
                    reasons=tuple(reasons),
                )
            )
    passing = tuple(
        (row.endpoint, row.candidate_view, row.reference_view)
        for row in decisions
        if row.passed
    )
    return P2GateDecision(
        status=(
            StageStatus.P2_GO
            if passing
            else StageStatus.MACK_EXTENSION_NO_GO
        ),
        passing_contrasts=passing,
        contrast_decisions=tuple(decisions),
    )
