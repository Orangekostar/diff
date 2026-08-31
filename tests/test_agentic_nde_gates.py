from __future__ import annotations

from cmc_bbdm.agentic_nde.contracts import (
    NOT_AUTHORIZED_STAGES,
    P0GateFacts,
    StageStatus,
    decide_p0,
)


def _passing_facts() -> P0GateFacts:
    return P0GateFacts(
        authorized_by_domain={
            "74t7kcdgkr": 45,
            "cgtnjyggtm": 49,
            "w68dtmpfyf": 43,
            "xcmzfsbd9t": 59,
            "yfxyg8jm46": 42,
            "ykhs7s2dck": 38,
        },
        exact_identity_hashes=True,
        orientation_resolved=True,
        deterministic_transform=True,
        deployable_evidence_only=True,
        replay_verified=True,
    )


def test_all_p0_requirements_yield_go() -> None:
    decision = decide_p0(_passing_facts())
    assert decision.status is StageStatus.P0_GO
    assert decision.reasons == ()


def test_unresolved_orientation_fails_closed() -> None:
    facts = _passing_facts()
    facts = P0GateFacts(**{**facts.as_dict(), "orientation_resolved": False})
    decision = decide_p0(facts)
    assert decision.status is StageStatus.P0_SPATIAL_REGISTRATION_NO_GO
    assert "orientation_unresolved" in decision.reasons
    assert decision.downstream == NOT_AUTHORIZED_STAGES


def test_coverage_threshold_applies_per_domain_and_total() -> None:
    facts = _passing_facts()
    counts = dict(facts.authorized_by_domain)
    counts["74t7kcdgkr"] = 40
    facts = P0GateFacts(**{**facts.as_dict(), "authorized_by_domain": counts})
    decision = decide_p0(facts)
    assert decision.status is StageStatus.P0_SPATIAL_REGISTRATION_NO_GO
    assert any(reason.startswith("coverage_below_90_percent:74t7kcdgkr") for reason in decision.reasons)


def test_identity_failure_has_distinct_status() -> None:
    facts = _passing_facts()
    facts = P0GateFacts(**{**facts.as_dict(), "exact_identity_hashes": False})
    assert decide_p0(facts).status is StageStatus.P0_IDENTITY_AUTHORITY_NO_GO
