from __future__ import annotations

import math

from cmc_bbdm.agentic_nde.contracts import (
    NOT_AUTHORIZED_STAGES,
    PRIMARY_COUNTS,
    P0RGateFacts,
    P0RStatus,
    StageStatus,
    decide_p0r,
)


def _passing_facts() -> P0RGateFacts:
    return P0RGateFacts(
        authorized_by_domain=dict(PRIMARY_COUNTS),
        exact_identity_hashes=True,
        author_statement_bound=True,
        global_orientation_rot90=True,
        all_panels_resolved=True,
        processing_provenance_deterministic=True,
        no_unsupported_rotation_reflection=True,
        composed_transform_replayable=True,
        no_result_driven_orientation=True,
        author_evidence_conflict=False,
        processing_provenance_unresolved=False,
    )


def test_all_p0r_requirements_yield_go() -> None:
    decision = decide_p0r(_passing_facts())

    assert decision.status is P0RStatus.GO
    assert decision.reasons == ()
    assert decision.downstream_registration_status is StageStatus.P0_GO
    assert decision.p1_authorized is True
    assert decision.downstream == ()


def test_author_evidence_conflict_has_highest_precedence() -> None:
    facts = _passing_facts()
    decision = decide_p0r(
        P0RGateFacts(
            **{
                **facts.as_dict(),
                "author_evidence_conflict": True,
                "processing_provenance_unresolved": True,
            }
        )
    )

    assert decision.status is P0RStatus.CONFLICT
    assert decision.reasons == ("author_evidence_conflict",)
    assert decision.p1_authorized is False
    assert decision.downstream == NOT_AUTHORIZED_STAGES


def test_unresolved_processing_provenance_has_distinct_status() -> None:
    facts = _passing_facts()
    decision = decide_p0r(
        P0RGateFacts(
            **{
                **facts.as_dict(),
                "processing_provenance_unresolved": True,
            }
        )
    )

    assert decision.status is P0RStatus.PROVENANCE_UNRESOLVED
    assert decision.reasons == ("processing_provenance_unresolved",)
    assert decision.downstream_registration_status is StageStatus.P0_SPATIAL_REGISTRATION_NO_GO


def test_result_driven_orientation_fails_closed() -> None:
    facts = _passing_facts()
    decision = decide_p0r(
        P0RGateFacts(
            **{
                **facts.as_dict(),
                "no_result_driven_orientation": False,
            }
        )
    )

    assert decision.status is P0RStatus.NO_GO
    assert decision.reasons == ("result_driven_orientation_not_excluded",)
    assert decision.p1_authorized is False


def test_each_boolean_gate_has_a_closed_reason() -> None:
    expected = {
        "exact_identity_hashes": "identity_or_hash_binding_failed",
        "author_statement_bound": "author_statement_not_bound",
        "global_orientation_rot90": "global_rot90_not_fixed",
        "all_panels_resolved": "specimen_panel_unresolved",
        "processing_provenance_deterministic": "processing_provenance_not_deterministic",
        "no_unsupported_rotation_reflection": "unsupported_rotation_or_reflection",
        "composed_transform_replayable": "composed_transform_not_replayable",
    }
    passing = _passing_facts().as_dict()

    for field, reason in expected.items():
        decision = decide_p0r(P0RGateFacts(**{**passing, field: False}))
        assert decision.status is P0RStatus.NO_GO
        assert reason in decision.reasons


def test_p0r_coverage_threshold_applies_per_domain_and_total() -> None:
    facts = _passing_facts()
    counts = dict(PRIMARY_COUNTS)
    domain = "74t7kcdgkr"
    counts[domain] = math.ceil(PRIMARY_COUNTS[domain] * 0.9) - 1
    decision = decide_p0r(
        P0RGateFacts(**{**facts.as_dict(), "authorized_by_domain": counts})
    )

    assert decision.status is P0RStatus.NO_GO
    assert any(reason.startswith(f"coverage_below_90_percent:{domain}:") for reason in decision.reasons)

    low_total = {key: 0 for key in PRIMARY_COUNTS}
    decision = decide_p0r(
        P0RGateFacts(**{**facts.as_dict(), "authorized_by_domain": low_total})
    )
    assert "coverage_below_240_total:0/276" in decision.reasons


def test_p0r_rejects_unexpected_domains() -> None:
    facts = _passing_facts()
    counts = {**dict(PRIMARY_COUNTS), "unexpected": 1}
    decision = decide_p0r(
        P0RGateFacts(**{**facts.as_dict(), "authorized_by_domain": counts})
    )

    assert decision.status is P0RStatus.NO_GO
    assert "unexpected_primary_domains:unexpected" in decision.reasons
