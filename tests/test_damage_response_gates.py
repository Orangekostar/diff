from __future__ import annotations

from dataclasses import replace

import pytest

from cmc_bbdm.damage_response.contracts import (
    PRIMARY_COUNTS,
    P0GateFacts,
    StageStatus,
    evaluate_p0_gate,
)


def _clear_facts() -> P0GateFacts:
    return P0GateFacts(
        exact_identity_pairing_possible=True,
        exact_pair_counts=PRIMARY_COUNTS,
        identity_guessed=False,
        all_sources_hash_bound=True,
        peak_reconciliation_passed=True,
        missing_primary_channel_fractions={domain: 0.0 for domain in PRIMARY_COUNTS},
    )


@pytest.mark.parametrize(
    ("facts", "reason"),
    (
        (
            replace(_clear_facts(), exact_identity_pairing_possible=False),
            "exact identity pairing",
        ),
        (replace(_clear_facts(), identity_guessed=True), "guessed identity"),
        (
            replace(_clear_facts(), all_sources_hash_bound=False),
            "source identity",
        ),
        (
            replace(_clear_facts(), peak_reconciliation_passed=False),
            "published peak",
        ),
    ),
)
def test_p0_no_go_reasons_fail_closed(facts: P0GateFacts, reason: str) -> None:
    decision = evaluate_p0_gate(facts)
    assert decision.status is StageStatus.P0_NO_GO
    assert reason in " ".join(decision.reasons)


def test_missing_primary_domain_is_no_go() -> None:
    counts = dict(PRIMARY_COUNTS)
    counts.pop("74t7kcdgkr")
    decision = evaluate_p0_gate(replace(_clear_facts(), exact_pair_counts=counts))
    assert decision.status is StageStatus.P0_NO_GO
    assert "missing primary domain" in " ".join(decision.reasons)


def test_fewer_than_twenty_exact_pairs_requires_human_review() -> None:
    counts = dict(PRIMARY_COUNTS)
    counts["74t7kcdgkr"] = 19
    decision = evaluate_p0_gate(replace(_clear_facts(), exact_pair_counts=counts))
    assert decision.status is StageStatus.P0_REQUIRES_HUMAN_REVIEW
    assert "fewer than 20" in " ".join(decision.reasons)


def test_more_than_twenty_percent_missing_channels_requires_human_review() -> None:
    fractions = {domain: 0.0 for domain in PRIMARY_COUNTS}
    fractions["cgtnjyggtm"] = 0.2000001
    decision = evaluate_p0_gate(
        replace(_clear_facts(), missing_primary_channel_fractions=fractions)
    )
    assert decision.status is StageStatus.P0_REQUIRES_HUMAN_REVIEW
    assert "more than 20%" in " ".join(decision.reasons)


def test_exactly_twenty_percent_missing_channels_is_allowed() -> None:
    fractions = {domain: 0.2 for domain in PRIMARY_COUNTS}
    decision = evaluate_p0_gate(
        replace(_clear_facts(), missing_primary_channel_fractions=fractions)
    )
    assert decision.status is StageStatus.P0_GO


def test_no_go_takes_precedence_over_human_review() -> None:
    counts = dict(PRIMARY_COUNTS)
    counts["74t7kcdgkr"] = 19
    decision = evaluate_p0_gate(
        replace(
            _clear_facts(),
            exact_pair_counts=counts,
            all_sources_hash_bound=False,
        )
    )
    assert decision.status is StageStatus.P0_NO_GO


def test_all_clear_facts_are_p0_go() -> None:
    decision = evaluate_p0_gate(_clear_facts())
    assert decision.status is StageStatus.P0_GO
    assert decision.reasons == ()
