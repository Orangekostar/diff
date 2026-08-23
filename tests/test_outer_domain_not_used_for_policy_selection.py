from __future__ import annotations

from cmc_bbdm.mva.crossfit import select_initial_survey

DOMAINS = tuple(f"d{index}" for index in range(6))


def test_initial_survey_uses_source_domain_scores_only() -> None:
    outer = "d5"
    full = {domain: 0.10 + index * 0.001 for index, domain in enumerate(DOMAINS)}
    candidate = {
        0.015625: {domain: 0.18 for domain in DOMAINS},
        0.03125: {domain: 0.13 for domain in DOMAINS},
        0.0625: {domain: 0.105 for domain in DOMAINS},
    }
    first = select_initial_survey(
        outer_domain=outer,
        domain_order=DOMAINS,
        full_domain_mae=full,
        candidate_domain_mae=candidate,
    )
    full[outer] = 10_000.0
    for values in candidate.values():
        values[outer] = 20_000.0
    second = select_initial_survey(
        outer_domain=outer,
        domain_order=DOMAINS,
        full_domain_mae=full,
        candidate_domain_mae=candidate,
    )

    assert first == second
    assert first.selected_budget == 0.03125
    assert first.status == "selected"


def test_initial_survey_marks_weak_headroom_without_rescue_search() -> None:
    full = {domain: 0.10 for domain in DOMAINS}
    candidates = {
        0.015625: {domain: 0.101 for domain in DOMAINS},
        0.03125: {domain: 0.102 for domain in DOMAINS},
        0.0625: {domain: 0.1024 for domain in DOMAINS},
    }

    result = select_initial_survey(
        outer_domain="d0",
        domain_order=DOMAINS,
        full_domain_mae=full,
        candidate_domain_mae=candidates,
    )

    assert result.selected_budget == 0.015625
    assert result.status == "weak_headroom"
