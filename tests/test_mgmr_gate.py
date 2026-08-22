from __future__ import annotations

from types import MappingProxyType

from cmc_bbdm.mgmr.statistics import (
    DomainMetric,
    MetricSummary,
    decide_m0,
)

DOMAINS = ("a", "b", "c", "d", "e", "f")


def _metric(method: str, maes: tuple[float, ...]) -> MetricSummary:
    rows = tuple(
        DomainMetric(method, domain, 1, value, 0.0, 0.0)
        for domain, value in zip(DOMAINS, maes, strict=True)
    )
    return MetricSummary(
        method=method,
        specimen_count=6,
        specimen_mae=sum(maes) / 6,
        equal_domain_mae=sum(maes) / 6,
        worst_domain_mae=max(maes),
        pearson=0.0,
        spearman=0.0,
        domain_metrics=rows,
    )


def test_gate_passes_only_when_a_b_and_d_are_all_true() -> None:
    baseline = _metric("coarse", (0.10,) * 6)
    real = _metric("coarse_residual", (0.09, 0.09, 0.09, 0.09, 0.11, 0.11))
    decision = decide_m0(
        direct=MappingProxyType(
            {
                "B1": baseline,
                "B2": _metric("B2", (0.12,) * 6),
                "B3": _metric("B3", (0.09, 0.09, 0.09, 0.09, 0.11, 0.11)),
            }
        ),
        coarse_baseline=baseline,
        coarse_corrected=real,
        full_baseline=_metric("full", (0.08,) * 6),
        full_corrected=_metric("full_residual", (0.07,) * 6),
        shuffled={
            1: _metric("shuffle1", (0.099,) * 6),
            2: _metric("shuffle2", (0.0995,) * 6),
            3: _metric("shuffle3", (0.0998,) * 6),
        },
        required_gates=("A", "B", "D"),
        minimum_positive_domains=4,
    )

    assert dict(decision.gates) == {"A": True, "B": True, "C": True, "D": True}
    assert decision.required_gates == ("A", "B", "D")
    assert decision.status == "MGMR_GO"
    assert decision.improved_domains["A"] == 4
    assert decision.improved_domains["B"] == 4
    assert decision.improved_domains["D"] == 4


def test_gate_boundaries_are_strict_and_issue_no_go() -> None:
    baseline = _metric("coarse", (0.10,) * 6)
    decision = decide_m0(
        direct={"B1": baseline, "B2": _metric("B2", (0.12,) * 6), "B3": baseline},
        coarse_baseline=baseline,
        coarse_corrected=baseline,
        full_baseline=_metric("full", (0.08,) * 6),
        full_corrected=_metric("full_residual", (0.08,) * 6),
        shuffled={1: baseline, 2: baseline, 3: baseline},
        required_gates=("A", "B", "D"),
        minimum_positive_domains=4,
    )

    assert dict(decision.gates) == {"A": False, "B": False, "C": False, "D": False}
    assert decision.status == "MGMR_NO_GO"
