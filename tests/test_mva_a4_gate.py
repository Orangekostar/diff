from __future__ import annotations

from dataclasses import replace

import pytest

from cmc_bbdm.mva.a4_evaluation import A4GateInputs, evaluate_a4_gate
from cmc_bbdm.mva.statistics import BootstrapEffect

INDICES_SHA256 = "a" * 64


def _effect(
    effect_id: str,
    domain_effects: tuple[float, ...],
    *,
    lower: float = 0.001,
) -> BootstrapEffect:
    return BootstrapEffect(
        effect_id=effect_id,
        point_estimate=sum(domain_effects) / 6.0,
        lower=lower,
        upper=max(domain_effects),
        improved_domains=sum(value > 0.0 for value in domain_effects),
        domain_effects=domain_effects,
        indices_sha256=INDICES_SHA256,
    )


def _inputs(*, oracle_auebc: float = 0.97) -> A4GateInputs:
    return A4GateInputs(
        uniform_effect=_effect("uniform_minus_global", (0.02,) * 6),
        reconstruction_effect=_effect("reconstruction_minus_global", (0.01,) * 6),
        appearance_effect=_effect("appearance_minus_global", (0.015,) * 6),
        adaptive_gap_effect=_effect(
            "global_minus_oracle", (1.0 - oracle_auebc,) * 6
        ),
        global_mechanical_auebc=1.0,
        mechanical_oracle_auebc=oracle_auebc,
    )


def test_a5_is_authorized_at_but_not_below_three_percent_gap() -> None:
    below = evaluate_a4_gate(_inputs(oracle_auebc=0.970000001))
    exact = evaluate_a4_gate(_inputs(oracle_auebc=0.97))

    assert below.relative_adaptive_gap < 0.03
    assert below.a5_status == "MVA_A5_NOT_AUTHORIZED"
    assert exact.relative_adaptive_gap == pytest.approx(0.03)
    assert exact.a5_status == "MVA_A5_AUTHORIZED"


def test_a4_requires_strict_lower_bounds_and_four_uniform_domains() -> None:
    baseline = _inputs()
    zero_lower = replace(
        baseline,
        reconstruction_effect=replace(baseline.reconstruction_effect, lower=0.0),
    )
    three_domains = replace(
        baseline,
        uniform_effect=_effect(
            "uniform_minus_global", (0.03, 0.03, 0.03, -0.01, -0.01, -0.01)
        ),
    )
    four_domains = replace(
        baseline,
        uniform_effect=_effect(
            "uniform_minus_global", (0.03, 0.03, 0.03, 0.03, -0.01, -0.01)
        ),
    )

    assert evaluate_a4_gate(zero_lower).global_mask_status == "MVA_A4_GLOBAL_NO_GO"
    assert evaluate_a4_gate(three_domains).global_mask_status == "MVA_A4_GLOBAL_NO_GO"
    assert evaluate_a4_gate(four_domains).global_mask_status == "MVA_A4_GLOBAL_GO"


def test_a4_and_a5_decisions_are_independent() -> None:
    inputs = _inputs()
    failed_a4 = replace(
        inputs,
        appearance_effect=replace(inputs.appearance_effect, lower=-0.001),
    )
    decision = evaluate_a4_gate(failed_a4)

    assert decision.global_mask_status == "MVA_A4_GLOBAL_NO_GO"
    assert decision.a5_status == "MVA_A5_AUTHORIZED"
