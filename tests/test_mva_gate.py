from __future__ import annotations

from dataclasses import replace

import pytest

from cmc_bbdm.mva.evaluation import A3GateInputs, evaluate_a3_gate
from cmc_bbdm.mva.statistics import BootstrapEffect


def _effect(
    effect_id: str, point: float = 0.01, lower: float = 0.001
) -> BootstrapEffect:
    return BootstrapEffect(
        effect_id=effect_id,
        point_estimate=point,
        lower=lower,
        upper=0.02,
        improved_domains=6,
        domain_effects=(point,) * 6,
        indices_sha256="a" * 64,
    )


def _passing() -> A3GateInputs:
    return A3GateInputs(
        uniform_low_mae=0.12,
        mechanical_low_mae=0.108,
        uniform_domain_low_mae=(0.12,) * 6,
        mechanical_domain_low_mae=(0.108,) * 6,
        reconstruction_minus_mechanical_auebc=_effect("reconstruction"),
        appearance_minus_mechanical_auebc=_effect("appearance"),
        uniform_auebc=0.030,
        random_median_auebc=0.028,
        mechanical_auebc=0.024,
        uniform_b5=0.25,
        random_median_b5=0.25,
        mechanical_b5=0.125,
    )


def test_all_four_hypotheses_are_required_for_go() -> None:
    result = evaluate_a3_gate(_passing())

    assert result.status == "MVA_ORACLE_GO"
    assert result.h1_pass and result.h2_pass and result.h3_pass and result.h4_pass
    assert result.h1_relative_improvement == pytest.approx(0.1)
    assert result.h1_improved_domains == 6


def test_each_failed_hypothesis_forces_no_go_without_rescue() -> None:
    passing = _passing()
    failures = (
        replace(passing, mechanical_low_mae=0.119),
        replace(
            passing,
            reconstruction_minus_mechanical_auebc=_effect(
                "reconstruction", point=0.01, lower=0.0
            ),
        ),
        replace(
            passing,
            appearance_minus_mechanical_auebc=_effect(
                "appearance", point=-0.01, lower=-0.02
            ),
        ),
        replace(
            passing,
            mechanical_auebc=0.027,
            uniform_b5=0.125,
            random_median_b5=0.125,
            mechanical_b5=0.125,
        ),
    )

    for inputs in failures:
        result = evaluate_a3_gate(inputs)
        assert result.status == "MVA_ORACLE_NO_GO"
        assert not all((result.h1_pass, result.h2_pass, result.h3_pass, result.h4_pass))


def test_missing_b5_does_not_get_imputed() -> None:
    inputs = replace(
        _passing(),
        mechanical_auebc=0.027,
        uniform_b5=None,
        random_median_b5=None,
        mechanical_b5=None,
    )

    result = evaluate_a3_gate(inputs)

    assert not result.h4_b5_pass
    assert result.h4_b5_saving is None
