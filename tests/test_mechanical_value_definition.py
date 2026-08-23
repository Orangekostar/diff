from __future__ import annotations

import pytest

from cmc_bbdm.mva.mechanical_value import mechanical_value


def test_mechanical_value_is_absolute_cai_error_reduction() -> None:
    result = mechanical_value(
        target=0.8, current_prediction=0.5, candidate_prediction=0.7
    )

    assert result.absolute_error_before == pytest.approx(0.3)
    assert result.absolute_error_after == pytest.approx(0.1)
    assert result.absolute_error_reduction == pytest.approx(0.2)
    assert result.squared_error_reduction == pytest.approx(0.08)


def test_mechanical_value_preserves_adverse_candidates() -> None:
    result = mechanical_value(
        target=0.5, current_prediction=0.5, candidate_prediction=0.9
    )

    assert result.absolute_error_reduction == pytest.approx(-0.4)
    assert result.squared_error_reduction == pytest.approx(-0.16)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_mechanical_value_rejects_nonfinite_inputs(value: float) -> None:
    with pytest.raises(ValueError):
        mechanical_value(target=value, current_prediction=0.5, candidate_prediction=0.6)
