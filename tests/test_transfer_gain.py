from __future__ import annotations

import pytest

from cmc_bbdm.msss.transfer_metrics import TransferMetricError, s2_gate, transfer_gain


def test_transfer_gain_sign_and_relative_gain() -> None:
    result = transfer_gain(full_mae=0.20, candidate_mae=0.15)

    assert result.tg == pytest.approx(0.05)
    assert result.rtg == pytest.approx(0.25)
    assert result.nonworse


def test_transfer_gain_rejects_zero_full_error() -> None:
    with pytest.raises(TransferMetricError):
        transfer_gain(full_mae=0.0, candidate_mae=0.1)


def test_s2_gate_uses_frozen_structured_and_domain_thresholds() -> None:
    result = s2_gate(
        domain_tg=(0.01, 0.0, 0.02, 0.03, -0.01, -0.02),
        ply_tg=(0.01, 0.02, -0.01),
        layup_tg=(0.01, -0.02),
    )

    assert result.status == "STRONG_GO"
    assert result.domain_support
