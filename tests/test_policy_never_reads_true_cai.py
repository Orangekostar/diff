from __future__ import annotations

import inspect

from cmc_bbdm.mva.oracle import choose_random_action, choose_uniform_action


def test_control_signatures_cannot_receive_true_cai() -> None:
    for function in (choose_uniform_action, choose_random_action):
        signature = inspect.signature(function)
        assert "target" not in signature.parameters
        assert "true_cai" not in signature.parameters
        assert "prediction" not in signature.parameters
