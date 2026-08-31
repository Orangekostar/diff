from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.stopping import (
    ReferenceEndpoint,
    earliest_sufficient_state,
    select_strongest_fixed_reference,
)


def test_fixed_reference_selection_equal_weights_source_domains() -> None:
    rows = (
        ReferenceEndpoint("A", "d1", "s1", 0.1),
        ReferenceEndpoint("A", "d1", "s2", 0.3),
        ReferenceEndpoint("A", "d2", "s3", 0.4),
        ReferenceEndpoint("B", "d1", "s1", 0.3),
        ReferenceEndpoint("B", "d1", "s2", 0.3),
        ReferenceEndpoint("B", "d2", "s3", 0.2),
    )
    selected = select_strongest_fixed_reference(rows, allowed_methods=("A", "B"))
    # A: (mean(d1)=.2 + mean(d2)=.4)/2=.3; B: (.3+.2)/2=.25.
    assert selected.method == "B"
    assert selected.equal_domain_loss == pytest.approx(0.25)


def test_earliest_sufficiency_reports_normalized_measurement_saving() -> None:
    result = earliest_sufficient_state(
        budgets=np.asarray((0.0, 0.05, 0.10, 0.20, 0.25)),
        losses=np.asarray((1.0, 0.5, 0.21, 0.19, 0.18)),
        reference_budget=0.25,
        reference_loss=0.20,
        tolerance=0.05,
    )
    assert result.reached
    assert result.stop_index == 2
    assert result.budget_to_sufficiency == pytest.approx(0.10)
    assert result.normalized_measurement_saving == pytest.approx(0.60)
    assert result.final_task_loss == pytest.approx(0.21)


def test_unreached_sufficiency_uses_endpoint_and_zero_saving() -> None:
    result = earliest_sufficient_state(
        budgets=np.asarray((0.0, 0.1, 0.25)),
        losses=np.asarray((1.0, 0.7, 0.5)),
        reference_budget=0.25,
        reference_loss=0.2,
        tolerance=0.05,
    )
    assert not result.reached
    assert result.stop_index == 2
    assert result.budget_to_sufficiency == 0.25
    assert result.normalized_measurement_saving == 0.0
