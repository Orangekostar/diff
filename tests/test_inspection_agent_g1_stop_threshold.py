from __future__ import annotations

import hashlib

import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.stopping_policy import (
    REGISTERED_STOP_THRESHOLDS,
    G1StoppingError,
    SourceStopValidationTrajectory,
    select_conservative_stop_threshold,
)

SOURCES = ("d1", "d2", "d3", "d4", "d5")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _trajectories(*, losses: tuple[float, ...] = (2.0, 1.0, 1.0)):
    return tuple(
        SourceStopValidationTrajectory(
            outer_target="d6",
            source_domain=domain,
            specimen_sha256=_sha(f"{domain}-sample"),
            task=InspectionTask.FIELD,
            budgets=(0.05, 0.10, 0.25),
            stop_probabilities=(0.60, 0.80, 0.99),
            true_task_losses=losses,
            reference_true_loss=1.0,
            endpoint_budget=0.25,
        )
        for domain in SOURCES
    )


def test_stop_threshold_roster_matches_the_preregistered_prompt() -> None:
    assert REGISTERED_STOP_THRESHOLDS == (0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)


def test_stop_threshold_maximizes_saving_subject_to_safety_constraints() -> None:
    selection = select_conservative_stop_threshold(_trajectories())
    assert selection.status == "STOP_AUTHORIZED_SOURCE_ONLY"
    assert selection.threshold == pytest.approx(0.80)
    selected = next(row for row in selection.candidates if row.threshold == 0.80)
    assert selected.equal_domain_saving == pytest.approx(0.60)
    assert selected.equal_domain_premature_rate == 0.0
    assert selected.equal_domain_task_loss_ratio == pytest.approx(1.0)


def test_stop_threshold_falls_back_when_no_candidate_is_safe() -> None:
    selection = select_conservative_stop_threshold(
        _trajectories(losses=(2.0, 2.0, 2.0))
    )
    assert selection.status == "STOP_NOT_AUTHORIZED"
    assert selection.threshold is None


def test_stop_threshold_rejects_target_trajectory() -> None:
    rows = list(_trajectories())
    rows[-1] = SourceStopValidationTrajectory(
        outer_target="d6",
        source_domain="d6",
        specimen_sha256=_sha("target"),
        task=InspectionTask.FIELD,
        budgets=(0.05, 0.10, 0.25),
        stop_probabilities=(0.60, 0.80, 0.99),
        true_task_losses=(2.0, 1.0, 1.0),
        reference_true_loss=1.0,
        endpoint_budget=0.25,
    )
    with pytest.raises(G1StoppingError, match="outer target"):
        select_conservative_stop_threshold(tuple(rows))
