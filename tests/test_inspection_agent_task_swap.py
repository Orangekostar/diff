from __future__ import annotations

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionDecision
from cmc_bbdm.inspection_agent.evaluation import (
    task_swap_advantages,
    trajectory_overlap,
)
from cmc_bbdm.inspection_agent.g0 import GateEvidence, task_conditioning_gate
from cmc_bbdm.inspection_agent.state import InspectionCellAction


def test_task_swap_advantage_is_wrong_task_loss_minus_correct_task_loss() -> None:
    result = task_swap_advantages(
        field_on_field=np.asarray((0.1, 0.2)),
        cai_on_field=np.asarray((0.3, 0.25)),
        cai_on_cai=np.asarray((0.2, 0.1)),
        field_on_cai=np.asarray((0.5, 0.4)),
    )
    np.testing.assert_allclose(result.field_advantage, (0.2, 0.05))
    np.testing.assert_allclose(result.cai_advantage, (0.3, 0.3))


def test_trajectory_overlap_reports_difference_without_claiming_task_value() -> None:
    field = (
        InspectionCellAction(0, -1, 0),
        InspectionCellAction(0, 0, 1),
        InspectionCellAction(1, -1, 0),
    )
    cai = (
        InspectionCellAction(0, -1, 0),
        InspectionCellAction(2, -1, 0),
        InspectionCellAction(2, 0, 1),
    )
    overlap = trajectory_overlap(
        field,
        cai,
        field_decisions=(
            InspectionDecision.FOCUS,
            InspectionDecision.REFINE,
            InspectionDecision.BROADEN,
        ),
        cai_decisions=(
            InspectionDecision.FOCUS,
            InspectionDecision.BROADEN,
            InspectionDecision.REFINE,
        ),
    )
    assert 0.0 < overlap.action_jaccard < 1.0
    assert 0.0 < overlap.cell_jaccard < 1.0
    assert overlap.normalized_edit_distance > 0.0


def test_task_conditioning_gate_requires_both_positive_wrong_task_swaps() -> None:
    positive = GateEvidence(0.1, 0.02, 0.2, 5)
    unsupported = GateEvidence(0.0, -0.01, 0.02, 3)
    assert task_conditioning_gate(positive, positive)
    assert not task_conditioning_gate(positive, unsupported)
