from __future__ import annotations

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionDecision, InspectionTask
from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.inspection_agent_g1.teacher import (
    PrivilegedTeacherLabel,
    TeacherCandidateRecord,
)
from cmc_bbdm.inspection_agent_g1.utility_distillation import teacher_distribution


def _label(utilities: tuple[float, ...]) -> PrivilegedTeacherLabel:
    candidates = tuple(
        TeacherCandidateRecord(
            slot=index,
            action=InspectionCellAction(index, -1, 0),
            decision=InspectionDecision.BROADEN,
            exact_added_cost=10,
            raw_value=utility * 10,
            objective_value=utility,
            task_loss_after=1.0 - utility,
            candidate_state_sha256=f"{index + 1:064x}",
            selected=index == 0,
        )
        for index, utility in enumerate(utilities)
    )
    return PrivilegedTeacherLabel(
        task=InspectionTask.FIELD,
        authorization_sha256="a" * 64,
        observation_sha256="b" * 64,
        policy_state_sha256="c" * 64,
        selected_slot=0,
        candidates=candidates,
    )


def test_teacher_distribution_retains_near_equivalent_legal_actions() -> None:
    distribution = teacher_distribution(_label((1.0, 1.0, 0.5)), tau=1.0)
    assert distribution.probabilities.shape == (192,)
    assert distribution.legal_action_mask.sum() == 3
    assert distribution.probabilities[0] == distribution.probabilities[1]
    assert distribution.probabilities[0] > distribution.probabilities[2] > 0.0
    assert np.sum(distribution.probabilities) == 1.0
    np.testing.assert_array_equal(distribution.probabilities[3:], 0.0)
    assert distribution.scale > 0.0


def test_all_effectively_tied_utilities_produce_uniform_legal_mass() -> None:
    distribution = teacher_distribution(
        _label((2.0, 2.0 + 1.0e-13, 2.0)),
        tau=0.25,
        tie_tolerance=1.0e-12,
    )
    np.testing.assert_allclose(distribution.probabilities[:3], 1 / 3, atol=0, rtol=0)
    assert distribution.all_tied is True


def test_lower_temperature_sharpens_a_nontied_teacher_distribution() -> None:
    label = _label((1.0, 0.8, 0.0))
    cold = teacher_distribution(label, tau=0.25)
    hot = teacher_distribution(label, tau=2.0)
    assert cold.probabilities[0] > hot.probabilities[0]
    assert cold.tau == 0.25
    assert hot.tau == 2.0
