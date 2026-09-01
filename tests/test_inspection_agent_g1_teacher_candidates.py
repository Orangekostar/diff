from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionDecision, InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.crossfit import build_crossfit_roster
from cmc_bbdm.inspection_agent_g1.features import canonical_action_from_slot
from cmc_bbdm.inspection_agent_g1.teacher import (
    authorize_source_teacher,
    field_teacher_label,
)
from cmc_bbdm.inspection_agent_g1.warm_start import (
    PRIMARY_WARM_START_CELLS,
    build_deployment_grid,
)
from cmc_bbdm.mavis.authority import MAVISAuthority

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _prior() -> SourceBackgroundPrior:
    return SourceBackgroundPrior(
        outer_domain="d6",
        source_domains=("d2", "d3", "d4", "d5"),
        fit_specimen_ids=("s2", "s3", "s4", "s5"),
        source_authority_sha256="a" * 64,
        domain_border_medians=np.full((4, 3), 80.0),
        background_rgb=np.full(3, 80, dtype=np.uint8),
    )


def _hypothesis() -> SurfaceHypothesis:
    scores = np.linspace(0.0, 1.0, 64)
    scores.setflags(write=False)
    median = np.zeros(3)
    median.setflags(write=False)
    return SurfaceHypothesis(scores, tuple(range(63, 55, -1)), median, "b" * 64)


def test_field_teacher_retains_every_legal_candidate_and_selected_action() -> None:
    rows, columns = np.indices((41, 43))
    image = np.stack((4 * rows, 3 * columns, rows + columns), axis=2).astype(np.uint8)
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("d1-sample",),
        dataset_ids=("d1",),
        images=(image,),
        targets=np.asarray([0.4]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )
    surface = np.zeros((80, 80, 3), dtype=np.uint8)
    grid = build_deployment_grid(image.shape[:2])
    world = CausalInspectionWorld(
        authority,
        specimen_id="d1-sample",
        task=InspectionTask.FIELD,
        surface_rgb=surface,
        surface_sha256=hashlib.sha256(surface.tobytes()).hexdigest(),
        grid=grid,
        endpoint_budget=0.25,
    )
    observation = world.reset()
    for cell in PRIMARY_WARM_START_CELLS:
        observation = world.step(observation, canonical_action_from_slot(cell))
    authorization = authorize_source_teacher(
        build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1"),
        query_domain="d1",
    )
    label = field_teacher_label(
        observation,
        grid,
        _prior(),
        _hypothesis(),
        authorization,
        full_scan=image,
        policy_state_sha256="c" * 64,
    )
    assert label.task is InspectionTask.FIELD
    assert len(label.candidates) == 64
    assert len({candidate.slot for candidate in label.candidates}) == 64
    assert label.selected_slot in {candidate.slot for candidate in label.candidates}
    assert sum(candidate.selected for candidate in label.candidates) == 1
    selected = next(candidate for candidate in label.candidates if candidate.selected)
    assert selected.slot == label.selected_slot
    assert selected.objective_value == max(
        candidate.objective_value for candidate in label.candidates
    )
    assert {candidate.decision for candidate in label.candidates} >= {
        InspectionDecision.FOCUS,
        InspectionDecision.BROADEN,
        InspectionDecision.REFINE,
    }
    assert all(candidate.exact_added_cost > 0 for candidate in label.candidates)
