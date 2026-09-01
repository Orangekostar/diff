from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent_g1.crossfit import build_crossfit_roster
from cmc_bbdm.inspection_agent_g1.teacher import (
    G1TeacherError,
    authorize_source_teacher,
    validate_source_teacher_dependencies,
)

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def test_source_teacher_authorization_binds_dual_exclusion_roster() -> None:
    roster = build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1")
    authorization = authorize_source_teacher(roster, query_domain="d1")
    assert authorization.outer_target == "d6"
    assert authorization.labeled_domain == "d1"
    assert authorization.fit_domains == ("d2", "d3", "d4", "d5")
    assert len(authorization.state_sha256) == 64


@pytest.mark.parametrize("query_domain", ("d6", "d2", "missing"))
def test_target_or_wrong_source_teacher_access_is_rejected(query_domain: str) -> None:
    roster = build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1")
    with pytest.raises(G1TeacherError, match="labeled source domain"):
        authorize_source_teacher(roster, query_domain=query_domain)


def test_teacher_dependency_roster_cannot_include_the_labeled_domain() -> None:
    roster = build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1")
    authorization = authorize_source_teacher(roster, query_domain="d1")
    contaminated = SourceBackgroundPrior(
        outer_domain="d6",
        source_domains=("d1", "d2", "d3", "d4"),
        fit_specimen_ids=("s1", "s2", "s3", "s4"),
        source_authority_sha256="a" * 64,
        domain_border_medians=np.zeros((4, 3)),
        background_rgb=np.zeros(3, dtype=np.uint8),
    )
    with pytest.raises(G1TeacherError, match="do not match authorization"):
        validate_source_teacher_dependencies(authorization, contaminated)
