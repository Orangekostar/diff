from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.inspection_agent_g1.crossfit import (
    G1CrossfitError,
    build_crossfit_roster,
    fit_crossfit_source_prior,
)
from cmc_bbdm.mavis.authority import MAVISAuthority

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _authority(*, excluded_offset: int = 0) -> MAVISAuthority:
    specimen_ids = tuple(f"{domain}-s{index}" for domain in DOMAINS for index in range(2))
    dataset_ids = tuple(domain for domain in DOMAINS for _ in range(2))
    images = []
    targets = []
    for domain_index, domain in enumerate(DOMAINS):
        value = 20 * (domain_index + 1)
        if domain in {"d1", "d6"}:
            value += excluded_offset
        for specimen_index in range(2):
            images.append(np.full((41, 43, 3), value + specimen_index, dtype=np.uint8))
            targets.append(domain_index + specimen_index / 10)
    return MAVISAuthority.from_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        images=tuple(images),
        targets=np.asarray(targets),
        metadata13=np.zeros((12, 13)),
        profile_stats21=np.zeros((12, 21)),
    )


def test_crossfit_roster_excludes_outer_and_labeled_domains() -> None:
    roster = build_crossfit_roster(
        DOMAINS,
        outer_target="d6",
        labeled_domain="d1",
    )
    assert roster.outer_target == "d6"
    assert roster.labeled_domain == "d1"
    assert roster.fit_domains == ("d2", "d3", "d4", "d5")
    assert len(roster.state_sha256) == 64


def test_crossfit_prior_reads_only_the_registered_four_domains() -> None:
    first = fit_crossfit_source_prior(
        _authority(excluded_offset=0),
        outer_target="d6",
        labeled_domain="d1",
    )
    changed_excluded = fit_crossfit_source_prior(
        _authority(excluded_offset=7),
        outer_target="d6",
        labeled_domain="d1",
    )
    assert first.roster.fit_domains == ("d2", "d3", "d4", "d5")
    assert first.prior.source_domains == first.roster.fit_domains
    assert all(
        specimen.startswith(("d2-", "d3-", "d4-", "d5-"))
        for specimen in first.prior.fit_specimen_ids
    )
    np.testing.assert_array_equal(first.prior.background_rgb, (70, 70, 70))
    assert first.prior.state_sha256 == changed_excluded.prior.state_sha256
    assert first.state_sha256 == changed_excluded.state_sha256


@pytest.mark.parametrize(
    ("outer_target", "labeled_domain"),
    (("d6", "d6"), ("missing", "d1"), ("d6", "missing")),
)
def test_crossfit_roster_rejects_invalid_domain_roles(
    outer_target: str, labeled_domain: str
) -> None:
    with pytest.raises(G1CrossfitError):
        build_crossfit_roster(
            DOMAINS,
            outer_target=outer_target,
            labeled_domain=labeled_domain,
        )
