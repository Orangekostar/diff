from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent_g1.rollout import (
    SurfaceVariant,
    controlled_surface_hypothesis,
    shuffled_surface_donors,
)


def _hypothesis(
    scores: np.ndarray,
    top_cells: tuple[int, ...],
    border_median: tuple[float, float, float],
    state_sha256: str,
) -> SurfaceHypothesis:
    scores = np.asarray(scores, dtype=np.float64)
    scores.setflags(write=False)
    median = np.asarray(border_median, dtype=np.float64)
    median.setflags(write=False)
    return SurfaceHypothesis(scores, top_cells, median, state_sha256)


def _recipient_and_donor() -> tuple[SurfaceHypothesis, SurfaceHypothesis]:
    recipient = _hypothesis(
        np.linspace(0.0, 1.0, 64),
        (63, 62, 61, 60),
        (11.0, 22.0, 33.0),
        "a" * 64,
    )
    donor = _hypothesis(
        np.linspace(1.0, 0.0, 64),
        (0, 1, 2, 3),
        (99.0, 88.0, 77.0),
        "b" * 64,
    )
    return recipient, donor


def test_surface_variants_preserve_only_the_registered_evidence() -> None:
    recipient, donor = _recipient_and_donor()

    correct = controlled_surface_hypothesis(
        recipient, SurfaceVariant.CORRECT_SURFACE
    )
    assert correct is recipient

    no_surface = controlled_surface_hypothesis(
        recipient, SurfaceVariant.NO_SURFACE
    )
    repeated = controlled_surface_hypothesis(
        recipient, SurfaceVariant.NO_SURFACE
    )
    np.testing.assert_array_equal(no_surface.scores, np.zeros(64))
    assert no_surface.top_cells == ()
    np.testing.assert_array_equal(
        no_surface.border_median_rgb, recipient.border_median_rgb
    )
    assert no_surface.state_sha256 == repeated.state_sha256
    assert len(no_surface.state_sha256) == 64

    shuffled = controlled_surface_hypothesis(
        recipient, SurfaceVariant.SHUFFLED_SURFACE, donor=donor
    )
    np.testing.assert_array_equal(shuffled.scores, donor.scores)
    assert shuffled.top_cells == donor.top_cells
    np.testing.assert_array_equal(
        shuffled.border_median_rgb, recipient.border_median_rgb
    )
    assert shuffled.state_sha256 not in {
        recipient.state_sha256,
        donor.state_sha256,
    }


def test_shuffled_surface_requires_a_nonself_donor() -> None:
    recipient, _donor = _recipient_and_donor()

    with pytest.raises(ValueError):
        controlled_surface_hypothesis(
            recipient, SurfaceVariant.SHUFFLED_SURFACE
        )
    with pytest.raises(ValueError):
        controlled_surface_hypothesis(
            recipient, SurfaceVariant.SHUFFLED_SURFACE, donor=recipient
        )


def test_shuffled_surface_donors_are_deterministic_domain_local_and_nonself() -> None:
    specimens = ("a0", "a1", "b0", "b1")
    domains = ("domain-a", "domain-a", "domain-b", "domain-b")
    first = shuffled_surface_donors(specimens, domains, seed=20260901)
    second = shuffled_surface_donors(specimens, domains, seed=20260901)

    assert first == second
    assert sorted(first) == sorted(specimens)
    domain_by_specimen = dict(zip(specimens, domains, strict=True))
    assert all(
        recipient != donor
        and domain_by_specimen[recipient] == domain_by_specimen[donor]
        for recipient, donor in zip(specimens, first, strict=True)
    )
