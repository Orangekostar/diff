from __future__ import annotations

import numpy as np

from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    fit_source_background_prior,
)
from cmc_bbdm.mavis.authority import MAVISAuthority


def _authority(target_color: int) -> MAVISAuthority:
    colors = (10, 20, 100, target_color)
    domains = ("source-a", "source-a", "source-b", "target")
    images = tuple(
        np.full((41, 43, 3), color, dtype=np.uint8) for color in colors
    )
    count = len(images)
    return MAVISAuthority.from_arrays(
        specimen_ids=tuple(f"s{index}" for index in range(count)),
        dataset_ids=domains,
        images=images,
        targets=np.linspace(0.1, 0.4, count),
        metadata13=np.zeros((count, 13), dtype=np.float64),
        profile_stats21=np.zeros((count, 21), dtype=np.float64),
    )


def test_source_prior_equal_weights_domains_and_excludes_outer_target() -> None:
    first = fit_source_background_prior(_authority(0), outer_domain="target")
    second = fit_source_background_prior(_authority(255), outer_domain="target")

    # source-a median mean is 15 and source-b is 100; equal-domain mean is 57.5,
    # rounded to even uint8 58.
    np.testing.assert_array_equal(first.background_rgb, (58, 58, 58))
    np.testing.assert_array_equal(first.background_rgb, second.background_rgb)
    assert first.source_domains == ("source-a", "source-b")
    assert first.fit_specimen_ids == ("s0", "s1", "s2")
    assert "s3" not in first.fit_specimen_ids
    assert first.outer_domain == "target"
    assert not first.background_rgb.flags.writeable
