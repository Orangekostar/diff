from __future__ import annotations

import numpy as np

from cmc_bbdm.mva.figures import REPRESENTATIVE_SPECIMEN, percentile_map


def test_percentile_map_is_aligned_finite_and_tie_stable() -> None:
    values = {0: 2.0, 1: 1.0, 2: 1.0, 63: -1.0}

    output = percentile_map(values)

    assert output.shape == (8, 8)
    assert np.all(np.isfinite(output))
    assert output[0, 0] == 1.0
    assert output[0, 1] == output[0, 2]
    assert output[7, 7] == 0.0
    assert REPRESENTATIVE_SPECIMEN == "c8-2"
