from __future__ import annotations

import numpy as np

from cmc_bbdm.inspection_agent.surface_hypothesis import (
    compute_surface_hypothesis,
)


def _boxes() -> np.ndarray:
    values = []
    for row in range(8):
        for column in range(8):
            values.append(
                (
                    float(column * 10),
                    float(row * 10),
                    float(79 if column == 7 else (column + 1) * 10),
                    float(79 if row == 7 else (row + 1) * 10),
                )
            )
    return np.asarray(values, dtype=np.float64)


def test_surface_hypothesis_uses_only_border_deviation_and_fixed_top_k() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[30:40, 30:40] = (120, 60, 30)
    image[40:50, 40:50] = (60, 30, 15)

    result = compute_surface_hypothesis(image, _boxes(), top_k=8)

    assert result.top_cells[0] == 27
    assert result.top_cells[1] == 36
    assert result.scores[27] == 1.0
    assert 0.0 < result.scores[36] < 1.0
    assert len(result.top_cells) == 8
    assert not result.scores.flags.writeable


def test_constant_surface_has_zero_scores_and_deterministic_cell_ties() -> None:
    image = np.full((80, 80, 3), 91, dtype=np.uint8)
    result = compute_surface_hypothesis(image, _boxes(), top_k=8)
    np.testing.assert_array_equal(result.scores, np.zeros(64))
    assert result.top_cells == tuple(range(8))
