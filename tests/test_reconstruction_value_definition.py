from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mva.appearance_value import appearance_intensity_value
from cmc_bbdm.mva.reconstruction_value import (
    normalized_rgb_mse,
    reconstruction_value,
)


def test_normalized_rgb_mse_uses_all_rgb_values_and_255_scale() -> None:
    reference = np.zeros((2, 2, 3), dtype=np.uint8)
    reconstruction = reference.copy()
    reconstruction[0, 0, 0] = 255

    assert normalized_rgb_mse(reference, reconstruction) == pytest.approx(1.0 / 12.0)


def test_reconstruction_value_is_current_minus_candidate_error() -> None:
    reference = np.zeros((2, 2, 3), dtype=np.uint8)
    current = np.full_like(reference, 100)
    candidate = np.full_like(reference, 50)

    value = reconstruction_value(reference, current, candidate)

    expected = (100.0 / 255.0) ** 2 - (50.0 / 255.0) ** 2
    assert value == pytest.approx(expected)
    assert reconstruction_value(reference, candidate, current) == pytest.approx(-value)


def test_appearance_value_uses_only_new_locations_and_border_median() -> None:
    image = np.full((5, 5, 3), 10, dtype=np.uint8)
    image[2, 2] = (110, 10, 10)
    current = np.zeros((5, 5), dtype=np.bool_)
    candidate = current.copy()
    candidate[2, 2] = True

    value = appearance_intensity_value(image, current, candidate)

    assert value == pytest.approx(100.0 / (3.0 * 255.0))
    assert appearance_intensity_value(image, candidate, candidate) == 0.0
