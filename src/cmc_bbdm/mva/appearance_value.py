"""Nondeployable RGB appearance-intensity comparator for MVA."""

from __future__ import annotations

import numpy as np


def _validate_mask(value: object, shape: tuple[int, int], label: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.bool_
        or value.shape != shape
    ):
        raise ValueError(f"{label} must be a native boolean mask")
    return value


def appearance_intensity_value(
    full_image: np.ndarray,
    current_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> float:
    """Mean newly revealed RGB deviation from the full-image border median."""

    if (
        not isinstance(full_image, np.ndarray)
        or full_image.dtype != np.uint8
        or full_image.ndim != 3
        or full_image.shape[2] != 3
        or min(full_image.shape[:2]) < 2
    ):
        raise ValueError("full image must be RGB uint8")
    current = _validate_mask(current_mask, full_image.shape[:2], "current mask")
    candidate = _validate_mask(candidate_mask, full_image.shape[:2], "candidate mask")
    if np.any(current & ~candidate):
        raise ValueError("candidate mask cannot remove observations")
    newly_revealed = candidate & ~current
    if not np.any(newly_revealed):
        return 0.0
    border_mask = np.zeros(full_image.shape[:2], dtype=np.bool_)
    border_mask[[0, -1], :] = True
    border_mask[:, [0, -1]] = True
    border = full_image[border_mask].astype(np.float64, copy=False)
    median = np.median(border, axis=0)
    deviation = np.abs(
        full_image[newly_revealed].astype(np.float64, copy=False) - median
    )
    return float(np.mean(deviation, dtype=np.float64) / 255.0)


__all__ = ["appearance_intensity_value"]
