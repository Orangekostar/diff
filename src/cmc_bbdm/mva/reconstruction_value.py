"""Registered reconstruction-error value for retrospective acquisition."""

from __future__ import annotations

import numpy as np


def _rgb(value: object, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError(f"{label} must be an RGB uint8 array")
    if value.ndim != 3 or value.shape[2] != 3 or not value.size:
        raise ValueError(f"{label} must be an RGB uint8 array")
    return value


def normalized_rgb_mse(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    """Mean squared RGB error after scaling each channel by 255."""

    target = _rgb(reference, "reference")
    estimate = _rgb(reconstruction, "reconstruction")
    if target.shape != estimate.shape:
        raise ValueError("reference and reconstruction shapes differ")
    difference = (
        target.astype(np.float64, copy=False) - estimate.astype(np.float64, copy=False)
    ) / 255.0
    return float(np.mean(np.square(difference), dtype=np.float64))


def reconstruction_value(
    reference: np.ndarray,
    current_reconstruction: np.ndarray,
    candidate_reconstruction: np.ndarray,
) -> float:
    """Return current normalized MSE minus candidate normalized MSE."""

    return normalized_rgb_mse(reference, current_reconstruction) - normalized_rgb_mse(
        reference, candidate_reconstruction
    )


__all__ = ["normalized_rgb_mse", "reconstruction_value"]
