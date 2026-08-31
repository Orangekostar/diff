"""Internal-signal discovery and full-field task metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.mva.reconstruction_value import normalized_rgb_mse


class FieldTaskError(ValueError):
    """Raised when an internal-signal or field metric request is invalid."""


def _rgb(image: object) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 2
    ):
        raise FieldTaskError("full C-scan must be RGB uint8")
    return image


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class InternalSignalSaliency:
    pixel_mass: np.ndarray
    border_median_rgb: np.ndarray
    total_mass: float


def internal_signal_saliency(full_scan: np.ndarray) -> InternalSignalSaliency:
    image = _rgb(full_scan)
    border = np.concatenate(
        (image[0], image[-1], image[1:-1, 0], image[1:-1, -1]),
        axis=0,
    )
    median = np.median(border.astype(np.float64), axis=0)
    mass = np.sum(
        np.abs(image.astype(np.float64, copy=False) - median),
        axis=2,
        dtype=np.float64,
    )
    total = float(np.sum(mass, dtype=np.float64))
    if not math.isfinite(total) or total < 0.0:
        raise FieldTaskError("internal-signal mass is invalid")
    return InternalSignalSaliency(
        pixel_mass=_readonly(mass.astype("<f8", copy=False)),
        border_median_rgb=_readonly(median.astype("<f8", copy=False)),
        total_mass=total,
    )


def signal_capture(mask: np.ndarray, saliency: InternalSignalSaliency) -> float:
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or type(saliency) is not InternalSignalSaliency
        or mask.shape != saliency.pixel_mass.shape
    ):
        raise FieldTaskError("signal-capture mask is invalid")
    if saliency.total_mass == 0.0:
        return 0.0
    value = float(np.sum(saliency.pixel_mass[mask], dtype=np.float64))
    return float(np.clip(value / saliency.total_mass, 0.0, 1.0))


def normalized_capture_auc(
    budgets: object,
    captures: object,
    *,
    scout_endpoint: float,
) -> float:
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(captures, dtype=np.float64)
    endpoint = float(scout_endpoint)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or x.size < 2
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(np.diff(x) <= 0.0)
        or x[0] != 0.0
        or not math.isfinite(endpoint)
        or endpoint <= 0.0
        or not math.isclose(x[-1], endpoint, rel_tol=0.0, abs_tol=1.0e-15)
        or np.any((y < 0.0) | (y > 1.0))
    ):
        raise FieldTaskError("capture-AUC curve is invalid")
    return float(np.trapezoid(y, x) / endpoint)


def field_loss(full_scan: np.ndarray, reconstruction: np.ndarray) -> float:
    try:
        return normalized_rgb_mse(_rgb(full_scan), _rgb(reconstruction))
    except ValueError as error:
        raise FieldTaskError("FIELD loss request is invalid") from error


__all__ = [
    "FieldTaskError",
    "InternalSignalSaliency",
    "field_loss",
    "internal_signal_saliency",
    "normalized_capture_auc",
    "signal_capture",
]
