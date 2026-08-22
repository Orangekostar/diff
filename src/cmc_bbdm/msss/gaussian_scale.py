"""Deterministic native-grid Gaussian scale-space for MSSS."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


class MSSSGaussianError(ValueError):
    """Raised when a Gaussian scale condition violates the registry."""


GAUSSIAN_SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


@dataclass(frozen=True, slots=True)
class GaussianScaleRecord:
    sigma_px: float
    sigma_mm: None
    boundary_mode: str
    native_height: int
    native_width: int
    input_sha256: str
    output_sha256: str
    shape_preserved: bool
    dtype_preserved: bool
    intensity_semantics_preserved: bool


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _image(value: object) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.uint8)
        or value.ndim != 3
        or value.shape[2] != 3
        or min(value.shape[:2]) < 2
    ):
        raise MSSSGaussianError("image must be uint8[H,W,3] RGB")
    return np.ascontiguousarray(value)


def _sigma(value: object) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise MSSSGaussianError("Gaussian sigma is not registered")
    sigma = float(value)
    if not math.isfinite(sigma) or sigma not in GAUSSIAN_SIGMAS:
        raise MSSSGaussianError("Gaussian sigma is not registered")
    return sigma


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.uint8)
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).reshape(array.shape)
    output.setflags(write=False)
    return output


def gaussian_scale(
    image: np.ndarray, *, sigma_px: float
) -> tuple[np.ndarray, GaussianScaleRecord]:
    """Apply one registered spatial bandwidth without changing the pixel grid."""

    source = _image(image)
    sigma = _sigma(sigma_px)
    if sigma == 0.0:
        restored = source.copy()
    else:
        filtered = gaussian_filter(
            source.astype(np.float64),
            sigma=(sigma, sigma, 0.0),
            mode="reflect",
        )
        if not np.all(np.isfinite(filtered)):
            raise MSSSGaussianError("Gaussian filter returned non-finite values")
        restored = np.clip(np.rint(filtered), 0.0, 255.0).astype(np.uint8)
    output = _readonly(restored)
    shape_preserved = output.shape == source.shape
    dtype_preserved = output.dtype == source.dtype == np.dtype(np.uint8)
    intensity_preserved = int(output.min()) >= 0 and int(output.max()) <= 255
    if not (shape_preserved and dtype_preserved and intensity_preserved):
        raise MSSSGaussianError("Gaussian scale invariant failed")
    return output, GaussianScaleRecord(
        sigma_px=sigma,
        sigma_mm=None,
        boundary_mode="reflect",
        native_height=int(source.shape[0]),
        native_width=int(source.shape[1]),
        input_sha256=_sha256(source),
        output_sha256=_sha256(output),
        shape_preserved=shape_preserved,
        dtype_preserved=dtype_preserved,
        intensity_semantics_preserved=intensity_preserved,
    )


__all__ = ["GAUSSIAN_SIGMAS", "GaussianScaleRecord", "MSSSGaussianError", "gaussian_scale"]
