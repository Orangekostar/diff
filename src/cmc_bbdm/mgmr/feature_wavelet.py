"""Deterministic one-level DWT for frozen NCHW feature maps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pywt


class FeatureWaveletError(ValueError):
    """Raised when a feature-map wavelet contract is violated."""


_WAVELETS = ("db2", "haar")
_MODE = "periodization"
_SPATIAL_SHAPES = {(14, 14), (28, 28)}


def _feature_maps(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise FeatureWaveletError("feature maps must use float32")
    if array.ndim != 4:
        raise FeatureWaveletError("feature maps must be NCHW")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise FeatureWaveletError("feature maps must have nonempty N and C axes")
    if tuple(array.shape[-2:]) not in _SPATIAL_SHAPES:
        raise FeatureWaveletError("feature-map size must be 14x14 or 28x28")
    if not np.all(np.isfinite(array)):
        raise FeatureWaveletError("feature maps must be finite")
    return np.ascontiguousarray(array, dtype=np.float32)


def _readonly(value: object, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32) or not np.all(np.isfinite(array)):
        raise FeatureWaveletError("wavelet coefficients must be finite float32")
    if shape is not None and array.shape != shape:
        raise FeatureWaveletError("wavelet coefficient shapes do not align")
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float32).reshape(
        contiguous.shape
    )
    output.setflags(write=False)
    return output


def _state_hash(
    coefficients: tuple[np.ndarray, ...],
    *,
    source_shape: tuple[int, int, int, int],
    wavelet: str,
    mode: str,
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "dtype": "<f4",
                "mode": mode,
                "source_shape": source_shape,
                "wavelet": wavelet,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for name, array in zip(("LL", "cH", "cV", "cD"), coefficients, strict=True):
        digest.update(name.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureWaveletBands:
    ll: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    diagonal: np.ndarray
    source_shape: tuple[int, int, int, int]
    wavelet: str
    mode: str
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if self.wavelet not in _WAVELETS:
            raise FeatureWaveletError("wavelet is not registered")
        if self.mode != _MODE:
            raise FeatureWaveletError("wavelet mode is not registered")
        if (
            len(self.source_shape) != 4
            or tuple(self.source_shape[-2:]) not in _SPATIAL_SHAPES
        ):
            raise FeatureWaveletError("source NCHW shape is invalid")
        expected = (
            int(self.source_shape[0]),
            int(self.source_shape[1]),
            int(self.source_shape[2]) // 2,
            int(self.source_shape[3]) // 2,
        )
        coefficients = tuple(
            _readonly(value, shape=expected)
            for value in (self.ll, self.horizontal, self.vertical, self.diagonal)
        )
        state = _state_hash(
            coefficients,
            source_shape=self.source_shape,
            wavelet=self.wavelet,
            mode=self.mode,
        )
        if self.state_sha256 and self.state_sha256 != state:
            raise FeatureWaveletError("wavelet state SHA-256 changed")
        object.__setattr__(self, "ll", coefficients[0])
        object.__setattr__(self, "horizontal", coefficients[1])
        object.__setattr__(self, "vertical", coefficients[2])
        object.__setattr__(self, "diagonal", coefficients[3])
        object.__setattr__(self, "state_sha256", state)

    @property
    def coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (self.ll, self.horizontal, self.vertical, self.diagonal)


def dwt2_feature_maps(
    maps: object, *, wavelet: str = "db2", mode: str = "periodization"
) -> FeatureWaveletBands:
    """Apply a one-level separable DWT over the final two NCHW axes."""

    if wavelet not in _WAVELETS:
        raise FeatureWaveletError("wavelet must be db2 or haar")
    if mode != _MODE:
        raise FeatureWaveletError("wavelet mode must be periodization")
    source = _feature_maps(maps)
    try:
        ll, (horizontal, vertical, diagonal) = pywt.dwt2(
            source,
            wavelet=wavelet,
            mode=mode,
            axes=(-2, -1),
        )
    except (TypeError, ValueError) as error:
        raise FeatureWaveletError("feature-map DWT failed") from error
    return FeatureWaveletBands(
        ll=ll,
        horizontal=horizontal,
        vertical=vertical,
        diagonal=diagonal,
        source_shape=tuple(int(value) for value in source.shape),
        wavelet=wavelet,
        mode=mode,
    )


def idwt2_feature_maps(bands: FeatureWaveletBands) -> np.ndarray:
    """Reconstruct the original NCHW feature maps from all four bands."""

    if type(bands) is not FeatureWaveletBands:
        raise FeatureWaveletError("issued feature wavelet bands are required")
    try:
        rebuilt = pywt.idwt2(
            (
                bands.ll,
                (bands.horizontal, bands.vertical, bands.diagonal),
            ),
            wavelet=bands.wavelet,
            mode=bands.mode,
            axes=(-2, -1),
        )
    except (TypeError, ValueError) as error:
        raise FeatureWaveletError("feature-map inverse DWT failed") from error
    return _readonly(rebuilt, shape=bands.source_shape)


def directional_gap(bands: FeatureWaveletBands) -> np.ndarray:
    """Concatenate channel-wise GAP of cH, cV, and cD without band averaging."""

    if type(bands) is not FeatureWaveletBands:
        raise FeatureWaveletError("issued feature wavelet bands are required")
    feature = np.concatenate(
        [
            np.mean(bands.horizontal, axis=(-2, -1), dtype=np.float32),
            np.mean(bands.vertical, axis=(-2, -1), dtype=np.float32),
            np.mean(bands.diagonal, axis=(-2, -1), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    return _readonly(feature)


__all__ = [
    "FeatureWaveletBands",
    "FeatureWaveletError",
    "directional_gap",
    "dwt2_feature_maps",
    "idwt2_feature_maps",
]
