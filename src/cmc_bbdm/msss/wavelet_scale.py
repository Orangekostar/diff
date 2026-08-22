"""Audited cumulative DWT reconstructions for the MSSS wavelet axis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pywt


class MSSSWaveletError(ValueError):
    """Raised when a wavelet condition or reconstruction is invalid."""


WAVELET_FAMILIES = ("db2", "haar", "db4")
WAVELET_LEVELS = (0, 1, 2, 3)
WAVELET_MODES = ("low_only", "low_plus_boundary_details")


@dataclass(frozen=True, slots=True)
class WaveletScaleRecord:
    wavelet: str
    level: int
    mode: str
    boundary_mode: str
    native_height: int
    native_width: int
    reconstruction_shape: tuple[int, int, int]
    retained_approximation: bool
    retained_detail_levels: tuple[int, ...]
    coefficient_sha256: str
    input_sha256: str
    output_sha256: str
    shape_preserved: bool
    dtype_preserved: bool


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
        raise MSSSWaveletError("image must be uint8[H,W,3] RGB")
    return np.ascontiguousarray(value)


def _condition(
    wavelet: object, level: object, mode: object, shape: tuple[int, int]
) -> tuple[str, int, str]:
    if type(wavelet) is not str or wavelet not in WAVELET_FAMILIES:
        raise MSSSWaveletError("wavelet condition is not registered")
    if isinstance(level, bool) or type(level) is not int or level not in WAVELET_LEVELS:
        raise MSSSWaveletError("wavelet level is not registered")
    if type(mode) is not str or mode not in WAVELET_MODES:
        raise MSSSWaveletError("wavelet mode is not registered")
    maximum = pywt.dwt_max_level(min(shape), pywt.Wavelet(wavelet).dec_len)
    if level > maximum:
        raise MSSSWaveletError("wavelet level is not registered for this image")
    return wavelet, level, mode


def _coefficient_digest(coefficients: list[object]) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(coefficients):
        arrays = (value,) if index == 0 else tuple(value)
        for band, array in enumerate(arrays):
            contiguous = np.ascontiguousarray(array, dtype="<f8")
            digest.update(f"{index}:{band}:{contiguous.shape}".encode("ascii"))
            digest.update(b"\0")
            digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _retained_coefficients(
    coefficients: list[object], *, mode: str
) -> list[object]:
    retained: list[object] = [np.asarray(coefficients[0], dtype=np.float64)]
    for index, detail in enumerate(coefficients[1:], start=1):
        if mode == "low_plus_boundary_details" and index == 1:
            retained.append(tuple(np.asarray(item, dtype=np.float64) for item in detail))
        else:
            retained.append(tuple(np.zeros_like(item, dtype=np.float64) for item in detail))
    return retained


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.uint8)
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).reshape(array.shape)
    output.setflags(write=False)
    return output


def wavelet_scale(
    image: np.ndarray,
    *,
    wavelet: str,
    level: int,
    mode: str,
) -> tuple[np.ndarray, WaveletScaleRecord]:
    """Reconstruct a registered cumulative candidate band on the native grid."""

    source = _image(image)
    family, decomposition_level, reconstruction_mode = _condition(
        wavelet, level, mode, source.shape[:2]
    )
    if decomposition_level == 0:
        output = _readonly(source)
        digest = hashlib.sha256(b"level-zero-full").hexdigest()
        details: tuple[int, ...] = ()
    else:
        channels: list[np.ndarray] = []
        coefficient_digest = hashlib.sha256()
        for channel in np.moveaxis(source.astype(np.float64), 2, 0):
            coefficients = pywt.wavedec2(
                channel,
                wavelet=family,
                level=decomposition_level,
                mode="periodization",
            )
            retained = _retained_coefficients(
                coefficients, mode=reconstruction_mode
            )
            coefficient_digest.update(_coefficient_digest(retained).encode("ascii"))
            restored = pywt.waverec2(
                retained, wavelet=family, mode="periodization"
            )
            channels.append(restored[: source.shape[0], : source.shape[1]])
        values = np.moveaxis(np.stack(channels), 0, 2)
        if values.shape != source.shape or not np.all(np.isfinite(values)):
            raise MSSSWaveletError("wavelet reconstruction is invalid")
        output = _readonly(
            np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)
        )
        digest = coefficient_digest.hexdigest()
        details = (
            (decomposition_level,)
            if reconstruction_mode == "low_plus_boundary_details"
            else ()
        )
    shape_preserved = output.shape == source.shape
    dtype_preserved = output.dtype == source.dtype == np.dtype(np.uint8)
    if not (shape_preserved and dtype_preserved):
        raise MSSSWaveletError("wavelet reconstruction invariant failed")
    return output, WaveletScaleRecord(
        wavelet=family,
        level=decomposition_level,
        mode=reconstruction_mode,
        boundary_mode="periodization",
        native_height=int(source.shape[0]),
        native_width=int(source.shape[1]),
        reconstruction_shape=tuple(int(value) for value in output.shape),
        retained_approximation=True,
        retained_detail_levels=details,
        coefficient_sha256=digest,
        input_sha256=_sha256(source),
        output_sha256=_sha256(output),
        shape_preserved=shape_preserved,
        dtype_preserved=dtype_preserved,
    )


__all__ = [
    "WAVELET_FAMILIES",
    "WAVELET_LEVELS",
    "WAVELET_MODES",
    "MSSSWaveletError",
    "WaveletScaleRecord",
    "wavelet_scale",
]
