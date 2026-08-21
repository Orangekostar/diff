"""Frequency decomposition and registered residual controls for D8."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter

from .residuals import P6ResidualBank, ResidualRecord

_FIELD_SHAPE = (3, 64, 64)
_BANDS = frozenset({"low", "mid", "high", "mid+high"})
_WAVELETS = frozenset({"haar", "db2", "db4", "sym4"})


def _field(value: object, *, label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if array.shape != _FIELD_SHAPE or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite float32 (3, 64, 64) field")
    return np.array(array, dtype=np.float32, copy=True, order="C")


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=np.float32, copy=True, order="C")
    output.setflags(write=False)
    return output


def _exact_keys(parameters: Mapping[str, object], *, expected: frozenset[str]) -> None:
    if type(parameters) is not dict or frozenset(parameters) != expected:
        raise ValueError("decomposition parameters do not match the registered schema")


def _finite_float(value: object, *, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_seed(seed: object) -> int:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return seed


def _state_digest(
    *,
    family: str,
    selected_band: str,
    parameters: Mapping[str, object],
    arrays: tuple[np.ndarray, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(family.encode("ascii"))
    digest.update(b"\0")
    digest.update(selected_band.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            parameters,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )
    digest.update(b"\0")
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ResidualBands:
    """One immutable three-band decomposition and its selected residual."""

    family: str
    selected_band: str
    selected: np.ndarray
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray
    energy_fraction: float
    reconstruction_error: float
    state_sha256: str


def _gaussian_bands(
    residual: np.ndarray, parameters: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _exact_keys(parameters, expected=frozenset({"band", "sigma"}))
    sigma = _finite_float(parameters["sigma"], label="Gaussian sigma")
    if not 0.5 <= sigma <= 8.0:
        raise ValueError("Gaussian sigma must lie in [0.5, 8.0]")
    inner_sigma = max(sigma / 2.0, 0.25)
    outer = gaussian_filter(
        residual.astype(np.float64), sigma=(0.0, sigma, sigma), mode="reflect"
    )
    inner = gaussian_filter(
        residual.astype(np.float64),
        sigma=(0.0, inner_sigma, inner_sigma),
        mode="reflect",
    )
    low = outer.astype(np.float32)
    mid = (inner - outer).astype(np.float32)
    high = residual - low - mid
    return low, mid, high


def _fourier_bands(
    residual: np.ndarray, parameters: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _exact_keys(parameters, expected=frozenset({"band", "cutoff", "transition"}))
    cutoff = _finite_float(parameters["cutoff"], label="Fourier cutoff")
    transition = _finite_float(parameters["transition"], label="Fourier transition")
    if not 0.04 <= cutoff <= 0.50 or not 0.01 <= transition <= 0.10:
        raise ValueError("Fourier cutoff or transition is outside the registered range")

    frequency = np.fft.fftfreq(64)
    radius = np.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
    lower_edge = max(cutoff - transition, 0.0)
    low_weight = np.zeros_like(radius)
    low_weight[radius <= lower_edge] = 1.0
    low_transition = (radius > lower_edge) & (radius < cutoff)
    low_width = cutoff - lower_edge
    low_position = (radius[low_transition] - lower_edge) / low_width
    low_weight[low_transition] = 0.5 * (
        1.0 + np.cos(np.pi * low_position)
    )

    high_weight = np.zeros_like(radius)
    high_weight[radius >= cutoff + transition] = 1.0
    high_transition = (radius > cutoff) & (radius < cutoff + transition)
    high_position = (radius[high_transition] - cutoff) / transition
    high_weight[high_transition] = 0.5 * (1.0 - np.cos(np.pi * high_position))
    mid_weight = 1.0 - low_weight - high_weight

    spectrum = np.fft.fft2(residual.astype(np.float64), axes=(-2, -1))
    low = np.fft.ifft2(spectrum * low_weight, axes=(-2, -1)).real.astype(np.float32)
    mid = np.fft.ifft2(spectrum * mid_weight, axes=(-2, -1)).real.astype(np.float32)
    high = residual - low - mid
    return low, mid, high


def _wavelet_component(
    coefficients: list[object],
    *,
    wavelet: str,
    approximation: bool,
    detail_index: int | None,
) -> np.ndarray:
    selected: list[object] = [
        np.asarray(coefficients[0]) if approximation else np.zeros_like(coefficients[0])
    ]
    for index, detail in enumerate(coefficients[1:], start=1):
        if index == detail_index:
            selected.append(tuple(np.asarray(value) for value in detail))
        else:
            selected.append(tuple(np.zeros_like(value) for value in detail))
    return pywt.waverec2(selected, wavelet=wavelet, mode="periodization")


def _wavelet_bands(
    residual: np.ndarray, parameters: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _exact_keys(parameters, expected=frozenset({"band", "wavelet", "level"}))
    wavelet = parameters["wavelet"]
    level = parameters["level"]
    if type(wavelet) is not str or wavelet not in _WAVELETS:
        raise ValueError("wavelet family is not registered")
    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("wavelet level must be 1, 2, or 3")

    low_channels: list[np.ndarray] = []
    mid_channels: list[np.ndarray] = []
    for channel in residual.astype(np.float64):
        coefficients = pywt.wavedec2(
            channel, wavelet=wavelet, level=level, mode="periodization"
        )
        low = _wavelet_component(
            coefficients,
            wavelet=wavelet,
            approximation=True,
            detail_index=None,
        )
        mid = np.zeros_like(channel)
        for detail_index in range(1, len(coefficients) - 1):
            mid += _wavelet_component(
                coefficients,
                wavelet=wavelet,
                approximation=False,
                detail_index=detail_index,
            )
        low_channels.append(low[:64, :64])
        mid_channels.append(mid[:64, :64])
    low_array = np.stack(low_channels).astype(np.float32)
    mid_array = np.stack(mid_channels).astype(np.float32)
    high_array = residual - low_array - mid_array
    return low_array, mid_array, high_array


def decompose_residual(
    residual: np.ndarray, *, family: str, parameters: Mapping[str, object]
) -> ResidualBands:
    """Return Gaussian, raised-cosine Fourier, or wavelet frequency bands."""

    source = _field(residual, label="residual")
    if type(family) is not str or type(parameters) is not dict:
        raise ValueError("decomposition family and parameters are invalid")
    band = parameters.get("band")
    if type(band) is not str or band not in _BANDS:
        raise ValueError("selected residual band is not registered")
    if family == "gaussian":
        low, mid, high = _gaussian_bands(source, parameters)
    elif family == "fourier":
        low, mid, high = _fourier_bands(source, parameters)
    elif family == "wavelet":
        low, mid, high = _wavelet_bands(source, parameters)
    else:
        raise ValueError("decomposition family is not registered")

    low = _readonly(low)
    mid = _readonly(mid)
    high = _readonly(source - low - mid)
    selected_values = {
        "low": low,
        "mid": mid,
        "high": high,
        "mid+high": mid + high,
    }
    selected = _readonly(selected_values[band])
    reconstruction_error = float(
        np.max(np.abs((low + mid + high).astype(np.float64) - source))
    )
    source_energy = float(np.sum(np.square(source, dtype=np.float64)))
    selected_energy = float(np.sum(np.square(selected, dtype=np.float64)))
    energy_fraction = 0.0 if source_energy == 0.0 else selected_energy / source_energy
    state_sha256 = _state_digest(
        family=family,
        selected_band=band,
        parameters=parameters,
        arrays=(selected, low, mid, high),
    )
    return ResidualBands(
        family=family,
        selected_band=band,
        selected=selected,
        low=low,
        mid=mid,
        high=high,
        energy_fraction=energy_fraction,
        reconstruction_error=reconstruction_error,
        state_sha256=state_sha256,
    )


def gaussian_control(residual: np.ndarray, *, seed: int) -> np.ndarray:
    """Return deterministic Gaussian noise with channel-wise mean and variance."""

    source = _field(residual, label="residual")
    generator = np.random.Generator(np.random.PCG64(_positive_seed(seed)))
    output = np.empty_like(source, dtype=np.float64)
    for index, channel in enumerate(source.astype(np.float64)):
        target_mean = float(np.mean(channel))
        target_std = float(np.std(channel))
        if target_std == 0.0:
            output[index].fill(target_mean)
            continue
        noise = generator.normal(size=(64, 64))
        noise = (noise - np.mean(noise)) / np.std(noise)
        output[index] = target_mean + target_std * noise
    return _readonly(output)


def phase_randomized_control(residual: np.ndarray, *, seed: int) -> np.ndarray:
    """Return a deterministic real field with the residual amplitude spectrum."""

    source = _field(residual, label="residual")
    generator = np.random.Generator(np.random.PCG64(_positive_seed(seed)))
    output: list[np.ndarray] = []
    for channel in source.astype(np.float64):
        source_spectrum = np.fft.fft2(channel)
        noise_spectrum = np.fft.fft2(generator.normal(size=(64, 64)))
        denominator = np.abs(noise_spectrum)
        phase = np.divide(
            noise_spectrum,
            denominator,
            out=np.ones_like(noise_spectrum),
            where=denominator > 0.0,
        )
        source_dc = source_spectrum[0, 0]
        phase[0, 0] = 1.0 if source_dc.real >= 0.0 else -1.0
        randomized = np.abs(source_spectrum) * phase
        output.append(np.fft.ifft2(randomized).real)
    return _readonly(np.stack(output))


def _identity_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} must be a nonempty tuple")
    if any(type(item) is not str or not item or item.strip() != item for item in value):
        raise ValueError(f"{label} contains an invalid identity")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must be unique")
    return value


def empirical_control(
    bank: P6ResidualBank,
    *,
    fit_domains: tuple[str, ...],
    query_ids: tuple[str, ...],
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Sample fit-domain residual donors without specimen self-donation."""

    if type(bank) is not P6ResidualBank or not isinstance(bank.records, tuple):
        raise ValueError("an exact P6 residual bank is required")
    domains = _identity_tuple(fit_domains, label="fit domains")
    queries = _identity_tuple(query_ids, label="query IDs")
    if any(type(record) is not ResidualRecord for record in bank.records):
        raise ValueError("residual bank contains an invalid record")
    generator = np.random.Generator(np.random.PCG64(_positive_seed(seed)))
    outputs: list[np.ndarray] = []
    for query_id in queries:
        eligible = tuple(
            record
            for record in bank.records
            if record.dataset_id in domains and record.specimen_id != query_id
        )
        if not eligible:
            raise ValueError("no fit-domain empirical donor remains after exclusion")
        index = int(generator.integers(0, len(eligible)))
        outputs.append(_readonly(eligible[index].residual_64))
    return tuple(outputs)


__all__ = [
    "ResidualBands",
    "decompose_residual",
    "empirical_control",
    "gaussian_control",
    "phase_randomized_control",
]
