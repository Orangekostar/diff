from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.damage_response.targets import ResponseTrace

BASELINE_SAMPLES = 50
GRID_POINTS = 101
MINIMUM_UNIQUE_EXTENSION_POSITIONS = 50


class ExtractionError(RuntimeError):
    """Raised when a pre-peak response cannot satisfy the frozen P1 contract."""


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def aggregate_extension_positions(
    extension_mm: np.ndarray, stress_mpa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse exact extension duplicates by median stress."""

    extension = np.asarray(extension_mm, dtype=np.float64)
    stress = np.asarray(stress_mpa, dtype=np.float64)
    if extension.ndim != 1 or stress.ndim != 1 or extension.shape != stress.shape:
        raise ExtractionError("extension and stress must be aligned one-dimensional arrays")
    if extension.size == 0 or not np.all(np.isfinite(extension)) or not np.all(
        np.isfinite(stress)
    ):
        raise ExtractionError("extension aggregation received empty or nonfinite values")
    order = np.argsort(extension, kind="stable")
    sorted_extension = extension[order]
    sorted_stress = stress[order]
    unique, starts, counts = np.unique(
        sorted_extension, return_index=True, return_counts=True
    )
    medians = np.asarray(
        [
            np.median(sorted_stress[start : start + count])
            for start, count in zip(starts, counts, strict=True)
        ],
        dtype=np.float64,
    )
    return _readonly(unique), _readonly(medians)


@dataclass(frozen=True)
class ExtractedResponse:
    specimen_id: str
    peak_row: int
    baseline_extension_mm: float
    baseline_stress_mpa: float
    extension_peak_mm: float
    zeroed_peak_stress_mpa: float
    unique_extension_positions: int
    u: np.ndarray
    extension_mm: np.ndarray
    zeroed_stress_mpa: np.ndarray
    normalized_stress: np.ndarray
    slope_u20_u60_mpa_per_mm: float
    normalized_prepeak_auc: float
    q_midpoint: float


def extract_prepeak_response(
    response: ResponseTrace,
    *,
    baseline_samples: int = BASELINE_SAMPLES,
    grid_points: int = GRID_POINTS,
    minimum_unique_extension_positions: int = MINIMUM_UNIQUE_EXTENSION_POSITIONS,
) -> ExtractedResponse:
    """Extract the preregistered stress-extension response and descriptors."""

    if (
        not isinstance(baseline_samples, int)
        or isinstance(baseline_samples, bool)
        or baseline_samples <= 0
    ):
        raise ExtractionError("baseline sample count must be a positive integer")
    if grid_points != 101:
        raise ExtractionError("P1 response grid must contain exactly 101 points")
    if minimum_unique_extension_positions != 50:
        raise ExtractionError("P1 minimum unique extension count must remain 50")
    if response.peak_row < baseline_samples:
        raise ExtractionError("peak row occurs before the fixed baseline window")

    extension = np.asarray(response.extension_mm, dtype=np.float64)
    stress = np.asarray(response.stress_mpa, dtype=np.float64)
    if extension.ndim != 1 or stress.ndim != 1 or extension.shape != stress.shape:
        raise ExtractionError("response extension/stress arrays are not aligned")
    if response.peak_row >= extension.size:
        raise ExtractionError("peak row is outside the response arrays")
    prepeak_extension = extension[: response.peak_row + 1]
    prepeak_stress = stress[: response.peak_row + 1]
    if not np.all(np.isfinite(prepeak_extension)) or not np.all(
        np.isfinite(prepeak_stress)
    ):
        raise ExtractionError("pre-peak response contains nonfinite values")

    extension_offset = float(np.median(prepeak_extension[:baseline_samples]))
    stress_offset = float(np.median(prepeak_stress[:baseline_samples]))
    extension_peak_delta = float(
        prepeak_extension[response.peak_row] - extension_offset
    )
    stress_peak_delta = float(prepeak_stress[response.peak_row] - stress_offset)
    if not math.isfinite(extension_peak_delta) or extension_peak_delta == 0.0:
        raise ExtractionError("offset-corrected peak extension must be nonzero")
    if not math.isfinite(stress_peak_delta) or stress_peak_delta == 0.0:
        raise ExtractionError("offset-corrected peak stress must be nonzero")

    extension_direction = 1.0 if extension_peak_delta > 0.0 else -1.0
    stress_direction = 1.0 if stress_peak_delta > 0.0 else -1.0
    oriented_extension = extension_direction * (
        prepeak_extension - extension_offset
    )
    oriented_stress = stress_direction * (prepeak_stress - stress_offset)
    extension_peak = extension_direction * extension_peak_delta
    stress_peak = stress_direction * stress_peak_delta
    if extension_peak <= 0.0:
        raise ExtractionError("offset-corrected peak extension must be positive")
    if stress_peak <= 0.0:
        raise ExtractionError("offset-corrected peak stress must be positive")

    eligible = (oriented_extension >= 0.0) & (
        oriented_extension <= extension_peak
    )
    grouped_extension, grouped_stress = aggregate_extension_positions(
        oriented_extension[eligible], oriented_stress[eligible]
    )
    if grouped_extension.size < minimum_unique_extension_positions:
        raise ExtractionError(
            "pre-peak response has fewer than 50 unique extension positions"
        )

    interior = (grouped_extension > 0.0) & (grouped_extension < extension_peak)
    interpolation_extension = np.concatenate(
        ([0.0], grouped_extension[interior], [extension_peak])
    )
    interpolation_stress = np.concatenate(
        ([0.0], grouped_stress[interior], [stress_peak])
    )
    if np.any(np.diff(interpolation_extension) <= 0.0):
        raise ExtractionError("extension positions are not strictly increasing")

    u = np.linspace(0.0, 1.0, grid_points, dtype=np.float64)
    grid_extension = u * extension_peak
    grid_stress = np.interp(
        grid_extension, interpolation_extension, interpolation_stress
    )
    normalized_stress = grid_stress / stress_peak
    normalized_stress[0] = 0.0
    normalized_stress[-1] = 1.0
    if not np.all(np.isfinite(normalized_stress)):
        raise ExtractionError("interpolated normalized response is nonfinite")

    slope_mask = (u >= 0.2) & (u <= 0.6)
    slope_extension = grid_extension[slope_mask]
    slope_stress = grid_stress[slope_mask]
    centered_extension = slope_extension - np.mean(slope_extension)
    denominator = float(np.dot(centered_extension, centered_extension))
    if denominator <= 0.0:
        raise ExtractionError("slope interval has zero extension range")
    slope = float(
        np.dot(centered_extension, slope_stress - np.mean(slope_stress))
        / denominator
    )
    auc = float(np.trapezoid(normalized_stress, u))
    if not math.isfinite(slope) or not math.isfinite(auc):
        raise ExtractionError("response descriptor is nonfinite")

    return ExtractedResponse(
        specimen_id=response.specimen_id.strip().casefold(),
        peak_row=response.peak_row,
        baseline_extension_mm=extension_offset,
        baseline_stress_mpa=stress_offset,
        extension_peak_mm=extension_peak,
        zeroed_peak_stress_mpa=stress_peak,
        unique_extension_positions=int(grouped_extension.size),
        u=_readonly(u),
        extension_mm=_readonly(grid_extension),
        zeroed_stress_mpa=_readonly(grid_stress),
        normalized_stress=_readonly(normalized_stress),
        slope_u20_u60_mpa_per_mm=slope,
        normalized_prepeak_auc=auc,
        q_midpoint=float(normalized_stress[50]),
    )
