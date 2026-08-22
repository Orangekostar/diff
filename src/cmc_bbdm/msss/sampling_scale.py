"""MSSS sampling axis with exact P5 digital-grid semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.cpb_sparse_scan.sampling import (
    FIELD_SIZE_MM,
    _indices,
    _interpolate,
    _sha256_array,
    _snapshot_rgb,
)


class MSSSSamplingError(ValueError):
    """Raised when a requested MSSS sampling condition is invalid."""


SAMPLING_DENSITIES = (1.0, 0.75, 0.625, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625)


@dataclass(frozen=True, slots=True)
class SamplingScaleRecord:
    specimen_id: str
    dataset_id: str
    requested_density: float
    effective_density: float
    interpolation: str
    native_height: int
    native_width: int
    row_count: int
    column_count: int
    measured_points: int
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    vertical_stride_px: float
    horizontal_stride_px: float
    vertical_spacing_mm: float
    horizontal_spacing_mm: float
    row_indices_sha256: str
    column_indices_sha256: str
    sampling_mask_sha256: str
    input_sha256: str
    output_sha256: str
    sampled_values_sha256: str
    measured_points_exact: bool
    shape_preserved: bool
    dtype_preserved: bool


def _identity(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\0" in value:
        raise MSSSSamplingError(f"{label} is invalid")
    return value


def _density(value: object) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise MSSSSamplingError("sampling density is not registered")
    density = float(value)
    if not math.isfinite(density) or density not in SAMPLING_DENSITIES:
        raise MSSSSamplingError("sampling density is not registered")
    return density


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.uint8)
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).reshape(array.shape)
    output.setflags(write=False)
    return output


def reconstruct_sampling_scale(
    image: np.ndarray,
    *,
    specimen_id: str,
    dataset_id: str,
    requested_density: float,
) -> tuple[np.ndarray, SamplingScaleRecord]:
    """Sample and bilinearly reconstruct one crop using exact P5 primitives."""

    try:
        source = _snapshot_rgb(image)
    except ValueError as error:
        raise MSSSSamplingError(str(error)) from error
    specimen = _identity(specimen_id, "specimen_id")
    dataset = _identity(dataset_id, "dataset_id")
    density = _density(requested_density)
    rows = _indices(source.shape[0], density)
    columns = _indices(source.shape[1], density)
    sampled = np.ascontiguousarray(source[np.ix_(rows, columns)])
    reconstructed = _interpolate(sampled, source.shape[:2], "bilinear")
    reconstructed[np.ix_(rows, columns)] = sampled
    output = _readonly(reconstructed)
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    mask[np.ix_(rows, columns)] = 1
    exact = bool(np.array_equal(output[np.ix_(rows, columns)], sampled))
    shape_preserved = output.shape == source.shape
    dtype_preserved = output.dtype == source.dtype == np.dtype(np.uint8)
    if not (exact and shape_preserved and dtype_preserved):
        raise MSSSSamplingError("sampling reconstruction invariant failed")
    row_count, column_count = int(rows.size), int(columns.size)
    measured = row_count * column_count
    record = SamplingScaleRecord(
        specimen_id=specimen,
        dataset_id=dataset,
        requested_density=density,
        effective_density=float(measured / (source.shape[0] * source.shape[1])),
        interpolation="bilinear",
        native_height=int(source.shape[0]),
        native_width=int(source.shape[1]),
        row_count=row_count,
        column_count=column_count,
        measured_points=measured,
        rows=tuple(int(value) for value in rows),
        columns=tuple(int(value) for value in columns),
        vertical_stride_px=float((source.shape[0] - 1) / (row_count - 1)),
        horizontal_stride_px=float((source.shape[1] - 1) / (column_count - 1)),
        vertical_spacing_mm=float(FIELD_SIZE_MM / (row_count - 1)),
        horizontal_spacing_mm=float(FIELD_SIZE_MM / (column_count - 1)),
        row_indices_sha256=_sha256_array(rows),
        column_indices_sha256=_sha256_array(columns),
        sampling_mask_sha256=_sha256_array(mask),
        input_sha256=_sha256_array(source),
        output_sha256=_sha256_array(output),
        sampled_values_sha256=_sha256_array(sampled),
        measured_points_exact=exact,
        shape_preserved=shape_preserved,
        dtype_preserved=dtype_preserved,
    )
    return output, record


__all__ = [
    "SAMPLING_DENSITIES",
    "MSSSSamplingError",
    "SamplingScaleRecord",
    "reconstruct_sampling_scale",
]
