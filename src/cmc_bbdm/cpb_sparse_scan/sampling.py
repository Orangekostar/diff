"""Deterministic digital sparse-grid reconstruction for the registered P5 study."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as torch_functional

SPARSE_DENSITIES = (0.5, 0.25, 0.125)
INTERPOLATIONS = ("nearest", "bilinear", "bicubic")
FIELD_SIZE_MM = 75.0


class SparseSamplingValidationError(ValueError):
    """Raised when a P5 sampling or reconstruction contract is violated."""


@dataclass(frozen=True, slots=True)
class SamplingCoordinates:
    native_height: int
    native_width: int
    nominal_density: float
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    actual_density: float
    vertical_spacing_mm: float
    horizontal_spacing_mm: float
    row_indices_sha256: str
    column_indices_sha256: str
    sampling_mask_sha256: str


@dataclass(frozen=True, slots=True)
class SamplingRecord:
    specimen_id: str
    dataset_id: str
    nominal_density: float
    actual_density: float
    interpolation: str
    native_height: int
    native_width: int
    row_count: int
    column_count: int
    rows: tuple[int, ...]
    columns: tuple[int, ...]
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
    invariants_passed: bool


def _sha256_array(value: np.ndarray) -> str:
    payload = np.ascontiguousarray(value).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _validate_dimension(value: object, label: str) -> int:
    if type(value) is not int or value < 2:
        raise SparseSamplingValidationError(f"{label} must be an integer >= 2")
    return value


def _validate_density(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SparseSamplingValidationError("density must be a registered finite number")
    density = float(value)
    if not math.isfinite(density) or density not in SPARSE_DENSITIES:
        raise SparseSamplingValidationError("density is not a registered sparse density")
    return density


def _validate_identity(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise SparseSamplingValidationError(f"{label} must be a canonical identifier")
    return value


def _indices(length: int, density: float) -> np.ndarray:
    count = max(2, int(np.rint(math.sqrt(density) * length)))
    values = np.rint(np.linspace(0.0, float(length - 1), count)).astype(
        "<i8", copy=False
    )
    if (
        values.shape != (count,)
        or values[0] != 0
        or values[-1] != length - 1
        or np.any(np.diff(values) <= 0)
    ):
        raise SparseSamplingValidationError("sampling indices violate the frozen grid")
    return np.ascontiguousarray(values)


def sampling_coordinates(
    height: int, width: int, *, density: float
) -> SamplingCoordinates:
    """Return the unique registered sparse grid for one native image shape."""

    native_height = _validate_dimension(height, "height")
    native_width = _validate_dimension(width, "width")
    nominal_density = _validate_density(density)
    rows = _indices(native_height, nominal_density)
    columns = _indices(native_width, nominal_density)
    mask = np.zeros((native_height, native_width), dtype=np.uint8)
    mask[np.ix_(rows, columns)] = 1
    return SamplingCoordinates(
        native_height=native_height,
        native_width=native_width,
        nominal_density=nominal_density,
        rows=tuple(int(value) for value in rows),
        columns=tuple(int(value) for value in columns),
        actual_density=float(rows.size * columns.size / (native_height * native_width)),
        vertical_spacing_mm=float(FIELD_SIZE_MM / (rows.size - 1)),
        horizontal_spacing_mm=float(FIELD_SIZE_MM / (columns.size - 1)),
        row_indices_sha256=_sha256_array(rows),
        column_indices_sha256=_sha256_array(columns),
        sampling_mask_sha256=_sha256_array(mask),
    )


def _snapshot_rgb(image: object) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] < 2
        or image.shape[1] < 2
        or image.dtype != np.uint8
    ):
        raise SparseSamplingValidationError("image must be uint8[H,W,3] RGB")
    first = np.ascontiguousarray(image).tobytes(order="C")
    second = np.ascontiguousarray(image).tobytes(order="C")
    if first != second:
        raise SparseSamplingValidationError("image changed while being snapshotted")
    return np.frombuffer(first, dtype=np.uint8).reshape(image.shape)


def _interpolate(sampled: np.ndarray, shape: tuple[int, int], method: str) -> np.ndarray:
    tensor = (
        torch.from_numpy(sampled.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device="cpu", dtype=torch.float64)
    )
    with torch.no_grad():
        if method == "nearest":
            expanded = torch_functional.interpolate(
                tensor, size=shape, mode="nearest-exact"
            )
        else:
            expanded = torch_functional.interpolate(
                tensor,
                size=shape,
                mode=method,
                align_corners=True,
                antialias=False,
            )
    if expanded.shape != (1, 3, *shape):
        raise SparseSamplingValidationError("interpolator output shape changed")
    if not bool(torch.isfinite(expanded).all().item()):
        raise SparseSamplingValidationError("interpolator output must be finite")
    return (
        torch.round(expanded)
        .clamp(0, 255)
        .to(torch.uint8)
        .squeeze(0)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )


def reconstruct_sparse_rgb(
    image: np.ndarray,
    *,
    specimen_id: str,
    dataset_id: str,
    density: float,
    interpolation: str,
) -> tuple[np.ndarray, SamplingRecord]:
    """Digitally sample and reconstruct one manifest-bound native RGB crop."""

    source = _snapshot_rgb(image)
    specimen = _validate_identity(specimen_id, "specimen_id")
    dataset = _validate_identity(dataset_id, "dataset_id")
    if not isinstance(interpolation, str) or interpolation not in INTERPOLATIONS:
        raise SparseSamplingValidationError("interpolation is not registered")
    coordinates = sampling_coordinates(
        source.shape[0], source.shape[1], density=density
    )
    rows = np.asarray(coordinates.rows, dtype=np.int64)
    columns = np.asarray(coordinates.columns, dtype=np.int64)
    sampled = np.ascontiguousarray(source[np.ix_(rows, columns)])
    reconstructed = _interpolate(
        sampled, (coordinates.native_height, coordinates.native_width), interpolation
    )
    reconstructed[np.ix_(rows, columns)] = sampled
    measured_points_exact = bool(
        np.array_equal(reconstructed[np.ix_(rows, columns)], sampled)
    )
    shape_preserved = reconstructed.shape == source.shape
    dtype_preserved = reconstructed.dtype == source.dtype == np.uint8
    output_payload = reconstructed.tobytes(order="C")
    output = np.frombuffer(output_payload, dtype=np.uint8).reshape(source.shape)
    invariants_passed = measured_points_exact and shape_preserved and dtype_preserved
    if not invariants_passed:
        raise SparseSamplingValidationError("reconstruction invariants failed")
    record = SamplingRecord(
        specimen_id=specimen,
        dataset_id=dataset,
        nominal_density=coordinates.nominal_density,
        actual_density=coordinates.actual_density,
        interpolation=interpolation,
        native_height=coordinates.native_height,
        native_width=coordinates.native_width,
        row_count=len(coordinates.rows),
        column_count=len(coordinates.columns),
        rows=coordinates.rows,
        columns=coordinates.columns,
        vertical_spacing_mm=coordinates.vertical_spacing_mm,
        horizontal_spacing_mm=coordinates.horizontal_spacing_mm,
        row_indices_sha256=coordinates.row_indices_sha256,
        column_indices_sha256=coordinates.column_indices_sha256,
        sampling_mask_sha256=coordinates.sampling_mask_sha256,
        input_sha256=_sha256_array(source),
        output_sha256=_sha256_array(output),
        sampled_values_sha256=_sha256_array(sampled),
        measured_points_exact=measured_points_exact,
        shape_preserved=shape_preserved,
        dtype_preserved=dtype_preserved,
        invariants_passed=invariants_passed,
    )
    return output, record
