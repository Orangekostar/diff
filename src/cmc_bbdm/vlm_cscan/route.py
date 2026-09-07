"""Common normalized primitive-route compiler."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from .contracts import RoutePlan


def compile_route(
    positions: np.ndarray,
    *,
    native_shape: tuple[int, int],
    start_position: tuple[float, float],
) -> RoutePlan:
    pixels = _validate_positions(positions, native_shape)
    unique = np.unique(pixels, axis=0)
    if not len(unique):
        return RoutePlan(
            pixel_order=np.empty((0, 2), dtype=np.int64),
            unique_revealed_count=0,
            transit_length=0.0,
            scan_length=0.0,
            total_length=0.0,
            turn_count=0,
            revisit_count=0,
            end_position=_validate_start(start_position),
        )
    start = _validate_start(start_position)
    row_forward = _snake(unique, primary_axis=0)
    column_forward = _snake(unique, primary_axis=1)
    candidates = (
        row_forward,
        row_forward[::-1],
        column_forward,
        column_forward[::-1],
    )
    ranked = []
    for candidate_index, candidate in enumerate(candidates):
        normalized = _normalized(candidate, native_shape)
        transit = _distance(np.asarray(start), normalized[0])
        scan = float(
            np.linalg.norm(np.diff(normalized, axis=0), axis=1).sum()
            if len(normalized) > 1
            else 0.0
        )
        ranked.append((transit + scan, transit, candidate_index, candidate, normalized))
    total, transit, _index, selected, normalized = min(
        ranked, key=lambda item: (item[0], item[1], item[2])
    )
    scan = total - transit
    return RoutePlan(
        pixel_order=selected,
        unique_revealed_count=len(selected),
        transit_length=transit,
        scan_length=scan,
        total_length=total,
        turn_count=_turn_count(normalized),
        revisit_count=0,
        end_position=(float(normalized[-1, 0]), float(normalized[-1, 1])),
    )


def reference_full_raster_length(
    native_shape: tuple[int, int], start_position: tuple[float, float]
) -> float:
    if (
        type(native_shape) is not tuple
        or len(native_shape) != 2
        or any(type(value) is not int or value < 2 for value in native_shape)
    ):
        raise ValueError("native route shape is invalid")
    positions = np.argwhere(np.ones(native_shape, dtype=np.bool_))
    return compile_route(
        positions,
        native_shape=native_shape,
        start_position=start_position,
    ).total_length


def _validate_positions(
    positions: np.ndarray, native_shape: tuple[int, int]
) -> np.ndarray:
    if (
        type(native_shape) is not tuple
        or len(native_shape) != 2
        or any(type(value) is not int or value < 2 for value in native_shape)
    ):
        raise ValueError("native route shape is invalid")
    values = np.asarray(positions)
    if (
        values.dtype.kind not in "iu"
        or values.ndim != 2
        or values.shape[1:] != (2,)
        or (
            values.size
            and (
                np.any(values < 0)
                or np.any(values[:, 0] >= native_shape[0])
                or np.any(values[:, 1] >= native_shape[1])
            )
        )
    ):
        raise ValueError("route positions are invalid")
    return np.ascontiguousarray(values, dtype=np.int64)


def _validate_start(start_position: tuple[float, float]) -> tuple[float, float]:
    if type(start_position) is not tuple or len(start_position) != 2:
        raise ValueError("route start is invalid")
    start = tuple(float(value) for value in start_position)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in start):
        raise ValueError("route start is invalid")
    return start


def _snake(points: np.ndarray, *, primary_axis: int) -> np.ndarray:
    secondary_axis = 1 - primary_axis
    output: list[np.ndarray] = []
    for group_index, primary in enumerate(sorted(set(points[:, primary_axis]))):
        group = points[points[:, primary_axis] == primary]
        order = np.argsort(group[:, secondary_axis], kind="stable")
        if group_index % 2:
            order = order[::-1]
        output.extend(group[order])
    return np.ascontiguousarray(output, dtype=np.int64)


def _normalized(points: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    output = np.empty(points.shape, dtype=np.float64)
    output[:, 0] = points[:, 1] / (native_shape[1] - 1)
    output[:, 1] = points[:, 0] / (native_shape[0] - 1)
    return output


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(right - left))


def _turn_count(points: np.ndarray) -> int:
    if len(points) < 3:
        return 0
    vectors = np.diff(points, axis=0)
    count = 0
    for left, right in pairwise(vectors):
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            continue
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
        if not math.isclose(cosine, 1.0, abs_tol=1.0e-12):
            count += 1
    return count


__all__ = ["compile_route", "reference_full_raster_length"]
