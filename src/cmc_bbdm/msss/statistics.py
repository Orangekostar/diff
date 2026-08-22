"""Synchronized specimen-level bootstrap statistics for MSSS."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


class MSSSStatisticsError(ValueError):
    """Raised when bootstrap evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class EffectInterval:
    name: str
    estimate: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class CommonBootstrap:
    seed: int
    resamples: int
    group_order: tuple[str, ...]
    quantiles: tuple[float, float]
    effects: Mapping[str, EffectInterval]
    draws_sha256: str


def common_stratified_bootstrap(
    effects: Mapping[str, object],
    *,
    groups: Sequence[str],
    group_order: Sequence[str],
    seed: int,
    resamples: int,
    quantiles: tuple[float, float],
) -> CommonBootstrap:
    """Bootstrap specimens within groups using shared draws for every effect."""

    if not isinstance(effects, Mapping) or not effects:
        raise MSSSStatisticsError("bootstrap effects must be a nonempty mapping")
    group_values = np.asarray(tuple(groups), dtype=str)
    order = tuple(group_order)
    if (
        group_values.ndim != 1
        or not len(group_values)
        or not order
        or tuple(dict.fromkeys(group_values.tolist())) != order
        or type(seed) is not int
        or isinstance(seed, bool)
        or seed < 0
        or type(resamples) is not int
        or resamples < 1
        or len(quantiles) != 2
        or not (0.0 < quantiles[0] < quantiles[1] < 1.0)
    ):
        raise MSSSStatisticsError("bootstrap registry is invalid")
    arrays: dict[str, np.ndarray] = {}
    for name, value in effects.items():
        if type(name) is not str or not name:
            raise MSSSStatisticsError("bootstrap effect name is invalid")
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise MSSSStatisticsError("bootstrap effect must be numeric") from error
        if array.shape != group_values.shape or not np.all(np.isfinite(array)):
            raise MSSSStatisticsError("bootstrap effect is incomplete")
        arrays[name] = np.ascontiguousarray(array)
    generator = np.random.Generator(np.random.PCG64(seed))
    draw_digest = hashlib.sha256()
    sampled = {name: np.zeros(resamples, dtype=np.float64) for name in arrays}
    point = {name: 0.0 for name in arrays}
    for group in order:
        indices = np.flatnonzero(group_values == group)
        if not len(indices):
            raise MSSSStatisticsError("bootstrap group is empty")
        draws = generator.integers(
            0, len(indices), size=(resamples, len(indices)), dtype=np.int64
        )
        draw_digest.update(group.encode("utf-8"))
        draw_digest.update(b"\0")
        draw_digest.update(draws.astype("<i8", copy=False).tobytes(order="C"))
        for name, array in arrays.items():
            values = array[indices]
            point[name] += float(np.mean(values)) / len(order)
            sampled[name] += np.mean(values[draws], axis=1) / len(order)
    intervals: dict[str, EffectInterval] = {}
    for name in arrays:
        low, high = np.quantile(
            sampled[name], quantiles, method="linear"
        ).tolist()
        if not all(math.isfinite(value) for value in (point[name], low, high)):
            raise MSSSStatisticsError("bootstrap interval is non-finite")
        intervals[name] = EffectInterval(
            name=name,
            estimate=point[name],
            low=float(low),
            high=float(high),
        )
    return CommonBootstrap(
        seed=seed,
        resamples=resamples,
        group_order=order,
        quantiles=quantiles,
        effects=MappingProxyType(intervals),
        draws_sha256=draw_digest.hexdigest(),
    )


__all__ = [
    "CommonBootstrap",
    "EffectInterval",
    "MSSSStatisticsError",
    "common_stratified_bootstrap",
]
