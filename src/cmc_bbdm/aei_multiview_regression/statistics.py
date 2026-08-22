"""Common six-domain bootstrap for the four registered confirmatory effects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapEffect:
    name: str
    point_estimate: float
    bootstrap_mean: float
    ordinary_interval: tuple[float, float]
    familywise_interval: tuple[float, float]
    probability_positive: float


@dataclass(frozen=True, slots=True)
class CommonBootstrap:
    seed: int
    resamples: int
    indices: np.ndarray
    effects: tuple[BootstrapEffect, ...]


def _readonly_indices(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<i8")
    result = np.frombuffer(array.tobytes(order="C"), dtype="<i8").reshape(array.shape)
    result.setflags(write=False)
    return result


def common_domain_bootstrap(
    effects: Mapping[str, object],
    *,
    seed: int = 20260811,
    resamples: int = 100_000,
) -> CommonBootstrap:
    """Bootstrap all registered FULL-minus-candidate domain effects together."""

    if (
        not isinstance(effects, Mapping)
        or len(effects) != 4
        or type(seed) is not int
        or seed != 20260811
        or type(resamples) is not int
        or resamples != 100_000
    ):
        raise ValueError(
            "bootstrap requires four effects and the registered seed/count"
        )
    normalized: list[tuple[str, np.ndarray]] = []
    for name, raw in effects.items():
        values = np.asarray(raw, dtype=np.float64)
        if (
            type(name) is not str
            or not name
            or values.shape != (6,)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "each bootstrap effect must contain six finite domain values"
            )
        normalized.append((name, np.array(values, copy=True)))
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(0, 6, size=(resamples, 6), dtype=np.int64)
    records: list[BootstrapEffect] = []
    for name, values in normalized:
        samples = np.mean(values[indices], axis=1)
        ordinary = np.quantile(samples, (0.025, 0.975), method="linear")
        familywise = np.quantile(samples, (0.00625, 0.99375), method="linear")
        records.append(
            BootstrapEffect(
                name=name,
                point_estimate=float(np.mean(values)),
                bootstrap_mean=float(np.mean(samples)),
                ordinary_interval=(float(ordinary[0]), float(ordinary[1])),
                familywise_interval=(float(familywise[0]), float(familywise[1])),
                probability_positive=float(np.mean(samples > 0.0)),
            )
        )
    return CommonBootstrap(
        seed=seed,
        resamples=resamples,
        indices=_readonly_indices(indices),
        effects=tuple(records),
    )
