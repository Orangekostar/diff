"""Synchronized held-out-domain bootstrap for MVA effects."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapEffect:
    effect_id: str
    point_estimate: float
    lower: float
    upper: float
    improved_domains: int
    domain_effects: tuple[float, ...]
    indices_sha256: str


def synchronized_bootstrap_indices(
    *, seed: int, resamples: int, domains: int = 6
) -> np.ndarray:
    """Return one immutable PCG64 domain-resampling matrix."""

    if type(seed) is not int or type(resamples) is not int or type(domains) is not int:
        raise ValueError("bootstrap seed and sizes must be integers")
    if resamples <= 0 or domains <= 1 or domains > np.iinfo(np.int16).max:
        raise ValueError("bootstrap sizes are invalid")
    generator = np.random.Generator(np.random.PCG64(seed))
    values = generator.integers(0, domains, size=(resamples, domains), dtype=np.int16)
    output = np.frombuffer(values.tobytes(order="C"), dtype=np.int16).reshape(
        values.shape
    )
    output.setflags(write=False)
    return output


def paired_domain_bootstrap(
    baseline: object,
    adaptive: object,
    *,
    indices: np.ndarray,
    effect_id: str,
) -> BootstrapEffect:
    """Bootstrap the paired domain mean of baseline minus adaptive."""

    try:
        first = np.asarray(baseline, dtype=np.float64)
        second = np.asarray(adaptive, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("bootstrap effects must be numeric") from error
    if (
        first.ndim != 1
        or first.shape != second.shape
        or first.size < 2
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
        or not isinstance(indices, np.ndarray)
        or indices.dtype != np.int16
        or indices.ndim != 2
        or indices.shape[1] != first.size
        or indices.size == 0
        or int(np.min(indices)) < 0
        or int(np.max(indices)) >= first.size
        or type(effect_id) is not str
        or not effect_id
    ):
        raise ValueError("bootstrap arrays are invalid")
    effects = first - second
    samples = np.mean(effects[indices], axis=1, dtype=np.float64)
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    digest = hashlib.sha256(
        np.ascontiguousarray(indices).tobytes(order="C")
    ).hexdigest()
    return BootstrapEffect(
        effect_id=effect_id,
        point_estimate=float(np.mean(effects, dtype=np.float64)),
        lower=float(lower),
        upper=float(upper),
        improved_domains=int(np.count_nonzero(effects > 0.0)),
        domain_effects=tuple(float(value) for value in effects),
        indices_sha256=digest,
    )


__all__ = [
    "BootstrapEffect",
    "paired_domain_bootstrap",
    "synchronized_bootstrap_indices",
]
