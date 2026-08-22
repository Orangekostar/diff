"""Transfer-gain metrics and the frozen S2 decision rule."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


class TransferMetricError(ValueError):
    """Raised when an S2 transfer metric is undefined."""


@dataclass(frozen=True, slots=True)
class TransferGain:
    full_mae: float
    candidate_mae: float
    tg: float
    rtg: float
    nonworse: bool


@dataclass(frozen=True, slots=True)
class S2Gate:
    status: str
    domain_nonworse: int
    domain_support: bool
    structured_nonworse: int
    ply_positive: int
    layup_positive: int


def transfer_gain(*, full_mae: float, candidate_mae: float) -> TransferGain:
    """Compute absolute and relative gain, positive when the candidate is better."""

    try:
        full = float(full_mae)
        candidate = float(candidate_mae)
    except (TypeError, ValueError, OverflowError) as error:
        raise TransferMetricError("transfer MAEs must be numeric") from error
    if not math.isfinite(full) or not math.isfinite(candidate) or full <= 0.0 or candidate < 0.0:
        raise TransferMetricError("transfer MAEs are invalid")
    gain = full - candidate
    return TransferGain(
        full_mae=full,
        candidate_mae=candidate,
        tg=gain,
        rtg=gain / full,
        nonworse=gain >= 0.0,
    )


def s2_gate(
    *, domain_tg: Sequence[float], ply_tg: Sequence[float], layup_tg: Sequence[float]
) -> S2Gate:
    """Apply the registered ordinary-support and structured-transfer thresholds."""

    try:
        domain = tuple(float(value) for value in domain_tg)
        ply = tuple(float(value) for value in ply_tg)
        layup = tuple(float(value) for value in layup_tg)
    except (TypeError, ValueError, OverflowError) as error:
        raise TransferMetricError("S2 gate effects must be numeric") from error
    if (
        len(domain) != 6
        or len(ply) != 3
        or len(layup) != 2
        or any(not math.isfinite(value) for value in (*domain, *ply, *layup))
    ):
        raise TransferMetricError("S2 gate roster is invalid")
    domain_nonworse = sum(value >= 0.0 for value in domain)
    structured_nonworse = sum(value >= 0.0 for value in (*ply, *layup))
    ply_positive = sum(value > 0.0 for value in ply)
    layup_positive = sum(value > 0.0 for value in layup)
    if ply_positive >= 2 and layup_positive >= 1:
        status = "STRONG_GO"
    elif structured_nonworse >= 3:
        status = "GO"
    else:
        status = "NO_GO"
    return S2Gate(
        status=status,
        domain_nonworse=domain_nonworse,
        domain_support=domain_nonworse >= 4,
        structured_nonworse=structured_nonworse,
        ply_positive=ply_positive,
        layup_positive=layup_positive,
    )


__all__ = [
    "S2Gate",
    "TransferGain",
    "TransferMetricError",
    "s2_gate",
    "transfer_gain",
]
