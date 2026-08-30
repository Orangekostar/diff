"""Deterministic nearest-strength response-curve pair selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CURVE_POINTS = 101
MAXIMUM_REPRESENTATIVE_PAIRS = 12


class RepresentativePairError(ValueError):
    """Raised when representative-pair inputs violate the frozen rule."""


@dataclass(frozen=True, slots=True)
class CurveRecord:
    specimen_id: str
    domain_id: str
    published_cai_strength_mpa: float
    normalized_curve: np.ndarray


@dataclass(frozen=True, slots=True)
class RepresentativePair:
    left_specimen_id: str
    right_specimen_id: str
    left_domain_id: str
    right_domain_id: str
    left_strength_mpa: float
    right_strength_mpa: float
    strength_abs_difference_mpa: float
    curve_rms: float


def _validated_records(records: tuple[CurveRecord, ...]) -> tuple[CurveRecord, ...]:
    if len(records) < 2:
        raise RepresentativePairError("at least two response curves are required")
    specimen_ids: set[str] = set()
    for record in records:
        if type(record) is not CurveRecord:
            raise RepresentativePairError("response curve record type changed")
        if (
            not record.specimen_id
            or record.specimen_id != record.specimen_id.strip().casefold()
            or record.specimen_id in specimen_ids
            or not record.domain_id
            or record.domain_id != record.domain_id.strip().casefold()
        ):
            raise RepresentativePairError(
                "response curve identity is empty, noncanonical, or duplicate"
            )
        strength = record.published_cai_strength_mpa
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(float(strength))
            or strength <= 0.0
        ):
            raise RepresentativePairError(
                f"response curve strength is invalid: {record.specimen_id}"
            )
        curve = np.asarray(record.normalized_curve, dtype=np.float64)
        if curve.shape != (CURVE_POINTS,):
            raise RepresentativePairError(
                f"response curve must have exactly 101 points: {record.specimen_id}"
            )
        if not np.all(np.isfinite(curve)):
            raise RepresentativePairError(
                f"response curve is nonfinite: {record.specimen_id}"
            )
        specimen_ids.add(record.specimen_id)
    return tuple(sorted(records, key=lambda record: record.specimen_id))


def _pair(left: CurveRecord, right: CurveRecord) -> RepresentativePair:
    if right.specimen_id < left.specimen_id:
        left, right = right, left
    difference = np.asarray(left.normalized_curve, dtype=np.float64) - np.asarray(
        right.normalized_curve, dtype=np.float64
    )
    rms = float(np.sqrt(np.mean(difference * difference, dtype=np.float64)))
    return RepresentativePair(
        left_specimen_id=left.specimen_id,
        right_specimen_id=right.specimen_id,
        left_domain_id=left.domain_id,
        right_domain_id=right.domain_id,
        left_strength_mpa=float(left.published_cai_strength_mpa),
        right_strength_mpa=float(right.published_cai_strength_mpa),
        strength_abs_difference_mpa=abs(
            float(left.published_cai_strength_mpa)
            - float(right.published_cai_strength_mpa)
        ),
        curve_rms=rms,
    )


def select_representative_pairs(
    records: tuple[CurveRecord, ...],
) -> tuple[RepresentativePair, ...]:
    """Apply nearest-strength matching, unordered deduplication, and top-12."""

    values = _validated_records(tuple(records))
    by_pair: dict[tuple[str, str], RepresentativePair] = {}
    for source in values:
        target = min(
            (candidate for candidate in values if candidate.specimen_id != source.specimen_id),
            key=lambda candidate: (
                abs(
                    float(candidate.published_cai_strength_mpa)
                    - float(source.published_cai_strength_mpa)
                ),
                candidate.specimen_id,
            ),
        )
        pair = _pair(source, target)
        key = (pair.left_specimen_id, pair.right_specimen_id)
        by_pair[key] = pair
    ranked = sorted(
        by_pair.values(),
        key=lambda pair: (
            -pair.curve_rms,
            pair.left_specimen_id,
            pair.right_specimen_id,
        ),
    )
    return tuple(ranked[:MAXIMUM_REPRESENTATIVE_PAIRS])
