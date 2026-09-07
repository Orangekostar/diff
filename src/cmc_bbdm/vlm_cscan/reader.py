"""Common sparse C-scan evidence reader."""

from __future__ import annotations

import numpy as np

from .contracts import EvidenceMap, EvidenceState


def read_sparse_evidence(
    *,
    native_shape: tuple[int, int],
    positions: np.ndarray,
    values: np.ndarray,
    background_rgb: np.ndarray,
    distance_threshold: float,
) -> EvidenceMap:
    if (
        type(native_shape) is not tuple
        or len(native_shape) != 2
        or any(type(value) is not int or value <= 0 for value in native_shape)
        or isinstance(distance_threshold, bool)
        or not 0.0 < float(distance_threshold) < 1.0
    ):
        raise ValueError("reader request is invalid")
    coordinates = np.asarray(positions)
    samples = np.asarray(values)
    background = np.asarray(background_rgb)
    if (
        coordinates.dtype.kind not in "iu"
        or coordinates.ndim != 2
        or coordinates.shape[1:] != (2,)
        or samples.dtype != np.uint8
        or samples.shape != (len(coordinates), 3)
        or background.dtype != np.uint8
        or background.shape != (3,)
        or (
            coordinates.size
            and (
                np.any(coordinates < 0)
                or np.any(coordinates[:, 0] >= native_shape[0])
                or np.any(coordinates[:, 1] >= native_shape[1])
                or len(np.unique(coordinates, axis=0)) != len(coordinates)
            )
        )
    ):
        raise ValueError("reader observations are invalid")
    sample_scores = np.linalg.norm(
        samples.astype(np.float64) - background.astype(np.float64), axis=1
    ) / np.sqrt(3.0 * 255.0**2)
    states = np.full(native_shape, EvidenceState.UNKNOWN, dtype=np.int8)
    scores = np.zeros(native_shape, dtype=np.float64)
    if len(coordinates):
        rows, columns = coordinates.T
        scores[rows, columns] = sample_scores
        states[rows, columns] = np.where(
            sample_scores >= float(distance_threshold),
            EvidenceState.MEASURED_INDICATION,
            EvidenceState.MEASURED_NO_INDICATION,
        )
    return EvidenceMap(
        states=states,
        scores=scores,
        measured_mask=states != EvidenceState.UNKNOWN,
        indication_mask=states == EvidenceState.MEASURED_INDICATION,
    )


__all__ = ["read_sparse_evidence"]
