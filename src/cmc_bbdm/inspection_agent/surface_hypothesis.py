"""Transparent surface-only hypothesis for zero-start inspection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.agentic_nde.surface_cells import integer_crop_box


class SurfaceHypothesisError(ValueError):
    """Raised when a surface hypothesis input or score is invalid."""


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise SurfaceHypothesisError("surface hypothesis array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class SurfaceHypothesis:
    scores: np.ndarray
    top_cells: tuple[int, ...]
    border_median_rgb: np.ndarray
    state_sha256: str


def compute_surface_hypothesis(
    surface_rgb: np.ndarray,
    cell_boxes: np.ndarray,
    *,
    top_k: int,
) -> SurfaceHypothesis:
    image = np.asarray(surface_rgb)
    boxes = np.asarray(cell_boxes, dtype=np.float64)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 9
        or boxes.shape != (64, 4)
        or not np.all(np.isfinite(boxes))
        or type(top_k) is not int
        or not 0 < top_k <= 64
    ):
        raise SurfaceHypothesisError("surface hypothesis request is invalid")
    border_mask = np.zeros(image.shape[:2], dtype=np.bool_)
    border_mask[[0, -1], :] = True
    border_mask[:, [0, -1]] = True
    median = np.median(image[border_mask].astype(np.float64), axis=0)
    raw = np.empty(64, dtype=np.float64)
    for cell, box in enumerate(boxes):
        left, top, right, bottom = integer_crop_box(
            tuple(float(value) for value in box),
            width=image.shape[1],
            height=image.shape[0],
        )
        patch = image[top:bottom, left:right].astype(np.float64, copy=False)
        raw[cell] = float(np.mean(np.abs(patch - median), dtype=np.float64))
    minimum = float(np.min(raw))
    span = float(np.max(raw) - minimum)
    scores = np.zeros(64, dtype=np.float64) if span == 0.0 else (raw - minimum) / span
    top_cells = tuple(
        sorted(range(64), key=lambda cell: (-float(scores[cell]), cell))[:top_k]
    )
    frozen_scores = _readonly(scores, dtype="<f8", shape=(64,))
    frozen_median = _readonly(median, dtype="<f8", shape=(3,))
    digest = hashlib.sha256()
    digest.update(b"inspection-agent-surface-hypothesis-v1")
    digest.update(
        json.dumps(
            {"shape": image.shape, "top_k": top_k, "top_cells": top_cells},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(np.ascontiguousarray(image).tobytes(order="C"))
    digest.update(np.ascontiguousarray(boxes, dtype="<f8").tobytes(order="C"))
    digest.update(frozen_scores.tobytes(order="C"))
    return SurfaceHypothesis(
        scores=frozen_scores,
        top_cells=top_cells,
        border_median_rgb=frozen_median,
        state_sha256=digest.hexdigest(),
    )


__all__ = [
    "SurfaceHypothesis",
    "SurfaceHypothesisError",
    "compute_surface_hypothesis",
]
