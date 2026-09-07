"""Reference import and hidden task evaluation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .contracts import BenchmarkTask, CScanReference, TaskReport, TaskScore


def evaluate_task_report(
    report: TaskReport, reference: CScanReference
) -> TaskScore:
    if type(report) is not TaskReport or type(reference) is not CScanReference:
        raise TypeError("typed report and reference are required")
    if report.predicted_mask.shape != reference.certain_mask.shape:
        raise ValueError("report and reference shapes differ")
    failures: list[str] = []
    if report.task is BenchmarkTask.LOCATE:
        iou, recall, area_error = _locate_score(report, reference, failures)
        passed = not failures and iou >= 0.50
    else:
        iou, recall, area_error = _characterize_score(report, reference, failures)
        passed = (
            not failures
            and iou >= 0.70
            and recall >= 0.90
            and area_error <= 0.10
        )
    return TaskScore(
        reference_eligible=reference.formal_eligible,
        formal_success=passed if reference.formal_eligible else None,
        proxy_success=passed,
        iou=float(iou),
        recall=float(recall),
        relative_area_error=float(area_error),
        failure_types=tuple(failures),
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if count == 0:
        return np.zeros_like(mask, dtype=np.bool_)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.argwhere(mask)
    if not len(points):
        return None
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()) + 1,
        int(points[:, 1].max()) + 1,
    )


def _bbox_iou(
    left: tuple[int, int, int, int] | None,
    right: tuple[int, int, int, int] | None,
) -> float:
    if left is None or right is None:
        return 0.0
    ly0, lx0, ly1, lx1 = left
    ry0, rx0, ry1, rx1 = right
    intersection = max(0, min(ly1, ry1) - max(ly0, ry0)) * max(
        0, min(lx1, rx1) - max(lx0, rx0)
    )
    union = (ly1 - ly0) * (lx1 - lx0) + (ry1 - ry0) * (rx1 - rx0) - intersection
    return float(intersection / union) if union else 0.0


def _locate_score(
    report: TaskReport, reference: CScanReference, failures: list[str]
) -> tuple[float, float, float]:
    target = _largest_component(reference.certain_mask)
    prediction = report.predicted_mask
    iou = _bbox_iou(_bbox(prediction), _bbox(target))
    if np.count_nonzero(prediction) / prediction.size >= 0.95:
        failures.append("FULL_FRAME_PREDICTION")
    supports = report.support_positions
    inside = (
        target[supports[:, 0], supports[:, 1]]
        if len(supports)
        else np.empty(0, dtype=np.bool_)
    )
    accepted = supports[inside]
    if (
        len(accepted) < 3
        or len(set(accepted[:, 0])) < 2
        or len(set(accepted[:, 1])) < 2
    ):
        failures.append("INSUFFICIENT_MEASURED_SUPPORT")
    if not np.any(target):
        failures.append("REFERENCE_HAS_NO_INDICATION")
    return iou, 1.0 if iou >= 0.50 else 0.0, 0.0


def _characterize_score(
    report: TaskReport, reference: CScanReference, failures: list[str]
) -> tuple[float, float, float]:
    certain = reference.certain_mask
    uncertain = reference.uncertain_mask
    valid = ~uncertain
    prediction = report.predicted_mask
    intersection = int(np.count_nonzero(prediction & certain & valid))
    union = int(np.count_nonzero((prediction | certain) & valid))
    iou = float(intersection / union) if union else 0.0
    certain_area = int(np.count_nonzero(certain))
    recall = float(intersection / certain_area) if certain_area else 0.0
    predicted_area = int(np.count_nonzero(prediction))
    upper = certain_area + int(np.count_nonzero(uncertain))
    if predicted_area < certain_area:
        area_distance = certain_area - predicted_area
    elif predicted_area > upper:
        area_distance = predicted_area - upper
    else:
        area_distance = 0
    area_error = float(area_distance / max(certain_area, 1))
    labels, count = ndimage.label(
        prediction,
        structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    )
    supports = report.support_positions
    supported_labels = (
        {int(value) for value in labels[supports[:, 0], supports[:, 1]] if value}
        if len(supports)
        else set()
    )
    if count and supported_labels != set(range(1, count + 1)):
        failures.append("UNSUPPORTED_PREDICTED_COMPONENT")
    if not certain_area:
        failures.append("REFERENCE_HAS_NO_INDICATION")
    return iou, recall, area_error


__all__ = ["evaluate_task_report"]
