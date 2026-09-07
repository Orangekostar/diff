"""Reference import and hidden task evaluation."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .contracts import (
    BenchmarkTask,
    CScanReference,
    ReferenceType,
    ReviewState,
    TaskReport,
    TaskScore,
)


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


def reference_from_payload(
    payload: object, *, native_shape: tuple[int, int]
) -> CScanReference:
    if (
        type(payload) is not dict
        or type(native_shape) is not tuple
        or len(native_shape) != 2
        or any(type(value) is not int or value < 2 for value in native_shape)
        or payload.get("frame") != "registered_cscan"
    ):
        raise ValueError("reference payload is invalid")
    try:
        reference_type = ReferenceType(payload["reference_type"])
        review_state = ReviewState(payload["review_state"])
        regions = payload["regions"]
        uncertain_regions = payload["uncertain_regions"]
        if type(regions) is not list or type(uncertain_regions) is not list:
            raise TypeError
        certain = _rasterize_regions(regions, native_shape, require_certain=True)
        uncertain = _rasterize_regions(
            uncertain_regions, native_shape, require_certain=False
        )
        uncertain = uncertain & ~certain
        return CScanReference(
            specimen_key=payload["specimen_key"],
            source_image_sha256=payload["source_image_sha256"],
            reference_type=reference_type,
            review_state=review_state,
            reviewer_alias=payload.get("reviewer_alias"),
            certain_mask=certain,
            uncertain_mask=uncertain,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"reference review payload is invalid: {error}") from error


def derive_proxy_reference(
    *,
    specimen_key: str,
    source_image_sha256: str,
    full_scan: np.ndarray,
    distance_threshold: float,
    minimum_component_pixels: int,
    uncertainty_band: float = 0.02,
    border_exclusion_fraction: float = 0.02,
) -> CScanReference:
    image = np.asarray(full_scan)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 9
        or not 0.0 < float(distance_threshold) < 1.0
        or type(minimum_component_pixels) is not int
        or minimum_component_pixels < 1
        or not 0.0 < float(uncertainty_band) < float(distance_threshold)
        or not 0.0 <= float(border_exclusion_fraction) < 0.25
    ):
        raise ValueError("proxy reference request is invalid")
    row_margin = max(1, round(image.shape[0] * border_exclusion_fraction))
    column_margin = max(1, round(image.shape[1] * border_exclusion_fraction))
    row_last = image.shape[0] - row_margin - 1
    column_last = image.shape[1] - column_margin - 1
    border = np.concatenate(
        (
            image[row_margin, column_margin : column_last + 1],
            image[row_last, column_margin : column_last + 1],
            image[row_margin + 1 : row_last, column_margin],
            image[row_margin + 1 : row_last, column_last],
        ),
        axis=0,
    )
    background = np.median(border.astype(np.float64), axis=0)
    scores = np.linalg.norm(image.astype(np.float64) - background, axis=2) / np.sqrt(
        3.0 * 255.0**2
    )
    allowed = np.ones(image.shape[:2], dtype=np.bool_)
    allowed[:row_margin] = False
    allowed[-row_margin:] = False
    allowed[:, :column_margin] = False
    allowed[:, -column_margin:] = False
    candidate = scores >= float(distance_threshold)
    candidate &= allowed
    candidate = ndimage.binary_closing(candidate, iterations=1)
    labels, count = ndimage.label(candidate)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    certain = np.zeros(candidate.shape, dtype=np.bool_)
    for label_id in range(1, count + 1):
        if sizes[label_id] >= minimum_component_pixels:
            certain |= labels == label_id
    uncertain = (
        (scores >= float(distance_threshold) - float(uncertainty_band))
        & ~certain
        & allowed
    )
    uncertain |= ndimage.binary_dilation(certain, iterations=1) & ~certain & allowed
    return CScanReference(
        specimen_key=specimen_key,
        source_image_sha256=source_image_sha256,
        reference_type=ReferenceType.ALGORITHM_DERIVED_NOT_REVIEWED,
        review_state=ReviewState.PENDING,
        reviewer_alias=None,
        certain_mask=certain,
        uncertain_mask=uncertain,
    )


def component_bbox_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    values = np.asarray(mask)
    if values.dtype != np.bool_ or values.ndim != 2:
        raise ValueError("component mask is invalid")
    labels, count = ndimage.label(values)
    height, width = values.shape
    output = []
    for label_id in range(1, count + 1):
        points = np.argwhere(labels == label_id)
        if not len(points):
            continue
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0)
        output.append(
            [
                [float(x0 / max(width - 1, 1)), float(y0 / max(height - 1, 1))],
                [float(x1 / max(width - 1, 1)), float(y0 / max(height - 1, 1))],
                [float(x1 / max(width - 1, 1)), float(y1 / max(height - 1, 1))],
                [float(x0 / max(width - 1, 1)), float(y1 / max(height - 1, 1))],
            ]
        )
    return output


def _rasterize_regions(
    regions: list[object],
    native_shape: tuple[int, int],
    *,
    require_certain: bool,
) -> np.ndarray:
    canvas = Image.new("1", (native_shape[1], native_shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for region in regions:
        if type(region) is not dict or type(region.get("polygon")) is not list:
            raise ValueError("reference polygon is invalid")
        if require_certain and region.get("certainty") != "certain":
            raise ValueError("reference certainty is invalid")
        points = region["polygon"]
        if len(points) < 3:
            raise ValueError("reference polygon is invalid")
        pixels = []
        for point in points:
            if (
                type(point) is not list
                or len(point) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= float(value) <= 1.0
                    for value in point
                )
            ):
                raise ValueError("reference polygon coordinate is invalid")
            pixels.append(
                (
                    float(point[0]) * (native_shape[1] - 1),
                    float(point[1]) * (native_shape[0] - 1),
                )
            )
        draw.polygon(pixels, fill=1)
    return np.asarray(canvas, dtype=np.bool_)


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


__all__ = [
    "component_bbox_polygons",
    "derive_proxy_reference",
    "evaluate_task_report",
    "reference_from_payload",
]
