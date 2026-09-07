"""Shared typed contracts for the VLM C-scan benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np


class BenchmarkTask(str, Enum):
    LOCATE = "LOCATE"
    CHARACTERIZE = "CHARACTERIZE"


class EvaluationMode(str, Enum):
    ANYTIME_REPORT = "ANYTIME_REPORT"
    AUTONOMOUS_REPORT = "AUTONOMOUS_REPORT"


class EvidenceState(IntEnum):
    UNKNOWN = -1
    MEASURED_NO_INDICATION = 0
    MEASURED_INDICATION = 1


class ReferenceType(str, Enum):
    EXPERT_REVIEWED = "EXPERT_REVIEWED"
    AUTHOR_PROVIDED = "AUTHOR_PROVIDED"
    ALGORITHM_DERIVED = "ALGORITHM_DERIVED"
    ALGORITHM_DERIVED_NOT_REVIEWED = "ALGORITHM_DERIVED_NOT_REVIEWED"
    UNKNOWN_PROVENANCE = "UNKNOWN_PROVENANCE"


class ReviewState(str, Enum):
    PENDING = "pending"
    REVIEWED = "reviewed"


class MethodId(str, Enum):
    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    B5 = "B5"
    B6 = "B6"
    B7 = "B7"


@dataclass(frozen=True, slots=True)
class SurfaceRegion:
    cells: tuple[int, ...]
    visible_cue: str
    alternative: str
    confidence: str

    def __post_init__(self) -> None:
        if (
            type(self.cells) is not tuple
            or not self.cells
            or len(set(self.cells)) != len(self.cells)
            or any(type(cell) is not int or not 0 <= cell < 64 for cell in self.cells)
            or type(self.visible_cue) is not str
            or not self.visible_cue
            or type(self.alternative) is not str
            or not self.alternative
            or self.confidence not in {"low", "medium", "high"}
        ):
            raise ValueError("surface region is invalid")


@dataclass(frozen=True, slots=True)
class SurfacePlan:
    regions: tuple[SurfaceRegion, ...]
    priority_cells: tuple[int, ...]
    initial_skill: str
    no_reliable_surface_cue: bool

    def __post_init__(self) -> None:
        cells = tuple(cell for region in self.regions for cell in region.cells)
        if (
            type(self.regions) is not tuple
            or len(self.regions) > 3
            or any(type(region) is not SurfaceRegion for region in self.regions)
            or type(self.priority_cells) is not tuple
            or len(self.priority_cells) > 8
            or len(set(self.priority_cells)) != len(self.priority_cells)
            or any(
                type(cell) is not int or not 0 <= cell < 64
                for cell in self.priority_cells
            )
            or len(set(cells)) != len(cells)
            or len(cells) > 8
            or not set(cells).issubset(self.priority_cells)
            or self.initial_skill not in {"SURVEY_ROI", "BROADEN_SEARCH"}
            or type(self.no_reliable_surface_cue) is not bool
            or (
                self.no_reliable_surface_cue
                and (self.regions or self.priority_cells or self.initial_skill != "BROADEN_SEARCH")
            )
            or (
                not self.priority_cells
                and self.initial_skill != "BROADEN_SEARCH"
            )
        ):
            raise ValueError("surface plan is invalid")


@dataclass(frozen=True, slots=True)
class EvidenceMap:
    states: np.ndarray
    scores: np.ndarray
    measured_mask: np.ndarray
    indication_mask: np.ndarray

    def __post_init__(self) -> None:
        states = _readonly(self.states, dtype=np.int8)
        scores = _readonly(self.scores, dtype=np.float64, shape=states.shape)
        measured = _readonly(self.measured_mask, dtype=np.bool_, shape=states.shape)
        indication = _readonly(
            self.indication_mask, dtype=np.bool_, shape=states.shape
        )
        if (
            states.ndim != 2
            or states.size == 0
            or not set(np.unique(states)).issubset({-1, 0, 1})
            or not np.all(np.isfinite(scores))
            or not np.array_equal(measured, states != EvidenceState.UNKNOWN)
            or not np.array_equal(
                indication, states == EvidenceState.MEASURED_INDICATION
            )
            or np.any(scores[~measured] != 0.0)
        ):
            raise ValueError("evidence map is invalid")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "measured_mask", measured)
        object.__setattr__(self, "indication_mask", indication)


@dataclass(frozen=True, slots=True)
class CScanReference:
    specimen_key: str
    source_image_sha256: str
    reference_type: ReferenceType
    review_state: ReviewState
    reviewer_alias: str | None
    certain_mask: np.ndarray
    uncertain_mask: np.ndarray

    def __post_init__(self) -> None:
        if (
            not self.specimen_key
            or not _is_sha256(self.source_image_sha256)
            or type(self.reference_type) is not ReferenceType
            or type(self.review_state) is not ReviewState
        ):
            raise ValueError("reference identity is invalid")
        certain = _readonly(self.certain_mask, dtype=np.bool_)
        uncertain = _readonly(
            self.uncertain_mask, dtype=np.bool_, shape=certain.shape
        )
        if certain.ndim != 2 or certain.size == 0 or np.any(certain & uncertain):
            raise ValueError("reference masks are invalid")
        if self.review_state is ReviewState.REVIEWED:
            if (
                self.reference_type
                not in {ReferenceType.EXPERT_REVIEWED, ReferenceType.AUTHOR_PROVIDED}
                or not self.reviewer_alias
            ):
                raise ValueError("only attributable human references may be reviewed")
        elif self.reviewer_alias is not None:
            raise ValueError("pending references cannot name a reviewer")
        object.__setattr__(self, "certain_mask", certain)
        object.__setattr__(self, "uncertain_mask", uncertain)

    @property
    def formal_eligible(self) -> bool:
        return self.review_state is ReviewState.REVIEWED and self.reference_type in {
            ReferenceType.EXPERT_REVIEWED,
            ReferenceType.AUTHOR_PROVIDED,
        }


@dataclass(frozen=True, slots=True)
class TaskReport:
    task: BenchmarkTask
    predicted_mask: np.ndarray
    support_positions: np.ndarray
    confidence: float
    public_complete: bool
    reason_code: str

    def __post_init__(self) -> None:
        prediction = _readonly(self.predicted_mask, dtype=np.bool_)
        positions = _readonly(self.support_positions, dtype=np.int64)
        confidence = float(self.confidence)
        if (
            type(self.task) is not BenchmarkTask
            or prediction.ndim != 2
            or prediction.size == 0
            or positions.ndim != 2
            or positions.shape[1:] != (2,)
            or (
                positions.size
                and (
                    np.any(positions < 0)
                    or np.any(positions[:, 0] >= prediction.shape[0])
                    or np.any(positions[:, 1] >= prediction.shape[1])
                )
            )
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or type(self.public_complete) is not bool
            or not self.reason_code
        ):
            raise ValueError("task report is invalid")
        object.__setattr__(self, "predicted_mask", prediction)
        object.__setattr__(self, "support_positions", positions)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class TaskScore:
    reference_eligible: bool
    formal_success: bool | None
    proxy_success: bool
    iou: float
    recall: float
    relative_area_error: float
    failure_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoutePlan:
    pixel_order: np.ndarray
    unique_revealed_count: int
    transit_length: float
    scan_length: float
    total_length: float
    turn_count: int
    revisit_count: int
    end_position: tuple[float, float]

    def __post_init__(self) -> None:
        order = _readonly(self.pixel_order, dtype=np.int64)
        lengths = (
            float(self.transit_length),
            float(self.scan_length),
            float(self.total_length),
        )
        if (
            order.ndim != 2
            or order.shape[1:] != (2,)
            or self.unique_revealed_count != len(order)
            or any(not math.isfinite(value) or value < 0.0 for value in lengths)
            or not math.isclose(lengths[0] + lengths[1], lengths[2], abs_tol=1e-12)
            or self.turn_count < 0
            or self.revisit_count < 0
            or len(self.end_position) != 2
        ):
            raise ValueError("route plan is invalid")
        object.__setattr__(self, "pixel_order", order)
        object.__setattr__(self, "transit_length", lengths[0])
        object.__setattr__(self, "scan_length", lengths[1])
        object.__setattr__(self, "total_length", lengths[2])


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _readonly(
    value: object, *, dtype: object, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError("array shape is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    output.setflags(write=False)
    return output


__all__ = [
    "BenchmarkTask",
    "CScanReference",
    "EvaluationMode",
    "EvidenceMap",
    "EvidenceState",
    "MethodId",
    "ReferenceType",
    "ReviewState",
    "RoutePlan",
    "SurfacePlan",
    "SurfaceRegion",
    "TaskReport",
    "TaskScore",
]
