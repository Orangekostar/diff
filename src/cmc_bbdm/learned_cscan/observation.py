"""Immutable same-perception packet exposed to every planner."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionObservation
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .contracts import Task
from .perception import SurfacePercept
from .readout import (
    BackgroundPrior,
    ReaderV2Result,
    TaskReportV2,
    build_task_report_v2,
    owned_cell_slices,
    read_visible_evidence,
)

CELL_FEATURE_COUNT = 17
SUBBLOCK_FEATURE_COUNT = 10
GLOBAL_FEATURE_COUNT = 9
HISTORY_FEATURE_COUNT = 5


@dataclass(frozen=True, slots=True, eq=False)
class ObservationPacket:
    task: Task
    measured_positions: np.ndarray
    measured_values: np.ndarray
    measured_mask: np.ndarray
    cell_levels: tuple[int, ...]
    cell_features: np.ndarray
    subblock_features: np.ndarray
    global_features: np.ndarray
    history_features: np.ndarray
    legal_mask: np.ndarray
    readout: ReaderV2Result
    report: TaskReportV2
    feature_sha256: str

    def __post_init__(self) -> None:
        positions = _readonly(self.measured_positions, dtype=np.int64)
        values = _readonly(
            self.measured_values, dtype=np.uint8, shape=(len(positions), 3)
        )
        mask = _readonly(self.measured_mask, dtype=np.bool_)
        cells = _readonly(
            self.cell_features, dtype=np.float32, shape=(64, CELL_FEATURE_COUNT)
        )
        subblocks = _readonly(
            self.subblock_features,
            dtype=np.float32,
            shape=(64, 16, SUBBLOCK_FEATURE_COUNT),
        )
        global_features = _readonly(
            self.global_features, dtype=np.float32, shape=(GLOBAL_FEATURE_COUNT,)
        )
        history = _readonly(
            self.history_features, dtype=np.float32, shape=(4, HISTORY_FEATURE_COUNT)
        )
        legal = _readonly(self.legal_mask, dtype=np.bool_, shape=(64,))
        if (
            type(self.task) is not Task
            or positions.ndim != 2
            or positions.shape[1:] != (2,)
            or mask.ndim != 2
            or type(self.cell_levels) is not tuple
            or len(self.cell_levels) != 64
            or any(level not in (-1, 0, 1, 2) for level in self.cell_levels)
            or not np.array_equal(legal, np.asarray(self.cell_levels) < 2)
            or not np.all(np.isfinite(cells))
            or not np.all(np.isfinite(subblocks))
            or not np.all(np.isfinite(global_features))
            or not np.all(np.isfinite(history))
            or type(self.readout) is not ReaderV2Result
            or type(self.report) is not TaskReportV2
            or self.report.task is not self.task
        ):
            raise ValueError("observation packet is invalid")
        digest = _packet_hash(
            self.task,
            positions,
            values,
            mask,
            self.cell_levels,
            cells,
            subblocks,
            global_features,
            history,
            legal,
        )
        if self.feature_sha256 not in ("", digest):
            raise ValueError("observation packet hash changed")
        object.__setattr__(self, "measured_positions", positions)
        object.__setattr__(self, "measured_values", values)
        object.__setattr__(self, "measured_mask", mask)
        object.__setattr__(self, "cell_features", cells)
        object.__setattr__(self, "subblock_features", subblocks)
        object.__setattr__(self, "global_features", global_features)
        object.__setattr__(self, "history_features", history)
        object.__setattr__(self, "legal_mask", legal)
        object.__setattr__(self, "feature_sha256", digest)

    def actor_tensors(self) -> dict[str, np.ndarray]:
        return {
            "cell_features": self.cell_features,
            "subblock_features": self.subblock_features,
            "global_features": self.global_features,
            "history_features": self.history_features,
            "legal_mask": self.legal_mask,
        }


def build_observation_packet(
    observation: InspectionObservation,
    *,
    grid: AcquisitionGrid,
    percept: SurfacePercept,
    task: Task,
    prior: BackgroundPrior,
    distance_threshold: float,
    probe_position: tuple[float, float],
    route_cost: float,
    include_actor_subblocks: bool = True,
) -> ObservationPacket:
    if (
        type(observation) is not InspectionObservation
        or type(grid) is not AcquisitionGrid
        or observation.grid_sha256 != grid.state_sha256
        or observation.native_shape != grid.native_shape
        or type(percept) is not SurfacePercept
        or type(task) is not Task
        or type(prior) is not BackgroundPrior
        or type(include_actor_subblocks) is not bool
        or type(probe_position) is not tuple
        or len(probe_position) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in probe_position
        )
        or isinstance(route_cost, bool)
        or not isinstance(route_cost, (int, float))
        or not math.isfinite(float(route_cost))
        or route_cost < 0.0
    ):
        raise ValueError("observation packet request is invalid")
    levels = observation.measurement_state.levels
    readout = read_visible_evidence(
        grid=grid,
        positions=observation.acquired_positions,
        values=observation.measurement_values,
        cell_levels=levels,
        prior=prior,
        distance_threshold=distance_threshold,
    )
    report = build_task_report_v2(readout, task=task)
    cue_strength = np.zeros(64, dtype=np.float64)
    confidence = np.zeros(64, dtype=np.float64)
    confidence_scale = {"unknown": 0.0, "low": 1.0 / 3.0, "medium": 2.0 / 3.0, "high": 1.0}
    for region in percept.regions:
        for cell in region.cells:
            cue_strength[cell] = 1.0
            confidence[cell] = max(confidence[cell], confidence_scale[region.confidence])
    cell_features = np.zeros((64, CELL_FEATURE_COUNT), dtype=np.float32)
    for cell in readout.cells:
        cell_features[cell.cell_index] = np.asarray(
            [
                (cell.cell_index // 8) / 7.0,
                (cell.cell_index % 8) / 7.0,
                (cell.level + 1) / 3.0,
                cell.measured_fraction,
                *(value / 255.0 for value in cell.observed_rgb_mean),
                *(value / 255.0 for value in cell.observed_rgb_std),
                cell.signal_mean,
                cell.signal_max,
                float(cell.candidate),
                float(cell.estimate_valid),
                float(cell.boundary_unverified),
                cue_strength[cell.cell_index],
                confidence[cell.cell_index],
            ],
            dtype=np.float32,
        )
    subblocks = (
        _subblock_features(grid, readout)
        if include_actor_subblocks
        else np.zeros((64, 16, SUBBLOCK_FEATURE_COUNT), dtype=np.float32)
    )
    task_locate = float(task is Task.LOCATE)
    global_features = np.asarray(
        [
            task_locate,
            1.0 - task_locate,
            observation.effective_budget,
            min(float(route_cost), 10.0) / 10.0,
            float(probe_position[0]),
            float(probe_position[1]),
            float(percept.no_reliable_cue),
            len(report.candidate_cells) / 64.0,
            observation.exact_acquired_count / observation.native_count,
        ],
        dtype=np.float32,
    )
    history = np.zeros((4, HISTORY_FEATURE_COUNT), dtype=np.float32)
    for row, action in enumerate(observation.action_history[-4:], start=max(0, 4 - len(observation.action_history))):
        history[row] = (
            1.0,
            (action.cell_index // 8) / 7.0,
            (action.cell_index % 8) / 7.0,
            (action.from_level + 1) / 3.0,
            (action.to_level + 1) / 3.0,
        )
    return ObservationPacket(
        task=task,
        measured_positions=observation.acquired_positions,
        measured_values=observation.measurement_values,
        measured_mask=readout.measured_mask,
        cell_levels=levels,
        cell_features=cell_features,
        subblock_features=subblocks,
        global_features=global_features,
        history_features=history,
        legal_mask=np.asarray(levels) < 2,
        readout=readout,
        report=report,
        feature_sha256="",
    )


def _subblock_features(
    grid: AcquisitionGrid, readout: ReaderV2Result
) -> np.ndarray:
    output = np.zeros((64, 16, SUBBLOCK_FEATURE_COUNT), dtype=np.float32)
    for cell in grid.cells:
        row_slice, column_slice = owned_cell_slices(grid, cell)
        cell_mask = readout.measured_mask[row_slice, column_slice]
        cell_rgb = readout.measured_rgb[row_slice, column_slice]
        cell_scores = readout.measured_scores[row_slice, column_slice]
        row_groups = np.array_split(
            np.arange(cell_mask.shape[0], dtype=np.int64), 4
        )
        column_groups = np.array_split(
            np.arange(cell_mask.shape[1], dtype=np.int64), 4
        )
        for subrow, rows in enumerate(row_groups):
            for subcolumn, columns in enumerate(column_groups):
                index = subrow * 4 + subcolumn
                if not len(rows) or not len(columns):
                    output[cell.index, index, 1] = 1.0
                    continue
                row_group = slice(int(rows[0]), int(rows[-1]) + 1)
                column_group = slice(int(columns[0]), int(columns[-1]) + 1)
                mask = cell_mask[row_group, column_group]
                count = int(np.count_nonzero(mask))
                total = int(mask.size)
                output[cell.index, index, 0] = count / total
                output[cell.index, index, 1] = float(count == 0)
                if not count:
                    continue
                rgb = cell_rgb[row_group, column_group][mask]
                scores = cell_scores[row_group, column_group][mask]
                output[cell.index, index, 2:5] = rgb.mean(axis=0) / 255.0
                output[cell.index, index, 5:8] = rgb.std(axis=0) / 255.0
                output[cell.index, index, 8] = scores.mean()
                output[cell.index, index, 9] = scores.max()
    return output


def _packet_hash(task: Task, *values: object) -> str:
    digest = hashlib.sha256(b"learned-cscan-observation-v1")
    digest.update(task.value.encode("ascii"))
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
            digest.update(value.tobytes(order="C"))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
            )
    return digest.hexdigest()


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
    "CELL_FEATURE_COUNT",
    "GLOBAL_FEATURE_COUNT",
    "HISTORY_FEATURE_COUNT",
    "SUBBLOCK_FEATURE_COUNT",
    "ObservationPacket",
    "build_observation_packet",
]
