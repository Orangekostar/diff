"""Rule controls and learned-policy interfaces for common C-scan perception."""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np

from cmc_bbdm.inspection_agent.state import (
    GeneralizedMeasurementState,
    InspectionCellAction,
    candidate_budget_record,
)
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .observation import ObservationPacket


class RuleMethod(StrEnum):
    R_GEOM = "R_GEOM"
    R_CENTER = "R_CENTER"
    R_VLM_OPEN = "R_VLM_OPEN"
    R_LEGACY = "R_LEGACY"
    R_BALANCED = "R_BALANCED"


def select_rule_action(
    method: RuleMethod,
    packet: ObservationPacket,
    *,
    grid: AcquisitionGrid,
    coverage_period: int = 4,
) -> InspectionCellAction:
    if (
        type(method) is not RuleMethod
        or type(packet) is not ObservationPacket
        or type(grid) is not AcquisitionGrid
        or grid.native_shape != packet.measured_mask.shape
        or type(coverage_period) is not int
        or coverage_period < 1
    ):
        raise ValueError("rule action request is invalid")
    legal_cells = tuple(index for index, level in enumerate(packet.cell_levels) if level < 2)
    if not legal_cells:
        raise ValueError("no legal scan action remains")
    if method is RuleMethod.R_GEOM:
        chosen = _geometry_cell(packet, legal_cells, coverage_only=True)
    elif method is RuleMethod.R_CENTER:
        chosen = min(
            legal_cells,
            key=lambda cell: (
                packet.cell_levels[cell] + 1,
                _center_distance(cell),
                _public_cost_key(packet, grid, cell),
            ),
        )
    elif method is RuleMethod.R_VLM_OPEN:
        chosen = min(
            legal_cells,
            key=lambda cell: (
                packet.cell_levels[cell] + 1,
                -float(packet.cell_features[cell, 16]),
                -float(packet.cell_features[cell, 15]),
                _center_distance(cell),
                _public_cost_key(packet, grid, cell),
            ),
        )
    elif method is RuleMethod.R_LEGACY:
        chosen = _legacy_cell(packet, grid, legal_cells)
    else:
        chosen = _balanced_cell(packet, grid, legal_cells, coverage_period)
    level = packet.cell_levels[chosen]
    return InspectionCellAction(chosen, level, level + 1)


def _balanced_cell(
    packet: ObservationPacket,
    grid: AcquisitionGrid,
    legal_cells: tuple[int, ...],
    coverage_period: int,
) -> int:
    coverage = tuple(cell for cell in legal_cells if packet.cell_levels[cell] == -1)
    primitive_count = sum(level + 1 for level in packet.cell_levels)
    if coverage and (primitive_count + 1) % coverage_period == 0:
        return _geometry_cell(packet, coverage, coverage_only=False)
    surface_unmeasured = tuple(
        cell
        for cell in coverage
        if packet.cell_features[cell, 15] > 0.0
    )
    if surface_unmeasured:
        return min(
            surface_unmeasured,
            key=lambda cell: (
                -float(packet.cell_features[cell, 16]),
                _public_cost_key(packet, grid, cell),
            ),
        )
    boundary = tuple(
        cell
        for cell in _neighbor_ring(packet.report.candidate_cells)
        if cell in coverage
    )
    if boundary:
        return min(boundary, key=lambda cell: _public_cost_key(packet, grid, cell))
    refinements = tuple(
        cell
        for cell in packet.report.candidate_cells
        if cell in legal_cells and packet.cell_levels[cell] >= 0
    )
    if refinements:
        return min(
            refinements,
            key=lambda cell: (
                packet.cell_levels[cell],
                _public_cost_key(packet, grid, cell),
            ),
        )
    if coverage:
        return _geometry_cell(packet, coverage, coverage_only=False)
    return min(legal_cells, key=lambda cell: _public_cost_key(packet, grid, cell))


def _legacy_cell(
    packet: ObservationPacket,
    grid: AcquisitionGrid,
    legal_cells: tuple[int, ...],
) -> int:
    positive = tuple(cell for cell in packet.report.candidate_cells if cell in legal_cells)
    if positive:
        return min(
            positive,
            key=lambda cell: (
                packet.cell_levels[cell],
                _public_cost_key(packet, grid, cell),
            ),
        )
    boundary = tuple(
        cell
        for cell in _neighbor_ring(packet.report.candidate_cells)
        if cell in legal_cells
    )
    if boundary:
        return min(boundary, key=lambda cell: _public_cost_key(packet, grid, cell))
    return min(
        legal_cells,
        key=lambda cell: (
            packet.cell_levels[cell] + 1,
            -float(packet.cell_features[cell, 16]),
            _public_cost_key(packet, grid, cell),
        ),
    )


def _geometry_cell(
    packet: ObservationPacket,
    candidates: tuple[int, ...],
    *,
    coverage_only: bool,
) -> int:
    eligible = (
        tuple(cell for cell in candidates if packet.cell_levels[cell] == -1)
        if coverage_only
        else candidates
    )
    if not eligible:
        eligible = candidates
    measured = tuple(cell for cell, level in enumerate(packet.cell_levels) if level >= 0)
    if not measured:
        anchors = ((0.0, 0.0), (0.0, 7.0), (7.0, 0.0), (7.0, 7.0))
        return min(
            eligible,
            key=lambda cell: (
                min(_squared_distance(cell, anchor) for anchor in anchors),
                cell,
            ),
        )
    return max(
        eligible,
        key=lambda cell: (
            min(_squared_distance(cell, _cell_coordinates(other)) for other in measured),
            -_center_distance(cell),
            -cell,
        ),
    )


def _public_cost_key(
    packet: ObservationPacket, grid: AcquisitionGrid, cell: int
) -> tuple[int, float, int]:
    state = GeneralizedMeasurementState(grid.state_sha256, packet.cell_levels)
    level = packet.cell_levels[cell]
    action = InspectionCellAction(cell, level, level + 1)
    candidate = candidate_budget_record(grid, state, action)
    current = int(np.count_nonzero(packet.measured_mask))
    probe_row = float(packet.global_features[4])
    probe_column = float(packet.global_features[5])
    row, column = _cell_coordinates(cell)
    distance = math.hypot(row / 7.0 - probe_row, column / 7.0 - probe_column)
    return candidate.measured_count - current, distance, cell


def _neighbor_ring(cells: tuple[int, ...]) -> tuple[int, ...]:
    source = set(cells)
    output: set[int] = set()
    for cell in source:
        row, column = divmod(cell, 8)
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                candidate_row = row + row_delta
                candidate_column = column + column_delta
                if (
                    0 <= candidate_row < 8
                    and 0 <= candidate_column < 8
                    and (row_delta or column_delta)
                ):
                    output.add(candidate_row * 8 + candidate_column)
    return tuple(sorted(output - source))


def _cell_coordinates(cell: int) -> tuple[float, float]:
    row, column = divmod(cell, 8)
    return float(row), float(column)


def _squared_distance(cell: int, point: tuple[float, float]) -> float:
    row, column = _cell_coordinates(cell)
    return (row - point[0]) ** 2 + (column - point[1]) ** 2


def _center_distance(cell: int) -> float:
    return _squared_distance(cell, (3.5, 3.5))


__all__ = ["RuleMethod", "select_rule_action"]
