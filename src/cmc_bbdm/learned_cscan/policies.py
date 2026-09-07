"""Rule controls and learned-policy interfaces for common C-scan perception."""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
import torch
from torch import nn

from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .observation import ObservationPacket


class RuleMethod(StrEnum):
    R_GEOM = "R_GEOM"
    R_CENTER = "R_CENTER"
    R_VLM_OPEN = "R_VLM_OPEN"
    R_LEGACY = "R_LEGACY"
    R_BALANCED = "R_BALANCED"


class LearnedPolicyError(ValueError):
    """Raised when learned actor inputs violate the visible packet contract."""


class LearnedCellActor(nn.Module):
    token_width = 128
    encoder_layers = 2
    attention_heads = 4

    def __init__(self) -> None:
        super().__init__()
        self.subblock_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(17 + 32, self.token_width),
            nn.GELU(),
            nn.Linear(self.token_width, self.token_width),
            nn.LayerNorm(self.token_width),
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 32)
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(9 + 32, self.token_width),
            nn.GELU(),
            nn.LayerNorm(self.token_width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_width,
            nhead=self.attention_heads,
            dim_feedforward=256,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.contextualizer = nn.TransformerEncoder(
            layer,
            num_layers=self.encoder_layers,
            enable_nested_tensor=False,
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(self.token_width * 2, self.token_width),
            nn.GELU(),
            nn.Linear(self.token_width, 1),
        )
        if self.parameter_count >= 1_000_000:
            raise LearnedPolicyError("learned actor exceeds the parameter cap")

    @property
    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        cell_features: torch.Tensor,
        subblock_features: torch.Tensor,
        global_features: torch.Tensor,
        history_features: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = _validate_actor_tensors(
            cell_features,
            subblock_features,
            global_features,
            history_features,
            legal_mask,
        )
        subblocks = self.subblock_encoder(subblock_features).mean(dim=2)
        cells = self.cell_encoder(torch.cat((cell_features, subblocks), dim=2))
        history = self.history_encoder(history_features).mean(dim=1)
        global_token = self.global_encoder(
            torch.cat((global_features, history), dim=1)
        )
        contextual = self.contextualizer(
            torch.cat((global_token.unsqueeze(1), cells), dim=1)
        )
        global_context = contextual[:, 0].unsqueeze(1).expand(
            batch, 64, self.token_width
        )
        logits = self.action_scorer(
            torch.cat((contextual[:, 1:], global_context), dim=2)
        ).squeeze(-1)
        return logits.masked_fill(~legal_mask, -torch.inf)

    def select_cell(self, packet: ObservationPacket, *, device: str = "cpu") -> int:
        if type(packet) is not ObservationPacket:
            raise TypeError("typed observation packet is required")
        tensors = packet.actor_tensors()
        target = torch.device(device)
        with torch.no_grad():
            logits = self(
                torch.tensor(tensors["cell_features"]).unsqueeze(0).to(target),
                torch.tensor(tensors["subblock_features"])
                .unsqueeze(0)
                .to(target),
                torch.tensor(tensors["global_features"])
                .unsqueeze(0)
                .to(target),
                torch.tensor(tensors["history_features"])
                .unsqueeze(0)
                .to(target),
                torch.tensor(tensors["legal_mask"]).unsqueeze(0).to(target),
            )
        return int(torch.argmax(logits[0]).item())


def _validate_actor_tensors(
    cell_features: torch.Tensor,
    subblock_features: torch.Tensor,
    global_features: torch.Tensor,
    history_features: torch.Tensor,
    legal_mask: torch.Tensor,
) -> int:
    values = (
        cell_features,
        subblock_features,
        global_features,
        history_features,
        legal_mask,
    )
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise LearnedPolicyError("learned actor inputs must be tensors")
    batch = cell_features.shape[0] if cell_features.ndim == 3 else -1
    if (
        batch < 1
        or cell_features.shape != (batch, 64, 17)
        or subblock_features.shape != (batch, 64, 16, 10)
        or global_features.shape != (batch, 9)
        or history_features.shape != (batch, 4, 5)
        or legal_mask.shape != (batch, 64)
        or legal_mask.dtype is not torch.bool
        or any(not value.is_floating_point() for value in values[:-1])
        or any(not torch.isfinite(value).all() for value in values[:-1])
        or not torch.all(legal_mask.any(dim=1))
        or len({value.device for value in values}) != 1
    ):
        raise LearnedPolicyError("learned actor tensor contract is invalid")
    return batch


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
    level = packet.cell_levels[cell]
    lattice = grid.cells[cell]
    rows = np.asarray(lattice.rows[level + 1], dtype=np.int64)
    columns = np.asarray(lattice.columns[level + 1], dtype=np.int64)
    added_count = int(np.count_nonzero(~packet.measured_mask[np.ix_(rows, columns)]))
    probe_row = float(packet.global_features[4])
    probe_column = float(packet.global_features[5])
    row, column = _cell_coordinates(cell)
    distance = math.hypot(row / 7.0 - probe_row, column / 7.0 - probe_column)
    return added_count, distance, cell


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


__all__ = [
    "LearnedCellActor",
    "LearnedPolicyError",
    "RuleMethod",
    "select_rule_action",
]
