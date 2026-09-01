"""Domain-independent scanner geometry and deterministic G1 warm start."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from cmc_bbdm.inspection_agent.state import (
    GeneralizedMeasurementState,
    InspectionCellAction,
    apply_action,
    budget_record,
    fitting_actions,
    zero_state,
)
from cmc_bbdm.mva.acquisition_grid import (
    INITIAL_BUDGETS,
    AcquisitionGrid,
    AcquisitionGridError,
    build_acquisition_grid,
)
from cmc_bbdm.mva.oracle import uniform_cell_order

DEPLOYMENT_INITIAL_NOMINAL_BUDGET = 0.015625
PRIMARY_WARM_START_K = 8
SENSITIVITY_WARM_START_K = (4, 16)
PRIMARY_WARM_START_CELLS = uniform_cell_order()[:PRIMARY_WARM_START_K]
_GEOMETRY_GO = "G1_DEPLOYMENT_GEOMETRY_GO"
_GEOMETRY_NO_GO = "G1_DEPLOYMENT_GEOMETRY_NO_GO"


class G1GeometryError(ValueError):
    """Raised when deployment geometry or warm-start evidence is invalid."""


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class GeometryPresetRecord:
    native_shape: tuple[int, int]
    g0_nominal_budget: float
    dataset_id: str
    count: int


@dataclass(frozen=True, slots=True)
class GeometryGridCandidate:
    native_shape: tuple[int, int]
    nominal_budget: float
    valid: bool
    grid_sha256: str


@dataclass(frozen=True, slots=True)
class DeploymentGeometryAudit:
    status: str
    specimen_count: int
    native_shapes: tuple[tuple[int, int], ...]
    domain_dependent_g0_budget: bool
    selected_nominal_budget: float | None
    preset_records: tuple[GeometryPresetRecord, ...]
    grid_candidates: tuple[GeometryGridCandidate, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class WarmStartAudit:
    k: int
    cells: tuple[int, ...]
    native_shape: tuple[int, int]
    measured_count: int
    native_count: int
    effective_budget: float
    broaden_action_count: int
    refine_action_count: int
    state_sha256: str


def build_deployment_grid(native_shape: tuple[int, int]) -> AcquisitionGrid:
    if (
        type(native_shape) is not tuple
        or len(native_shape) != 2
        or any(type(value) is not int or value < 9 for value in native_shape)
    ):
        raise G1GeometryError("native deployment geometry is invalid")
    try:
        return build_acquisition_grid(
            *native_shape,
            initial_budget=DEPLOYMENT_INITIAL_NOMINAL_BUDGET,
        )
    except AcquisitionGridError as error:
        raise G1GeometryError("deployment grid cannot be built") from error


def _warm_start_actions(k: int) -> tuple[InspectionCellAction, ...]:
    if type(k) is not int or not 1 <= k <= 64:
        raise G1GeometryError("warm-start size must be between 1 and 64")
    return tuple(
        InspectionCellAction(cell_index, -1, 0)
        for cell_index in uniform_cell_order()[:k]
    )


def apply_warm_start(
    grid: AcquisitionGrid,
    *,
    k: int = PRIMARY_WARM_START_K,
) -> GeneralizedMeasurementState:
    if type(grid) is not AcquisitionGrid:
        raise G1GeometryError("issued acquisition grid is required")
    state = zero_state(grid)
    for action in _warm_start_actions(k):
        state = apply_action(grid, state, action)
    return state


def warm_start_audit(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
    *,
    k: int = PRIMARY_WARM_START_K,
    endpoint_budget: float = 0.25,
) -> WarmStartAudit:
    if (
        type(grid) is not AcquisitionGrid
        or type(state) is not GeneralizedMeasurementState
        or state != apply_warm_start(grid, k=k)
        or isinstance(endpoint_budget, bool)
        or not math.isfinite(float(endpoint_budget))
        or not 0.0 < float(endpoint_budget) <= 1.0
    ):
        raise G1GeometryError("warm-start audit request is invalid")
    record = budget_record(grid, state)
    if record.effective_budget > float(endpoint_budget) + 1.0e-15:
        raise G1GeometryError("warm start exceeds the endpoint")
    actions = fitting_actions(grid, state, float(endpoint_budget))
    broaden = sum(action.from_level == -1 for action in actions)
    refine = sum(action.from_level >= 0 for action in actions)
    cells = uniform_cell_order()[:k]
    digest = _sha(
        {
            "schema": 1,
            "kind": "g1-warm-start-audit",
            "grid": grid.state_sha256,
            "state": state.state_sha256,
            "k": k,
            "cells": cells,
            "measured_count": record.measured_count,
            "native_count": record.native_count,
            "endpoint_budget": float(endpoint_budget),
            "broaden": broaden,
            "refine": refine,
        }
    )
    return WarmStartAudit(
        k=k,
        cells=cells,
        native_shape=grid.native_shape,
        measured_count=record.measured_count,
        native_count=record.native_count,
        effective_budget=record.effective_budget,
        broaden_action_count=broaden,
        refine_action_count=refine,
        state_sha256=digest,
    )


def _roster_records(path: Path) -> tuple[GeometryPresetRecord, ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = tuple(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise G1GeometryError("authorized roster cannot be read") from error
    required = {
        "authorization_status",
        "dataset_id",
        "initial_nominal_budget",
        "native_height",
        "native_width",
        "roster_index",
        "specimen_id",
    }
    if not rows or not required <= set(rows[0]):
        raise G1GeometryError("authorized roster schema is invalid")
    grouped: dict[tuple[int, int, float, str], int] = {}
    identities: set[tuple[str, str]] = set()
    roster_indices: set[int] = set()
    try:
        for row in rows:
            height = int(row["native_height"])
            width = int(row["native_width"])
            budget = float(row["initial_nominal_budget"])
            dataset_id = row["dataset_id"]
            specimen_id = row["specimen_id"]
            roster_index = int(row["roster_index"])
            if (
                row["authorization_status"] != "AUTHORIZED"
                or height < 9
                or width < 9
                or budget not in INITIAL_BUDGETS
                or not dataset_id
                or not specimen_id
                or (dataset_id, specimen_id) in identities
                or roster_index in roster_indices
            ):
                raise G1GeometryError("authorized roster row is invalid")
            identities.add((dataset_id, specimen_id))
            roster_indices.add(roster_index)
            key = (height, width, budget, dataset_id)
            grouped[key] = grouped.get(key, 0) + 1
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, G1GeometryError):
            raise
        raise G1GeometryError("authorized roster row is invalid") from error
    if roster_indices != set(range(len(rows))):
        raise G1GeometryError("authorized roster indices are incomplete")
    return tuple(
        GeometryPresetRecord((height, width), budget, domain, count)
        for (height, width, budget, domain), count in sorted(grouped.items())
    )


def audit_deployment_geometry(roster_path: str | Path) -> DeploymentGeometryAudit:
    path = Path(roster_path)
    records = _roster_records(path)
    specimen_count = sum(record.count for record in records)
    shapes = tuple(sorted({record.native_shape for record in records}))
    budgets_by_shape = {
        shape: {
            record.g0_nominal_budget
            for record in records
            if record.native_shape == shape
        }
        for shape in shapes
    }
    domain_dependent = any(len(values) > 1 for values in budgets_by_shape.values())
    candidates: list[GeometryGridCandidate] = []
    valid_by_budget = {budget: True for budget in INITIAL_BUDGETS}
    for shape in shapes:
        for budget in INITIAL_BUDGETS:
            try:
                grid = build_acquisition_grid(*shape, initial_budget=budget)
            except AcquisitionGridError:
                valid_by_budget[budget] = False
                candidates.append(GeometryGridCandidate(shape, budget, False, ""))
            else:
                candidates.append(
                    GeometryGridCandidate(shape, budget, True, grid.state_sha256)
                )
    valid_budgets = tuple(
        budget for budget in INITIAL_BUDGETS if valid_by_budget[budget]
    )
    selected = min(valid_budgets) if valid_budgets else None
    status = _GEOMETRY_GO if selected is not None else _GEOMETRY_NO_GO
    payload = {
        "schema": 1,
        "kind": "g1-deployment-geometry-audit",
        "status": status,
        "specimen_count": specimen_count,
        "domain_dependent_g0_budget": domain_dependent,
        "selected_nominal_budget": selected,
        "preset_records": [
            (record.native_shape, record.g0_nominal_budget, record.dataset_id, record.count)
            for record in records
        ],
        "grid_candidates": [
            (
                candidate.native_shape,
                candidate.nominal_budget,
                candidate.valid,
                candidate.grid_sha256,
            )
            for candidate in candidates
        ],
    }
    return DeploymentGeometryAudit(
        status=status,
        specimen_count=specimen_count,
        native_shapes=shapes,
        domain_dependent_g0_budget=domain_dependent,
        selected_nominal_budget=selected,
        preset_records=records,
        grid_candidates=tuple(candidates),
        state_sha256=_sha(payload),
    )


__all__ = [
    "DEPLOYMENT_INITIAL_NOMINAL_BUDGET",
    "PRIMARY_WARM_START_CELLS",
    "PRIMARY_WARM_START_K",
    "SENSITIVITY_WARM_START_K",
    "DeploymentGeometryAudit",
    "G1GeometryError",
    "GeometryGridCandidate",
    "GeometryPresetRecord",
    "WarmStartAudit",
    "apply_warm_start",
    "audit_deployment_geometry",
    "build_deployment_grid",
    "warm_start_audit",
]
