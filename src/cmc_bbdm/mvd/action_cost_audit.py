"""Exact CandidateBank action-cost evidence for MVD M0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .authority import CompactMVDAuthority, MVDAuthorityError


@dataclass(frozen=True, slots=True)
class ActionCostSummary:
    initial_budget: float
    action_count: int
    minimum: int
    maximum: int
    mean: float
    standard_deviation: float
    coefficient_of_variation: float
    unique_costs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ActionCostAudit:
    rows: tuple[dict[str, object], ...]
    summaries: tuple[ActionCostSummary, ...]
    state_sha256: str


def _state_sha256(
    rows: tuple[dict[str, object], ...], summaries: tuple[ActionCostSummary, ...]
) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("ascii")
        )
    digest.update(
        json.dumps(
            [
                {
                    "action_count": value.action_count,
                    "coefficient_of_variation": value.coefficient_of_variation,
                    "initial_budget": value.initial_budget,
                    "maximum": value.maximum,
                    "mean": value.mean,
                    "minimum": value.minimum,
                    "standard_deviation": value.standard_deviation,
                    "unique_costs": value.unique_costs,
                }
                for value in summaries
            ],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    )
    return digest.hexdigest()


def build_action_cost_audit(authority: CompactMVDAuthority) -> ActionCostAudit:
    """Audit every specimen/budget/cell cost already bound by CandidateBank."""

    if type(authority) is not CompactMVDAuthority:
        raise MVDAuthorityError("issued compact MVD authority is required")
    rows: list[dict[str, object]] = []
    summaries: list[ActionCostSummary] = []
    for budget in sorted(authority.candidate_banks):
        bank = authority.candidate_banks[budget]
        costs = np.asarray(bank.added_measurements, dtype=np.int64)
        for specimen_index, (specimen_id, dataset_id, native_shape) in enumerate(
            zip(
                bank.specimen_ids,
                bank.dataset_ids,
                bank.native_shapes,
                strict=True,
            )
        ):
            native_count = int(bank.native_counts[specimen_index])
            for cell_index in range(64):
                row, column = divmod(cell_index, 8)
                added = int(costs[specimen_index, cell_index])
                rows.append(
                    {
                        "specimen_id": specimen_id,
                        "dataset_id": dataset_id,
                        "initial_budget": budget,
                        "cell_index": cell_index,
                        "row": row,
                        "column": column,
                        "boundary_or_interior": (
                            "boundary"
                            if row in {0, 7} or column in {0, 7}
                            else "interior"
                        ),
                        "native_height": native_shape[0],
                        "native_width": native_shape[1],
                        "native_count": native_count,
                        "added_measurements": added,
                        "added_fraction": float(added / native_count),
                        "grid_state_sha256": bank.grid_state_sha256[specimen_index],
                        "candidate_bank_state_sha256": bank.state_sha256,
                    }
                )
        mean = float(np.mean(costs, dtype=np.float64))
        standard_deviation = float(np.std(costs, dtype=np.float64))
        summaries.append(
            ActionCostSummary(
                initial_budget=budget,
                action_count=int(costs.size),
                minimum=int(np.min(costs)),
                maximum=int(np.max(costs)),
                mean=mean,
                standard_deviation=standard_deviation,
                coefficient_of_variation=float(standard_deviation / mean),
                unique_costs=tuple(int(value) for value in np.unique(costs)),
            )
        )
    frozen_rows = tuple(rows)
    frozen_summaries = tuple(summaries)
    return ActionCostAudit(
        rows=frozen_rows,
        summaries=frozen_summaries,
        state_sha256=_state_sha256(frozen_rows, frozen_summaries),
    )


__all__ = ["ActionCostAudit", "ActionCostSummary", "build_action_cost_audit"]
