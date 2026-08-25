"""Machine-readable row contracts for the MAVIS P1 state bank."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from .state_bank import StateSnapshot, StateTrajectory
from .state_candidates import StateCandidateBatch
from .teacher import FoldStateLabels


class MAVISStateBankArtifactError(ValueError):
    """Raised when a P1 state or action row is inconsistent."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class StateBankSample:
    dataset_id: str
    trajectory: StateTrajectory
    snapshot: StateSnapshot
    candidates: StateCandidateBatch
    fold_labels: tuple[FoldStateLabels, ...]

    def __post_init__(self) -> None:
        if (
            type(self.dataset_id) is not str
            or not self.dataset_id
            or type(self.trajectory) is not StateTrajectory
            or type(self.snapshot) is not StateSnapshot
            or type(self.candidates) is not StateCandidateBatch
            or type(self.fold_labels) is not tuple
            or not self.fold_labels
            or any(type(value) is not FoldStateLabels for value in self.fold_labels)
        ):
            raise MAVISStateBankArtifactError("state-bank sample inputs are invalid")
        state = self.snapshot.inspection_state
        expected_history = tuple(
            item.action for item in self.trajectory.actions[: self.snapshot.step]
        )
        if (
            self.trajectory.specimen_id != state.specimen_id
            or self.candidates.specimen_id != state.specimen_id
            or self.candidates.dataset_id != self.dataset_id
            or self.candidates.inspection_state_sha256 != state.state_sha256
            or state.action_history != expected_history
            or not any(
                value.step == self.snapshot.step
                and value.nominal_checkpoint == self.snapshot.nominal_checkpoint
                and value.inspection_state.state_sha256 == state.state_sha256
                for value in self.trajectory.snapshots
            )
        ):
            raise MAVISStateBankArtifactError("state-bank sample state is misaligned")
        outer_domains = tuple(value.outer_domain for value in self.fold_labels)
        if (
            len(set(outer_domains)) != len(outer_domains)
            or self.dataset_id in outer_domains
        ):
            raise MAVISStateBankArtifactError("state-bank fold roster is invalid")
        for fold in self.fold_labels:
            if (
                fold.query_domain != self.dataset_id
                or not math.isfinite(float(fold.current_prediction))
                or not _is_sha256(fold.teacher_state_sha256)
                or not _is_sha256(fold.predictor_state_sha256)
                or len(fold.action_values) != len(self.candidates.actions)
            ):
                raise MAVISStateBankArtifactError("state-bank fold labels are invalid")
            for action, cost, label in zip(
                self.candidates.actions,
                self.candidates.candidate_costs,
                fold.action_values,
                strict=True,
            ):
                if (
                    label.specimen_id != state.specimen_id
                    or label.dataset_id != self.dataset_id
                    or label.action != action
                    or label.exact_added_cost != cost
                    or label.current_prediction != fold.current_prediction
                    or label.predictor_state_sha256 != fold.predictor_state_sha256
                ):
                    raise MAVISStateBankArtifactError(
                        "state-bank action labels are misaligned"
                    )


def _state_id(sample: StateBankSample) -> str:
    return _hash(
        {
            "schema": 1,
            "trajectory_state_sha256": sample.trajectory.state_sha256,
            "inspection_state_sha256": sample.snapshot.inspection_state.state_sha256,
            "step": sample.snapshot.step,
            "nominal_checkpoint": sample.snapshot.nominal_checkpoint,
        }
    )


def state_bank_rows(
    sample: StateBankSample,
    *,
    authority_state_sha256: str,
    endpoint_budget: float,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    if type(sample) is not StateBankSample or not _is_sha256(authority_state_sha256):
        raise MAVISStateBankArtifactError("state-bank row authority is invalid")
    endpoint = float(endpoint_budget)
    state = sample.snapshot.inspection_state
    if (
        isinstance(endpoint_budget, bool)
        or not math.isfinite(endpoint)
        or not state.effective_budget <= endpoint <= state.checkpoint
    ):
        raise MAVISStateBankArtifactError("state-bank endpoint is invalid")
    folds = tuple(sorted(sample.fold_labels, key=lambda value: value.outer_domain))
    state_id = _state_id(sample)
    endpoint_count = math.floor(endpoint * state.native_count)
    action_history = state.action_history
    positions = state.acquired_positions
    values = state.measurement_values
    state_row: dict[str, object] = {
        "schema_version": 1,
        "specimen_id": state.specimen_id,
        "domain_id": sample.dataset_id,
        "trajectory_id": sample.trajectory.state_sha256,
        "method": sample.trajectory.method,
        "seed": sample.trajectory.seed,
        "state_id": state_id,
        "inspection_state_sha256": state.state_sha256,
        "step": sample.snapshot.step,
        "nominal_checkpoint": sample.snapshot.nominal_checkpoint,
        "initial_budget": state.initial_budget,
        "endpoint_budget": endpoint,
        "exact_acquired_cost": state.exact_acquired_count,
        "native_count": state.native_count,
        "effective_budget": state.effective_budget,
        "remaining_cost_to_endpoint": endpoint_count - state.exact_acquired_count,
        "context_features": state.context_features.tolist(),
        "measurement_levels": list(state.measurement_levels),
        "acquired_action_cell_indices": [
            action.cell_index for action in action_history
        ],
        "acquired_action_from_levels": [
            action.from_level for action in action_history
        ],
        "acquired_action_to_levels": [action.to_level for action in action_history],
        "revealed_rows": positions[:, 0].tolist(),
        "revealed_columns": positions[:, 1].tolist(),
        "revealed_red": values[:, 0].tolist(),
        "revealed_green": values[:, 1].tolist(),
        "revealed_blue": values[:, 2].tolist(),
        "candidate_cell_indices": [
            action.cell_index for action in sample.candidates.actions
        ],
        "candidate_from_levels": [
            action.from_level for action in sample.candidates.actions
        ],
        "candidate_to_levels": [
            action.to_level for action in sample.candidates.actions
        ],
        "candidate_exact_added_costs": list(sample.candidates.candidate_costs),
        "teacher_outer_domains": [fold.outer_domain for fold in folds],
        "strict_oof_cai_predictions": [
            float(fold.current_prediction) for fold in folds
        ],
        "teacher_state_sha256": [fold.teacher_state_sha256 for fold in folds],
        "teacher_predictor_state_sha256": [
            fold.predictor_state_sha256 for fold in folds
        ],
        "grid_state_sha256": state.grid_state_sha256,
        "candidate_batch_state_sha256": sample.candidates.state_sha256,
        "current_reconstruction_sha256": (
            sample.candidates.current_reconstruction_sha256
        ),
        "authority_state_sha256": authority_state_sha256,
    }
    action_rows: list[dict[str, object]] = []
    for fold in folds:
        for candidate_index, (label, reconstruction_sha256) in enumerate(
            zip(
                fold.action_values,
                sample.candidates.candidate_reconstruction_sha256,
                strict=True,
            )
        ):
            cost_after = state.exact_acquired_count + label.exact_added_cost
            action_rows.append(
                {
                    "schema_version": 1,
                    "specimen_id": state.specimen_id,
                    "domain_id": sample.dataset_id,
                    "outer_domain": fold.outer_domain,
                    "trajectory_id": sample.trajectory.state_sha256,
                    "method": sample.trajectory.method,
                    "state_id": state_id,
                    "step": sample.snapshot.step,
                    "nominal_checkpoint": sample.snapshot.nominal_checkpoint,
                    "candidate_index": candidate_index,
                    "cell_index": label.action.cell_index,
                    "from_level": label.action.from_level,
                    "to_level": label.action.to_level,
                    "exact_added_cost": label.exact_added_cost,
                    "candidate_exact_cost_after": cost_after,
                    "candidate_effective_budget_after": (
                        cost_after / state.native_count
                    ),
                    "candidate_remaining_cost_to_endpoint": (
                        endpoint_count - cost_after
                    ),
                    "teacher_true_cai": label.true_cai,
                    "current_prediction": label.current_prediction,
                    "candidate_prediction": label.candidate_prediction,
                    "error_before": label.error_before,
                    "error_after": label.error_after,
                    "primary_value": label.primary_value,
                    "secondary_value": label.secondary_value,
                    "primary_value_per_cost": (
                        label.primary_value / label.exact_added_cost
                    ),
                    "teacher_state_sha256": fold.teacher_state_sha256,
                    "teacher_predictor_state_sha256": (
                        fold.predictor_state_sha256
                    ),
                    "candidate_reconstruction_sha256": reconstruction_sha256,
                    "authority_state_sha256": authority_state_sha256,
                }
            )
    return state_row, tuple(action_rows)


__all__ = [
    "MAVISStateBankArtifactError",
    "StateBankSample",
    "state_bank_rows",
]
