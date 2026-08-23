"""Observed-only A5 selectors and retrospective target execution."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Protocol

import numpy as np

from .acquisition_grid import INITIAL_BUDGETS, AcquisitionGrid, build_acquisition_grid
from .interpolation import (
    RefinementPatchCache,
    reconstruct_measurement_state,
    refine_reconstruction,
)
from .measurement_state import (
    MeasurementState,
    apply_action,
    budget_record,
    candidate_budget_record,
    fitting_actions,
    initial_state,
    measurement_mask,
)
from .policy_state import PolicyObservation, build_policy_observation
from .ranking_policy import TrainedRankingPolicy


class A5DeploymentError(ValueError):
    """Raised when deployable selection or execution violates A5."""


class _Encoder(Protocol):
    def encode(self, images: list[np.ndarray]) -> np.ndarray: ...

    def validate(self) -> None: ...


class _Predictor(Protocol):
    fit_domains: tuple[str, ...]
    state_sha256: str

    def predict(self, metadata: object, embeddings: object) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class DeployableAction:
    step: int
    checkpoint: float
    cell_index: int
    from_level: int
    to_level: int
    selector_score: float
    budget_before: float
    budget_after: float
    p_a_prediction_before: float
    p_a_prediction_after: float


@dataclass(frozen=True, slots=True)
class DeployableSnapshot:
    checkpoint: float
    state: MeasurementState
    image: np.ndarray
    embedding: np.ndarray
    p_a_prediction: float
    measured_count: int
    native_count: int
    effective_budget: float
    cumulative_actions: int


@dataclass(frozen=True, slots=True)
class DeployableTrajectory:
    specimen_id: str
    dataset_id: str
    method: str
    grid: AcquisitionGrid
    predictor_state_sha256: str
    policy_state_sha256: str | None
    actions: tuple[DeployableAction, ...]
    snapshots: tuple[DeployableSnapshot, ...]
    state_sha256: str


_METHODS = (
    "center_first",
    "observed_gradient",
    "observed_uncertainty",
    "imitation_policy",
)


def _selector_scores(
    observation: PolicyObservation,
    method: str,
    policy: TrainedRankingPolicy | None,
) -> np.ndarray:
    features = observation.candidate_features
    if method == "center_first":
        scores = -((features[:, 0] - 0.5) ** 2 + (features[:, 1] - 0.5) ** 2)
    elif method == "observed_gradient":
        scores = features[:, 3] * features[:, 5]
    elif method == "observed_uncertainty":
        scores = features[:, 3] * (features[:, 6] + features[:, 7])
    elif method == "imitation_policy":
        if type(policy) is not TrainedRankingPolicy:
            raise A5DeploymentError("imitation policy package is required")
        scores = policy.score_features(
            observation.global_features, observation.candidate_features
        )
    else:
        raise A5DeploymentError("deployable method is not registered")
    output = np.asarray(scores, dtype=np.float64)
    if output.shape != (len(observation.actions),) or not np.all(np.isfinite(output)):
        raise A5DeploymentError("deployable selector scores are invalid")
    return output


def select_deployable_action(
    observation: PolicyObservation,
    *,
    method: str,
    policy: TrainedRankingPolicy | None = None,
):
    """Select one action using only the frozen current observation tensors."""

    if type(observation) is not PolicyObservation or not observation.actions:
        raise A5DeploymentError("a nonempty policy observation is required")
    if method != "imitation_policy" and policy is not None:
        raise A5DeploymentError("heuristic selectors do not accept a policy")
    scores = _selector_scores(observation, method, policy)
    selected = max(
        range(len(observation.actions)),
        key=lambda index: (
            float(scores[index]),
            -observation.actions[index].cell_index,
            -observation.actions[index].to_level,
        ),
    )
    return observation.actions[selected]


def _readonly(value: object, shape: tuple[int, ...], dtype: object) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise A5DeploymentError("deployment snapshot array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _trajectory_state(result: DeployableTrajectory) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "actions": [
                    (
                        value.step,
                        value.checkpoint,
                        value.cell_index,
                        value.from_level,
                        value.to_level,
                        value.selector_score,
                        value.budget_before,
                        value.budget_after,
                        value.p_a_prediction_before,
                        value.p_a_prediction_after,
                    )
                    for value in result.actions
                ],
                "dataset_id": result.dataset_id,
                "grid_state_sha256": result.grid.state_sha256,
                "method": result.method,
                "policy_state_sha256": result.policy_state_sha256,
                "predictor_state_sha256": result.predictor_state_sha256,
                "snapshots": [
                    (
                        value.checkpoint,
                        value.state.levels,
                        value.p_a_prediction,
                        value.measured_count,
                        value.native_count,
                        value.effective_budget,
                        value.cumulative_actions,
                    )
                    for value in result.snapshots
                ],
                "specimen_id": result.specimen_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for snapshot in result.snapshots:
        digest.update(snapshot.image.tobytes(order="C"))
        digest.update(snapshot.embedding.tobytes(order="C"))
    return digest.hexdigest()


def run_deployable_trajectory(
    *,
    specimen_id: str,
    dataset_id: str,
    image: np.ndarray,
    metadata: np.ndarray,
    initial_budget: float,
    checkpoints: tuple[float, ...],
    predictor: _Predictor,
    encoder: _Encoder,
    method: str,
    policy: TrainedRankingPolicy | None = None,
) -> DeployableTrajectory:
    """Reveal selected measurements only after observed-state action choice."""

    if (
        type(specimen_id) is not str
        or not specimen_id
        or type(dataset_id) is not str
        or not dataset_id
        or initial_budget not in INITIAL_BUDGETS
        or type(checkpoints) is not tuple
        or not checkpoints
        or any(
            not math.isfinite(float(value)) or not 0.0 < float(value) <= 0.25
            for value in checkpoints
        )
        or any(second <= first for first, second in pairwise(checkpoints))
        or method not in _METHODS
        or (method == "imitation_policy") != (policy is not None)
    ):
        raise A5DeploymentError("deployment trajectory authority changed")
    meta = np.asarray(metadata, dtype=np.float64)
    if meta.ndim != 1 or not np.all(np.isfinite(meta)):
        raise A5DeploymentError("deployment metadata is invalid")
    encoder.validate()
    grid = build_acquisition_grid(
        image.shape[0], image.shape[1], initial_budget=initial_budget
    )
    state = initial_state(grid)
    current = reconstruct_measurement_state(
        image,
        grid,
        state,
        interpolation="bilinear",
        specimen_id=specimen_id,
        dataset_id=dataset_id,
    ).image
    encoded = np.asarray(encoder.encode([current]), dtype=np.float64)
    if encoded.shape != (1, 512) or not np.all(np.isfinite(encoded)):
        raise A5DeploymentError("deployment encoder output is invalid")
    current_embedding = encoded[0]
    prediction = np.asarray(
        predictor.predict(meta.reshape(1, -1), encoded), dtype=np.float64
    )
    if prediction.shape != (1,) or not np.all(np.isfinite(prediction)):
        raise A5DeploymentError("deployment P-A prediction is invalid")
    current_prediction = float(prediction[0])
    patch_cache = RefinementPatchCache(image=image, grid=grid)
    actions: list[DeployableAction] = []
    snapshots: list[DeployableSnapshot] = []
    step = 0
    for checkpoint in checkpoints:
        while fitting_actions(grid, state, checkpoint):
            observation = build_policy_observation(
                grid,
                state,
                current_reconstruction=current,
                current_embedding=current_embedding,
                current_prediction=current_prediction,
                checkpoint=checkpoint,
                maximum_budget=0.25,
            )
            selected = select_deployable_action(
                observation, method=method, policy=policy
            )
            scores = _selector_scores(observation, method, policy)
            selected_index = observation.actions.index(selected)
            candidate = candidate_budget_record(grid, state, selected)
            current_mask = measurement_mask(grid, state)
            refined = refine_reconstruction(
                image,
                grid,
                state,
                current,
                selected,
                interpolation="bilinear",
                current_mask=current_mask,
                patch_cache=patch_cache,
            )
            next_embedding_batch = np.asarray(
                encoder.encode([refined]), dtype=np.float64
            )
            if (
                next_embedding_batch.shape != (1, 512)
                or not np.all(np.isfinite(next_embedding_batch))
            ):
                raise A5DeploymentError("deployment update embedding is invalid")
            next_prediction_array = np.asarray(
                predictor.predict(
                    meta.reshape(1, -1), next_embedding_batch
                ),
                dtype=np.float64,
            )
            if (
                next_prediction_array.shape != (1,)
                or not np.all(np.isfinite(next_prediction_array))
            ):
                raise A5DeploymentError("deployment update prediction is invalid")
            next_prediction = float(next_prediction_array[0])
            actions.append(
                DeployableAction(
                    step=step,
                    checkpoint=float(checkpoint),
                    cell_index=selected.cell_index,
                    from_level=selected.from_level,
                    to_level=selected.to_level,
                    selector_score=float(scores[selected_index]),
                    budget_before=observation.used_budget,
                    budget_after=candidate.effective_budget,
                    p_a_prediction_before=current_prediction,
                    p_a_prediction_after=next_prediction,
                )
            )
            state = apply_action(grid, state, selected)
            current = refined
            current_embedding = next_embedding_batch[0]
            current_prediction = next_prediction
            step += 1
        budget = budget_record(grid, state)
        snapshots.append(
            DeployableSnapshot(
                checkpoint=float(checkpoint),
                state=state,
                image=_readonly(current, (*grid.native_shape, 3), np.uint8),
                embedding=_readonly(current_embedding, (512,), np.float64),
                p_a_prediction=current_prediction,
                measured_count=budget.measured_count,
                native_count=budget.native_count,
                effective_budget=budget.effective_budget,
                cumulative_actions=len(actions),
            )
        )
    result = DeployableTrajectory(
        specimen_id=specimen_id,
        dataset_id=dataset_id,
        method=method,
        grid=grid,
        predictor_state_sha256=predictor.state_sha256,
        policy_state_sha256=(
            policy.state_sha256 if method == "imitation_policy" and policy else None
        ),
        actions=tuple(actions),
        snapshots=tuple(snapshots),
        state_sha256="",
    )
    return replace(result, state_sha256=_trajectory_state(result))


__all__ = [
    "A5DeploymentError",
    "DeployableAction",
    "DeployableSnapshot",
    "DeployableTrajectory",
    "run_deployable_trajectory",
    "select_deployable_action",
]
