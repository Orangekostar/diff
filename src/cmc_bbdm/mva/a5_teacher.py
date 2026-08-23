"""Outer-safe oracle teachers for the supervised MVA A5 policy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np

from .acquisition_grid import (
    INITIAL_BUDGETS,
    AcquisitionGrid,
    build_acquisition_grid,
)
from .cai_evaluator import CAIPredictor
from .crossfit import FitAudit, fit_outer_source_predictor
from .interpolation import (
    RefinementPatchCache,
    reconstruct_measurement_state,
    refine_reconstruction,
)
from .measurement_state import (
    MeasurementState,
    RefinementAction,
    apply_action,
    candidate_budget_record,
    fitting_actions,
    initial_state,
    measurement_mask,
)
from .policy_state import build_policy_observation


class A5TeacherError(ValueError):
    """Raised when teacher fitting or trajectory generation leaks or drifts."""


class _Encoder(Protocol):
    def encode(self, images: list[np.ndarray]) -> np.ndarray: ...

    def validate(self) -> None: ...


class _Predictor(Protocol):
    fit_domains: tuple[str, ...]
    state_sha256: str

    def predict(self, metadata: object, embeddings: object) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class TeacherFitAudit:
    stage: str
    held_out_target_domain: str
    query_source_domain: str
    query_domains: tuple[str, ...]
    fit_domains: tuple[str, ...]
    query_specimen_ids: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    pca_dimension: int
    predictor_state_sha256: str


@dataclass(frozen=True, slots=True)
class TeacherModelBundle:
    outer_domain: str
    source_domains: tuple[str, ...]
    models: Mapping[str, CAIPredictor]
    fit_audits: tuple[TeacherFitAudit, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class TeacherState:
    step: int
    checkpoint: float
    actions: tuple[RefinementAction, ...]
    global_features: np.ndarray
    candidate_features: np.ndarray
    values: np.ndarray
    selected_index: int
    budget_before: float
    budget_after: float
    current_prediction: float
    selected_prediction: float


@dataclass(frozen=True, slots=True)
class TeacherTrajectory:
    specimen_id: str
    dataset_id: str
    predictor_state_sha256: str
    states: tuple[TeacherState, ...]
    selected_actions: tuple[RefinementAction, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class TeacherTrajectoryInput:
    specimen_id: str
    dataset_id: str
    image: np.ndarray
    target: float
    metadata: np.ndarray
    predictor: _Predictor
    initial_embedding: np.ndarray | None = None


@dataclass(slots=True)
class _TeacherContext:
    request: TeacherTrajectoryInput
    response: float
    metadata: np.ndarray
    grid: AcquisitionGrid
    state: MeasurementState
    current: np.ndarray
    current_embedding: np.ndarray
    current_prediction: float
    patch_cache: RefinementPatchCache
    records: list[TeacherState]
    selected_actions: list[RefinementAction]
    step: int = 0


def _state(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
    return digest.hexdigest()


def _readonly(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise A5TeacherError("teacher array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(shape)
    output.setflags(write=False)
    return output


def _audit(
    value: FitAudit, *, outer_domain: str, query_domain: str
) -> TeacherFitAudit:
    return TeacherFitAudit(
        stage=value.stage,
        held_out_target_domain=outer_domain,
        query_source_domain=query_domain,
        query_domains=value.query_domains,
        fit_domains=value.fit_domains,
        query_specimen_ids=value.query_specimen_ids,
        fit_specimen_ids=value.fit_specimen_ids,
        pca_dimension=value.pca_dimension,
        predictor_state_sha256=value.predictor_state_sha256,
    )


def fit_outer_safe_teacher_models(
    *,
    outer_domain: str,
    domain_order: tuple[str, ...],
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    targets: object,
    metadata: object,
    full_embeddings: object,
    pca_dimensions: tuple[int, ...],
    ridge_alpha: float,
    tie_tolerance: float = 1.0e-12,
) -> TeacherModelBundle:
    """Fit one four-domain P-A oracle teacher for every source query domain."""

    if (
        type(domain_order) is not tuple
        or len(domain_order) != 6
        or outer_domain not in domain_order
    ):
        raise A5TeacherError("teacher domain authority changed")
    samples = tuple(specimen_ids)
    domains = tuple(dataset_ids)
    response = np.asarray(targets, dtype=np.float64)
    meta = np.asarray(metadata, dtype=np.float64)
    embeddings = np.asarray(full_embeddings, dtype=np.float64)
    if (
        len(samples) == 0
        or len(domains) != len(samples)
        or response.shape != (len(samples),)
        or meta.ndim != 2
        or meta.shape[0] != len(samples)
        or embeddings.ndim != 2
        or embeddings.shape[0] != len(samples)
    ):
        raise A5TeacherError("teacher fit arrays are invalid")
    dataset_array = np.asarray(domains, dtype=object)
    source_indices = np.flatnonzero(dataset_array != outer_domain)
    source_domains = tuple(domain for domain in domain_order if domain != outer_domain)
    source_ids = tuple(samples[index] for index in source_indices)
    source_dataset_ids = tuple(domains[index] for index in source_indices)
    models: dict[str, CAIPredictor] = {}
    audits: list[TeacherFitAudit] = []
    for query_domain in source_domains:
        fitted = fit_outer_source_predictor(
            method=f"MVA_A5_P_A_{outer_domain}_{query_domain}",
            outer_domain=query_domain,
            specimen_ids=source_ids,
            dataset_ids=source_dataset_ids,
            domain_order=source_domains,
            targets=response[source_indices],
            metadata=meta[source_indices],
            embeddings=embeddings[source_indices],
            pca_dimensions=pca_dimensions,
            ridge_alpha=ridge_alpha,
            tie_tolerance=tie_tolerance,
        )
        if (
            outer_domain in fitted.model.fit_domains
            or query_domain in fitted.model.fit_domains
            or set(fitted.model.fit_domains)
            != set(domain_order) - {outer_domain, query_domain}
        ):
            raise A5TeacherError("teacher fit crossed an outer/query barrier")
        models[query_domain] = fitted.model
        audits.extend(
            _audit(value, outer_domain=outer_domain, query_domain=query_domain)
            for value in fitted.fit_audits
        )
    audit_tokens = tuple(
        (
            value.stage,
            value.held_out_target_domain,
            value.query_source_domain,
            value.query_domains,
            value.fit_domains,
            value.query_specimen_ids,
            value.fit_specimen_ids,
            value.pca_dimension,
            value.predictor_state_sha256,
        )
        for value in audits
    )
    state_sha256 = _state(
        "mva-a5-teacher-models",
        outer_domain,
        source_domains,
        tuple((domain, models[domain].state_sha256) for domain in source_domains),
        audit_tokens,
    )
    return TeacherModelBundle(
        outer_domain=outer_domain,
        source_domains=source_domains,
        models=MappingProxyType(models),
        fit_audits=tuple(audits),
        state_sha256=state_sha256,
    )


def _encode(encoder: _Encoder, images: list[np.ndarray]) -> np.ndarray:
    batches = [
        np.asarray(encoder.encode(images[start : start + 256]), dtype=np.float64)
        for start in range(0, len(images), 256)
    ]
    output = np.ascontiguousarray(np.vstack(batches), dtype=np.float64)
    if output.shape != (len(images), 512) or not np.all(np.isfinite(output)):
        raise A5TeacherError("teacher encoder output is invalid")
    return output


def _trajectory_state(trajectory: TeacherTrajectory) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "dataset_id": trajectory.dataset_id,
                "predictor_state_sha256": trajectory.predictor_state_sha256,
                "selected_actions": [
                    (action.cell_index, action.from_level, action.to_level)
                    for action in trajectory.selected_actions
                ],
                "specimen_id": trajectory.specimen_id,
                "states": [
                    {
                        "actions": [
                            (action.cell_index, action.from_level, action.to_level)
                            for action in state.actions
                        ],
                        "budget_after": state.budget_after,
                        "budget_before": state.budget_before,
                        "checkpoint": state.checkpoint,
                        "current_prediction": state.current_prediction,
                        "selected_index": state.selected_index,
                        "selected_prediction": state.selected_prediction,
                        "step": state.step,
                    }
                    for state in trajectory.states
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for state in trajectory.states:
        for name in ("global_features", "candidate_features", "values"):
            value = np.asarray(getattr(state, name))
            digest.update(name.encode("ascii"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _validate_trajectory_authority(
    inputs: tuple[TeacherTrajectoryInput, ...],
    initial_budget: float,
    checkpoints: tuple[float, ...],
) -> None:
    if (
        type(inputs) is not tuple
        or not inputs
        or any(type(value) is not TeacherTrajectoryInput for value in inputs)
        or len({value.specimen_id for value in inputs}) != len(inputs)
        or initial_budget not in INITIAL_BUDGETS
        or not checkpoints
        or tuple(sorted(set(checkpoints))) != checkpoints
        or any(
            not math.isfinite(float(value)) or not 0.0 < float(value) <= 0.25
            for value in checkpoints
        )
    ):
        raise A5TeacherError("teacher trajectory authority changed")


def _prepare_contexts(
    inputs: tuple[TeacherTrajectoryInput, ...],
    *,
    initial_budget: float,
    encoder: _Encoder,
) -> list[_TeacherContext]:
    prepared: list[
        tuple[
            TeacherTrajectoryInput,
            float,
            np.ndarray,
            AcquisitionGrid,
            MeasurementState,
            np.ndarray,
        ]
    ] = []
    for request in inputs:
        if (
            type(request.specimen_id) is not str
            or not request.specimen_id
            or type(request.dataset_id) is not str
            or not request.dataset_id
            or not isinstance(request.image, np.ndarray)
            or request.image.dtype != np.uint8
            or request.image.ndim != 3
            or request.image.shape[2] != 3
        ):
            raise A5TeacherError("teacher trajectory identity or image is invalid")
        response = float(request.target)
        metadata = np.asarray(request.metadata, dtype=np.float64)
        if (
            not math.isfinite(response)
            or metadata.ndim != 1
            or not np.all(np.isfinite(metadata))
        ):
            raise A5TeacherError("teacher target or metadata is invalid")
        grid = build_acquisition_grid(
            request.image.shape[0],
            request.image.shape[1],
            initial_budget=initial_budget,
        )
        state = initial_state(grid)
        current = reconstruct_measurement_state(
            request.image,
            grid,
            state,
            interpolation="bilinear",
            specimen_id=request.specimen_id,
            dataset_id=request.dataset_id,
        ).image
        prepared.append((request, response, metadata, grid, state, current))
    missing = [index for index, value in enumerate(inputs) if value.initial_embedding is None]
    encoded_missing = (
        _encode(encoder, [prepared[index][5] for index in missing])
        if missing
        else np.empty((0, 512), dtype=np.float64)
    )
    missing_embeddings = {
        index: encoded_missing[position] for position, index in enumerate(missing)
    }
    contexts: list[_TeacherContext] = []
    for index, (request, response, metadata, grid, state, current) in enumerate(prepared):
        current_embedding = np.asarray(
            missing_embeddings.get(index, request.initial_embedding), dtype=np.float64
        )
        if current_embedding.shape != (512,) or not np.all(np.isfinite(current_embedding)):
            raise A5TeacherError("initial teacher embedding is invalid")
        prediction = np.asarray(
            request.predictor.predict(
                metadata.reshape(1, -1), current_embedding.reshape(1, -1)
            ),
            dtype=np.float64,
        )
        if prediction.shape != (1,) or not np.all(np.isfinite(prediction)):
            raise A5TeacherError("initial teacher prediction is nonfinite")
        contexts.append(
            _TeacherContext(
                request=request,
                response=response,
                metadata=metadata,
                grid=grid,
                state=state,
                current=current,
                current_embedding=current_embedding,
                current_prediction=float(prediction[0]),
                patch_cache=RefinementPatchCache(image=request.image, grid=grid),
                records=[],
                selected_actions=[],
            )
        )
    return contexts


def generate_teacher_trajectories(
    inputs: tuple[TeacherTrajectoryInput, ...],
    *,
    initial_budget: float,
    checkpoints: tuple[float, ...],
    encoder: _Encoder,
) -> tuple[TeacherTrajectory, ...]:
    """Generate multiple independent oracle trajectories in encoder lockstep."""

    _validate_trajectory_authority(inputs, initial_budget, checkpoints)
    encoder.validate()
    contexts = _prepare_contexts(
        inputs, initial_budget=initial_budget, encoder=encoder
    )
    for checkpoint in checkpoints:
        while True:
            pending: list[tuple[_TeacherContext, object, list[np.ndarray], int, int]] = []
            all_candidates: list[np.ndarray] = []
            for context in contexts:
                if not fitting_actions(context.grid, context.state, checkpoint):
                    continue
                observation = build_policy_observation(
                    context.grid,
                    context.state,
                    current_reconstruction=context.current,
                    current_embedding=context.current_embedding,
                    current_prediction=context.current_prediction,
                    checkpoint=checkpoint,
                    maximum_budget=0.25,
                )
                current_mask = measurement_mask(context.grid, context.state)
                candidates = [
                    refine_reconstruction(
                        context.request.image,
                        context.grid,
                        context.state,
                        context.current,
                        action,
                        interpolation="bilinear",
                        current_mask=current_mask,
                        patch_cache=context.patch_cache,
                    )
                    for action in observation.actions
                ]
                start = len(all_candidates)
                all_candidates.extend(candidates)
                pending.append(
                    (context, observation, candidates, start, len(all_candidates))
                )
            if not pending:
                break
            all_embeddings = _encode(encoder, all_candidates)
            for context, observation, candidates, start, stop in pending:
                candidate_embeddings = all_embeddings[start:stop]
                candidate_predictions = np.asarray(
                    context.request.predictor.predict(
                        np.repeat(
                            context.metadata.reshape(1, -1), len(candidates), axis=0
                        ),
                        candidate_embeddings,
                    ),
                    dtype=np.float64,
                )
                if (
                    candidate_predictions.shape != (len(candidates),)
                    or not np.all(np.isfinite(candidate_predictions))
                ):
                    raise A5TeacherError("teacher candidate predictions are invalid")
                before = abs(context.response - context.current_prediction)
                values = before - np.abs(context.response - candidate_predictions)
                selected = max(
                    range(len(observation.actions)),
                    key=lambda index: (
                        float(values[index]),
                        -observation.actions[index].cell_index,
                        -observation.actions[index].to_level,
                    ),
                )
                action = observation.actions[selected]
                selected_budget = candidate_budget_record(
                    context.grid, context.state, action
                )
                context.records.append(
                    TeacherState(
                        step=context.step,
                        checkpoint=float(checkpoint),
                        actions=observation.actions,
                        global_features=observation.global_features,
                        candidate_features=observation.candidate_features,
                        values=_readonly(values, (len(candidates),)),
                        selected_index=selected,
                        budget_before=observation.used_budget,
                        budget_after=selected_budget.effective_budget,
                        current_prediction=context.current_prediction,
                        selected_prediction=float(candidate_predictions[selected]),
                    )
                )
                context.selected_actions.append(action)
                context.state = apply_action(context.grid, context.state, action)
                context.current = candidates[selected]
                context.current_embedding = candidate_embeddings[selected]
                context.current_prediction = float(candidate_predictions[selected])
                context.step += 1
    output: list[TeacherTrajectory] = []
    for context in contexts:
        result = TeacherTrajectory(
            specimen_id=context.request.specimen_id,
            dataset_id=context.request.dataset_id,
            predictor_state_sha256=context.request.predictor.state_sha256,
            states=tuple(context.records),
            selected_actions=tuple(context.selected_actions),
            state_sha256="",
        )
        output.append(replace(result, state_sha256=_trajectory_state(result)))
    return tuple(output)


def generate_teacher_trajectory(
    *,
    specimen_id: str,
    dataset_id: str,
    image: np.ndarray,
    target: float,
    metadata: np.ndarray,
    initial_budget: float,
    checkpoints: tuple[float, ...],
    predictor: _Predictor,
    encoder: _Encoder,
    initial_embedding: np.ndarray | None = None,
) -> TeacherTrajectory:
    """Generate one full oracle ranking trajectory through the batch engine."""

    return generate_teacher_trajectories(
        (
            TeacherTrajectoryInput(
                specimen_id=specimen_id,
                dataset_id=dataset_id,
                image=image,
                target=target,
                metadata=metadata,
                predictor=predictor,
                initial_embedding=initial_embedding,
            ),
        ),
        initial_budget=initial_budget,
        checkpoints=checkpoints,
        encoder=encoder,
    )[0]


def save_teacher_trajectory(
    path: str | Path, trajectory: TeacherTrajectory
) -> Path:
    """Atomically save one variable-candidate teacher trajectory without pickle."""

    if (
        type(trajectory) is not TeacherTrajectory
        or _trajectory_state(trajectory) != trajectory.state_sha256
    ):
        raise A5TeacherError("teacher trajectory content changed")
    state_count = len(trajectory.states)
    offsets = np.zeros(state_count + 1, dtype=np.int64)
    for index, state in enumerate(trajectory.states):
        offsets[index + 1] = offsets[index] + len(state.actions)
    if state_count:
        global_features = np.vstack(
            [state.global_features for state in trajectory.states]
        )
        candidate_features = np.vstack(
            [state.candidate_features for state in trajectory.states]
        )
        values = np.concatenate([state.values for state in trajectory.states])
        action_rows = np.asarray(
            [
                (action.cell_index, action.from_level, action.to_level)
                for state in trajectory.states
                for action in state.actions
            ],
            dtype=np.int64,
        )
    else:
        global_features = np.empty((0, 579), dtype=np.float64)
        candidate_features = np.empty((0, 8), dtype=np.float64)
        values = np.empty(0, dtype=np.float64)
        action_rows = np.empty((0, 3), dtype=np.int64)
    metadata = json.dumps(
        {
            "dataset_id": trajectory.dataset_id,
            "predictor_state_sha256": trajectory.predictor_state_sha256,
            "schema_version": 1,
            "specimen_id": trajectory.specimen_id,
            "state_sha256": trajectory.state_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    arrays = {
        "action_rows": action_rows,
        "budget_after": np.asarray(
            [state.budget_after for state in trajectory.states], dtype=np.float64
        ),
        "budget_before": np.asarray(
            [state.budget_before for state in trajectory.states], dtype=np.float64
        ),
        "candidate_features": candidate_features,
        "checkpoint": np.asarray(
            [state.checkpoint for state in trajectory.states], dtype=np.float64
        ),
        "current_prediction": np.asarray(
            [state.current_prediction for state in trajectory.states], dtype=np.float64
        ),
        "global_features": global_features,
        "metadata": np.frombuffer(metadata, dtype=np.uint8),
        "offsets": offsets,
        "selected_actions": np.asarray(
            [
                (action.cell_index, action.from_level, action.to_level)
                for action in trajectory.selected_actions
            ],
            dtype=np.int64,
        ).reshape(state_count, 3),
        "selected_index": np.asarray(
            [state.selected_index for state in trajectory.states], dtype=np.int64
        ),
        "selected_prediction": np.asarray(
            [state.selected_prediction for state in trajectory.states], dtype=np.float64
        ),
        "step": np.asarray(
            [state.step for state in trajectory.states], dtype=np.int64
        ),
        "values": values,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".npz"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if load_teacher_trajectory(destination).state_sha256 != trajectory.state_sha256:
        raise A5TeacherError("saved teacher trajectory validation failed")
    return destination


def load_teacher_trajectory(path: str | Path) -> TeacherTrajectory:
    """Load and independently validate one teacher trajectory cache."""

    expected_names = {
        "action_rows",
        "budget_after",
        "budget_before",
        "candidate_features",
        "checkpoint",
        "current_prediction",
        "global_features",
        "metadata",
        "offsets",
        "selected_actions",
        "selected_index",
        "selected_prediction",
        "step",
        "values",
    }
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != expected_names:
                raise A5TeacherError("teacher cache file roster changed")
            metadata = json.loads(
                np.asarray(archive["metadata"], dtype=np.uint8).tobytes()
            )
            if set(metadata) != {
                "dataset_id",
                "predictor_state_sha256",
                "schema_version",
                "specimen_id",
                "state_sha256",
            } or metadata["schema_version"] != 1:
                raise A5TeacherError("teacher cache metadata changed")
            offsets = np.asarray(archive["offsets"], dtype=np.int64)
            if (
                offsets.ndim != 1
                or offsets.size < 1
                or offsets[0] != 0
                or np.any(np.diff(offsets) < 1)
            ):
                raise A5TeacherError("teacher cache offsets changed")
            state_count = offsets.size - 1
            total = int(offsets[-1])
            global_features = np.asarray(
                archive["global_features"], dtype=np.float64
            )
            candidate_features = np.asarray(
                archive["candidate_features"], dtype=np.float64
            )
            values = np.asarray(archive["values"], dtype=np.float64)
            action_rows = np.asarray(archive["action_rows"], dtype=np.int64)
            selected_actions_array = np.asarray(
                archive["selected_actions"], dtype=np.int64
            )
            one_dimensional = {
                name: np.asarray(
                    archive[name],
                    dtype=(
                        np.int64
                        if name in {"selected_index", "step"}
                        else np.float64
                    ),
                )
                for name in (
                    "budget_after",
                    "budget_before",
                    "checkpoint",
                    "current_prediction",
                    "selected_index",
                    "selected_prediction",
                    "step",
                )
            }
            if (
                global_features.shape != (state_count, 579)
                or candidate_features.shape != (total, 8)
                or values.shape != (total,)
                or action_rows.shape != (total, 3)
                or selected_actions_array.shape != (state_count, 3)
                or any(value.shape != (state_count,) for value in one_dimensional.values())
                or not np.all(np.isfinite(global_features))
                or not np.all(np.isfinite(candidate_features))
                or not np.all(np.isfinite(values))
            ):
                raise A5TeacherError("teacher cache arrays changed")
            states: list[TeacherState] = []
            for index in range(state_count):
                start, stop = int(offsets[index]), int(offsets[index + 1])
                actions = tuple(
                    RefinementAction(*(int(item) for item in row))
                    for row in action_rows[start:stop]
                )
                selected_index = int(one_dimensional["selected_index"][index])
                if not 0 <= selected_index < len(actions):
                    raise A5TeacherError("teacher cache selected index changed")
                states.append(
                    TeacherState(
                        step=int(one_dimensional["step"][index]),
                        checkpoint=float(one_dimensional["checkpoint"][index]),
                        actions=actions,
                        global_features=_readonly(
                            global_features[index], (579,)
                        ),
                        candidate_features=_readonly(
                            candidate_features[start:stop], (stop - start, 8)
                        ),
                        values=_readonly(values[start:stop], (stop - start,)),
                        selected_index=selected_index,
                        budget_before=float(one_dimensional["budget_before"][index]),
                        budget_after=float(one_dimensional["budget_after"][index]),
                        current_prediction=float(
                            one_dimensional["current_prediction"][index]
                        ),
                        selected_prediction=float(
                            one_dimensional["selected_prediction"][index]
                        ),
                    )
                )
            selected_actions = tuple(
                RefinementAction(*(int(item) for item in row))
                for row in selected_actions_array
            )
    except A5TeacherError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise A5TeacherError("teacher trajectory cache cannot be loaded") from error
    result = TeacherTrajectory(
        specimen_id=str(metadata["specimen_id"]),
        dataset_id=str(metadata["dataset_id"]),
        predictor_state_sha256=str(metadata["predictor_state_sha256"]),
        states=tuple(states),
        selected_actions=selected_actions,
        state_sha256=str(metadata["state_sha256"]),
    )
    if (
        len(result.states) != len(result.selected_actions)
        or any(
            state.actions[state.selected_index] != selected
            for state, selected in zip(
                result.states, result.selected_actions, strict=True
            )
        )
        or _trajectory_state(result) != result.state_sha256
    ):
        raise A5TeacherError("teacher trajectory cache content digest changed")
    return result


__all__ = [
    "A5TeacherError",
    "TeacherFitAudit",
    "TeacherModelBundle",
    "TeacherState",
    "TeacherTrajectory",
    "TeacherTrajectoryInput",
    "fit_outer_safe_teacher_models",
    "generate_teacher_trajectories",
    "generate_teacher_trajectory",
    "load_teacher_trajectory",
    "save_teacher_trajectory",
]
