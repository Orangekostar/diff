"""Fold-safe supervised training for observable G1 action policies."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from torch import nn

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .contracts import (
    ACTION_SLOT_COUNT,
    CANDIDATE_FEATURE_DIMENSION,
    CELL_FEATURE_DIMENSION,
    GLOBAL_SCALAR_DIMENSION,
    RECONSTRUCTION_EMBEDDING_DIMENSION,
    CAIContextMode,
    G1PolicyState,
    TaskTokenMode,
)
from .dagger import REGISTERED_DAGGER_ITERATIONS
from .policy_model import SharedActionMLP, StructuredInspectionPolicy
from .rollout import ObservablePolicyScores
from .teacher import PrivilegedTeacherLabel
from .utility_distillation import (
    REGISTERED_TEMPERATURES,
    soft_utility_distillation_loss,
    teacher_distribution,
)

REGISTERED_LEARNING_RATES = (0.0001, 0.0003)
REGISTERED_WEIGHT_DECAYS = (0.0001, 0.001)
REGISTERED_INITIALIZATION_SEED = 2026090102
REGISTERED_MAX_EPOCHS = 80
REGISTERED_EARLY_STOPPING_PATIENCE = 12
REGISTERED_BATCH_SPECIMENS = 8
REGISTERED_GRADIENT_CLIP_NORM = 1.0


class G1PolicyTrainingError(ValueError):
    """Raised when supervised policy targets violate legal-action contracts."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _readonly(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise G1PolicyTrainingError("policy normalizer array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(shape)
    output.setflags(write=False)
    return output


class TrainingRoute(str, Enum):
    HARD_BC = "HARD_BC"
    SOFT_UTILITY_DISTILL = "SOFT_UTILITY_DISTILL"


class PolicyModelName(str, Enum):
    SHARED_ACTION_MLP = "SharedActionMLP"
    STRUCTURED_INSPECTION_POLICY = "StructuredInspectionPolicy"


@dataclass(frozen=True, slots=True)
class G1PolicyTrainingExample:
    outer_target: str
    source_domain: str
    specimen_sha256: str
    task: InspectionTask
    dagger_iteration: int
    policy_state: G1PolicyState
    teacher_label: PrivilegedTeacherLabel
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        label_slots = tuple(candidate.slot for candidate in self.teacher_label.candidates)
        legal_slots = tuple(np.flatnonzero(self.policy_state.legal_action_mask).tolist())
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domain) is not str
            or not self.source_domain
            or self.source_domain == self.outer_target
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or self.dagger_iteration not in REGISTERED_DAGGER_ITERATIONS
            or type(self.policy_state) is not G1PolicyState
            or type(self.teacher_label) is not PrivilegedTeacherLabel
            or self.policy_state.task is not self.task
            or self.teacher_label.task is not self.task
            or self.teacher_label.policy_state_sha256
            != self.policy_state.state_sha256
            or self.teacher_label.observation_sha256
            != self.policy_state.observation_sha256
            or label_slots != legal_slots
        ):
            message = (
                "outer target is forbidden from policy training"
                if self.source_domain == self.outer_target
                else "policy training example is invalid"
            )
            raise G1PolicyTrainingError(message)
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-policy-training-example",
                    "outer_target": self.outer_target,
                    "source_domain": self.source_domain,
                    "specimen": self.specimen_sha256,
                    "task": self.task.value,
                    "dagger_iteration": self.dagger_iteration,
                    "policy_state": self.policy_state.state_sha256,
                    "teacher_label": self.teacher_label.state_sha256,
                }
            ),
        )


def rebind_training_example_modes(
    example: G1PolicyTrainingExample,
    *,
    cai_context_mode: CAIContextMode,
    task_token_mode: TaskTokenMode,
) -> G1PolicyTrainingExample:
    """Derive registered actor controls from one maximal observable state."""

    if (
        type(example) is not G1PolicyTrainingExample
        or type(cai_context_mode) is not CAIContextMode
        or task_token_mode not in (TaskTokenMode.CORRECT, TaskTokenMode.NO_TASK)
    ):
        raise G1PolicyTrainingError("policy feature-mode rebinding is invalid")
    source = example.policy_state
    if (
        cai_context_mode is CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT
        and source.global_scalars[13] != 1.0
    ):
        raise G1PolicyTrainingError("shared CAI context is absent from the source state")
    global_scalars = np.array(source.global_scalars, dtype=np.float64, copy=True)
    if (
        cai_context_mode is CAIContextMode.TASK_SPECIFIC_MASKED
        and source.task is InspectionTask.FIELD
    ):
        global_scalars[12:14] = 0.0
    task_token = np.zeros(2, dtype=np.float64)
    if task_token_mode is TaskTokenMode.CORRECT:
        task_token[0 if source.task is InspectionTask.FIELD else 1] = 1.0
    state = G1PolicyState(
        task=source.task,
        task_token_mode=task_token_mode,
        cai_context_mode=cai_context_mode,
        observation_sha256=source.observation_sha256,
        reconstruction_sha256=source.reconstruction_sha256,
        surface_hypothesis_sha256=source.surface_hypothesis_sha256,
        grid_sha256=source.grid_sha256,
        reconstruction_embedding=source.reconstruction_embedding,
        global_scalars=global_scalars,
        task_token=task_token,
        cell_features=source.cell_features,
        candidate_features=source.candidate_features,
        legal_action_mask=source.legal_action_mask,
    )
    label = PrivilegedTeacherLabel(
        task=example.teacher_label.task,
        authorization_sha256=example.teacher_label.authorization_sha256,
        observation_sha256=example.teacher_label.observation_sha256,
        policy_state_sha256=state.state_sha256,
        selected_slot=example.teacher_label.selected_slot,
        candidates=example.teacher_label.candidates,
    )
    return G1PolicyTrainingExample(
        outer_target=example.outer_target,
        source_domain=example.source_domain,
        specimen_sha256=example.specimen_sha256,
        task=example.task,
        dagger_iteration=example.dagger_iteration,
        policy_state=state,
        teacher_label=label,
    )


@dataclass(frozen=True, slots=True)
class PolicyTrainingHyperparameters:
    model_name: PolicyModelName
    route: TrainingRoute
    cai_context_mode: CAIContextMode
    task_token_mode: TaskTokenMode
    tau: float | None
    learning_rate: float
    weight_decay: float
    dagger_iterations: int
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        temperature = None if self.tau is None else float(self.tau)
        learning_rate = float(self.learning_rate)
        weight_decay = float(self.weight_decay)
        if (
            type(self.model_name) is not PolicyModelName
            or type(self.route) is not TrainingRoute
            or type(self.cai_context_mode) is not CAIContextMode
            or self.task_token_mode not in (
                TaskTokenMode.CORRECT,
                TaskTokenMode.NO_TASK,
            )
            or learning_rate not in REGISTERED_LEARNING_RATES
            or weight_decay not in REGISTERED_WEIGHT_DECAYS
            or self.dagger_iterations not in REGISTERED_DAGGER_ITERATIONS
            or (
                self.route is TrainingRoute.HARD_BC
                and temperature is not None
            )
            or (
                self.route is TrainingRoute.SOFT_UTILITY_DISTILL
                and temperature not in REGISTERED_TEMPERATURES
            )
        ):
            raise G1PolicyTrainingError("policy hyperparameters leave the registered roster")
        object.__setattr__(self, "tau", temperature)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-policy-hyperparameters",
                    "model_name": self.model_name.value,
                    "route": self.route.value,
                    "cai_context_mode": self.cai_context_mode.value,
                    "task_token_mode": self.task_token_mode.value,
                    "tau": temperature,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "dagger_iterations": self.dagger_iterations,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class NormalizedPolicyTensors:
    reconstruction_embedding: np.ndarray
    global_scalars: np.ndarray
    task_token: np.ndarray
    cell_features: np.ndarray
    candidate_features: np.ndarray
    legal_action_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class G1ActorNormalizer:
    outer_target: str
    fit_domains: tuple[str, ...]
    fit_specimen_sha256s: tuple[str, ...]
    fit_state_sha256s: tuple[str, ...]
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray
    global_mean: np.ndarray
    global_scale: np.ndarray
    cell_mean: np.ndarray
    cell_scale: np.ndarray
    candidate_mean: np.ndarray
    candidate_scale: np.ndarray
    state_sha256: str

    def transform(self, state: G1PolicyState) -> NormalizedPolicyTensors:
        if type(state) is not G1PolicyState:
            raise G1PolicyTrainingError("issued observable policy state is required")
        return NormalizedPolicyTensors(
            reconstruction_embedding=np.array(
                (state.reconstruction_embedding - self.embedding_mean)
                / self.embedding_scale,
                dtype=np.float32,
                order="C",
                copy=True,
            ),
            global_scalars=np.array(
                (state.global_scalars - self.global_mean) / self.global_scale,
                dtype=np.float32,
                order="C",
                copy=True,
            ),
            task_token=np.array(
                state.task_token, dtype=np.float32, order="C", copy=True
            ),
            cell_features=np.array(
                (state.cell_features - self.cell_mean) / self.cell_scale,
                dtype=np.float32,
                order="C",
                copy=True,
            ),
            candidate_features=np.array(
                (state.candidate_features - self.candidate_mean)
                / self.candidate_scale,
                dtype=np.float32,
                order="C",
                copy=True,
            ),
            legal_action_mask=np.array(
                state.legal_action_mask,
                dtype=np.bool_,
                order="C",
                copy=True,
            ),
        )


def _checked_examples(
    examples: tuple[G1PolicyTrainingExample, ...],
) -> tuple[G1PolicyTrainingExample, ...]:
    if (
        type(examples) is not tuple
        or not examples
        or any(type(row) is not G1PolicyTrainingExample for row in examples)
        or len({row.state_sha256 for row in examples}) != len(examples)
        or len({row.outer_target for row in examples}) != 1
    ):
        raise G1PolicyTrainingError("policy training roster is invalid")
    specimen_domains: dict[str, str] = {}
    for row in examples:
        previous = specimen_domains.setdefault(row.specimen_sha256, row.source_domain)
        if previous != row.source_domain:
            raise G1PolicyTrainingError("one physical specimen appears in multiple domains")
    return examples


def _sorted_examples(
    examples: tuple[G1PolicyTrainingExample, ...],
) -> tuple[G1PolicyTrainingExample, ...]:
    return tuple(
        sorted(
            examples,
            key=lambda row: (
                row.source_domain,
                row.specimen_sha256,
                row.task.value,
                row.dagger_iteration,
                row.policy_state.state_sha256,
            ),
        )
    )


def equal_policy_training_weights(
    examples: tuple[G1PolicyTrainingExample, ...],
) -> np.ndarray:
    """Give every physical specimen equal mass and split it equally by task/state."""

    checked = _checked_examples(examples)
    tasks_by_specimen: dict[str, set[InspectionTask]] = {}
    states_by_specimen_task = Counter((row.specimen_sha256, row.task) for row in checked)
    for row in checked:
        tasks_by_specimen.setdefault(row.specimen_sha256, set()).add(row.task)
    expected_tasks = {InspectionTask.FIELD, InspectionTask.CAI}
    if any(tasks != expected_tasks for tasks in tasks_by_specimen.values()):
        raise G1PolicyTrainingError("each training specimen must contain both tasks")
    specimen_count = len(tasks_by_specimen)
    weights = np.asarray(
        [
            1.0
            / specimen_count
            / 2.0
            / states_by_specimen_task[(row.specimen_sha256, row.task)]
            for row in checked
        ],
        dtype=np.float64,
    )
    if not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1.0e-12):
        raise G1PolicyTrainingError("policy training weights do not sum to one")
    return weights


def _mean_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, dtype=np.float64)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    return mean, scale


def fit_policy_normalizer(
    examples: tuple[G1PolicyTrainingExample, ...],
    *,
    fit_domains: tuple[str, ...],
) -> G1ActorNormalizer:
    checked = _sorted_examples(_checked_examples(examples))
    if (
        type(fit_domains) is not tuple
        or not fit_domains
        or fit_domains != tuple(sorted(set(fit_domains)))
    ):
        raise G1PolicyTrainingError("normalizer fit-domain roster is invalid")
    outer_target = checked[0].outer_target
    available = {row.source_domain for row in checked}
    if outer_target in fit_domains or not set(fit_domains) <= available:
        raise G1PolicyTrainingError("outer or unavailable domain entered normalization")
    rows = tuple(row for row in checked if row.source_domain in fit_domains)
    if {row.source_domain for row in rows} != set(fit_domains):
        raise G1PolicyTrainingError("normalizer fit-domain roster is incomplete")
    embedding_mean, embedding_scale = _mean_scale(
        np.stack([row.policy_state.reconstruction_embedding for row in rows])
    )
    global_mean, global_scale = _mean_scale(
        np.stack([row.policy_state.global_scalars for row in rows])
    )
    cell_mean, cell_scale = _mean_scale(
        np.concatenate([row.policy_state.cell_features for row in rows], axis=0)
    )
    legal_candidates = np.concatenate(
        [
            row.policy_state.candidate_features[row.policy_state.legal_action_mask]
            for row in rows
        ],
        axis=0,
    )
    candidate_mean, candidate_scale = _mean_scale(legal_candidates)
    arrays = {
        "embedding_mean": _readonly(
            embedding_mean, (RECONSTRUCTION_EMBEDDING_DIMENSION,)
        ),
        "embedding_scale": _readonly(
            embedding_scale, (RECONSTRUCTION_EMBEDDING_DIMENSION,)
        ),
        "global_mean": _readonly(global_mean, (GLOBAL_SCALAR_DIMENSION,)),
        "global_scale": _readonly(global_scale, (GLOBAL_SCALAR_DIMENSION,)),
        "cell_mean": _readonly(cell_mean, (CELL_FEATURE_DIMENSION,)),
        "cell_scale": _readonly(cell_scale, (CELL_FEATURE_DIMENSION,)),
        "candidate_mean": _readonly(
            candidate_mean, (CANDIDATE_FEATURE_DIMENSION,)
        ),
        "candidate_scale": _readonly(
            candidate_scale, (CANDIDATE_FEATURE_DIMENSION,)
        ),
    }
    fit_specimens = tuple(sorted({row.specimen_sha256 for row in rows}))
    fit_states = tuple(sorted(row.state_sha256 for row in rows))
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": 1,
                "kind": "g1-actor-normalizer",
                "outer_target": outer_target,
                "fit_domains": fit_domains,
                "fit_specimens": fit_specimens,
                "fit_states": fit_states,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for name, array in sorted(arrays.items()):
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return G1ActorNormalizer(
        outer_target=outer_target,
        fit_domains=fit_domains,
        fit_specimen_sha256s=fit_specimens,
        fit_state_sha256s=fit_states,
        state_sha256=digest.hexdigest(),
        **arrays,
    )


def hard_behavior_cloning_loss(
    action_logits: torch.Tensor,
    selected_slots: torch.Tensor,
    legal_action_mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        not isinstance(action_logits, torch.Tensor)
        or not isinstance(selected_slots, torch.Tensor)
        or not isinstance(legal_action_mask, torch.Tensor)
        or action_logits.ndim != 2
        or action_logits.shape[1] != ACTION_SLOT_COUNT
        or selected_slots.shape != (action_logits.shape[0],)
        or selected_slots.dtype is not torch.long
        or legal_action_mask.shape != action_logits.shape
        or legal_action_mask.dtype is not torch.bool
        or not action_logits.is_floating_point()
        or torch.any(selected_slots < 0)
        or torch.any(selected_slots >= ACTION_SLOT_COUNT)
        or not torch.isfinite(action_logits[legal_action_mask]).all()
    ):
        raise G1PolicyTrainingError("hard behavior-cloning tensors are invalid")
    rows = torch.arange(action_logits.shape[0], device=action_logits.device)
    if not torch.all(legal_action_mask[rows, selected_slots]):
        raise G1PolicyTrainingError("selected action must be legal")
    log_probabilities = torch.log_softmax(
        action_logits.masked_fill(~legal_action_mask, -torch.inf), dim=1
    )
    per_state = -log_probabilities[rows, selected_slots]
    if sample_weights is None:
        return per_state.mean()
    if (
        not isinstance(sample_weights, torch.Tensor)
        or sample_weights.shape != per_state.shape
        or not torch.isfinite(sample_weights).all()
        or torch.any(sample_weights <= 0)
    ):
        raise G1PolicyTrainingError("sample weights are invalid")
    return torch.sum(per_state * sample_weights) / torch.sum(sample_weights)


@dataclass(frozen=True, slots=True)
class PolicyFitAudit:
    outer_target: str
    validation_domain: str | None
    fit_domains: tuple[str, ...]
    fit_specimen_sha256s: tuple[str, ...]
    validation_specimen_sha256s: tuple[str, ...]
    epochs_run: int
    selected_epoch: int
    best_validation_regret: float | None
    training_objectives: tuple[float, ...]
    normalizer_state_sha256: str
    training_roster_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or (
                self.validation_domain is not None
                and (
                    type(self.validation_domain) is not str
                    or self.validation_domain in self.fit_domains
                )
            )
            or self.outer_target in self.fit_domains
            or self.fit_domains != tuple(sorted(set(self.fit_domains)))
            or not self.fit_domains
            or type(self.epochs_run) is not int
            or type(self.selected_epoch) is not int
            or not 1 <= self.selected_epoch <= self.epochs_run
            or self.epochs_run != len(self.training_objectives)
            or not all(
                math.isfinite(value) and value >= 0.0
                for value in self.training_objectives
            )
            or (
                self.validation_domain is None
                and self.best_validation_regret is not None
            )
            or (
                self.validation_domain is not None
                and (
                    self.best_validation_regret is None
                    or not math.isfinite(self.best_validation_regret)
                    or self.best_validation_regret < 0.0
                )
            )
            or not _valid_sha256(self.normalizer_state_sha256)
            or not _valid_sha256(self.training_roster_sha256)
        ):
            raise G1PolicyTrainingError("policy fit audit is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-policy-fit-audit",
                    "outer_target": self.outer_target,
                    "validation_domain": self.validation_domain,
                    "fit_domains": self.fit_domains,
                    "fit_specimens": self.fit_specimen_sha256s,
                    "validation_specimens": self.validation_specimen_sha256s,
                    "epochs_run": self.epochs_run,
                    "selected_epoch": self.selected_epoch,
                    "best_validation_regret": self.best_validation_regret,
                    "training_objectives": self.training_objectives,
                    "normalizer": self.normalizer_state_sha256,
                    "training_roster": self.training_roster_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainedObservablePolicy:
    model: SharedActionMLP | StructuredInspectionPolicy
    normalizer: G1ActorNormalizer
    hyperparameters: PolicyTrainingHyperparameters
    audit: PolicyFitAudit
    model_state_sha256: str

    def __call__(self, state: G1PolicyState) -> ObservablePolicyScores:
        tensors = self.normalizer.transform(state)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.inference_mode():
            output = self.model(
                torch.from_numpy(tensors.reconstruction_embedding[None]).to(device),
                torch.from_numpy(tensors.global_scalars[None]).to(device),
                torch.from_numpy(tensors.task_token[None]).to(device),
                torch.from_numpy(tensors.cell_features[None]).to(device),
                torch.from_numpy(tensors.candidate_features[None]).to(device),
                torch.from_numpy(tensors.legal_action_mask[None]).to(device),
            )
            logits = np.asarray(
                output.action_logits[0].detach().cpu().numpy(), dtype=np.float64
            )
            stop_probability = float(torch.sigmoid(output.stop_logits[0]).cpu())
        return ObservablePolicyScores(
            policy_state_sha256=state.state_sha256,
            model_sha256=self.model_state_sha256,
            action_logits=logits,
            stop_probability=stop_probability,
        )


@dataclass(frozen=True, slots=True)
class _PolicyArrays:
    reconstruction_embedding: np.ndarray
    global_scalars: np.ndarray
    task_token: np.ndarray
    cell_features: np.ndarray
    candidate_features: np.ndarray
    legal_action_mask: np.ndarray
    selected_slots: np.ndarray
    target_probabilities: np.ndarray
    utilities: np.ndarray


def _policy_arrays(
    rows: tuple[G1PolicyTrainingExample, ...],
    normalizer: G1ActorNormalizer,
    hyperparameters: PolicyTrainingHyperparameters,
) -> _PolicyArrays:
    transformed = tuple(normalizer.transform(row.policy_state) for row in rows)
    probabilities: list[np.ndarray] = []
    utilities: list[np.ndarray] = []
    for row in rows:
        temperature = 1.0 if hyperparameters.tau is None else hyperparameters.tau
        distribution = teacher_distribution(row.teacher_label, tau=temperature)
        probabilities.append(distribution.probabilities)
        utilities.append(distribution.utilities)
    return _PolicyArrays(
        reconstruction_embedding=np.stack(
            [value.reconstruction_embedding for value in transformed]
        ),
        global_scalars=np.stack([value.global_scalars for value in transformed]),
        task_token=np.stack([value.task_token for value in transformed]),
        cell_features=np.stack([value.cell_features for value in transformed]),
        candidate_features=np.stack(
            [value.candidate_features for value in transformed]
        ),
        legal_action_mask=np.stack(
            [value.legal_action_mask for value in transformed]
        ),
        selected_slots=np.asarray(
            [row.teacher_label.selected_slot for row in rows], dtype=np.int64
        ),
        target_probabilities=np.asarray(probabilities, dtype=np.float32),
        utilities=np.asarray(utilities, dtype=np.float32),
    )


def _tensor(
    values: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.from_numpy(values[indices]).to(device)


def _forward_batch(
    model: SharedActionMLP | StructuredInspectionPolicy,
    arrays: _PolicyArrays,
    indices: np.ndarray,
    *,
    device: torch.device,
):
    return model(
        _tensor(arrays.reconstruction_embedding, indices, device=device),
        _tensor(arrays.global_scalars, indices, device=device),
        _tensor(arrays.task_token, indices, device=device),
        _tensor(arrays.cell_features, indices, device=device),
        _tensor(arrays.candidate_features, indices, device=device),
        _tensor(arrays.legal_action_mask, indices, device=device),
    )


def _specimen_batches(
    rows: tuple[G1PolicyTrainingExample, ...],
    *,
    epoch: int,
    seed: int,
    batch_specimens: int,
) -> tuple[np.ndarray, ...]:
    specimens = np.asarray(
        sorted({row.specimen_sha256 for row in rows}), dtype=object
    )
    order = np.random.Generator(np.random.PCG64(seed + epoch)).permutation(specimens)
    batches: list[np.ndarray] = []
    specimen_values = np.asarray([row.specimen_sha256 for row in rows], dtype=object)
    for start in range(0, order.size, batch_specimens):
        selected = set(order[start : start + batch_specimens].tolist())
        batches.append(np.flatnonzero(np.isin(specimen_values, tuple(selected))))
    return tuple(batches)


def _validation_regret(
    model: SharedActionMLP | StructuredInspectionPolicy,
    arrays: _PolicyArrays,
    weights: np.ndarray,
    *,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    all_indices = np.arange(weights.size, dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(all_indices), 64):
            indices = all_indices[start : start + 64]
            output = _forward_batch(model, arrays, indices, device=device)
            mask = _tensor(arrays.legal_action_mask, indices, device=device)
            utilities = _tensor(arrays.utilities, indices, device=device)
            probabilities = torch.softmax(
                output.action_logits.masked_fill(~mask, -torch.inf), dim=1
            )
            best = utilities.masked_fill(~mask, -torch.inf).max(dim=1).values
            regret = (
                probabilities
                * torch.where(mask, best.unsqueeze(1) - utilities, 0.0)
            ).sum(dim=1)
            total += float(
                torch.sum(
                    regret
                    * torch.from_numpy(weights[indices]).to(
                        device=device, dtype=regret.dtype
                    )
                ).cpu()
            )
    if not math.isfinite(total) or total < -1.0e-9:
        raise G1PolicyTrainingError("policy validation regret is invalid")
    return max(total, 0.0)


def _build_model(
    model_name: PolicyModelName,
) -> SharedActionMLP | StructuredInspectionPolicy:
    if model_name is PolicyModelName.SHARED_ACTION_MLP:
        return SharedActionMLP()
    if model_name is PolicyModelName.STRUCTURED_INSPECTION_POLICY:
        return StructuredInspectionPolicy()
    raise G1PolicyTrainingError("policy model name is invalid")


@contextmanager
def _deterministic_torch(seed: int, *, cpu: bool) -> Iterator[None]:
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_threads = torch.get_num_threads()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if cpu:
            torch.set_num_threads(1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(old_deterministic)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = old_cuda_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        if cpu:
            torch.set_num_threads(old_threads)


def _model_hash(
    model: SharedActionMLP | StructuredInspectionPolicy,
    normalizer: G1ActorNormalizer,
    hyperparameters: PolicyTrainingHyperparameters,
    audit: PolicyFitAudit,
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": 1,
                "kind": "g1-trained-observable-policy",
                "normalizer": normalizer.state_sha256,
                "hyperparameters": hyperparameters.state_sha256,
                "audit": audit.state_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for name, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_fit_request(
    examples: tuple[G1PolicyTrainingExample, ...],
    hyperparameters: PolicyTrainingHyperparameters,
    *,
    validation_domain: str | None,
    max_epochs: int,
    patience: int | None,
    batch_specimens: int,
    gradient_clip_norm: float,
    seed: int,
    device: str,
) -> tuple[G1PolicyTrainingExample, ...]:
    checked = _sorted_examples(_checked_examples(examples))
    source_domains = tuple(sorted({row.source_domain for row in checked}))
    if (
        type(hyperparameters) is not PolicyTrainingHyperparameters
        or len(source_domains) != 5
        or checked[0].outer_target in source_domains
        or (
            validation_domain is not None
            and validation_domain not in source_domains
        )
        or type(max_epochs) is not int
        or not 1 <= max_epochs <= REGISTERED_MAX_EPOCHS
        or (
            validation_domain is not None
            and (type(patience) is not int or not 1 <= patience <= REGISTERED_EARLY_STOPPING_PATIENCE)
        )
        or (validation_domain is None and patience is not None)
        or type(batch_specimens) is not int
        or not 1 <= batch_specimens <= REGISTERED_BATCH_SPECIMENS
        or float(gradient_clip_norm) != REGISTERED_GRADIENT_CLIP_NORM
        or seed != REGISTERED_INITIALIZATION_SEED
        or type(device) is not str
        or not device
    ):
        raise G1PolicyTrainingError("policy fit request leaves the registered protocol")
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise G1PolicyTrainingError("policy training device is invalid") from error
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise G1PolicyTrainingError("registered CUDA policy device is unavailable")
    selected = tuple(
        row
        for row in checked
        if row.dagger_iteration <= hyperparameters.dagger_iterations
    )
    if any(
        row.policy_state.cai_context_mode is not hyperparameters.cai_context_mode
        or row.policy_state.task_token_mode is not hyperparameters.task_token_mode
        for row in selected
    ):
        raise G1PolicyTrainingError("policy feature mode contradicts hyperparameters")
    equal_policy_training_weights(selected)
    return selected


def _fit_observable_policy(
    examples: tuple[G1PolicyTrainingExample, ...],
    *,
    validation_domain: str | None,
    hyperparameters: PolicyTrainingHyperparameters,
    max_epochs: int,
    patience: int | None,
    batch_specimens: int,
    gradient_clip_norm: float,
    seed: int,
    device: str,
) -> TrainedObservablePolicy:
    checked = _validate_fit_request(
        examples,
        hyperparameters,
        validation_domain=validation_domain,
        max_epochs=max_epochs,
        patience=patience,
        batch_specimens=batch_specimens,
        gradient_clip_norm=gradient_clip_norm,
        seed=seed,
        device=device,
    )
    fit_domains = tuple(
        sorted(
            {
                row.source_domain
                for row in checked
                if row.source_domain != validation_domain
            }
        )
    )
    fit_rows = tuple(row for row in checked if row.source_domain in fit_domains)
    validation_rows = tuple(
        row for row in checked if row.source_domain == validation_domain
    )
    if not fit_rows or (validation_domain is not None and not validation_rows):
        raise G1PolicyTrainingError("policy fit or validation roster is empty")
    normalizer = fit_policy_normalizer(checked, fit_domains=fit_domains)
    fit_weights = equal_policy_training_weights(fit_rows)
    fit_arrays = _policy_arrays(fit_rows, normalizer, hyperparameters)
    validation_weights = (
        None
        if validation_domain is None
        else equal_policy_training_weights(validation_rows)
    )
    validation_arrays = (
        None
        if validation_domain is None
        else _policy_arrays(validation_rows, normalizer, hyperparameters)
    )
    torch_device = torch.device(device)
    trace: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_regret = math.inf
    selected_epoch = max_epochs
    stale = 0
    with _deterministic_torch(seed, cpu=torch_device.type == "cpu"):
        model = _build_model(hyperparameters.model_name).to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hyperparameters.learning_rate,
            weight_decay=hyperparameters.weight_decay,
        )
        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            objective = 0.0
            for indices in _specimen_batches(
                fit_rows,
                epoch=epoch,
                seed=seed,
                batch_specimens=batch_specimens,
            ):
                output = _forward_batch(
                    model, fit_arrays, indices, device=torch_device
                )
                mask = _tensor(
                    fit_arrays.legal_action_mask, indices, device=torch_device
                )
                weights = torch.from_numpy(fit_weights[indices]).to(
                    device=torch_device, dtype=output.action_logits.dtype
                )
                if hyperparameters.route is TrainingRoute.HARD_BC:
                    loss = hard_behavior_cloning_loss(
                        output.action_logits,
                        _tensor(
                            fit_arrays.selected_slots, indices, device=torch_device
                        ),
                        mask,
                        sample_weights=weights,
                    )
                else:
                    loss = soft_utility_distillation_loss(
                        output.action_logits,
                        _tensor(
                            fit_arrays.target_probabilities,
                            indices,
                            device=torch_device,
                        ),
                        mask,
                        sample_weights=weights,
                    )
                mass = float(np.sum(fit_weights[indices]))
                (loss * mass).backward()
                objective += float(loss.detach().cpu()) * mass
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip_norm
            )
            optimizer.step()
            trace.append(objective)
            if validation_domain is None:
                continue
            assert validation_arrays is not None
            assert validation_weights is not None
            regret = _validation_regret(
                model,
                validation_arrays,
                validation_weights,
                device=torch_device,
            )
            if regret < best_regret - 1.0e-12:
                best_regret = regret
                selected_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= int(patience):
                    break
        if validation_domain is not None:
            if best_state is None:
                raise G1PolicyTrainingError("early stopping selected no policy")
            model.load_state_dict(best_state, strict=True)
        model.cpu().eval()
    training_roster_sha256 = _json_sha(
        tuple(row.state_sha256 for row in fit_rows)
    )
    audit = PolicyFitAudit(
        outer_target=checked[0].outer_target,
        validation_domain=validation_domain,
        fit_domains=fit_domains,
        fit_specimen_sha256s=tuple(
            sorted({row.specimen_sha256 for row in fit_rows})
        ),
        validation_specimen_sha256s=tuple(
            sorted({row.specimen_sha256 for row in validation_rows})
        ),
        epochs_run=len(trace),
        selected_epoch=selected_epoch,
        best_validation_regret=(
            None if validation_domain is None else float(best_regret)
        ),
        training_objectives=tuple(trace),
        normalizer_state_sha256=normalizer.state_sha256,
        training_roster_sha256=training_roster_sha256,
    )
    model_state_sha256 = _model_hash(model, normalizer, hyperparameters, audit)
    return TrainedObservablePolicy(
        model=model,
        normalizer=normalizer,
        hyperparameters=hyperparameters,
        audit=audit,
        model_state_sha256=model_state_sha256,
    )


def fit_inner_observable_policy(
    examples: tuple[G1PolicyTrainingExample, ...],
    *,
    validation_domain: str,
    hyperparameters: PolicyTrainingHyperparameters,
    max_epochs: int = REGISTERED_MAX_EPOCHS,
    patience: int = REGISTERED_EARLY_STOPPING_PATIENCE,
    batch_specimens: int = REGISTERED_BATCH_SPECIMENS,
    gradient_clip_norm: float = REGISTERED_GRADIENT_CLIP_NORM,
    seed: int = REGISTERED_INITIALIZATION_SEED,
    device: str = "cpu",
) -> TrainedObservablePolicy:
    return _fit_observable_policy(
        examples,
        validation_domain=validation_domain,
        hyperparameters=hyperparameters,
        max_epochs=max_epochs,
        patience=patience,
        batch_specimens=batch_specimens,
        gradient_clip_norm=gradient_clip_norm,
        seed=seed,
        device=device,
    )


def fit_final_observable_policy(
    examples: tuple[G1PolicyTrainingExample, ...],
    *,
    hyperparameters: PolicyTrainingHyperparameters,
    selected_epochs: int,
    batch_specimens: int = REGISTERED_BATCH_SPECIMENS,
    gradient_clip_norm: float = REGISTERED_GRADIENT_CLIP_NORM,
    seed: int = REGISTERED_INITIALIZATION_SEED,
    device: str = "cpu",
) -> TrainedObservablePolicy:
    return _fit_observable_policy(
        examples,
        validation_domain=None,
        hyperparameters=hyperparameters,
        max_epochs=selected_epochs,
        patience=None,
        batch_specimens=batch_specimens,
        gradient_clip_norm=gradient_clip_norm,
        seed=seed,
        device=device,
    )


__all__ = [
    "REGISTERED_BATCH_SPECIMENS",
    "REGISTERED_EARLY_STOPPING_PATIENCE",
    "REGISTERED_GRADIENT_CLIP_NORM",
    "REGISTERED_INITIALIZATION_SEED",
    "REGISTERED_LEARNING_RATES",
    "REGISTERED_MAX_EPOCHS",
    "REGISTERED_WEIGHT_DECAYS",
    "G1ActorNormalizer",
    "G1PolicyTrainingError",
    "G1PolicyTrainingExample",
    "NormalizedPolicyTensors",
    "PolicyFitAudit",
    "PolicyModelName",
    "PolicyTrainingHyperparameters",
    "TrainedObservablePolicy",
    "TrainingRoute",
    "equal_policy_training_weights",
    "fit_final_observable_policy",
    "fit_inner_observable_policy",
    "fit_policy_normalizer",
    "hard_behavior_cloning_loss",
    "rebind_training_example_modes",
]
