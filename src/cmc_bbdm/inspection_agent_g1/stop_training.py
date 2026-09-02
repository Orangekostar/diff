"""Source-only training for the independent observable G1 STOP head."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .contracts import ACTION_SLOT_COUNT, G1PolicyState
from .policy_model import SharedActionMLP, StructuredInspectionPolicy
from .policy_training import (
    REGISTERED_BATCH_SPECIMENS,
    REGISTERED_EARLY_STOPPING_PATIENCE,
    REGISTERED_GRADIENT_CLIP_NORM,
    REGISTERED_INITIALIZATION_SEED,
    REGISTERED_MAX_EPOCHS,
    G1ActorNormalizer,
    TrainedObservablePolicy,
    _deterministic_torch,
    rebind_training_example_modes,
)
from .rollout import ObservablePolicyScores
from .stop_bank import G1StopBankRecord
from .stopping_policy import SourceStopLabel, observable_stop_loss
from .teacher_bank import G1TeacherBankRecord


class G1StopTrainingError(ValueError):
    """Raised when STOP training crosses a source or actor boundary."""


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


@dataclass(frozen=True, slots=True)
class G1StopTrainingExample:
    outer_target: str
    source_domain: str
    specimen_sha256: str
    task: InspectionTask
    source_policy_state_sha256: str
    action_example_sha256: str
    policy_state: G1PolicyState
    label: SourceStopLabel
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domain) is not str
            or not self.source_domain
            or self.source_domain == self.outer_target
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or not _valid_sha256(self.source_policy_state_sha256)
            or not _valid_sha256(self.action_example_sha256)
            or type(self.policy_state) is not G1PolicyState
            or self.policy_state.task is not self.task
            or type(self.label) is not SourceStopLabel
            or self.label.source_domain != self.source_domain
            or self.label.specimen_sha256 != self.specimen_sha256
            or self.label.task is not self.task
            or self.label.policy_state_sha256 != self.source_policy_state_sha256
        ):
            raise G1StopTrainingError("STOP training example is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-stop-training-example",
                    "outer_target": self.outer_target,
                    "source_domain": self.source_domain,
                    "specimen_sha256": self.specimen_sha256,
                    "task": self.task.value,
                    "source_policy_state_sha256": self.source_policy_state_sha256,
                    "action_example_sha256": self.action_example_sha256,
                    "policy_state": self.policy_state.state_sha256,
                    "label": self.label.state_sha256,
                }
            ),
        )


def join_g1_stop_training_examples(
    action_records: tuple[G1TeacherBankRecord, ...],
    stop_records: tuple[G1StopBankRecord, ...],
    *,
    action_policy: TrainedObservablePolicy,
) -> tuple[G1StopTrainingExample, ...]:
    if (
        type(action_records) is not tuple
        or not action_records
        or any(type(row) is not G1TeacherBankRecord for row in action_records)
        or type(stop_records) is not tuple
        or not stop_records
        or any(type(row) is not G1StopBankRecord for row in stop_records)
        or type(action_policy) is not TrainedObservablePolicy
    ):
        raise G1StopTrainingError("STOP training join request is invalid")
    action_by_sha = {row.example.state_sha256: row for row in action_records}
    if len(action_by_sha) != len(action_records):
        raise G1StopTrainingError("action-bank example identity is duplicated")
    output = []
    consumed: set[str] = set()
    hp = action_policy.hyperparameters
    for stop in stop_records:
        try:
            action = action_by_sha[stop.action_example_sha256]
        except KeyError as error:
            raise G1StopTrainingError("STOP label has no action-bank state") from error
        original = action.example
        if (
            original.outer_target != stop.outer_target
            or original.source_domain != stop.source_domain
            or original.specimen_sha256 != stop.specimen_sha256
            or original.task is not stop.task
            or original.policy_state.state_sha256 != stop.policy_state_sha256
        ):
            raise G1StopTrainingError("STOP/action-bank join identity changed")
        rebound = rebind_training_example_modes(
            original,
            cai_context_mode=hp.cai_context_mode,
            task_token_mode=hp.task_token_mode,
        )
        output.append(
            G1StopTrainingExample(
                outer_target=stop.outer_target,
                source_domain=stop.source_domain,
                specimen_sha256=stop.specimen_sha256,
                task=stop.task,
                source_policy_state_sha256=stop.policy_state_sha256,
                action_example_sha256=stop.action_example_sha256,
                policy_state=rebound.policy_state,
                label=stop.label,
            )
        )
        consumed.add(original.state_sha256)
    if consumed != set(action_by_sha):
        raise G1StopTrainingError("STOP and action-bank state rosters differ")
    return tuple(output)


def _checked_examples(
    examples: tuple[G1StopTrainingExample, ...],
) -> tuple[G1StopTrainingExample, ...]:
    if (
        type(examples) is not tuple
        or not examples
        or any(type(row) is not G1StopTrainingExample for row in examples)
        or len({row.state_sha256 for row in examples}) != len(examples)
        or len({row.outer_target for row in examples}) != 1
    ):
        raise G1StopTrainingError("STOP training roster is invalid")
    return tuple(
        sorted(
            examples,
            key=lambda row: (
                row.source_domain,
                row.specimen_sha256,
                row.task.value,
                row.source_policy_state_sha256,
            ),
        )
    )


def equal_stop_training_weights(
    examples: tuple[G1StopTrainingExample, ...],
) -> np.ndarray:
    rows = _checked_examples(examples)
    specimens = tuple(sorted({row.specimen_sha256 for row in rows}))
    groups = {
        (specimen, task): tuple(
            index
            for index, row in enumerate(rows)
            if row.specimen_sha256 == specimen and row.task is task
        )
        for specimen in specimens
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    }
    if any(not indices for indices in groups.values()):
        raise G1StopTrainingError("STOP roster must contain both tasks per specimen")
    weights = np.zeros(len(rows), dtype=np.float64)
    for indices in groups.values():
        weights[np.asarray(indices, dtype=np.int64)] = 1.0 / (
            len(specimens) * 2 * len(indices)
        )
    if not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1.0e-12):
        raise G1StopTrainingError("STOP training weights are invalid")
    return weights


@dataclass(frozen=True, slots=True)
class StopFitAudit:
    outer_target: str
    validation_domain: str | None
    fit_domains: tuple[str, ...]
    fit_specimen_sha256s: tuple[str, ...]
    validation_specimen_sha256s: tuple[str, ...]
    epochs_run: int
    selected_epoch: int
    best_validation_loss: float | None
    training_objectives: tuple[float, ...]
    base_action_model_sha256: str
    action_backbone_sha256: str
    training_roster_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or self.outer_target in self.fit_domains
            or self.fit_domains != tuple(sorted(set(self.fit_domains)))
            or not self.fit_domains
            or (
                self.validation_domain is not None
                and self.validation_domain in self.fit_domains
            )
            or type(self.epochs_run) is not int
            or type(self.selected_epoch) is not int
            or not 1 <= self.selected_epoch <= self.epochs_run
            or len(self.training_objectives) != self.epochs_run
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.training_objectives
            )
            or (
                self.validation_domain is None
                and self.best_validation_loss is not None
            )
            or (
                self.validation_domain is not None
                and (
                    self.best_validation_loss is None
                    or not math.isfinite(self.best_validation_loss)
                    or self.best_validation_loss < 0.0
                )
            )
            or not _valid_sha256(self.base_action_model_sha256)
            or not _valid_sha256(self.action_backbone_sha256)
            or not _valid_sha256(self.training_roster_sha256)
        ):
            raise G1StopTrainingError("STOP fit audit is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-stop-fit-audit",
                    "outer_target": self.outer_target,
                    "validation_domain": self.validation_domain,
                    "fit_domains": self.fit_domains,
                    "fit_specimens": self.fit_specimen_sha256s,
                    "validation_specimens": self.validation_specimen_sha256s,
                    "epochs_run": self.epochs_run,
                    "selected_epoch": self.selected_epoch,
                    "best_validation_loss": self.best_validation_loss,
                    "training_objectives": self.training_objectives,
                    "base_action_model": self.base_action_model_sha256,
                    "action_backbone": self.action_backbone_sha256,
                    "training_roster": self.training_roster_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainedObservableStopPolicy:
    model: SharedActionMLP | StructuredInspectionPolicy
    normalizer: G1ActorNormalizer
    audit: StopFitAudit
    base_action_model_sha256: str
    action_backbone_sha256: str
    model_state_sha256: str

    def __call__(self, state: G1PolicyState) -> ObservablePolicyScores:
        return self.score_batch((state,))[0]

    def score_batch(
        self,
        states: tuple[G1PolicyState, ...],
    ) -> tuple[ObservablePolicyScores, ...]:
        if (
            type(states) is not tuple
            or not states
            or any(type(state) is not G1PolicyState for state in states)
        ):
            raise G1StopTrainingError("observable STOP batch is invalid")
        tensors = tuple(self.normalizer.transform(state) for state in states)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.inference_mode():
            output = self.model(
                torch.from_numpy(
                    np.stack([value.reconstruction_embedding for value in tensors])
                ).to(device),
                torch.from_numpy(
                    np.stack([value.global_scalars for value in tensors])
                ).to(device),
                torch.from_numpy(
                    np.stack([value.task_token for value in tensors])
                ).to(device),
                torch.from_numpy(
                    np.stack([value.cell_features for value in tensors])
                ).to(device),
                torch.from_numpy(
                    np.stack([value.candidate_features for value in tensors])
                ).to(device),
                torch.from_numpy(
                    np.stack([value.legal_action_mask for value in tensors])
                ).to(device),
            )
        logits = np.asarray(
            output.action_logits.detach().cpu().numpy(), dtype=np.float64
        )
        probabilities = np.asarray(
            torch.sigmoid(output.stop_logits).detach().cpu().numpy(),
            dtype=np.float64,
        )
        if (
            logits.shape != (len(states), ACTION_SLOT_COUNT)
            or probabilities.shape != (len(states),)
        ):
            raise G1StopTrainingError("observable STOP batch output is invalid")
        return tuple(
            ObservablePolicyScores(
                policy_state_sha256=state.state_sha256,
                model_sha256=self.model_state_sha256,
                action_logits=row_logits,
                stop_probability=float(probability),
            )
            for state, row_logits, probability in zip(
                states,
                logits,
                probabilities,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class _ObservableArrays:
    reconstruction_embedding: np.ndarray
    global_scalars: np.ndarray
    task_token: np.ndarray
    cell_features: np.ndarray
    candidate_features: np.ndarray
    legal_action_mask: np.ndarray


def _observable_arrays(
    rows: tuple[G1StopTrainingExample, ...],
    normalizer: G1ActorNormalizer,
) -> _ObservableArrays:
    transformed = tuple(normalizer.transform(row.policy_state) for row in rows)
    return _ObservableArrays(
        reconstruction_embedding=np.stack(
            [row.reconstruction_embedding for row in transformed]
        ),
        global_scalars=np.stack([row.global_scalars for row in transformed]),
        task_token=np.stack([row.task_token for row in transformed]),
        cell_features=np.stack([row.cell_features for row in transformed]),
        candidate_features=np.stack([row.candidate_features for row in transformed]),
        legal_action_mask=np.stack([row.legal_action_mask for row in transformed]),
    )


def _tensor(value: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value[indices]).to(device)


def _observable_contexts(
    model: SharedActionMLP | StructuredInspectionPolicy,
    arrays: _ObservableArrays,
    *,
    device: torch.device,
) -> np.ndarray:
    count = len(arrays.reconstruction_embedding)
    contexts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, 64):
            indices = np.arange(start, min(start + 64, count), dtype=np.int64)
            output = model(
                _tensor(arrays.reconstruction_embedding, indices, device),
                _tensor(arrays.global_scalars, indices, device),
                _tensor(arrays.task_token, indices, device),
                _tensor(arrays.cell_features, indices, device),
                _tensor(arrays.candidate_features, indices, device),
                _tensor(arrays.legal_action_mask, indices, device),
            )
            contexts.append(output.global_context.detach().cpu().numpy())
    result = np.ascontiguousarray(np.concatenate(contexts), dtype=np.float32)
    if result.shape != (count, 128) or not np.all(np.isfinite(result)):
        raise G1StopTrainingError("observable STOP contexts are invalid")
    return result


def _backbone_hash(model: SharedActionMLP | StructuredInspectionPolicy) -> str:
    digest = hashlib.sha256(b"inspection-agent-g1-action-backbone-v1")
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("stop_head."):
            continue
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _model_hash(
    model: SharedActionMLP | StructuredInspectionPolicy,
    audit: StopFitAudit,
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": 1,
                "kind": "g1-trained-observable-stop-policy",
                "audit": audit.state_sha256,
                "base_action_model": audit.base_action_model_sha256,
                "action_backbone": audit.action_backbone_sha256,
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


def _specimen_batches(
    rows: tuple[G1StopTrainingExample, ...],
    *,
    epoch: int,
    seed: int,
    batch_specimens: int,
) -> tuple[np.ndarray, ...]:
    specimens = np.asarray(sorted({row.specimen_sha256 for row in rows}), dtype=object)
    order = np.random.Generator(np.random.PCG64(seed + epoch)).permutation(specimens)
    values = np.asarray([row.specimen_sha256 for row in rows], dtype=object)
    return tuple(
        np.flatnonzero(np.isin(values, tuple(order[start : start + batch_specimens])))
        for start in range(0, len(order), batch_specimens)
    )


def _fit_observable_stop_head(
    action_policy: TrainedObservablePolicy,
    examples: tuple[G1StopTrainingExample, ...],
    *,
    validation_domain: str | None,
    max_epochs: int,
    patience: int | None,
    batch_specimens: int,
    gradient_clip_norm: float,
    seed: int,
    device: str,
) -> TrainedObservableStopPolicy:
    rows = _checked_examples(examples)
    if type(action_policy) is not TrainedObservablePolicy:
        raise G1StopTrainingError("issued action policy is required for STOP fitting")
    domains = tuple(sorted({row.source_domain for row in rows}))
    fit_domains = tuple(domain for domain in domains if domain != validation_domain)
    audit = action_policy.audit
    hp = action_policy.hyperparameters
    if (
        len(domains) != 5
        or rows[0].outer_target in domains
        or validation_domain != audit.validation_domain
        or fit_domains != audit.fit_domains
        or rows[0].outer_target != audit.outer_target
        or any(
            row.policy_state.cai_context_mode is not hp.cai_context_mode
            or row.policy_state.task_token_mode is not hp.task_token_mode
            for row in rows
        )
        or type(max_epochs) is not int
        or not 1 <= max_epochs <= REGISTERED_MAX_EPOCHS
        or (
            validation_domain is not None
            and (
                validation_domain not in domains
                or type(patience) is not int
                or not 1 <= patience <= REGISTERED_EARLY_STOPPING_PATIENCE
            )
        )
        or (validation_domain is None and patience is not None)
        or type(batch_specimens) is not int
        or not 1 <= batch_specimens <= REGISTERED_BATCH_SPECIMENS
        or float(gradient_clip_norm) != REGISTERED_GRADIENT_CLIP_NORM
        or seed != REGISTERED_INITIALIZATION_SEED
        or type(device) is not str
        or not device
    ):
        raise G1StopTrainingError("STOP fit request leaves the registered protocol")
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise G1StopTrainingError("STOP training device is invalid") from error
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise G1StopTrainingError("registered CUDA STOP device is unavailable")
    fit_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.source_domain in fit_domains],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.source_domain == validation_domain],
        dtype=np.int64,
    )
    if not len(fit_indices) or (validation_domain is not None and not len(validation_indices)):
        raise G1StopTrainingError("STOP fit or validation roster is empty")
    fit_rows = tuple(rows[index] for index in fit_indices)
    validation_rows = tuple(rows[index] for index in validation_indices)
    fit_weights = equal_stop_training_weights(fit_rows)
    validation_weights = (
        None
        if validation_domain is None
        else equal_stop_training_weights(validation_rows)
    )
    model = copy.deepcopy(action_policy.model).to(torch_device)
    base_backbone_sha = _backbone_hash(model)
    arrays = _observable_arrays(rows, action_policy.normalizer)
    contexts = _observable_contexts(model, arrays, device=torch_device)
    labels = np.asarray([row.label.is_sufficient for row in rows], dtype=np.bool_)
    fit_contexts = contexts[fit_indices]
    fit_labels = labels[fit_indices]
    validation_contexts = contexts[validation_indices]
    validation_labels = labels[validation_indices]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.stop_head.parameters():
        parameter.requires_grad_(True)
    trace: list[float] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    selected_epoch = max_epochs
    stale = 0
    with _deterministic_torch(seed, cpu=torch_device.type == "cpu"):
        optimizer = torch.optim.AdamW(
            model.stop_head.parameters(),
            lr=hp.learning_rate,
            weight_decay=hp.weight_decay,
        )
        for epoch in range(1, max_epochs + 1):
            model.stop_head.train()
            optimizer.zero_grad(set_to_none=True)
            objective = 0.0
            for indices in _specimen_batches(
                fit_rows,
                epoch=epoch,
                seed=seed,
                batch_specimens=batch_specimens,
            ):
                logits = model.stop_head(
                    torch.from_numpy(fit_contexts[indices]).to(torch_device)
                ).squeeze(-1)
                weights = torch.from_numpy(fit_weights[indices]).to(
                    device=torch_device,
                    dtype=logits.dtype,
                )
                loss = observable_stop_loss(
                    logits,
                    torch.from_numpy(fit_labels[indices]).to(torch_device),
                    sample_weights=weights,
                )
                mass = float(np.sum(fit_weights[indices]))
                (loss * mass).backward()
                objective += float(loss.detach().cpu()) * mass
            nn.utils.clip_grad_norm_(
                model.stop_head.parameters(), max_norm=gradient_clip_norm
            )
            optimizer.step()
            trace.append(objective)
            if validation_domain is None:
                continue
            assert validation_weights is not None
            model.stop_head.eval()
            with torch.inference_mode():
                validation_logits = model.stop_head(
                    torch.from_numpy(validation_contexts).to(torch_device)
                ).squeeze(-1)
                validation_loss = float(
                    observable_stop_loss(
                        validation_logits,
                        torch.from_numpy(validation_labels).to(torch_device),
                        sample_weights=torch.from_numpy(validation_weights).to(
                            device=torch_device,
                            dtype=validation_logits.dtype,
                        ),
                    ).cpu()
                )
            if validation_loss < best_loss - 1.0e-12:
                best_loss = validation_loss
                selected_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.stop_head.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= int(patience):
                    break
        if validation_domain is not None:
            if best_state is None:
                raise G1StopTrainingError("early stopping selected no STOP head")
            model.stop_head.load_state_dict(best_state, strict=True)
    model.cpu().eval()
    final_backbone_sha = _backbone_hash(model)
    if final_backbone_sha != base_backbone_sha:
        raise G1StopTrainingError("STOP fitting changed action-policy parameters")
    stop_audit = StopFitAudit(
        outer_target=rows[0].outer_target,
        validation_domain=validation_domain,
        fit_domains=fit_domains,
        fit_specimen_sha256s=tuple(sorted({row.specimen_sha256 for row in fit_rows})),
        validation_specimen_sha256s=tuple(
            sorted({row.specimen_sha256 for row in validation_rows})
        ),
        epochs_run=len(trace),
        selected_epoch=selected_epoch,
        best_validation_loss=None if validation_domain is None else best_loss,
        training_objectives=tuple(trace),
        base_action_model_sha256=action_policy.model_state_sha256,
        action_backbone_sha256=final_backbone_sha,
        training_roster_sha256=_json_sha(tuple(row.state_sha256 for row in fit_rows)),
    )
    model_state_sha = _model_hash(model, stop_audit)
    return TrainedObservableStopPolicy(
        model=model,
        normalizer=action_policy.normalizer,
        audit=stop_audit,
        base_action_model_sha256=action_policy.model_state_sha256,
        action_backbone_sha256=final_backbone_sha,
        model_state_sha256=model_state_sha,
    )


def fit_inner_observable_stop_head(
    action_policy: TrainedObservablePolicy,
    examples: tuple[G1StopTrainingExample, ...],
    *,
    validation_domain: str,
    max_epochs: int = REGISTERED_MAX_EPOCHS,
    patience: int = REGISTERED_EARLY_STOPPING_PATIENCE,
    batch_specimens: int = REGISTERED_BATCH_SPECIMENS,
    gradient_clip_norm: float = REGISTERED_GRADIENT_CLIP_NORM,
    seed: int = REGISTERED_INITIALIZATION_SEED,
    device: str = "cpu",
) -> TrainedObservableStopPolicy:
    return _fit_observable_stop_head(
        action_policy,
        examples,
        validation_domain=validation_domain,
        max_epochs=max_epochs,
        patience=patience,
        batch_specimens=batch_specimens,
        gradient_clip_norm=gradient_clip_norm,
        seed=seed,
        device=device,
    )


def fit_final_observable_stop_head(
    action_policy: TrainedObservablePolicy,
    examples: tuple[G1StopTrainingExample, ...],
    *,
    selected_epochs: int,
    batch_specimens: int = REGISTERED_BATCH_SPECIMENS,
    gradient_clip_norm: float = REGISTERED_GRADIENT_CLIP_NORM,
    seed: int = REGISTERED_INITIALIZATION_SEED,
    device: str = "cpu",
) -> TrainedObservableStopPolicy:
    return _fit_observable_stop_head(
        action_policy,
        examples,
        validation_domain=None,
        max_epochs=selected_epochs,
        patience=None,
        batch_specimens=batch_specimens,
        gradient_clip_norm=gradient_clip_norm,
        seed=seed,
        device=device,
    )


__all__ = [
    "G1StopTrainingError",
    "G1StopTrainingExample",
    "StopFitAudit",
    "TrainedObservableStopPolicy",
    "equal_stop_training_weights",
    "fit_final_observable_stop_head",
    "fit_inner_observable_stop_head",
    "join_g1_stop_training_examples",
]
