"""Deterministic source-only training for dynamic mechanical VoI."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dynamic_data import DynamicStateGroup
from .dynamic_voi import ActionScoreBatch, CandidateDescriptor, DynamicActionScorer
from .losses import grouped_dynamic_voi_loss


class MAVISDynamicTrainingError(ValueError):
    """Raised when P3 training violates its source-only fold contract."""


@dataclass(frozen=True, slots=True)
class DynamicTrainingAudit:
    outer_domain: str
    validation_domain: str | None
    fit_domains: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    validation_specimen_ids: tuple[str, ...]
    epochs_run: int
    selected_epoch: int
    best_validation_regret: float | None


class DynamicVoITrainingModel(nn.Module):
    def __init__(self, *, mris_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.scorer = DynamicActionScorer(
            mris_dimension=mris_dimension,
            hidden_dimension=hidden_dimension,
        )
        self.cai_head = nn.Linear(mris_dimension, 1)

    def forward_grouped(
        self,
        mris: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_group_indices: torch.Tensor,
    ) -> tuple[ActionScoreBatch, torch.Tensor]:
        scores = self.scorer.forward_grouped(
            mris,
            candidate_features,
            candidate_group_indices,
        )
        cai = self.cai_head(mris).squeeze(-1)
        if cai.shape != (mris.shape[0],) or not torch.isfinite(cai).all():
            raise MAVISDynamicTrainingError("dynamic CAI auxiliary output is invalid")
        return scores, cai


@dataclass(frozen=True, slots=True)
class FittedDynamicVoI:
    outer_domain: str
    mris_dimension: int
    hidden_dimension: int
    model: DynamicVoITrainingModel
    audit: DynamicTrainingAudit
    loss_weights: tuple[tuple[str, float], ...]
    model_state_sha256: str

    def score_actions(
        self,
        mris: torch.Tensor,
        candidates: tuple[CandidateDescriptor, ...],
    ) -> ActionScoreBatch:
        return self.model.scorer.score_actions(mris, candidates)


def _loss_weights(value: object) -> dict[str, float]:
    if type(value) is not dict or set(value) != {"cai", "pair", "list", "value"}:
        raise MAVISDynamicTrainingError("dynamic loss-weight schema changed")
    result = {key: float(item) for key, item in value.items()}
    if any(not math.isfinite(item) or item < 0.0 for item in result.values()):
        raise MAVISDynamicTrainingError("dynamic loss weights are invalid")
    return result


def _validate(
    groups: object,
    embeddings: object,
    *,
    validation_domain: str | None,
    hidden_dimension: int,
    learning_rate: float,
    epochs: int,
    patience: int | None,
    batch_size: int,
    seed: int,
    device: str,
    loss_weights: object,
) -> tuple[tuple[DynamicStateGroup, ...], np.ndarray, dict[str, float]]:
    if (
        type(groups) is not tuple
        or not groups
        or any(type(group) is not DynamicStateGroup for group in groups)
        or type(hidden_dimension) is not int
        or hidden_dimension <= 0
        or isinstance(learning_rate, bool)
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
        or type(epochs) is not int
        or epochs <= 0
        or type(batch_size) is not int
        or batch_size <= 0
        or type(seed) is not int
        or type(device) is not str
        or not device
    ):
        raise MAVISDynamicTrainingError("dynamic training request is invalid")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(groups)
        or matrix.shape[1] == 0
        or not np.all(np.isfinite(matrix))
    ):
        raise MAVISDynamicTrainingError("dynamic MRIS matrix is invalid")
    outer_domains = {group.outer_domain for group in groups}
    domains = {group.domain_id for group in groups}
    if (
        len(outer_domains) != 1
        or next(iter(outer_domains)) in domains
        or (validation_domain is not None and validation_domain not in domains)
        or (
            validation_domain is not None
            and (type(patience) is not int or patience <= 0)
        )
        or (validation_domain is None and patience is not None)
    ):
        raise MAVISDynamicTrainingError("dynamic fold roster is invalid")
    return groups, matrix, _loss_weights(loss_weights)


def _batch_arrays(
    groups: tuple[DynamicStateGroup, ...],
    embeddings: np.ndarray,
    candidate_features: tuple[np.ndarray, ...],
    teacher_values: tuple[np.ndarray, ...],
    indices: np.ndarray,
    *,
    device: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    candidate_arrays: list[np.ndarray] = []
    teacher_arrays: list[np.ndarray] = []
    group_indices: list[int] = []
    targets: list[float] = []
    for local_index, index in enumerate(indices):
        group = groups[int(index)]
        features = candidate_features[int(index)]
        candidate_arrays.append(features)
        teacher_arrays.append(teacher_values[int(index)])
        count = features.shape[0]
        group_indices.extend([local_index] * count)
        targets.append(group.true_cai)
    return (
        torch.tensor(embeddings[indices], dtype=torch.float32, device=device),
        torch.tensor(np.concatenate(candidate_arrays), dtype=torch.float32, device=device),
        torch.tensor(group_indices, dtype=torch.long, device=device),
        torch.tensor(np.concatenate(teacher_arrays), dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
    )


def _validation_regret(
    model: DynamicVoITrainingModel,
    groups: tuple[DynamicStateGroup, ...],
    embeddings: np.ndarray,
    candidate_features: tuple[np.ndarray, ...],
    teacher_values: tuple[np.ndarray, ...],
    indices: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> float:
    regrets: dict[str, list[float]] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, indices.size, batch_size):
            batch = indices[start : start + batch_size]
            mris, features, candidate_groups, _teachers, _targets = _batch_arrays(
                groups,
                embeddings,
                candidate_features,
                teacher_values,
                batch,
                device=device,
            )
            result = model.scorer.forward_grouped(mris, features, candidate_groups)
            scores = result.scores.detach().cpu().numpy()
            local_groups = candidate_groups.detach().cpu().numpy()
            for local_index, index in enumerate(batch):
                group = groups[int(index)]
                local_scores = scores[local_groups == local_index]
                selected = int(np.argmax(local_scores))
                regret = float(
                    np.max(group.teacher_values) - group.teacher_values[selected]
                )
                regrets.setdefault(group.specimen_id, []).append(regret)
    value = float(
        np.mean(
            [np.mean(items, dtype=np.float64) for items in regrets.values()],
            dtype=np.float64,
        )
    )
    if not math.isfinite(value) or value < -1.0e-12:
        raise MAVISDynamicTrainingError("dynamic validation regret is invalid")
    return max(value, 0.0)


def _model_hash(
    model: DynamicVoITrainingModel,
    *,
    outer_domain: str,
    selected_epoch: int,
    loss_weights: dict[str, float],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema": 1,
                "outer_domain": outer_domain,
                "selected_epoch": selected_epoch,
                "loss_weights": loss_weights,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fit(
    groups: tuple[DynamicStateGroup, ...],
    embeddings: np.ndarray,
    *,
    validation_domain: str | None,
    hidden_dimension: int,
    learning_rate: float,
    epochs: int,
    patience: int | None,
    batch_size: int,
    seed: int,
    device: str,
    loss_weights: dict[str, float],
) -> FittedDynamicVoI:
    groups, embeddings, weights = _validate(
        groups,
        embeddings,
        validation_domain=validation_domain,
        hidden_dimension=hidden_dimension,
        learning_rate=learning_rate,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        device=device,
        loss_weights=loss_weights,
    )
    domains = np.asarray([group.domain_id for group in groups], dtype=object)
    fit_indices = np.flatnonzero(domains != validation_domain)
    validation_indices = (
        np.empty(0, dtype=np.int64)
        if validation_domain is None
        else np.flatnonzero(domains == validation_domain)
    )
    if fit_indices.size == 0 or (validation_domain is not None and not validation_indices.size):
        raise MAVISDynamicTrainingError("dynamic fit or validation roster is empty")
    torch.manual_seed(seed)
    model = DynamicVoITrainingModel(
        mris_dimension=embeddings.shape[1],
        hidden_dimension=hidden_dimension,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    candidate_features = tuple(
        np.ascontiguousarray(
            np.stack([candidate.features() for candidate in group.candidates]),
            dtype=np.float32,
        )
        for group in groups
    )
    teacher_values = tuple(
        np.ascontiguousarray(group.teacher_values, dtype=np.float32) for group in groups
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_regret = math.inf
    selected_epoch = epochs
    stale = 0
    epochs_run = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.Generator(np.random.PCG64(seed + epoch)).permutation(
            fit_indices
        )
        for start in range(0, order.size, batch_size):
            batch = order[start : start + batch_size]
            mris, features, candidate_groups, teachers, targets = _batch_arrays(
                groups,
                embeddings,
                candidate_features,
                teacher_values,
                batch,
                device=device,
            )
            output, cai_predictions = model.forward_grouped(
                mris,
                features,
                candidate_groups,
            )
            total = grouped_dynamic_voi_loss(
                cai_predictions=cai_predictions,
                cai_targets=targets,
                action_scores=output.scores,
                value_predictions=output.value_predictions,
                teacher_values=teachers,
                candidate_group_indices=candidate_groups,
                cai_weight=weights["cai"],
                pair_weight=weights["pair"],
                list_weight=weights["list"],
                value_weight=weights["value"],
            ).total
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
        epochs_run = epoch
        if validation_domain is None:
            continue
        regret = _validation_regret(
            model,
            groups,
            embeddings,
            candidate_features,
            teacher_values,
            validation_indices,
            batch_size=batch_size,
            device=device,
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
            raise MAVISDynamicTrainingError("dynamic early stopping selected no model")
        model.load_state_dict(best_state, strict=True)
    model.cpu().eval()
    fit_domains = tuple(sorted(set(domains[fit_indices].tolist())))
    fit_ids = tuple(sorted({groups[int(index)].specimen_id for index in fit_indices}))
    validation_ids = tuple(
        sorted({groups[int(index)].specimen_id for index in validation_indices})
    )
    outer_domain = groups[0].outer_domain
    audit = DynamicTrainingAudit(
        outer_domain=outer_domain,
        validation_domain=validation_domain,
        fit_domains=fit_domains,
        fit_specimen_ids=fit_ids,
        validation_specimen_ids=validation_ids,
        epochs_run=epochs_run,
        selected_epoch=selected_epoch,
        best_validation_regret=(
            None if validation_domain is None else float(best_regret)
        ),
    )
    return FittedDynamicVoI(
        outer_domain=outer_domain,
        mris_dimension=embeddings.shape[1],
        hidden_dimension=hidden_dimension,
        model=model,
        audit=audit,
        loss_weights=tuple(sorted(weights.items())),
        model_state_sha256=_model_hash(
            model,
            outer_domain=outer_domain,
            selected_epoch=selected_epoch,
            loss_weights=weights,
        ),
    )


def fit_inner_dynamic_voi(
    groups: tuple[DynamicStateGroup, ...],
    embeddings: np.ndarray,
    *,
    validation_domain: str,
    hidden_dimension: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
    loss_weights: dict[str, float],
) -> FittedDynamicVoI:
    return _fit(
        groups,
        embeddings,
        validation_domain=validation_domain,
        hidden_dimension=hidden_dimension,
        learning_rate=learning_rate,
        epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        device=device,
        loss_weights=loss_weights,
    )


def fit_final_dynamic_voi(
    groups: tuple[DynamicStateGroup, ...],
    embeddings: np.ndarray,
    *,
    hidden_dimension: int,
    learning_rate: float,
    selected_epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    loss_weights: dict[str, float],
) -> FittedDynamicVoI:
    return _fit(
        groups,
        embeddings,
        validation_domain=None,
        hidden_dimension=hidden_dimension,
        learning_rate=learning_rate,
        epochs=selected_epochs,
        patience=None,
        batch_size=batch_size,
        seed=seed,
        device=device,
        loss_weights=loss_weights,
    )


def save_fitted_dynamic_checkpoint(
    fitted: FittedDynamicVoI,
    path: str | Path,
) -> None:
    if type(fitted) is not FittedDynamicVoI:
        raise MAVISDynamicTrainingError("issued dynamic model is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    weights = sorted(fitted.model.state_dict().items())
    metadata = {
        "schema_version": 1,
        "outer_domain": fitted.outer_domain,
        "mris_dimension": fitted.mris_dimension,
        "hidden_dimension": fitted.hidden_dimension,
        "model_state_sha256": fitted.model_state_sha256,
        "loss_weights": dict(fitted.loss_weights),
        "weight_names": [name for name, _value in weights],
        "audit": {
            "outer_domain": fitted.audit.outer_domain,
            "validation_domain": fitted.audit.validation_domain,
            "fit_domains": fitted.audit.fit_domains,
            "fit_specimen_ids": fitted.audit.fit_specimen_ids,
            "validation_specimen_ids": fitted.audit.validation_specimen_ids,
            "epochs_run": fitted.audit.epochs_run,
            "selected_epoch": fitted.audit.selected_epoch,
            "best_validation_regret": fitted.audit.best_validation_regret,
        },
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            arrays = {
                f"weight_{index:02d}": value.detach().cpu().numpy()
                for index, (_name, value) in enumerate(weights)
            }
            np.savez_compressed(
                handle,
                metadata_json=np.frombuffer(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8"),
                    dtype=np.uint8,
                ),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise MAVISDynamicTrainingError(f"dynamic checkpoint {label} is invalid")
    return tuple(value)


def load_fitted_dynamic_checkpoint(
    path: str | Path,
    *,
    expected_model_state_sha256: str | None = None,
) -> FittedDynamicVoI:
    if expected_model_state_sha256 is not None and (
        type(expected_model_state_sha256) is not str
        or len(expected_model_state_sha256) != 64
    ):
        raise MAVISDynamicTrainingError("expected dynamic model hash is invalid")
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(
                np.asarray(archive["metadata_json"], dtype=np.uint8)
                .tobytes()
                .decode("utf-8")
            )
            if type(metadata) is not dict or metadata.get("schema_version") != 1:
                raise MAVISDynamicTrainingError(
                    "dynamic checkpoint metadata changed"
                )
            weight_names = _text_tuple(metadata.get("weight_names"), "weights")
            expected_files = {
                "metadata_json",
                *(f"weight_{index:02d}" for index in range(len(weight_names))),
            }
            if set(archive.files) != expected_files:
                raise MAVISDynamicTrainingError(
                    "dynamic checkpoint weight roster changed"
                )
            weight_arrays = tuple(
                np.asarray(archive[f"weight_{index:02d}"])
                for index in range(len(weight_names))
            )
    except MAVISDynamicTrainingError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise MAVISDynamicTrainingError("dynamic checkpoint is invalid") from error
    outer_domain = metadata.get("outer_domain")
    mris_dimension = metadata.get("mris_dimension")
    hidden_dimension = metadata.get("hidden_dimension")
    stored_hash = metadata.get("model_state_sha256")
    audit_raw = metadata.get("audit")
    if (
        type(outer_domain) is not str
        or not outer_domain
        or type(mris_dimension) is not int
        or type(hidden_dimension) is not int
        or min(mris_dimension, hidden_dimension) <= 0
        or type(stored_hash) is not str
        or len(stored_hash) != 64
        or type(audit_raw) is not dict
    ):
        raise MAVISDynamicTrainingError("dynamic checkpoint metadata is invalid")
    weights = _loss_weights(metadata.get("loss_weights"))
    fit_domains = _text_tuple(audit_raw.get("fit_domains"), "fit domains")
    fit_ids = _text_tuple(audit_raw.get("fit_specimen_ids"), "fit specimen IDs")
    validation_ids = _text_tuple(
        audit_raw.get("validation_specimen_ids"),
        "validation specimen IDs",
    )
    validation_domain = audit_raw.get("validation_domain")
    epochs_run = audit_raw.get("epochs_run")
    selected_epoch = audit_raw.get("selected_epoch")
    regret_raw = audit_raw.get("best_validation_regret")
    regret = None if regret_raw is None else float(regret_raw)
    if (
        audit_raw.get("outer_domain") != outer_domain
        or (validation_domain is not None and (type(validation_domain) is not str or not validation_domain))
        or type(epochs_run) is not int
        or type(selected_epoch) is not int
        or not 0 < selected_epoch <= epochs_run
        or (regret is not None and (not math.isfinite(regret) or regret < 0.0))
        or (validation_domain is None) != (regret is None)
    ):
        raise MAVISDynamicTrainingError("dynamic checkpoint audit is invalid")
    model = DynamicVoITrainingModel(
        mris_dimension=mris_dimension,
        hidden_dimension=hidden_dimension,
    )
    expected_state = model.state_dict()
    if tuple(sorted(expected_state)) != weight_names:
        raise MAVISDynamicTrainingError("dynamic checkpoint model graph changed")
    try:
        model.load_state_dict(
            {
                name: torch.tensor(array.copy(), dtype=expected_state[name].dtype)
                for name, array in zip(weight_names, weight_arrays, strict=True)
            },
            strict=True,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise MAVISDynamicTrainingError("dynamic checkpoint weights changed") from error
    observed_hash = _model_hash(
        model,
        outer_domain=outer_domain,
        selected_epoch=selected_epoch,
        loss_weights=weights,
    )
    if observed_hash != stored_hash or (
        expected_model_state_sha256 is not None
        and observed_hash != expected_model_state_sha256
    ):
        raise MAVISDynamicTrainingError("dynamic checkpoint model hash changed")
    audit = DynamicTrainingAudit(
        outer_domain=outer_domain,
        validation_domain=validation_domain,
        fit_domains=fit_domains,
        fit_specimen_ids=fit_ids,
        validation_specimen_ids=validation_ids,
        epochs_run=epochs_run,
        selected_epoch=selected_epoch,
        best_validation_regret=regret,
    )
    return FittedDynamicVoI(
        outer_domain=outer_domain,
        mris_dimension=mris_dimension,
        hidden_dimension=hidden_dimension,
        model=model.eval(),
        audit=audit,
        loss_weights=tuple(sorted(weights.items())),
        model_state_sha256=observed_hash,
    )


__all__ = [
    "DynamicTrainingAudit",
    "DynamicVoITrainingModel",
    "FittedDynamicVoI",
    "MAVISDynamicTrainingError",
    "fit_final_dynamic_voi",
    "fit_inner_dynamic_voi",
    "load_fitted_dynamic_checkpoint",
    "save_fitted_dynamic_checkpoint",
]
