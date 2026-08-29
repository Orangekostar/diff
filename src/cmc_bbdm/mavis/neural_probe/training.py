"""Deterministic nested-LODO P2 training for the spatial neural probe."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..mechanics_head import (
    FoldNormalizer,
    fit_fold_normalizer,
    fold_normalizer_state_sha256,
)
from ..mris_data import MRISFeatureBank
from ..state_encoder import build_mris_input, summarize_mris_input
from .mechanics import SpatialMRISMechanicsModel
from .state_encoder import SpatialGridMRISStateEncoder


class SpatialMRISTrainingError(ValueError):
    """Raised when spatial P2 training or provenance is invalid."""


_TRAINABLE_MODES = frozenset(("static", "positions_only", "real", "shuffled"))
_CHECKPOINT_TYPE = "mavis_spatial_mris"
_CHECKPOINT_SCHEMA = 1
_ARCHITECTURE = "spatial_grid_cnn_v1"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class SpatialMRISTrainingAudit:
    mode: str
    outer_domain: str
    validation_domains: tuple[str, ...]
    fit_domains: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    validation_specimen_ids: tuple[str, ...]
    epochs_run: int
    selected_epoch: int
    best_validation_mae: float | None
    normalizer_state_sha256: str
    target_data_used_for_selection: bool = False


@dataclass(frozen=True, slots=True)
class FittedSpatialMRISModel:
    mode: str
    outer_domain: str
    architecture_name: str
    model: SpatialMRISMechanicsModel
    normalizer: FoldNormalizer
    audit: SpatialMRISTrainingAudit
    model_state_sha256: str
    state_dict_sha256: str
    base_commit: str
    feature_bank_input_sha256: str
    feature_bank_target_sha256: str
    config_sha256: str
    learning_rate: float
    max_epochs: int
    patience: int | None
    batch_size: int
    seed: int

    @property
    def mris_dimension(self) -> int:
        return self.model.encoder.output_dimension

    def _inspection_arrays(
        self,
        states: tuple[object, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.mode not in {"static", "positions_only", "real"}:
            raise SpatialMRISTrainingError(
                "shuffled inspection encoding requires an explicit donor"
            )
        if type(states) is not tuple or not states:
            raise SpatialMRISTrainingError("inspection state roster is invalid")
        summaries = tuple(
            summarize_mris_input(build_mris_input(state, mode=self.mode))
            for state in states
        )
        return (
            self.normalizer.transform_context(
                np.stack([summary.context_features for summary in summaries])
            ),
            np.stack([summary.token_features for summary in summaries]),
            np.stack([summary.token_mask for summary in summaries]),
            np.stack([summary.cost_features for summary in summaries]),
        )

    def encode_inspection_state(self, state: object, *, device: str) -> np.ndarray:
        return self.encode_inspection_states(
            (state,), batch_size=1, device=device
        )[0]

    def encode_inspection_states(
        self,
        states: tuple[object, ...],
        *,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        contexts, tokens, masks, costs = self._inspection_arrays(states)
        indices = np.arange(len(states), dtype=np.int64)
        output = _infer(
            self.model,
            contexts=contexts,
            tokens=tokens,
            masks=masks,
            costs=costs,
            indices=indices,
            batch_size=batch_size,
            device=device,
            return_embeddings=True,
        )
        return _readonly_f8(output)

    def predict_inspection_state(self, state: object, *, device: str) -> float:
        return float(
            self.predict_inspection_states(
                (state,), batch_size=1, device=device
            )[0]
        )

    def predict_inspection_states(
        self,
        states: tuple[object, ...],
        *,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        contexts, tokens, masks, costs = self._inspection_arrays(states)
        normalized = _infer(
            self.model,
            contexts=contexts,
            tokens=tokens,
            masks=masks,
            costs=costs,
            indices=np.arange(len(states), dtype=np.int64),
            batch_size=batch_size,
            device=device,
            return_embeddings=False,
        )
        return _readonly_f8(self.normalizer.inverse_target(normalized))

    def encode(
        self,
        bank: MRISFeatureBank,
        *,
        state_ids: tuple[str, ...],
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        _validate_inference_request(bank, batch_size=batch_size, device=device)
        if (
            type(state_ids) is not tuple
            or not state_ids
            or len(set(state_ids)) != len(state_ids)
        ):
            raise SpatialMRISTrainingError("spatial encoding roster is invalid")
        index_by_state = {value: index for index, value in enumerate(bank.state_ids)}
        if any(value not in index_by_state for value in state_ids):
            raise SpatialMRISTrainingError("spatial encoding state is unavailable")
        inputs = bank.model_inputs(self.mode, outer_domain=self.outer_domain)
        output = _infer(
            self.model,
            contexts=self.normalizer.transform_context(inputs.context_features),
            tokens=inputs.token_features,
            masks=inputs.token_masks,
            costs=inputs.cost_features,
            indices=np.asarray([index_by_state[value] for value in state_ids]),
            batch_size=batch_size,
            device=device,
            return_embeddings=True,
        )
        return _readonly_f8(output)

    def predict(
        self,
        bank: MRISFeatureBank,
        *,
        domain: str,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        _validate_inference_request(bank, batch_size=batch_size, device=device)
        if domain not in bank.domain_order:
            raise SpatialMRISTrainingError("spatial prediction domain is invalid")
        inputs = bank.model_inputs(self.mode, outer_domain=self.outer_domain)
        indices = np.flatnonzero(np.asarray(bank.domain_ids, dtype=object) == domain)
        normalized = _infer(
            self.model,
            contexts=self.normalizer.transform_context(inputs.context_features),
            tokens=inputs.token_features,
            masks=inputs.token_masks,
            costs=inputs.cost_features,
            indices=indices,
            batch_size=batch_size,
            device=device,
            return_embeddings=False,
        )
        return _readonly_f8(self.normalizer.inverse_target(normalized))


def _readonly_f8(value: object) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    if not np.all(np.isfinite(array)):
        raise SpatialMRISTrainingError("spatial model output is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    output.setflags(write=False)
    return output


def _is_sha(value: object, pattern: re.Pattern[str]) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None


def _validate_inference_request(
    bank: object,
    *,
    batch_size: int,
    device: str,
) -> None:
    if (
        type(bank) is not MRISFeatureBank
        or type(batch_size) is not int
        or batch_size <= 0
        or type(device) is not str
        or not device
    ):
        raise SpatialMRISTrainingError("spatial inference request is invalid")


def _specimen_arrays(
    bank: MRISFeatureBank,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    first_row: dict[str, int] = {}
    for index, specimen_id in enumerate(bank.specimen_ids):
        first_row.setdefault(specimen_id, index)
    specimen_ids = tuple(first_row)
    indices = np.asarray([first_row[value] for value in specimen_ids], dtype=np.int64)
    domains = tuple(bank.domain_ids[index] for index in indices)
    for index, specimen_id in enumerate(bank.specimen_ids):
        reference = first_row[specimen_id]
        if (
            bank.domain_ids[index] != bank.domain_ids[reference]
            or not np.array_equal(
                bank.context_features[index], bank.context_features[reference]
            )
            or bank.targets[index] != bank.targets[reference]
        ):
            raise SpatialMRISTrainingError("per-specimen spatial roster is inconsistent")
    return (
        np.asarray(bank.context_features[indices], dtype=np.float64),
        np.asarray(bank.targets[indices], dtype=np.float64),
        specimen_ids,
        domains,
    )


def _batch_tensors(
    *,
    contexts: np.ndarray,
    tokens: np.ndarray,
    masks: np.ndarray,
    costs: np.ndarray,
    indices: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(contexts[indices], dtype=torch.float32, device=device),
        torch.tensor(tokens[indices], dtype=torch.float32, device=device),
        torch.tensor(masks[indices], dtype=torch.bool, device=device),
        torch.tensor(costs[indices], dtype=torch.float32, device=device),
    )


def _infer(
    model: SpatialMRISMechanicsModel,
    *,
    contexts: np.ndarray,
    tokens: np.ndarray,
    masks: np.ndarray,
    costs: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: str,
    return_embeddings: bool,
) -> np.ndarray:
    if (
        type(batch_size) is not int
        or batch_size <= 0
        or type(device) is not str
        or not device
        or indices.ndim != 1
        or indices.size == 0
    ):
        raise SpatialMRISTrainingError("spatial inference roster is invalid")
    model = model.to(device).eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, indices.size, batch_size):
            batch = indices[start : start + batch_size]
            embedding, prediction = model.forward_batch(
                *_batch_tensors(
                    contexts=contexts,
                    tokens=tokens,
                    masks=masks,
                    costs=costs,
                    indices=batch,
                    device=device,
                )
            )
            selected = embedding if return_embeddings else prediction
            output.append(selected.detach().cpu().numpy().astype(np.float64))
    model.cpu()
    combined = np.concatenate(output)
    expected = (
        (indices.size, model.encoder.output_dimension)
        if return_embeddings
        else (indices.size,)
    )
    if combined.shape != expected or not np.all(np.isfinite(combined)):
        raise SpatialMRISTrainingError("spatial inference output is invalid")
    return combined


def _specimen_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
    specimen_ids: tuple[str, ...],
) -> float:
    if predictions.shape != targets.shape or predictions.shape != (len(specimen_ids),):
        raise SpatialMRISTrainingError("spatial validation rows are misaligned")
    errors: dict[str, list[float]] = {}
    for specimen_id, prediction, target in zip(
        specimen_ids, predictions, targets, strict=True
    ):
        errors.setdefault(specimen_id, []).append(abs(float(prediction - target)))
    value = float(
        np.mean(
            [np.mean(rows, dtype=np.float64) for rows in errors.values()],
            dtype=np.float64,
        )
    )
    if not math.isfinite(value):
        raise SpatialMRISTrainingError("spatial validation MAE is invalid")
    return value


def _new_model(seed: int) -> SpatialMRISMechanicsModel:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return SpatialMRISMechanicsModel(
        SpatialGridMRISStateEncoder(context_dimension=34, output_dimension=64)
    )


def _state_dict_sha256(model: SpatialMRISMechanicsModel) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _model_state_sha256(
    model: SpatialMRISMechanicsModel,
    *,
    mode: str,
    outer_domain: str,
    normalizer: FoldNormalizer,
    selected_epoch: int,
) -> str:
    state_dict_sha256 = _state_dict_sha256(model)
    payload = {
        "architecture_name": _ARCHITECTURE,
        "mode": mode,
        "normalizer_state_sha256": normalizer.state_sha256,
        "outer_domain": outer_domain,
        "schema": 1,
        "selected_epoch": selected_epoch,
        "state_dict_sha256": state_dict_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fit(
    bank: MRISFeatureBank,
    *,
    mode: str,
    outer_domain: str,
    validation_domain: str | None,
    learning_rate: float,
    epochs: int,
    patience: int | None,
    batch_size: int,
    seed: int,
    device: str,
    base_commit: str,
    config_sha256: str,
) -> FittedSpatialMRISModel:
    rate = float(learning_rate)
    if (
        type(bank) is not MRISFeatureBank
        or mode not in _TRAINABLE_MODES
        or outer_domain not in bank.domain_order
        or isinstance(learning_rate, bool)
        or not math.isfinite(rate)
        or rate <= 0.0
        or type(epochs) is not int
        or epochs <= 0
        or type(batch_size) is not int
        or batch_size <= 0
        or type(seed) is not int
        or type(device) is not str
        or not device
        or not _is_sha(base_commit, _HEX40)
        or not _is_sha(config_sha256, _HEX64)
    ):
        raise SpatialMRISTrainingError("spatial training request is invalid")
    if validation_domain is not None and (
        validation_domain not in bank.domain_order
        or validation_domain == outer_domain
        or type(patience) is not int
        or patience <= 0
    ):
        raise SpatialMRISTrainingError("spatial inner validation request is invalid")

    specimen_contexts, specimen_targets, specimen_ids, specimen_domains = (
        _specimen_arrays(bank)
    )
    additional = () if validation_domain is None else (validation_domain,)
    normalizer = fit_fold_normalizer(
        contexts=specimen_contexts,
        targets=specimen_targets,
        specimen_ids=specimen_ids,
        dataset_ids=specimen_domains,
        outer_domain=outer_domain,
        additional_excluded_domains=additional,
    )
    inputs = bank.model_inputs(mode, outer_domain=outer_domain)
    contexts = normalizer.transform_context(inputs.context_features)
    targets = normalizer.transform_target(bank.targets)
    row_domains = np.asarray(bank.domain_ids, dtype=object)
    excluded = (outer_domain, *additional)
    fit_indices = np.flatnonzero(~np.isin(row_domains, excluded))
    validation_indices = (
        np.empty(0, dtype=np.int64)
        if validation_domain is None
        else np.flatnonzero(row_domains == validation_domain)
    )
    if fit_indices.size == 0 or (
        validation_domain is not None and validation_indices.size == 0
    ):
        raise SpatialMRISTrainingError("spatial training roster is empty")

    model = _new_model(seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = math.inf
    selected_epoch = epochs
    epochs_run = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.Generator(np.random.PCG64(seed + epoch)).permutation(
            fit_indices
        )
        for start in range(0, order.size, batch_size):
            batch = order[start : start + batch_size]
            target = torch.tensor(targets[batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            _embedding, prediction = model.forward_batch(
                *_batch_tensors(
                    contexts=contexts,
                    tokens=inputs.token_features,
                    masks=inputs.token_masks,
                    costs=inputs.cost_features,
                    indices=batch,
                    device=device,
                )
            )
            loss = torch.mean(torch.abs(prediction - target))
            if not torch.isfinite(loss):
                raise SpatialMRISTrainingError("spatial training loss is invalid")
            loss.backward()
            optimizer.step()
        epochs_run = epoch
        if validation_domain is None:
            continue
        prediction = normalizer.inverse_target(
            _infer(
                model,
                contexts=contexts,
                tokens=inputs.token_features,
                masks=inputs.token_masks,
                costs=inputs.cost_features,
                indices=validation_indices,
                batch_size=batch_size,
                device=device,
                return_embeddings=False,
            )
        )
        mae = _specimen_mae(
            prediction,
            bank.targets[validation_indices],
            tuple(bank.specimen_ids[index] for index in validation_indices),
        )
        if mae < best_mae - 1.0e-12:
            best_mae = mae
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            selected_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
        model.to(device)
    if validation_domain is not None:
        if best_state is None or not math.isfinite(best_mae):
            raise SpatialMRISTrainingError("spatial early stopping failed")
        model.load_state_dict(best_state)
        best_validation_mae: float | None = best_mae
    else:
        best_validation_mae = None
    model.cpu().eval()
    validation_ids = tuple(
        specimen_id
        for specimen_id, domain in zip(specimen_ids, specimen_domains, strict=True)
        if domain == validation_domain
    )
    audit = SpatialMRISTrainingAudit(
        mode=mode,
        outer_domain=outer_domain,
        validation_domains=additional,
        fit_domains=normalizer.fit_domains,
        fit_specimen_ids=normalizer.fit_specimen_ids,
        validation_specimen_ids=validation_ids,
        epochs_run=epochs_run,
        selected_epoch=selected_epoch,
        best_validation_mae=best_validation_mae,
        normalizer_state_sha256=normalizer.state_sha256,
    )
    return FittedSpatialMRISModel(
        mode=mode,
        outer_domain=outer_domain,
        architecture_name=_ARCHITECTURE,
        model=model,
        normalizer=normalizer,
        audit=audit,
        model_state_sha256=_model_state_sha256(
            model,
            mode=mode,
            outer_domain=outer_domain,
            normalizer=normalizer,
            selected_epoch=selected_epoch,
        ),
        state_dict_sha256=_state_dict_sha256(model),
        base_commit=base_commit,
        feature_bank_input_sha256=bank.input_state_sha256,
        feature_bank_target_sha256=bank.target_state_sha256,
        config_sha256=config_sha256,
        learning_rate=rate,
        max_epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
    )


def fit_inner_spatial_mris_fold(
    bank: MRISFeatureBank,
    *,
    mode: str,
    outer_domain: str,
    validation_domain: str,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
    base_commit: str,
    config_sha256: str,
) -> FittedSpatialMRISModel:
    return _fit(
        bank,
        mode=mode,
        outer_domain=outer_domain,
        validation_domain=validation_domain,
        learning_rate=learning_rate,
        epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        device=device,
        base_commit=base_commit,
        config_sha256=config_sha256,
    )


def fit_final_spatial_mris_model(
    bank: MRISFeatureBank,
    *,
    mode: str,
    outer_domain: str,
    learning_rate: float,
    selected_epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    base_commit: str,
    config_sha256: str,
) -> FittedSpatialMRISModel:
    return _fit(
        bank,
        mode=mode,
        outer_domain=outer_domain,
        validation_domain=None,
        learning_rate=learning_rate,
        epochs=selected_epochs,
        patience=None,
        batch_size=batch_size,
        seed=seed,
        device=device,
        base_commit=base_commit,
        config_sha256=config_sha256,
    )


def _metadata(fitted: FittedSpatialMRISModel) -> dict[str, object]:
    weights = sorted(fitted.model.state_dict())
    return {
        "architecture_name": fitted.architecture_name,
        "audit": {
            "best_validation_mae": fitted.audit.best_validation_mae,
            "epochs_run": fitted.audit.epochs_run,
            "fit_domains": fitted.audit.fit_domains,
            "fit_specimen_ids": fitted.audit.fit_specimen_ids,
            "mode": fitted.audit.mode,
            "normalizer_state_sha256": fitted.audit.normalizer_state_sha256,
            "outer_domain": fitted.audit.outer_domain,
            "selected_epoch": fitted.audit.selected_epoch,
            "target_data_used_for_selection": False,
            "validation_domains": fitted.audit.validation_domains,
            "validation_specimen_ids": fitted.audit.validation_specimen_ids,
        },
        "base_commit": fitted.base_commit,
        "checkpoint_type": _CHECKPOINT_TYPE,
        "config_sha256": fitted.config_sha256,
        "feature_bank_input_sha256": fitted.feature_bank_input_sha256,
        "feature_bank_target_sha256": fitted.feature_bank_target_sha256,
        "hyperparameters": {
            "batch_size": fitted.batch_size,
            "learning_rate": fitted.learning_rate,
            "max_epochs": fitted.max_epochs,
            "patience": fitted.patience,
            "seed": fitted.seed,
        },
        "mode": fitted.mode,
        "model_state_sha256": fitted.model_state_sha256,
        "normalizer": {
            "excluded_domains": fitted.normalizer.excluded_domains,
            "fit_dataset_ids": fitted.normalizer.fit_dataset_ids,
            "fit_domains": fitted.normalizer.fit_domains,
            "fit_specimen_ids": fitted.normalizer.fit_specimen_ids,
            "outer_domain": fitted.normalizer.outer_domain,
            "state_sha256": fitted.normalizer.state_sha256,
            "target_mean": fitted.normalizer.target_mean,
            "target_scale": fitted.normalizer.target_scale,
        },
        "outer_domain": fitted.outer_domain,
        "schema_version": _CHECKPOINT_SCHEMA,
        "state_dict_sha256": fitted.state_dict_sha256,
        "weight_names": weights,
    }


def save_fitted_spatial_mris_checkpoint(
    fitted: FittedSpatialMRISModel,
    path: str | Path,
) -> None:
    if type(fitted) is not FittedSpatialMRISModel:
        raise SpatialMRISTrainingError("issued fitted spatial model is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    weights = sorted(fitted.model.state_dict().items())
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
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
                        _metadata(fitted),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8"),
                    dtype=np.uint8,
                ),
                context_mean=fitted.normalizer.context_mean,
                context_scale=fitted.normalizer.context_scale,
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_spatial_mris_checkpoint_metadata(path: str | Path) -> dict[str, object]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(
                np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes().decode(
                    "utf-8"
                )
            )
    except (OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SpatialMRISTrainingError("spatial checkpoint is invalid") from error
    if type(metadata) is not dict:
        raise SpatialMRISTrainingError("spatial checkpoint metadata is invalid")
    return metadata


def _metadata_tuple(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise SpatialMRISTrainingError(f"checkpoint {label} is invalid")
    return tuple(value)


def _expected_hash(metadata: dict[str, object], key: str, expected: str | None) -> None:
    if expected is not None and metadata.get(key) != expected:
        label = {
            "base_commit": "base commit",
            "config_sha256": "config hash",
            "feature_bank_input_sha256": "feature-bank input hash",
            "feature_bank_target_sha256": "feature-bank target hash",
            "model_state_sha256": "model state hash",
        }[key]
        raise SpatialMRISTrainingError(f"checkpoint {label} changed")


def load_fitted_spatial_mris_checkpoint(
    path: str | Path,
    *,
    expected_model_state_sha256: str | None = None,
    expected_base_commit: str | None = None,
    expected_feature_bank_input_sha256: str | None = None,
    expected_feature_bank_target_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> FittedSpatialMRISModel:
    metadata = read_spatial_mris_checkpoint_metadata(path)
    if (
        metadata.get("checkpoint_type") != _CHECKPOINT_TYPE
        or metadata.get("schema_version") != _CHECKPOINT_SCHEMA
        or metadata.get("architecture_name") != _ARCHITECTURE
    ):
        raise SpatialMRISTrainingError("spatial checkpoint schema changed")
    for key, expected in (
        ("model_state_sha256", expected_model_state_sha256),
        ("base_commit", expected_base_commit),
        ("feature_bank_input_sha256", expected_feature_bank_input_sha256),
        ("feature_bank_target_sha256", expected_feature_bank_target_sha256),
        ("config_sha256", expected_config_sha256),
    ):
        _expected_hash(metadata, key, expected)

    weight_names = _metadata_tuple(metadata.get("weight_names"), "weights")
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            expected_files = {
                "metadata_json",
                "context_mean",
                "context_scale",
                *(f"weight_{index:02d}" for index in range(len(weight_names))),
            }
            if set(archive.files) != expected_files or len(set(weight_names)) != len(
                weight_names
            ):
                raise SpatialMRISTrainingError("spatial checkpoint roster changed")
            context_mean = np.asarray(archive["context_mean"], dtype=np.float64)
            context_scale = np.asarray(archive["context_scale"], dtype=np.float64)
            weight_arrays = tuple(
                np.asarray(archive[f"weight_{index:02d}"])
                for index in range(len(weight_names))
            )
    except SpatialMRISTrainingError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise SpatialMRISTrainingError("spatial checkpoint arrays are invalid") from error

    mode = metadata.get("mode")
    outer_domain = metadata.get("outer_domain")
    normalizer_raw = metadata.get("normalizer")
    audit_raw = metadata.get("audit")
    hyperparameters = metadata.get("hyperparameters")
    if (
        mode not in _TRAINABLE_MODES
        or type(outer_domain) is not str
        or not outer_domain
        or type(normalizer_raw) is not dict
        or type(audit_raw) is not dict
        or type(hyperparameters) is not dict
        or context_mean.shape != (34,)
        or context_scale.shape != (34,)
        or not np.all(np.isfinite(context_mean))
        or not np.all(np.isfinite(context_scale))
        or np.any(context_scale <= 0.0)
        or not _is_sha(metadata.get("base_commit"), _HEX40)
        or not _is_sha(metadata.get("config_sha256"), _HEX64)
        or not _is_sha(metadata.get("feature_bank_input_sha256"), _HEX64)
        or not _is_sha(metadata.get("feature_bank_target_sha256"), _HEX64)
        or not _is_sha(metadata.get("model_state_sha256"), _HEX64)
        or not _is_sha(metadata.get("state_dict_sha256"), _HEX64)
    ):
        raise SpatialMRISTrainingError("spatial checkpoint metadata is invalid")

    excluded = _metadata_tuple(normalizer_raw.get("excluded_domains"), "excluded domains")
    fit_specimen_ids = _metadata_tuple(
        normalizer_raw.get("fit_specimen_ids"), "fit specimen IDs"
    )
    fit_dataset_ids = _metadata_tuple(
        normalizer_raw.get("fit_dataset_ids"), "fit dataset IDs"
    )
    fit_domains = _metadata_tuple(normalizer_raw.get("fit_domains"), "fit domains")
    target_mean = float(normalizer_raw.get("target_mean"))
    target_scale = float(normalizer_raw.get("target_scale"))
    frozen_mean = _readonly_f8(context_mean)
    frozen_scale = _readonly_f8(context_scale)
    normalizer_state = fold_normalizer_state_sha256(
        outer_domain=outer_domain,
        excluded_domains=excluded,
        context_mean=frozen_mean,
        context_scale=frozen_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        fit_specimen_ids=fit_specimen_ids,
        fit_dataset_ids=fit_dataset_ids,
    )
    if (
        normalizer_raw.get("outer_domain") != outer_domain
        or normalizer_raw.get("state_sha256") != normalizer_state
        or not fit_specimen_ids
        or len(fit_specimen_ids) != len(fit_dataset_ids)
        or not math.isfinite(target_mean)
        or not math.isfinite(target_scale)
        or target_scale <= 0.0
    ):
        raise SpatialMRISTrainingError("spatial checkpoint normalizer changed")
    normalizer = FoldNormalizer(
        outer_domain=outer_domain,
        excluded_domains=excluded,
        context_mean=frozen_mean,
        context_scale=frozen_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        fit_specimen_ids=fit_specimen_ids,
        fit_dataset_ids=fit_dataset_ids,
        fit_domains=fit_domains,
        state_sha256=normalizer_state,
    )

    model = _new_model(0)
    expected_state = model.state_dict()
    if tuple(sorted(expected_state)) != weight_names:
        raise SpatialMRISTrainingError("spatial checkpoint model graph changed")
    state = {
        name: torch.tensor(value.copy(), dtype=expected_state[name].dtype)
        for name, value in zip(weight_names, weight_arrays, strict=True)
    }
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise SpatialMRISTrainingError("spatial checkpoint weights changed") from error

    validation_domains = _metadata_tuple(
        audit_raw.get("validation_domains"), "validation domains"
    )
    audit_fit_domains = _metadata_tuple(audit_raw.get("fit_domains"), "audit domains")
    audit_fit_ids = _metadata_tuple(
        audit_raw.get("fit_specimen_ids"), "audit fit specimen IDs"
    )
    validation_ids = _metadata_tuple(
        audit_raw.get("validation_specimen_ids"), "validation specimen IDs"
    )
    epochs_run = audit_raw.get("epochs_run")
    selected_epoch = audit_raw.get("selected_epoch")
    best_raw = audit_raw.get("best_validation_mae")
    best_mae = None if best_raw is None else float(best_raw)
    if (
        audit_raw.get("mode") != mode
        or audit_raw.get("outer_domain") != outer_domain
        or audit_raw.get("normalizer_state_sha256") != normalizer_state
        or audit_raw.get("target_data_used_for_selection") is not False
        or audit_fit_domains != fit_domains
        or audit_fit_ids != fit_specimen_ids
        or type(epochs_run) is not int
        or type(selected_epoch) is not int
        or not 0 < selected_epoch <= epochs_run
        or (best_mae is not None and (not math.isfinite(best_mae) or best_mae < 0.0))
    ):
        raise SpatialMRISTrainingError("spatial checkpoint audit changed")
    audit = SpatialMRISTrainingAudit(
        mode=mode,
        outer_domain=outer_domain,
        validation_domains=validation_domains,
        fit_domains=audit_fit_domains,
        fit_specimen_ids=audit_fit_ids,
        validation_specimen_ids=validation_ids,
        epochs_run=epochs_run,
        selected_epoch=selected_epoch,
        best_validation_mae=best_mae,
        normalizer_state_sha256=normalizer_state,
    )
    observed_state_dict = _state_dict_sha256(model)
    observed_model_state = _model_state_sha256(
        model,
        mode=mode,
        outer_domain=outer_domain,
        normalizer=normalizer,
        selected_epoch=selected_epoch,
    )
    if (
        observed_state_dict != metadata.get("state_dict_sha256")
        or observed_model_state != metadata.get("model_state_sha256")
    ):
        raise SpatialMRISTrainingError("spatial checkpoint model state hash changed")

    learning_rate = float(hyperparameters.get("learning_rate"))
    max_epochs = hyperparameters.get("max_epochs")
    patience = hyperparameters.get("patience")
    batch_size = hyperparameters.get("batch_size")
    seed = hyperparameters.get("seed")
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0.0
        or type(max_epochs) is not int
        or max_epochs <= 0
        or (patience is not None and (type(patience) is not int or patience <= 0))
        or type(batch_size) is not int
        or batch_size <= 0
        or type(seed) is not int
    ):
        raise SpatialMRISTrainingError("spatial checkpoint hyperparameters changed")
    return FittedSpatialMRISModel(
        mode=mode,
        outer_domain=outer_domain,
        architecture_name=_ARCHITECTURE,
        model=model.eval(),
        normalizer=normalizer,
        audit=audit,
        model_state_sha256=observed_model_state,
        state_dict_sha256=observed_state_dict,
        base_commit=metadata["base_commit"],
        feature_bank_input_sha256=metadata["feature_bank_input_sha256"],
        feature_bank_target_sha256=metadata["feature_bank_target_sha256"],
        config_sha256=metadata["config_sha256"],
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
    )


__all__ = [
    "FittedSpatialMRISModel",
    "SpatialMRISTrainingAudit",
    "SpatialMRISTrainingError",
    "fit_final_spatial_mris_model",
    "fit_inner_spatial_mris_fold",
    "load_fitted_spatial_mris_checkpoint",
    "read_spatial_mris_checkpoint_metadata",
    "save_fitted_spatial_mris_checkpoint",
]
