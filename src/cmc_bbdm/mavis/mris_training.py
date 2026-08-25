"""Deterministic nested-LODO training for the MAVIS mechanics head."""

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

from .mechanics_head import (
    FoldNormalizer,
    MRISMechanicsModel,
    fit_fold_normalizer,
    fold_normalizer_state_sha256,
)
from .mris_data import MRISFeatureBank
from .state_encoder import (
    MRISStateEncoder,
    build_mris_input,
    summarize_mris_input,
)


class MAVISMRISTrainingError(ValueError):
    """Raised when nested MRIS training violates its roster or numeric contract."""


_TRAINABLE_MODES = frozenset(("static", "positions_only", "real", "shuffled"))


@dataclass(frozen=True, slots=True)
class MRISTrainingAudit:
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


@dataclass(frozen=True, slots=True)
class FittedMRISModel:
    mode: str
    outer_domain: str
    hidden_dimension: int
    mris_dimension: int
    model: MRISMechanicsModel
    normalizer: FoldNormalizer
    audit: MRISTrainingAudit
    model_state_sha256: str

    def encode_inspection_state(
        self,
        state: object,
        *,
        device: str,
    ) -> np.ndarray:
        return self.encode_inspection_states(
            (state,),
            batch_size=1,
            device=device,
        )[0]

    def encode_inspection_states(
        self,
        states: tuple[object, ...],
        *,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        if self.mode not in {"static", "positions_only", "real"}:
            raise MAVISMRISTrainingError(
                "shuffled state encoding requires an explicit donor"
            )
        if (
            type(states) is not tuple
            or not states
            or type(batch_size) is not int
            or batch_size <= 0
            or type(device) is not str
            or not device
        ):
            raise MAVISMRISTrainingError("MRIS inspection encoding request is invalid")
        summaries = tuple(
            summarize_mris_input(build_mris_input(state, mode=self.mode))
            for state in states
        )
        normalized = self.normalizer.transform_context(
            np.stack([summary.context_features for summary in summaries])
        )
        tokens = np.stack([summary.token_features for summary in summaries])
        masks = np.stack([summary.token_mask for summary in summaries])
        costs = np.stack([summary.cost_features for summary in summaries])
        model = self.model.to(device).eval()
        parameter = next(model.parameters())
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(states), batch_size):
                stop = min(start + batch_size, len(states))
                encoded = model.encoder.forward_batch(
                    torch.tensor(
                        normalized[start:stop],
                        dtype=parameter.dtype,
                        device=device,
                    ),
                    torch.tensor(
                        tokens[start:stop],
                        dtype=parameter.dtype,
                        device=device,
                    ),
                    torch.tensor(
                        masks[start:stop],
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.tensor(
                        costs[start:stop],
                        dtype=parameter.dtype,
                        device=device,
                    ),
                )
                outputs.append(encoded.detach().cpu().numpy().astype(np.float64))
        model.cpu()
        output = np.ascontiguousarray(np.concatenate(outputs), dtype="<f8")
        if output.shape != (len(states), self.mris_dimension) or not np.all(
            np.isfinite(output)
        ):
            raise MAVISMRISTrainingError("MRIS inspection-state encoding is invalid")
        frozen = np.frombuffer(output.tobytes(order="C"), dtype="<f8").reshape(
            output.shape
        )
        frozen.setflags(write=False)
        return frozen

    def predict_inspection_state(
        self,
        state: object,
        *,
        device: str,
    ) -> float:
        return float(
            self.predict_inspection_states(
                (state,),
                batch_size=1,
                device=device,
            )[0]
        )

    def predict_inspection_states(
        self,
        states: tuple[object, ...],
        *,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        if (
            self.mode not in {"static", "positions_only", "real"}
            or type(states) is not tuple
            or not states
            or type(batch_size) is not int
            or batch_size <= 0
            or type(device) is not str
            or not device
        ):
            raise MAVISMRISTrainingError("MRIS inspection curve request is invalid")
        summaries = tuple(
            summarize_mris_input(build_mris_input(state, mode=self.mode))
            for state in states
        )
        contexts = self.normalizer.transform_context(
            np.stack([summary.context_features for summary in summaries])
        )
        tokens = np.stack([summary.token_features for summary in summaries])
        masks = np.stack([summary.token_mask for summary in summaries])
        costs = np.stack([summary.cost_features for summary in summaries])
        model = self.model.to(device).eval()
        parameter = next(model.parameters())
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(states), batch_size):
                stop = min(start + batch_size, len(states))
                _embedding, normalized = model.forward_batch(
                    torch.tensor(
                        contexts[start:stop],
                        dtype=parameter.dtype,
                        device=device,
                    ),
                    torch.tensor(
                        tokens[start:stop],
                        dtype=parameter.dtype,
                        device=device,
                    ),
                    torch.tensor(
                        masks[start:stop],
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.tensor(
                        costs[start:stop],
                        dtype=parameter.dtype,
                        device=device,
                    ),
                )
                outputs.append(normalized.detach().cpu().numpy().astype(np.float64))
        model.cpu()
        predictions = np.ascontiguousarray(
            self.normalizer.inverse_target(np.concatenate(outputs)),
            dtype="<f8",
        )
        if predictions.shape != (len(states),) or not np.all(np.isfinite(predictions)):
            raise MAVISMRISTrainingError("MRIS inspection curve prediction is invalid")
        result = np.frombuffer(predictions.tobytes(order="C"), dtype="<f8")
        result.setflags(write=False)
        return result

    def encode(
        self,
        bank: MRISFeatureBank,
        *,
        state_ids: tuple[str, ...],
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        if (
            type(bank) is not MRISFeatureBank
            or type(state_ids) is not tuple
            or not state_ids
            or len(set(state_ids)) != len(state_ids)
            or any(type(value) is not str or not value for value in state_ids)
            or type(batch_size) is not int
            or batch_size <= 0
            or type(device) is not str
            or not device
        ):
            raise MAVISMRISTrainingError("MRIS encoding request is invalid")
        index_by_state = {value: index for index, value in enumerate(bank.state_ids)}
        if any(value not in index_by_state for value in state_ids):
            raise MAVISMRISTrainingError("MRIS encoding state roster is unavailable")
        indices = np.asarray([index_by_state[value] for value in state_ids], dtype=np.int64)
        inputs = bank.model_inputs(self.mode, outer_domain=self.outer_domain)
        contexts = self.normalizer.transform_context(inputs.context_features)
        model = self.model.to(device).eval()
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, indices.size, batch_size):
                batch = indices[start : start + batch_size]
                tensors = _batch_tensors(
                    contexts=contexts,
                    tokens=inputs.token_features,
                    masks=inputs.token_masks,
                    costs=inputs.cost_features,
                    indices=batch,
                    device=device,
                )
                encoded = model.encoder.forward_batch(*tensors)
                outputs.append(encoded.detach().cpu().numpy().astype(np.float64))
        model.cpu()
        combined = np.ascontiguousarray(np.concatenate(outputs), dtype="<f8")
        if combined.shape != (len(state_ids), self.mris_dimension) or not np.all(
            np.isfinite(combined)
        ):
            raise MAVISMRISTrainingError("MRIS encoded state matrix is invalid")
        frozen = np.frombuffer(combined.tobytes(order="C"), dtype="<f8").reshape(
            combined.shape
        )
        frozen.setflags(write=False)
        return frozen

    def predict(
        self,
        bank: MRISFeatureBank,
        *,
        domain: str,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        if (
            type(bank) is not MRISFeatureBank
            or domain not in bank.domain_order
            or type(batch_size) is not int
            or batch_size <= 0
            or type(device) is not str
            or not device
        ):
            raise MAVISMRISTrainingError("MRIS prediction request is invalid")
        inputs = bank.model_inputs(self.mode, outer_domain=self.outer_domain)
        indices = np.flatnonzero(np.asarray(bank.domain_ids, dtype=object) == domain)
        normalized_context = self.normalizer.transform_context(inputs.context_features)
        model = self.model.to(device).eval()
        predictions = _predict_normalized(
            model,
            contexts=normalized_context,
            tokens=inputs.token_features,
            masks=inputs.token_masks,
            costs=inputs.cost_features,
            indices=indices,
            batch_size=batch_size,
            device=device,
        )
        model.cpu()
        return self.normalizer.inverse_target(predictions)


def _specimen_arrays(
    bank: MRISFeatureBank,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    first_row: dict[str, int] = {}
    for index, specimen_id in enumerate(bank.specimen_ids):
        first_row.setdefault(specimen_id, index)
    specimen_ids = tuple(first_row)
    indices = np.asarray([first_row[value] for value in specimen_ids], dtype=np.int64)
    domains = tuple(bank.domain_ids[index] for index in indices)
    contexts = np.asarray(bank.context_features[indices], dtype=np.float64)
    targets = np.asarray(bank.targets[indices], dtype=np.float64)
    for row_index, specimen_id in enumerate(bank.specimen_ids):
        reference = first_row[specimen_id]
        if (
            bank.domain_ids[row_index] != bank.domain_ids[reference]
            or not np.array_equal(
                bank.context_features[row_index], bank.context_features[reference]
            )
            or bank.targets[row_index] != bank.targets[reference]
        ):
            raise MAVISMRISTrainingError("per-specimen P2 roster is inconsistent")
    return contexts, targets, specimen_ids, domains


def _validate_hyperparameters(
    *,
    bank: MRISFeatureBank,
    mode: str,
    outer_domain: str,
    hidden_dimension: int,
    mris_dimension: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> float:
    rate = float(learning_rate)
    if (
        type(bank) is not MRISFeatureBank
        or mode not in _TRAINABLE_MODES
        or outer_domain not in bank.domain_order
        or type(hidden_dimension) is not int
        or type(mris_dimension) is not int
        or min(hidden_dimension, mris_dimension) <= 0
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
    ):
        raise MAVISMRISTrainingError("MRIS training request is invalid")
    return rate


def _model(
    *, hidden_dimension: int, mris_dimension: int, seed: int
) -> MRISMechanicsModel:
    torch.manual_seed(seed)
    return MRISMechanicsModel(
        MRISStateEncoder(
            context_dimension=34,
            hidden_dimension=hidden_dimension,
            output_dimension=mris_dimension,
        )
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


def _predict_normalized(
    model: MRISMechanicsModel,
    *,
    contexts: np.ndarray,
    tokens: np.ndarray,
    masks: np.ndarray,
    costs: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, indices.size, batch_size):
            batch = indices[start : start + batch_size]
            tensors = _batch_tensors(
                contexts=contexts,
                tokens=tokens,
                masks=masks,
                costs=costs,
                indices=batch,
                device=device,
            )
            _mris, prediction = model.forward_batch(*tensors)
            output.append(prediction.detach().cpu().numpy().astype(np.float64))
    if not output:
        raise MAVISMRISTrainingError("MRIS prediction roster is empty")
    combined = np.concatenate(output)
    if combined.shape != (indices.size,) or not np.all(np.isfinite(combined)):
        raise MAVISMRISTrainingError("MRIS predictions are invalid")
    return combined


def _specimen_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
    specimen_ids: tuple[str, ...],
) -> float:
    if predictions.shape != targets.shape or predictions.shape != (len(specimen_ids),):
        raise MAVISMRISTrainingError("validation predictions are misaligned")
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
        raise MAVISMRISTrainingError("validation MAE is invalid")
    return value


def _state_dict_cpu(model: MRISMechanicsModel) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _model_hash(
    model: MRISMechanicsModel,
    *,
    mode: str,
    outer_domain: str,
    normalizer: FoldNormalizer,
    selected_epoch: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema": 1,
                "mode": mode,
                "outer_domain": outer_domain,
                "normalizer_state_sha256": normalizer.state_sha256,
                "selected_epoch": selected_epoch,
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
    bank: MRISFeatureBank,
    *,
    mode: str,
    outer_domain: str,
    validation_domain: str | None,
    hidden_dimension: int,
    mris_dimension: int,
    learning_rate: float,
    epochs: int,
    patience: int | None,
    batch_size: int,
    seed: int,
    device: str,
) -> FittedMRISModel:
    rate = _validate_hyperparameters(
        bank=bank,
        mode=mode,
        outer_domain=outer_domain,
        hidden_dimension=hidden_dimension,
        mris_dimension=mris_dimension,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    if validation_domain is not None and (
        validation_domain not in bank.domain_order
        or validation_domain == outer_domain
        or type(patience) is not int
        or patience <= 0
    ):
        raise MAVISMRISTrainingError("inner validation request is invalid")
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
    normalized_targets = normalizer.transform_target(bank.targets)
    row_domains = np.asarray(bank.domain_ids, dtype=object)
    excluded = (outer_domain, *additional)
    fit_indices = np.flatnonzero(~np.isin(row_domains, excluded))
    validation_indices = (
        np.empty(0, dtype=np.int64)
        if validation_domain is None
        else np.flatnonzero(row_domains == validation_domain)
    )
    if fit_indices.size == 0 or (validation_domain is not None and validation_indices.size == 0):
        raise MAVISMRISTrainingError("MRIS training roster is empty")

    model = _model(
        hidden_dimension=hidden_dimension,
        mris_dimension=mris_dimension,
        seed=seed,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = math.inf
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
            tensors = _batch_tensors(
                contexts=contexts,
                tokens=inputs.token_features,
                masks=inputs.token_masks,
                costs=inputs.cost_features,
                indices=batch,
                device=device,
            )
            target = torch.tensor(
                normalized_targets[batch],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            _mris, prediction = model.forward_batch(*tensors)
            loss = torch.mean(torch.abs(prediction - target))
            if not torch.isfinite(loss):
                raise MAVISMRISTrainingError("MRIS training loss is invalid")
            loss.backward()
            optimizer.step()
        epochs_run = epoch
        if validation_domain is None:
            continue
        model.eval()
        prediction = normalizer.inverse_target(
            _predict_normalized(
                model,
                contexts=contexts,
                tokens=inputs.token_features,
                masks=inputs.token_masks,
                costs=inputs.cost_features,
                indices=validation_indices,
                batch_size=batch_size,
                device=device,
            )
        )
        mae = _specimen_mae(
            prediction,
            bank.targets[validation_indices],
            tuple(bank.specimen_ids[index] for index in validation_indices),
        )
        if mae < best_mae - 1.0e-12:
            best_mae = mae
            best_state = _state_dict_cpu(model)
            selected_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if validation_domain is not None:
        if best_state is None or not math.isfinite(best_mae):
            raise MAVISMRISTrainingError("MRIS early stopping did not select a model")
        model.load_state_dict(best_state)
        validation_mae: float | None = best_mae
    else:
        validation_mae = None
    model.cpu().eval()
    validation_ids = tuple(
        specimen_ids[index]
        for index, domain in enumerate(specimen_domains)
        if domain == validation_domain
    )
    audit = MRISTrainingAudit(
        mode=mode,
        outer_domain=outer_domain,
        validation_domains=additional,
        fit_domains=normalizer.fit_domains,
        fit_specimen_ids=normalizer.fit_specimen_ids,
        validation_specimen_ids=validation_ids,
        epochs_run=epochs_run,
        selected_epoch=selected_epoch,
        best_validation_mae=validation_mae,
        normalizer_state_sha256=normalizer.state_sha256,
    )
    return FittedMRISModel(
        mode=mode,
        outer_domain=outer_domain,
        hidden_dimension=hidden_dimension,
        mris_dimension=mris_dimension,
        model=model,
        normalizer=normalizer,
        audit=audit,
        model_state_sha256=_model_hash(
            model,
            mode=mode,
            outer_domain=outer_domain,
            normalizer=normalizer,
            selected_epoch=selected_epoch,
        ),
    )


def fit_inner_mris_fold(
    bank: MRISFeatureBank,
    *,
    mode: str,
    outer_domain: str,
    validation_domain: str,
    hidden_dimension: int,
    mris_dimension: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
) -> FittedMRISModel:
    return _fit(
        bank,
        mode=mode,
        outer_domain=outer_domain,
        validation_domain=validation_domain,
        hidden_dimension=hidden_dimension,
        mris_dimension=mris_dimension,
        learning_rate=learning_rate,
        epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )


def fit_final_mris_model(
    bank: MRISFeatureBank,
    *,
    mode: str,
    outer_domain: str,
    hidden_dimension: int,
    mris_dimension: int,
    learning_rate: float,
    selected_epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> FittedMRISModel:
    return _fit(
        bank,
        mode=mode,
        outer_domain=outer_domain,
        validation_domain=None,
        hidden_dimension=hidden_dimension,
        mris_dimension=mris_dimension,
        learning_rate=learning_rate,
        epochs=selected_epochs,
        patience=None,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )


def save_fitted_mris_checkpoint(
    fitted: FittedMRISModel,
    path: str | Path,
) -> None:
    if type(fitted) is not FittedMRISModel:
        raise MAVISMRISTrainingError("issued fitted MRIS model is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    weights = sorted(fitted.model.state_dict().items())
    metadata = {
        "schema_version": 1,
        "mode": fitted.mode,
        "outer_domain": fitted.outer_domain,
        "hidden_dimension": fitted.hidden_dimension,
        "mris_dimension": fitted.mris_dimension,
        "model_state_sha256": fitted.model_state_sha256,
        "weight_names": [name for name, _value in weights],
        "normalizer": {
            "outer_domain": fitted.normalizer.outer_domain,
            "excluded_domains": fitted.normalizer.excluded_domains,
            "target_mean": fitted.normalizer.target_mean,
            "target_scale": fitted.normalizer.target_scale,
            "fit_specimen_ids": fitted.normalizer.fit_specimen_ids,
            "fit_dataset_ids": fitted.normalizer.fit_dataset_ids,
            "fit_domains": fitted.normalizer.fit_domains,
            "state_sha256": fitted.normalizer.state_sha256,
        },
        "audit": {
            "mode": fitted.audit.mode,
            "outer_domain": fitted.audit.outer_domain,
            "validation_domains": fitted.audit.validation_domains,
            "fit_domains": fitted.audit.fit_domains,
            "fit_specimen_ids": fitted.audit.fit_specimen_ids,
            "validation_specimen_ids": fitted.audit.validation_specimen_ids,
            "epochs_run": fitted.audit.epochs_run,
            "selected_epoch": fitted.audit.selected_epoch,
            "best_validation_mae": fitted.audit.best_validation_mae,
            "normalizer_state_sha256": fitted.audit.normalizer_state_sha256,
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


def _metadata_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or any(type(item) is not str or not item for item in value)
    ):
        raise MAVISMRISTrainingError(f"checkpoint {label} is invalid")
    return tuple(value)


def load_fitted_mris_checkpoint(
    path: str | Path,
    *,
    expected_model_state_sha256: str | None = None,
) -> FittedMRISModel:
    if expected_model_state_sha256 is not None and (
        type(expected_model_state_sha256) is not str
        or len(expected_model_state_sha256) != 64
    ):
        raise MAVISMRISTrainingError("expected model state hash is invalid")
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(
                np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes().decode(
                    "utf-8"
                )
            )
            if type(metadata) is not dict or metadata.get("schema_version") != 1:
                raise MAVISMRISTrainingError("checkpoint metadata schema changed")
            weight_names = _metadata_tuple(metadata.get("weight_names"), "weights")
            expected_files = {
                "metadata_json",
                "context_mean",
                "context_scale",
                *(f"weight_{index:02d}" for index in range(len(weight_names))),
            }
            if set(archive.files) != expected_files or len(set(weight_names)) != len(
                weight_names
            ):
                raise MAVISMRISTrainingError("checkpoint weight roster changed")
            context_mean = np.asarray(archive["context_mean"], dtype=np.float64)
            context_scale = np.asarray(archive["context_scale"], dtype=np.float64)
            weight_arrays = tuple(
                np.asarray(archive[f"weight_{index:02d}"])
                for index in range(len(weight_names))
            )
    except MAVISMRISTrainingError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise MAVISMRISTrainingError("MRIS checkpoint is invalid") from error
    mode = metadata.get("mode")
    outer_domain = metadata.get("outer_domain")
    hidden_dimension = metadata.get("hidden_dimension")
    mris_dimension = metadata.get("mris_dimension")
    stored_model_hash = metadata.get("model_state_sha256")
    normalizer_raw = metadata.get("normalizer")
    audit_raw = metadata.get("audit")
    if (
        mode not in _TRAINABLE_MODES
        or type(outer_domain) is not str
        or not outer_domain
        or type(hidden_dimension) is not int
        or type(mris_dimension) is not int
        or min(hidden_dimension, mris_dimension) <= 0
        or type(stored_model_hash) is not str
        or len(stored_model_hash) != 64
        or type(normalizer_raw) is not dict
        or type(audit_raw) is not dict
        or context_mean.shape != (34,)
        or context_scale.shape != (34,)
        or not np.all(np.isfinite(context_mean))
        or not np.all(np.isfinite(context_scale))
        or np.any(context_scale <= 0.0)
    ):
        raise MAVISMRISTrainingError("MRIS checkpoint metadata is invalid")
    excluded_domains = _metadata_tuple(
        normalizer_raw.get("excluded_domains"), "excluded domains"
    )
    fit_specimen_ids = _metadata_tuple(
        normalizer_raw.get("fit_specimen_ids"), "fit specimen IDs"
    )
    fit_dataset_ids = _metadata_tuple(
        normalizer_raw.get("fit_dataset_ids"), "fit dataset IDs"
    )
    fit_domains = _metadata_tuple(normalizer_raw.get("fit_domains"), "fit domains")
    target_mean = float(normalizer_raw.get("target_mean"))
    target_scale = float(normalizer_raw.get("target_scale"))
    if (
        normalizer_raw.get("outer_domain") != outer_domain
        or len(fit_specimen_ids) != len(fit_dataset_ids)
        or not fit_specimen_ids
        or not math.isfinite(target_mean)
        or not math.isfinite(target_scale)
        or target_scale <= 0.0
    ):
        raise MAVISMRISTrainingError("checkpoint normalizer metadata is invalid")
    frozen_mean = np.frombuffer(
        np.ascontiguousarray(context_mean, dtype="<f8").tobytes(order="C"),
        dtype="<f8",
    )
    frozen_scale = np.frombuffer(
        np.ascontiguousarray(context_scale, dtype="<f8").tobytes(order="C"),
        dtype="<f8",
    )
    frozen_mean.setflags(write=False)
    frozen_scale.setflags(write=False)
    normalizer_state = fold_normalizer_state_sha256(
        outer_domain=outer_domain,
        excluded_domains=excluded_domains,
        context_mean=frozen_mean,
        context_scale=frozen_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        fit_specimen_ids=fit_specimen_ids,
        fit_dataset_ids=fit_dataset_ids,
    )
    if normalizer_state != normalizer_raw.get("state_sha256"):
        raise MAVISMRISTrainingError("checkpoint normalizer state hash changed")
    normalizer = FoldNormalizer(
        outer_domain=outer_domain,
        excluded_domains=excluded_domains,
        context_mean=frozen_mean,
        context_scale=frozen_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        fit_specimen_ids=fit_specimen_ids,
        fit_dataset_ids=fit_dataset_ids,
        fit_domains=fit_domains,
        state_sha256=normalizer_state,
    )
    model = _model(
        hidden_dimension=hidden_dimension,
        mris_dimension=mris_dimension,
        seed=0,
    )
    expected_state = model.state_dict()
    if tuple(sorted(expected_state)) != weight_names:
        raise MAVISMRISTrainingError("checkpoint model graph changed")
    state = {
        name: torch.tensor(value.copy(), dtype=expected_state[name].dtype)
        for name, value in zip(weight_names, weight_arrays, strict=True)
    }
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise MAVISMRISTrainingError("checkpoint model weights changed") from error
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
    best_mae_raw = audit_raw.get("best_validation_mae")
    best_mae = None if best_mae_raw is None else float(best_mae_raw)
    if (
        audit_raw.get("mode") != mode
        or audit_raw.get("outer_domain") != outer_domain
        or audit_raw.get("normalizer_state_sha256") != normalizer_state
        or audit_fit_domains != fit_domains
        or audit_fit_ids != fit_specimen_ids
        or type(epochs_run) is not int
        or type(selected_epoch) is not int
        or not 0 < selected_epoch <= epochs_run
        or (best_mae is not None and (not math.isfinite(best_mae) or best_mae < 0.0))
    ):
        raise MAVISMRISTrainingError("checkpoint training audit changed")
    audit = MRISTrainingAudit(
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
    observed_model_hash = _model_hash(
        model,
        mode=mode,
        outer_domain=outer_domain,
        normalizer=normalizer,
        selected_epoch=selected_epoch,
    )
    if observed_model_hash != stored_model_hash or (
        expected_model_state_sha256 is not None
        and observed_model_hash != expected_model_state_sha256
    ):
        raise MAVISMRISTrainingError("checkpoint model state hash changed")
    return FittedMRISModel(
        mode=mode,
        outer_domain=outer_domain,
        hidden_dimension=hidden_dimension,
        mris_dimension=mris_dimension,
        model=model.eval(),
        normalizer=normalizer,
        audit=audit,
        model_state_sha256=observed_model_hash,
    )


__all__ = [
    "FittedMRISModel",
    "MAVISMRISTrainingError",
    "MRISTrainingAudit",
    "fit_final_mris_model",
    "fit_inner_mris_fold",
    "load_fitted_mris_checkpoint",
    "save_fitted_mris_checkpoint",
]
