"""Deterministic inner-fold training for D8 residual diffusion."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .authority import (
    D8InnerFold,
    D8SearchView,
    validate_inner_fold,
    validate_search_view,
)
from .residual_config import (
    ResidualCandidate,
    ResidualDiffusionConfig,
)
from .residual_model import (
    ResidualCheckpoint,
    _configure_determinism,
    _is_registered_runtime_device,
    build_residual_unet,
    build_train_scheduler,
    freeze_residual_checkpoint,
    residual_diffusion_loss,
)
from .residual_targets import (
    ResidualTargetBatch,
    validate_residual_target_batch,
)


class ResidualTrainingError(ValueError):
    """Raised when inner-fold training loses its registered authority."""


@dataclass(frozen=True, slots=True)
class EpochLossRecord:
    """Sample-weighted finite loss components for one final-order epoch."""

    epoch: int
    total: float
    diffusion: float
    spectral: float
    low_pass: float
    sample_count: int
    batch_count: int

    def __post_init__(self) -> None:
        if (
            type(self.epoch) is not int
            or self.epoch < 1
            or type(self.sample_count) is not int
            or self.sample_count < 1
            or type(self.batch_count) is not int
            or self.batch_count < 1
            or any(
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
                for value in (
                    self.total,
                    self.diffusion,
                    self.spectral,
                    self.low_pass,
                )
            )
        ):
            raise ResidualTrainingError("epoch loss record is invalid")


@dataclass(frozen=True, slots=True)
class ResidualTrainingResult:
    """One replayable final-epoch checkpoint and its fit-only evidence."""

    outer_domain: str
    query_domain: str
    candidate_id: str
    seed: int
    epochs: int
    fit_specimen_ids: tuple[str, ...]
    fit_dataset_ids: tuple[str, ...]
    target_state_sha256: str
    split_sha256: str
    epoch_losses: tuple[EpochLossRecord, ...]
    checkpoint: ResidualCheckpoint
    sample_count: int
    batch_count: int
    response_read_count: int
    test_scale_override: bool
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "outer_domain": self.outer_domain,
            "query_domain": self.query_domain,
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "epochs": self.epochs,
            "fit_specimen_ids": self.fit_specimen_ids,
            "fit_dataset_ids": self.fit_dataset_ids,
            "target_state_sha256": self.target_state_sha256,
            "split_sha256": self.split_sha256,
            "epoch_losses": [asdict(value) for value in self.epoch_losses],
            "checkpoint_scientific_digest": self.checkpoint.scientific_digest,
            "sample_count": self.sample_count,
            "batch_count": self.batch_count,
            "response_read_count": self.response_read_count,
            "test_scale_override": self.test_scale_override,
        }
        if (
            len(self.epoch_losses) != self.epochs
            or self.response_read_count != 0
            or self.sample_count != len(self.fit_specimen_ids)
            or len(self.fit_dataset_ids) != self.sample_count
            or self.checkpoint.candidate_id != self.candidate_id
            or self.checkpoint.training_seed != self.seed
            or self.checkpoint.split_sha256 != self.split_sha256
        ):
            raise ResidualTrainingError("training result state is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class ResidualFinalTrainingResult:
    """One complete five-domain checkpoint frozen before outer evaluation."""

    outer_domain: str
    candidate_id: str
    seed: int
    epochs: int
    fit_specimen_ids: tuple[str, ...]
    fit_dataset_ids: tuple[str, ...]
    target_state_sha256: str
    split_sha256: str
    epoch_losses: tuple[EpochLossRecord, ...]
    checkpoint: ResidualCheckpoint
    sample_count: int
    batch_count: int
    response_read_count: int
    test_scale_override: bool
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "outer_domain": self.outer_domain,
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "epochs": self.epochs,
            "fit_specimen_ids": self.fit_specimen_ids,
            "fit_dataset_ids": self.fit_dataset_ids,
            "target_state_sha256": self.target_state_sha256,
            "split_sha256": self.split_sha256,
            "epoch_losses": [asdict(value) for value in self.epoch_losses],
            "checkpoint_scientific_digest": self.checkpoint.scientific_digest,
            "sample_count": self.sample_count,
            "batch_count": self.batch_count,
            "response_read_count": self.response_read_count,
            "test_scale_override": self.test_scale_override,
        }
        if (
            len(self.epoch_losses) != self.epochs
            or self.response_read_count != 0
            or self.sample_count != len(self.fit_specimen_ids)
            or len(self.fit_dataset_ids) != self.sample_count
            or self.checkpoint.candidate_id != self.candidate_id
            or self.checkpoint.training_seed != self.seed
            or self.checkpoint.split_sha256 != self.split_sha256
        ):
            raise ResidualTrainingError("final training result state is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest(),
        )


def _derived_seed(*values: object) -> int:
    payload = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _validate_training_authority(
    inner_fold: object,
    target_batch: object,
    *,
    config: object,
    candidate: object,
) -> tuple[
    D8InnerFold,
    ResidualTargetBatch,
    ResidualDiffusionConfig,
    ResidualCandidate,
    str,
    str,
]:
    if type(inner_fold) is not D8InnerFold:
        raise TypeError("exact D8InnerFold is required")
    if type(target_batch) is not ResidualTargetBatch:
        raise TypeError("exact ResidualTargetBatch is required")
    if type(config) is not ResidualDiffusionConfig:
        raise TypeError("exact ResidualDiffusionConfig is required")
    if type(candidate) is not ResidualCandidate:
        raise TypeError("exact ResidualCandidate is required")
    split_state = validate_inner_fold(inner_fold)
    target_state = validate_residual_target_batch(target_batch)
    try:
        registered_candidate = config.candidate(candidate.candidate_id)
    except ValueError as error:
        raise ResidualTrainingError("candidate is not registered") from error
    if (
        candidate != registered_candidate
        or target_batch.role != "inner_fit"
        or target_batch.outer_domain != inner_fold.outer_domain
        or target_batch.authority_sha256 != split_state
        or target_batch.specimen_ids != inner_fold.fit_specimen_ids
        or target_batch.dataset_ids != inner_fold.fit_dataset_ids
    ):
        raise ResidualTrainingError("training target authority differs from inner fold")
    return (
        inner_fold,
        target_batch,
        config,
        candidate,
        split_state,
        target_state,
    )


def _validate_outer_training_authority(
    search_view: object,
    target_batch: object,
    *,
    config: object,
    candidate: object,
) -> tuple[
    D8SearchView,
    ResidualTargetBatch,
    ResidualDiffusionConfig,
    ResidualCandidate,
    str,
    str,
]:
    if type(search_view) is not D8SearchView:
        raise TypeError("exact D8SearchView is required")
    if type(target_batch) is not ResidualTargetBatch:
        raise TypeError("exact ResidualTargetBatch is required")
    if type(config) is not ResidualDiffusionConfig:
        raise TypeError("exact ResidualDiffusionConfig is required")
    if type(candidate) is not ResidualCandidate:
        raise TypeError("exact ResidualCandidate is required")
    split_state = validate_search_view(search_view)
    target_state = validate_residual_target_batch(target_batch)
    try:
        registered_candidate = config.candidate(candidate.candidate_id)
    except ValueError as error:
        raise ResidualTrainingError("candidate is not registered") from error
    if (
        candidate != registered_candidate
        or target_batch.role != "outer_fit"
        or target_batch.outer_domain != search_view.outer_domain
        or target_batch.authority_sha256 != split_state
        or target_batch.specimen_ids != search_view.specimen_ids
        or target_batch.dataset_ids != search_view.dataset_ids
    ):
        raise ResidualTrainingError(
            "final training target authority differs from search view"
        )
    return (
        search_view,
        target_batch,
        config,
        candidate,
        split_state,
        target_state,
    )


def _epoch_record(
    epoch: int,
    components: list[tuple[int, tuple[float, float, float, float]]],
    *,
    expected_samples: int,
) -> EpochLossRecord:
    sample_count = sum(count for count, _values in components)
    if sample_count != expected_samples or not components:
        raise ResidualTrainingError("epoch did not consume each fit identity once")
    averages = tuple(
        float(
            math.fsum(count * values[index] for count, values in components)
            / sample_count
        )
        for index in range(4)
    )
    return EpochLossRecord(
        epoch=epoch,
        total=averages[0],
        diffusion=averages[1],
        spectral=averages[2],
        low_pass=averages[3],
        sample_count=sample_count,
        batch_count=len(components),
    )


def _train_registered_checkpoint(
    targets: ResidualTargetBatch,
    *,
    config: ResidualDiffusionConfig,
    candidate: ResidualCandidate,
    split_state: str,
    epochs: int,
    seed: int,
    device: str,
    test_scale_override: bool,
) -> tuple[tuple[EpochLossRecord, ...], ResidualCheckpoint, int, int]:
    if (
        type(test_scale_override) is not bool
        or type(epochs) is not int
        or (
            epochs != 1
            if test_scale_override
            else epochs
            not in {config.screening_epochs, config.rerank_epochs}
        )
        or type(seed) is not int
        or seed not in config.training_seeds
        or not _is_registered_runtime_device(device, allow_cpu=False)
    ):
        raise ResidualTrainingError("training runtime or hyperparameters changed")
    _configure_determinism(seed)
    model = build_residual_unet(candidate).to(device)
    model.train()
    scheduler = build_train_scheduler(
        candidate,
        train_timesteps=config.train_timesteps,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
        fused=False,
    )
    epoch_records: list[EpochLossRecord] = []
    rows = len(targets.specimen_ids)
    for epoch_index in range(epochs):
        order_seed = _derived_seed(
            "order",
            seed,
            split_state,
            candidate.candidate_id,
            epoch_index,
        )
        order = np.random.Generator(np.random.PCG64(order_seed)).permutation(rows)
        components: list[tuple[int, tuple[float, float, float, float]]] = []
        for start in range(0, rows, config.batch_size):
            indices = order[start : start + config.batch_size]
            batch_target = torch.from_numpy(
                np.array(targets.training_target[indices], copy=True, order="C")
            ).to(device)
            batch_condition = torch.from_numpy(
                np.array(targets.stable_condition[indices], copy=True, order="C")
            ).to(device)
            generator = torch.Generator(device=device).manual_seed(
                _derived_seed(
                    "noise",
                    seed,
                    split_state,
                    candidate.candidate_id,
                    epoch_index,
                    tuple(targets.specimen_ids[int(index)] for index in indices),
                )
            )
            noise = torch.randn(
                batch_target.shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            timesteps = torch.randint(
                0,
                config.train_timesteps,
                (len(indices),),
                dtype=torch.int64,
                device=device,
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = residual_diffusion_loss(
                model,
                scheduler,
                batch_target,
                batch_condition,
                timesteps,
                noise,
                candidate,
            )
            loss.total.backward()
            optimizer.step()
            values = tuple(
                float(value.detach().to(device="cpu").item())
                for value in (
                    loss.total,
                    loss.diffusion,
                    loss.spectral,
                    loss.low_pass,
                )
            )
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ResidualTrainingError("training loss is not finite")
            components.append((len(indices), values))
        if any(
            parameter.is_floating_point()
            and not bool(torch.isfinite(parameter.detach()).all().item())
            for parameter in model.parameters()
        ):
            raise ResidualTrainingError("training produced a nonfinite model")
        epoch_records.append(
            _epoch_record(
                epoch_index + 1,
                components,
                expected_samples=rows,
            )
        )
    checkpoint = freeze_residual_checkpoint(
        model,
        candidate=candidate,
        config_sha256=config.config_sha256,
        split_sha256=split_state,
        training_seed=seed,
    )
    losses = tuple(epoch_records)
    return (
        losses,
        checkpoint,
        rows,
        sum(record.batch_count for record in losses),
    )


def train_inner_residual_model(
    inner_fold: D8InnerFold,
    target_batch: ResidualTargetBatch,
    *,
    config: ResidualDiffusionConfig,
    candidate: ResidualCandidate,
    epochs: int,
    seed: int,
    device: str,
    test_scale_override: bool = False,
) -> ResidualTrainingResult:
    """Train one response-free final-epoch model on four inner-fit domains."""

    inner, targets, clean_config, clean_candidate, split_state, target_state = (
        _validate_training_authority(
            inner_fold,
            target_batch,
            config=config,
            candidate=candidate,
        )
    )
    epoch_records, checkpoint, rows, batch_count = _train_registered_checkpoint(
        targets,
        config=clean_config,
        candidate=clean_candidate,
        split_state=split_state,
        epochs=epochs,
        seed=seed,
        device=device,
        test_scale_override=test_scale_override,
    )
    return ResidualTrainingResult(
        outer_domain=inner.outer_domain,
        query_domain=inner.query_domain,
        candidate_id=clean_candidate.candidate_id,
        seed=seed,
        epochs=epochs,
        fit_specimen_ids=targets.specimen_ids,
        fit_dataset_ids=targets.dataset_ids,
        target_state_sha256=target_state,
        split_sha256=split_state,
        epoch_losses=epoch_records,
        checkpoint=checkpoint,
        sample_count=rows,
        batch_count=batch_count,
        response_read_count=0,
        test_scale_override=test_scale_override,
    )


def train_outer_fit_residual_model(
    search_view: D8SearchView,
    target_batch: ResidualTargetBatch,
    *,
    config: ResidualDiffusionConfig,
    candidate: ResidualCandidate,
    epochs: int,
    seed: int,
    device: str,
    test_scale_override: bool = False,
) -> ResidualFinalTrainingResult:
    """Train one final response-free model on the complete five-domain view."""

    search, targets, clean_config, clean_candidate, split_state, target_state = (
        _validate_outer_training_authority(
            search_view,
            target_batch,
            config=config,
            candidate=candidate,
        )
    )
    epoch_records, checkpoint, rows, batch_count = _train_registered_checkpoint(
        targets,
        config=clean_config,
        candidate=clean_candidate,
        split_state=split_state,
        epochs=epochs,
        seed=seed,
        device=device,
        test_scale_override=test_scale_override,
    )
    return ResidualFinalTrainingResult(
        outer_domain=search.outer_domain,
        candidate_id=clean_candidate.candidate_id,
        seed=seed,
        epochs=epochs,
        fit_specimen_ids=targets.specimen_ids,
        fit_dataset_ids=targets.dataset_ids,
        target_state_sha256=target_state,
        split_sha256=split_state,
        epoch_losses=epoch_records,
        checkpoint=checkpoint,
        sample_count=rows,
        batch_count=batch_count,
        response_read_count=0,
        test_scale_override=test_scale_override,
    )


__all__ = [
    "EpochLossRecord",
    "ResidualFinalTrainingResult",
    "ResidualTrainingError",
    "ResidualTrainingResult",
    "train_inner_residual_model",
    "train_outer_fit_residual_model",
]
