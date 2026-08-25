"""Fold-local normalization and CAI mechanics head for MAVIS MRIS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .state_encoder import MRISInput, MRISStateEncoder


class MAVISMechanicsHeadError(ValueError):
    """Raised when mechanics normalization or prediction is invalid."""


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.frombuffer(
        np.ascontiguousarray(value, dtype="<f8").tobytes(order="C"),
        dtype="<f8",
    ).reshape(value.shape)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True, eq=False)
class FoldNormalizer:
    outer_domain: str
    excluded_domains: tuple[str, ...]
    context_mean: np.ndarray
    context_scale: np.ndarray
    target_mean: float
    target_scale: float
    fit_specimen_ids: tuple[str, ...]
    fit_dataset_ids: tuple[str, ...]
    fit_domains: tuple[str, ...]
    state_sha256: str

    def transform_context(self, contexts: object) -> np.ndarray:
        values = np.asarray(contexts, dtype=np.float64)
        if values.shape[-1:] != self.context_mean.shape:
            raise MAVISMechanicsHeadError("normalizer context shape is invalid")
        output = (values - self.context_mean) / self.context_scale
        if not np.all(np.isfinite(output)):
            raise MAVISMechanicsHeadError("normalized context is invalid")
        return np.asarray(output, dtype=np.float64)

    def transform_target(self, targets: object) -> np.ndarray:
        values = np.asarray(targets, dtype=np.float64)
        output = (values - self.target_mean) / self.target_scale
        if not np.all(np.isfinite(output)):
            raise MAVISMechanicsHeadError("normalized target is invalid")
        return np.asarray(output, dtype=np.float64)

    def inverse_target(self, values: object) -> np.ndarray:
        output = np.asarray(values, dtype=np.float64) * self.target_scale + self.target_mean
        if not np.all(np.isfinite(output)):
            raise MAVISMechanicsHeadError("inverse target is invalid")
        return np.asarray(output, dtype=np.float64)

    def __eq__(self, other: object) -> bool:
        return type(other) is FoldNormalizer and self.state_sha256 == other.state_sha256


def fold_normalizer_state_sha256(
    *,
    outer_domain: str,
    excluded_domains: tuple[str, ...],
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    target_mean: float,
    target_scale: float,
    fit_specimen_ids: tuple[str, ...],
    fit_dataset_ids: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema": 1,
                "outer_domain": outer_domain,
                "excluded_domains": excluded_domains,
                "target_mean": target_mean,
                "target_scale": target_scale,
                "fit_specimen_ids": fit_specimen_ids,
                "fit_dataset_ids": fit_dataset_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(context_mean, dtype="<f8").tobytes(order="C"))
    digest.update(np.ascontiguousarray(context_scale, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def fit_fold_normalizer(
    *,
    contexts: object,
    targets: object,
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    outer_domain: str,
    additional_excluded_domains: tuple[str, ...] = (),
) -> FoldNormalizer:
    values = np.asarray(contexts, dtype=np.float64)
    response = np.asarray(targets, dtype=np.float64)
    count = len(specimen_ids)
    if (
        type(specimen_ids) is not tuple
        or type(dataset_ids) is not tuple
        or count == 0
        or len(set(specimen_ids)) != count
        or len(dataset_ids) != count
        or outer_domain not in dataset_ids
        or type(additional_excluded_domains) is not tuple
        or len(set(additional_excluded_domains)) != len(additional_excluded_domains)
        or outer_domain in additional_excluded_domains
        or any(domain not in dataset_ids for domain in additional_excluded_domains)
        or values.shape != (count, 34)
        or response.shape != (count,)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(response))
    ):
        raise MAVISMechanicsHeadError("normalizer inputs are invalid")
    excluded_domains = (outer_domain, *additional_excluded_domains)
    fit_indices = np.flatnonzero(
        ~np.isin(np.asarray(dataset_ids, dtype=object), excluded_domains)
    )
    if fit_indices.size == 0:
        raise MAVISMechanicsHeadError("normalizer has no source rows")
    context_mean = np.mean(values[fit_indices], axis=0, dtype=np.float64)
    context_scale = np.std(values[fit_indices], axis=0, dtype=np.float64)
    context_scale[context_scale < 1.0e-12] = 1.0
    target_mean = float(np.mean(response[fit_indices], dtype=np.float64))
    target_scale = float(np.std(response[fit_indices], dtype=np.float64))
    if target_scale < 1.0e-12:
        target_scale = 1.0
    frozen_mean = _readonly(context_mean)
    frozen_scale = _readonly(context_scale)
    fit_ids = tuple(specimen_ids[index] for index in fit_indices)
    fit_dataset_ids = tuple(dataset_ids[index] for index in fit_indices)
    fit_domains = tuple(dict.fromkeys(fit_dataset_ids))
    state = fold_normalizer_state_sha256(
        outer_domain=outer_domain,
        excluded_domains=excluded_domains,
        context_mean=frozen_mean,
        context_scale=frozen_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        fit_specimen_ids=fit_ids,
        fit_dataset_ids=fit_dataset_ids,
    )
    return FoldNormalizer(
        outer_domain=outer_domain,
        excluded_domains=excluded_domains,
        context_mean=frozen_mean,
        context_scale=frozen_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        fit_specimen_ids=fit_ids,
        fit_dataset_ids=fit_dataset_ids,
        fit_domains=fit_domains,
        state_sha256=state,
    )


class MechanicsHead(nn.Module):
    def __init__(self, mris_dimension: int) -> None:
        super().__init__()
        if type(mris_dimension) is not int or mris_dimension <= 0:
            raise MAVISMechanicsHeadError("mechanics head dimension is invalid")
        self.regressor = nn.Linear(mris_dimension, 1)

    def forward(self, mris: torch.Tensor) -> torch.Tensor:
        output = self.regressor(mris).squeeze(-1)
        if not torch.isfinite(output).all():
            raise MAVISMechanicsHeadError("mechanics prediction is invalid")
        return output


class MRISMechanicsModel(nn.Module):
    def __init__(self, encoder: MRISStateEncoder) -> None:
        super().__init__()
        if not isinstance(encoder, MRISStateEncoder):
            raise MAVISMechanicsHeadError("issued MRIS encoder is required")
        self.encoder = encoder
        self.head = MechanicsHead(encoder.output_dimension)

    def forward(self, state: MRISInput) -> tuple[torch.Tensor, torch.Tensor]:
        mris = self.encoder(state)
        return mris, self.head(mris)

    def forward_batch(
        self,
        contexts: torch.Tensor,
        token_features: torch.Tensor,
        token_masks: torch.Tensor,
        cost_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mris = self.encoder.forward_batch(
            contexts,
            token_features,
            token_masks,
            cost_features,
        )
        return mris, self.head(mris)

__all__ = [
    "FoldNormalizer",
    "MAVISMechanicsHeadError",
    "MRISMechanicsModel",
    "MechanicsHead",
    "fit_fold_normalizer",
    "fold_normalizer_state_sha256",
]
