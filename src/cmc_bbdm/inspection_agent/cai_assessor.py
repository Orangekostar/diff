"""Metadata-free fold-local CAI assessor for observed inspection states."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.mva.cai_evaluator import CAIPredictor, fit_cai_predictor
from cmc_bbdm.mva.encoder_session import MVAEncoderSession

from .contracts import InspectionObservation


class CAIAssessorError(ValueError):
    """Raised when state features or a fold-local assessor violate G0."""


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise CAIAssessorError("CAI state array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


@dataclass(frozen=True, slots=True)
class StateFeatureRow:
    sample_id: str
    specimen_id: str
    dataset_id: str
    policy: str
    observation_sha256: str
    embedding: np.ndarray
    effective_budget: float
    observed_cell_fraction: float
    mean_observed_level: float
    true_cai: float

    def __post_init__(self) -> None:
        numbers = (
            float(self.effective_budget),
            float(self.observed_cell_fraction),
            float(self.mean_observed_level),
            float(self.true_cai),
        )
        if (
            any(type(value) is not str or not value for value in (
                self.sample_id,
                self.specimen_id,
                self.dataset_id,
                self.policy,
            ))
            or not _valid_sha256(self.observation_sha256)
            or not all(math.isfinite(value) for value in numbers)
            or not 0.0 <= numbers[0] <= 1.0
            or not 0.0 <= numbers[1] <= 1.0
            or not 0.0 <= numbers[2] <= 2.0
        ):
            raise CAIAssessorError("CAI state feature row is invalid")
        embedding = _readonly(self.embedding, dtype="<f8", shape=(512,))
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "effective_budget", numbers[0])
        object.__setattr__(self, "observed_cell_fraction", numbers[1])
        object.__setattr__(self, "mean_observed_level", numbers[2])
        object.__setattr__(self, "true_cai", numbers[3])

    @property
    def scalars(self) -> tuple[float, float, float]:
        return (
            self.effective_budget,
            self.observed_cell_fraction,
            self.mean_observed_level,
        )


@dataclass(frozen=True, slots=True)
class StateCAIAssessor:
    outer_domain: str
    predictor: CAIPredictor
    fit_sample_ids: tuple[str, ...]
    fit_physical_specimen_ids: tuple[str, ...]
    fit_domains: tuple[str, ...]
    state_scalar_count: int
    state_bank_sha256: str
    model_state_sha256: str

    def predict(self, embeddings: object, scalars: object) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float64)
        state_values = np.asarray(scalars, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 512
            or state_values.shape != (values.shape[0], 3)
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(state_values))
        ):
            raise CAIAssessorError("CAI assessor query is invalid")
        predictions = self.predictor.predict(state_values, values)
        if predictions.shape != (values.shape[0],) or not np.all(
            np.isfinite(predictions)
        ):
            raise CAIAssessorError("CAI assessor prediction is invalid")
        return np.asarray(predictions, dtype=np.float64)


def state_scalars(observation: InspectionObservation) -> np.ndarray:
    if type(observation) is not InspectionObservation:
        raise CAIAssessorError("issued observation is required")
    levels = observation.measurement_state.levels
    observed = tuple(level for level in levels if level >= 0)
    values = np.asarray(
        (
            observation.effective_budget,
            len(observed) / 64.0,
            0.0 if not observed else float(np.mean(observed, dtype=np.float64)),
        ),
        dtype=np.float64,
    )
    values.setflags(write=False)
    return values


def encode_reconstruction_images(
    images: tuple[np.ndarray, ...],
    encoder: MVAEncoderSession,
    *,
    chunk_size: int = 276,
) -> np.ndarray:
    if (
        type(images) is not tuple
        or not images
        or not isinstance(encoder, MVAEncoderSession)
        or type(chunk_size) is not int
        or not 0 < chunk_size <= 276
    ):
        raise CAIAssessorError("CAI encoding request is invalid")
    chunks = [
        encoder.encode(images[start : start + chunk_size])
        for start in range(0, len(images), chunk_size)
    ]
    values = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
    if values.shape != (len(images), 512) or not np.all(np.isfinite(values)):
        raise CAIAssessorError("CAI embeddings are invalid")
    return values


def fit_state_cai_assessor(
    rows: tuple[StateFeatureRow, ...],
    *,
    outer_domain: str,
    pca_dimension: int,
    ridge_alpha: float,
) -> StateCAIAssessor:
    if (
        type(rows) is not tuple
        or not rows
        or any(type(row) is not StateFeatureRow for row in rows)
        or type(outer_domain) is not str
        or not outer_domain
        or pca_dimension != 32
        or float(ridge_alpha) != 10.0
    ):
        raise CAIAssessorError("CAI assessor fit request is invalid")
    if any(row.dataset_id == outer_domain for row in rows):
        raise CAIAssessorError("outer target rows are forbidden from assessor fit")
    sample_ids = tuple(row.sample_id for row in rows)
    if len(set(sample_ids)) != len(sample_ids):
        raise CAIAssessorError("CAI state sample IDs must be unique")
    specimen_domains: dict[str, str] = {}
    specimen_targets: dict[str, float] = {}
    for row in rows:
        previous_domain = specimen_domains.setdefault(row.specimen_id, row.dataset_id)
        previous_target = specimen_targets.setdefault(row.specimen_id, row.true_cai)
        if previous_domain != row.dataset_id or previous_target != row.true_cai:
            raise CAIAssessorError("physical specimen identity changed across states")
    counts = Counter(row.specimen_id for row in rows)
    if len(set(counts.values())) != 1:
        raise CAIAssessorError("every physical specimen must have equal state count")
    domains = tuple(row.dataset_id for row in rows)
    physical_ids = tuple(dict.fromkeys(row.specimen_id for row in rows))
    embeddings = np.asarray([row.embedding for row in rows], dtype=np.float64)
    scalars = np.asarray([row.scalars for row in rows], dtype=np.float64)
    targets = np.asarray([row.true_cai for row in rows], dtype=np.float64)
    predictor = fit_cai_predictor(
        method="inspection_agent_state_cai_assessor",
        outer_domain=outer_domain,
        specimen_ids=sample_ids,
        dataset_ids=domains,
        targets=targets,
        metadata=scalars,
        embeddings=embeddings,
        dimension=pca_dimension,
        fit_indices=np.arange(len(rows), dtype=np.int64),
        ridge_alpha=ridge_alpha,
    )
    digest = hashlib.sha256()
    digest.update(b"inspection-agent-state-bank-features-v1")
    for row in rows:
        digest.update(
            json.dumps(
                {
                    "sample_id": row.sample_id,
                    "specimen_id": row.specimen_id,
                    "dataset_id": row.dataset_id,
                    "policy": row.policy,
                    "observation_sha256": row.observation_sha256,
                    "scalars": row.scalars,
                    "true_cai": row.true_cai,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(row.embedding.tobytes(order="C"))
    bank_hash = digest.hexdigest()
    model_hash = hashlib.sha256(
        (
            f"inspection-agent-state-cai-assessor-v1|{outer_domain}|"
            f"{predictor.state_sha256}|{bank_hash}"
        ).encode("ascii")
    ).hexdigest()
    return StateCAIAssessor(
        outer_domain=outer_domain,
        predictor=predictor,
        fit_sample_ids=sample_ids,
        fit_physical_specimen_ids=physical_ids,
        fit_domains=tuple(dict.fromkeys(domains)),
        state_scalar_count=3,
        state_bank_sha256=bank_hash,
        model_state_sha256=model_hash,
    )


__all__ = [
    "CAIAssessorError",
    "StateCAIAssessor",
    "StateFeatureRow",
    "encode_reconstruction_images",
    "fit_state_cai_assessor",
    "state_scalars",
]
