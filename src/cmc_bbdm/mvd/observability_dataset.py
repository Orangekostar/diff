"""Observed-only M1 inputs aligned with strict Mechanical Value teachers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.candidate_features import build_candidate_features
from cmc_bbdm.mva.interpolation import reconstruct_measurement_state
from cmc_bbdm.mva.measurement_state import initial_state

from .authority import CompactMVDAuthority, MVDAuthorityError
from .initial_value_dataset import build_source_initial_value_dataset


@dataclass(frozen=True, slots=True)
class StudentInputs:
    initial_embedding: np.ndarray
    current_prediction: float
    candidate_features: np.ndarray


@dataclass(frozen=True, slots=True)
class ObservedValueExamples:
    outer_domain: str
    role: str
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    initial_embeddings: np.ndarray
    current_predictions: np.ndarray
    candidate_features: np.ndarray
    initial_used_budgets: np.ndarray
    mechanical_values: np.ndarray
    candidate_costs: np.ndarray
    teacher_predictor_state_sha256: tuple[str, ...]
    candidate_bank_state_sha256: str
    observed_feature_state_sha256: str
    state_sha256: str

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)

    def student_inputs(self, index: int) -> StudentInputs:
        if type(index) is not int or not 0 <= index < self.specimen_count:
            raise IndexError("student input index is invalid")
        return StudentInputs(
            initial_embedding=self.initial_embeddings[index],
            current_prediction=float(self.current_predictions[index]),
            candidate_features=self.candidate_features[index],
        )


@dataclass(frozen=True, slots=True)
class ObservedCandidateFeatureBank:
    initial_budget: float
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    candidate_features: np.ndarray
    candidate_costs: np.ndarray
    grid_state_sha256: tuple[str, ...]
    candidate_bank_state_sha256: str
    state_sha256: str


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise MVDAuthorityError("observability array changed")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _bank_state(bank: ObservedCandidateFeatureBank) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "candidate_bank_state_sha256": bank.candidate_bank_state_sha256,
                "dataset_ids": bank.dataset_ids,
                "grid_state_sha256": bank.grid_state_sha256,
                "initial_budget": bank.initial_budget,
                "specimen_ids": bank.specimen_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(bank.candidate_features.tobytes(order="C"))
    digest.update(bank.candidate_costs.tobytes(order="C"))
    return digest.hexdigest()


def build_observed_candidate_feature_bank(
    authority: object,
    compact: CompactMVDAuthority,
    *,
    initial_budget: float,
) -> ObservedCandidateFeatureBank:
    """Simulate S0, retain only its eight observed-only features, then discard RGB."""

    candidate_bank = compact.candidate_banks[initial_budget]
    if (
        tuple(authority.specimen_ids) != compact.specimen_ids
        or tuple(authority.dataset_ids) != compact.dataset_ids
        or tuple(authority.image_sha256) != compact.image_sha256
    ):
        raise MVDAuthorityError("feature-bank runtime authority changed")
    rows = np.empty((compact.specimen_count, 64, 8), dtype="<f8")
    for index, image in enumerate(authority.images):
        grid = build_acquisition_grid(
            image.shape[0], image.shape[1], initial_budget=initial_budget
        )
        state = initial_state(grid)
        current = reconstruct_measurement_state(
            image,
            grid,
            state,
            interpolation="bilinear",
            specimen_id=compact.specimen_ids[index],
            dataset_id=compact.dataset_ids[index],
        ).image
        actions, features = build_candidate_features(
            grid,
            state,
            current_reconstruction=current,
            checkpoint=0.25,
        )
        if (
            tuple(action.cell_index for action in actions) != tuple(range(64))
            or any(action.from_level != 0 or action.to_level != 1 for action in actions)
            or grid.state_sha256 != candidate_bank.grid_state_sha256[index]
        ):
            raise MVDAuthorityError("observed candidate roster changed")
        rows[index] = features
        if (index + 1) % 16 == 0 or index + 1 == compact.specimen_count:
            print(
                f"[observability features {initial_budget}] "
                f"{index + 1}/{compact.specimen_count}",
                flush=True,
            )
    result = ObservedCandidateFeatureBank(
        initial_budget=initial_budget,
        specimen_ids=compact.specimen_ids,
        dataset_ids=compact.dataset_ids,
        candidate_features=_readonly(
            rows, dtype="<f8", shape=(compact.specimen_count, 64, 8)
        ),
        candidate_costs=_readonly(
            candidate_bank.added_measurements,
            dtype="<i8",
            shape=(compact.specimen_count, 64),
        ),
        grid_state_sha256=candidate_bank.grid_state_sha256,
        candidate_bank_state_sha256=candidate_bank.state_sha256,
        state_sha256="",
    )
    object.__setattr__(result, "state_sha256", _bank_state(result))
    return result


def save_observed_candidate_feature_bank(
    path: str | Path, bank: ObservedCandidateFeatureBank
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        initial_budget=np.asarray([bank.initial_budget], dtype="<f8"),
        specimen_ids=np.asarray(bank.specimen_ids),
        dataset_ids=np.asarray(bank.dataset_ids),
        candidate_features=bank.candidate_features,
        candidate_costs=bank.candidate_costs,
        grid_state_sha256=np.asarray(bank.grid_state_sha256),
        candidate_bank_state_sha256=np.asarray([bank.candidate_bank_state_sha256]),
        state_sha256=np.asarray([bank.state_sha256]),
    )


def load_observed_candidate_feature_bank(
    path: str | Path,
    *,
    compact: CompactMVDAuthority,
    initial_budget: float,
) -> ObservedCandidateFeatureBank:
    candidate_bank = compact.candidate_banks[initial_budget]
    try:
        with np.load(path, allow_pickle=False) as archive:
            result = ObservedCandidateFeatureBank(
                initial_budget=float(archive["initial_budget"][0]),
                specimen_ids=tuple(str(value) for value in archive["specimen_ids"]),
                dataset_ids=tuple(str(value) for value in archive["dataset_ids"]),
                candidate_features=_readonly(
                    archive["candidate_features"],
                    dtype="<f8",
                    shape=(compact.specimen_count, 64, 8),
                ),
                candidate_costs=_readonly(
                    archive["candidate_costs"],
                    dtype="<i8",
                    shape=(compact.specimen_count, 64),
                ),
                grid_state_sha256=tuple(
                    str(value) for value in archive["grid_state_sha256"]
                ),
                candidate_bank_state_sha256=str(
                    archive["candidate_bank_state_sha256"][0]
                ),
                state_sha256=str(archive["state_sha256"][0]),
            )
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        raise MVDAuthorityError("observed candidate feature bank cannot be read") from error
    if (
        result.initial_budget != initial_budget
        or result.specimen_ids != compact.specimen_ids
        or result.dataset_ids != compact.dataset_ids
        or result.grid_state_sha256 != candidate_bank.grid_state_sha256
        or result.candidate_bank_state_sha256 != candidate_bank.state_sha256
        or not np.array_equal(result.candidate_costs, candidate_bank.added_measurements)
        or _bank_state(result) != result.state_sha256
    ):
        raise MVDAuthorityError("observed candidate feature authority changed")
    return result


def _examples_state(value: ObservedValueExamples) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "candidate_bank_state_sha256": value.candidate_bank_state_sha256,
                "dataset_ids": value.dataset_ids,
                "observed_feature_state_sha256": value.observed_feature_state_sha256,
                "outer_domain": value.outer_domain,
                "role": value.role,
                "specimen_ids": value.specimen_ids,
                "teacher_predictor_state_sha256": value.teacher_predictor_state_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for name in (
        "initial_embeddings",
        "current_predictions",
        "candidate_features",
        "initial_used_budgets",
        "mechanical_values",
        "candidate_costs",
    ):
        digest.update(np.asarray(getattr(value, name)).tobytes(order="C"))
    return digest.hexdigest()


def build_outer_observability_examples(
    compact: CompactMVDAuthority,
    feature_bank: ObservedCandidateFeatureBank,
    target_initial_values: pl.DataFrame,
    *,
    outer_domain: str,
) -> tuple[ObservedValueExamples, ObservedValueExamples]:
    """Return strict source-training and untouched outer-target examples."""

    source = build_source_initial_value_dataset(compact, outer_domain=outer_domain)
    if (
        feature_bank.specimen_ids != compact.specimen_ids
        or feature_bank.candidate_bank_state_sha256
        != source.candidate_bank_state_sha256
    ):
        raise MVDAuthorityError("observability feature and teacher banks differ")
    source_indices = np.asarray(
        [compact.specimen_ids.index(value) for value in source.specimen_ids],
        dtype=np.int64,
    )
    target_indices = np.flatnonzero(
        np.asarray(compact.dataset_ids, dtype=object) == outer_domain
    )
    target_rows = target_initial_values.filter(
        pl.col("outer_domain") == outer_domain
    ).sort(["specimen_id", "cell_index"])
    grouped = {
        specimen_id: rows
        for specimen_id, rows in target_rows.partition_by(
            "specimen_id", as_dict=True, include_key=False
        ).items()
    }
    # Polars tuple-keys are normalized here to the specimen string.
    grouped = {
        key[0] if isinstance(key, tuple) else key: value for key, value in grouped.items()
    }

    def freeze_examples(
        *,
        role: str,
        indices: np.ndarray,
        specimen_ids: tuple[str, ...],
        dataset_ids: tuple[str, ...],
        current: np.ndarray,
        values: np.ndarray,
        hashes: tuple[str, ...],
    ) -> ObservedValueExamples:
        count = len(specimen_ids)
        result = ObservedValueExamples(
            outer_domain=outer_domain,
            role=role,
            specimen_ids=specimen_ids,
            dataset_ids=dataset_ids,
            initial_embeddings=_readonly(
                compact.candidate_banks[source.initial_budget].initial_embeddings[indices],
                dtype="<f8",
                shape=(count, 512),
            ),
            current_predictions=_readonly(current, dtype="<f8", shape=(count,)),
            candidate_features=_readonly(
                feature_bank.candidate_features[indices],
                dtype="<f8",
                shape=(count, 64, 8),
            ),
            initial_used_budgets=_readonly(
                np.asarray(
                    [
                        compact.candidate_banks[source.initial_budget]
                        .initial_measured_counts[index]
                        / compact.candidate_banks[source.initial_budget].native_counts[
                            index
                        ]
                        for index in indices
                    ],
                    dtype="<f8",
                ),
                dtype="<f8",
                shape=(count,),
            ),
            mechanical_values=_readonly(values, dtype="<f8", shape=(count, 64)),
            candidate_costs=_readonly(
                feature_bank.candidate_costs[indices],
                dtype="<i8",
                shape=(count, 64),
            ),
            teacher_predictor_state_sha256=hashes,
            candidate_bank_state_sha256=source.candidate_bank_state_sha256,
            observed_feature_state_sha256=feature_bank.state_sha256,
            state_sha256="",
        )
        object.__setattr__(result, "state_sha256", _examples_state(result))
        return result

    source_examples = freeze_examples(
        role="source_train",
        indices=source_indices,
        specimen_ids=source.specimen_ids,
        dataset_ids=source.dataset_ids,
        current=source.current_predictions,
        values=source.mechanical_values,
        hashes=source.predictor_state_sha256,
    )
    target_ids = tuple(compact.specimen_ids[index] for index in target_indices)
    target_current = np.empty(len(target_ids), dtype="<f8")
    target_values = np.empty((len(target_ids), 64), dtype="<f8")
    target_hashes: list[str] = []
    for row_index, specimen_id in enumerate(target_ids):
        rows = grouped.get(specimen_id)
        if rows is None or rows.height != 64:
            raise MVDAuthorityError("outer target teacher roster changed")
        current_values = set(rows["current_prediction"])
        hashes = set(rows["p_a_predictor_state_sha256"])
        if len(current_values) != 1 or len(hashes) != 1:
            raise MVDAuthorityError("outer target teacher binding changed")
        target_current[row_index] = float(current_values.pop())
        target_values[row_index] = rows["initial_mechanical_value"].to_numpy()
        target_hashes.append(str(hashes.pop()))
    target_examples = freeze_examples(
        role="outer_target",
        indices=target_indices,
        specimen_ids=target_ids,
        dataset_ids=tuple(compact.dataset_ids[index] for index in target_indices),
        current=target_current,
        values=target_values,
        hashes=tuple(target_hashes),
    )
    return source_examples, target_examples


def subset_observed_examples(
    examples: ObservedValueExamples,
    *,
    included_domains: tuple[str, ...],
    role: str = "source_train",
) -> ObservedValueExamples:
    """Create a hash-bound source-domain subset for inner grouped CV."""

    if (
        type(examples) is not ObservedValueExamples
        or not included_domains
        or len(set(included_domains)) != len(included_domains)
        or not set(included_domains) <= set(examples.dataset_ids)
        or role not in {"source_train", "source_validation"}
    ):
        raise MVDAuthorityError("observability subset request changed")
    indices = np.asarray(
        [
            index
            for index, domain in enumerate(examples.dataset_ids)
            if domain in included_domains
        ],
        dtype=np.int64,
    )
    count = int(indices.size)
    result = ObservedValueExamples(
        outer_domain=examples.outer_domain,
        role=role,
        specimen_ids=tuple(examples.specimen_ids[index] for index in indices),
        dataset_ids=tuple(examples.dataset_ids[index] for index in indices),
        initial_embeddings=_readonly(
            examples.initial_embeddings[indices], dtype="<f8", shape=(count, 512)
        ),
        current_predictions=_readonly(
            examples.current_predictions[indices], dtype="<f8", shape=(count,)
        ),
        candidate_features=_readonly(
            examples.candidate_features[indices], dtype="<f8", shape=(count, 64, 8)
        ),
        initial_used_budgets=_readonly(
            examples.initial_used_budgets[indices], dtype="<f8", shape=(count,)
        ),
        mechanical_values=_readonly(
            examples.mechanical_values[indices], dtype="<f8", shape=(count, 64)
        ),
        candidate_costs=_readonly(
            examples.candidate_costs[indices], dtype="<i8", shape=(count, 64)
        ),
        teacher_predictor_state_sha256=tuple(
            examples.teacher_predictor_state_sha256[index] for index in indices
        ),
        candidate_bank_state_sha256=examples.candidate_bank_state_sha256,
        observed_feature_state_sha256=examples.observed_feature_state_sha256,
        state_sha256="",
    )
    object.__setattr__(result, "state_sha256", _examples_state(result))
    return result


__all__ = [
    "ObservedCandidateFeatureBank",
    "ObservedValueExamples",
    "StudentInputs",
    "build_observed_candidate_feature_bank",
    "build_outer_observability_examples",
    "load_observed_candidate_feature_bank",
    "save_observed_candidate_feature_bank",
    "subset_observed_examples",
]
