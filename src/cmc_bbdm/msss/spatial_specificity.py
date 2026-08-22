"""Post-scale registered spatial destruction and MSSS specificity metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from cmc_bbdm.cpb_spatial.controls import ControlRecord, patch_shuffle_rgb

from .scale_evaluator import AxisEvaluation, evaluate_axis
from .scale_features import (
    ScaleCondition,
    ScaleFeatureBank,
    ScaleFeatureBuild,
    encode_condition_images,
)


class SpatialSpecificityError(ValueError):
    """Raised when spatial-specificity evidence is invalid."""


@dataclass(frozen=True, slots=True)
class SpecificityResult:
    domain_effects: tuple[float, ...]
    estimate: float
    positive_domains: int
    status: str


@dataclass(frozen=True, slots=True)
class ShuffledFeatureBank:
    base_condition_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    features: Mapping[tuple[str, int], np.ndarray]
    feature_sha256: Mapping[tuple[str, int], str]
    transform_state_sha256: Mapping[tuple[str, int], str]
    encoder_provenance: Mapping[str, object]
    state_sha256: str

    @classmethod
    def issue(
        cls,
        *,
        base_condition_ids: Sequence[str],
        seeds: Sequence[int],
        specimen_ids: Sequence[str],
        dataset_ids: Sequence[str],
        features: Mapping[tuple[str, int], object],
        transform_state_sha256: Mapping[tuple[str, int], str],
        encoder_provenance: Mapping[str, object],
    ) -> ShuffledFeatureBank:
        bases = tuple(base_condition_ids)
        seed_tuple = tuple(seeds)
        specimens = tuple(specimen_ids)
        datasets = tuple(dataset_ids)
        if (
            not bases
            or len(set(bases)) != len(bases)
            or any(type(value) is not str or not value for value in bases)
            or not seed_tuple
            or len(set(seed_tuple)) != len(seed_tuple)
            or any(type(value) is not int or isinstance(value, bool) or value < 0 for value in seed_tuple)
            or not specimens
            or len(set(specimens)) != len(specimens)
            or len(datasets) != len(specimens)
        ):
            raise SpatialSpecificityError("shuffled feature authority is invalid")
        expected = {(base, seed) for base in bases for seed in seed_tuple}
        if set(features) != expected or set(transform_state_sha256) != expected:
            raise SpatialSpecificityError("shuffled feature condition mapping is incomplete")
        frozen_features: dict[tuple[str, int], np.ndarray] = {}
        feature_hashes: dict[tuple[str, int], str] = {}
        transform_hashes: dict[tuple[str, int], str] = {}
        for key in sorted(expected):
            try:
                array = np.asarray(features[key], dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as error:
                raise SpatialSpecificityError("shuffled features must be numeric") from error
            if array.shape != (len(specimens), 512) or not np.all(np.isfinite(array)):
                raise SpatialSpecificityError("shuffled feature shape is invalid")
            contiguous = np.ascontiguousarray(array, dtype="<f8")
            snapshot = np.frombuffer(
                contiguous.tobytes(order="C"), dtype="<f8"
            ).reshape(contiguous.shape)
            snapshot.setflags(write=False)
            frozen_features[key] = snapshot
            feature_hashes[key] = hashlib.sha256(
                snapshot.tobytes(order="C")
            ).hexdigest()
            state = transform_state_sha256[key]
            if (
                type(state) is not str
                or len(state) != 64
                or any(character not in "0123456789abcdef" for character in state)
            ):
                raise SpatialSpecificityError("shuffle transform state is invalid")
            transform_hashes[key] = state
        provenance = MappingProxyType(dict(encoder_provenance))
        state_payload = {
            "bases": bases,
            "seeds": seed_tuple,
            "specimens": specimens,
            "datasets": datasets,
            "features": {f"{base}|{seed}": feature_hashes[(base, seed)] for base, seed in sorted(expected)},
            "transforms": {f"{base}|{seed}": transform_hashes[(base, seed)] for base, seed in sorted(expected)},
        }
        state_sha256 = hashlib.sha256(
            json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        return cls(
            base_condition_ids=bases,
            seeds=seed_tuple,
            specimen_ids=specimens,
            dataset_ids=datasets,
            features=MappingProxyType(frozen_features),
            feature_sha256=MappingProxyType(feature_hashes),
            transform_state_sha256=MappingProxyType(transform_hashes),
            encoder_provenance=provenance,
            state_sha256=state_sha256,
        )


@dataclass(frozen=True, slots=True)
class SpatialPrediction:
    axis: str
    base_condition_id: str
    seed: int
    specimen_id: str
    dataset_id: str
    target: float
    regular_prediction: float
    shuffled_prediction: float
    regular_absolute_error: float
    shuffled_absolute_error: float
    selected_pca_dimension: int


@dataclass(frozen=True, slots=True)
class SpatialSpecificityEvaluation:
    axis: str
    predictions: tuple[SpatialPrediction, ...]
    regular_domain_mae: tuple[float, ...]
    shuffled_domain_mae: tuple[float, ...]
    result: SpecificityResult
    state_sha256: str


def specificity_gate(
    domain_effects: Sequence[float], *, minimum_positive_domains: int = 4
) -> SpecificityResult:
    try:
        effects = tuple(float(value) for value in domain_effects)
    except (TypeError, ValueError, OverflowError) as error:
        raise SpatialSpecificityError("specificity effects must be numeric") from error
    if (
        len(effects) != 6
        or any(not math.isfinite(value) for value in effects)
        or type(minimum_positive_domains) is not int
        or not 1 <= minimum_positive_domains <= 6
    ):
        raise SpatialSpecificityError("specificity effects are invalid")
    estimate = float(math.fsum(effects) / len(effects))
    positive = sum(value > 0.0 for value in effects)
    status = (
        "PASS"
        if estimate > 0.0 and positive >= minimum_positive_domains
        else "FAIL"
    )
    return SpecificityResult(
        domain_effects=effects,
        estimate=estimate,
        positive_domains=positive,
        status=status,
    )


def compute_specificity(
    *,
    regular_domain_mae: Sequence[float],
    shuffled_domain_mae: Sequence[float],
    minimum_positive_domains: int = 4,
) -> SpecificityResult:
    try:
        regular = tuple(float(value) for value in regular_domain_mae)
        shuffled = tuple(float(value) for value in shuffled_domain_mae)
    except (TypeError, ValueError, OverflowError) as error:
        raise SpatialSpecificityError("specificity MAE values must be numeric") from error
    if (
        len(regular) != 6
        or len(shuffled) != 6
        or any(not math.isfinite(value) or value < 0.0 for value in (*regular, *shuffled))
    ):
        raise SpatialSpecificityError("specificity MAE values are invalid")
    return specificity_gate(
        tuple(control - reference for reference, control in zip(regular, shuffled, strict=True)),
        minimum_positive_domains=minimum_positive_domains,
    )


def apply_post_scale_patch_shuffle(
    scaled_image: np.ndarray,
    *,
    specimen_id: str,
    dataset_id: str,
    seed: int,
) -> tuple[np.ndarray, ControlRecord]:
    """Apply the exact P3 primary control after a scale transform."""

    try:
        return patch_shuffle_rgb(
            scaled_image,
            specimen_id=specimen_id,
            dataset_id=dataset_id,
            seed=seed,
            grid=(8, 8),
        )
    except ValueError as error:
        raise SpatialSpecificityError("post-scale P3 patch shuffle failed") from error


def build_shuffled_feature_bank(
    feature_build: ScaleFeatureBuild,
    *,
    selected_condition_ids: Sequence[str],
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    seeds: Sequence[int],
    encoder: object,
) -> ShuffledFeatureBank:
    """Encode exact P3 controls for the union of source-selected scales."""

    if type(feature_build) is not ScaleFeatureBuild:
        raise SpatialSpecificityError("issued ScaleFeatureBuild is required")
    bases = tuple(dict.fromkeys(selected_condition_ids))
    specimens = tuple(specimen_ids)
    datasets = tuple(dataset_ids)
    seed_tuple = tuple(seeds)
    feature_values: dict[tuple[str, int], np.ndarray] = {}
    transform_states: dict[tuple[str, int], str] = {}
    for base in bases:
        materialized = feature_build.materializations.get(base)
        if materialized is None:
            raise SpatialSpecificityError("selected scale materialization is unavailable")
        for seed in seed_tuple:
            shuffled_images: list[np.ndarray] = []
            output_hashes: list[str] = []
            for image, specimen, dataset in zip(
                materialized.images, specimens, datasets, strict=True
            ):
                shuffled, record = apply_post_scale_patch_shuffle(
                    image,
                    specimen_id=specimen,
                    dataset_id=dataset,
                    seed=seed,
                )
                shuffled_images.append(shuffled)
                output_hashes.append(record.output_sha256)
            transform_states[(base, seed)] = hashlib.sha256(
                json.dumps(
                    {
                        "base": base,
                        "seed": seed,
                        "specimens": specimens,
                        "outputs": output_hashes,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest()
            encoded = encode_condition_images(encoder, tuple(shuffled_images))
            feature_values[(base, seed)] = encoded.features
    return ShuffledFeatureBank.issue(
        base_condition_ids=bases,
        seeds=seed_tuple,
        specimen_ids=specimens,
        dataset_ids=datasets,
        features=feature_values,
        transform_state_sha256=transform_states,
        encoder_provenance=feature_build.bank.encoder_provenance,
    )


def evaluate_spatial_specificity(
    regular: AxisEvaluation,
    *,
    regular_bank: ScaleFeatureBank,
    shuffled_bank: ShuffledFeatureBank,
    targets: object,
    metadata13: object,
    pca_dimensions: Sequence[int] = (8, 16, 32),
) -> SpatialSpecificityEvaluation:
    """Evaluate shuffled predictors at the already selected regular scales."""

    if type(regular) is not AxisEvaluation or type(regular_bank) is not ScaleFeatureBank:
        raise SpatialSpecificityError("regular axis authority is invalid")
    if type(shuffled_bank) is not ShuffledFeatureBank:
        raise SpatialSpecificityError("issued ShuffledFeatureBank is required")
    if (
        regular_bank.specimen_ids != shuffled_bank.specimen_ids
        or regular_bank.dataset_ids != shuffled_bank.dataset_ids
        or regular.group_order != tuple(dict.fromkeys(regular_bank.dataset_ids))
    ):
        raise SpatialSpecificityError("regular and shuffled rosters differ")
    regular_by_specimen = {row.specimen_id: row for row in regular.selected_predictions}
    if set(regular_by_specimen) != set(regular_bank.specimen_ids):
        raise SpatialSpecificityError("regular selected predictions are incomplete")
    selected_by_group = {
        row.outer_group: row.selected_condition_id for row in regular.scale_selections
    }
    required_bases = tuple(dict.fromkeys(selected_by_group.values()))
    if any(base not in shuffled_bank.base_condition_ids for base in required_bases):
        raise SpatialSpecificityError("a selected scale lacks shuffled features")

    evaluated: dict[tuple[str, int], dict[str, object]] = {}
    for base in required_bases:
        for seed in shuffled_bank.seeds:
            condition_id = f"shuffle:{base}:seed={seed}"
            condition = ScaleCondition(
                condition_id=condition_id,
                axis=regular.axis,
                value=0.0,
                coarse_rank=0,
                primary_eligible=True,
                is_full_identity=True,
            )
            temporary = ScaleFeatureBank.issue(
                conditions=(condition,),
                specimen_ids=regular_bank.specimen_ids,
                dataset_ids=regular_bank.dataset_ids,
                features={condition_id: shuffled_bank.features[(base, seed)]},
                transform_state_sha256={
                    condition_id: shuffled_bank.transform_state_sha256[(base, seed)]
                },
                encoder_provenance=shuffled_bank.encoder_provenance,
            )
            result = evaluate_axis(
                temporary,
                targets=targets,
                metadata13=metadata13,
                axis=regular.axis,
                pca_dimensions=pca_dimensions,
            )
            evaluated[(base, seed)] = {
                row.specimen_id: row for row in result.candidate_predictions
            }

    prediction_rows: list[SpatialPrediction] = []
    for specimen, dataset in zip(
        regular_bank.specimen_ids, regular_bank.dataset_ids, strict=True
    ):
        base = selected_by_group[dataset]
        reference = regular_by_specimen[specimen]
        for seed in shuffled_bank.seeds:
            shuffled = evaluated[(base, seed)][specimen]
            prediction_rows.append(
                SpatialPrediction(
                    axis=regular.axis,
                    base_condition_id=base,
                    seed=seed,
                    specimen_id=specimen,
                    dataset_id=dataset,
                    target=reference.target,
                    regular_prediction=reference.prediction,
                    shuffled_prediction=float(shuffled.prediction),
                    regular_absolute_error=reference.absolute_error,
                    shuffled_absolute_error=float(shuffled.absolute_error),
                    selected_pca_dimension=int(shuffled.selected_pca_dimension),
                )
            )
    regular_domain: list[float] = []
    shuffled_domain: list[float] = []
    for group in regular.group_order:
        group_rows = tuple(row for row in prediction_rows if row.dataset_id == group)
        regular_domain.append(
            float(np.mean([row.regular_absolute_error for row in group_rows[:: len(shuffled_bank.seeds)]]))
        )
        shuffled_domain.append(
            float(np.mean([row.shuffled_absolute_error for row in group_rows]))
        )
    result = compute_specificity(
        regular_domain_mae=regular_domain,
        shuffled_domain_mae=shuffled_domain,
    )
    state_sha256 = hashlib.sha256(
        json.dumps(
            {
                "axis": regular.axis,
                "regular": regular_domain,
                "shuffled": shuffled_domain,
                "effects": result.domain_effects,
                "predictions": [
                    (
                        row.base_condition_id,
                        row.seed,
                        row.specimen_id,
                        row.regular_prediction,
                        row.shuffled_prediction,
                    )
                    for row in prediction_rows
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return SpatialSpecificityEvaluation(
        axis=regular.axis,
        predictions=tuple(prediction_rows),
        regular_domain_mae=tuple(regular_domain),
        shuffled_domain_mae=tuple(shuffled_domain),
        result=result,
        state_sha256=state_sha256,
    )


__all__ = [
    "ShuffledFeatureBank",
    "SpatialPrediction",
    "SpatialSpecificityError",
    "SpatialSpecificityEvaluation",
    "SpecificityResult",
    "apply_post_scale_patch_shuffle",
    "build_shuffled_feature_bank",
    "compute_specificity",
    "evaluate_spatial_specificity",
    "specificity_gate",
]
