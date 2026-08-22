"""Registered scale conditions and one-pass frozen feature extraction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from cmc_bbdm.cpb_v3.config import load_config
from cmc_bbdm.cpb_v3.embeddings import encode_resnet18

from .authority import MSSSAuthority
from .gaussian_scale import gaussian_scale
from .protocol import MSSSProtocol
from .sampling_scale import reconstruct_sampling_scale
from .wavelet_scale import wavelet_scale


class MSSSFeatureError(ValueError):
    """Raised when a scale condition or feature bank is incomplete."""


@dataclass(frozen=True, slots=True)
class ScaleCondition:
    condition_id: str
    axis: str
    value: float
    coarse_rank: int
    primary_eligible: bool
    is_full_identity: bool
    wavelet: str | None = None
    level: int | None = None
    mode: str | None = None


@dataclass(frozen=True, slots=True)
class MaterializedCondition:
    condition: ScaleCondition
    images: tuple[np.ndarray, ...]
    records: tuple[object, ...]
    output_sha256: tuple[str, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class EncodedCondition:
    features: np.ndarray
    sha256: str
    provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ScaleFeatureBank:
    conditions: tuple[ScaleCondition, ...]
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    features: Mapping[str, np.ndarray]
    feature_sha256: Mapping[str, str]
    transform_state_sha256: Mapping[str, str]
    encoder_provenance: Mapping[str, object]
    state_sha256: str

    @classmethod
    def issue(
        cls,
        *,
        conditions: Sequence[ScaleCondition],
        specimen_ids: Sequence[str],
        dataset_ids: Sequence[str],
        features: Mapping[str, object],
        transform_state_sha256: Mapping[str, str],
        encoder_provenance: Mapping[str, object],
    ) -> ScaleFeatureBank:
        registry = tuple(conditions)
        specimens = _identities(specimen_ids, "specimen IDs", unique=True)
        datasets = _identities(dataset_ids, "dataset IDs", unique=False)
        if (
            not registry
            or any(type(item) is not ScaleCondition for item in registry)
            or len({item.condition_id for item in registry}) != len(registry)
            or len(datasets) != len(specimens)
        ):
            raise MSSSFeatureError("feature-bank registry is invalid")
        expected = tuple(item.condition_id for item in registry)
        if set(features) != set(expected) or set(transform_state_sha256) != set(expected):
            raise MSSSFeatureError("feature condition mapping is incomplete")
        frozen_features: dict[str, np.ndarray] = {}
        feature_hashes: dict[str, str] = {}
        transform_hashes: dict[str, str] = {}
        for condition_id in expected:
            array = _feature_array(features[condition_id], rows=len(specimens))
            frozen_features[condition_id] = array
            feature_hashes[condition_id] = _array_sha256(array)
            transform_hashes[condition_id] = _sha256_text(
                transform_state_sha256[condition_id], "transform state"
            )
        provenance = _freeze_mapping(encoder_provenance)
        state_payload = {
            "conditions": [
                (
                    item.condition_id,
                    item.axis,
                    item.value,
                    item.coarse_rank,
                    item.primary_eligible,
                    item.is_full_identity,
                    item.wavelet,
                    item.level,
                    item.mode,
                )
                for item in registry
            ],
            "specimen_ids": specimens,
            "dataset_ids": datasets,
            "feature_sha256": feature_hashes,
            "transform_state_sha256": transform_hashes,
            "encoder_provenance": _jsonable(provenance),
        }
        state = hashlib.sha256(
            json.dumps(
                state_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        return cls(
            conditions=registry,
            specimen_ids=specimens,
            dataset_ids=datasets,
            features=MappingProxyType(frozen_features),
            feature_sha256=MappingProxyType(feature_hashes),
            transform_state_sha256=MappingProxyType(transform_hashes),
            encoder_provenance=provenance,
            state_sha256=state,
        )


@dataclass(frozen=True, slots=True)
class ScaleFeatureBuild:
    bank: ScaleFeatureBank
    materializations: Mapping[str, MaterializedCondition]


def _token(value: float) -> str:
    return format(float(value), ".10g")


def build_condition_registry(protocol: MSSSProtocol) -> tuple[ScaleCondition, ...]:
    """Return the frozen primary and sensitivity condition order."""

    if type(protocol) is not MSSSProtocol:
        raise MSSSFeatureError("issued MSSSProtocol is required")
    conditions: list[ScaleCondition] = []
    for rank, density in enumerate(protocol.sampling_densities):
        conditions.append(
            ScaleCondition(
                condition_id=f"sampling:density={_token(density)}",
                axis="sampling",
                value=density,
                coarse_rank=rank,
                primary_eligible=True,
                is_full_identity=density == 1.0,
            )
        )
    for rank, sigma in enumerate(protocol.gaussian_sigmas):
        conditions.append(
            ScaleCondition(
                condition_id=f"gaussian:sigma={_token(sigma)}",
                axis="gaussian",
                value=sigma,
                coarse_rank=rank,
                primary_eligible=True,
                is_full_identity=sigma == 0.0,
            )
        )
    for level in protocol.wavelet_levels:
        conditions.append(
            ScaleCondition(
                condition_id=f"wavelet:db2:low_only:level={level}",
                axis="wavelet",
                value=float(level),
                coarse_rank=level,
                primary_eligible=True,
                is_full_identity=level == 0,
                wavelet="db2",
                level=level,
                mode="low_only",
            )
        )
    for family in ("haar", "db4"):
        for level in protocol.wavelet_levels[1:]:
            conditions.append(
                ScaleCondition(
                    condition_id=f"wavelet:{family}:low_only:level={level}",
                    axis="wavelet",
                    value=float(level),
                    coarse_rank=level,
                    primary_eligible=False,
                    is_full_identity=False,
                    wavelet=family,
                    level=level,
                    mode="low_only",
                )
            )
    for family in protocol.wavelet_families:
        for level in protocol.wavelet_levels[1:]:
            conditions.append(
                ScaleCondition(
                    condition_id=f"wavelet:{family}:low_plus_boundary_details:level={level}",
                    axis="wavelet",
                    value=float(level),
                    coarse_rank=level,
                    primary_eligible=False,
                    is_full_identity=False,
                    wavelet=family,
                    level=level,
                    mode="low_plus_boundary_details",
                )
            )
    if len(conditions) != 37 or len({item.condition_id for item in conditions}) != 37:
        raise MSSSFeatureError("condition registry construction changed")
    return tuple(conditions)


def _identities(
    value: Sequence[str], label: str, *, unique: bool
) -> tuple[str, ...]:
    result = tuple(value)
    if (
        not result
        or any(type(item) is not str or not item or item.strip() != item for item in result)
        or (unique and len(set(result)) != len(result))
    ):
        raise MSSSFeatureError(f"{label} are invalid")
    return result


def _readonly_uint8(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.uint8)
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).reshape(array.shape)
    output.setflags(write=False)
    return output


def _raw_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def materialize_condition(
    images: Sequence[np.ndarray],
    *,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    condition: ScaleCondition,
) -> MaterializedCondition:
    """Materialize one complete condition before any frozen preprocessing."""

    source_images = tuple(images)
    specimens = _identities(specimen_ids, "specimen IDs", unique=True)
    datasets = _identities(dataset_ids, "dataset IDs", unique=False)
    if (
        type(condition) is not ScaleCondition
        or len(source_images) != len(specimens)
        or len(datasets) != len(specimens)
    ):
        raise MSSSFeatureError("condition input roster is incomplete")
    outputs: list[np.ndarray] = []
    records: list[object] = []
    hashes: list[str] = []
    for image, specimen, dataset in zip(
        source_images, specimens, datasets, strict=True
    ):
        if condition.axis == "sampling":
            output, record = reconstruct_sampling_scale(
                image,
                specimen_id=specimen,
                dataset_id=dataset,
                requested_density=condition.value,
            )
        elif condition.axis == "gaussian":
            output, record = gaussian_scale(image, sigma_px=condition.value)
        elif condition.axis == "wavelet":
            if condition.wavelet is None or condition.level is None or condition.mode is None:
                raise MSSSFeatureError("wavelet condition is incomplete")
            output, record = wavelet_scale(
                image,
                wavelet=condition.wavelet,
                level=condition.level,
                mode=condition.mode,
            )
        else:
            raise MSSSFeatureError("condition axis is unsupported")
        snapshot = _readonly_uint8(output)
        outputs.append(snapshot)
        records.append(record)
        hashes.append(_raw_sha256(snapshot))
    state_payload = {
        "condition_id": condition.condition_id,
        "specimen_ids": specimens,
        "dataset_ids": datasets,
        "output_sha256": hashes,
    }
    state = hashlib.sha256(
        json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    return MaterializedCondition(
        condition=condition,
        images=tuple(outputs),
        records=tuple(records),
        output_sha256=tuple(hashes),
        state_sha256=state,
    )


def _feature_array(value: object, *, rows: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise MSSSFeatureError("condition features must be numeric") from error
    if array.shape != (rows, 512) or not np.all(np.isfinite(array)):
        raise MSSSFeatureError(f"condition features must have shape ({rows}, 512)")
    contiguous = np.ascontiguousarray(array, dtype="<f8")
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype="<f8").reshape(
        contiguous.shape
    )
    output.setflags(write=False)
    return output


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MSSSFeatureError("encoder provenance must be a mapping")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise MSSSFeatureError("encoder provenance keys must be strings")
        if isinstance(item, Mapping):
            frozen[key] = _freeze_mapping(item)
        elif isinstance(item, list):
            frozen[key] = tuple(item)
        else:
            frozen[key] = item
    return MappingProxyType(frozen)


def _sha256_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MSSSFeatureError(f"{label} is not SHA-256")
    return value


def encode_condition_images(
    encoder: object, images: Sequence[np.ndarray]
) -> EncodedCondition:
    """Encode one full condition with the already constructed frozen encoder."""

    image_tuple = tuple(images)
    encode = getattr(encoder, "encode", None)
    provenance_method = getattr(encoder, "provenance", None)
    if not image_tuple or not callable(encode) or not callable(provenance_method):
        raise MSSSFeatureError("frozen encoder interface is unavailable")
    try:
        features = _feature_array(encode(image_tuple), rows=len(image_tuple))
        provenance = _freeze_mapping(provenance_method())
    except MSSSFeatureError:
        raise
    except (TypeError, ValueError, RuntimeError) as error:
        raise MSSSFeatureError("condition encoding failed") from error
    return EncodedCondition(
        features=features,
        sha256=_array_sha256(features),
        provenance=provenance,
    )


def _protocol_source(protocol: MSSSProtocol, name: str) -> Path:
    matches = tuple(item.path for item in protocol.sources if item.name == name)
    if len(matches) != 1:
        raise MSSSFeatureError(f"protocol source is unavailable: {name}")
    return matches[0]


def load_frozen_encoder(
    protocol: MSSSProtocol, *, project_root: str | Path
) -> object:
    """Construct the registered frozen encoder once for an MSSS execution."""

    if type(protocol) is not MSSSProtocol:
        raise MSSSFeatureError("issued MSSSProtocol is required")
    root = Path(project_root).resolve(strict=True)
    config = load_config(_protocol_source(protocol, "p1_config"), project_root=root)
    return encode_resnet18(
        weight_path=config.frozen_encoder.weight_path.relative_to(root),
        project_root=root,
        device=protocol.device,
        batch_size=32,
    )


def _execution_indices(value: object, *, rows: int) -> np.ndarray:
    if value is None:
        return np.arange(rows, dtype=np.int64)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise MSSSFeatureError("execution indices must be integers") from error
    if (
        raw.ndim != 1
        or not len(raw)
        or raw.dtype.kind not in {"i", "u"}
        or np.any(raw < 0)
        or np.any(raw >= rows)
    ):
        raise MSSSFeatureError("execution indices are invalid")
    indices = np.asarray(raw, dtype=np.int64)
    if len(set(indices.tolist())) != len(indices):
        raise MSSSFeatureError("execution indices must be unique")
    return indices


def build_scale_feature_bank(
    protocol: MSSSProtocol,
    authority: MSSSAuthority,
    *,
    project_root: str | Path,
    encoder: object | None = None,
    indices: object | None = None,
    retain_materializations: bool = False,
) -> ScaleFeatureBuild:
    """Materialize and encode every registered S1 condition exactly once."""

    if type(protocol) is not MSSSProtocol or type(authority) is not MSSSAuthority:
        raise MSSSFeatureError("issued MSSS protocol and authority are required")
    root = Path(project_root).resolve(strict=True)
    if type(retain_materializations) is not bool:
        raise MSSSFeatureError("materialization retention flag must be boolean")
    selected = _execution_indices(indices, rows=authority.specimen_count)
    specimen_ids = tuple(authority.specimen_ids[int(index)] for index in selected)
    dataset_ids = tuple(authority.dataset_ids[int(index)] for index in selected)
    images = tuple(authority.registered_inputs.images[int(index)] for index in selected)
    active_encoder = encoder
    if active_encoder is None:
        active_encoder = load_frozen_encoder(protocol, project_root=root)
    registry = build_condition_registry(protocol)
    feature_values: dict[str, np.ndarray] = {}
    transform_states: dict[str, str] = {}
    materializations: dict[str, MaterializedCondition] = {}
    provenance: Mapping[str, object] | None = None
    authority_aliases = {
        "sampling:density=1": authority.full_features[selected],
        "sampling:density=0.5": authority.bilinear50_features[selected],
        "sampling:density=0.25": authority.bilinear25_features[selected],
        "gaussian:sigma=0": authority.full_features[selected],
        "wavelet:db2:low_only:level=0": authority.full_features[selected],
    }
    for condition in registry:
        materialized = materialize_condition(
            images,
            specimen_ids=specimen_ids,
            dataset_ids=dataset_ids,
            condition=condition,
        )
        if retain_materializations:
            materializations[condition.condition_id] = materialized
        transform_states[condition.condition_id] = materialized.state_sha256
        alias = authority_aliases.get(condition.condition_id)
        if alias is not None:
            feature_values[condition.condition_id] = alias
            continue
        encoded = encode_condition_images(active_encoder, materialized.images)
        feature_values[condition.condition_id] = encoded.features
        if provenance is None:
            provenance = encoded.provenance
        elif _jsonable(provenance) != _jsonable(encoded.provenance):
            raise MSSSFeatureError("encoder provenance changed between conditions")
    if provenance is None:
        provenance_method = getattr(active_encoder, "provenance", None)
        if not callable(provenance_method):
            raise MSSSFeatureError("encoder provenance is unavailable")
        provenance = _freeze_mapping(provenance_method())
    bank = ScaleFeatureBank.issue(
        conditions=registry,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        features=feature_values,
        transform_state_sha256=transform_states,
        encoder_provenance=provenance,
    )
    return ScaleFeatureBuild(
        bank=bank,
        materializations=MappingProxyType(materializations),
    )


def rematerialize_conditions(
    bank: ScaleFeatureBank,
    *,
    images: Sequence[np.ndarray],
    condition_ids: Sequence[str],
) -> ScaleFeatureBuild:
    """Recreate selected images and verify them against the frozen transform hashes."""

    if type(bank) is not ScaleFeatureBank:
        raise MSSSFeatureError("issued ScaleFeatureBank is required")
    image_tuple = tuple(images)
    if len(image_tuple) != len(bank.specimen_ids):
        raise MSSSFeatureError("rematerialization roster is incomplete")
    requested = tuple(dict.fromkeys(condition_ids))
    registry = {item.condition_id: item for item in bank.conditions}
    if not requested or any(item not in registry for item in requested):
        raise MSSSFeatureError("rematerialization condition is not registered")
    materializations: dict[str, MaterializedCondition] = {}
    for condition_id in requested:
        materialized = materialize_condition(
            image_tuple,
            specimen_ids=bank.specimen_ids,
            dataset_ids=bank.dataset_ids,
            condition=registry[condition_id],
        )
        if materialized.state_sha256 != bank.transform_state_sha256[condition_id]:
            raise MSSSFeatureError("rematerialized transform state changed")
        materializations[condition_id] = materialized
    return ScaleFeatureBuild(
        bank=bank,
        materializations=MappingProxyType(materializations),
    )


__all__ = [
    "EncodedCondition",
    "MSSSFeatureError",
    "MaterializedCondition",
    "ScaleCondition",
    "ScaleFeatureBank",
    "ScaleFeatureBuild",
    "build_condition_registry",
    "build_scale_feature_bank",
    "encode_condition_images",
    "load_frozen_encoder",
    "materialize_condition",
    "rematerialize_conditions",
]
