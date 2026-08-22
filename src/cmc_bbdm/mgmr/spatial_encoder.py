"""FULL and P5-coarse spatial feature extraction for MGMR M0."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmc_bbdm.cpb_v3.embeddings import FrozenResNet18Encoder, encode_resnet18
from cmc_bbdm.msss.sampling_scale import (
    SamplingScaleRecord,
    reconstruct_sampling_scale,
)

from .authority import MGMRM0Authority
from .feature_bank import MGMRFeatureBank, make_feature_bank
from .protocol import MGMRProtocol


class MGMRSpatialEncoderError(ValueError):
    """Raised when ordered M0 spatial views cannot be constructed."""


@dataclass(frozen=True, slots=True)
class M0SpatialViews:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    full_layer3: np.ndarray
    coarse_layer3: np.ndarray
    sampling_records: tuple[SamplingScaleRecord, ...]


@dataclass(frozen=True, slots=True)
class M0FeatureExtraction:
    bank: MGMRFeatureBank
    sampling_records: tuple[SamplingScaleRecord, ...]
    encoder_provenance: dict[str, object]


def _ids(value: Sequence[str], *, length: int, label: str) -> tuple[str, ...]:
    result = tuple(value)
    if (
        len(result) != length
        or any(type(item) is not str or not item for item in result)
        or (label == "specimen IDs" and len(set(result)) != len(result))
    ):
        raise MGMRSpatialEncoderError(f"{label} do not align with images")
    return result


def encode_m0_views(
    encoder: FrozenResNet18Encoder,
    *,
    images: Sequence[np.ndarray],
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    coarse_density: float,
) -> M0SpatialViews:
    """Encode ordered FULL and exact P5-coarse crops with one frozen encoder."""

    if type(encoder) is not FrozenResNet18Encoder:
        raise MGMRSpatialEncoderError("issued frozen encoder is required")
    image_values = tuple(images)
    if not image_values:
        raise MGMRSpatialEncoderError("M0 images must not be empty")
    specimens = _ids(specimen_ids, length=len(image_values), label="specimen IDs")
    datasets = _ids(dataset_ids, length=len(image_values), label="dataset IDs")
    if coarse_density != 0.25:
        raise MGMRSpatialEncoderError("M0 coarse density must be 0.25")
    coarse_images: list[np.ndarray] = []
    records: list[SamplingScaleRecord] = []
    for image, specimen_id, dataset_id in zip(
        image_values, specimens, datasets, strict=True
    ):
        reconstructed, record = reconstruct_sampling_scale(
            image,
            specimen_id=specimen_id,
            dataset_id=dataset_id,
            requested_density=coarse_density,
        )
        coarse_images.append(reconstructed)
        records.append(record)
    full = encoder.encode_spatial(image_values, layer="layer3")
    coarse = encoder.encode_spatial(tuple(coarse_images), layer="layer3")
    expected = (len(image_values), 256, 14, 14)
    if full.shape != expected or coarse.shape != expected:
        raise MGMRSpatialEncoderError("M0 layer3 feature shapes changed")
    return M0SpatialViews(
        specimen_ids=specimens,
        dataset_ids=datasets,
        full_layer3=full,
        coarse_layer3=coarse,
        sampling_records=tuple(records),
    )


def extract_m0_feature_bank(
    protocol: MGMRProtocol,
    authority: MGMRM0Authority,
    *,
    project_root: str | Path,
    device: str,
    status_hook: Callable[[str], None] | None = None,
) -> M0FeatureExtraction:
    """Construct the production M0 feature bank without accessing targets."""

    if type(protocol) is not MGMRProtocol or type(authority) is not MGMRM0Authority:
        raise MGMRSpatialEncoderError("issued protocol and authority are required")
    if device != protocol.device:
        raise MGMRSpatialEncoderError("feature extraction device changed")
    root = Path(project_root).resolve(strict=True)
    notify = status_hook if status_hook is not None else lambda _message: None
    notify("loading frozen ResNet18")
    encoder = encode_resnet18(
        weight_path=protocol.sources["resnet_weights"].path,
        project_root=root,
        device=device,
        batch_size=protocol.batch_size,
    )
    if not isinstance(encoder, FrozenResNet18Encoder):
        raise MGMRSpatialEncoderError("frozen encoder construction failed")
    notify("encoding FULL and 25% layer3 maps")
    views = encode_m0_views(
        encoder,
        images=authority.images,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        coarse_density=protocol.coarse_density,
    )
    notify("deriving coarse and directional components")
    bank = make_feature_bank(
        specimen_ids=views.specimen_ids,
        dataset_ids=views.dataset_ids,
        full_global=authority.full_global,
        full_layer3=views.full_layer3,
        coarse_layer3=views.coarse_layer3,
        config_sha256=protocol.config_sha256,
        source_sha256={
            "authority": authority.state_sha256,
            "p1_config": protocol.sources["p1_config"].sha256,
            "p5_config": protocol.sources["p5_config"].sha256,
            "paired_feature_bank": authority.paired_feature_bank_sha256,
            "resnet_weights": protocol.sources["resnet_weights"].sha256,
        },
        wavelet=protocol.wavelet,
        wavelet_mode=protocol.wavelet_mode,
    )
    return M0FeatureExtraction(
        bank=bank,
        sampling_records=views.sampling_records,
        encoder_provenance=encoder.provenance(),
    )


__all__ = [
    "M0FeatureExtraction",
    "M0SpatialViews",
    "MGMRSpatialEncoderError",
    "encode_m0_views",
    "extract_m0_feature_bank",
]
