"""Frozen ResNet feature extraction and D8 marginalization operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmc_bbdm.cpb_v3.embeddings import (
    EMBEDDING_BATCH_SIZE,
    RESNET18_WEIGHTS_RELATIVE_PATH,
    FrozenResNet18Encoder,
    _deterministic_runtime,
    encode_resnet18,
    preprocess_full_field,
)

_LAYERS = frozenset({"global", "layer3", "multi_layer"})
_FEATURE_AGGREGATIONS = frozenset({"mean", "median", "trimmed", "mean_variance"})
_PREDICTION_AGGREGATIONS = frozenset(
    {"mean", "median", "trimmed", "morphology_weighted"}
)


def _readonly(value: object, *, dtype: np.dtype) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("feature values must be real")
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("feature values must be numeric") from error
    if not np.all(np.isfinite(array)):
        raise ValueError("feature values must be finite")
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(
        contiguous.shape
    )
    output.setflags(write=False)
    return output


def _image(value: object) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.uint8)
        or value.ndim != 3
        or value.shape[2] != 3
        or min(value.shape[:2]) < 1
    ):
        raise ValueError("D8 encoder input must be a uint8 RGB image")
    contiguous = np.ascontiguousarray(value)
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.uint8).reshape(
        contiguous.shape
    )
    output.setflags(write=False)
    return output


def _variant_grid(
    variants: tuple[tuple[np.ndarray, ...], ...],
) -> tuple[tuple[np.ndarray, ...], int]:
    if not isinstance(variants, tuple) or not variants:
        raise ValueError("variants must be a nonempty specimen tuple")
    if any(not isinstance(specimen, tuple) or not specimen for specimen in variants):
        raise ValueError("every specimen must contain a nonempty variant tuple")
    count = len(variants[0])
    if count not in (1, 2, 4, 8, 16) or any(
        len(specimen) != count for specimen in variants
    ):
        raise ValueError("all specimens must share a registered K")
    return tuple(
        tuple(_image(value) for value in specimen) for specimen in variants
    ), count


@dataclass(frozen=True, slots=True)
class D8FrozenEncoder:
    """D8 feature views over the exact frozen P1 ResNet18 authority."""

    base_encoder: FrozenResNet18Encoder

    def __post_init__(self) -> None:
        if type(self.base_encoder) is not FrozenResNet18Encoder:
            raise TypeError("D8 requires the exact frozen ResNet18 encoder")
        self.base_encoder._validate_encoder_state()

    def _intermediate(
        self, images: tuple[np.ndarray, ...], *, layer: str
    ) -> np.ndarray:
        encoder = self.base_encoder
        encoder._validate_encoder_state()
        torch = encoder.torch
        model = encoder._model
        outputs: list[np.ndarray] = []
        try:
            with _deterministic_runtime(torch), torch.inference_mode():
                for start in range(0, len(images), EMBEDDING_BATCH_SIZE):
                    values = images[start : start + EMBEDDING_BATCH_SIZE]
                    tensors = [
                        torch.from_numpy(preprocess_full_field(value, size=224))
                        for value in values
                    ]
                    batch = torch.stack(tensors).to(encoder.device, non_blocking=False)
                    value = model.conv1(batch)
                    value = model.bn1(value)
                    value = model.relu(value)
                    value = model.maxpool(value)
                    layer1 = model.layer1(value)
                    layer2 = model.layer2(layer1)
                    layer3 = model.layer3(layer2)
                    if layer == "layer3":
                        output = torch.mean(layer3, dim=(-2, -1))
                    else:
                        layer4 = model.layer4(layer3)
                        output = torch.cat(
                            tuple(
                                torch.mean(item, dim=(-2, -1))
                                for item in (layer1, layer2, layer3, layer4)
                            ),
                            dim=1,
                        )
                    outputs.append(output.detach().to("cpu").numpy())
        except Exception as error:
            raise ValueError("D8 intermediate ResNet extraction failed") from error
        encoder._validate_encoder_state()
        expected = 256 if layer == "layer3" else 960
        result = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        if result.shape != (len(images), expected):
            raise ValueError("D8 intermediate feature shape changed")
        return result

    def encode(
        self,
        variants: tuple[tuple[np.ndarray, ...], ...],
        *,
        layer: str,
    ) -> np.ndarray:
        """Return immutable ``(specimen, K, feature)`` frozen embeddings."""

        grid, count = _variant_grid(variants)
        if type(layer) is not str or layer not in _LAYERS:
            raise ValueError("feature layer is not registered")
        images = tuple(value for specimen in grid for value in specimen)
        if layer == "global":
            chunks = [
                self.base_encoder.encode(images[start : start + 276])
                for start in range(0, len(images), 276)
            ]
            flat = np.concatenate(chunks, axis=0)
        else:
            flat = self._intermediate(images, layer=layer)
        return _readonly(
            flat.reshape(len(grid), count, flat.shape[1]),
            dtype=np.dtype(np.float32),
        )


def create_d8_frozen_encoder(
    *, project_root: str | Path, device: str = "cuda:0"
) -> D8FrozenEncoder:
    """Load the exact P1 encoder for D8 feature marginalization."""

    result = encode_resnet18(
        weight_path=RESNET18_WEIGHTS_RELATIVE_PATH,
        project_root=project_root,
        device=device,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    if type(result) is not FrozenResNet18Encoder:
        raise TypeError("frozen ResNet18 construction failed")
    return D8FrozenEncoder(result)


def aggregate_features(features: np.ndarray, *, method: str) -> np.ndarray:
    """Aggregate only the variant axis while preserving specimen identity."""

    values = _readonly(features, dtype=np.dtype(np.float64))
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("variant features must have shape (specimen, K, feature)")
    if type(method) is not str or method not in _FEATURE_AGGREGATIONS:
        raise ValueError("feature aggregation is not registered")
    if method == "mean":
        result = np.mean(values, axis=1, dtype=np.float64)
    elif method == "median":
        result = np.median(values, axis=1)
    elif method == "trimmed":
        rows: list[np.ndarray] = []
        keep = max(1, math.ceil(values.shape[1] * 0.8))
        for specimen in values:
            center = np.median(specimen, axis=0)
            distances = np.linalg.norm(specimen - center, axis=1)
            indices = np.argsort(distances, kind="stable")[:keep]
            rows.append(np.mean(specimen[indices], axis=0, dtype=np.float64))
        result = np.stack(rows)
    else:
        mean = np.mean(values, axis=1, dtype=np.float64)
        variance = np.var(values, axis=1, dtype=np.float64)
        result = np.concatenate((mean, np.log1p(variance)), axis=1)
    return _readonly(result, dtype=np.dtype(np.float64))


def aggregate_predictions(
    predictions: np.ndarray,
    *,
    method: str,
    morphology_distances: np.ndarray | None = None,
    beta: float | None = None,
) -> np.ndarray:
    """Return one CAI prediction per specimen from the registered K axis."""

    values = _readonly(predictions, dtype=np.dtype(np.float64))
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("predictions must have shape (specimen, K)")
    if type(method) is not str or method not in _PREDICTION_AGGREGATIONS:
        raise ValueError("prediction aggregation is not registered")
    if method == "mean":
        result = np.mean(values, axis=1, dtype=np.float64)
    elif method == "median":
        result = np.median(values, axis=1)
    elif method == "trimmed":
        ordered = np.sort(values, axis=1)
        trim = math.floor(values.shape[1] * 0.1)
        selected = ordered[:, trim : values.shape[1] - trim] if trim else ordered
        result = np.mean(selected, axis=1, dtype=np.float64)
    else:
        distances = _readonly(morphology_distances, dtype=np.dtype(np.float64))
        if (
            distances.shape != values.shape
            or np.any(distances < 0.0)
            or type(beta) not in (int, float)
            or not math.isfinite(beta)
            or not 0.1 <= float(beta) <= 100.0
        ):
            raise ValueError("morphology-weighted aggregation inputs are invalid")
        logits = -float(beta) * distances
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=1, keepdims=True)
        result = np.sum(weights * values, axis=1, dtype=np.float64)
    return _readonly(result, dtype=np.dtype(np.float64))


def variant_training_weights(specimen_ids: tuple[str, ...]) -> np.ndarray:
    """Assign each replicated variant reciprocal within-specimen weight."""

    if (
        not isinstance(specimen_ids, tuple)
        or not specimen_ids
        or any(type(item) is not str or not item for item in specimen_ids)
    ):
        raise ValueError("specimen IDs must be a nonempty tuple")
    counts: dict[str, int] = {}
    for specimen in specimen_ids:
        counts[specimen] = counts.get(specimen, 0) + 1
    result = np.asarray(
        [1.0 / counts[specimen] for specimen in specimen_ids], dtype=np.float64
    )
    for specimen, count in counts.items():
        mask = np.asarray([item == specimen for item in specimen_ids])
        result[mask] /= np.sum(result[mask], dtype=np.float64)
        if np.sum(mask) != count:
            raise ValueError("specimen weight roster changed")
    return _readonly(result, dtype=np.dtype(np.float64))


__all__ = [
    "D8FrozenEncoder",
    "aggregate_features",
    "aggregate_predictions",
    "create_d8_frozen_encoder",
    "variant_training_weights",
]
