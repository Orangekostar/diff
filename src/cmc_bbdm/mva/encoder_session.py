"""Validated frozen-encoder session for sequential MVA oracle queries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from cmc_bbdm.cpb_v3.embeddings import (
    _ENCODER_EXECUTION_LOCK,
    EMBEDDING_DIMENSION,
    FeatureValidationError,
    FrozenResNet18Encoder,
    _deterministic_runtime,
    _load_image_item,
    preprocess_full_field,
)


class MVAEncoderSession:
    """Amortize integrity checks while preserving the registered forward path."""

    def __init__(self, encoder: FrozenResNet18Encoder) -> None:
        if not isinstance(encoder, FrozenResNet18Encoder):
            raise TypeError("a frozen ResNet18 encoder is required")
        encoder._validate_encoder_state()
        self._encoder = encoder

    def validate(self) -> None:
        """Revalidate weights, graph, runtime, and device before publication."""

        self._encoder._validate_encoder_state()

    def encode(self, images: Iterable[object]) -> np.ndarray:
        """Run the same preprocessing and frozen forward graph within the session."""

        values = list(images)
        if not values:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        encoder = self._encoder
        with _ENCODER_EXECUTION_LOCK:
            tensors = [
                encoder.torch.from_numpy(
                    preprocess_full_field(_load_image_item(item), size=224)
                )
                for item in values
            ]
            outputs: list[np.ndarray] = []
            try:
                with (
                    _deterministic_runtime(encoder.torch),
                    encoder.torch.inference_mode(),
                ):
                    for start in range(0, len(tensors), encoder.batch_size):
                        batch = encoder.torch.stack(
                            tensors[start : start + encoder.batch_size]
                        ).to(encoder.device, non_blocking=False)
                        output = encoder._model(batch).detach().to("cpu").numpy()
                        output = np.asarray(output, dtype=np.float32)
                        if output.ndim != 2 or output.shape[1] != EMBEDDING_DIMENSION:
                            raise FeatureValidationError(
                                "ResNet18 output is not 512-dimensional"
                            )
                        outputs.append(output)
            except FeatureValidationError:
                raise
            except Exception as error:
                raise FeatureValidationError(
                    "MVA ResNet18 embedding inference failed"
                ) from error
        combined = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        if combined.shape != (len(values), EMBEDDING_DIMENSION) or not np.all(
            np.isfinite(combined)
        ):
            raise FeatureValidationError("MVA ResNet18 output is incomplete")
        return combined


__all__ = ["MVAEncoderSession"]
