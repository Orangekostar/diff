from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.cpb_v3.embeddings import (
    FeatureValidationError,
    FrozenResNet18Encoder,
    encode_resnet18,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path("paper_v3/assets/resnet18-f37072fd.pth")


@pytest.fixture(scope="module")
def encoder() -> FrozenResNet18Encoder:
    result = encode_resnet18(
        weight_path=WEIGHTS,
        project_root=ROOT,
        device="cuda:0",
        batch_size=32,
    )
    assert isinstance(result, FrozenResNet18Encoder)
    return result


@pytest.fixture(scope="module")
def images() -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros((31, 29, 3), dtype=np.uint8)
    first[4:18, 3:21, 0] = 220
    first[9:26, 11:27, 1] = 80
    second = np.flip(first, axis=1).copy()
    return first, second


def test_spatial_encoder_returns_registered_layer_shapes(
    encoder: FrozenResNet18Encoder, images: tuple[np.ndarray, np.ndarray]
) -> None:
    layer2 = encoder.encode_spatial(images, layer="layer2")
    layer3 = encoder.encode_spatial(images, layer="layer3")

    assert layer2.shape == (2, 128, 28, 28)
    assert layer3.shape == (2, 256, 14, 14)
    assert layer2.dtype == np.float32
    assert layer3.dtype == np.float32
    assert np.all(np.isfinite(layer2))
    assert np.all(np.isfinite(layer3))
    assert layer2.flags.writeable is False
    assert layer3.flags.writeable is False


def test_spatial_encoder_is_deterministic_and_ordered(
    encoder: FrozenResNet18Encoder, images: tuple[np.ndarray, np.ndarray]
) -> None:
    forward = encoder.encode_spatial(images, layer="layer3")
    repeated = encoder.encode_spatial(images, layer="layer3")
    reversed_output = encoder.encode_spatial(tuple(reversed(images)), layer="layer3")

    np.testing.assert_array_equal(forward, repeated)
    np.testing.assert_array_equal(forward[0], reversed_output[1])
    np.testing.assert_array_equal(forward[1], reversed_output[0])
    assert not np.array_equal(forward[0], forward[1])


def test_layer4_gap_matches_existing_global_encoder(
    encoder: FrozenResNet18Encoder, images: tuple[np.ndarray, np.ndarray]
) -> None:
    before = encoder.provenance()
    layer4 = encoder.encode_spatial(images, layer="layer4")
    global_features = encoder.encode(images)
    after = encoder.provenance()

    np.testing.assert_allclose(
        layer4.mean(axis=(-2, -1)), global_features, rtol=0.0, atol=1.0e-6
    )
    assert before["weights_sha256"] == after["weights_sha256"]
    assert before["model_state_sha256"] == after["model_state_sha256"]
    assert before["architecture_sha256"] == after["architecture_sha256"]


def test_spatial_encoder_rejects_unregistered_layer(
    encoder: FrozenResNet18Encoder, images: tuple[np.ndarray, np.ndarray]
) -> None:
    with pytest.raises(FeatureValidationError, match="spatial layer"):
        encoder.encode_spatial(images, layer="layer5")
