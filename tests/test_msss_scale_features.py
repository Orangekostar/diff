from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.msss.protocol import load_protocol
from cmc_bbdm.msss.scale_features import (
    MSSSFeatureError,
    ScaleFeatureBank,
    build_condition_registry,
    encode_condition_images,
    materialize_condition,
)

ROOT = Path(__file__).resolve().parents[1]


class _FakeEncoder:
    def encode(self, images: tuple[np.ndarray, ...]) -> np.ndarray:
        output = np.zeros((len(images), 512), dtype=np.float64)
        for index, image in enumerate(images):
            output[index, 0] = float(np.mean(image))
            output[index, 1] = float(np.std(image))
        return output

    def provenance(self) -> dict[str, object]:
        return {"encoder": "fake", "frozen": True}


def _images() -> tuple[np.ndarray, ...]:
    first = np.arange(17 * 19 * 3, dtype=np.int64).reshape(17, 19, 3) % 256
    second = np.flip(first, axis=1)
    return (np.asarray(first, dtype=np.uint8), np.asarray(second, dtype=np.uint8))


def test_condition_registry_freezes_primary_and_sensitivity_conditions() -> None:
    protocol = load_protocol(ROOT / "paper_v3/configs/msss.yaml", project_root=ROOT)
    registry = build_condition_registry(protocol)

    assert len(registry) == 37
    assert tuple(item.axis for item in registry[:9]) == ("sampling",) * 9
    assert tuple(item.axis for item in registry[9:18]) == ("gaussian",) * 9
    assert sum(item.primary_eligible for item in registry if item.axis == "wavelet") == 4
    assert tuple(item.condition_id for item in registry if item.is_full_identity) == (
        "sampling:density=1",
        "gaussian:sigma=0",
        "wavelet:db2:low_only:level=0",
    )


def test_materialization_uses_registered_transform_and_roster() -> None:
    protocol = load_protocol(ROOT / "paper_v3/configs/msss.yaml", project_root=ROOT)
    condition = next(
        item
        for item in build_condition_registry(protocol)
        if item.condition_id == "sampling:density=0.25"
    )
    result = materialize_condition(
        _images(),
        specimen_ids=("s1", "s2"),
        dataset_ids=("d1", "d2"),
        condition=condition,
    )

    assert len(result.images) == 2
    assert len(result.output_sha256) == 2
    assert len(set(result.output_sha256)) == 2
    assert all(not image.flags.writeable for image in result.images)
    assert result.condition == condition


def test_condition_encoder_returns_immutable_512d_features() -> None:
    encoded = encode_condition_images(_FakeEncoder(), _images())

    assert encoded.features.shape == (2, 512)
    assert encoded.features.dtype == np.float64
    assert not encoded.features.flags.writeable
    assert len(encoded.sha256) == 64
    assert encoded.provenance == {"encoder": "fake", "frozen": True}


def test_feature_bank_requires_complete_registered_mapping() -> None:
    protocol = load_protocol(ROOT / "paper_v3/configs/msss.yaml", project_root=ROOT)
    registry = build_condition_registry(protocol)
    features = {
        condition.condition_id: np.zeros((2, 512), dtype=np.float64)
        for condition in registry
    }
    bank = ScaleFeatureBank.issue(
        conditions=registry,
        specimen_ids=("s1", "s2"),
        dataset_ids=("d1", "d2"),
        features=features,
        transform_state_sha256={item.condition_id: "0" * 64 for item in registry},
        encoder_provenance={"encoder": "fake", "frozen": True},
    )

    assert tuple(bank.features) == tuple(item.condition_id for item in registry)
    assert all(not value.flags.writeable for value in bank.features.values())
    assert len(bank.state_sha256) == 64

    missing = dict(features)
    missing.pop(registry[-1].condition_id)
    with pytest.raises(MSSSFeatureError, match="condition mapping"):
        ScaleFeatureBank.issue(
            conditions=registry,
            specimen_ids=("s1", "s2"),
            dataset_ids=("d1", "d2"),
            features=missing,
            transform_state_sha256={item.condition_id: "0" * 64 for item in registry},
            encoder_provenance={"encoder": "fake", "frozen": True},
        )
