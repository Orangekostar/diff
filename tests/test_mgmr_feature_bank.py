from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.mgmr.feature_bank import (
    MGMRFeatureBankError,
    load_feature_bank,
    make_feature_bank,
    publish_feature_bank,
)
from cmc_bbdm.mgmr.feature_wavelet import directional_gap, dwt2_feature_maps


def _bank():
    rng = np.random.Generator(np.random.PCG64(20260822))
    full = rng.normal(size=(3, 256, 14, 14)).astype(np.float32)
    coarse = rng.normal(size=(3, 256, 14, 14)).astype(np.float32)
    global_features = rng.normal(size=(3, 512)).astype(np.float32)
    return make_feature_bank(
        specimen_ids=("c8-1", "q16-2", "c24-3"),
        dataset_ids=("74t7kcdgkr", "w68dtmpfyf", "xcmzfsbd9t"),
        full_global=global_features,
        full_layer3=full,
        coarse_layer3=coarse,
        config_sha256="1" * 64,
        source_sha256={"data": "2" * 64, "weights": "3" * 64},
        wavelet="db2",
        wavelet_mode="periodization",
    )


def test_feature_bank_derives_registered_components() -> None:
    bank = _bank()

    assert bank.specimen_ids == ("c8-1", "q16-2", "c24-3")
    assert bank.full_layer3.shape == (3, 256, 14, 14)
    assert bank.coarse_layer3.shape == bank.full_layer3.shape
    assert bank.full_global.shape == (3, 512)
    assert bank.coarse_gap.shape == (3, 256)
    assert bank.full_directional.shape == (3, 768)
    np.testing.assert_array_equal(
        bank.coarse_gap,
        bank.coarse_layer3.mean(axis=(-2, -1), dtype=np.float32),
    )
    expected_directional = directional_gap(
        dwt2_feature_maps(
            bank.full_layer3, wavelet="db2", mode="periodization"
        )
    )
    np.testing.assert_array_equal(bank.full_directional, expected_directional)
    for array in bank.arrays:
        assert array.dtype == np.float32
        assert array.flags.writeable is False


def test_feature_bank_round_trip_is_hash_bound(tmp_path: Path) -> None:
    original = _bank()
    output = tmp_path / "feature_bank"

    published = publish_feature_bank(output, original)
    loaded = load_feature_bank(
        output,
        expected_manifest_sha256=published.manifest_sha256,
        expected_specimen_ids=original.specimen_ids,
        expected_dataset_ids=original.dataset_ids,
        expected_config_sha256=original.config_sha256,
    )

    assert loaded.state_sha256 == original.state_sha256
    assert published.state_sha256 == original.state_sha256
    assert set(published.files) == {
        "full_global.npy",
        "full_layer3.npy",
        "coarse_layer3.npy",
        "coarse_gap.npy",
        "full_directional.npy",
    }
    for expected, actual in zip(original.arrays, loaded.arrays, strict=True):
        np.testing.assert_array_equal(expected, actual)


def test_feature_bank_rejects_specimen_reordering(tmp_path: Path) -> None:
    original = _bank()
    output = tmp_path / "feature_bank"
    published = publish_feature_bank(output, original)

    with pytest.raises(MGMRFeatureBankError, match="specimen order"):
        load_feature_bank(
            output,
            expected_manifest_sha256=published.manifest_sha256,
            expected_specimen_ids=tuple(reversed(original.specimen_ids)),
            expected_dataset_ids=original.dataset_ids,
            expected_config_sha256=original.config_sha256,
        )


def test_feature_bank_rejects_file_tampering(tmp_path: Path) -> None:
    original = _bank()
    output = tmp_path / "feature_bank"
    published = publish_feature_bank(output, original)
    target = output / "coarse_gap.npy"
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(MGMRFeatureBankError, match="file SHA-256"):
        load_feature_bank(
            output,
            expected_manifest_sha256=published.manifest_sha256,
            expected_specimen_ids=original.specimen_ids,
            expected_dataset_ids=original.dataset_ids,
            expected_config_sha256=original.config_sha256,
        )
