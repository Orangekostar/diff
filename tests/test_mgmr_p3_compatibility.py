from __future__ import annotations

import numpy as np

from cmc_bbdm.cpb_spatial.controls import patch_shuffle_rgb
from cmc_bbdm.mgmr.m0_residual_audit import patch_shuffle_m0_images
from cmc_bbdm.mgmr.specificity_bank import (
    load_specificity_bank,
    make_specificity_bank,
    publish_specificity_bank,
)


def test_mgmr_p3_control_is_the_registered_8x8_primitive() -> None:
    image = np.arange(17 * 19 * 3, dtype=np.uint8).reshape(17, 19, 3)
    images = (image, np.flip(image, axis=0).copy())
    specimen_ids = ("s0", "s1")
    dataset_ids = ("d0", "d1")

    output, records = patch_shuffle_m0_images(
        images,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        seed=20260831,
    )

    assert len(output) == len(records) == 2
    for index in range(2):
        expected, expected_record = patch_shuffle_rgb(
            images[index],
            specimen_id=specimen_ids[index],
            dataset_id=dataset_ids[index],
            seed=20260831,
            grid=(8, 8),
        )
        np.testing.assert_array_equal(output[index], expected)
        assert records[index] == expected_record


def test_all_registered_seeds_reuse_the_same_coarse_residual_targets() -> None:
    from test_mgmr_residual_target_is_strict_oof import _problem

    values = _problem()
    from cmc_bbdm.mgmr.m0_residual_audit import audit_residual_arrays

    audit = audit_residual_arrays(
        specimen_ids=values[2],
        dataset_ids=values[1],
        domain_order=values[0],
        targets=values[7],
        metadata=values[3],
        full=values[4],
        coarse=values[5],
        boundary=values[6],
        coarse_outer_predictions=values[8],
        full_outer_predictions=values[9],
        shuffled_boundary=values[10],
        pca_dimensions=(2,),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )

    assert audit.specificity_seeds == (20260831, 20260901, 20260902)
    assert all(
        branch.source_residual_state_sha256
        == audit.coarse.source_residual_state_sha256
        for branch in audit.shuffles.values()
    )


def test_specificity_features_round_trip_under_seed_hashes(tmp_path) -> None:
    rng = np.random.Generator(np.random.PCG64(20260828))
    features = {
        seed: rng.normal(size=(3, 768)).astype(np.float32)
        for seed in (20260831, 20260901, 20260902)
    }
    bank = make_specificity_bank(
        specimen_ids=("s0", "s1", "s2"),
        dataset_ids=("d0", "d1", "d2"),
        directional=features,
        config_sha256="1" * 64,
        source_sha256={"feature_bank": "2" * 64},
        control_sha256={seed: str(index + 3) * 64 for index, seed in enumerate(features)},
    )
    output = tmp_path / "p3"

    publication = publish_specificity_bank(output, bank)
    loaded = load_specificity_bank(
        output,
        expected_manifest_sha256=publication.manifest_sha256,
        expected_specimen_ids=bank.specimen_ids,
        expected_dataset_ids=bank.dataset_ids,
        expected_config_sha256=bank.config_sha256,
    )

    assert loaded.state_sha256 == bank.state_sha256
    for seed in bank.seeds:
        np.testing.assert_array_equal(loaded.directional[seed], bank.directional[seed])
        assert loaded.directional[seed].flags.writeable is False
