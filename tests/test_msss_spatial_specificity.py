from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.cpb_spatial.controls import patch_shuffle_rgb
from cmc_bbdm.msss.gaussian_scale import gaussian_scale
from cmc_bbdm.msss.scale_evaluator import evaluate_axis
from cmc_bbdm.msss.scale_features import ScaleCondition, ScaleFeatureBank
from cmc_bbdm.msss.spatial_specificity import (
    ShuffledFeatureBank,
    apply_post_scale_patch_shuffle,
    compute_specificity,
    evaluate_spatial_specificity,
    specificity_gate,
)
from cmc_bbdm.msss.statistics import common_stratified_bootstrap


def test_spatial_specificity_requires_four_strictly_positive_domains() -> None:
    passing = specificity_gate((0.02, 0.01, 0.03, 0.01, -0.01, 0.0))
    failing = specificity_gate((0.02, 0.01, 0.03, 0.0, -0.01, -0.02))

    assert passing.status == "PASS"
    assert passing.positive_domains == 4
    assert failing.status == "FAIL"
    assert failing.positive_domains == 3


def test_specificity_is_shuffle_error_minus_regular_error() -> None:
    result = compute_specificity(
        regular_domain_mae=(0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
        shuffled_domain_mae=(0.12, 0.19, 0.34, 0.45, 0.50, 0.70),
    )

    np.testing.assert_allclose(
        result.domain_effects, (0.02, -0.01, 0.04, 0.05, 0.0, 0.10), atol=1e-15
    )
    assert result.estimate == pytest.approx(np.mean(result.domain_effects))
    assert result.positive_domains == 4
    assert result.status == "PASS"


def test_patch_shuffle_is_applied_after_scale_transform() -> None:
    generator = np.random.Generator(np.random.PCG64(31))
    image = generator.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)
    scaled, _ = gaussian_scale(image, sigma_px=2.0)
    actual, actual_record = apply_post_scale_patch_shuffle(
        scaled,
        specimen_id="s1",
        dataset_id="d1",
        seed=20260831,
    )
    expected, expected_record = patch_shuffle_rgb(
        scaled,
        specimen_id="s1",
        dataset_id="d1",
        seed=20260831,
        grid=(8, 8),
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual_record == expected_record


def test_common_bootstrap_uses_synchronized_within_group_specimen_draws() -> None:
    groups = ("a", "a", "b", "b")
    first = np.asarray([0.0, 2.0, 10.0, 14.0])
    result = common_stratified_bootstrap(
        {"first": first, "double": first * 2.0},
        groups=groups,
        group_order=("a", "b"),
        seed=7,
        resamples=2000,
        quantiles=(0.025, 0.975),
    )

    assert result.effects["first"].estimate == pytest.approx(6.5)
    assert result.effects["double"].estimate == pytest.approx(13.0)
    assert result.effects["double"].low == pytest.approx(
        2.0 * result.effects["first"].low
    )
    assert result.effects["double"].high == pytest.approx(
        2.0 * result.effects["first"].high
    )
    assert len(result.draws_sha256) == 64


def test_spatial_evaluation_uses_regular_scale_selection_and_all_seeds() -> None:
    generator = np.random.Generator(np.random.PCG64(37))
    groups = tuple(group for group in "abcdef" for _ in range(6))
    rows = len(groups)
    latent = np.linspace(-1.0, 1.0, rows)
    targets = 0.65 + 0.08 * latent
    metadata = np.zeros((rows, 13), dtype=np.float64)
    metadata[:, 0] = latent
    conditions = tuple(
        ScaleCondition(
            condition_id=f"sampling:density={value}",
            axis="sampling",
            value=value,
            coarse_rank=rank,
            primary_eligible=True,
            is_full_identity=rank == 0,
        )
        for rank, value in enumerate((1.0, 0.25))
    )
    base = generator.normal(scale=0.05, size=(rows, 512))
    base[:, 0] = latent
    features = {item.condition_id: base.copy() for item in conditions}
    specimen_ids = tuple(f"s{index}" for index in range(rows))
    bank = ScaleFeatureBank.issue(
        conditions=conditions,
        specimen_ids=specimen_ids,
        dataset_ids=groups,
        features=features,
        transform_state_sha256={item.condition_id: "3" * 64 for item in conditions},
        encoder_provenance={"encoder": "synthetic", "frozen": True},
    )
    regular = evaluate_axis(
        bank,
        targets=targets,
        metadata13=metadata,
        axis="sampling",
        pca_dimensions=(2,),
    )
    selected_ids = tuple(
        dict.fromkeys(item.selected_condition_id for item in regular.scale_selections)
    )
    seeds = (20260831, 20260901, 20260902)
    shuffled = ShuffledFeatureBank.issue(
        base_condition_ids=selected_ids,
        seeds=seeds,
        specimen_ids=specimen_ids,
        dataset_ids=groups,
        features={
            (condition_id, seed): features[condition_id]
            for condition_id in selected_ids
            for seed in seeds
        },
        transform_state_sha256={
            (condition_id, seed): "4" * 64
            for condition_id in selected_ids
            for seed in seeds
        },
        encoder_provenance=bank.encoder_provenance,
    )
    result = evaluate_spatial_specificity(
        regular,
        regular_bank=bank,
        shuffled_bank=shuffled,
        targets=targets,
        metadata13=metadata,
        pca_dimensions=(2,),
    )

    assert len(result.predictions) == rows * len(seeds)
    assert result.result.status == "FAIL"
    np.testing.assert_allclose(result.result.domain_effects, 0.0, atol=1e-15)
    assert tuple(dict.fromkeys(row.base_condition_id for row in result.predictions)) == selected_ids
