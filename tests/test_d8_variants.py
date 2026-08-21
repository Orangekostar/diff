from __future__ import annotations

import numpy as np
import pytest

import cmc_bbdm.cpb_diffusion_marginalization.variants as variants_module
from cmc_bbdm.cpb_cscan_morphology import CscanMorphologyRule
from cmc_bbdm.cpb_diffusion_marginalization.variants import (
    MorphologyThresholds,
    _relative,
    build_variant,
    build_variant_batch,
    evaluate_candidate_acceptance,
)
from cmc_bbdm.cpb_physical_descriptors import PhysicalCalibration


def test_zero_reference_relative_deviation_is_conservative_and_finite() -> None:
    assert _relative(0.0, 0.0) == 0.0
    assert _relative(1.0, 0.0) == 1.0
    with pytest.raises(ValueError, match="nonnegative"):
        _relative(-1.0, 0.0)


@pytest.fixture
def source() -> np.ndarray:
    rgb = np.full((64, 64, 3), 16, dtype=np.uint8)
    rows, columns = np.indices((64, 64))
    disk = (rows - 31.5) ** 2 + (columns - 31.5) ** 2 <= 12.0**2
    rgb[disk] = np.asarray((230, 80, 30), dtype=np.uint8)
    return (rgb.astype(np.float32) / np.float32(127.5) - 1.0).transpose(2, 0, 1)


@pytest.fixture
def native_source(source: np.ndarray) -> np.ndarray:
    return np.rint((source.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)


@pytest.fixture
def rule() -> CscanMorphologyRule:
    return CscanMorphologyRule(
        name="d8-test-rule",
        background_border_fraction=0.10,
        background_distance_min=20.0,
        chroma_min=20.0,
        white_channel_min=245,
        central_radius_mm=30.0,
        closing_radius_mm=0.0,
        minimum_component_area_mm2=1.0,
    )


@pytest.fixture
def calibration() -> PhysicalCalibration:
    return PhysicalCalibration(
        dataset_id="test-domain",
        field_width_mm=75.0,
        field_height_mm=75.0,
        calibration_basis="test-only-registered-frame",
        evidence_path="test-only",
        evidence_sha256="a" * 64,
    )


@pytest.fixture
def thresholds() -> MorphologyThresholds:
    return MorphologyThresholds(
        area_relative_deviation=0.10,
        width_relative_deviation=0.10,
        height_relative_deviation=0.10,
        centroid_shift_mm=2.0,
        low_frequency_correlation_minimum=0.99,
        radial_spearman_minimum=0.98,
        low_frequency_sigma_pixels=2.0,
        radial_profile_bins=16,
    )


def test_zero_alpha_preserves_every_morphology_measure(
    source: np.ndarray,
    native_source: np.ndarray,
    rule: CscanMorphologyRule,
    calibration: PhysicalCalibration,
    thresholds: MorphologyThresholds,
) -> None:
    result = build_variant(
        source,
        np.ones_like(source),
        native_source=native_source,
        alpha=0.0,
        rule=rule,
        calibration=calibration,
        thresholds=thresholds,
    )
    assert result.accepted
    assert result.area_deviation == 0.0
    assert result.width_deviation == 0.0
    assert result.height_deviation == 0.0
    assert result.centroid_shift_mm == 0.0
    assert result.low_frequency_correlation == 1.0
    assert result.radial_profile_correlation == 1.0
    assert result.failed_conditions == ()
    assert result.variant.flags.writeable is False
    np.testing.assert_array_equal(result.variant, source)
    np.testing.assert_array_equal(result.encoder_image, native_source)


def test_zero_alpha_accepts_a_registered_source_with_no_damage_footprint(
    rule: CscanMorphologyRule,
    calibration: PhysicalCalibration,
    thresholds: MorphologyThresholds,
) -> None:
    native = np.full((64, 64, 3), 16, dtype=np.uint8)
    source = (native.astype(np.float32) / np.float32(127.5) - 1.0).transpose(
        2, 0, 1
    )

    result = build_variant(
        source,
        np.zeros_like(source),
        native_source=native,
        alpha=0.0,
        rule=rule,
        calibration=calibration,
        thresholds=thresholds,
    )

    assert result.accepted
    assert result.area_deviation == 0.0
    assert result.width_deviation == 0.0
    assert result.height_deviation == 0.0


def test_variant_rejects_large_low_frequency_change(
    source: np.ndarray,
    native_source: np.ndarray,
    rule: CscanMorphologyRule,
    calibration: PhysicalCalibration,
    thresholds: MorphologyThresholds,
) -> None:
    result = build_variant(
        source,
        -2.0 * source,
        native_source=native_source,
        alpha=1.0,
        rule=rule,
        calibration=calibration,
        thresholds=thresholds,
    )
    assert not result.accepted
    assert "low_frequency_correlation" in result.failed_conditions


def test_variant_batch_falls_back_to_raw_after_registered_proposal_cap(
    source: np.ndarray,
    native_source: np.ndarray,
    rule: CscanMorphologyRule,
    calibration: PhysicalCalibration,
    thresholds: MorphologyThresholds,
) -> None:
    rejected = tuple(-2.0 * source for _ in range(32))
    batch = build_variant_batch(
        source,
        rejected,
        native_source=native_source,
        alpha=1.0,
        requested_count=4,
        rule=rule,
        calibration=calibration,
        thresholds=thresholds,
        maximum_proposals=32,
    )
    assert batch.proposal_count == 32
    assert batch.accepted_count == 0
    assert batch.fallback_count == 4
    assert len(batch.variants) == 4
    assert all(np.array_equal(value, source) for value in batch.variants)
    assert all(
        np.array_equal(value, native_source) for value in batch.encoder_images
    )
    assert all(not value.flags.writeable for value in batch.variants)


def test_variant_batch_reuses_exact_source_morphology(
    source: np.ndarray,
    native_source: np.ndarray,
    rule: CscanMorphologyRule,
    calibration: PhysicalCalibration,
    thresholds: MorphologyThresholds,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    extract = variants_module.extract_damage_footprint

    def counted_extract(*args, **kwargs):
        nonlocal calls
        calls += 1
        return extract(*args, **kwargs)

    monkeypatch.setattr(
        variants_module,
        "extract_damage_footprint",
        counted_extract,
    )
    residuals = tuple(
        np.full_like(source, (index + 1) * 0.01, dtype=np.float32)
        for index in range(4)
    )
    batch = build_variant_batch(
        source,
        residuals,
        native_source=native_source,
        alpha=0.1,
        requested_count=4,
        rule=rule,
        calibration=calibration,
        thresholds=thresholds,
    )

    assert calls == 1 + batch.proposal_count
    reference = tuple(
        build_variant(
            source,
            residual,
            native_source=native_source,
            alpha=0.1,
            rule=rule,
            calibration=calibration,
            thresholds=thresholds,
        )
        for residual in residuals
    )
    assert tuple(record.state_sha256 for record in batch.records) == tuple(
        record.state_sha256 for record in reference
    )
    for actual, expected in zip(batch.records, reference, strict=True):
        np.testing.assert_array_equal(actual.variant, expected.variant)
        np.testing.assert_array_equal(actual.encoder_image, expected.encoder_image)


def test_candidate_acceptance_requires_overall_and_every_domain_threshold() -> None:
    accepted = {
        "domain-a": (8, 10),
        "domain-b": (7, 10),
    }
    passed = evaluate_candidate_acceptance(
        accepted,
        minimum_overall=0.70,
        minimum_domain=0.60,
    )
    assert passed.eligible
    assert passed.overall_rate == 0.75

    failed = evaluate_candidate_acceptance(
        {"domain-a": (9, 10), "domain-b": (5, 10)},
        minimum_overall=0.70,
        minimum_domain=0.60,
    )
    assert not failed.eligible
    assert failed.failed_domains == ("domain-b",)


def test_variant_rejects_out_of_contract_inputs(
    source: np.ndarray,
    native_source: np.ndarray,
    rule: CscanMorphologyRule,
    calibration: PhysicalCalibration,
    thresholds: MorphologyThresholds,
) -> None:
    with pytest.raises(ValueError):
        build_variant(
            source,
            np.ones_like(source),
            native_source=native_source,
            alpha=1.01,
            rule=rule,
            calibration=calibration,
            thresholds=thresholds,
        )
    with pytest.raises(ValueError):
        build_variant_batch(
            source,
            (np.ones_like(source),),
            native_source=native_source,
            alpha=0.1,
            requested_count=17,
            rule=rule,
            calibration=calibration,
            thresholds=thresholds,
            maximum_proposals=32,
        )

    changed_native = native_source.copy()
    changed_native[0, 0, 0] += 1
    with pytest.raises(ValueError, match="64x64 source"):
        build_variant(
            source,
            np.ones_like(source),
            native_source=changed_native,
            alpha=0.0,
            rule=rule,
            calibration=calibration,
            thresholds=thresholds,
        )
