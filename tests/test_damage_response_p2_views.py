from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.damage_response.p2_features import P2FeatureAuthority
from cmc_bbdm.damage_response.p2_views import (
    P2_VIEW_FIELDS,
    P2ViewError,
    fit_p2_preprocessor,
    transform_p2_view,
    validate_p2_view,
    view_feature_names,
)


def _authority(*, query_sentinel: bool = False) -> P2FeatureAuthority:
    rng = np.random.default_rng(11)
    specimen_ids = tuple(f"s-{index:02d}" for index in range(15))
    domain_ids = tuple(f"d{index // 5 + 1}" for index in range(15))
    laminate_types = tuple(
        "cross_ply" if index % 2 == 0 else "quasi_isotropic"
        for index in range(15)
    )
    impactors = ("hemia", "hemib", "hemic", "coni60", "coni120") * 3
    ply = np.asarray([8 + index % 3 * 8 for index in range(15)])
    width = np.linspace(50.0, 64.0, 15)
    thickness = np.linspace(1.5, 2.9, 15)
    surface = rng.normal(size=(15, 21))
    scalar = np.abs(rng.normal(size=(15, 3)))
    embedding = rng.normal(size=(15, 512)).astype(np.float32)
    total_energy = np.linspace(4.0, 18.0, 15)
    if query_sentinel:
        query = np.asarray([domain == "d3" for domain in domain_ids])
        ply[query] = 1_000_000
        width[query] = 2_000_000.0
        thickness[query] = 3_000_000.0
        surface[query] = 4_000_000.0
        scalar[query] = 5_000_000.0
        embedding[query] = 6_000_000.0
        total_energy[query] = 7_000_000.0
    return P2FeatureAuthority(
        specimen_ids=specimen_ids,
        domain_ids=domain_ids,
        laminate_types=laminate_types,
        ply_counts=ply,
        widths_mm=width,
        thicknesses_mm=thickness,
        surface_profile_stats=surface,
        scalar_damage=scalar,
        full_cscan_embedding=embedding,
        privileged_total_energy_j=total_energy,
        privileged_impactors=impactors,
        full_embedding_view="FULL",
        encoder_sha256="a" * 64,
        embedding_state_sha256="b" * 64,
        source_sha256={
            "feature_bank": "1" * 64,
            "feature_cache": "2" * 64,
            "physical_descriptors": "3" * 64,
            "provenance_specimens": "4" * 64,
            "lvi_workbook": "5" * 64,
        },
    )


def test_registered_view_membership_is_exact() -> None:
    assert P2_VIEW_FIELDS == {
        "F0": ("laminate_type", "ply_count", "width_mm", "thickness_mm"),
        "F1": ("F0", "surface_profile_stats21"),
        "F2": ("F0", "projected_damage_area", "damage_height", "damage_width"),
        "F3": ("F0", "full_cscan_embedding512"),
        "F4": ("F0", "surface_profile_stats21", "full_cscan_embedding512"),
        "F5": (
            "F4",
            "privileged_total_impact_energy_j",
            "privileged_impactor",
        ),
    }
    for name, fields in P2_VIEW_FIELDS.items():
        assert validate_p2_view(name, fields) == fields


@pytest.mark.parametrize("view_name", tuple(P2_VIEW_FIELDS))
@pytest.mark.parametrize(
    "forbidden",
    (
        "published_cai_strength_mpa",
        "true_cai_strength",
        "raw_cai_trace",
        "extension_peak_mm",
        "normalized_prepeak_auc",
        "post_cai_image",
    ),
)
def test_outcomes_and_response_fields_cannot_enter_p2_views(
    view_name: str, forbidden: str
) -> None:
    with pytest.raises(P2ViewError):
        validate_p2_view(view_name, (*P2_VIEW_FIELDS[view_name], forbidden))


@pytest.mark.parametrize(
    ("view_name", "pca_dimension", "expected_dimension"),
    (
        ("F0", None, 5),
        ("F1", None, 26),
        ("F2", None, 8),
        ("F3", 2, 7),
        ("F4", 2, 28),
        ("F5", 2, 35),
    ),
)
def test_fold_preprocessor_has_fixed_names_and_dimensions(
    view_name: str, pca_dimension: int | None, expected_dimension: int
) -> None:
    authority = _authority()
    fit = np.flatnonzero(np.asarray(authority.domain_ids) != "d3")

    state = fit_p2_preprocessor(
        authority, view_name, fit, pca_dimension=pca_dimension
    )
    matrix = transform_p2_view(authority, state, np.arange(15))

    assert state.feature_names == view_feature_names(view_name, pca_dimension)
    assert len(state.feature_names) == expected_dimension
    assert matrix.shape == (15, expected_dimension)
    assert matrix.flags.writeable is False
    assert np.all(np.isfinite(matrix))
    assert state.fit_specimen_ids == tuple(authority.specimen_ids[index] for index in fit)


def test_held_out_feature_sentinels_do_not_change_fitted_state() -> None:
    reference = _authority()
    sentinel = _authority(query_sentinel=True)
    fit = np.flatnonzero(np.asarray(reference.domain_ids) != "d3")

    first = fit_p2_preprocessor(reference, "F5", fit, pca_dimension=2)
    second = fit_p2_preprocessor(sentinel, "F5", fit, pca_dimension=2)

    assert first.state_sha256 == second.state_sha256
    assert first.fit_specimen_ids == second.fit_specimen_ids
    np.testing.assert_array_equal(first.imputer_statistics, second.imputer_statistics)
    np.testing.assert_array_equal(first.numeric_means, second.numeric_means)
    np.testing.assert_array_equal(first.numeric_scales, second.numeric_scales)
    np.testing.assert_array_equal(first.pca_mean, second.pca_mean)
    np.testing.assert_array_equal(first.pca_components, second.pca_components)


def test_pca_is_fit_only_to_embedding_block_and_has_canonical_sign() -> None:
    authority = _authority()
    fit = np.arange(10)

    state = fit_p2_preprocessor(authority, "F4", fit, pca_dimension=2)
    centered = authority.full_cscan_embedding[fit].astype(np.float64) - np.mean(
        authority.full_cscan_embedding[fit], axis=0, dtype=np.float64
    )
    _left, _singular, right = np.linalg.svd(centered, full_matrices=False)
    expected = right[:2].copy()
    for row in expected:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0

    np.testing.assert_allclose(state.pca_components, expected, rtol=0.0, atol=0.0)
    for row in state.pca_components:
        assert row[int(np.argmax(np.abs(row)))] >= 0.0


def test_categories_are_fixed_one_hot_and_not_scaled() -> None:
    authority = _authority()
    fit = np.arange(10)
    state = fit_p2_preprocessor(authority, "F5", fit, pca_dimension=2)

    matrix = transform_p2_view(authority, state, [0, 1])
    category_start = len(state.numeric_feature_names)
    categories = matrix[:, category_start:]

    assert state.categorical_feature_names == (
        "laminate_type=cross_ply",
        "laminate_type=quasi_isotropic",
        "impactor=coni120",
        "impactor=coni60",
        "impactor=flat",
        "impactor=hemia",
        "impactor=hemib",
        "impactor=hemic",
    )
    np.testing.assert_array_equal(categories[0], [1, 0, 0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(categories[1], [0, 1, 0, 0, 0, 0, 1, 0])


@pytest.mark.parametrize(
    ("view_name", "pca_dimension"),
    (("F0", 2), ("F1", 2), ("F2", 2), ("F3", None), ("F4", None), ("F5", None)),
)
def test_pca_dimension_is_required_only_for_embedding_views(
    view_name: str, pca_dimension: int | None
) -> None:
    with pytest.raises(P2ViewError, match="PCA dimension"):
        fit_p2_preprocessor(
            _authority(), view_name, np.arange(10), pca_dimension=pca_dimension
        )


@pytest.mark.parametrize("indices", ([], [0, 0], [-1, 1], [0, 15]))
def test_fit_indices_fail_closed(indices: list[int]) -> None:
    with pytest.raises(P2ViewError, match="fit indices"):
        fit_p2_preprocessor(_authority(), "F0", indices, pca_dimension=None)
