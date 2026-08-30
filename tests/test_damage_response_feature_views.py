from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cmc_bbdm.damage_response.feature_views import (
    FeatureViewError,
    FieldRole,
    field_role,
    fit_fold_local_design_encoder,
    validate_p1_redundancy_view,
)
from cmc_bbdm.damage_response.sources import DesignMetadata

REGISTERED_VIEWS = {
    "strength_only": ("published_cai_strength_mpa",),
    "strength_plus_design": (
        "published_cai_strength_mpa",
        "ply_count",
        "width_mm",
        "thickness_mm",
        "laminate_type",
        "impactor",
    ),
}


def _records() -> tuple[DesignMetadata, ...]:
    return (
        DesignMetadata("a-1", "domain-a", "cross_ply", 10, "coni120", 10.0, 1.0),
        DesignMetadata(
            "a-2", "domain-a", "quasi_isotropic", 20, "coni60", 20.0, 2.0
        ),
        DesignMetadata("b-1", "domain-b", "cross_ply", 30, "flat", 30.0, 3.0),
        DesignMetadata(
            "b-2", "domain-b", "quasi_isotropic", 40, "hemia", 40.0, 4.0
        ),
        DesignMetadata("c-1", "domain-c", "cross_ply", 1000, "hemib", 1000.0, 100.0),
        DesignMetadata(
            "c-2", "domain-c", "quasi_isotropic", 2000, "hemic", 2000.0, 200.0
        ),
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("specimen_id", FieldRole.IDENTITY),
        ("domain_id", FieldRole.SPLIT),
        ("extension_peak_mm", FieldRole.TARGET),
        ("slope_u20_u60_mpa_per_mm", FieldRole.TARGET),
        ("normalized_prepeak_auc", FieldRole.TARGET),
        ("published_cai_strength_mpa", FieldRole.REDUNDANCY_REFERENCE),
        ("ply_count", FieldRole.DEPLOYABLE_DESIGN),
        ("width_mm", FieldRole.DEPLOYABLE_DESIGN),
        ("thickness_mm", FieldRole.DEPLOYABLE_DESIGN),
        ("laminate_type", FieldRole.DEPLOYABLE_DESIGN),
        ("impactor", FieldRole.PRIVILEGED_DESIGN),
        ("true_cai_trace", FieldRole.FORBIDDEN),
        ("derived_response", FieldRole.FORBIDDEN),
        ("response_curve", FieldRole.FORBIDDEN),
        ("post_cai_image", FieldRole.FORBIDDEN),
    ),
)
def test_field_roles_are_exact(name: str, expected: FieldRole) -> None:
    assert field_role(name) is expected


def test_field_role_enum_names_and_values_are_frozen() -> None:
    expected = (
        "IDENTITY",
        "SPLIT",
        "TARGET",
        "REDUNDANCY_REFERENCE",
        "DEPLOYABLE_DESIGN",
        "PRIVILEGED_DESIGN",
        "FORBIDDEN",
    )
    assert tuple(role.name for role in FieldRole) == expected
    assert tuple(role.value for role in FieldRole) == expected


@pytest.mark.parametrize("view_name, expected", REGISTERED_VIEWS.items())
def test_registered_p1_redundancy_views_accept_exact_fields(
    view_name: str, expected: tuple[str, ...]
) -> None:
    assert validate_p1_redundancy_view(view_name, expected) == expected


@pytest.mark.parametrize(
    ("view_name", "fields"),
    (
        ("strength_only", ("published_cai_strength_mpa", "specimen_id")),
        ("strength_only", ()),
        (
            "strength_plus_design",
            REGISTERED_VIEWS["strength_plus_design"] + ("specimen_id",),
        ),
        ("strength_plus_design", REGISTERED_VIEWS["strength_plus_design"][:-1]),
        (
            "strength_plus_design",
            (
                "ply_count",
                "published_cai_strength_mpa",
                "width_mm",
                "thickness_mm",
                "laminate_type",
                "impactor",
            ),
        ),
    ),
)
def test_registered_p1_redundancy_views_reject_membership_or_order_drift(
    view_name: str, fields: tuple[str, ...]
) -> None:
    with pytest.raises(FeatureViewError):
        validate_p1_redundancy_view(view_name, fields)


def test_fold_local_encoder_uses_source_only_population_statistics() -> None:
    records = _records()
    encoder = fit_fold_local_design_encoder(records, "domain-c")
    source_numeric = np.asarray(
        [[record.ply_count, record.width_mm, record.thickness_mm] for record in records[:4]],
        dtype=np.float64,
    )

    np.testing.assert_allclose(encoder.means, np.mean(source_numeric, axis=0))
    np.testing.assert_allclose(encoder.scales, np.std(source_numeric, axis=0, ddof=0))
    assert encoder.held_out_domain == "domain-c"
    assert encoder.fit_specimen_ids == ("a-1", "a-2", "b-1", "b-2")
    assert not set(encoder.fit_specimen_ids) & {"c-1", "c-2"}


def test_fold_local_encoder_replaces_zero_source_scales() -> None:
    records = (
        DesignMetadata("a-1", "domain-a", "cross_ply", 10, "coni120", 10.0, 1.0),
        DesignMetadata("b-1", "domain-b", "quasi_isotropic", 10, "coni60", 10.0, 1.0),
        DesignMetadata("c-1", "domain-c", "cross_ply", 1000, "flat", 1000.0, 100.0),
    )

    encoder = fit_fold_local_design_encoder(records, "domain-c")

    np.testing.assert_array_equal(encoder.means, np.asarray([10.0, 10.0, 1.0]))
    np.testing.assert_array_equal(encoder.scales, np.ones(3))


def test_fold_local_encoder_has_fixed_feature_order_and_read_only_transform() -> None:
    records = _records()
    encoder = fit_fold_local_design_encoder(records, "domain-c")
    expected_names = (
        "ply_count",
        "width_mm",
        "thickness_mm",
        "laminate_type=cross_ply",
        "laminate_type=quasi_isotropic",
        "impactor=coni120",
        "impactor=coni60",
        "impactor=flat",
        "impactor=hemia",
        "impactor=hemib",
        "impactor=hemic",
    )

    transformed = encoder.transform(records)

    assert encoder.feature_names == expected_names
    assert transformed.shape == (len(records), 11)
    np.testing.assert_array_equal(transformed[0, 3:], np.asarray([1, 0, 1, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(transformed[5, 3:], np.asarray([0, 1, 0, 0, 0, 0, 0, 1]))
    assert transformed.flags.writeable is False
    with pytest.raises(ValueError):
        transformed[0, 0] = 0.0


def test_fold_local_encoder_transform_and_state_are_deterministic() -> None:
    records = _records()
    first = fit_fold_local_design_encoder(records, "domain-c")
    second = fit_fold_local_design_encoder(records, "DOMAIN-C")

    np.testing.assert_array_equal(first.transform(records), first.transform(records))
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.scales, second.scales)
    assert first.state_sha256 == second.state_sha256


def test_fold_local_encoder_rejects_unknown_category() -> None:
    records = _records()

    with pytest.raises(FeatureViewError, match="category"):
        fit_fold_local_design_encoder(
            (replace(records[0], impactor="unknown"), *records[1:]),
            "domain-c",
        )


def test_fold_local_encoder_rejects_absent_held_out_domain() -> None:
    with pytest.raises(FeatureViewError, match="absent"):
        fit_fold_local_design_encoder(_records(), "domain-missing")
