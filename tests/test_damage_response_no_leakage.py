from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cmc_bbdm.damage_response.contracts import validate_input_names
from cmc_bbdm.damage_response.feature_views import (
    FeatureViewError,
    fit_fold_local_design_encoder,
    validate_p1_redundancy_view,
)
from cmc_bbdm.damage_response.p2_views import P2ViewError, validate_p2_view
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
TARGET_AND_FORBIDDEN_FIELDS = (
    "extension_peak_mm",
    "slope_u20_u60_mpa_per_mm",
    "normalized_prepeak_auc",
    "true_cai_trace",
    "derived_response",
    "response_curve",
    "post_cai_image",
)


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


@pytest.mark.parametrize("view_name", REGISTERED_VIEWS)
@pytest.mark.parametrize("field_name", TARGET_AND_FORBIDDEN_FIELDS)
def test_target_and_forbidden_fields_cannot_enter_registered_views(
    view_name: str, field_name: str
) -> None:
    with pytest.raises(FeatureViewError):
        validate_p1_redundancy_view(
            view_name,
            REGISTERED_VIEWS[view_name] + (field_name,),
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "derived_response",
        "true_cai_trace",
        "true_peak_strength",
        "post_cai_image",
        "impactor",
    ),
)
def test_deployable_validation_rejects_forbidden_and_privileged_fields(
    field_name: str,
) -> None:
    with pytest.raises(ValueError):
        validate_input_names(("laminate", field_name))


def test_held_out_numeric_sentinels_do_not_change_encoder_state() -> None:
    records = _records()
    reference = fit_fold_local_design_encoder(records, "domain-c")
    held_out_ids = {record.specimen_id for record in records if record.domain_id == "domain-c"}
    mutated = tuple(
        replace(
            record,
            ply_count=1_000_000_000 + index,
            width_mm=2_000_000_000.0 + index,
            thickness_mm=3_000_000_000.0 + index,
        )
        if record.domain_id == "domain-c"
        else record
        for index, record in enumerate(records)
    )

    sentinel_encoder = fit_fold_local_design_encoder(mutated, "domain-c")

    assert reference.means.tobytes() == sentinel_encoder.means.tobytes()
    assert reference.scales.tobytes() == sentinel_encoder.scales.tobytes()
    assert reference.state_sha256 == sentinel_encoder.state_sha256
    assert reference.fit_specimen_ids == sentinel_encoder.fit_specimen_ids
    assert not set(reference.fit_specimen_ids) & held_out_ids
    assert not set(sentinel_encoder.fit_specimen_ids) & held_out_ids
    np.testing.assert_array_equal(reference.means, sentinel_encoder.means)
    np.testing.assert_array_equal(reference.scales, sentinel_encoder.scales)


@pytest.mark.parametrize(
    "forbidden",
    (
        "true_cai_strength",
        "raw_cai_trace",
        "response_curve",
        "post_cai_image",
        "privileged_total_impact_energy_j",
        "privileged_impactor",
    ),
)
def test_p2_deployable_view_rejects_outcome_and_privileged_fields(
    forbidden: str,
) -> None:
    with pytest.raises(P2ViewError):
        validate_p2_view("F4", ("F0", "surface_profile_stats21", forbidden))
