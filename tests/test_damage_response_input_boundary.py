from __future__ import annotations

import pytest

from cmc_bbdm.damage_response.contracts import InputRole, validate_input_names


def test_post_cai_image_is_forbidden_as_input() -> None:
    with pytest.raises(ValueError, match="post-CAI"):
        validate_input_names(("laminate", "post_cai_image"))


def test_true_response_is_never_a_deployable_input() -> None:
    assert InputRole.TRUE_CAI_TRACE.deployable is False
    assert InputRole.TRUE_PEAK_STRENGTH.deployable is False


@pytest.mark.parametrize(
    "forbidden_name",
    ("true_cai_trace", "true_peak_strength", "post_cai_image"),
)
def test_forbidden_input_names_fail_closed(forbidden_name: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_input_names(("geometry", forbidden_name))


def test_pre_cai_observation_names_are_accepted() -> None:
    assert validate_input_names(
        ("laminate", "geometry", "surface_profile", "cscan")
    ) == ("laminate", "geometry", "surface_profile", "cscan")


def test_unknown_input_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown input"):
        validate_input_names(("geometry", "mystery_signal"))
