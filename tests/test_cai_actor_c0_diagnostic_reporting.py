import re

import numpy as np
import pytest

from scripts.cai_actor_c0_diagnostic.reporting import (
    _carry_render_timing,
    _write_csv,
    derive_state_provenance,
    hash_named_arrays,
    policy_input_hash,
    prior_channels,
)


def test_carry_render_timing_preserves_only_valid_exact_measurement():
    current = {"status": "PASS_CACHE_ONLY_PROVENANCE", "model_forwards": 0}
    existing = {
        "status": "STALE",
        "cache_report_render_seconds_upper_bound": 268,
        "timing_measurement": "MONOTONIC_WALL_SECONDS_ROUNDED_UP",
    }

    carried = _carry_render_timing(current, existing)
    assert carried == {
        **current,
        "cache_report_render_seconds_upper_bound": 268,
        "timing_measurement": "MONOTONIC_WALL_SECONDS_ROUNDED_UP",
    }
    assert _carry_render_timing(
        current,
        existing | {"cache_report_render_seconds_upper_bound": -1},
    ) == current
    assert _carry_render_timing(
        current,
        existing | {"timing_measurement": "ESTIMATE"},
    ) == current


def _prior_row() -> dict[str, str]:
    indicator = [0.0, 1.0, 0.25] + [0.0] * 61
    confidence = [0.0, 0.5, 1.0] + [0.0] * 61
    return {
        "vlm_available": "True",
        "no_reliable_cue": "False",
        "region_indicator": ";".join(str(value) for value in indicator),
        "confidence": ";".join(str(value) for value in confidence),
    }


def _provenance_row(*, action_count: str = "2", prefix_lengths: str = "[2, 2]") -> dict[str, str]:
    return {
        "sources": "['C_NATIVE_PREFIX', 'N_NATIVE_PREFIX']",
        "source_models": "['C', 'N']",
        "prefix_lengths": prefix_lengths,
        "action_count": action_count,
    }


def _trajectories() -> dict[str, dict[str, object]]:
    return {
        "C": {"actions": [7, 3, 11]},
        "N": {"actions": [7, 3, 12]},
    }


def test_hash_named_arrays_is_order_insensitive_but_binds_name_dtype_shape_and_values():
    arrays = {
        "region_indicator": np.arange(64, dtype=np.float32),
        "confidence": np.linspace(0.0, 1.0, 64, dtype=np.float32),
    }
    digest = hash_named_arrays(arrays)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest == hash_named_arrays(dict(reversed(tuple(arrays.items()))))

    changed_dtype = {
        "region_indicator": arrays["region_indicator"].astype(np.float64),
        "confidence": arrays["confidence"],
    }
    changed_shape = {
        "region_indicator": arrays["region_indicator"].reshape(8, 8),
        "confidence": arrays["confidence"],
    }
    changed_value = {
        "region_indicator": arrays["region_indicator"].copy(),
        "confidence": arrays["confidence"].copy(),
    }
    changed_value["confidence"][0] = np.float32(0.75)
    changed_name = {
        "region_indicator": arrays["region_indicator"],
        "ordinal_confidence": arrays["confidence"],
    }

    for variant in (changed_dtype, changed_shape, changed_value, changed_name):
        variant_digest = hash_named_arrays(variant)
        assert re.fullmatch(r"[0-9a-f]{64}", variant_digest)
        assert variant_digest != digest


def test_prior_channels_parses_scalar_booleans_and_float32_vectors():
    channels = prior_channels(_prior_row())

    assert set(channels) == {
        "vlm_available",
        "no_reliable_cue",
        "region_indicator",
        "confidence",
    }
    assert channels["vlm_available"].dtype == np.float32
    assert channels["no_reliable_cue"].dtype == np.float32
    assert channels["vlm_available"].shape == (1,)
    assert channels["no_reliable_cue"].shape == (1,)
    assert np.array_equal(channels["vlm_available"], np.ones(1, dtype=np.float32))
    assert np.array_equal(channels["no_reliable_cue"], np.zeros(1, dtype=np.float32))

    expected_indicator = np.asarray([0.0, 1.0, 0.25] + [0.0] * 61, dtype=np.float32)
    expected_confidence = np.asarray([0.0, 0.5, 1.0] + [0.0] * 61, dtype=np.float32)
    assert channels["region_indicator"].dtype == np.float32
    assert channels["confidence"].dtype == np.float32
    assert channels["region_indicator"].shape == (64,)
    assert channels["confidence"].shape == (64,)
    assert np.array_equal(channels["region_indicator"], expected_indicator)
    assert np.array_equal(channels["confidence"], expected_confidence)


@pytest.mark.parametrize("field", ["region_indicator", "confidence"])
@pytest.mark.parametrize("length", [63, 65])
def test_prior_channels_rejects_vector_lengths_other_than_64(field: str, length: int):
    row = _prior_row()
    row[field] = ";".join(["0.0"] * length)
    with pytest.raises(ValueError):
        prior_channels(row)


def test_policy_input_hash_is_repeatable_model_bound_and_prior_channel_bound():
    channels = prior_channels(_prior_row())
    physical_state_sha256 = "a" * 64

    c_hash = policy_input_hash(
        physical_state_sha256,
        model="C",
        channels=channels,
    )
    assert re.fullmatch(r"[0-9a-f]{64}", c_hash)
    assert c_hash == policy_input_hash(
        physical_state_sha256,
        model="C",
        channels=channels,
    )
    assert c_hash != policy_input_hash(
        physical_state_sha256,
        model="N",
        channels=channels,
    )

    for name in channels:
        changed = {channel: values.copy() for channel, values in channels.items()}
        changed[name][0] += np.float32(1.0)
        assert c_hash != policy_input_hash(
            physical_state_sha256,
            model="C",
            channels=changed,
        )


def test_derive_state_provenance_reconstructs_ordered_shared_prefix_and_origins():
    provenance = derive_state_provenance(_provenance_row(), _trajectories())

    assert provenance["t"] == 2
    assert provenance["prefix_cells_in_order"] == [7, 3]
    assert provenance["state_origins"] == [
        {
            "label": "C_NATIVE_PREFIX",
            "model": "C",
            "t": 2,
            "prefix_cells_in_order": [7, 3],
        },
        {
            "label": "N_NATIVE_PREFIX",
            "model": "N",
            "t": 2,
            "prefix_cells_in_order": [7, 3],
        },
    ]


def test_derive_state_provenance_rejects_inconsistent_prefixes():
    trajectories = _trajectories()
    trajectories["N"] = {"actions": [7, 4, 12]}
    with pytest.raises(ValueError):
        derive_state_provenance(_provenance_row(), trajectories)


def test_derive_state_provenance_rejects_prefix_length_out_of_bounds():
    with pytest.raises(ValueError):
        derive_state_provenance(
            _provenance_row(prefix_lengths="[4, 2]"),
            _trajectories(),
        )


def test_derive_state_provenance_rejects_action_count_mismatch():
    with pytest.raises(ValueError):
        derive_state_provenance(
            _provenance_row(action_count="1"),
            _trajectories(),
        )


def test_report_csv_uses_lf_without_carriage_returns(tmp_path):
    path = tmp_path / "report.csv"
    _write_csv(path, [{"name": "value", "count": 1}])
    assert path.read_bytes() == b"name,count\nvalue,1\n"
