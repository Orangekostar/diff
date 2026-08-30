from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.damage_response.response_extraction import (
    ExtractionError,
    aggregate_extension_positions,
    extract_prepeak_response,
)
from cmc_bbdm.damage_response.targets import ResponseTrace


def _quadratic_response(*, post_peak: bool = False) -> ResponseTrace:
    baseline_extension = np.full(50, 0.25, dtype=np.float64)
    baseline_stress = np.full(50, 3.0, dtype=np.float64)
    x = np.linspace(0.0, 2.0, 101, dtype=np.float64)
    loading_extension = 0.25 - x
    loading_stress = 3.0 - 100.0 * (x / 2.0) ** 2
    extension = np.concatenate((baseline_extension, loading_extension[1:]))
    stress = np.concatenate((baseline_stress, loading_stress[1:]))
    load = stress * 0.01
    peak_row = len(stress) - 1
    if post_peak:
        extension = np.concatenate((extension, [-9.0, -10.0]))
        stress = np.concatenate((stress, [5000.0, -5000.0]))
        load = np.concatenate((load, [50.0, -50.0]))
    return ResponseTrace(
        specimen_id="c8-2",
        extension_mm=extension,
        load_kn=load,
        stress_mpa=stress,
        peak_row=peak_row,
        peak_absolute_stress_mpa=97.0,
    )


def test_prepeak_extraction_has_fixed_grid_anchors_and_descriptors() -> None:
    extracted = extract_prepeak_response(_quadratic_response())

    assert extracted.specimen_id == "c8-2"
    assert extracted.u.shape == (101,)
    np.testing.assert_allclose(extracted.u, np.linspace(0.0, 1.0, 101))
    np.testing.assert_allclose(extracted.normalized_stress, extracted.u**2)
    assert extracted.normalized_stress[0] == 0.0
    assert extracted.normalized_stress[-1] == 1.0
    assert extracted.extension_peak_mm == pytest.approx(2.0)
    assert extracted.zeroed_peak_stress_mpa == pytest.approx(100.0)
    assert extracted.slope_u20_u60_mpa_per_mm == pytest.approx(40.0)
    assert extracted.normalized_prepeak_auc == pytest.approx(
        np.trapezoid(extracted.u**2, extracted.u)
    )
    assert extracted.q_midpoint == pytest.approx(0.25)
    assert extracted.extension_mm.flags.writeable is False
    assert extracted.normalized_stress.flags.writeable is False


def test_post_peak_rows_do_not_change_primary_extraction() -> None:
    baseline = extract_prepeak_response(_quadratic_response())
    with_post_peak = extract_prepeak_response(_quadratic_response(post_peak=True))

    np.testing.assert_array_equal(
        baseline.normalized_stress, with_post_peak.normalized_stress
    )
    assert baseline.extension_peak_mm == with_post_peak.extension_peak_mm
    assert baseline.normalized_prepeak_auc == with_post_peak.normalized_prepeak_auc


@pytest.mark.parametrize(
    ("reverse_extension", "reverse_stress"),
    ((True, False), (False, True), (True, True)),
)
def test_peak_signs_orient_extension_and_stress_independently(
    reverse_extension: bool, reverse_stress: bool
) -> None:
    reference = _quadratic_response()
    extension = reference.extension_mm.copy()
    stress = reference.stress_mpa.copy()
    if reverse_extension:
        extension = 0.5 - extension
    if reverse_stress:
        stress = 6.0 - stress
    transformed = ResponseTrace(
        specimen_id=reference.specimen_id,
        extension_mm=extension,
        load_kn=stress * 0.01,
        stress_mpa=stress,
        peak_row=reference.peak_row,
        peak_absolute_stress_mpa=float(abs(stress[reference.peak_row])),
    )

    extracted = extract_prepeak_response(transformed)

    u = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(extracted.extension_mm, u * 2.0)
    np.testing.assert_allclose(extracted.normalized_stress, u**2)


def test_duplicate_extension_positions_use_median_stress() -> None:
    extension, stress = aggregate_extension_positions(
        np.asarray([0.0, 0.5, 0.5, 0.5, 1.0]),
        np.asarray([0.0, 10.0, 30.0, 1000.0, 40.0]),
    )

    np.testing.assert_array_equal(extension, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(stress, [0.0, 30.0, 40.0])


def test_extraction_is_byte_deterministic() -> None:
    first = extract_prepeak_response(_quadratic_response())
    second = extract_prepeak_response(_quadratic_response())

    assert first.u.tobytes() == second.u.tobytes()
    assert first.normalized_stress.tobytes() == second.normalized_stress.tobytes()


def test_extraction_rejects_peak_before_baseline_window() -> None:
    response = _quadratic_response()
    early = ResponseTrace(
        specimen_id=response.specimen_id,
        extension_mm=response.extension_mm,
        load_kn=response.load_kn,
        stress_mpa=response.stress_mpa,
        peak_row=49,
        peak_absolute_stress_mpa=response.peak_absolute_stress_mpa,
    )

    with pytest.raises(ExtractionError, match="baseline window"):
        extract_prepeak_response(early)


def test_extraction_rejects_nonpositive_offset_corrected_peak() -> None:
    response = _quadratic_response()
    flat = ResponseTrace(
        specimen_id=response.specimen_id,
        extension_mm=np.zeros_like(response.extension_mm),
        load_kn=response.load_kn,
        stress_mpa=response.stress_mpa,
        peak_row=response.peak_row,
        peak_absolute_stress_mpa=response.peak_absolute_stress_mpa,
    )

    with pytest.raises(ExtractionError, match="peak extension"):
        extract_prepeak_response(flat)


def test_extraction_rejects_zero_offset_corrected_stress_peak() -> None:
    response = _quadratic_response()
    flat_stress = np.full_like(response.stress_mpa, 3.0)
    flat = ResponseTrace(
        specimen_id=response.specimen_id,
        extension_mm=response.extension_mm,
        load_kn=flat_stress * 0.01,
        stress_mpa=flat_stress,
        peak_row=response.peak_row,
        peak_absolute_stress_mpa=3.0,
    )

    with pytest.raises(ExtractionError, match="peak stress"):
        extract_prepeak_response(flat)


def test_extraction_requires_fifty_unique_extension_positions() -> None:
    response = _quadratic_response()
    quantized = ResponseTrace(
        specimen_id=response.specimen_id,
        extension_mm=np.round(response.extension_mm, 0),
        load_kn=response.load_kn,
        stress_mpa=response.stress_mpa,
        peak_row=response.peak_row,
        peak_absolute_stress_mpa=response.peak_absolute_stress_mpa,
    )

    with pytest.raises(ExtractionError, match="unique extension"):
        extract_prepeak_response(quantized)


def test_extraction_rejects_nonfinite_prepeak_values() -> None:
    response = _quadratic_response()
    stress = response.stress_mpa.copy()
    stress[75] = np.inf
    invalid = ResponseTrace(
        specimen_id=response.specimen_id,
        extension_mm=response.extension_mm,
        load_kn=response.load_kn,
        stress_mpa=stress,
        peak_row=response.peak_row,
        peak_absolute_stress_mpa=response.peak_absolute_stress_mpa,
    )

    with pytest.raises(ExtractionError, match="nonfinite"):
        extract_prepeak_response(invalid)


def test_extraction_rejects_misaligned_response_arrays() -> None:
    response = _quadratic_response()
    invalid = ResponseTrace(
        specimen_id=response.specimen_id,
        extension_mm=response.extension_mm[:-1],
        load_kn=response.load_kn,
        stress_mpa=response.stress_mpa,
        peak_row=response.peak_row,
        peak_absolute_stress_mpa=response.peak_absolute_stress_mpa,
    )

    with pytest.raises(ExtractionError, match="not aligned"):
        extract_prepeak_response(invalid)


def test_extraction_rejects_peak_row_outside_response() -> None:
    response = _quadratic_response()
    invalid = ResponseTrace(
        specimen_id=response.specimen_id,
        extension_mm=response.extension_mm,
        load_kn=response.load_kn,
        stress_mpa=response.stress_mpa,
        peak_row=len(response.stress_mpa),
        peak_absolute_stress_mpa=response.peak_absolute_stress_mpa,
    )

    with pytest.raises(ExtractionError, match="outside"):
        extract_prepeak_response(invalid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"baseline_samples": 0}, "baseline sample"),
        ({"grid_points": 100}, "exactly 101"),
        ({"minimum_unique_extension_positions": 49}, "remain 50"),
    ),
)
def test_extraction_rejects_changed_fixed_parameters(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ExtractionError, match=message):
        extract_prepeak_response(_quadratic_response(), **kwargs)
