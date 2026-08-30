from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.damage_response.raw_cai import StrainUnitStatus, decode_raw_cai_csv
from cmc_bbdm.damage_response.targets import (
    PublishedPeak,
    TargetError,
    convert_trace_to_response,
    decimal_places_from_excel_format,
    derive_global_absolute_tolerance,
    reconcile_published_peak,
    require_strain_endpoints_authorized,
)


def _trace_payload() -> bytes:
    lines = (
        '"ID番号","EDX-100A"',
        '"タイトル","c8-2"',
        '"試験日時","2022/01/24","12:10:45"',
        '"測定CH数",6',
        '"デジタル入力","OFF"',
        '"サンプリング周波数(Hz)",50',
        '"集録データ数/CH",3',
        '"測定時間(sec)",0.06',
        '"CH名称","Extension","Load","Strain-FL","Strain-FR","Strain-BL","Strain-BR"',
        '"CH No","CH1","CH2","CH3","CH4","CH5","CH6"',
        '"レンジ",10,10,50000,50000,50000,50000',
        '"校正係数",1,1,1,1,1,1',
        '"オフセット",0,0,0,0,0,0',
        '"単位","V","V","με","με","με","με"',
        "0.00,0.0,-0.1,1,2,3,4",
        "0.02,0.1,-0.4,5,6,7,8",
        "0.04,0.2,-0.2,9,10,11,12",
    )
    return ("\n".join(lines) + "\n").encode("cp932")


def test_trace_conversion_uses_registered_global_calibrations() -> None:
    response = convert_trace_to_response(
        decode_raw_cai_csv(_trace_payload()),
        width_mm=10.0,
        thickness_mm=2.0,
    )

    np.testing.assert_allclose(response.extension_mm, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(response.load_kn, [-2.5, -10.0, -5.0])
    np.testing.assert_allclose(response.stress_mpa, [-125.0, -500.0, -250.0])
    assert response.peak_row == 1
    assert response.peak_absolute_stress_mpa == 500.0


@pytest.mark.parametrize(
    ("width_mm", "thickness_mm"),
    ((0.0, 2.0), (10.0, -1.0), (float("nan"), 2.0), (True, 2.0)),
)
def test_trace_conversion_rejects_invalid_dimensions(
    width_mm: float, thickness_mm: float
) -> None:
    with pytest.raises(TargetError, match="dimension"):
        convert_trace_to_response(
            decode_raw_cai_csv(_trace_payload()),
            width_mm=width_mm,
            thickness_mm=thickness_mm,
        )


def test_global_tolerance_comes_from_one_published_precision() -> None:
    peaks = (
        PublishedPeak("c8-2", 499.996, decimal_places=2),
        PublishedPeak("c8-3", 200.0, decimal_places=2),
    )
    assert derive_global_absolute_tolerance(peaks) == pytest.approx(0.005)


@pytest.mark.parametrize(
    "number_format",
    ("0.00", "#,##0.00_);[Red](#,##0.00)"),
)
def test_excel_display_precision_is_parsed_for_global_tolerance(
    number_format: str,
) -> None:
    assert decimal_places_from_excel_format(number_format) == 2


def test_excel_general_format_has_no_authorized_rounding_precision() -> None:
    with pytest.raises(TargetError, match="decimal precision"):
        decimal_places_from_excel_format("General")


def test_global_tolerance_rejects_specimen_specific_precision() -> None:
    peaks = (
        PublishedPeak("c8-2", 500.0, decimal_places=2),
        PublishedPeak("c8-3", 200.0, decimal_places=3),
    )
    with pytest.raises(TargetError, match="one global"):
        derive_global_absolute_tolerance(peaks)


@pytest.mark.parametrize(
    ("published", "passed"),
    ((499.996, True), (499.994, False)),
)
def test_peak_reconciliation_uses_absolute_raw_stress_and_global_tolerance(
    published: float, passed: bool
) -> None:
    response = convert_trace_to_response(
        decode_raw_cai_csv(_trace_payload()),
        width_mm=10.0,
        thickness_mm=2.0,
    )
    record = reconcile_published_peak(
        response,
        PublishedPeak("c8-2", published, decimal_places=2),
        absolute_tolerance_mpa=0.005,
    )

    assert record.raw_peak_mpa == 500.0
    assert record.signed_error_mpa == pytest.approx(500.0 - published)
    assert record.passed is passed


def test_unresolved_strain_unit_disables_strain_endpoints() -> None:
    with pytest.raises(TargetError, match="unauthorized"):
        require_strain_endpoints_authorized(
            StrainUnitStatus.STRAIN_UNIT_UNRESOLVED
        )
