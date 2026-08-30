from __future__ import annotations

import pytest

from cmc_bbdm.damage_response.raw_cai import (
    RawCaiError,
    StrainUnitStatus,
    decode_raw_cai_csv,
)


def _payload(
    *,
    channel_names: tuple[str, ...] = (
        "Extension",
        "Load",
        "Strain-FL",
        "Strain-FR",
        "Strain-BL",
        "Strain-BR",
    ),
    data_rows: tuple[tuple[str, ...], ...] = (
        ("0.00", "0.0", "-0.1", "1", "2", "3", "4"),
        ("0.02", "0.1", "-0.4", "5", "6", "7", "8"),
        ("0.04", "0.2", "-0.2", "9", "10", "11", "12"),
    ),
    declared_rows: int | None = None,
) -> bytes:
    n_rows = len(data_rows) if declared_rows is None else declared_rows
    names = ",".join(f'"{name}"' for name in channel_names)
    lines = [
        '"ID番号","EDX-100A"',
        '"タイトル","c8-2"',
        '"試験日時","2022/01/24","12:10:45"',
        '"測定CH数",6',
        '"デジタル入力","OFF"',
        '"サンプリング周波数(Hz)",50',
        f'"集録データ数/CH",{n_rows}',
        '"測定時間(sec)",0.06',
        f'"CH名称",{names}',
        '"CH No","CH1","CH2","CH3","CH4","CH5","CH6"',
        '"レンジ",10,10,50000,50000,50000,50000',
        '"校正係数",1,1,1,1,1,1',
        '"オフセット",0,0,0,0,0,0',
        '"単位","V","V","με","με","με","με"',
    ]
    lines.extend(",".join(row) for row in data_rows)
    return ("\n".join(lines) + "\n").encode("cp932")


def test_decoder_preserves_registered_schema_and_peak_row() -> None:
    trace = decode_raw_cai_csv(_payload())

    assert trace.specimen_id == "c8-2"
    assert trace.n_rows == 3
    assert trace.sampling_hz == 50.0
    assert trace.original_channel_names == (
        "Extension",
        "Load",
        "Strain-FL",
        "Strain-FR",
        "Strain-BL",
        "Strain-BR",
    )
    assert trace.original_channel_units == ("V", "V", "με", "με", "με", "με")
    assert trace.extension_volts.tolist() == [0.0, 0.1, 0.2]
    assert trace.load_volts.tolist() == [-0.1, -0.4, -0.2]
    assert trace.peak_row == 1
    assert trace.finite_counts == {
        "Extension": 3,
        "Load": 3,
        "Strain-FL": 3,
        "Strain-FR": 3,
        "Strain-BL": 3,
        "Strain-BR": 3,
    }
    assert trace.strain_unit_status is StrainUnitStatus.STRAIN_UNIT_UNRESOLVED


def test_decoder_accepts_exporter_trailing_empty_metadata_cells() -> None:
    payload = _payload().replace(
        '"測定CH数",6'.encode("cp932"),
        '"測定CH数",6,,,,,'.encode("cp932"),
    )
    assert decode_raw_cai_csv(payload).n_rows == 3


def test_decoder_rejects_missing_load_channel() -> None:
    names = (
        "Extension",
        "Force",
        "Strain-FL",
        "Strain-FR",
        "Strain-BL",
        "Strain-BR",
    )
    with pytest.raises(RawCaiError, match="registered channels"):
        decode_raw_cai_csv(_payload(channel_names=names))


def test_decoder_rejects_duplicate_header() -> None:
    names = (
        "Load",
        "Load",
        "Strain-FL",
        "Strain-FR",
        "Strain-BL",
        "Strain-BR",
    )
    with pytest.raises(RawCaiError, match="duplicate channel"):
        decode_raw_cai_csv(_payload(channel_names=names))


def test_decoder_rejects_nonfinite_load_value() -> None:
    rows = (
        ("0.00", "0.0", "-0.1", "1", "2", "3", "4"),
        ("0.02", "0.1", "inf", "5", "6", "7", "8"),
    )
    with pytest.raises(RawCaiError, match="finite Load"):
        decode_raw_cai_csv(_payload(data_rows=rows))


def test_decoder_requires_two_finite_load_and_extension_rows() -> None:
    rows = (("0.00", "", "", "1", "2", "3", "4"),)
    with pytest.raises(RawCaiError, match="two finite rows"):
        decode_raw_cai_csv(_payload(data_rows=rows))


def test_decoder_rejects_declared_row_count_mismatch() -> None:
    with pytest.raises(RawCaiError, match="declared row count"):
        decode_raw_cai_csv(_payload(declared_rows=99))


def test_decoder_rejects_non_cp932_payload() -> None:
    with pytest.raises(RawCaiError, match="CP932"):
        decode_raw_cai_csv(b"\x81")
