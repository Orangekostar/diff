from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from enum import Enum
from io import StringIO

import numpy as np

_REGISTERED_CHANNELS = (
    "Extension",
    "Load",
    "Strain-FL",
    "Strain-FR",
    "Strain-BL",
    "Strain-BR",
)
_REGISTERED_ALIASES = {
    "Extension": "extension_volts",
    "Load": "load_volts",
    "Strain-FL": "strain_fl",
    "Strain-FR": "strain_fr",
    "Strain-BL": "strain_bl",
    "Strain-BR": "strain_br",
}


class RawCaiError(RuntimeError):
    """Raised when an official raw CAI CSV violates its registered schema."""


class StrainUnitStatus(str, Enum):
    STRAIN_UNIT_UNRESOLVED = "STRAIN_UNIT_UNRESOLVED"
    MICROSTRAIN_SIGN_RESOLVED = "MICROSTRAIN_SIGN_RESOLVED"


def _readonly(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RawCaiTrace:
    specimen_id: str
    sampling_hz: float
    original_channel_names: tuple[str, ...]
    original_channel_units: tuple[str, ...]
    time_seconds: np.ndarray
    extension_volts: np.ndarray
    load_volts: np.ndarray
    strain_fl: np.ndarray
    strain_fr: np.ndarray
    strain_bl: np.ndarray
    strain_br: np.ndarray
    peak_row: int
    strain_unit_status: StrainUnitStatus

    @property
    def n_rows(self) -> int:
        return int(self.time_seconds.shape[0])

    @property
    def finite_counts(self) -> dict[str, int]:
        arrays = (
            self.extension_volts,
            self.load_volts,
            self.strain_fl,
            self.strain_fr,
            self.strain_bl,
            self.strain_br,
        )
        return {
            name: int(np.count_nonzero(np.isfinite(values)))
            for name, values in zip(
                self.original_channel_names, arrays, strict=True
            )
        }


def _metadata_rows(rows: list[list[str]]) -> tuple[dict[str, list[str]], int]:
    metadata: dict[str, list[str]] = {}
    units_index: int | None = None
    for index, row in enumerate(rows):
        if not row:
            continue
        label = row[0].strip()
        if label in {
            "タイトル",
            "測定CH数",
            "サンプリング周波数(Hz)",
            "集録データ数/CH",
            "CH名称",
            "単位",
        }:
            if label in metadata:
                raise RawCaiError(f"duplicate metadata row: {label}")
            metadata[label] = row[1:]
            if label == "単位":
                units_index = index
                break
    required = {
        "タイトル",
        "測定CH数",
        "サンプリング周波数(Hz)",
        "集録データ数/CH",
        "CH名称",
        "単位",
    }
    missing = sorted(required - metadata.keys())
    if missing or units_index is None:
        raise RawCaiError(f"raw CSV lacks registered metadata rows: {missing}")
    return metadata, units_index


def _single_value(metadata: dict[str, list[str]], label: str) -> str:
    values = [value.strip() for value in metadata[label] if value.strip()]
    if len(values) != 1:
        raise RawCaiError(f"metadata {label!r} must contain one value")
    return values[0]


def _finite_metadata_number(
    metadata: dict[str, list[str]], label: str, *, integer: bool
) -> float | int:
    raw = _single_value(metadata, label)
    try:
        value = float(raw)
    except ValueError as error:
        raise RawCaiError(f"metadata {label!r} must be numeric") from error
    if not math.isfinite(value) or (integer and not value.is_integer()):
        raise RawCaiError(f"metadata {label!r} must be finite")
    return int(value) if integer else value


def _data_value(raw: str, *, channel: str, row_number: int) -> float:
    value = raw.strip()
    if not value:
        return math.nan
    try:
        result = float(value)
    except ValueError as error:
        raise RawCaiError(
            f"nonnumeric {channel} value at data row {row_number}"
        ) from error
    if math.isinf(result):
        raise RawCaiError(f"expected finite {channel} value at data row {row_number}")
    return result


def decode_raw_cai_csv(payload: bytes) -> RawCaiTrace:
    """Decode one source-format CAI trace without smoothing or curve repair."""

    if not isinstance(payload, bytes):
        raise TypeError("raw CAI payload must be bytes")
    try:
        text = payload.decode("cp932", errors="strict")
    except UnicodeDecodeError as error:
        raise RawCaiError("raw CAI payload is not valid CP932") from error
    try:
        rows = list(csv.reader(StringIO(text, newline=""), delimiter=","))
    except csv.Error as error:
        raise RawCaiError("raw CAI payload is not valid source-format CSV") from error

    metadata, units_index = _metadata_rows(rows)
    channel_count = _finite_metadata_number(metadata, "測定CH数", integer=True)
    declared_rows = _finite_metadata_number(metadata, "集録データ数/CH", integer=True)
    sampling_hz = _finite_metadata_number(
        metadata, "サンプリング周波数(Hz)", integer=False
    )
    if channel_count != len(_REGISTERED_CHANNELS):
        raise RawCaiError("registered channel count changed")
    if sampling_hz != 50.0:
        raise RawCaiError(f"registered sampling frequency changed: {sampling_hz}")

    channel_names = tuple(value.strip() for value in metadata["CH名称"])
    if len(set(channel_names)) != len(channel_names):
        raise RawCaiError("duplicate channel name in raw CSV")
    if channel_names != _REGISTERED_CHANNELS:
        raise RawCaiError(
            "registered channels changed: "
            f"expected {_REGISTERED_CHANNELS!r}, observed {channel_names!r}"
        )
    if set(_REGISTERED_ALIASES) != set(channel_names):
        raise RawCaiError("explicit registered channel alias table is incomplete")

    channel_units = tuple(value.strip() for value in metadata["単位"])
    if len(channel_units) != channel_count:
        raise RawCaiError("registered channel unit count changed")

    data_rows = [row for row in rows[units_index + 1 :] if any(cell.strip() for cell in row)]
    if len(data_rows) != declared_rows:
        raise RawCaiError(
            "declared row count does not match data rows: "
            f"declared {declared_rows}, observed {len(data_rows)}"
        )

    time_values: list[float] = []
    channel_values = {name: [] for name in channel_names}
    for row_number, row in enumerate(data_rows, start=1):
        if len(row) != channel_count + 1:
            raise RawCaiError(
                f"data row {row_number} has {len(row)} columns, expected {channel_count + 1}"
            )
        time_value = _data_value(row[0], channel="time", row_number=row_number)
        if not math.isfinite(time_value):
            raise RawCaiError(f"expected finite time value at data row {row_number}")
        time_values.append(time_value)
        for name, raw in zip(channel_names, row[1:], strict=True):
            channel_values[name].append(
                _data_value(raw, channel=name, row_number=row_number)
            )

    arrays = {name: _readonly(values) for name, values in channel_values.items()}
    extension_finite = np.isfinite(arrays["Extension"])
    load_finite = np.isfinite(arrays["Load"])
    if np.count_nonzero(extension_finite) < 2 or np.count_nonzero(load_finite) < 2:
        raise RawCaiError("raw trace requires at least two finite rows for load and extension")
    peak_row = int(np.nanargmax(np.abs(arrays["Load"])))

    return RawCaiTrace(
        specimen_id=_single_value(metadata, "タイトル").strip().casefold(),
        sampling_hz=float(sampling_hz),
        original_channel_names=channel_names,
        original_channel_units=channel_units,
        time_seconds=_readonly(time_values),
        extension_volts=arrays["Extension"],
        load_volts=arrays["Load"],
        strain_fl=arrays["Strain-FL"],
        strain_fr=arrays["Strain-FR"],
        strain_bl=arrays["Strain-BL"],
        strain_br=arrays["Strain-BR"],
        peak_row=peak_row,
        strain_unit_status=StrainUnitStatus.STRAIN_UNIT_UNRESOLVED,
    )
