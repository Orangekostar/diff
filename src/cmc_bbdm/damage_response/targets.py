from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.damage_response.contracts import (
    DISPLACEMENT_MM_PER_VOLT,
    LOAD_KN_PER_VOLT,
)
from cmc_bbdm.damage_response.raw_cai import RawCaiTrace, StrainUnitStatus


class TargetError(RuntimeError):
    """Raised when a response target cannot be derived under the frozen rules."""


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _positive_dimension(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetError(f"specimen dimension {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TargetError(f"specimen dimension {name} must be finite and positive")
    return result


@dataclass(frozen=True)
class ResponseTrace:
    specimen_id: str
    extension_mm: np.ndarray
    load_kn: np.ndarray
    stress_mpa: np.ndarray
    peak_row: int
    peak_absolute_stress_mpa: float


def convert_trace_to_response(
    trace: RawCaiTrace,
    *,
    width_mm: float,
    thickness_mm: float,
    canonical_specimen_id: str | None = None,
) -> ResponseTrace:
    """Apply the one registered load/displacement conversion and measured area."""

    width = _positive_dimension(width_mm, "width_mm")
    thickness = _positive_dimension(thickness_mm, "thickness_mm")
    if canonical_specimen_id is None:
        specimen_id = trace.specimen_id
    elif not isinstance(canonical_specimen_id, str) or not canonical_specimen_id.strip():
        raise TargetError("canonical specimen identity must be nonempty")
    else:
        specimen_id = canonical_specimen_id.strip().casefold()
    extension_mm = trace.extension_volts * DISPLACEMENT_MM_PER_VOLT
    load_kn = trace.load_volts * LOAD_KN_PER_VOLT
    stress_mpa = load_kn * 1000.0 / (width * thickness)
    finite = np.isfinite(stress_mpa)
    if np.count_nonzero(finite) < 2:
        raise TargetError("converted response has fewer than two finite stress rows")
    peak_row = int(np.nanargmax(np.abs(stress_mpa)))
    return ResponseTrace(
        specimen_id=specimen_id,
        extension_mm=_readonly(extension_mm),
        load_kn=_readonly(load_kn),
        stress_mpa=_readonly(stress_mpa),
        peak_row=peak_row,
        peak_absolute_stress_mpa=float(abs(stress_mpa[peak_row])),
    )


@dataclass(frozen=True)
class PublishedPeak:
    specimen_id: str
    value_mpa: float
    decimal_places: int

    def __post_init__(self) -> None:
        if not isinstance(self.specimen_id, str) or not self.specimen_id.strip():
            raise TargetError("published peak specimen identity must be nonempty")
        if isinstance(self.value_mpa, bool) or not isinstance(
            self.value_mpa, (int, float)
        ):
            raise TargetError("published peak must be numeric")
        if not math.isfinite(float(self.value_mpa)) or float(self.value_mpa) <= 0.0:
            raise TargetError("published peak must be finite and positive")
        if (
            not isinstance(self.decimal_places, int)
            or isinstance(self.decimal_places, bool)
            or not 0 <= self.decimal_places <= 12
        ):
            raise TargetError("published decimal precision must be an integer in [0, 12]")


def decimal_places_from_excel_format(number_format: str) -> int:
    """Read display precision from the positive section of an Excel format."""

    if not isinstance(number_format, str) or not number_format.strip():
        raise TargetError("published decimal precision is absent")
    positive_section = number_format.split(";", maxsplit=1)[0]
    match = re.search(r"\.([0#]+)", positive_section)
    if match is None:
        raise TargetError(
            f"published number format has no decimal precision: {number_format!r}"
        )
    return len(match.group(1))


def derive_global_absolute_tolerance(peaks: Iterable[PublishedPeak]) -> float:
    """Derive one half-rounding-unit tolerance shared by every specimen."""

    records = tuple(peaks)
    if not records:
        raise TargetError("published peak records are empty")
    precisions = {record.decimal_places for record in records}
    if len(precisions) != 1:
        raise TargetError("published peaks do not define one global decimal precision")
    identities = [record.specimen_id.strip().casefold() for record in records]
    if len(set(identities)) != len(identities):
        raise TargetError("published peak specimen identities are duplicate")
    decimal_places = next(iter(precisions))
    return 0.5 * 10.0 ** (-decimal_places)


@dataclass(frozen=True)
class PeakReconciliation:
    specimen_id: str
    raw_peak_mpa: float
    published_peak_mpa: float
    signed_error_mpa: float
    absolute_error_mpa: float
    absolute_tolerance_mpa: float
    passed: bool


def reconcile_published_peak(
    response: ResponseTrace,
    published: PublishedPeak,
    *,
    absolute_tolerance_mpa: float,
) -> PeakReconciliation:
    """Compare the unsmoothed absolute raw peak with one published value."""

    if response.specimen_id.strip().casefold() != published.specimen_id.strip().casefold():
        raise TargetError("response and published peak specimen identities differ")
    if (
        isinstance(absolute_tolerance_mpa, bool)
        or not isinstance(absolute_tolerance_mpa, (int, float))
        or not math.isfinite(float(absolute_tolerance_mpa))
        or absolute_tolerance_mpa < 0.0
    ):
        raise TargetError("global absolute tolerance must be finite and nonnegative")
    raw_peak = response.peak_absolute_stress_mpa
    published_peak = float(published.value_mpa)
    signed_error = raw_peak - published_peak
    absolute_error = abs(signed_error)
    tolerance = float(absolute_tolerance_mpa)
    return PeakReconciliation(
        specimen_id=response.specimen_id.strip().casefold(),
        raw_peak_mpa=raw_peak,
        published_peak_mpa=published_peak,
        signed_error_mpa=signed_error,
        absolute_error_mpa=absolute_error,
        absolute_tolerance_mpa=tolerance,
        passed=absolute_error <= tolerance,
    )


def require_strain_endpoints_authorized(status: StrainUnitStatus) -> None:
    if status is not StrainUnitStatus.MICROSTRAIN_SIGN_RESOLVED:
        raise TargetError(
            "strain-dependent endpoints are unauthorized while unit/sign is unresolved"
        )
