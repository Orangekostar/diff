"""Morphology-preserving variant construction and acceptance gates for D8."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

from cmc_bbdm.cpb_cscan_morphology import (
    CscanFootprintMeasurement,
    CscanMorphologyRule,
    extract_damage_footprint,
    physical_morphology_descriptors,
)
from cmc_bbdm.cpb_physical_descriptors import PhysicalCalibration

_FIELD_SHAPE = (3, 64, 64)
_COUNTS = frozenset({1, 2, 4, 8, 16})


def _registered_float(value: object, *, label: str, choices: frozenset[float]) -> float:
    if type(value) is not float or not math.isfinite(value) or value not in choices:
        raise ValueError(f"{label} is not a registered threshold")
    return value


@dataclass(frozen=True, slots=True)
class MorphologyThresholds:
    """One inner-frozen morphology gate configuration."""

    area_relative_deviation: float
    width_relative_deviation: float
    height_relative_deviation: float
    centroid_shift_mm: float
    low_frequency_correlation_minimum: float
    radial_spearman_minimum: float
    low_frequency_sigma_pixels: float = 2.0
    radial_profile_bins: int = 16

    def __post_init__(self) -> None:
        _registered_float(
            self.area_relative_deviation,
            label="area deviation",
            choices=frozenset({0.025, 0.05, 0.075, 0.10}),
        )
        _registered_float(
            self.width_relative_deviation,
            label="width deviation",
            choices=frozenset({0.025, 0.05, 0.075, 0.10}),
        )
        _registered_float(
            self.height_relative_deviation,
            label="height deviation",
            choices=frozenset({0.025, 0.05, 0.075, 0.10}),
        )
        _registered_float(
            self.centroid_shift_mm,
            label="centroid shift",
            choices=frozenset({0.5, 1.0, 2.0}),
        )
        _registered_float(
            self.low_frequency_correlation_minimum,
            label="low-frequency correlation",
            choices=frozenset({0.95, 0.97, 0.98, 0.99}),
        )
        _registered_float(
            self.radial_spearman_minimum,
            label="radial correlation",
            choices=frozenset({0.90, 0.95, 0.98}),
        )
        if self.low_frequency_sigma_pixels != 2.0:
            raise ValueError("low-frequency sigma is not registered")
        if type(self.radial_profile_bins) is not int or self.radial_profile_bins != 16:
            raise ValueError("radial profile bin count is not registered")


@dataclass(frozen=True, slots=True)
class VariantRecord:
    """One proposed field and its complete morphology-gate record."""

    variant: np.ndarray
    encoder_image: np.ndarray
    accepted: bool
    area_deviation: float
    width_deviation: float
    height_deviation: float
    centroid_shift_mm: float
    low_frequency_correlation: float
    radial_profile_correlation: float
    failed_conditions: tuple[str, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class VariantBatch:
    """Accepted variants plus explicit raw-field fallbacks for one specimen."""

    variants: tuple[np.ndarray, ...]
    encoder_images: tuple[np.ndarray, ...]
    records: tuple[VariantRecord, ...]
    proposal_count: int
    accepted_count: int
    fallback_count: int
    acceptance_rate: float
    state_sha256: str


@dataclass(frozen=True, slots=True)
class AcceptanceAudit:
    """Overall and per-domain proposal acceptance for candidate eligibility."""

    eligible: bool
    overall_rate: float
    domain_rates: tuple[tuple[str, float], ...]
    failed_domains: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _VariantSourceContext:
    original: np.ndarray
    native: np.ndarray
    footprint: CscanFootprintMeasurement
    descriptors: Mapping[str, float]
    rule: CscanMorphologyRule
    calibration: PhysicalCalibration
    thresholds: MorphologyThresholds


def _field(value: object, *, label: str, bounded: bool) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if array.shape != _FIELD_SHAPE or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite float32 (3, 64, 64) field")
    if bounded and (np.min(array) < -1.0 or np.max(array) > 1.0):
        raise ValueError(f"{label} must lie in [-1, 1]")
    return np.array(array, dtype=np.float32, copy=True, order="C")


def _readonly(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float32).reshape(
        contiguous.shape
    )
    output.setflags(write=False)
    return output


def _native_rgb(value: object) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.uint8)
        or value.ndim != 3
        or value.shape[2] != 3
        or min(value.shape[:2]) < 5
    ):
        raise ValueError("native source must be a uint8 RGB image")
    contiguous = np.ascontiguousarray(value)
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.uint8).reshape(
        contiguous.shape
    )
    output.setflags(write=False)
    return output


def _resize_native_to_64(native: np.ndarray) -> np.ndarray:
    image = Image.fromarray(native, mode="RGB").resize(
        (64, 64), resample=Image.Resampling.BICUBIC
    )
    rgb = np.asarray(image, dtype=np.uint8)
    return (
        rgb.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    ).transpose(2, 0, 1)


def _lift_to_native(
    native: np.ndarray, source: np.ndarray, variant: np.ndarray
) -> np.ndarray:
    if np.array_equal(source, variant):
        return native
    height, width = native.shape[:2]
    delta = variant.astype(np.float32) - source.astype(np.float32)
    lifted_channels: list[np.ndarray] = []
    for channel in delta:
        resized = Image.fromarray(channel, mode="F").resize(
            (width, height), resample=Image.Resampling.BICUBIC
        )
        lifted_channels.append(np.asarray(resized, dtype=np.float32))
    lifted_delta = np.stack(lifted_channels, axis=2)
    native_normalized = (
        native.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    )
    lifted = np.rint(
        (np.clip(native_normalized + lifted_delta, -1.0, 1.0) + 1.0) * 127.5
    ).astype(np.uint8)
    return _native_rgb(lifted)


def _relative(candidate: float, reference: float) -> float:
    if (
        not math.isfinite(candidate)
        or not math.isfinite(reference)
        or candidate < 0.0
        or reference < 0.0
    ):
        raise ValueError("morphology descriptors must be finite and nonnegative")
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else 1.0
    return abs(candidate - reference) / reference


def _centroid(
    mask: np.ndarray, calibration: PhysicalCalibration
) -> tuple[float, float]:
    rows, columns = np.nonzero(np.asarray(mask) > 0)
    if len(rows) == 0:
        return math.nan, math.nan
    height, width = mask.shape
    return (
        float(np.mean(columns + 0.5) * calibration.field_width_mm / width),
        float(np.mean(rows + 0.5) * calibration.field_height_mm / height),
    )


def _centroid_shift(
    source_mask: np.ndarray,
    candidate_mask: np.ndarray,
    calibration: PhysicalCalibration,
) -> float:
    source_foreground = np.asarray(source_mask) > 0
    candidate_foreground = np.asarray(candidate_mask) > 0
    if not source_foreground.any() and not candidate_foreground.any():
        return 0.0
    source = _centroid(source_mask, calibration)
    candidate = _centroid(candidate_mask, calibration)
    if any(not math.isfinite(value) for value in (*source, *candidate)):
        return math.hypot(calibration.field_width_mm, calibration.field_height_mm)
    return math.hypot(candidate[0] - source[0], candidate[1] - source[1])


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    if np.array_equal(left, right):
        return 1.0
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(float(left @ right) / denominator, -1.0, 1.0))


def _low_frequency_correlation(
    source: np.ndarray, candidate: np.ndarray, *, sigma: float
) -> float:
    source_low = gaussian_filter(
        source.astype(np.float64), sigma=(0.0, sigma, sigma), mode="reflect"
    )
    candidate_low = gaussian_filter(
        candidate.astype(np.float64), sigma=(0.0, sigma, sigma), mode="reflect"
    )
    return _correlation(source_low, candidate_low)


def _radial_profile(
    mask: np.ndarray, calibration: PhysicalCalibration, *, bins: int
) -> np.ndarray:
    foreground = np.asarray(mask) > 0
    center = _centroid(mask, calibration)
    if not foreground.any() or any(not math.isfinite(value) for value in center):
        return np.zeros(bins, dtype=np.float64)
    height, width = foreground.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    x = (columns + 0.5) * calibration.field_width_mm / width
    y = (rows + 0.5) * calibration.field_height_mm / height
    radius = np.hypot(x - center[0], y - center[1])
    maximum = math.hypot(calibration.field_width_mm, calibration.field_height_mm)
    indices = np.minimum((radius / maximum * bins).astype(np.int64), bins - 1)
    total = np.bincount(indices.ravel(), minlength=bins).astype(np.float64)
    occupied = np.bincount(
        indices.ravel(), weights=foreground.ravel(), minlength=bins
    ).astype(np.float64)
    return np.divide(occupied, total, out=np.zeros_like(occupied), where=total > 0.0)


def _radial_correlation(
    source_mask: np.ndarray,
    candidate_mask: np.ndarray,
    calibration: PhysicalCalibration,
    *,
    bins: int,
) -> float:
    source = _radial_profile(source_mask, calibration, bins=bins)
    candidate = _radial_profile(candidate_mask, calibration, bins=bins)
    if np.array_equal(source, candidate):
        return 1.0
    return _correlation(rankdata(source), rankdata(candidate))


def _state_digest(payload: Mapping[str, object], arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _source_context(
    source: np.ndarray,
    native_source: np.ndarray,
    *,
    rule: CscanMorphologyRule,
    calibration: PhysicalCalibration,
    thresholds: MorphologyThresholds,
) -> _VariantSourceContext:
    original = _readonly(_field(source, label="source", bounded=True))
    native = _native_rgb(native_source)
    if not np.array_equal(_resize_native_to_64(native), original):
        raise ValueError("native source does not reproduce the registered 64x64 source")
    if type(rule) is not CscanMorphologyRule:
        raise TypeError("an exact CscanMorphologyRule is required")
    if type(calibration) is not PhysicalCalibration:
        raise TypeError("an exact PhysicalCalibration is required")
    if type(thresholds) is not MorphologyThresholds:
        raise TypeError("exact morphology thresholds are required")
    footprint = extract_damage_footprint(native, calibration, rule)
    descriptors = physical_morphology_descriptors(footprint.mask, calibration)
    return _VariantSourceContext(
        original=original,
        native=native,
        footprint=footprint,
        descriptors=descriptors,
        rule=rule,
        calibration=calibration,
        thresholds=thresholds,
    )


def _build_variant_from_context(
    context: _VariantSourceContext,
    residual: np.ndarray,
    *,
    alpha: float,
) -> VariantRecord:
    original = context.original
    native = context.native
    rule = context.rule
    calibration = context.calibration
    thresholds = context.thresholds
    perturbation = _field(residual, label="residual", bounded=False)
    if type(alpha) not in (int, float) or not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    alpha_value = float(alpha)
    if not -0.5 <= alpha_value <= 1.0:
        raise ValueError("alpha must lie in [-0.5, 1.0]")

    variant = _readonly(np.clip(original + alpha_value * perturbation, -1.0, 1.0))
    encoder_image = _lift_to_native(native, original, variant)
    candidate_footprint = extract_damage_footprint(encoder_image, calibration, rule)
    source_values = context.descriptors
    candidate_values = physical_morphology_descriptors(
        candidate_footprint.mask, calibration
    )
    area = _relative(
        float(candidate_values["projected_damage_area"]),
        float(source_values["projected_damage_area"]),
    )
    width = _relative(
        float(candidate_values["damage_width"]),
        float(source_values["damage_width"]),
    )
    height = _relative(
        float(candidate_values["damage_height"]),
        float(source_values["damage_height"]),
    )
    centroid = _centroid_shift(
        context.footprint.mask, candidate_footprint.mask, calibration
    )
    low_frequency = _low_frequency_correlation(
        original,
        variant,
        sigma=thresholds.low_frequency_sigma_pixels,
    )
    radial = _radial_correlation(
        context.footprint.mask,
        candidate_footprint.mask,
        calibration,
        bins=thresholds.radial_profile_bins,
    )
    checks = (
        ("area_deviation", area <= thresholds.area_relative_deviation),
        ("width_deviation", width <= thresholds.width_relative_deviation),
        ("height_deviation", height <= thresholds.height_relative_deviation),
        ("centroid_shift", centroid <= thresholds.centroid_shift_mm),
        (
            "low_frequency_correlation",
            low_frequency >= thresholds.low_frequency_correlation_minimum,
        ),
        (
            "radial_profile_correlation",
            radial >= thresholds.radial_spearman_minimum,
        ),
    )
    failed = tuple(name for name, passed in checks if not passed)
    metrics = {
        "alpha": alpha_value,
        "rule": rule.sort_key(),
        "calibration": asdict(calibration),
        "thresholds": asdict(thresholds),
        "area_deviation": area,
        "width_deviation": width,
        "height_deviation": height,
        "centroid_shift_mm": centroid,
        "low_frequency_correlation": low_frequency,
        "radial_profile_correlation": radial,
        "failed_conditions": failed,
    }
    return VariantRecord(
        variant=variant,
        encoder_image=encoder_image,
        accepted=not failed,
        area_deviation=area,
        width_deviation=width,
        height_deviation=height,
        centroid_shift_mm=centroid,
        low_frequency_correlation=low_frequency,
        radial_profile_correlation=radial,
        failed_conditions=failed,
        state_sha256=_state_digest(
            metrics, (original, perturbation, variant, encoder_image)
        ),
    )


def build_variant(
    source: np.ndarray,
    residual: np.ndarray,
    *,
    native_source: np.ndarray,
    alpha: float,
    rule: CscanMorphologyRule,
    calibration: PhysicalCalibration,
    thresholds: MorphologyThresholds,
) -> VariantRecord:
    """Construct one clipped D8 variant and apply the complete morphology gate."""
    return _build_variant_from_context(
        _source_context(
            source,
            native_source,
            rule=rule,
            calibration=calibration,
            thresholds=thresholds,
        ),
        residual,
        alpha=alpha,
    )


def build_variant_batch(
    source: np.ndarray,
    residuals: tuple[np.ndarray, ...],
    *,
    native_source: np.ndarray,
    alpha: float,
    requested_count: int,
    rule: CscanMorphologyRule,
    calibration: PhysicalCalibration,
    thresholds: MorphologyThresholds,
    maximum_proposals: int = 32,
) -> VariantBatch:
    """Keep accepted proposals in draw order and fill missing slots with raw input."""

    context = _source_context(
        source,
        native_source,
        rule=rule,
        calibration=calibration,
        thresholds=thresholds,
    )
    original = context.original
    native = context.native
    if not isinstance(residuals, tuple) or not residuals:
        raise ValueError("residual proposals must be a nonempty tuple")
    if type(requested_count) is not int or requested_count not in _COUNTS:
        raise ValueError("requested variant count is not registered")
    if type(maximum_proposals) is not int or maximum_proposals != 32:
        raise ValueError("maximum proposal count is frozen at 32")
    accepted: list[np.ndarray] = []
    records: list[VariantRecord] = []
    for residual in residuals[:maximum_proposals]:
        record = _build_variant_from_context(
            context,
            residual,
            alpha=alpha,
        )
        records.append(record)
        if record.accepted:
            accepted.append(record.variant)
            if len(accepted) == requested_count:
                break
    fallback_count = requested_count - len(accepted)
    variants = tuple(accepted) + tuple(
        _readonly(original) for _ in range(fallback_count)
    )
    accepted_images = tuple(
        record.encoder_image for record in records if record.accepted
    )[:requested_count]
    encoder_images = accepted_images + tuple(native for _ in range(fallback_count))
    proposal_count = len(records)
    accepted_count = len(accepted)
    acceptance_rate = accepted_count / proposal_count if proposal_count else 0.0
    payload = {
        "alpha": float(alpha),
        "requested_count": requested_count,
        "proposal_count": proposal_count,
        "accepted_count": accepted_count,
        "fallback_count": fallback_count,
        "record_states": [record.state_sha256 for record in records],
    }
    return VariantBatch(
        variants=variants,
        encoder_images=encoder_images,
        records=tuple(records),
        proposal_count=proposal_count,
        accepted_count=accepted_count,
        fallback_count=fallback_count,
        acceptance_rate=acceptance_rate,
        state_sha256=_state_digest(payload, (*variants, *encoder_images)),
    )


def evaluate_candidate_acceptance(
    domain_counts: Mapping[str, tuple[int, int]],
    *,
    minimum_overall: float = 0.80,
    minimum_domain: float = 0.60,
) -> AcceptanceAudit:
    """Apply registered overall and per-inner-domain proposal acceptance gates."""

    if (
        not isinstance(domain_counts, Mapping)
        or not domain_counts
        or type(minimum_overall) is not float
        or type(minimum_domain) is not float
        or not 0.0 <= minimum_overall <= 1.0
        or not 0.0 <= minimum_domain <= 1.0
    ):
        raise ValueError("candidate acceptance inputs are invalid")
    rates: list[tuple[str, float]] = []
    accepted_total = 0
    proposed_total = 0
    for domain, counts in domain_counts.items():
        if (
            type(domain) is not str
            or not domain
            or not isinstance(counts, tuple)
            or len(counts) != 2
            or any(type(value) is not int for value in counts)
        ):
            raise ValueError("domain acceptance counts are invalid")
        accepted, proposed = counts
        if proposed < 1 or accepted < 0 or accepted > proposed:
            raise ValueError("domain acceptance counts are invalid")
        accepted_total += accepted
        proposed_total += proposed
        rates.append((domain, accepted / proposed))
    overall = accepted_total / proposed_total
    failed = tuple(domain for domain, rate in rates if rate < minimum_domain)
    return AcceptanceAudit(
        eligible=overall >= minimum_overall and not failed,
        overall_rate=overall,
        domain_rates=tuple(rates),
        failed_domains=failed,
    )


__all__ = [
    "AcceptanceAudit",
    "MorphologyThresholds",
    "VariantBatch",
    "VariantRecord",
    "build_variant",
    "build_variant_batch",
    "evaluate_candidate_acceptance",
]
