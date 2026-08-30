from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from cmc_bbdm.damage_response.sources import (
    IMPACTOR_CATEGORIES,
    LAMINATE_CATEGORIES,
    DesignMetadata,
)

PRIMARY_TARGET_FIELDS = (
    "extension_peak_mm",
    "slope_u20_u60_mpa_per_mm",
    "normalized_prepeak_auc",
)
STRENGTH_REFERENCE_FIELD = "published_cai_strength_mpa"
DESIGN_NUMERIC_FIELDS = ("ply_count", "width_mm", "thickness_mm")
DESIGN_CATEGORICAL_FIELDS = ("laminate_type", "impactor")
FORBIDDEN_RESPONSE_FIELDS = (
    "true_cai_trace",
    "true_peak_strength",
    "derived_response",
    "response_curve",
    "post_cai_image",
)
P1_REDUNDANCY_VIEWS = {
    "strength_only": (STRENGTH_REFERENCE_FIELD,),
    "strength_plus_design": (
        STRENGTH_REFERENCE_FIELD,
        *DESIGN_NUMERIC_FIELDS,
        *DESIGN_CATEGORICAL_FIELDS,
    ),
}
DESIGN_FEATURE_NAMES = (
    *DESIGN_NUMERIC_FIELDS,
    *(f"laminate_type={value}" for value in LAMINATE_CATEGORIES),
    *(f"impactor={value}" for value in IMPACTOR_CATEGORIES),
)


class FeatureViewError(ValueError):
    """Raised when a P1 field role or fold-local view is invalid."""


class FieldRole(str, Enum):
    IDENTITY = "IDENTITY"
    SPLIT = "SPLIT"
    TARGET = "TARGET"
    REDUNDANCY_REFERENCE = "REDUNDANCY_REFERENCE"
    DEPLOYABLE_DESIGN = "DEPLOYABLE_DESIGN"
    PRIVILEGED_DESIGN = "PRIVILEGED_DESIGN"
    FORBIDDEN = "FORBIDDEN"


_FIELD_ROLES = {
    "specimen_id": FieldRole.IDENTITY,
    "domain_id": FieldRole.SPLIT,
    **{name: FieldRole.TARGET for name in PRIMARY_TARGET_FIELDS},
    STRENGTH_REFERENCE_FIELD: FieldRole.REDUNDANCY_REFERENCE,
    "ply_count": FieldRole.DEPLOYABLE_DESIGN,
    "width_mm": FieldRole.DEPLOYABLE_DESIGN,
    "thickness_mm": FieldRole.DEPLOYABLE_DESIGN,
    "laminate_type": FieldRole.DEPLOYABLE_DESIGN,
    "impactor": FieldRole.PRIVILEGED_DESIGN,
    "impact_energy": FieldRole.PRIVILEGED_DESIGN,
    "energy_per_thickness_j_per_mm": FieldRole.PRIVILEGED_DESIGN,
    **{name: FieldRole.FORBIDDEN for name in FORBIDDEN_RESPONSE_FIELDS},
}


def field_role(name: str) -> FieldRole:
    """Return the frozen P1 role for one named field."""

    try:
        return _FIELD_ROLES[name]
    except (KeyError, TypeError) as error:
        raise FeatureViewError(f"unknown P1 field: {name!r}") from error


def validate_p1_redundancy_view(
    view_name: str, fields: Iterable[str]
) -> tuple[str, ...]:
    """Require exact membership for a registered P1 diagnostic view."""

    expected = P1_REDUNDANCY_VIEWS.get(view_name)
    if expected is None:
        raise FeatureViewError(f"unknown P1 redundancy view: {view_name!r}")
    observed = tuple(fields)
    for name in observed:
        role = field_role(name)
        if role in {FieldRole.TARGET, FieldRole.FORBIDDEN}:
            raise FeatureViewError(f"forbidden P1 redundancy field: {name}")
    if observed != expected:
        raise FeatureViewError(
            f"P1 redundancy view {view_name!r} must equal {expected!r}"
        )
    return observed


def _readonly(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype="<f8").reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _validated_records(records: Iterable[DesignMetadata]) -> tuple[DesignMetadata, ...]:
    values = tuple(records)
    if not values:
        raise FeatureViewError("design metadata is empty")
    specimen_ids: set[str] = set()
    for record in values:
        if not isinstance(record, DesignMetadata):
            raise FeatureViewError("design metadata record type changed")
        if not record.specimen_id or record.specimen_id in specimen_ids:
            raise FeatureViewError("design specimen identities are empty or duplicate")
        if not record.domain_id:
            raise FeatureViewError("design domain identity is empty")
        if record.laminate_type not in LAMINATE_CATEGORIES:
            raise FeatureViewError(
                f"unknown laminate category: {record.laminate_type!r}"
            )
        if record.impactor not in IMPACTOR_CATEGORIES:
            raise FeatureViewError(f"unknown impactor category: {record.impactor!r}")
        numeric = (record.ply_count, record.width_mm, record.thickness_mm)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in numeric
        ):
            raise FeatureViewError(
                f"invalid numeric design metadata: {record.specimen_id}"
            )
        specimen_ids.add(record.specimen_id)
    return values


def _numeric_matrix(records: Sequence[DesignMetadata]) -> np.ndarray:
    return np.asarray(
        [
            (float(record.ply_count), record.width_mm, record.thickness_mm)
            for record in records
        ],
        dtype=np.float64,
    )


def _design_matrix(
    records: Sequence[DesignMetadata], means: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    numeric = (_numeric_matrix(records) - means) / scales
    laminate = np.asarray(
        [
            [float(record.laminate_type == category) for category in LAMINATE_CATEGORIES]
            for record in records
        ],
        dtype=np.float64,
    )
    impactor = np.asarray(
        [
            [float(record.impactor == category) for category in IMPACTOR_CATEGORIES]
            for record in records
        ],
        dtype=np.float64,
    )
    return np.column_stack((numeric, laminate, impactor))


@dataclass(frozen=True)
class FoldLocalDesignEncoder:
    held_out_domain: str
    fit_specimen_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    state_sha256: str

    def transform(self, records: Iterable[DesignMetadata]) -> np.ndarray:
        values = _validated_records(records)
        return _readonly(_design_matrix(values, self.means, self.scales))


def fit_fold_local_design_encoder(
    records: Iterable[DesignMetadata], held_out_domain: str
) -> FoldLocalDesignEncoder:
    """Fit numeric design scaling on source domains only; categories are fixed."""

    values = _validated_records(records)
    if not isinstance(held_out_domain, str) or not held_out_domain.strip():
        raise FeatureViewError("held-out domain must be nonempty")
    target_domain = held_out_domain.strip().casefold()
    query = tuple(record for record in values if record.domain_id == target_domain)
    source = tuple(record for record in values if record.domain_id != target_domain)
    if not query:
        raise FeatureViewError(f"held-out domain is absent: {target_domain}")
    if not source:
        raise FeatureViewError("fold has no source-domain design metadata")

    source_numeric = _numeric_matrix(source)
    means = np.mean(source_numeric, axis=0, dtype=np.float64)
    scales = np.std(source_numeric, axis=0, ddof=0, dtype=np.float64)
    scales[scales == 0.0] = 1.0
    immutable_means = _readonly(means)
    immutable_scales = _readonly(scales)
    fit_specimen_ids = tuple(record.specimen_id for record in source)

    digest = hashlib.sha256()
    digest.update(target_domain.encode("utf-8"))
    digest.update(b"\0")
    for specimen_id in fit_specimen_ids:
        digest.update(specimen_id.encode("utf-8"))
        digest.update(b"\0")
    for name in DESIGN_FEATURE_NAMES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    digest.update(immutable_means.tobytes(order="C"))
    digest.update(immutable_scales.tobytes(order="C"))

    return FoldLocalDesignEncoder(
        held_out_domain=target_domain,
        fit_specimen_ids=fit_specimen_ids,
        feature_names=tuple(DESIGN_FEATURE_NAMES),
        means=immutable_means,
        scales=immutable_scales,
        state_sha256=digest.hexdigest(),
    )
