"""Closed P0 contracts for evidence, geometry, and stage authorization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

PRIMARY_COUNTS = MappingProxyType(
    {
        "74t7kcdgkr": 45,
        "cgtnjyggtm": 49,
        "w68dtmpfyf": 43,
        "xcmzfsbd9t": 59,
        "yfxyg8jm46": 42,
        "ykhs7s2dck": 38,
    }
)


class EvidenceClass(str, Enum):
    """Permitted transform-evidence classes, ordered by preference."""

    A_DIRECT_METADATA = "A_DIRECT_METADATA"
    B_GEOMETRY_ONLY = "B_GEOMETRY_ONLY"
    C_SOURCE_ONLY_LEARNED = "C_SOURCE_ONLY_LEARNED"


class EvidenceRole(str, Enum):
    """Typed roles prevent target evidence entering registration."""

    AUTHOR_CORRESPONDENCE = "AUTHOR_CORRESPONDENCE"
    SURFACE_METADATA = "SURFACE_METADATA"
    DATASET_METADATA = "DATASET_METADATA"
    SPECIMEN_GEOMETRY = "SPECIMEN_GEOMETRY"
    INSTRUMENT_COORDINATES = "INSTRUMENT_COORDINATES"
    SOURCE_ONLY_REGISTRATION = "SOURCE_ONLY_REGISTRATION"
    HIDDEN_CSCAN_PIXELS = "HIDDEN_CSCAN_PIXELS"
    CSCAN_MASK = "CSCAN_MASK"
    DAMAGE_CENTROID = "DAMAGE_CENTROID"
    CAI = "CAI"
    ORACLE_VALUE = "ORACLE_VALUE"
    TARGET_DOMAIN_LABEL = "TARGET_DOMAIN_LABEL"
    MANUAL_TARGET_ALIGNMENT = "MANUAL_TARGET_ALIGNMENT"


_FORBIDDEN_ROLES = frozenset(
    {
        EvidenceRole.HIDDEN_CSCAN_PIXELS,
        EvidenceRole.CSCAN_MASK,
        EvidenceRole.DAMAGE_CENTROID,
        EvidenceRole.CAI,
        EvidenceRole.ORACLE_VALUE,
        EvidenceRole.TARGET_DOMAIN_LABEL,
        EvidenceRole.MANUAL_TARGET_ALIGNMENT,
    }
)


class Orientation(str, Enum):
    """The eight orientation-preserving/reflection cases of a rectangle."""

    IDENTITY = "IDENTITY"
    ROT90 = "ROT90"
    ROT180 = "ROT180"
    ROT270 = "ROT270"
    FLIP_X = "FLIP_X"
    FLIP_Y = "FLIP_Y"
    TRANSPOSE = "TRANSPOSE"
    ANTI_TRANSPOSE = "ANTI_TRANSPOSE"


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    """Native pixel geometry and declared physical extent of one frame."""

    width_px: int
    height_px: int
    width_mm: float
    height_mm: float

    def __post_init__(self) -> None:
        if (
            type(self.width_px) is not int
            or type(self.height_px) is not int
            or self.width_px <= 0
            or self.height_px <= 0
            or not math.isfinite(float(self.width_mm))
            or not math.isfinite(float(self.height_mm))
            or self.width_mm <= 0
            or self.height_mm <= 0
        ):
            raise ValueError("frame dimensions and extents must be positive")
        if self.width_px < 2 or self.height_px < 2:
            raise ValueError("frame pixel dimensions must be at least 2")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "width_px": self.width_px,
            "height_px": self.height_px,
            "width_mm": float(self.width_mm),
            "height_mm": float(self.height_mm),
        }


def validate_evidence_roles(roles: tuple[EvidenceRole, ...]) -> None:
    """Reject any role that can reveal the held-out mechanical target."""

    if not roles or any(type(role) is not EvidenceRole for role in roles):
        raise ValueError("registration evidence roles are invalid")
    forbidden = sorted(role.value for role in roles if role in _FORBIDDEN_ROLES)
    if forbidden:
        raise ValueError(f"forbidden registration evidence: {','.join(forbidden)}")
    if len(set(roles)) != len(roles):
        raise ValueError("registration evidence roles contain duplicates")


class StageStatus(str, Enum):
    """Closed stage-decision vocabulary."""

    P0_GO = "P0_REGISTRATION_GO"
    P0_SPATIAL_REGISTRATION_NO_GO = "P0_SPATIAL_REGISTRATION_NO_GO"
    P0_IDENTITY_AUTHORITY_NO_GO = "P0_IDENTITY_AUTHORITY_NO_GO"
    NOT_RUN_NOT_AUTHORIZED = "NOT_RUN_NOT_AUTHORIZED"


class P0RStatus(str, Enum):
    """Closed decision vocabulary for the author-registration re-audit."""

    GO = "P0R_AUTHOR_REGISTRATION_GO"
    NO_GO = "P0R_AUTHOR_REGISTRATION_NO_GO"
    CONFLICT = "P0R_AUTHOR_EVIDENCE_CONFLICT"
    PROVENANCE_UNRESOLVED = "P0R_PROCESSING_PROVENANCE_UNRESOLVED"


NOT_AUTHORIZED_STAGES = (
    ("P1", StageStatus.NOT_RUN_NOT_AUTHORIZED.value),
    ("P2", StageStatus.NOT_RUN_NOT_AUTHORIZED.value),
    ("P3", StageStatus.NOT_RUN_NOT_AUTHORIZED.value),
    ("P4", StageStatus.NOT_RUN_NOT_AUTHORIZED.value),
)


@dataclass(frozen=True, slots=True)
class P0GateFacts:
    authorized_by_domain: Mapping[str, int]
    exact_identity_hashes: bool
    orientation_resolved: bool
    deterministic_transform: bool
    deployable_evidence_only: bool
    replay_verified: bool

    def __post_init__(self) -> None:
        counts = dict(self.authorized_by_domain)
        if any(type(key) is not str or type(value) is not int or value < 0 for key, value in counts.items()):
            raise ValueError("authorized registration counts are invalid")
        object.__setattr__(self, "authorized_by_domain", MappingProxyType(counts))
        for value in (
            self.exact_identity_hashes,
            self.orientation_resolved,
            self.deterministic_transform,
            self.deployable_evidence_only,
            self.replay_verified,
        ):
            if type(value) is not bool:
                raise ValueError("P0 gate flags must be boolean")

    def as_dict(self) -> dict[str, Any]:
        return {
            "authorized_by_domain": dict(self.authorized_by_domain),
            "exact_identity_hashes": self.exact_identity_hashes,
            "orientation_resolved": self.orientation_resolved,
            "deterministic_transform": self.deterministic_transform,
            "deployable_evidence_only": self.deployable_evidence_only,
            "replay_verified": self.replay_verified,
        }


@dataclass(frozen=True, slots=True)
class P0Decision:
    status: StageStatus
    reasons: tuple[str, ...]
    downstream: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reasons": list(self.reasons),
            "downstream": {stage: status for stage, status in self.downstream},
        }


def decide_p0(facts: P0GateFacts) -> P0Decision:
    """Apply the preregistered identity, registration, and coverage gate."""

    if not facts.exact_identity_hashes:
        return P0Decision(
            status=StageStatus.P0_IDENTITY_AUTHORITY_NO_GO,
            reasons=("identity_or_hash_binding_failed",),
            downstream=NOT_AUTHORIZED_STAGES,
        )

    reasons: list[str] = []
    if not facts.orientation_resolved:
        reasons.append("orientation_unresolved")
    if not facts.deterministic_transform:
        reasons.append("deterministic_transform_unavailable")
    if not facts.deployable_evidence_only:
        reasons.append("registration_evidence_not_deployable")
    if not facts.replay_verified:
        reasons.append("transform_replay_unverified")

    counts = facts.authorized_by_domain
    for domain, expected in PRIMARY_COUNTS.items():
        actual = counts.get(domain, 0)
        minimum = math.ceil(0.9 * expected)
        if actual < minimum:
            reasons.append(f"coverage_below_90_percent:{domain}:{actual}/{expected}")
    unexpected = sorted(set(counts) - set(PRIMARY_COUNTS))
    if unexpected:
        reasons.append(f"unexpected_primary_domains:{','.join(unexpected)}")
    total = sum(counts.get(domain, 0) for domain in PRIMARY_COUNTS)
    if total < 240:
        reasons.append(f"coverage_below_240_total:{total}/276")

    if reasons:
        return P0Decision(
            status=StageStatus.P0_SPATIAL_REGISTRATION_NO_GO,
            reasons=tuple(reasons),
            downstream=NOT_AUTHORIZED_STAGES,
        )
    return P0Decision(status=StageStatus.P0_GO, reasons=(), downstream=())


@dataclass(frozen=True, slots=True)
class P0RGateFacts:
    """Machine-recomputable facts for the separate P0R gate."""

    authorized_by_domain: Mapping[str, int]
    exact_identity_hashes: bool
    author_statement_bound: bool
    global_orientation_rot90: bool
    all_panels_resolved: bool
    processing_provenance_deterministic: bool
    no_unsupported_rotation_reflection: bool
    composed_transform_replayable: bool
    no_result_driven_orientation: bool
    author_evidence_conflict: bool
    processing_provenance_unresolved: bool

    def __post_init__(self) -> None:
        counts = dict(self.authorized_by_domain)
        if any(
            type(key) is not str or type(value) is not int or value < 0
            for key, value in counts.items()
        ):
            raise ValueError("authorized P0R registration counts are invalid")
        object.__setattr__(self, "authorized_by_domain", MappingProxyType(counts))
        for value in (
            self.exact_identity_hashes,
            self.author_statement_bound,
            self.global_orientation_rot90,
            self.all_panels_resolved,
            self.processing_provenance_deterministic,
            self.no_unsupported_rotation_reflection,
            self.composed_transform_replayable,
            self.no_result_driven_orientation,
            self.author_evidence_conflict,
            self.processing_provenance_unresolved,
        ):
            if type(value) is not bool:
                raise ValueError("P0R gate flags must be boolean")

    def as_dict(self) -> dict[str, Any]:
        return {
            "authorized_by_domain": dict(self.authorized_by_domain),
            "exact_identity_hashes": self.exact_identity_hashes,
            "author_statement_bound": self.author_statement_bound,
            "global_orientation_rot90": self.global_orientation_rot90,
            "all_panels_resolved": self.all_panels_resolved,
            "processing_provenance_deterministic": (
                self.processing_provenance_deterministic
            ),
            "no_unsupported_rotation_reflection": (
                self.no_unsupported_rotation_reflection
            ),
            "composed_transform_replayable": self.composed_transform_replayable,
            "no_result_driven_orientation": self.no_result_driven_orientation,
            "author_evidence_conflict": self.author_evidence_conflict,
            "processing_provenance_unresolved": (
                self.processing_provenance_unresolved
            ),
        }


@dataclass(frozen=True, slots=True)
class P0RDecision:
    status: P0RStatus
    reasons: tuple[str, ...]
    downstream_registration_status: StageStatus
    p1_authorized: bool
    downstream: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reasons": list(self.reasons),
            "downstream_registration_status": self.downstream_registration_status.value,
            "p1_authorized": self.p1_authorized,
            "downstream": {stage: status for stage, status in self.downstream},
        }


def _blocked_p0r(status: P0RStatus, reasons: tuple[str, ...]) -> P0RDecision:
    return P0RDecision(
        status=status,
        reasons=reasons,
        downstream_registration_status=StageStatus.P0_SPATIAL_REGISTRATION_NO_GO,
        p1_authorized=False,
        downstream=NOT_AUTHORIZED_STAGES,
    )


def decide_p0r(facts: P0RGateFacts) -> P0RDecision:
    """Apply the preregistered author-registration evidence and coverage gate."""

    if facts.author_evidence_conflict:
        return _blocked_p0r(P0RStatus.CONFLICT, ("author_evidence_conflict",))
    if facts.processing_provenance_unresolved:
        return _blocked_p0r(
            P0RStatus.PROVENANCE_UNRESOLVED,
            ("processing_provenance_unresolved",),
        )

    boolean_requirements = (
        (facts.exact_identity_hashes, "identity_or_hash_binding_failed"),
        (facts.author_statement_bound, "author_statement_not_bound"),
        (facts.global_orientation_rot90, "global_rot90_not_fixed"),
        (facts.all_panels_resolved, "specimen_panel_unresolved"),
        (
            facts.processing_provenance_deterministic,
            "processing_provenance_not_deterministic",
        ),
        (
            facts.no_unsupported_rotation_reflection,
            "unsupported_rotation_or_reflection",
        ),
        (facts.composed_transform_replayable, "composed_transform_not_replayable"),
        (
            facts.no_result_driven_orientation,
            "result_driven_orientation_not_excluded",
        ),
    )
    reasons = [reason for passed, reason in boolean_requirements if not passed]
    counts = facts.authorized_by_domain
    for domain, expected in PRIMARY_COUNTS.items():
        actual = counts.get(domain, 0)
        minimum = math.ceil(0.9 * expected)
        if actual < minimum:
            reasons.append(f"coverage_below_90_percent:{domain}:{actual}/{expected}")
    unexpected = sorted(set(counts) - set(PRIMARY_COUNTS))
    if unexpected:
        reasons.append(f"unexpected_primary_domains:{','.join(unexpected)}")
    total = sum(counts.get(domain, 0) for domain in PRIMARY_COUNTS)
    if total < 240:
        reasons.append(f"coverage_below_240_total:{total}/276")
    if reasons:
        return _blocked_p0r(P0RStatus.NO_GO, tuple(reasons))
    return P0RDecision(
        status=P0RStatus.GO,
        reasons=(),
        downstream_registration_status=StageStatus.P0_GO,
        p1_authorized=True,
        downstream=(),
    )


__all__ = [
    "NOT_AUTHORIZED_STAGES",
    "PRIMARY_COUNTS",
    "EvidenceClass",
    "EvidenceRole",
    "FrameGeometry",
    "Orientation",
    "P0Decision",
    "P0GateFacts",
    "P0RDecision",
    "P0RGateFacts",
    "P0RStatus",
    "StageStatus",
    "decide_p0",
    "decide_p0r",
    "validate_evidence_roles",
]
