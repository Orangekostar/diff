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


__all__ = [
    "NOT_AUTHORIZED_STAGES",
    "PRIMARY_COUNTS",
    "EvidenceClass",
    "EvidenceRole",
    "FrameGeometry",
    "Orientation",
    "P0Decision",
    "P0GateFacts",
    "StageStatus",
    "decide_p0",
    "validate_evidence_roles",
]
