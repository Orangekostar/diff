"""Immutable authority for the author-supplied surface registration rule."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .contracts import EvidenceRole, Orientation

USER_ATTESTED_SOURCE = (
    "USER_ATTESTED_PERSONAL_COMMUNICATION_WITH_DATASET_AUTHOR"
)
VERBATIM_STATEMENT = (
    "hasebe所有表面rgb的png图像顺时针旋转90度就是扫描图像jpg，\n"
    "比如Q24-7astm.png顺时针旋转90度得到Q24-7astm.jpg，\n"
    "其外边框的大小未做裁切，只是比例不同"
)
EXPECTED_STATEMENT_SHA256 = (
    "3560662d4509ea3e059d597cedca15950cce02f706a992330b161381acfba6ba"
)
OUTER_FRAME_CROP = "NONE_AT_SPECIMEN_FRAME"
MAPPING_BASIS = "AUTHOR_FULL_FRAME_PIXEL_CORRESPONDENCE"
EXAMPLE_SURFACE = "Q24-7astm.png"
EXAMPLE_SCAN = "Q24-7astm.jpg"
_HEX = frozenset("0123456789abcdef")
_EVIDENCE_ROLES = (
    EvidenceRole.AUTHOR_CORRESPONDENCE,
    EvidenceRole.DATASET_METADATA,
    EvidenceRole.SURFACE_METADATA,
)


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and not set(value) - _HEX


@dataclass(frozen=True, slots=True)
class AuthorRegistrationAuthority:
    """Closed record that cannot carry C-scan pixels or outcome evidence."""

    schema_version: int
    source_type: str
    verbatim_statement: str
    statement_sha256: str
    orientation: Orientation
    outer_frame_crop: str
    mapping_basis: str
    example_surface: str
    example_scan: str
    physical_mm_used_for_cross_modal_mapping: bool
    original_artifact_sha256: str | None
    evidence_roles: tuple[EvidenceRole, ...]

    def __post_init__(self) -> None:
        statement_hash = hashlib.sha256(
            self.verbatim_statement.encode("utf-8")
        ).hexdigest()
        if (
            self.schema_version != 1
            or self.source_type != USER_ATTESTED_SOURCE
            or self.verbatim_statement != VERBATIM_STATEMENT
            or self.statement_sha256 != EXPECTED_STATEMENT_SHA256
            or statement_hash != EXPECTED_STATEMENT_SHA256
            or self.orientation is not Orientation.ROT90
            or self.outer_frame_crop != OUTER_FRAME_CROP
            or self.mapping_basis != MAPPING_BASIS
            or self.example_surface != EXAMPLE_SURFACE
            or self.example_scan != EXAMPLE_SCAN
            or self.physical_mm_used_for_cross_modal_mapping is not False
            or self.evidence_roles != _EVIDENCE_ROLES
        ):
            raise ValueError("author registration authority is invalid")
        if self.original_artifact_sha256 is not None and not _is_sha256(
            self.original_artifact_sha256
        ):
            raise ValueError("original artifact SHA-256 is invalid")

    def constructor_dict(self) -> dict[str, object]:
        """Return exact constructor values for contract mutation tests."""

        return {
            "schema_version": self.schema_version,
            "source_type": self.source_type,
            "verbatim_statement": self.verbatim_statement,
            "statement_sha256": self.statement_sha256,
            "orientation": self.orientation,
            "outer_frame_crop": self.outer_frame_crop,
            "mapping_basis": self.mapping_basis,
            "example_surface": self.example_surface,
            "example_scan": self.example_scan,
            "physical_mm_used_for_cross_modal_mapping": (
                self.physical_mm_used_for_cross_modal_mapping
            ),
            "original_artifact_sha256": self.original_artifact_sha256,
            "evidence_roles": self.evidence_roles,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_type": self.source_type,
            "verbatim_statement": self.verbatim_statement,
            "statement_sha256": self.statement_sha256,
            "orientation": self.orientation.value,
            "outer_frame_crop": self.outer_frame_crop,
            "mapping_basis": self.mapping_basis,
            "example_surface": self.example_surface,
            "example_scan": self.example_scan,
            "physical_mm_used_for_cross_modal_mapping": (
                self.physical_mm_used_for_cross_modal_mapping
            ),
            "original_artifact_sha256": self.original_artifact_sha256,
            "evidence_roles": [role.value for role in self.evidence_roles],
        }


def build_author_registration_authority(
    *, original_artifact_sha256: str | None = None
) -> AuthorRegistrationAuthority:
    """Build the one authorized interpretation of the attested statement."""

    return AuthorRegistrationAuthority(
        schema_version=1,
        source_type=USER_ATTESTED_SOURCE,
        verbatim_statement=VERBATIM_STATEMENT,
        statement_sha256=EXPECTED_STATEMENT_SHA256,
        orientation=Orientation.ROT90,
        outer_frame_crop=OUTER_FRAME_CROP,
        mapping_basis=MAPPING_BASIS,
        example_surface=EXAMPLE_SURFACE,
        example_scan=EXAMPLE_SCAN,
        physical_mm_used_for_cross_modal_mapping=False,
        original_artifact_sha256=original_artifact_sha256,
        evidence_roles=_EVIDENCE_ROLES,
    )


__all__ = [
    "EXPECTED_STATEMENT_SHA256",
    "MAPPING_BASIS",
    "USER_ATTESTED_SOURCE",
    "VERBATIM_STATEMENT",
    "AuthorRegistrationAuthority",
    "build_author_registration_authority",
]
