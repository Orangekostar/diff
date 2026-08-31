from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest

from cmc_bbdm.agentic_nde.author_authority import (
    EXPECTED_STATEMENT_SHA256,
    MAPPING_BASIS,
    USER_ATTESTED_SOURCE,
    VERBATIM_STATEMENT,
    AuthorRegistrationAuthority,
    build_author_registration_authority,
)
from cmc_bbdm.agentic_nde.contracts import EvidenceRole, Orientation


def test_author_authority_binds_exact_user_attested_statement() -> None:
    authority = build_author_registration_authority()

    assert authority.schema_version == 1
    assert authority.source_type == USER_ATTESTED_SOURCE
    assert authority.verbatim_statement == VERBATIM_STATEMENT
    assert authority.statement_sha256 == EXPECTED_STATEMENT_SHA256
    assert authority.orientation is Orientation.ROT90
    assert authority.outer_frame_crop == "NONE_AT_SPECIMEN_FRAME"
    assert authority.mapping_basis == MAPPING_BASIS
    assert authority.example_surface == "Q24-7astm.png"
    assert authority.example_scan == "Q24-7astm.jpg"
    assert authority.physical_mm_used_for_cross_modal_mapping is False
    assert authority.original_artifact_sha256 is None
    assert authority.evidence_roles == (
        EvidenceRole.AUTHOR_CORRESPONDENCE,
        EvidenceRole.DATASET_METADATA,
        EvidenceRole.SURFACE_METADATA,
    )


def test_author_authority_is_immutable_and_hashable() -> None:
    authority = build_author_registration_authority()

    assert hash(authority) == hash(authority)
    with pytest.raises(FrozenInstanceError):
        authority.mapping_basis = "changed"  # type: ignore[misc]


def test_author_authority_accepts_only_a_valid_optional_archive_hash() -> None:
    digest = "a" * 64
    assert (
        build_author_registration_authority(
            original_artifact_sha256=digest
        ).original_artifact_sha256
        == digest
    )
    with pytest.raises(ValueError, match="original artifact SHA-256"):
        build_author_registration_authority(original_artifact_sha256="invalid")


def test_author_authority_api_has_no_result_or_cscan_inputs() -> None:
    forbidden = {
        "cscan_pixels",
        "cai",
        "mechanical_value",
        "oracle_action",
        "damage_mask",
        "damage_centroid",
        "target_domain_label",
        "manual_alignment",
    }
    constructor = set(inspect.signature(AuthorRegistrationAuthority).parameters)
    builder = set(inspect.signature(build_author_registration_authority).parameters)

    assert forbidden.isdisjoint(constructor)
    assert forbidden.isdisjoint(builder)


def test_author_authority_rejects_any_semantic_mutation() -> None:
    authority = build_author_registration_authority()
    payload = authority.as_dict()

    assert payload["orientation"] == "ROT90"
    assert payload["physical_mm_used_for_cross_modal_mapping"] is False
    assert not any(
        token in payload
        for token in (
            "cscan_pixels",
            "cai",
            "oracle_value",
            "damage_mask",
            "damage_centroid",
        )
    )
    with pytest.raises(ValueError, match="author registration authority"):
        AuthorRegistrationAuthority(
            **{
                **authority.constructor_dict(),
                "orientation": Orientation.IDENTITY,
            }
        )
