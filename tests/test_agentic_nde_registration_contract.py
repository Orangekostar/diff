from __future__ import annotations

import pytest

from cmc_bbdm.agentic_nde.contracts import (
    PRIMARY_COUNTS,
    EvidenceClass,
    EvidenceRole,
    FrameGeometry,
    Orientation,
    validate_evidence_roles,
)


def test_primary_roster_is_frozen() -> None:
    assert PRIMARY_COUNTS == {
        "74t7kcdgkr": 45,
        "cgtnjyggtm": 49,
        "w68dtmpfyf": 43,
        "xcmzfsbd9t": 59,
        "yfxyg8jm46": 42,
        "ykhs7s2dck": 38,
    }
    assert sum(PRIMARY_COUNTS.values()) == 276


def test_registration_enums_are_closed() -> None:
    assert {item.value for item in EvidenceClass} == {
        "A_DIRECT_METADATA",
        "B_GEOMETRY_ONLY",
        "C_SOURCE_ONLY_LEARNED",
    }
    assert len(Orientation) == 8


def test_frame_geometry_rejects_invalid_extent() -> None:
    with pytest.raises(ValueError, match="positive"):
        FrameGeometry(width_px=0, height_px=10, width_mm=75.0, height_mm=75.0)


def test_forbidden_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_evidence_roles((EvidenceRole.SURFACE_METADATA, EvidenceRole.CAI))


def test_source_metadata_role_is_accepted() -> None:
    validate_evidence_roles((EvidenceRole.SURFACE_METADATA, EvidenceRole.DATASET_METADATA))
