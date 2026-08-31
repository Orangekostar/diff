from __future__ import annotations

import pytest

from cmc_bbdm.agentic_nde.contracts import (
    EvidenceClass,
    EvidenceRole,
    FrameGeometry,
    Orientation,
)
from cmc_bbdm.agentic_nde.registration import create_transform


@pytest.mark.parametrize(
    "role",
    [
        EvidenceRole.HIDDEN_CSCAN_PIXELS,
        EvidenceRole.CSCAN_MASK,
        EvidenceRole.DAMAGE_CENTROID,
        EvidenceRole.CAI,
        EvidenceRole.ORACLE_VALUE,
        EvidenceRole.TARGET_DOMAIN_LABEL,
        EvidenceRole.MANUAL_TARGET_ALIGNMENT,
    ],
)
def test_forbidden_registration_evidence_is_rejected(role: EvidenceRole) -> None:
    frame = FrameGeometry(width_px=10, height_px=10, width_mm=75.0, height_mm=75.0)
    with pytest.raises(ValueError, match="forbidden"):
        create_transform(
            source=frame,
            destination=frame,
            orientation=Orientation.IDENTITY,
            evidence_class=EvidenceClass.B_GEOMETRY_ONLY,
            evidence_roles=(EvidenceRole.SURFACE_METADATA, role),
            evidence_hashes=("a" * 64,),
        )


def test_class_c_requires_source_only_isolation() -> None:
    frame = FrameGeometry(width_px=10, height_px=10, width_mm=75.0, height_mm=75.0)
    with pytest.raises(ValueError, match="source-only"):
        create_transform(
            source=frame,
            destination=frame,
            orientation=Orientation.IDENTITY,
            evidence_class=EvidenceClass.C_SOURCE_ONLY_LEARNED,
            evidence_roles=(EvidenceRole.SURFACE_METADATA,),
            evidence_hashes=("a" * 64,),
            source_only_isolated=False,
        )
