from __future__ import annotations

import pytest

from cmc_bbdm.damage_response.post_cai import audit_post_cai_images
from cmc_bbdm.damage_response.sources import OfficialFileRecord, SourceError


def _record(filename: str, file_id: str) -> OfficialFileRecord:
    return OfficialFileRecord(
        dataset_id="8scdmfdcfb",
        file_id=file_id,
        filename=filename,
        folder="3_Specimen image",
        relative_path=f"8scdmfdcfb/v3/3_Specimen image/{filename}",
        sha256=("a" if "front" in filename else "b") * 64,
        size=100,
        version=3,
    )


def test_post_cai_audit_requires_exact_front_and_back_views() -> None:
    rows = audit_post_cai_images(
        (_record("c8-2_front.jpg", "front"), _record("c8-2_back.jpg", "back")),
        raw_specimen_ids={"c8-2"},
    )

    assert [(row.specimen_id, row.view) for row in rows] == [
        ("c8-2", "back"),
        ("c8-2", "front"),
    ]
    assert all(row.input_forbidden for row in rows)
    assert all(row.integrity_status == "REMOTE_OFFICIAL_HASH_BOUND" for row in rows)


def test_post_cai_audit_rejects_missing_view() -> None:
    with pytest.raises(SourceError, match="front/back"):
        audit_post_cai_images(
            (_record("c8-2_front.jpg", "front"),),
            raw_specimen_ids={"c8-2"},
        )


def test_post_cai_audit_rejects_identity_without_raw_trace() -> None:
    with pytest.raises(SourceError, match="raw identity"):
        audit_post_cai_images(
            (
                _record("c8-3_front.jpg", "front"),
                _record("c8-3_back.jpg", "back"),
            ),
            raw_specimen_ids={"c8-2"},
        )
