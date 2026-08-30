from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from cmc_bbdm.damage_response.contracts import POST_CAI_IMAGE_INPUT_FORBIDDEN
from cmc_bbdm.damage_response.sources import OfficialFileRecord, SourceError

_IMAGE_NAME_RE = re.compile(
    r"(?P<specimen>(?:[cq](?:8|16|24)|r(?:0|45))-\d+t?)_"
    r"(?P<view>front|back)\.jpg",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PostCaiImageAudit:
    specimen_id: str
    view: str
    file_id: str
    relative_path: str
    sha256: str
    size: int
    integrity_status: str
    input_forbidden: bool


def audit_post_cai_images(
    records: Iterable[OfficialFileRecord], *, raw_specimen_ids: AbstractSet[str]
) -> tuple[PostCaiImageAudit, ...]:
    """Audit official front/back membership without treating images as inputs."""

    raw_ids = {identity.strip().casefold() for identity in raw_specimen_ids}
    if not raw_ids or "" in raw_ids:
        raise SourceError("raw identity set is empty or invalid")
    views: dict[str, dict[str, OfficialFileRecord]] = defaultdict(dict)
    for record in records:
        if record.folder != "3_Specimen image":
            raise SourceError(f"non-image official record in post-CAI audit: {record.filename}")
        match = _IMAGE_NAME_RE.fullmatch(record.filename)
        if match is None:
            raise SourceError(f"post-CAI image filename is not registered: {record.filename}")
        specimen_id = match.group("specimen").casefold()
        view = match.group("view").casefold()
        if specimen_id not in raw_ids:
            raise SourceError(
                f"post-CAI image has no exact raw identity: {record.filename}"
            )
        if view in views[specimen_id]:
            raise SourceError(f"duplicate post-CAI image view: {record.filename}")
        views[specimen_id][view] = record

    if set(views) != raw_ids:
        missing = sorted(raw_ids - views.keys())
        raise SourceError(f"post-CAI image identities lack front/back records: {missing!r}")
    incomplete = sorted(
        specimen_id
        for specimen_id, by_view in views.items()
        if set(by_view) != {"front", "back"}
    )
    if incomplete:
        raise SourceError(
            f"post-CAI image identities lack exact front/back views: {incomplete!r}"
        )

    output: list[PostCaiImageAudit] = []
    for specimen_id in sorted(views):
        for view in ("back", "front"):
            record = views[specimen_id][view]
            output.append(
                PostCaiImageAudit(
                    specimen_id=specimen_id,
                    view=view,
                    file_id=record.file_id,
                    relative_path=record.relative_path,
                    sha256=record.sha256,
                    size=record.size,
                    integrity_status="REMOTE_OFFICIAL_HASH_BOUND",
                    input_forbidden=POST_CAI_IMAGE_INPUT_FORBIDDEN,
                )
            )
    return tuple(output)
