"""Replay the frozen G0 surface/internal agreement strata for G1 diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

FROZEN_G0_INITIALIZATION_CURVES_SHA256 = (
    "519b96c0cf525cda32ff54e91792524c77fcc858e3fa33b10ee292d3be8141e6"
)
FROZEN_G0_SURFACE_STRATA_SOURCE = (
    "results/inspection_agent/g0/initialization_curves.csv"
)
FROZEN_G0_SURFACE_STRATUM_COUNTS = (
    ("SURFACE_INTERNAL_AGREE", 1),
    ("SURFACE_INTERNAL_PARTIAL", 114),
    ("SURFACE_INTERNAL_MISLEADING", 161),
)
_STRATA = tuple(name for name, _count in FROZEN_G0_SURFACE_STRATUM_COUNTS)


class G1SurfaceStratumError(ValueError):
    """Raised when the frozen G0 surface-stratum authority changes."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class G1SurfaceStratumRecord:
    outer_target: str
    specimen_id: str
    stratum: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.specimen_id) is not str
            or not self.specimen_id
            or self.stratum not in _STRATA
        ):
            raise G1SurfaceStratumError("surface-stratum record is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-frozen-surface-stratum",
                    "outer_target": self.outer_target,
                    "specimen_id": self.specimen_id,
                    "stratum": self.stratum,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1SurfaceStratumAuthority:
    source: str
    file_sha256: str
    record_count: int
    stratum_counts: tuple[tuple[str, int], ...]
    records_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.source) is not str
            or not self.source
            or not _valid_sha256(self.file_sha256)
            or type(self.record_count) is not int
            or self.record_count <= 0
            or type(self.stratum_counts) is not tuple
            or not self.stratum_counts
            or any(
                name not in _STRATA or type(count) is not int or count < 0
                for name, count in self.stratum_counts
            )
            or sum(count for _name, count in self.stratum_counts)
            != self.record_count
            or not _valid_sha256(self.records_sha256)
        ):
            raise G1SurfaceStratumError("surface-stratum authority is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-frozen-surface-stratum-authority",
                    "source": self.source,
                    "file": self.file_sha256,
                    "record_count": self.record_count,
                    "stratum_counts": self.stratum_counts,
                    "records": self.records_sha256,
                }
            ),
        )


def surface_stratum_records_sha256(
    rows: tuple[G1SurfaceStratumRecord, ...],
) -> str:
    if (
        type(rows) is not tuple
        or not rows
        or any(type(row) is not G1SurfaceStratumRecord for row in rows)
        or len({(row.outer_target, row.specimen_id) for row in rows}) != len(rows)
    ):
        raise G1SurfaceStratumError("surface-stratum record roster is invalid")
    ordered = tuple(
        sorted(rows, key=lambda row: (row.outer_target, row.specimen_id))
    )
    return _json_sha(tuple(row.state_sha256 for row in ordered))


def load_g1_frozen_surface_strata(
    path: str | Path,
) -> tuple[G1SurfaceStratumAuthority, tuple[G1SurfaceStratumRecord, ...]]:
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise G1SurfaceStratumError("surface-stratum authority cannot be read") from error
    file_sha = hashlib.sha256(payload).hexdigest()
    if file_sha != FROZEN_G0_INITIALIZATION_CURVES_SHA256:
        raise G1SurfaceStratumError("frozen G0 surface-stratum SHA changed")
    try:
        decoded = payload.decode("ascii")
        reader = csv.DictReader(decoded.splitlines())
        if reader.fieldnames is None or not {
            "dataset_id",
            "specimen_id",
            "surface_internal_stratum",
        } <= set(reader.fieldnames):
            raise G1SurfaceStratumError("surface-stratum columns changed")
        mapping: dict[tuple[str, str], str] = {}
        for raw in reader:
            key = str(raw["dataset_id"]), str(raw["specimen_id"])
            stratum = str(raw["surface_internal_stratum"])
            if (
                not all(key)
                or stratum not in _STRATA
                or (key in mapping and mapping[key] != stratum)
            ):
                raise G1SurfaceStratumError("surface-stratum rows changed")
            mapping[key] = stratum
    except (UnicodeError, csv.Error, KeyError) as error:
        if isinstance(error, G1SurfaceStratumError):
            raise
        raise G1SurfaceStratumError("surface-stratum authority is invalid") from error
    rows = tuple(
        G1SurfaceStratumRecord(
            outer_target=outer,
            specimen_id=specimen,
            stratum=stratum,
        )
        for (outer, specimen), stratum in sorted(mapping.items())
    )
    counts = Counter(row.stratum for row in rows)
    stratum_counts = tuple((name, counts[name]) for name in _STRATA)
    if len(rows) != 276 or stratum_counts != FROZEN_G0_SURFACE_STRATUM_COUNTS:
        raise G1SurfaceStratumError("frozen G0 surface-stratum roster changed")
    records_sha = surface_stratum_records_sha256(rows)
    return (
        G1SurfaceStratumAuthority(
            source=FROZEN_G0_SURFACE_STRATA_SOURCE,
            file_sha256=file_sha,
            record_count=len(rows),
            stratum_counts=stratum_counts,
            records_sha256=records_sha,
        ),
        rows,
    )


__all__ = [
    "FROZEN_G0_INITIALIZATION_CURVES_SHA256",
    "FROZEN_G0_SURFACE_STRATA_SOURCE",
    "FROZEN_G0_SURFACE_STRATUM_COUNTS",
    "G1SurfaceStratumAuthority",
    "G1SurfaceStratumError",
    "G1SurfaceStratumRecord",
    "load_g1_frozen_surface_strata",
    "surface_stratum_records_sha256",
]
