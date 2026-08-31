"""Deterministic P0R surface-cell crops and spatial controls for P1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import EvidenceClass, EvidenceRole, FrameGeometry, Orientation
from .grid import render_surface_grid
from .registration import create_transform

_HEX = frozenset("0123456789abcdef")
_WRONG_ORIENTATIONS = (
    Orientation.IDENTITY,
    Orientation.ROT180,
    Orientation.ROT270,
    Orientation.FLIP_X,
    Orientation.FLIP_Y,
    Orientation.TRANSPOSE,
    Orientation.ANTI_TRANSPOSE,
)
_BOX_TOLERANCE = 1.0e-9


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and not (set(value) - _HEX)


def _read_csv(path: str | Path, label: str) -> tuple[dict[str, str], ...]:
    source = Path(path).resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(
                reader.fieldnames
            ):
                raise ValueError(f"{label} header is invalid")
            rows = tuple(dict(row) for row in reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"{label} cannot be read") from error
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _relative_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{label} is not a repository-relative path")
    return path


def _integer(value: str, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not an integer") from error
    return number


def _number(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _readonly_boxes(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("surface-cell boxes are invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(shape)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class SurfaceCellRecord:
    specimen_id: str
    dataset_id: str
    surface_path: Path
    surface_sha256: str
    source: FrameGeometry
    destination: FrameGeometry
    evidence_class: EvidenceClass
    evidence_roles: tuple[EvidenceRole, ...]
    evidence_hashes: tuple[str, ...]
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    transform_sha256: str
    cell_boxes: np.ndarray


@dataclass(frozen=True, slots=True)
class SurfaceCellAuthority:
    records: tuple[SurfaceCellRecord, ...]
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    surface_paths: tuple[Path, ...]
    surface_sha256: tuple[str, ...]
    cell_boxes: np.ndarray
    state_sha256: str

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)


def _transform(record: SurfaceCellRecord, orientation: Orientation):
    return create_transform(
        source=record.source,
        destination=record.destination,
        orientation=orientation,
        evidence_class=record.evidence_class,
        evidence_roles=record.evidence_roles,
        evidence_hashes=record.evidence_hashes,
        source_only_isolated=True,
        scale_x=record.scale_x,
        scale_y=record.scale_y,
        offset_x=record.offset_x,
        offset_y=record.offset_y,
    )


def oriented_surface_boxes(
    record: SurfaceCellRecord, orientation: Orientation
) -> np.ndarray:
    """Render all 64 inverse P0R cell boxes under one declared orientation."""

    if type(record) is not SurfaceCellRecord or type(orientation) is not Orientation:
        raise ValueError("surface record and D4 orientation are required")
    rows = render_surface_grid(_transform(record, orientation))
    if tuple(row["cell_id"] for row in rows) != tuple(range(64)):
        raise ValueError("surface-cell roster changed")
    return _readonly_boxes(
        [tuple(float(value) for value in row["surface_box"]) for row in rows],
        (64, 4),
    )


def _record_from_rows(
    surface: dict[str, str],
    registration: dict[str, str],
    grid_rows: tuple[dict[str, str], ...],
) -> SurfaceCellRecord:
    specimen = surface["specimen_id"]
    domain = surface["dataset_id"]
    if (
        registration["specimen_id"] != specimen
        or registration["dataset_id"] != domain
        or surface.get("p0r_roster_status") != "AUTHORIZED"
        or registration.get("status") != "AUTHORIZED"
        or registration.get("orientation") != Orientation.ROT90.value
        or registration.get("mapping_basis")
        != "AUTHOR_FULL_FRAME_PIXEL_CORRESPONDENCE"
        or registration.get("physical_mm_used_for_cross_modal_mapping") != "false"
    ):
        raise ValueError("P0R registration identity changed")
    surface_hash = surface["surface_sha256"]
    transform_hash = registration["transform_sha256"]
    evidence_hashes = tuple(registration["evidence_hashes"].split(";"))
    if (
        not _valid_sha256(surface_hash)
        or not _valid_sha256(transform_hash)
        or not evidence_hashes
        or any(not _valid_sha256(value) for value in evidence_hashes)
    ):
        raise ValueError("P0R hash authority changed")
    try:
        evidence_class = EvidenceClass(registration["evidence_class"])
        evidence_roles = tuple(
            EvidenceRole(value) for value in registration["evidence_roles"].split(";")
        )
    except ValueError as error:
        raise ValueError("P0R evidence vocabulary changed") from error
    source = FrameGeometry(
        _integer(registration["source_width_px"], "source width"),
        _integer(registration["source_height_px"], "source height"),
        1.0,
        1.0,
    )
    destination = FrameGeometry(
        _integer(registration["destination_width_px"], "destination width"),
        _integer(registration["destination_height_px"], "destination height"),
        1.0,
        1.0,
    )
    placeholder = SurfaceCellRecord(
        specimen_id=specimen,
        dataset_id=domain,
        surface_path=_relative_path(surface["impacted_surface_path"], "surface path"),
        surface_sha256=surface_hash,
        source=source,
        destination=destination,
        evidence_class=evidence_class,
        evidence_roles=evidence_roles,
        evidence_hashes=evidence_hashes,
        scale_x=_number(registration["scale_x"], "scale x"),
        scale_y=_number(registration["scale_y"], "scale y"),
        offset_x=_number(registration["offset_x"], "offset x"),
        offset_y=_number(registration["offset_y"], "offset y"),
        transform_sha256=transform_hash,
        cell_boxes=np.empty((0, 4), dtype="<f8"),
    )
    transform = _transform(placeholder, Orientation.ROT90)
    if transform.sha256 != transform_hash:
        raise ValueError("P0R transform hash cannot be replayed")
    boxes = oriented_surface_boxes(placeholder, Orientation.ROT90)
    indexed: dict[int, dict[str, str]] = {}
    for row in grid_rows:
        cell = _integer(row["cell_id"], "grid cell")
        if (
            cell in indexed
            or row["specimen_id"] != specimen
            or row["dataset_id"] != domain
            or row["transform_sha256"] != transform_hash
            or row.get("round_trip_status") != "PASS"
        ):
            raise ValueError("P0R grid mapping identity changed")
        indexed[cell] = row
    if set(indexed) != set(range(64)):
        raise ValueError("P0R grid mapping roster changed")
    for cell, expected in enumerate(boxes):
        row = indexed[cell]
        recorded = np.asarray(
            [
                _number(row["surface_x0"], "surface x0"),
                _number(row["surface_y0"], "surface y0"),
                _number(row["surface_x1"], "surface x1"),
                _number(row["surface_y1"], "surface y1"),
            ],
            dtype=np.float64,
        )
        if (
            _integer(row["row"], "grid row") != cell // 8
            or _integer(row["column"], "grid column") != cell % 8
            or not np.allclose(recorded, expected, rtol=0.0, atol=_BOX_TOLERANCE)
            or _number(row["round_trip_max_abs_error_px"], "round-trip error")
            > _BOX_TOLERANCE
        ):
            raise ValueError("P0R rendered surface grid changed")
    return SurfaceCellRecord(
        specimen_id=placeholder.specimen_id,
        dataset_id=placeholder.dataset_id,
        surface_path=placeholder.surface_path,
        surface_sha256=placeholder.surface_sha256,
        source=placeholder.source,
        destination=placeholder.destination,
        evidence_class=placeholder.evidence_class,
        evidence_roles=placeholder.evidence_roles,
        evidence_hashes=placeholder.evidence_hashes,
        scale_x=placeholder.scale_x,
        scale_y=placeholder.scale_y,
        offset_x=placeholder.offset_x,
        offset_y=placeholder.offset_y,
        transform_sha256=placeholder.transform_sha256,
        cell_boxes=boxes,
    )


def load_surface_cell_authority(
    surface_manifest_path: str | Path,
    registration_path: str | Path,
    grid_mapping_path: str | Path,
) -> SurfaceCellAuthority:
    """Rebuild and validate every authorized P0R surface cell without labels."""

    surfaces = _read_csv(surface_manifest_path, "P0R surface manifest")
    registrations = _read_csv(registration_path, "P0R registration")
    grids = _read_csv(grid_mapping_path, "P0R grid mapping")
    if len(surfaces) != 276 or len(registrations) != 276 or len(grids) != 17_664:
        raise ValueError("P0R surface-cell authority count changed")

    def key(row: dict[str, str]) -> tuple[str, str]:
        return row["dataset_id"], row["specimen_id"]

    registrations_by_key = {key(row): row for row in registrations}
    grids_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in grids:
        grids_by_key[key(row)].append(row)
    surface_keys = tuple(key(row) for row in surfaces)
    if (
        len(set(surface_keys)) != 276
        or len(registrations_by_key) != 276
        or set(surface_keys) != set(registrations_by_key)
        or set(surface_keys) != set(grids_by_key)
    ):
        raise ValueError("P0R surface-cell join changed")
    records = tuple(
        _record_from_rows(
            surface,
            registrations_by_key[key(surface)],
            tuple(grids_by_key[key(surface)]),
        )
        for surface in surfaces
    )
    boxes = _readonly_boxes([record.cell_boxes for record in records], (276, 64, 4))
    payload = {
        "dataset_ids": [record.dataset_id for record in records],
        "specimen_ids": [record.specimen_id for record in records],
        "surface_paths": [record.surface_path.as_posix() for record in records],
        "surface_sha256": [record.surface_sha256 for record in records],
        "transform_sha256": [record.transform_sha256 for record in records],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    digest.update(boxes.tobytes(order="C"))
    return SurfaceCellAuthority(
        records=records,
        specimen_ids=tuple(record.specimen_id for record in records),
        dataset_ids=tuple(record.dataset_id for record in records),
        surface_paths=tuple(record.surface_path for record in records),
        surface_sha256=tuple(record.surface_sha256 for record in records),
        cell_boxes=boxes,
        state_sha256=digest.hexdigest(),
    )


def _axis_bounds(lower: float, upper: float, length: int) -> tuple[int, int]:
    if (
        type(length) is not int
        or length < 2
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower >= upper
        or lower < -_BOX_TOLERANCE
        or upper > length - 1 + _BOX_TOLERANCE
    ):
        raise ValueError("surface crop axis bounds are invalid")
    bounded_lower = min(float(length - 1), max(0.0, lower))
    bounded_upper = min(float(length - 1), max(0.0, upper))
    start = max(0, min(length - 1, math.ceil(bounded_lower)))
    stop = (
        length
        if bounded_upper >= length - 1 - _BOX_TOLERANCE
        else max(1, min(length, math.ceil(bounded_upper)))
    )
    if start >= stop:
        raise ValueError("surface crop rounds to an empty axis")
    return start, stop


def integer_crop_box(
    box: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[int, int, int, int]:
    """Convert a continuous P0R pixel-center box to the frozen half-open crop."""

    if len(box) != 4:
        raise ValueError("surface crop box must have four values")
    x0, y0, x1, y1 = (float(value) for value in box)
    left, right = _axis_bounds(x0, x1, width)
    top, bottom = _axis_bounds(y0, y1, height)
    return left, top, right, bottom


def crop_rgb_patch(
    image: Image.Image, box: tuple[float, float, float, float]
) -> Image.Image:
    """Decode/copy one exact RGB patch without resize, padding, or labels."""

    if not isinstance(image, Image.Image):
        raise TypeError("surface crop requires a Pillow image")
    image.load()
    rgb = image.convert("RGB")
    crop = integer_crop_box(box, width=rgb.width, height=rgb.height)
    return rgb.crop(crop)


def wrong_orientation(
    specimen_id: str, *, dataset_id: str, seed: str
) -> Orientation:
    """Select one preregistered incorrect D4 orientation from immutable identity."""

    if not specimen_id or not dataset_id or not seed:
        raise ValueError("wrong-orientation identity is incomplete")
    token = _sha256_text(f"{seed}|{dataset_id}|{specimen_id}")
    return _WRONG_ORIENTATIONS[int(token[:16], 16) % len(_WRONG_ORIENTATIONS)]


def shuffled_surface_donors(
    specimen_ids: tuple[str, ...], dataset_ids: tuple[str, ...], *, seed: str
) -> tuple[str, ...]:
    """Return a domain-local bijection with a deterministic nonzero offset."""

    if (
        not specimen_ids
        or len(specimen_ids) != len(dataset_ids)
        or len(set(specimen_ids)) != len(specimen_ids)
        or any(not value for value in (*specimen_ids, *dataset_ids))
        or not seed
    ):
        raise ValueError("shuffled-surface roster is invalid")
    by_domain: dict[str, list[str]] = defaultdict(list)
    for specimen, domain in zip(specimen_ids, dataset_ids, strict=True):
        by_domain[domain].append(specimen)
    donors: dict[str, str] = {}
    for domain, values in by_domain.items():
        ordered = sorted(values)
        if len(ordered) < 2:
            raise ValueError("shuffled-surface domain has no legal donor")
        token = _sha256_text(f"{seed}|{domain}")
        offset = 1 + int(token[:16], 16) % (len(ordered) - 1)
        donors.update(
            {
                specimen: ordered[(index + offset) % len(ordered)]
                for index, specimen in enumerate(ordered)
            }
        )
    output = tuple(donors[specimen] for specimen in specimen_ids)
    if any(left == right for left, right in zip(specimen_ids, output, strict=True)):
        raise ValueError("shuffled-surface donor contains a self mapping")
    return output


def spatial_derangement(
    specimen_id: str, *, dataset_id: str, seed: str
) -> tuple[int, ...]:
    """Build a hash-driven Sattolo 64-cycle with no fixed cell."""

    if not specimen_id or not dataset_id or not seed:
        raise ValueError("spatial-derangement identity is incomplete")
    values = list(range(64))
    for index in range(63, 0, -1):
        token = _sha256_text(f"{seed}|{dataset_id}|{specimen_id}|{index}")
        swap = int(token[:16], 16) % index
        values[index], values[swap] = values[swap], values[index]
    output = tuple(values)
    if set(output) != set(range(64)) or any(
        index == value for index, value in enumerate(output)
    ):
        raise ValueError("spatial derangement is not a fixed-point-free permutation")
    visited: set[int] = set()
    current = 0
    while current not in visited:
        visited.add(current)
        current = output[current]
    if current != 0 or len(visited) != 64:
        raise ValueError("spatial derangement is not one Sattolo cycle")
    return output


__all__ = [
    "SurfaceCellAuthority",
    "SurfaceCellRecord",
    "crop_rgb_patch",
    "integer_crop_box",
    "load_surface_cell_authority",
    "oriented_surface_boxes",
    "shuffled_surface_donors",
    "spatial_derangement",
    "wrong_orientation",
]
