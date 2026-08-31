"""Deterministic diagnostic overlays for a frozen P0R registration package."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, UnidentifiedImageError

from .contracts import PRIMARY_COUNTS
from .p0r_artifacts import P0RArtifactError, replay_p0r_package


class P0RQCError(ValueError):
    """Raised when deterministic P0R diagnostic rendering cannot proceed."""


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_PANELS = (
    "Surface original",
    "Surface CW90",
    "Surface inverse grid",
    "Registered C-scan grid",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise P0RQCError("QC source file cannot be read") from error
    return digest.hexdigest()


def _json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if (
                reader.fieldnames is None
                or len(set(reader.fieldnames)) != len(reader.fieldnames)
            ):
                raise P0RQCError(f"{label} schema is invalid")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise P0RQCError(f"{label} cannot be read") from error
    if not rows:
        raise P0RQCError(f"{label} contains no rows")
    return rows


def _index(
    rows: Sequence[Mapping[str, str]], label: str
) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for source in rows:
        row = dict(source)
        key = (row.get("dataset_id", ""), row.get("specimen_id", ""))
        if not all(key) or key in result:
            raise P0RQCError(f"{label} specimen keys are missing or duplicated")
        result[key] = row
    return result


def _resolve_source(root: Path, raw_path: str, expected_hash: str, label: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise P0RQCError(f"{label} path is unsafe")
    unresolved = root
    for part in relative.parts:
        unresolved /= part
        if unresolved.is_symlink():
            raise P0RQCError(f"{label} contains a symlink")
    try:
        path = unresolved.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise P0RQCError(f"{label} escapes the surface root") from error
    if not path.is_file() or _sha256(path) != expected_hash:
        raise P0RQCError(f"{label} SHA-256 changed")
    return path


def _decode_rgb(path: Path, label: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise P0RQCError(f"{label} mode is not RGB")
            image.load()
            return image.copy()
    except P0RQCError:
        raise
    except (OSError, UnidentifiedImageError) as error:
        raise P0RQCError(f"{label} cannot be decoded") from error


def _panel(
    image: Image.Image,
    *,
    label: str,
    width: int,
    height: int,
    boxes: Sequence[tuple[float, float, float, float]] = (),
) -> Image.Image:
    header = 26
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    available_height = height - header
    scale = min(width / image.width, available_height / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize(
        (resized_width, resized_height), resample=Image.Resampling.LANCZOS
    )
    left = (width - resized_width) // 2
    top = header + (available_height - resized_height) // 2
    canvas.paste(resized, (left, top))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, header - 1), fill=(32, 36, 40))
    draw.text((8, 7), label, fill=(255, 255, 255))
    if boxes:
        scale_x = resized_width / image.width
        scale_y = resized_height / image.height
        for box in boxes:
            x0, y0, x1, y1 = box
            draw.rectangle(
                (
                    left + x0 * scale_x,
                    top + y0 * scale_y,
                    left + x1 * scale_x,
                    top + y1 * scale_y,
                ),
                outline=(230, 30, 45),
                width=1,
            )
    draw.rectangle((0, 0, width - 1, height - 1), outline=(96, 100, 104), width=1)
    return canvas


def _render_overlay(
    *,
    surface: Image.Image,
    crop: Image.Image,
    grid_rows: Sequence[Mapping[str, str]],
    panel_width: int,
    panel_height: int,
) -> Image.Image:
    surface_boxes = [
        tuple(float(row[field]) for field in ("surface_x0", "surface_y0", "surface_x1", "surface_y1"))
        for row in grid_rows
    ]
    cscan_boxes = [
        tuple(float(row[field]) for field in ("cscan_x0", "cscan_y0", "cscan_x1", "cscan_y1"))
        for row in grid_rows
    ]
    panels = (
        _panel(
            surface,
            label=_PANELS[0],
            width=panel_width,
            height=panel_height,
        ),
        _panel(
            surface.transpose(Image.Transpose.ROTATE_270),
            label=_PANELS[1],
            width=panel_width,
            height=panel_height,
        ),
        _panel(
            surface,
            label=_PANELS[2],
            width=panel_width,
            height=panel_height,
            boxes=surface_boxes,
        ),
        _panel(
            crop,
            label=_PANELS[3],
            width=panel_width,
            height=panel_height,
            boxes=cscan_boxes,
        ),
    )
    output = Image.new("RGB", (panel_width * 2, panel_height * 2), "white")
    output.paste(panels[0], (0, 0))
    output.paste(panels[1], (panel_width, 0))
    output.paste(panels[2], (0, panel_height))
    output.paste(panels[3], (panel_width, panel_height))
    return output


def _selection_score(seed: str, domain: str, specimen: str) -> str:
    return hashlib.sha256(f"{seed}\0{domain}\0{specimen}".encode()).hexdigest()


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    fields = tuple(sorted({field for row in rows for field in row}))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def render_p0r_qc(
    *,
    package: str | Path,
    surface_root: str | Path,
    output: str | Path,
) -> Path:
    """Render a balanced, hash-selected diagnostic subset without mutation."""

    package_root = Path(package)
    try:
        summary = replay_p0r_package(package_root)
    except P0RArtifactError as error:
        raise P0RQCError("P0R package integrity failed before QC") from error
    if summary.get("status") != "P0R_AUTHOR_REGISTRATION_GO":
        raise P0RQCError("P0R QC requires an authorized GO package")
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise P0RQCError("P0R QC output already exists")
    external = Path(surface_root)
    if external.is_symlink() or not external.is_dir():
        raise P0RQCError("P0R QC surface root is unavailable")
    try:
        external = external.resolve(strict=True)
        config = yaml.safe_load((package_root / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise P0RQCError("P0R QC config cannot be read") from error
    qc = config.get("qc") if type(config) is dict else None
    if (
        type(qc) is not dict
        or type(qc.get("selection_seed")) is not str
        or qc.get("specimens_per_domain") != 2
        or type(qc.get("panel_width_px")) is not int
        or type(qc.get("panel_height_px")) is not int
        or qc["panel_width_px"] <= 0
        or qc["panel_height_px"] <= 0
    ):
        raise P0RQCError("P0R QC config changed")

    surfaces = _index(
        _read_csv(package_root / "surface_manifest.csv", "P0R surface manifest"),
        "P0R surface manifest",
    )
    provenance = _index(
        _read_csv(
            package_root / "scan_processing_provenance.csv",
            "P0R scan processing provenance",
        ),
        "P0R scan processing provenance",
    )
    registration = _index(
        _read_csv(package_root / "registration.csv", "P0R registration"),
        "P0R registration",
    )
    grid_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(package_root / "grid_mapping_qc.csv", "P0R grid mapping"):
        key = (row.get("dataset_id", ""), row.get("specimen_id", ""))
        if not all(key):
            raise P0RQCError("P0R grid specimen key is missing")
        grid_by_key[key].append(row)
    if not (set(surfaces) == set(provenance) == set(registration) == set(grid_by_key)):
        raise P0RQCError("P0R QC source tables have different rosters")

    candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seed = qc["selection_seed"]
    for key, row in registration.items():
        domain, specimen = key
        if (
            domain not in PRIMARY_COUNTS
            or row.get("status") != "AUTHORIZED"
            or row.get("orientation") != "ROT90"
            or surfaces[key].get("p0r_roster_status") != "AUTHORIZED"
        ):
            raise P0RQCError("P0R QC encountered an unauthorized registration row")
        score = _selection_score(seed, domain, specimen)
        candidates[domain].append((score, specimen))
    if set(candidates) != set(PRIMARY_COUNTS) or any(
        len(candidates[domain]) < qc["specimens_per_domain"] for domain in PRIMARY_COUNTS
    ):
        raise P0RQCError("P0R QC balanced selection is unavailable")

    selections: list[tuple[str, int, str, str]] = []
    for domain in PRIMARY_COUNTS:
        ranked = sorted(candidates[domain])[: qc["specimens_per_domain"]]
        selections.extend(
            (domain, rank, specimen, score)
            for rank, (score, specimen) in enumerate(ranked, start=1)
        )
    if Counter(domain for domain, _, _, _ in selections) != Counter(
        {domain: 2 for domain in PRIMARY_COUNTS}
    ):
        raise P0RQCError("P0R QC selection balance changed")

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise P0RQCError("P0R QC output parent is unavailable")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        manifest_rows: list[dict[str, object]] = []
        png_records: dict[str, dict[str, object]] = {}
        for domain, rank, specimen, score in selections:
            key = (domain, specimen)
            surface_row = surfaces[key]
            provenance_row = provenance[key]
            registration_row = registration[key]
            cell_rows = sorted(grid_by_key[key], key=lambda row: int(row["cell_id"]))
            if (
                len(cell_rows) != 64
                or [int(row["cell_id"]) for row in cell_rows] != list(range(64))
                or any(row.get("round_trip_status") != "PASS" for row in cell_rows)
                or any(
                    row.get("transform_sha256")
                    != registration_row.get("transform_sha256")
                    for row in cell_rows
                )
            ):
                raise P0RQCError("P0R QC grid mapping changed")
            surface_path = _resolve_source(
                external,
                surface_row["impacted_surface_path"],
                surface_row["surface_sha256"],
                f"QC surface {domain}/{specimen}",
            )
            crop_path = _resolve_source(
                external,
                provenance_row["registered_cscan_crop_path"],
                provenance_row["registered_crop_file_sha256"],
                f"QC registered crop {domain}/{specimen}",
            )
            surface_image = _decode_rgb(surface_path, "QC surface")
            crop_image = _decode_rgb(crop_path, "QC registered crop")
            if (
                surface_image.size
                != (
                    int(registration_row["source_width_px"]),
                    int(registration_row["source_height_px"]),
                )
                or crop_image.size
                != (
                    int(registration_row["destination_width_px"]),
                    int(registration_row["destination_height_px"]),
                )
            ):
                raise P0RQCError("P0R QC image geometry changed")
            if not _SAFE_NAME.fullmatch(domain) or not _SAFE_NAME.fullmatch(specimen):
                raise P0RQCError("P0R QC specimen name is unsafe")
            filename = f"{domain}__{specimen}.png"
            overlay = _render_overlay(
                surface=surface_image,
                crop=crop_image,
                grid_rows=cell_rows,
                panel_width=qc["panel_width_px"],
                panel_height=qc["panel_height_px"],
            )
            destination = temporary / filename
            overlay.save(destination, format="PNG", optimize=False, compress_level=9)
            png_hash = _sha256(destination)
            png_records[filename] = {
                "sha256": png_hash,
                "size": destination.stat().st_size,
            }
            manifest_rows.append(
                {
                    "dataset_id": domain,
                    "specimen_id": specimen,
                    "selection_rank": rank,
                    "selection_sha256": score,
                    "filename": filename,
                    "sha256": png_hash,
                    "width_px": overlay.width,
                    "height_px": overlay.height,
                    "panel_count": 4,
                    "panels": ";".join(_PANELS),
                    "transform_sha256": registration_row["transform_sha256"],
                    "surface_sha256": surface_row["surface_sha256"],
                    "registered_cscan_sha256": provenance_row[
                        "registered_crop_file_sha256"
                    ],
                }
            )
        manifest_payload = _csv_bytes(manifest_rows)
        (temporary / "overlay_manifest.csv").write_bytes(manifest_payload)
        file_records = {
            **png_records,
            "overlay_manifest.csv": {
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "size": len(manifest_payload),
            },
        }
        artifact_manifest = _json(
            {
                "schema_version": 1,
                "stage": "P0R_QC_DIAGNOSTIC_ONLY",
                "selection_seed": seed,
                "specimens_per_domain": 2,
                "panel_layout": "2x2",
                "panel_count": 4,
                "p0r_checksums_sha256": _sha256(
                    package_root / "CHECKSUMS.sha256"
                ),
                "files": file_records,
            }
        )
        (temporary / "artifact_manifest.json").write_bytes(artifact_manifest)
        checksum_names = sorted({*file_records, "artifact_manifest.json"})
        checksums = "".join(
            f"{_sha256(temporary / name)}  {name}\n" for name in checksum_names
        ).encode("ascii")
        (temporary / "CHECKSUMS.sha256").write_bytes(checksums)
        if target.exists() or target.is_symlink():
            raise P0RQCError("P0R QC output already exists")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


__all__ = ["P0RQCError", "render_p0r_qc"]
