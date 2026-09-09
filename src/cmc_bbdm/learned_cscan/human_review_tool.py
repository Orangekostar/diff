"""Local packet and return adapters for C-scan human review."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl
import yaml
from PIL import Image, ImageDraw

from cmc_bbdm.inspection_agent.state import (
    GeneralizedMeasurementState,
    InspectionCellAction,
    action_added_positions_from_mask,
    apply_action,
    zero_state,
)
from cmc_bbdm.learned_cscan.benchmark import _report_digest
from cmc_bbdm.learned_cscan.contracts import Split, Task
from cmc_bbdm.learned_cscan.frozen_evidence_finalize import (
    BLIND_REVIEW_DECISIONS,
    summarize_blind_reviews,
)
from cmc_bbdm.learned_cscan.frozen_process_analysis import (
    load_frozen_process_config,
)
from cmc_bbdm.learned_cscan.frozen_process_recovery import (
    FrozenVisibleReportReader,
)
from cmc_bbdm.learned_cscan.readout import TaskReportV2
from cmc_bbdm.learned_cscan.runtime import (
    load_study_config,
    load_study_context,
    load_study_roster,
    open_study_specimen,
)
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.vlm_cscan.references import reference_from_payload
from cmc_bbdm.vlm_cscan.runtime import InputSpecimen

TOOL_STAGE = "CSCAN_HUMAN_REVIEW_HTML"
TOOL_BASE_SHA = "2102cc4a1726910931dfaaf20e29ad29a20eaf2e"
BLINDING_CONDITION = "FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1"
REFERENCE_PACKET_KIND = "REFERENCE_ANNOTATION"
REFERENCE_SESSION_KIND = "CSCAN_REFERENCE_SESSION"
BLIND_PACKET_KIND = "BLIND_FIRST_STOP_REVIEW"
BLIND_SESSION_KIND = "CSCAN_BLIND_REVIEW_SESSION"
BLIND_APPLICATION_STATES = {"UNREVIEWED", "DRAFT", "CONFIRMED"}
REFERENCE_STATES = {
    "UNANNOTATED",
    "DRAFT",
    "CONFIRMED",
    "CONFIRMED_NO_CERTAIN_REGION",
    "UNABLE_TO_JUDGE",
}
HUMAN_REFERENCE_TYPES = {"EXPERT_REVIEWED", "AUTHOR_PROVIDED"}
ANNOTATION_ORIGINS = {"FROM_SCRATCH", "WITH_PREANNOTATION"}
REFERENCE_PACKET_FIELDS = {
    "schema_version",
    "packet_kind",
    "packet_id",
    "source_base_sha",
    "packet_number",
    "item_count",
    "test_only",
    "practice_notice",
    "items",
}
REFERENCE_ITEM_FIELDS = {
    "item_id",
    "specimen_key",
    "native_width",
    "native_height",
    "source_image_sha256",
    "display_image_sha256",
    "frame",
    "image_mime",
    "image_data",
}
BLIND_PACKET_FIELDS = {
    "schema_version",
    "packet_kind",
    "packet_id",
    "source_base_sha",
    "packet_number",
    "item_count",
    "blinding_condition",
    "items",
}
BLIND_ITEM_FIELDS = {
    "item_id",
    "display_number",
    "report_id",
    "task",
    "blinding_condition",
    "native_height",
    "native_width",
    "measured_image_mime",
    "measured_image_data",
    "measured_image_sha256",
    "report_overlay_mime",
    "report_overlay_data",
    "report_overlay_sha256",
    "support_overlay_mime",
    "support_overlay_data",
    "support_overlay_sha256",
}


@dataclass(frozen=True, slots=True)
class HumanReviewConfig:
    path: Path
    project_root: Path
    config_sha256: str
    repository_base_sha: str
    frozen_process_config: Path
    first_stop_path: Path
    trajectories_path: Path
    cohort_path: Path
    web_root: Path
    dist_root: Path
    result_root: Path
    artifact_root: Path
    local_root: Path
    reference_packet_size: int
    blind_packet_size: int
    shuffle_seed: int
    blinding_condition: str
    resource_limits: MappingProxyType[str, int]
    values: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class ReplayedEndpoint:
    state: GeneralizedMeasurementState
    measured_mask: np.ndarray
    measured_count: int
    measured_cost: float
    action_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(root: Path, value: object, *, must_exist: bool) -> Path:
    if type(value) is not str or not value:
        raise ValueError("human-review path binding is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("human-review path binding is invalid")
    path = (root / relative).resolve()
    if must_exist and not path.is_file():
        raise ValueError(f"human-review frozen input is missing: {relative}")
    return path


def load_human_review_config(
    path: str | Path, *, project_root: str | Path
) -> HumanReviewConfig:
    root = Path(project_root).resolve(strict=True)
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    payload = yaml.safe_load(raw)
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("stage") != TOOL_STAGE
        or payload.get("repository_base_sha") != TOOL_BASE_SHA
    ):
        raise ValueError("human-review config identity is invalid")
    paths = payload.get("paths")
    packets = payload.get("packets")
    frozen = payload.get("frozen_inputs")
    resources = payload.get("resource_limits")
    if not all(type(value) is dict for value in (paths, packets, frozen, resources)):
        raise ValueError("human-review config sections are invalid")

    input_names = ("frozen_process_config", "first_stop", "trajectories", "cohort")
    input_paths = {
        name: _project_path(root, paths.get(name), must_exist=True)
        for name in input_names
    }
    for name, input_path in input_paths.items():
        expected = frozen.get(name)
        if type(expected) is not str or _sha256(input_path) != expected:
            raise ValueError(f"human-review frozen input changed: {name}")

    reference_packet_size = packets.get("reference_packet_size")
    blind_packet_size = packets.get("blind_packet_size")
    shuffle_seed = packets.get("shuffle_seed")
    if (
        packets.get("schema_version") != 1
        or type(reference_packet_size) is not int
        or reference_packet_size < 1
        or type(blind_packet_size) is not int
        or blind_packet_size < 1
        or type(shuffle_seed) is not int
        or packets.get("blinding_condition") != BLINDING_CONDITION
    ):
        raise ValueError("human-review packet config is invalid")
    required_resources = {
        "training_updates": 0,
        "vlm_calls": 0,
        "actor_stop_forward_calls": 0,
        "cpu_processes": 1,
    }
    if resources != required_resources:
        raise ValueError("human-review resource contract is invalid")

    return HumanReviewConfig(
        path=source,
        project_root=root,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        repository_base_sha=TOOL_BASE_SHA,
        frozen_process_config=input_paths["frozen_process_config"],
        first_stop_path=input_paths["first_stop"],
        trajectories_path=input_paths["trajectories"],
        cohort_path=input_paths["cohort"],
        web_root=_project_path(root, paths.get("web_root"), must_exist=False),
        dist_root=_project_path(root, paths.get("dist_root"), must_exist=False),
        result_root=_project_path(root, paths.get("result_root"), must_exist=False),
        artifact_root=_project_path(root, paths.get("artifact_root"), must_exist=False),
        local_root=_project_path(root, paths.get("local_root"), must_exist=False),
        reference_packet_size=reference_packet_size,
        blind_packet_size=blind_packet_size,
        shuffle_seed=shuffle_seed,
        blinding_condition=BLINDING_CONDITION,
        resource_limits=MappingProxyType(required_resources),
        values=MappingProxyType(payload),
    )


def build_delivery_ui(
    config: HumanReviewConfig, *, output_path: str | Path | None = None
) -> dict[str, object]:
    if type(config) is not HumanReviewConfig:
        raise TypeError("issued human-review config is required")
    template_path = config.web_root / "index.template.html"
    styles_path = config.web_root / "styles.css"
    app_path = config.web_root / "app.js"
    for path in (template_path, styles_path, app_path):
        if not path.is_file():
            raise ValueError(f"UI source is missing: {path.name}")
    template = template_path.read_text(encoding="utf-8")
    if template.count("/*__CSCAN_STYLES__*/") != 1 or template.count("/*__CSCAN_APP__*/") != 1:
        raise ValueError("UI template markers are invalid")
    html = template.replace(
        "/*__CSCAN_STYLES__*/", styles_path.read_text(encoding="utf-8")
    ).replace("/*__CSCAN_APP__*/", app_path.read_text(encoding="utf-8"))
    lowered = html.lower()
    if any(
        token in lowered
        for token in ('src="http://', 'src="https://', 'href="http://', 'href="https://')
    ):
        raise ValueError("delivery UI cannot use network resources")
    output = Path(output_path or config.dist_root / "index.html").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8", newline="\n")
    return {
        "schema_version": 1,
        "stage": TOOL_STAGE,
        "tool_status": "READY",
        "output_path": str(output),
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
        "network_dependencies": 0,
    }


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _packet_id(kind: str, source_base_sha: str, items: Sequence[Mapping[str, object]]) -> str:
    identity = {
        "kind": kind,
        "source_base_sha": source_base_sha,
        "items": [
            {
                key: item[key]
                for key in (
                    "item_id",
                    "specimen_key",
                    "source_image_sha256",
                    "display_image_sha256",
                )
                if key in item
            }
            for item in items
        ],
    }
    return hashlib.sha256(_json_bytes(identity)).hexdigest()


def _blind_packet_id(
    source_base_sha: str, items: Sequence[Mapping[str, object]]
) -> str:
    return hashlib.sha256(
        _json_bytes(
            {
                "kind": BLIND_PACKET_KIND,
                "source_base_sha": source_base_sha,
                "report_ids": [item["report_id"] for item in items],
                "layer_hashes": [
                    (
                        item["measured_image_sha256"],
                        item["report_overlay_sha256"],
                        item["support_overlay_sha256"],
                    )
                    for item in items
                ],
            }
        )
    ).hexdigest()


def _validate_embedded_png(
    item: Mapping[str, object],
    *,
    data_field: str,
    sha_field: str,
    expected_mode: str,
) -> None:
    value = item.get(data_field)
    expected_sha = item.get(sha_field)
    if (
        type(value) is not str
        or not value.startswith("data:image/png;base64,")
        or type(expected_sha) is not str
    ):
        raise ValueError("packet image identity is invalid")
    try:
        payload = base64.b64decode(value.split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            size = image.size
            mode = image.mode
            format_name = image.format
    except (OSError, ValueError) as error:
        raise ValueError("packet image payload is invalid") from error
    if hashlib.sha256(payload).hexdigest() != expected_sha or format_name != "PNG":
        raise ValueError("packet image hash is invalid")
    if size != (item.get("native_width"), item.get("native_height")):
        raise ValueError("packet image shape is invalid")
    if mode != expected_mode:
        raise ValueError("packet image mode is invalid")


def _validate_packet_integrity(packet: Mapping[str, object]) -> None:
    items = packet.get("items")
    source_base_sha = packet.get("source_base_sha")
    if (
        type(items) is not list
        or not items
        or packet.get("item_count") != len(items)
        or type(source_base_sha) is not str
        or len(source_base_sha) != 40
        or set(source_base_sha) - set("0123456789abcdef")
        or any(type(item) is not dict for item in items)
    ):
        raise ValueError("packet integrity is invalid")
    item_ids = tuple(str(item.get("item_id") or "") for item in items)
    if not all(item_ids) or len(set(item_ids)) != len(item_ids):
        raise ValueError("packet item identity is invalid")

    if packet.get("packet_kind") == REFERENCE_PACKET_KIND:
        if set(packet) - REFERENCE_PACKET_FIELDS:
            raise ValueError("reference packet public fields are invalid")
        for item in items:
            if (
                set(item) != REFERENCE_ITEM_FIELDS
                or item.get("frame") != "registered_cscan"
                or item.get("image_mime") != "image/png"
                or type(item.get("specimen_key")) is not str
                or not item["specimen_key"]
                or type(item.get("source_image_sha256")) is not str
                or len(item["source_image_sha256"]) != 64
                or type(item.get("native_width")) is not int
                or item["native_width"] < 2
                or type(item.get("native_height")) is not int
                or item["native_height"] < 2
            ):
                raise ValueError("reference packet item identity is invalid")
            _validate_embedded_png(
                item,
                data_field="image_data",
                sha_field="display_image_sha256",
                expected_mode="RGB",
            )
        if packet.get("packet_id") != _packet_id(
            REFERENCE_PACKET_KIND, source_base_sha, items
        ):
            raise ValueError("reference packet identity changed")
        return

    if packet.get("packet_kind") == BLIND_PACKET_KIND:
        if set(packet) != BLIND_PACKET_FIELDS:
            raise ValueError("blind packet public fields are invalid")
        report_ids = tuple(str(item.get("report_id") or "") for item in items)
        if not all(report_ids) or len(set(report_ids)) != len(report_ids):
            raise ValueError("blind packet report identity is invalid")
        for item in items:
            if (
                set(item) != BLIND_ITEM_FIELDS
                or item.get("task") not in {"LOCATE", "CHARACTERIZE"}
                or item.get("blinding_condition") != BLINDING_CONDITION
                or type(item.get("display_number")) is not int
                or type(item.get("native_width")) is not int
                or item["native_width"] < 2
                or type(item.get("native_height")) is not int
                or item["native_height"] < 2
            ):
                raise ValueError("blind packet public fields are invalid")
            for prefix, mode in (
                ("measured_image", "RGB"),
                ("report_overlay", "RGBA"),
                ("support_overlay", "RGBA"),
            ):
                if item.get(f"{prefix}_mime") != "image/png":
                    raise ValueError("blind packet image type is invalid")
                _validate_embedded_png(
                    item,
                    data_field=f"{prefix}_data",
                    sha_field=f"{prefix}_sha256",
                    expected_mode=mode,
                )
        if (
            packet.get("blinding_condition") != BLINDING_CONDITION
            or packet.get("packet_id") != _blind_packet_id(source_base_sha, items)
        ):
            raise ValueError("blind packet identity changed")
        return
    raise ValueError("unsupported packet kind")


def _display_png(record: InputSpecimen) -> tuple[bytes, int, int]:
    if _sha256(record.cscan_path) != record.cscan_sha256:
        raise ValueError(f"registered source image hash changed: {record.specimen_key}")
    with Image.open(record.cscan_path) as source:
        source.load()
        if source.size != (record.native_shape[1], record.native_shape[0]):
            raise ValueError(f"registered source image shape changed: {record.specimen_key}")
        image = source.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=9)
    return buffer.getvalue(), image.height, image.width


def build_reference_packets(
    records: tuple[InputSpecimen, ...],
    *,
    packet_size: int,
    source_base_sha: str,
) -> dict[str, object]:
    if (
        type(records) is not tuple
        or not records
        or any(type(record) is not InputSpecimen for record in records)
        or type(packet_size) is not int
        or packet_size < 1
        or type(source_base_sha) is not str
        or len(source_base_sha) != 40
        or set(source_base_sha) - set("0123456789abcdef")
    ):
        raise ValueError("reference packet request is invalid")
    if len({record.specimen_key for record in records}) != len(records):
        raise ValueError("reference packet specimens are duplicated")

    items = []
    manifest = []
    for index, record in enumerate(records, start=1):
        png, height, width = _display_png(record)
        display_sha = hashlib.sha256(png).hexdigest()
        item_id = f"reference-{index:03d}-{hashlib.sha256(record.specimen_key.encode()).hexdigest()[:8]}"
        items.append(
            {
                "item_id": item_id,
                "specimen_key": record.specimen_key,
                "native_width": width,
                "native_height": height,
                "source_image_sha256": record.cscan_sha256,
                "display_image_sha256": display_sha,
                "frame": "registered_cscan",
                "image_mime": "image/png",
                "image_data": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
            }
        )
        manifest.append(
            {
                "schema_version": 1,
                "item_id": item_id,
                "dataset_id": record.dataset_id,
                "specimen_id": record.specimen_id,
                "specimen_key": record.specimen_key,
                "native_height": height,
                "native_width": width,
                "source_image_path": str(record.cscan_path),
                "source_image_sha256": record.cscan_sha256,
                "display_image_sha256": display_sha,
                "status": "PREPARED",
            }
        )

    packets = []
    for packet_number, offset in enumerate(range(0, len(items), packet_size), start=1):
        chunk = items[offset : offset + packet_size]
        packet_id = _packet_id(REFERENCE_PACKET_KIND, source_base_sha, chunk)
        packet = {
            "schema_version": 1,
            "packet_kind": REFERENCE_PACKET_KIND,
            "packet_id": packet_id,
            "source_base_sha": source_base_sha,
            "packet_number": packet_number,
            "item_count": len(chunk),
            "items": chunk,
        }
        packets.append(packet)
        for row in manifest[offset : offset + packet_size]:
            row["packet_id"] = packet_id
            row["packet_number"] = packet_number
    return {"packets": tuple(packets), "manifest": tuple(manifest)}


def replay_stored_prefix(
    *,
    grid: AcquisitionGrid,
    trajectory_rows: tuple[Mapping[str, object], ...],
    stop_step: int,
) -> ReplayedEndpoint:
    if (
        type(grid) is not AcquisitionGrid
        or type(trajectory_rows) is not tuple
        or any(not isinstance(row, Mapping) for row in trajectory_rows)
        or type(stop_step) is not int
        or stop_step < 0
    ):
        raise ValueError("stored prefix request is invalid")
    steps = tuple(row.get("step") for row in trajectory_rows)
    if len(steps) <= stop_step or steps[: stop_step + 1] != tuple(range(stop_step + 1)):
        raise ValueError("stored trajectory step order changed")
    state = zero_state(grid)
    measured = np.zeros(grid.native_shape, dtype=np.bool_)
    for row in trajectory_rows[:stop_step]:
        fields = tuple(row.get(name) for name in (
            "action_cell",
            "action_from_level",
            "action_to_level",
        ))
        if any(type(value) is not int for value in fields):
            raise ValueError("stored trajectory action is invalid")
        action = InspectionCellAction(*fields)
        positions = action_added_positions_from_mask(
            grid, state, action, measured
        )
        measured[positions[:, 0], positions[:, 1]] = True
        state = apply_action(grid, state, action)
    measured_count = int(np.count_nonzero(measured))
    measured.setflags(write=False)
    return ReplayedEndpoint(
        state=state,
        measured_mask=measured,
        measured_count=measured_count,
        measured_cost=float(measured_count / measured.size),
        action_count=stop_step,
    )


def _png_data(image: Image.Image) -> tuple[str, str]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=9)
    payload = buffer.getvalue()
    return (
        "data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
        hashlib.sha256(payload).hexdigest(),
    )


def render_blind_layers(
    *,
    full_scan: np.ndarray,
    measured_mask: np.ndarray,
    report: TaskReportV2,
) -> dict[str, object]:
    scan = np.asarray(full_scan)
    measured = np.asarray(measured_mask)
    if (
        scan.dtype != np.uint8
        or scan.ndim != 3
        or scan.shape[2] != 3
        or measured.dtype != np.bool_
        or measured.shape != scan.shape[:2]
        or type(report) is not TaskReportV2
        or report.predicted_mask.shape != measured.shape
    ):
        raise ValueError("blind rendering request is invalid")
    support = np.asarray(report.support_positions, dtype=np.int64)
    if len(support) and not np.all(measured[support[:, 0], support[:, 1]]):
        raise ValueError("report support lies outside measured evidence")

    rows, columns = np.indices(measured.shape)
    checker = ((rows // 8 + columns // 8) % 2).astype(np.uint8)
    sanitized = np.empty_like(scan)
    sanitized[:] = np.where(checker[..., None] == 0, 38, 52)
    sanitized[measured] = scan[measured]

    report_overlay = np.zeros((*measured.shape, 4), dtype=np.uint8)
    report_overlay[report.predicted_mask] = (255, 184, 0, 118)
    support_overlay = Image.new("RGBA", (scan.shape[1], scan.shape[0]), (0, 0, 0, 0))
    support_draw = ImageDraw.Draw(support_overlay)
    radius = max(2, round(min(measured.shape) / 170))
    for row, column in support:
        support_draw.ellipse(
            (
                int(column) - radius,
                int(row) - radius,
                int(column) + radius,
                int(row) + radius,
            ),
            fill=(0, 214, 190, 255),
            outline=(4, 40, 38, 255),
            width=1,
        )
    measured_data, measured_sha = _png_data(Image.fromarray(sanitized, mode="RGB"))
    report_data, report_sha = _png_data(Image.fromarray(report_overlay, mode="RGBA"))
    support_data, support_sha = _png_data(support_overlay)
    return {
        "native_height": int(scan.shape[0]),
        "native_width": int(scan.shape[1]),
        "measured_image_mime": "image/png",
        "measured_image_data": measured_data,
        "measured_image_sha256": measured_sha,
        "report_overlay_mime": "image/png",
        "report_overlay_data": report_data,
        "report_overlay_sha256": report_sha,
        "support_overlay_mime": "image/png",
        "support_overlay_data": support_data,
        "support_overlay_sha256": support_sha,
    }


def build_blind_packets(
    reports_and_layers: tuple[tuple[Mapping[str, object], Mapping[str, object]], ...],
    *,
    packet_size: int,
    shuffle_seed: int,
    source_base_sha: str,
    blinding_condition: str,
) -> dict[str, object]:
    if (
        type(reports_and_layers) is not tuple
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or not isinstance(pair[0], Mapping)
            or not isinstance(pair[1], Mapping)
            for pair in reports_and_layers
        )
        or type(packet_size) is not int
        or packet_size < 1
        or type(shuffle_seed) is not int
        or len(source_base_sha) != 40
        or blinding_condition != BLINDING_CONDITION
    ):
        raise ValueError("blind packet request is invalid")
    report_ids = tuple(str(row.get("report_id") or "") for row, _ in reports_and_layers)
    if not all(report_ids) or len(set(report_ids)) != len(report_ids):
        raise ValueError("blind report identity is missing or duplicated")
    order = list(range(len(reports_and_layers)))
    random.Random(shuffle_seed).shuffle(order)
    public_items = []
    private_index = {}
    manifest = []
    for display_number, source_index in enumerate(order, start=1):
        row, layers = reports_and_layers[source_index]
        report_id = str(row["report_id"])
        task = str(row.get("task") or "")
        if task not in {"LOCATE", "CHARACTERIZE"}:
            raise ValueError("blind report task is invalid")
        required_layer_fields = (
            "native_height",
            "native_width",
            "measured_image_mime",
            "measured_image_data",
            "measured_image_sha256",
            "report_overlay_mime",
            "report_overlay_data",
            "report_overlay_sha256",
            "support_overlay_mime",
            "support_overlay_data",
            "support_overlay_sha256",
        )
        if any(field not in layers for field in required_layer_fields):
            raise ValueError("blind report layer is incomplete")
        item_id = f"blind-{display_number:03d}-{hashlib.sha256(report_id.encode()).hexdigest()[:8]}"
        item = {
            "item_id": item_id,
            "display_number": display_number,
            "report_id": report_id,
            "task": task,
            "blinding_condition": blinding_condition,
            **{field: layers[field] for field in required_layer_fields},
        }
        public_items.append(item)
        private_index[report_id] = dict(row)
        manifest.append(
            {
                "schema_version": 1,
                "item_id": item_id,
                "display_number": display_number,
                "report_id": report_id,
                "task": task,
                "public_material_status": "PREPARED",
            }
        )

    packets = []
    for packet_number, offset in enumerate(
        range(0, len(public_items), packet_size), start=1
    ):
        chunk = public_items[offset : offset + packet_size]
        packet_id = _blind_packet_id(source_base_sha, chunk)
        packets.append(
            {
                "schema_version": 1,
                "packet_kind": BLIND_PACKET_KIND,
                "packet_id": packet_id,
                "source_base_sha": source_base_sha,
                "packet_number": packet_number,
                "item_count": len(chunk),
                "blinding_condition": blinding_condition,
                "items": chunk,
            }
        )
        for row in manifest[offset : offset + packet_size]:
            row["packet_id"] = packet_id
            row["packet_number"] = packet_number
    return {
        "packets": tuple(packets),
        "private_index": private_index,
        "manifest": tuple(manifest),
    }


def _private_report_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "report_id": str(row["report_id"]),
        "specimen_key": str(row["specimen_key"]),
        "task": str(row["task"]),
        "method": str(row["method"]),
        "seed": int(row["seed"]),
        "stop_system": str(row["stop_system"]),
        "stop_step": int(row["stop_step"]),
        "report_sha256": str(row["stop_report_sha256"]),
        "objective_success": bool(row["success_at_stop"]),
        "objective_scope": "PROXY_LEGACY",
        "reference_version": str(row.get("reference_version") or "PROXY_LEGACY"),
    }


def select_blind_stop_rows(config: HumanReviewConfig) -> dict[str, object]:
    if type(config) is not HumanReviewConfig:
        raise TypeError("issued human-review config is required")
    table = pl.read_csv(config.first_stop_path, infer_schema_length=None)
    selected = table.filter(
        pl.col("method").is_in(["R_BALANCED_P8", "BC_S1", "BC_S2", "BC_S3"])
        & pl.col("task").is_in(["LOCATE", "CHARACTERIZE"])
        & (pl.col("stop_system") == "S_BC_CAL")
    ).sort(["specimen_key", "task", "method", "seed"])
    if selected.height != 192:
        raise ValueError("blind review frozen run matrix changed")
    stopped = selected.filter(pl.col("stopped"))
    no_stop = selected.filter(~pl.col("stopped"))
    stopped_rows = tuple(stopped.to_dicts())
    no_stop_rows = tuple(no_stop.to_dicts())
    if (
        any(not row.get("report_id") or not row.get("stop_report_sha256") for row in stopped_rows)
        or any(row.get("report_id") for row in no_stop_rows)
        or len({row["report_id"] for row in stopped_rows}) != len(stopped_rows)
    ):
        raise ValueError("blind review report identity changed")
    return {"stopped_rows": stopped_rows, "no_stop_rows": no_stop_rows}


def recover_blind_reports(
    config: HumanReviewConfig,
    *,
    source_root: str | Path,
    stop_rows: tuple[Mapping[str, object], ...],
    trajectories: pl.DataFrame,
    cache_root: str | Path,
) -> dict[str, object]:
    if (
        type(config) is not HumanReviewConfig
        or type(stop_rows) is not tuple
        or not stop_rows
        or any(not isinstance(row, Mapping) for row in stop_rows)
        or type(trajectories) is not pl.DataFrame
    ):
        raise ValueError("blind report recovery request is invalid")
    report_ids = tuple(str(row.get("report_id") or "") for row in stop_rows)
    if not all(report_ids) or len(set(report_ids)) != len(report_ids):
        raise ValueError("blind report recovery identity is invalid")
    for row in stop_rows:
        if (
            row.get("stopped") is not True
            or row.get("method") not in {"R_BALANCED_P8", "BC_S1", "BC_S2", "BC_S3"}
            or row.get("task") not in {"LOCATE", "CHARACTERIZE"}
            or row.get("stop_system") != "S_BC_CAL"
            or type(row.get("stop_step")) is not int
            or not row.get("stop_report_sha256")
        ):
            raise ValueError("blind report recovery row is invalid")

    frames = {}
    for frame in trajectories.partition_by(
        ["specimen_key", "task", "method", "seed"], maintain_order=True
    ):
        first = frame.row(0, named=True)
        key = (
            str(first["specimen_key"]),
            str(first["task"]),
            str(first["method"]),
            int(first["seed"]),
        )
        frames[key] = tuple(frame.sort("step").to_dicts())

    frozen = load_frozen_process_config(
        config.frozen_process_config, project_root=config.project_root
    )
    parent = load_study_config(
        frozen.parent_config_path, project_root=config.project_root
    )
    context = load_study_context(parent, source_root=source_root)
    records = {
        assignment.record.specimen_key: assignment.record
        for assignment in context.roster.assignments
        if assignment.split is Split.TEST
    }
    cache_path = Path(cache_root).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    runtime_cache = {}
    reader_cache = {}
    full_scan_cache = {}
    reports_and_layers = []
    reader_count = 0
    cache_hit_count = 0
    action_count = 0
    for stop_row in stop_rows:
        report_id = str(stop_row["report_id"])
        identity = {
            "schema_version": 1,
            "repository_base_sha": config.repository_base_sha,
            "tool_config_sha256": config.config_sha256,
            "trajectories_sha256": config.values["frozen_inputs"]["trajectories"],
            "first_stop_sha256": config.values["frozen_inputs"]["first_stop"],
            "report_id": report_id,
            "report_sha256": str(stop_row["stop_report_sha256"]),
            "stop_step": int(stop_row["stop_step"]),
        }
        cached_path = cache_path / f"{report_id}.json"
        private = _private_report_row(stop_row)
        if cached_path.is_file():
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            if cached.get("identity") == identity and type(cached.get("layers")) is dict:
                reports_and_layers.append((private, cached["layers"]))
                cache_hit_count += 1
                action_count += int(cached["action_count"])
                continue
        specimen_key = str(stop_row["specimen_key"])
        record = records.get(specimen_key)
        if record is None:
            raise ValueError("blind report specimen is outside frozen TEST")
        if specimen_key not in runtime_cache:
            runtime = open_study_specimen(context, record)
            full_scan = np.asarray(
                context.authority.source_teacher_view(record.specimen_id).full_scan,
                dtype=np.uint8,
            )
            runtime_cache[specimen_key] = runtime
            full_scan_cache[specimen_key] = full_scan
            reader_cache[specimen_key] = FrozenVisibleReportReader(
                grid=runtime.grid,
                full_scan=full_scan,
                prior=context.background_prior,
                distance_threshold=float(
                    context.config.values["reader"]["distance_threshold"]
                ),
            )
        runtime = runtime_cache[specimen_key]
        trajectory_key = (
            specimen_key,
            str(stop_row["task"]),
            str(stop_row["method"]),
            int(stop_row["seed"]),
        )
        trajectory_rows = frames.get(trajectory_key)
        if trajectory_rows is None:
            raise ValueError("blind report trajectory is missing")
        endpoint = replay_stored_prefix(
            grid=runtime.grid,
            trajectory_rows=trajectory_rows,
            stop_step=int(stop_row["stop_step"]),
        )
        stop_step = int(stop_row["stop_step"])
        stored_state = trajectory_rows[stop_step]
        if (
            not math.isclose(
                endpoint.measured_cost, float(stop_row["stop_cost"]), abs_tol=1e-12
            )
            or not math.isclose(
                endpoint.measured_cost, float(stored_state["cost"]), abs_tol=1e-12
            )
        ):
            raise ValueError("blind report STOP cost changed")
        report = reader_cache[specimen_key].report(
            state=endpoint.state,
            measured_mask=endpoint.measured_mask,
            task=Task(str(stop_row["task"])),
        )
        digest = _report_digest(report)
        if digest != stop_row["stop_report_sha256"] or digest != stored_state["report_sha256"]:
            raise ValueError("blind report STOP digest changed")
        layers = render_blind_layers(
            full_scan=full_scan_cache[specimen_key],
            measured_mask=endpoint.measured_mask,
            report=report,
        )
        reports_and_layers.append((private, layers))
        reader_count += 1
        action_count += endpoint.action_count
        _write_json(
            cached_path,
            {
                "schema_version": 1,
                "identity": identity,
                "action_count": endpoint.action_count,
                "measured_count": endpoint.measured_count,
                "measured_cost": endpoint.measured_cost,
                "layers": layers,
            },
        )
    return {
        "reports_and_layers": tuple(reports_and_layers),
        "recovery_summary": {
            "schema_version": 1,
            "stage": "BLIND_FIRST_STOP_RECOVERY_COMPLETE",
            "requested_report_count": len(stop_rows),
            "recovered_endpoint_count": reader_count,
            "cached_endpoint_count": cache_hit_count,
            "reader_report_count": reader_count,
            "stored_action_transition_count": action_count,
            "opened_specimen_count": len(runtime_cache),
            "world_step_count": 0,
            "report_digest_mismatch_count": 0,
            "cost_mismatch_count": 0,
            "training_updates": 0,
            "vlm_calls": 0,
            "actor_stop_forward_calls": 0,
        },
    }


def write_blind_packet_bundle(
    *,
    built: Mapping[str, object],
    no_stop_rows: tuple[Mapping[str, object], ...],
    recovery_summary: Mapping[str, object],
    packet_root: str | Path,
    private_index_root: str | Path,
    result_root: str | Path,
) -> dict[str, object]:
    packets = built.get("packets")
    manifest = built.get("manifest")
    private_index = built.get("private_index")
    if (
        type(packets) is not tuple
        or not packets
        or type(manifest) is not tuple
        or not isinstance(private_index, Mapping)
        or type(no_stop_rows) is not tuple
        or not isinstance(recovery_summary, Mapping)
    ):
        raise ValueError("blind packet bundle is invalid")
    public_root = Path(packet_root).resolve()
    private_root = Path(private_index_root).resolve()
    results = Path(result_root).resolve()
    public_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    packet_files = []
    for index, packet in enumerate(packets, start=1):
        path = public_root / f"blind_packet_{index:03d}.json"
        _write_json(path, packet)
        packet_files.append(str(path))

    manifest_rows = []
    for row in manifest:
        packet_number = int(row["packet_number"])
        manifest_rows.append({**row, "packet_path": packet_files[packet_number - 1]})
    manifest_path = results / "blind_packet_manifest.csv"
    _write_csv(manifest_path, manifest_rows)

    report_index_path = private_root / "report_index.json"
    _write_json(report_index_path, private_index)
    order_rows = []
    for public in manifest:
        private = private_index[str(public["report_id"])]
        order_rows.append(
            {
                **public,
                "specimen_key": private["specimen_key"],
                "method": private["method"],
                "seed": private["seed"],
                "stop_step": private["stop_step"],
                "report_sha256": private["report_sha256"],
                "objective_scope": private["objective_scope"],
            }
        )
    _write_csv(private_root / "blind_order_private.csv", order_rows)

    no_stop_coverage = tuple(
        {
            "schema_version": 1,
            "dataset_id": row["dataset_id"],
            "specimen_id": row["specimen_id"],
            "specimen_key": row["specimen_key"],
            "task": row["task"],
            "method": row["method"],
            "seed": row["seed"],
            "stop_system": row["stop_system"],
            "stopped": False,
            "exhausted": bool(row.get("exhausted")),
            "report_status": "NO_FIRST_STOP_REPORT",
        }
        for row in no_stop_rows
    )
    if no_stop_coverage:
        _write_csv(results / "no_stop_coverage.csv", no_stop_coverage)
    summary = {
        "schema_version": 1,
        "stage": TOOL_STAGE,
        "blind_packet_status": "PREPARED",
        "candidate_run_count": len(manifest_rows) + len(no_stop_rows),
        "eligible_report_count": len(manifest_rows),
        "no_stop_count": len(no_stop_rows),
        "packet_count": len(packet_files),
        "packet_root": str(public_root),
        "packet_paths": packet_files,
        "packet_bytes": sum(Path(path).stat().st_size for path in packet_files),
        "manifest_path": str(manifest_path),
        "private_report_index_path": str(report_index_path),
        "real_blind_review_count": 0,
        "scientific_evaluation_status": "NOT_RUN_NO_HUMAN_INPUT",
        "recovery": dict(recovery_summary),
    }
    _write_json(results / "blind_recovery_summary.json", summary)
    return summary


def prepare_blind_packets(
    config: HumanReviewConfig,
    *,
    source_root: str | Path,
    packet_root: str | Path | None = None,
    private_index_root: str | Path | None = None,
    result_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> dict[str, object]:
    if type(config) is not HumanReviewConfig:
        raise TypeError("issued human-review config is required")
    selected = select_blind_stop_rows(config)
    recovery = recover_blind_reports(
        config,
        source_root=source_root,
        stop_rows=selected["stopped_rows"],
        trajectories=pl.read_parquet(config.trajectories_path),
        cache_root=cache_root or config.local_root / "stop_report_cache",
    )
    if recovery["recovery_summary"]["stored_action_transition_count"] > 36864:
        raise ValueError("blind report stored-action cap exceeded")
    built = build_blind_packets(
        recovery["reports_and_layers"],
        packet_size=config.blind_packet_size,
        shuffle_seed=config.shuffle_seed,
        source_base_sha=config.repository_base_sha,
        blinding_condition=config.blinding_condition,
    )
    return write_blind_packet_bundle(
        built=built,
        no_stop_rows=selected["no_stop_rows"],
        recovery_summary=recovery["recovery_summary"],
        packet_root=packet_root or config.local_root / "blind_packets",
        private_index_root=(
            private_index_root or config.local_root / "private_report_index"
        ),
        result_root=result_root or config.result_root,
    )


def _validated_polygon(polygon: object) -> list[list[float]]:
    if type(polygon) is not list or len(polygon) < 3:
        raise ValueError("reference polygon needs at least three points")
    output = []
    for point in polygon:
        if (
            type(point) is not list
            or len(point) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in point
            )
        ):
            raise ValueError("reference polygon coordinate is invalid")
        output.append([float(point[0]), float(point[1])])
    if len({tuple(point) for point in output}) < 3:
        raise ValueError("reference polygon needs three distinct points")
    doubled_area = abs(
        sum(
            output[index][0] * output[(index + 1) % len(output)][1]
            - output[(index + 1) % len(output)][0] * output[index][1]
            for index in range(len(output))
        )
    )
    if doubled_area <= 1e-12:
        raise ValueError("reference polygon area is zero")
    return output


def _validated_regions(regions: object, *, certainty: str) -> list[dict[str, object]]:
    if type(regions) is not list:
        raise ValueError("reference regions are invalid")
    output = []
    for index, region in enumerate(regions, start=1):
        if type(region) is not dict or region.get("certainty") != certainty:
            raise ValueError("reference region certainty is invalid")
        output.append(
            {
                "id": str(region.get("id") or f"{certainty}-{index}"),
                "certainty": certainty,
                "polygon": _validated_polygon(region.get("polygon")),
            }
        )
    return output


def _validate_open_polygon(value: object) -> None:
    if value is None:
        return
    if type(value) is not dict or value.get("certainty") not in {"certain", "uncertain"}:
        raise ValueError("open reference polygon is invalid")
    polygon = value.get("polygon")
    if type(polygon) is not list:
        raise ValueError("open reference polygon is invalid")
    for point in polygon:
        if (
            type(point) is not list
            or len(point) != 2
            or any(
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(float(coordinate))
                or not 0.0 <= float(coordinate) <= 1.0
                for coordinate in point
            )
        ):
            raise ValueError("open reference polygon coordinate is invalid")


def validate_return_session(
    *,
    session: object,
    packet: object,
    private_index: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if (
        type(session) is not dict
        or type(packet) is not dict
        or session.get("schema_version") != 1
        or packet.get("schema_version") != 1
        or session.get("packet_id") != packet.get("packet_id")
        or type(packet.get("items")) is not list
        or type(session.get("items")) is not dict
    ):
        raise ValueError("return session identity is invalid")
    _validate_packet_integrity(packet)
    packet_items = packet["items"]
    items_by_id = {
        str(item.get("item_id")): item
        for item in packet_items
        if type(item) is dict and item.get("item_id")
    }
    if len(items_by_id) != len(packet_items) or set(session["items"]) - set(items_by_id):
        raise ValueError("return session item cannot be matched")

    if packet.get("packet_kind") == REFERENCE_PACKET_KIND:
        if session.get("session_kind") != REFERENCE_SESSION_KIND:
            raise ValueError("reference return session kind is invalid")
        counts = {state_name: 0 for state_name in REFERENCE_STATES}
        for item_id in items_by_id:
            value = session["items"].get(item_id, {"state": "UNANNOTATED"})
            if type(value) is not dict or value.get("state") not in REFERENCE_STATES:
                raise ValueError("reference application state is invalid")
            application_state = str(value["state"])
            regions = _validated_regions(value.get("regions", []), certainty="certain")
            _validated_regions(value.get("uncertain_regions", []), certainty="uncertain")
            _validate_open_polygon(value.get("open_polygon"))
            if value.get("open_polygon") is not None and application_state != "DRAFT":
                raise ValueError("only a draft may contain an open polygon")
            if application_state == "CONFIRMED" and not regions:
                raise ValueError("confirmed reference needs a certain region")
            if application_state == "CONFIRMED_NO_CERTAIN_REGION" and regions:
                raise ValueError("no-certain reference cannot contain a certain region")
            if application_state in {"CONFIRMED", "CONFIRMED_NO_CERTAIN_REGION"}:
                reviewer = session.get("reviewer")
                if (
                    type(reviewer) is not dict
                    or type(reviewer.get("reviewer_alias")) is not str
                    or not reviewer["reviewer_alias"].strip()
                    or reviewer.get("reference_type") not in HUMAN_REFERENCE_TYPES
                    or type(reviewer.get("participated_in_method_development")) is not bool
                    or type(reviewer.get("saw_model_outputs")) is not bool
                    or reviewer.get("annotation_origin") not in ANNOTATION_ORIGINS
                ):
                    raise ValueError("confirmed reference provenance is incomplete")
            counts[application_state] += 1
        canonical_ready = counts["CONFIRMED"] + counts["CONFIRMED_NO_CERTAIN_REGION"]
        return {
            "schema_version": 1,
            "status": "VALID_SESSION_BACKUP",
            "session_kind": REFERENCE_SESSION_KIND,
            "item_count": len(items_by_id),
            "draft_count": counts["DRAFT"],
            "canonical_ready_count": canonical_ready,
            "reviewed_no_certain_region_count": counts[
                "CONFIRMED_NO_CERTAIN_REGION"
            ],
            "unable_to_judge_count": counts["UNABLE_TO_JUDGE"],
            "unannotated_count": counts["UNANNOTATED"],
        }

    if packet.get("packet_kind") == BLIND_PACKET_KIND:
        if (
            session.get("session_kind") != BLIND_SESSION_KIND
            or session.get("blinding_condition") != BLINDING_CONDITION
            or packet.get("blinding_condition") != BLINDING_CONDITION
        ):
            raise ValueError("blind return session kind is invalid")
        report_ids = {str(item["report_id"]) for item in packet_items}
        if private_index is not None and not report_ids.issubset(private_index):
            raise ValueError("blind private index does not match packet")
        counts = {state_name: 0 for state_name in BLIND_APPLICATION_STATES}
        unable = 0
        for item_id in items_by_id:
            value = session["items"].get(item_id, {"state": "UNREVIEWED"})
            if type(value) is not dict or value.get("state") not in BLIND_APPLICATION_STATES:
                raise ValueError("blind application state is invalid")
            application_state = str(value["state"])
            if application_state == "CONFIRMED":
                if (
                    type(session.get("reviewer_id")) is not str
                    or not session["reviewer_id"].strip()
                    or value.get("decision") not in BLIND_REVIEW_DECISIONS
                ):
                    raise ValueError("confirmed blind review is incomplete")
                unable += value["decision"] == "UNABLE_TO_JUDGE"
            counts[application_state] += 1
        return {
            "schema_version": 1,
            "status": "VALID_SESSION_BACKUP",
            "session_kind": BLIND_SESSION_KIND,
            "item_count": len(items_by_id),
            "draft_count": counts["DRAFT"],
            "confirmed_review_count": counts["CONFIRMED"],
            "unable_to_judge_count": unable,
            "unreviewed_count": counts["UNREVIEWED"],
        }
    raise ValueError("unsupported return packet kind")


def _safe_component(value: str, *, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not safe:
        safe = fallback + "-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    if safe in {".", ".."} or len(safe) > 80:
        raise ValueError("output identity is invalid")
    return safe


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(_json_bytes(payload))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("CSV rows cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _practice_reference_packet(source_base_sha: str) -> dict[str, object]:
    height, width = 120, 210
    rows, columns = np.indices((height, width))
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = np.clip(38 + columns * 3 // 5, 0, 255)
    pixels[..., 1] = np.clip(54 + rows, 0, 255)
    pixels[..., 2] = 112
    ellipse = ((rows - 70) / 19) ** 2 + ((columns - 142) / 34) ** 2 <= 1
    pixels[ellipse] = (231, 82, 48)
    data, display_sha = _png_data(Image.fromarray(pixels, mode="RGB"))
    item = {
        "item_id": "reference-practice-001",
        "specimen_key": "TEST_ONLY:asymmetric-practice",
        "native_width": width,
        "native_height": height,
        "source_image_sha256": display_sha,
        "display_image_sha256": display_sha,
        "frame": "registered_cscan",
        "image_mime": "image/png",
        "image_data": data,
    }
    return {
        "schema_version": 1,
        "packet_kind": REFERENCE_PACKET_KIND,
        "packet_id": _packet_id(REFERENCE_PACKET_KIND, source_base_sha, (item,)),
        "source_base_sha": source_base_sha,
        "packet_number": 0,
        "item_count": 1,
        "test_only": True,
        "practice_notice": "仅用于练习界面操作，不得导入正式研究结果。",
        "items": [item],
    }


def prepare_reference_packets(
    config: HumanReviewConfig,
    *,
    source_root: str | Path,
    packet_root: str | Path | None = None,
    result_root: str | Path | None = None,
    dist_root: str | Path | None = None,
) -> dict[str, object]:
    if type(config) is not HumanReviewConfig:
        raise TypeError("issued human-review config is required")
    frozen = load_frozen_process_config(
        config.frozen_process_config, project_root=config.project_root
    )
    parent = load_study_config(
        frozen.parent_config_path, project_root=config.project_root
    )
    roster = load_study_roster(parent, source_root=source_root)
    records = tuple(
        assignment.record
        for assignment in roster.assignments
        if assignment.split is Split.TEST
    )
    cohort = pl.read_csv(config.cohort_path, infer_schema_length=None).filter(
        pl.col("split") == Split.TEST.value
    )
    cohort_keys = set(cohort["specimen_key"].to_list())
    record_keys = {record.specimen_key for record in records}
    if record_keys != cohort_keys:
        raise ValueError("reference TEST roster differs from frozen cohort")
    built = build_reference_packets(
        records,
        packet_size=config.reference_packet_size,
        source_base_sha=config.repository_base_sha,
    )
    packets_path = Path(packet_root or config.local_root / "reference_packets").resolve()
    results_path = Path(result_root or config.result_root).resolve()
    delivery_path = Path(dist_root or config.dist_root).resolve()
    packets_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    delivery_path.mkdir(parents=True, exist_ok=True)
    packet_files = []
    for index, packet in enumerate(built["packets"], start=1):
        path = packets_path / f"reference_packet_{index:03d}.json"
        _write_json(path, packet)
        packet_files.append(str(path))
    manifest_rows = []
    for row in built["manifest"]:
        packet_number = int(row["packet_number"])
        manifest_rows.append(
            {
                **row,
                "packet_path": packet_files[packet_number - 1],
            }
        )
    manifest_path = results_path / "reference_packet_manifest.csv"
    _write_csv(manifest_path, manifest_rows)
    practice_path = delivery_path / "reference_practice_TEST_ONLY.json"
    _write_json(practice_path, _practice_reference_packet(config.repository_base_sha))
    summary = {
        "schema_version": 1,
        "stage": TOOL_STAGE,
        "reference_packet_status": "PREPARED",
        "physical_specimen_count": len(records),
        "domain_count": len({record.dataset_id for record in records}),
        "packet_count": len(packet_files),
        "missing_source_count": 0,
        "source_image_bytes": sum(record.cscan_path.stat().st_size for record in records),
        "packet_root": str(packets_path),
        "packet_paths": packet_files,
        "manifest_path": str(manifest_path),
        "practice_packet_path": str(practice_path),
        "real_reference_return_count": 0,
        "scientific_evaluation_status": "NOT_RUN_NO_HUMAN_INPUT",
        "resource_use": dict(config.resource_limits),
    }
    _write_json(results_path / "reference_build_summary.json", summary)
    return summary


def export_reference_session(
    *,
    session: object,
    packet: object,
    output_root: str | Path,
) -> dict[str, object]:
    validate_return_session(session=session, packet=packet)
    if (
        type(session) is not dict
        or type(packet) is not dict
        or session.get("schema_version") != 1
        or session.get("session_kind") != REFERENCE_SESSION_KIND
        or packet.get("schema_version") != 1
        or packet.get("packet_kind") != REFERENCE_PACKET_KIND
        or session.get("packet_id") != packet.get("packet_id")
    ):
        raise ValueError("reference session identity is invalid")
    reviewer = session.get("reviewer")
    states = session.get("items")
    export_id = session.get("export_id")
    if type(reviewer) is not dict or type(states) is not dict or type(export_id) is not str:
        raise ValueError("reference session fields are invalid")
    reviewer_alias = reviewer.get("reviewer_alias")
    reference_type = reviewer.get("reference_type")
    if (
        type(reviewer_alias) is not str
        or not reviewer_alias.strip()
        or reference_type not in HUMAN_REFERENCE_TYPES
        or type(reviewer.get("participated_in_method_development")) is not bool
        or type(reviewer.get("saw_model_outputs")) is not bool
        or reviewer.get("annotation_origin") not in ANNOTATION_ORIGINS
    ):
        raise ValueError("reference reviewer provenance is incomplete")
    packet_items = packet.get("items")
    if type(packet_items) is not list:
        raise ValueError("reference packet items are invalid")
    items_by_id = {
        str(item.get("item_id")): item
        for item in packet_items
        if type(item) is dict and item.get("item_id")
    }
    if len(items_by_id) != len(packet_items) or set(states) - set(items_by_id):
        raise ValueError("reference session item cannot be matched")

    reviewer_dir = _safe_component(reviewer_alias.strip(), fallback="reviewer")
    export_dir = _safe_component(export_id, fallback="export")
    root = Path(output_root).resolve() / reviewer_dir / export_dir
    if root.exists():
        raise ValueError("reference export already exists")
    references_root = root / "references"
    references_root.mkdir(parents=True)
    audit_rows = []
    exported = 0
    no_certain = 0
    for item_id, item in items_by_id.items():
        state_payload = states.get(item_id, {"state": "UNANNOTATED"})
        if type(state_payload) is not dict or state_payload.get("state") not in REFERENCE_STATES:
            raise ValueError("reference application state is invalid")
        state = str(state_payload["state"])
        regions = _validated_regions(state_payload.get("regions", []), certainty="certain")
        uncertain = _validated_regions(
            state_payload.get("uncertain_regions", []), certainty="uncertain"
        )
        if state == "CONFIRMED" and not regions:
            raise ValueError("confirmed reference needs a certain region")
        if state == "CONFIRMED_NO_CERTAIN_REGION" and regions:
            raise ValueError("no-certain reference cannot contain a certain region")
        exportable = state in {"CONFIRMED", "CONFIRMED_NO_CERTAIN_REGION"}
        status = state
        if exportable:
            payload = {
                "specimen_key": item["specimen_key"],
                "source_image_sha256": item["source_image_sha256"],
                "reference_type": reference_type,
                "review_state": "reviewed",
                "reviewer_alias": reviewer_alias.strip(),
                "frame": "registered_cscan",
                "regions": regions,
                "uncertain_regions": uncertain,
                "notes": str(state_payload.get("notes") or ""),
            }
            parsed = reference_from_payload(
                payload,
                native_shape=(int(item["native_height"]), int(item["native_width"])),
            )
            if (
                parsed.specimen_key != item["specimen_key"]
                or parsed.source_image_sha256 != item["source_image_sha256"]
            ):
                raise ValueError("reference export identity changed")
            output_name = f"{item_id}.json"
            _write_json(references_root / output_name, payload)
            exported += 1
            if state == "CONFIRMED_NO_CERTAIN_REGION":
                no_certain += 1
                status = "REVIEWED_NO_CERTAIN_REGION"
            else:
                status = "EXPORTED_REVIEWED_REFERENCE"
        audit_rows.append(
            {
                "schema_version": 1,
                "packet_id": packet["packet_id"],
                "item_id": item_id,
                "specimen_key": item["specimen_key"],
                "source_image_sha256": item["source_image_sha256"],
                "reviewer_alias": reviewer_alias.strip(),
                "reference_type": reference_type,
                "application_state": state,
                "export_status": status,
                "notes": str(state_payload.get("notes") or ""),
            }
        )

    raw_session_path = root / "session.json"
    audit_path = root / "annotation_audit.csv"
    _write_json(raw_session_path, session)
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(audit_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    return {
        "schema_version": 1,
        "status": "VALIDATED",
        "references_root": str(references_root),
        "annotation_audit_path": str(audit_path),
        "raw_session_path": str(raw_session_path),
        "exported_reference_count": exported,
        "reviewed_no_certain_region_count": no_certain,
        "not_exported_count": len(audit_rows) - exported,
    }


def export_blind_review_session(
    *,
    session: object,
    packet: object,
    private_index: Mapping[str, Mapping[str, object]],
    output_root: str | Path,
) -> dict[str, object]:
    validate_return_session(
        session=session, packet=packet, private_index=private_index
    )
    if (
        type(session) is not dict
        or type(packet) is not dict
        or session.get("schema_version") != 1
        or session.get("session_kind") != BLIND_SESSION_KIND
        or packet.get("schema_version") != 1
        or packet.get("packet_kind") != BLIND_PACKET_KIND
        or session.get("packet_id") != packet.get("packet_id")
        or session.get("blinding_condition") != BLINDING_CONDITION
        or packet.get("blinding_condition") != BLINDING_CONDITION
        or not isinstance(private_index, Mapping)
    ):
        raise ValueError("blind session identity is invalid")
    reviewer_id = session.get("reviewer_id")
    export_id = session.get("export_id")
    states = session.get("items")
    packet_items = packet.get("items")
    if (
        type(reviewer_id) is not str
        or not reviewer_id.strip()
        or type(export_id) is not str
        or not export_id
        or type(states) is not dict
        or type(packet_items) is not list
    ):
        raise ValueError("blind session fields are invalid")
    items_by_id = {}
    report_ids = set()
    for item in packet_items:
        if type(item) is not dict or not item.get("item_id") or not item.get("report_id"):
            raise ValueError("blind packet item is invalid")
        item_id = str(item["item_id"])
        report_id = str(item["report_id"])
        if item_id in items_by_id or report_id in report_ids:
            raise ValueError("blind packet item is duplicated")
        items_by_id[item_id] = item
        report_ids.add(report_id)
    if set(states) - set(items_by_id):
        raise ValueError("blind session item cannot be matched")
    if not report_ids.issubset(private_index):
        raise ValueError("blind private index does not match packet")

    rows = []
    for item_id, item in items_by_id.items():
        state_payload = states.get(item_id, {"state": "UNREVIEWED"})
        if (
            type(state_payload) is not dict
            or state_payload.get("state") not in BLIND_APPLICATION_STATES
        ):
            raise ValueError("blind application state is invalid")
        if state_payload["state"] != "CONFIRMED":
            continue
        decision = state_payload.get("decision")
        if decision not in BLIND_REVIEW_DECISIONS:
            raise ValueError("confirmed blind review needs a valid decision")
        rows.append(
            {
                "report_id": item["report_id"],
                "reviewer_id": reviewer_id.strip(),
                "decision": decision,
                "problem_type": str(state_payload.get("problem_type") or ""),
                "review_basis": str(state_payload.get("review_basis") or ""),
                "blinding_condition": BLINDING_CONDITION,
                "notes": str(state_payload.get("notes") or ""),
            }
        )
    summary = summarize_blind_reviews(tuple(rows), report_index=private_index)

    reviewer_dir = _safe_component(reviewer_id.strip(), fallback="reviewer")
    export_dir = _safe_component(export_id, fallback="export")
    root = Path(output_root).resolve() / reviewer_dir / export_dir
    if root.exists():
        raise ValueError("blind review export already exists")
    root.mkdir(parents=True)
    raw_session_path = root / "session.json"
    reviews_path = root / "blind_reviews.csv"
    summary_path = root / "summary.json"
    _write_json(raw_session_path, session)
    fields = (
        "report_id",
        "reviewer_id",
        "decision",
        "problem_type",
        "review_basis",
        "blinding_condition",
        "notes",
    )
    with reviews_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_json(summary_path, summary)
    return {
        "schema_version": 1,
        "status": "VALIDATED",
        "blind_reviews_path": str(reviews_path),
        "raw_session_path": str(raw_session_path),
        "summary_path": str(summary_path),
        "confirmed_review_count": len(rows),
        "unreviewed_count": len(items_by_id) - len(rows),
        "summary": summary,
    }


__all__ = [
    "HumanReviewConfig",
    "ReplayedEndpoint",
    "build_blind_packets",
    "build_delivery_ui",
    "build_reference_packets",
    "export_blind_review_session",
    "export_reference_session",
    "load_human_review_config",
    "prepare_blind_packets",
    "prepare_reference_packets",
    "recover_blind_reports",
    "render_blind_layers",
    "replay_stored_prefix",
    "select_blind_stop_rows",
    "validate_return_session",
    "write_blind_packet_bundle",
]
