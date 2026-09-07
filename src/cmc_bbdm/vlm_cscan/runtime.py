"""Real surface and causal-world adapters."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from cmc_bbdm.agentic_nde.surface_cells import (
    SurfaceCellAuthority,
    SurfaceCellRecord,
    load_surface_cell_authority,
)
from cmc_bbdm.inspection_agent.contracts import InspectionObservation, InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    SourceBackgroundPrior,
)
from cmc_bbdm.inspection_agent.surface_hypothesis import compute_surface_hypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mavis.authority import MAVISAuthority, load_mavis_authority
from cmc_bbdm.mavis.config import load_mavis_config
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid, build_acquisition_grid


@dataclass(frozen=True, slots=True)
class SurfaceRender:
    clean: Image.Image
    gridded: Image.Image
    clean_sha256: str
    gridded_sha256: str


@dataclass(frozen=True, slots=True)
class InputSpecimen:
    dataset_id: str
    specimen_id: str
    surface_path: Path
    surface_sha256: str
    cscan_path: Path
    cscan_sha256: str
    native_shape: tuple[int, int]
    transform_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.dataset_id
            or not self.specimen_id
            or not isinstance(self.surface_path, Path)
            or not isinstance(self.cscan_path, Path)
            or any(
                len(value) != 64 or set(value) - set("0123456789abcdef")
                for value in (
                    self.surface_sha256,
                    self.cscan_sha256,
                    self.transform_sha256,
                )
            )
            or type(self.native_shape) is not tuple
            or len(self.native_shape) != 2
            or any(type(value) is not int or value < 9 for value in self.native_shape)
        ):
            raise ValueError("input specimen is invalid")

    @property
    def specimen_key(self) -> str:
        return f"{self.dataset_id}:{self.specimen_id}"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    path: Path
    project_root: Path
    config_sha256: str
    domain_order: tuple[str, ...]
    pilot_per_domain: int
    pilot_seed: str
    values: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeRoster:
    records: tuple[InputSpecimen, ...]
    pilot_records: tuple[InputSpecimen, ...]
    smoke_records: tuple[InputSpecimen, ...]
    surface_authority: SurfaceCellAuthority


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    config: BenchmarkConfig
    roster: RuntimeRoster
    authority: MAVISAuthority
    background_priors: MappingProxyType[str, SourceBackgroundPrior]
    source_root: Path


@dataclass(frozen=True, slots=True)
class SpecimenRuntime:
    record: InputSpecimen
    surface_record: SurfaceCellRecord
    render: SurfaceRender
    grid: AcquisitionGrid
    world: CausalInspectionWorld
    observation: InspectionObservation
    saliency_scores: np.ndarray
    background_rgb: np.ndarray


def render_surface_inputs(image: Image.Image, *, max_edge: int) -> SurfaceRender:
    if (
        not isinstance(image, Image.Image)
        or type(max_edge) is not int
        or max_edge < 10
        or image.width < 10
        or image.height < 10
    ):
        raise ValueError("surface render request is invalid")
    image.load()
    clean = image.convert("RGB").transpose(Image.Transpose.ROTATE_270)
    longest = max(clean.size)
    if longest > max_edge:
        scale = max_edge / longest
        size = tuple(max(10, math.floor(axis * scale + 0.5)) for axis in clean.size)
        clean = clean.resize(size, Image.Resampling.LANCZOS)
    gridded = clean.copy()
    draw = ImageDraw.Draw(gridded, "RGBA")
    width, height = gridded.size
    line_width = max(1, round(max(width, height) / 512))
    for index in range(1, 8):
        x = round(index * width / 8)
        y = round(index * height / 8)
        draw.line((x, 0, x, height - 1), fill=(0, 255, 255, 110), width=line_width)
        draw.line((0, y, width - 1, y), fill=(0, 255, 255, 110), width=line_width)
    for row in range(8):
        for column in range(8):
            draw.text(
                (round(column * width / 8) + 2, round(row * height / 8) + 1),
                str(row * 8 + column),
                fill=(255, 255, 255, 230),
                stroke_width=1,
                stroke_fill=(0, 0, 0, 220),
            )
    return SurfaceRender(
        clean=clean,
        gridded=gridded,
        clean_sha256=_image_sha256(clean),
        gridded_sha256=_image_sha256(gridded),
    )


def _image_sha256(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def select_pilot_records(
    records: tuple[InputSpecimen, ...],
    *,
    domain_order: tuple[str, ...],
    per_domain: int,
    seed: str,
) -> tuple[InputSpecimen, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(record) is not InputSpecimen for record in records)
        or type(domain_order) is not tuple
        or not domain_order
        or len(set(domain_order)) != len(domain_order)
        or type(per_domain) is not int
        or per_domain < 1
        or not seed
    ):
        raise ValueError("pilot selection request is invalid")
    output: list[InputSpecimen] = []
    seen_keys = set()
    for domain in domain_order:
        candidates = [record for record in records if record.dataset_id == domain]
        if len(candidates) < per_domain:
            raise ValueError("pilot domain has too few specimens")
        ranked = sorted(
            candidates,
            key=lambda record: (
                hashlib.sha256(
                    f"{seed}|{record.dataset_id}|{record.specimen_id}|"
                    f"{record.surface_sha256}|{record.cscan_sha256}".encode("ascii")
                ).hexdigest(),
                record.specimen_id,
            ),
        )
        output.extend(ranked[:per_domain])
        seen_keys.update(record.specimen_key for record in candidates)
    if len(seen_keys) != len(records):
        raise ValueError("input roster contains duplicate or unregistered domains")
    return tuple(output)


def load_benchmark_config(
    path: str | Path, *, project_root: str | Path
) -> BenchmarkConfig:
    root = Path(project_root).resolve(strict=True)
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError("benchmark config is invalid YAML") from error
    if type(payload) is not dict:
        raise ValueError("benchmark config is invalid")
    cohort = payload.get("cohort")
    model = payload.get("model")
    acquisition = payload.get("acquisition")
    surface = payload.get("surface")
    if (
        payload.get("schema_version") != 1
        or payload.get("stage") != "VLM_CSCAN_SUCCESS_EFFICIENCY"
        or payload.get("mode") != "pilot"
        or payload.get("repository_base_sha")
        != "78453de3fe01c261887bc56c41e04aa505526fa1"
        or payload.get("configuration_frozen") is not True
        or type(cohort) is not dict
        or type(model) is not dict
        or type(acquisition) is not dict
        or type(surface) is not dict
    ):
        raise ValueError("benchmark config identity is invalid")
    domains_raw = cohort.get("domain_order")
    if (
        type(domains_raw) is not list
        or not domains_raw
        or any(type(value) is not str or not value for value in domains_raw)
        or len(set(domains_raw)) != len(domains_raw)
        or type(cohort.get("pilot_per_domain")) is not int
        or cohort["pilot_per_domain"] < 1
        or type(cohort.get("pilot_seed")) is not str
        or not cohort["pilot_seed"]
        or model.get("repository") != "Qwen/Qwen2.5-VL-7B-Instruct"
        or model.get("revision") != "cc594898137f460bfe9f0759e9844b3ce807cfb5"
        or model.get("do_sample") is not False
        or surface.get("orientation") != "ROT90"
        or surface.get("grid_shape") != [8, 8]
        or surface.get("preprocessing_sha256")
        != "b341e99fb74d9f815be810b2e8b1d91e59f236540072577c244313c584281f40"
        or float(acquisition.get("endpoint_budget", 0.0)) != 1.0
        or float(acquisition.get("initial_nominal_budget", 0.0)) != 0.015625
    ):
        raise ValueError("benchmark config contract is invalid")
    sources = payload.get("sources")
    if type(sources) is not dict or not sources:
        raise ValueError("benchmark config sources are invalid")
    for binding in sources.values():
        if type(binding) is not dict or set(binding) != {"path", "sha256"}:
            raise ValueError("benchmark config source binding is invalid")
        relative = Path(binding["path"])
        expected = binding["sha256"]
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(expected) != 64
            or set(expected) - set("0123456789abcdef")
        ):
            raise ValueError("benchmark config source binding is invalid")
        actual = _file_sha256((root / relative).resolve(strict=True))
        if actual != expected:
            raise ValueError(f"source hash changed: {relative.as_posix()}")
    return BenchmarkConfig(
        path=source,
        project_root=root,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        domain_order=tuple(domains_raw),
        pilot_per_domain=cohort["pilot_per_domain"],
        pilot_seed=cohort["pilot_seed"],
        values=MappingProxyType(payload),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inner_border_median(
    image: np.ndarray, *, border_fraction: float
) -> np.ndarray:
    values = np.asarray(image)
    if (
        values.dtype != np.uint8
        or values.ndim != 3
        or values.shape[2] != 3
        or min(values.shape[:2]) < 9
        or not 0.0 <= float(border_fraction) < 0.25
    ):
        raise ValueError("inner-border background request is invalid")
    row_margin = max(1, round(values.shape[0] * float(border_fraction)))
    column_margin = max(1, round(values.shape[1] * float(border_fraction)))
    row_last = values.shape[0] - row_margin - 1
    column_last = values.shape[1] - column_margin - 1
    ring = np.concatenate(
        (
            values[row_margin, column_margin : column_last + 1],
            values[row_last, column_margin : column_last + 1],
            values[row_margin + 1 : row_last, column_margin],
            values[row_margin + 1 : row_last, column_last],
        ),
        axis=0,
    )
    return np.median(ring.astype(np.float64), axis=0)


def _fit_inner_source_background_priors(
    authority: MAVISAuthority,
    *,
    domain_order: tuple[str, ...],
    border_fraction: float,
) -> MappingProxyType[str, SourceBackgroundPrior]:
    if (
        type(authority) is not MAVISAuthority
        or type(domain_order) is not tuple
        or not domain_order
        or len(set(domain_order)) != len(domain_order)
    ):
        raise ValueError("source background prior request is invalid")
    domain_medians: dict[str, np.ndarray] = {}
    for domain in domain_order:
        medians = [
            _inner_border_median(
                authority.source_teacher_view(specimen).full_scan,
                border_fraction=border_fraction,
            )
            for specimen, specimen_domain in zip(
                authority.specimen_ids,
                authority.dataset_ids,
                strict=True,
            )
            if specimen_domain == domain
        ]
        if not medians:
            raise ValueError("source background prior domain is empty")
        domain_medians[domain] = np.mean(
            np.asarray(medians, dtype=np.float64),
            axis=0,
            dtype=np.float64,
        )
    priors = {}
    for outer_domain in domain_order:
        source_domains = tuple(
            domain for domain in domain_order if domain != outer_domain
        )
        domain_rows = np.asarray(
            [domain_medians[domain] for domain in source_domains],
            dtype=np.float64,
        )
        background = np.rint(domain_rows.mean(axis=0)).clip(0, 255).astype(np.uint8)
        fit_ids = tuple(
            specimen
            for specimen, domain in zip(
                authority.specimen_ids,
                authority.dataset_ids,
                strict=True,
            )
            if domain != outer_domain
        )
        priors[outer_domain] = SourceBackgroundPrior(
            outer_domain=outer_domain,
            source_domains=source_domains,
            fit_specimen_ids=fit_ids,
            source_authority_sha256=authority.state_sha256,
            domain_border_medians=domain_rows,
            background_rgb=background,
        )
    return MappingProxyType(priors)


def load_input_records(
    config: BenchmarkConfig,
    *,
    source_root: str | Path,
    verify_pilot_hashes: bool = True,
) -> RuntimeRoster:
    if type(config) is not BenchmarkConfig or type(verify_pilot_hashes) is not bool:
        raise ValueError("runtime roster request is invalid")
    external = Path(source_root).resolve(strict=True)
    bindings = config.values["sources"]
    authority = load_surface_cell_authority(
        config.project_root / bindings["p0r_surface_manifest"]["path"],
        config.project_root / bindings["p0r_registration"]["path"],
        config.project_root / bindings["p0r_grid_mapping"]["path"],
    )
    manifest_path = config.project_root / bindings["p0r_surface_manifest"]["path"]
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    rows_by_key = {(row["dataset_id"], row["specimen_id"]): row for row in rows}
    if len(rows_by_key) != len(rows) or len(rows) != authority.specimen_count:
        raise ValueError("P0R runtime roster changed")
    records = []
    for surface in authority.records:
        key = (surface.dataset_id, surface.specimen_id)
        row = rows_by_key.get(key)
        if row is None or Path(row["impacted_surface_path"]) != surface.surface_path:
            raise ValueError("P0R surface runtime identity changed")
        surface_path = _resolve_external(external, row["impacted_surface_path"])
        cscan_path = _resolve_external(external, row["registered_cscan_crop_path"])
        if not surface_path.is_file() or not cscan_path.is_file():
            raise ValueError("registered runtime image is unavailable")
        native_shape = (
            int(row["registered_cscan_height_px"]),
            int(row["registered_cscan_width_px"]),
        )
        if native_shape != (
            surface.destination.height_px,
            surface.destination.width_px,
        ):
            raise ValueError("registered C-scan shape changed")
        records.append(
            InputSpecimen(
                dataset_id=surface.dataset_id,
                specimen_id=surface.specimen_id,
                surface_path=surface_path,
                surface_sha256=surface.surface_sha256,
                cscan_path=cscan_path,
                cscan_sha256=row["registered_cscan_crop_sha256"],
                native_shape=native_shape,
                transform_sha256=surface.transform_sha256,
            )
        )
    roster = tuple(records)
    pilot = select_pilot_records(
        roster,
        domain_order=config.domain_order,
        per_domain=config.pilot_per_domain,
        seed=config.pilot_seed,
    )
    smoke = tuple(
        next(record for record in pilot if record.dataset_id == domain)
        for domain in config.domain_order
    )
    if verify_pilot_hashes:
        for record in pilot:
            if (
                _file_sha256(record.surface_path) != record.surface_sha256
                or _file_sha256(record.cscan_path) != record.cscan_sha256
            ):
                raise ValueError(f"pilot image hash changed: {record.specimen_key}")
    return RuntimeRoster(roster, pilot, smoke, authority)


def _resolve_external(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("external image path is invalid")
    resolved = (root / relative).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("external image escapes the research root")
    return resolved


def load_runtime_context(
    config: BenchmarkConfig,
    *,
    source_root: str | Path,
    verify_pilot_hashes: bool = True,
) -> RuntimeContext:
    if type(config) is not BenchmarkConfig:
        raise TypeError("issued benchmark config is required")
    external = Path(source_root).resolve(strict=True)
    roster = load_input_records(
        config,
        source_root=external,
        verify_pilot_hashes=verify_pilot_hashes,
    )
    mavis_binding = config.values["sources"]["mavis_config"]
    mavis_config = load_mavis_config(
        config.project_root / mavis_binding["path"],
        project_root=config.project_root,
    )
    authority = load_mavis_authority(mavis_config, source_project_root=external)
    authority_keys = set(
        zip(authority.dataset_ids, authority.specimen_ids, strict=True)
    )
    roster_keys = {(record.dataset_id, record.specimen_id) for record in roster.records}
    if authority_keys != roster_keys:
        raise ValueError("MAVIS and P0R rosters differ")
    priors = _fit_inner_source_background_priors(
        authority,
        domain_order=config.domain_order,
        border_fraction=float(config.values["reader"]["border_exclusion_fraction"]),
    )
    return RuntimeContext(
        config=config,
        roster=roster,
        authority=authority,
        background_priors=priors,
        source_root=external,
    )


def open_specimen_runtime(
    context: RuntimeContext, record: InputSpecimen
) -> SpecimenRuntime:
    if (
        type(context) is not RuntimeContext
        or type(record) is not InputSpecimen
        or record not in context.roster.records
    ):
        raise ValueError("specimen runtime request is invalid")
    surface_index = next(
        index
        for index, item in enumerate(context.roster.records)
        if item.specimen_key == record.specimen_key
    )
    surface_record = context.roster.surface_authority.records[surface_index]
    with Image.open(record.surface_path) as image:
        image.load()
        source_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        render = render_surface_inputs(
            image,
            max_edge=int(context.config.values["surface"]["max_edge"]),
        )
    hypothesis = compute_surface_hypothesis(
        source_rgb,
        surface_record.cell_boxes,
        top_k=int(context.config.values["planning"]["initial_priority_limit"]),
    )
    grid = build_acquisition_grid(
        *record.native_shape,
        initial_budget=float(
            context.config.values["acquisition"]["initial_nominal_budget"]
        ),
    )
    world = CausalInspectionWorld(
        context.authority,
        specimen_id=record.specimen_id,
        task=InspectionTask.FIELD,
        surface_rgb=np.asarray(render.clean, dtype=np.uint8),
        surface_sha256=render.clean_sha256,
        grid=grid,
        endpoint_budget=float(context.config.values["acquisition"]["endpoint_budget"]),
    )
    observation = world.reset()
    return SpecimenRuntime(
        record=record,
        surface_record=surface_record,
        render=render,
        grid=grid,
        world=world,
        observation=observation,
        saliency_scores=hypothesis.scores,
        background_rgb=context.background_priors[record.dataset_id].background_rgb,
    )


__all__ = [
    "BenchmarkConfig",
    "InputSpecimen",
    "RuntimeContext",
    "RuntimeRoster",
    "SpecimenRuntime",
    "SurfaceRender",
    "load_benchmark_config",
    "load_input_records",
    "load_runtime_context",
    "open_specimen_runtime",
    "render_surface_inputs",
    "select_pilot_records",
]
