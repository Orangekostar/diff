"""Frozen cohort and real-input adapters for the learned C-scan study."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from PIL import Image

from cmc_bbdm.vlm_cscan.runtime import (
    InputSpecimen,
    RuntimeRoster,
    SurfaceRender,
    load_benchmark_config,
    load_input_records,
    render_surface_inputs,
)

from .contracts import Split

STUDY_SPLIT_SEED = "learned-cscan-same-perception-pilot-v1"
STUDY_BASE_SHA = "59a67511c0ee6395b68220c6a1644f40c383dfa2"


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    record: InputSpecimen
    split: Split
    domain_rank: int


@dataclass(frozen=True, slots=True)
class StudyConfig:
    path: Path
    project_root: Path
    config_sha256: str
    prior_config_path: Path
    domain_order: tuple[str, ...]
    split_seed: str
    values: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class StudyRoster:
    records: tuple[InputSpecimen, ...]
    pilot_records: tuple[InputSpecimen, ...]
    assignments: tuple[SplitAssignment, ...]
    legacy_roster: RuntimeRoster


@dataclass(frozen=True, slots=True)
class RegisteredSurface:
    render: SurfaceRender
    clockwise_quarter_turns: int


def hash_split(
    records: tuple[InputSpecimen, ...],
    *,
    seed: str,
    train_per_domain: int = 4,
    valid_per_domain: int = 2,
    test_per_domain: int = 4,
) -> tuple[SplitAssignment, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(record) is not InputSpecimen for record in records)
        or not seed
        or any(
            type(value) is not int or value < 1
            for value in (train_per_domain, valid_per_domain, test_per_domain)
        )
    ):
        raise ValueError("study split request is invalid")
    per_domain = train_per_domain + valid_per_domain + test_per_domain
    keys = [record.specimen_key for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("study split contains duplicate specimens")
    assignments: list[SplitAssignment] = []
    for domain in sorted({record.dataset_id for record in records}):
        candidates = [record for record in records if record.dataset_id == domain]
        if len(candidates) != per_domain:
            raise ValueError("study split requires the frozen per-domain roster")
        ranked = sorted(
            candidates,
            key=lambda record: (
                hashlib.sha256(f"{seed}|{record.specimen_key}".encode()).hexdigest(),
                record.specimen_key,
            ),
        )
        for rank, record in enumerate(ranked):
            if rank < train_per_domain:
                split = Split.TRAIN
            elif rank < train_per_domain + valid_per_domain:
                split = Split.VALID
            else:
                split = Split.TEST
            assignments.append(SplitAssignment(record, split, rank))
    return tuple(assignments)


def require_fit_split(split: Split) -> None:
    if type(split) is not Split:
        raise TypeError("typed split is required")
    if split is not Split.TRAIN:
        raise ValueError("fitting is restricted to TRAIN")


def load_study_config(
    path: str | Path, *, project_root: str | Path
) -> StudyConfig:
    root = Path(project_root).resolve(strict=True)
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError("study config is invalid YAML") from error
    if type(payload) is not dict:
        raise ValueError("study config is invalid")
    cohort = payload.get("cohort")
    legacy = payload.get("legacy_runtime")
    if (
        payload.get("schema_version") != 1
        or payload.get("stage") != "LEARNED_CSCAN_SAME_PERCEPTION"
        or payload.get("repository_base_sha") != STUDY_BASE_SHA
        or payload.get("configuration_frozen") is not True
        or type(cohort) is not dict
        or type(legacy) is not dict
        or cohort.get("split_seed") != STUDY_SPLIT_SEED
        or cohort.get("train_per_domain") != 4
        or cohort.get("valid_per_domain") != 2
        or cohort.get("test_per_domain") != 4
    ):
        raise ValueError("study config identity is invalid")
    domains = cohort.get("domain_order")
    if (
        type(domains) is not list
        or len(domains) != 6
        or any(type(domain) is not str or not domain for domain in domains)
        or len(set(domains)) != len(domains)
    ):
        raise ValueError("study domains are invalid")
    relative = Path(legacy.get("config_path", ""))
    expected = legacy.get("config_sha256")
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or type(expected) is not str
        or len(expected) != 64
    ):
        raise ValueError("legacy runtime binding is invalid")
    prior_path = (root / relative).resolve(strict=True)
    if _file_sha256(prior_path) != expected:
        raise ValueError("legacy runtime config hash changed")
    load_benchmark_config(prior_path, project_root=root)
    return StudyConfig(
        path=source,
        project_root=root,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        prior_config_path=prior_path,
        domain_order=tuple(domains),
        split_seed=cohort["split_seed"],
        values=MappingProxyType(payload),
    )


def load_study_roster(
    config: StudyConfig,
    *,
    source_root: str | Path,
    verify_pilot_hashes: bool = True,
) -> StudyRoster:
    if type(config) is not StudyConfig:
        raise TypeError("issued study config is required")
    legacy_config = load_benchmark_config(
        config.prior_config_path, project_root=config.project_root
    )
    legacy_roster = load_input_records(
        legacy_config,
        source_root=source_root,
        verify_pilot_hashes=verify_pilot_hashes,
    )
    assignments = hash_split(
        legacy_roster.pilot_records,
        seed=config.split_seed,
        train_per_domain=4,
        valid_per_domain=2,
        test_per_domain=4,
    )
    if tuple(sorted(config.domain_order)) != tuple(
        sorted({row.record.dataset_id for row in assignments})
    ):
        raise ValueError("study roster domains differ from config")
    return StudyRoster(
        records=legacy_roster.records,
        pilot_records=legacy_roster.pilot_records,
        assignments=assignments,
        legacy_roster=legacy_roster,
    )


def render_registered_surface(
    record: InputSpecimen, *, max_edge: int
) -> RegisteredSurface:
    if type(record) is not InputSpecimen:
        raise TypeError("issued input specimen is required")
    with Image.open(record.surface_path) as image:
        image.load()
        render = render_surface_inputs(image, max_edge=max_edge)
    return RegisteredSurface(render=render, clockwise_quarter_turns=1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "STUDY_BASE_SHA",
    "STUDY_SPLIT_SEED",
    "RegisteredSurface",
    "SplitAssignment",
    "StudyConfig",
    "StudyRoster",
    "hash_split",
    "load_study_config",
    "load_study_roster",
    "render_registered_surface",
    "require_fit_split",
]
