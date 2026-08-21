"""Frozen pre-outer configuration for D8 residual diffusion training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import diffusers
import numpy as np
import PIL
import pywt
import scipy
import sklearn
import torch
import torchvision
import yaml

from .config import SourceRecord

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_CONFIG_BYTES = 256 * 1024
_MAX_SOURCE_BYTES = 512 * 1024 * 1024
_CANDIDATE_IDS = tuple(f"RD{index}" for index in range(8))
_SOURCE_PATHS = {
    "prompt": (
        "docs/Codex Optimization Prompt — Result-Oriented Diffusion "
        "Marginalization for Cross-Domain CAI.md"
    ),
    "pilot_decision": "docs/D8_PILOT_DECISION.md",
    "residual_training_design": (
        "docs/superpowers/specs/"
        "2026-08-18-d8-residual-diffusion-training-design.md"
    ),
    "residual_training_plan": (
        "docs/superpowers/plans/"
        "2026-08-18-d8-residual-diffusion-training.md"
    ),
    "exploration_config": "paper_v3/configs/d8_exploration.yaml",
    "pilot_manifest": "results/d8_search/artifact_manifest.json",
    "escalation_evidence": "results/d8_search/escalation_evidence.json",
    "d8_requirements": "environment/requirements-d8.txt",
    "resnet_weights": "paper_v3/assets/resnet18-f37072fd.pth",
}


class ResidualConfigError(ValueError):
    """Raised when the frozen residual-diffusion contract drifts."""


@dataclass(frozen=True, slots=True)
class ResidualCandidate:
    """One registered compact conditional diffusion candidate."""

    candidate_id: str
    base_channels: int
    prediction_type: str
    beta_schedule: str
    bottleneck_attention: bool
    spectral_weight: float
    low_pass_weight: float


@dataclass(frozen=True, slots=True)
class ResidualDiffusionConfig:
    """Immutable authority for the pre-outer training branch."""

    schema_version: int
    scope: str
    outer_evaluation_count: int
    output_dir: str
    replay_output_dir: str
    sources: Mapping[str, SourceRecord]
    runtime: Mapping[str, str]
    candidates: Mapping[str, ResidualCandidate]
    candidate_ids: tuple[str, ...]
    pilot_decision: str
    pilot_outer_evaluation_count: int
    pilot_scientific_digest: str
    screening_epochs: int
    screening_seed: int
    finalists_per_outer: int
    rerank_epochs: int
    training_seeds: tuple[int, ...]
    objective_weights: tuple[float, float, float]
    minimum_overall_acceptance: float
    minimum_domain_acceptance: float
    promotion_margin: float
    ensemble_margin: float
    train_timesteps: int
    sample_steps: int
    sample_eta: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    config_sha256: str

    def candidate(self, candidate_id: str) -> ResidualCandidate:
        if type(candidate_id) is not str or candidate_id not in self.candidates:
            raise ResidualConfigError("candidate ID is not registered")
        return self.candidates[candidate_id]


class _UniqueLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ResidualConfigError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _read_regular(path: Path, label: str, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ResidualConfigError(f"{label} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise ResidualConfigError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ResidualConfigError(f"{label} is not a bounded regular file")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise ResidualConfigError(f"{label} changed while reading")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise ResidualConfigError(f"{label} changed while reading")
    return payload


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ResidualConfigError(f"{label} must be a string mapping")
    return value


def _strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(expected, float) and not math.isfinite(actual):
        return False
    return actual == expected


def _expect(actual: object, expected: object, label: str) -> None:
    if not _strict_equal(actual, expected):
        raise ResidualConfigError(f"{label} changed")


def _safe_relative(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ResidualConfigError(f"{label} path must be text")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ResidualConfigError(f"{label} path is unsafe")
    return path.as_posix()


def _distribution(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise ResidualConfigError(f"runtime package is missing: {name}") from error


def _runtime_values() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy_module": np.__version__,
        "numpy_distribution": _distribution("numpy"),
        "scipy_module": scipy.__version__,
        "scipy_distribution": _distribution("scipy"),
        "scikit_learn_module": sklearn.__version__,
        "scikit_learn_distribution": _distribution("scikit-learn"),
        "pil_module": PIL.__version__,
        "pillow_distribution": _distribution("Pillow"),
        "torch_module": str(torch.__version__),
        "torch_distribution": _distribution("torch"),
        "torchvision_module": torchvision.__version__,
        "torchvision_distribution": _distribution("torchvision"),
        "diffusers_module": diffusers.__version__,
        "diffusers_distribution": _distribution("diffusers"),
        "pywavelets_module": pywt.__version__,
        "pywavelets_distribution": _distribution("PyWavelets"),
        "pyyaml_module": yaml.__version__,
        "pyyaml_distribution": _distribution("PyYAML"),
    }


def _expected_candidates() -> dict[str, dict[str, object]]:
    return {
        "RD0": {
            "base_channels": 32,
            "prediction_type": "epsilon",
            "beta_schedule": "squared_cosine",
            "bottleneck_attention": False,
            "spectral_weight": 0.00,
            "low_pass_weight": 0.00,
        },
        "RD1": {
            "base_channels": 32,
            "prediction_type": "epsilon",
            "beta_schedule": "squared_cosine",
            "bottleneck_attention": False,
            "spectral_weight": 0.05,
            "low_pass_weight": 0.10,
        },
        "RD2": {
            "base_channels": 32,
            "prediction_type": "v_prediction",
            "beta_schedule": "squared_cosine",
            "bottleneck_attention": False,
            "spectral_weight": 0.05,
            "low_pass_weight": 0.10,
        },
        "RD3": {
            "base_channels": 32,
            "prediction_type": "sample",
            "beta_schedule": "squared_cosine",
            "bottleneck_attention": False,
            "spectral_weight": 0.05,
            "low_pass_weight": 0.10,
        },
        "RD4": {
            "base_channels": 64,
            "prediction_type": "epsilon",
            "beta_schedule": "squared_cosine",
            "bottleneck_attention": True,
            "spectral_weight": 0.05,
            "low_pass_weight": 0.10,
        },
        "RD5": {
            "base_channels": 64,
            "prediction_type": "v_prediction",
            "beta_schedule": "squared_cosine",
            "bottleneck_attention": True,
            "spectral_weight": 0.05,
            "low_pass_weight": 0.10,
        },
        "RD6": {
            "base_channels": 32,
            "prediction_type": "epsilon",
            "beta_schedule": "linear",
            "bottleneck_attention": False,
            "spectral_weight": 0.05,
            "low_pass_weight": 0.10,
        },
        "RD7": {
            "base_channels": 32,
            "prediction_type": "v_prediction",
            "beta_schedule": "linear",
            "bottleneck_attention": False,
            "spectral_weight": 0.05,
            "low_pass_weight": 0.10,
        },
    }


def _expected_sections() -> dict[str, object]:
    return {
        "pilot": {
            "decision": "TRAIN_RESIDUAL_DIFFUSION",
            "outer_evaluation_count": 0,
            "config_sha256": (
                "a040eb25a95b166ca674a1744b53088b7c1b5ec14c11a88fbada953846ff119b"
            ),
            "residual_bank_sha256": (
                "a8b4723cc343a7bb4480b24fd9495f08b856f798a30e9faf3a587cc0890be18b"
            ),
            "scientific_digest": (
                "3478d97858236c1873c88d8fc3e910dbe659e05d2c4e472eac15825e999474ca"
            ),
            "output_tree_sha256": (
                "685798b852590b37c2d3857b95e7de212ca32cc087cd306b99fa748d7946eb2d"
            ),
        },
        "search": {
            "screening_epochs": 24,
            "screening_seed": 20260823,
            "finalists_per_outer": 2,
            "rerank_epochs": 120,
            "training_seeds": [20260823, 20260824, 20260825],
            "objective_mean_weight": 1.0,
            "objective_worst_weight": 0.25,
            "objective_domain_sd_weight": 0.10,
            "minimum_overall_acceptance": 0.80,
            "minimum_domain_acceptance": 0.60,
            "promotion_margin": 0.0001,
            "ensemble_margin": 0.0001,
            "final_checkpoint_seeds": [20260823, 20260824, 20260825],
        },
        "training": {
            "resolution": 64,
            "input_channels": 6,
            "output_channels": 3,
            "layers_per_block": 1,
            "normalization": "group",
            "optimizer": "AdamW",
            "batch_size": 32,
            "learning_rate": 0.0002,
            "weight_decay": 0.0001,
            "train_timesteps": 1000,
            "residual_scale": 2.0,
            "checkpoint_selection": "final_epoch",
            "early_stopping": False,
        },
        "sampling": {
            "sampler": "DDIM",
            "steps": 25,
            "eta": 1.0,
            "seed_authority": "specimen_candidate_training_seed",
        },
        "execution": {
            "gpu_type": "NVIDIA A40",
            "gpu_count": 3,
            "prospective_outers_per_gpu": 2,
            "screening_model_count": 240,
            "rerank_model_count": 180,
            "maximum_final_checkpoint_count": 18,
            "blas_threads": 1,
        },
        "outputs": {
            "required_entries": [
                "config.yaml",
                "candidate_index.csv",
                "training.csv",
                "inner_predictions.csv",
                "inner_metrics.csv",
                "checkpoint_index.csv",
                "selected_generators.json",
                "frozen_pipelines.json",
                "models",
                "REPORT.md",
                "artifact_manifest.json",
                "CHECKSUMS.sha256",
            ],
            "formal_outer_status": "BLOCKED_PENDING_AUTHORIZED_ONE_WAY_RUN",
        },
    }


def _load_sources(
    value: object, *, project_root: Path
) -> tuple[Mapping[str, SourceRecord], dict[str, bytes]]:
    mapping = _mapping(value, "sources")
    if set(mapping) != set(_SOURCE_PATHS):
        raise ResidualConfigError("source keys changed")
    records: dict[str, SourceRecord] = {}
    payloads: dict[str, bytes] = {}
    for name, expected_path in _SOURCE_PATHS.items():
        record = _mapping(mapping[name], f"source {name}")
        if set(record) != {"path", "sha256"}:
            raise ResidualConfigError(f"source {name} keys changed")
        relative_path = _safe_relative(record["path"], f"source {name}")
        if relative_path != expected_path:
            raise ResidualConfigError(f"source {name} path changed")
        expected_sha = record["sha256"]
        if type(expected_sha) is not str or _SHA256.fullmatch(expected_sha) is None:
            raise ResidualConfigError(f"source {name} hash is invalid")
        payload = _read_regular(
            project_root / relative_path,
            f"source {name}",
            maximum=_MAX_SOURCE_BYTES,
        )
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected_sha:
            raise ResidualConfigError(f"source {name} hash mismatch")
        records[name] = SourceRecord(path=relative_path, sha256=expected_sha)
        payloads[name] = payload
    return MappingProxyType(records), payloads


def _validate_pilot_sources(payloads: Mapping[str, bytes]) -> None:
    try:
        manifest = json.loads(payloads["pilot_manifest"].decode("ascii"))
        escalation = json.loads(payloads["escalation_evidence"].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ResidualConfigError("Pilot JSON source is invalid") from error
    if (
        type(manifest) is not dict
        or manifest.get("scope") != "cpb_d8_pilot_search_package"
        or manifest.get("outer_evaluation_count") != 0
        or manifest.get("scientific_digest")
        != "3478d97858236c1873c88d8fc3e910dbe659e05d2c4e472eac15825e999474ca"
        or type(escalation) is not dict
        or escalation.get("decision") != "TRAIN_RESIDUAL_DIFFUSION"
        or escalation.get("config_sha256")
        != "a040eb25a95b166ca674a1744b53088b7c1b5ec14c11a88fbada953846ff119b"
    ):
        raise ResidualConfigError("Pilot decision authority changed")


def load_residual_diffusion_config(
    config_path: str | Path, *, project_root: str | Path
) -> ResidualDiffusionConfig:
    """Load and verify the exact residual-diffusion pre-outer authority."""

    root = Path(project_root).resolve()
    config_bytes = _read_regular(
        Path(config_path), "residual configuration", maximum=_MAX_CONFIG_BYTES
    )
    try:
        raw = yaml.load(config_bytes.decode("utf-8"), Loader=_UniqueLoader)
    except ResidualConfigError:
        raise
    except (UnicodeError, yaml.YAMLError) as error:
        raise ResidualConfigError("residual configuration is invalid YAML") from error
    payload = _mapping(raw, "root")
    root_keys = {
        "schema_version",
        "scope",
        "outer_evaluation_count",
        "output_dir",
        "replay_output_dir",
        "runtime",
        "sources",
        "pilot",
        "search",
        "training",
        "sampling",
        "candidates",
        "execution",
        "outputs",
    }
    if set(payload) != root_keys:
        raise ResidualConfigError("root keys changed")
    _expect(payload["schema_version"], 1, "schema version")
    _expect(payload["scope"], "cpb_d8_residual_diffusion_preouter", "scope")
    _expect(payload["outer_evaluation_count"], 0, "outer evaluation count")
    _expect(
        payload["output_dir"],
        "results/d8_residual_diffusion_search",
        "output directory",
    )
    _expect(
        payload["replay_output_dir"],
        "results/replay/d8_residual_diffusion_search",
        "replay output directory",
    )
    runtime = _mapping(payload["runtime"], "runtime")
    _expect(runtime, _runtime_values(), "runtime")
    sources, source_payloads = _load_sources(payload["sources"], project_root=root)
    _validate_pilot_sources(source_payloads)
    sections = _expected_sections()
    for name, expected in sections.items():
        _expect(payload[name], expected, name)
    candidates_payload = _mapping(payload["candidates"], "candidates")
    expected_candidates = _expected_candidates()
    if tuple(candidates_payload) != _CANDIDATE_IDS:
        raise ResidualConfigError("candidate order or roster changed")
    candidates: dict[str, ResidualCandidate] = {}
    for candidate_id in _CANDIDATE_IDS:
        candidate_payload = _mapping(
            candidates_payload[candidate_id], f"candidate {candidate_id}"
        )
        _expect(
            candidate_payload,
            expected_candidates[candidate_id],
            f"candidate {candidate_id}",
        )
        candidates[candidate_id] = ResidualCandidate(
            candidate_id=candidate_id,
            base_channels=int(candidate_payload["base_channels"]),
            prediction_type=str(candidate_payload["prediction_type"]),
            beta_schedule=str(candidate_payload["beta_schedule"]),
            bottleneck_attention=bool(candidate_payload["bottleneck_attention"]),
            spectral_weight=float(candidate_payload["spectral_weight"]),
            low_pass_weight=float(candidate_payload["low_pass_weight"]),
        )
    search = sections["search"]
    training = sections["training"]
    sampling = sections["sampling"]
    assert isinstance(search, dict)
    assert isinstance(training, dict)
    assert isinstance(sampling, dict)
    return ResidualDiffusionConfig(
        schema_version=1,
        scope="cpb_d8_residual_diffusion_preouter",
        outer_evaluation_count=0,
        output_dir="results/d8_residual_diffusion_search",
        replay_output_dir="results/replay/d8_residual_diffusion_search",
        sources=sources,
        runtime=MappingProxyType(dict(runtime)),
        candidates=MappingProxyType(candidates),
        candidate_ids=_CANDIDATE_IDS,
        pilot_decision="TRAIN_RESIDUAL_DIFFUSION",
        pilot_outer_evaluation_count=0,
        pilot_scientific_digest=str(sections["pilot"]["scientific_digest"]),
        screening_epochs=int(search["screening_epochs"]),
        screening_seed=int(search["screening_seed"]),
        finalists_per_outer=int(search["finalists_per_outer"]),
        rerank_epochs=int(search["rerank_epochs"]),
        training_seeds=tuple(search["training_seeds"]),
        objective_weights=(
            float(search["objective_mean_weight"]),
            float(search["objective_worst_weight"]),
            float(search["objective_domain_sd_weight"]),
        ),
        minimum_overall_acceptance=float(search["minimum_overall_acceptance"]),
        minimum_domain_acceptance=float(search["minimum_domain_acceptance"]),
        promotion_margin=float(search["promotion_margin"]),
        ensemble_margin=float(search["ensemble_margin"]),
        train_timesteps=int(training["train_timesteps"]),
        sample_steps=int(sampling["steps"]),
        sample_eta=float(sampling["eta"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )


__all__ = [
    "ResidualCandidate",
    "ResidualConfigError",
    "ResidualDiffusionConfig",
    "load_residual_diffusion_config",
]

