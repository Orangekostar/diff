"""Fail-closed P1 visual-observability configuration and orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from yaml.nodes import MappingNode

_CONFIG_SHA256 = "00f3e0cf45d45dd64c20852513d9b23c69a3c29ad7e0d0d7220fb13f86bfe92e"
_DOMAINS = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)
_ROSTER_SHA256 = "4fd8c6076dd3fcdf908a73739251db215fcb01f570f1a930b7faf250fe6d285a"
_REGISTRATION_SHA256 = (
    "38ab3cf32e866cda447a5edf2637fa502406c4c5c574bc966c13cc1cbbd2553a"
)
_TRANSFORM_SHA256 = "2b275ebbc220e6a0376d305d0996f4ffe80509fc8b27223fd919331a100acbe5"
_WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


class P1ConfigError(ValueError):
    """Raised when the preregistered P1 protocol or a source drifts."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    output: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in output
        except TypeError as error:
            raise P1ConfigError("P1 config key is not hashable") from error
        if duplicate:
            raise P1ConfigError("P1 config contains a duplicate key")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class P1Source:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class P1Config:
    config_path: Path
    config_sha256: str
    project_root: Path
    sources: Mapping[str, P1Source]
    raw: Mapping[str, Any]
    domain_order: tuple[str, ...]
    authorized_specimen_count: int
    authorized_roster_sha256: str
    registration_authority_sha256: str
    target_rows: int
    target_path: Path
    encoder_weight_sha256: str
    surface_transform_sha256: str
    bootstrap_resamples: int
    output_work: Path
    output_result: Path
    output_replay: Path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise P1ConfigError(f"{label} must be a string-keyed mapping")
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _load_yaml(path: Path) -> tuple[Mapping[str, object], bytes]:
    if not path.is_file() or path.is_symlink():
        raise P1ConfigError("P1 config must be a regular file")
    try:
        payload = path.read_bytes()
        text = payload.decode("ascii")
        parsed = yaml.load(text, Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise P1ConfigError("P1 config cannot be read") from error
    return _mapping(parsed, "P1 config"), payload


def _load_sources(
    raw: object, *, root: Path
) -> Mapping[str, P1Source]:
    values = _mapping(raw, "P1 sources")
    if len(values) != 20:
        raise P1ConfigError("P1 source registry changed")
    output: dict[str, P1Source] = {}
    for name, raw_entry in values.items():
        entry = _mapping(raw_entry, f"P1 source {name}")
        if set(entry) != {"path", "sha256"}:
            raise P1ConfigError("P1 source registry changed")
        path_text = entry["path"]
        expected_hash = entry["sha256"]
        if (
            type(path_text) is not str
            or not path_text
            or Path(path_text).is_absolute()
            or ".." in Path(path_text).parts
            or type(expected_hash) is not str
            or len(expected_hash) != 64
        ):
            raise P1ConfigError("P1 source binding is invalid")
        unresolved = root / path_text
        try:
            resolved = unresolved.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise P1ConfigError(f"P1 source is unavailable: {name}") from error
        if (
            unresolved.is_symlink()
            or not resolved.is_file()
            or _sha256_file(resolved) != expected_hash
        ):
            raise P1ConfigError(f"P1 source changed: {name}")
        output[name] = P1Source(name=name, path=Path(path_text), sha256=expected_hash)
    return MappingProxyType(output)


def _validate_registered_content(raw: Mapping[str, object]) -> None:
    p0r = _mapping(raw.get("p0r_authority"), "P0R authority")
    target = _mapping(raw.get("target"), "P1 target")
    surface = _mapping(raw.get("surface_features"), "P1 surface features")
    transform = _mapping(surface.get("transform"), "P1 surface transform")
    encoder = _mapping(surface.get("encoder"), "P1 surface encoder")
    models = _mapping(raw.get("models"), "P1 models")
    controls = _mapping(raw.get("controls"), "P1 controls")
    c4 = _mapping(controls.get("C4"), "P1 wrong-orientation control")
    bootstrap = _mapping(raw.get("bootstrap"), "P1 bootstrap")
    outputs = _mapping(raw.get("outputs"), "P1 outputs")
    if (
        raw.get("schema_version") != 1
        or raw.get("stage") != "P1_VISUAL_OBSERVABILITY"
        or p0r.get("status") != "P0R_AUTHOR_REGISTRATION_GO"
        or p0r.get("authorized_specimen_count") != 276
        or p0r.get("authorized_roster_sha256") != _ROSTER_SHA256
        or p0r.get("registration_authority_sha256") != _REGISTRATION_SHA256
        or p0r.get("orientation") != "ROT90"
        or tuple(p0r.get("domain_order", ())) != _DOMAINS
        or target.get("expected_rows") != 17_664
        or target.get("expected_specimens") != 276
        or target.get("expected_cells_per_specimen") != 64
        or target.get("outer_target_labels")
        != "evaluation_only_after_score_freeze"
        or surface.get("encoder_roster") != ["resnet18_imagenet1k_v1_rgb"]
        or surface.get("optional_encoder_roster") != []
        or encoder.get("weights_sha256") != _WEIGHT_SHA256
        or encoder.get("frozen") is not True
        or encoder.get("train_encoder") is not False
        or transform.get("sha256") != _TRANSFORM_SHA256
        or models.get("outer_split") != "leave_one_domain_out"
        or models.get("inner_split") != "leave_one_source_domain_out"
        or models.get("target_scores_frozen_before_target_labels") is not True
        or "ROT90" in tuple(c4.get("roster", ()))
        or c4.get("excludes_correct_ROT90") is not True
        or bootstrap.get("resamples") != 100_000
        or outputs.get("result")
        != "results/agentic_task_driven_nde/p1_visual_observability"
    ):
        raise P1ConfigError("P1 frozen protocol changed")
    spec = transform.get("spec")
    try:
        spec_payload = json.dumps(
            spec, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    except (TypeError, UnicodeError, ValueError) as error:
        raise P1ConfigError("P1 transform specification is invalid") from error
    if _sha256_bytes(spec_payload) != _TRANSFORM_SHA256:
        raise P1ConfigError("P1 transform specification changed")


def load_p1_config(
    path: str | Path, *, project_root: str | Path
) -> P1Config:
    """Load P1 only when the exact preregistration and every source still match."""

    try:
        root = Path(project_root).resolve(strict=True)
        config_path = Path(path).resolve(strict=True)
    except OSError as error:
        raise P1ConfigError("P1 project or config path is unavailable") from error
    if not root.is_dir():
        raise P1ConfigError("P1 project root is invalid")
    raw, payload = _load_yaml(config_path)
    config_hash = _sha256_bytes(payload)
    if config_hash != _CONFIG_SHA256:
        raise P1ConfigError("P1 preregistered config bytes changed")
    _validate_registered_content(raw)
    sources = _load_sources(raw.get("sources"), root=root)
    p0r = _mapping(raw["p0r_authority"], "P0R authority")
    target = _mapping(raw["target"], "P1 target")
    surface = _mapping(raw["surface_features"], "P1 surface features")
    encoder = _mapping(surface["encoder"], "P1 encoder")
    transform = _mapping(surface["transform"], "P1 transform")
    bootstrap = _mapping(raw["bootstrap"], "P1 bootstrap")
    outputs = _mapping(raw["outputs"], "P1 outputs")
    return P1Config(
        config_path=config_path,
        config_sha256=config_hash,
        project_root=root,
        sources=sources,
        raw=_freeze(raw),  # type: ignore[arg-type]
        domain_order=_DOMAINS,
        authorized_specimen_count=int(p0r["authorized_specimen_count"]),
        authorized_roster_sha256=str(p0r["authorized_roster_sha256"]),
        registration_authority_sha256=str(p0r["registration_authority_sha256"]),
        target_rows=int(target["expected_rows"]),
        target_path=Path(str(target["path"])),
        encoder_weight_sha256=str(encoder["weights_sha256"]),
        surface_transform_sha256=str(transform["sha256"]),
        bootstrap_resamples=int(bootstrap["resamples"]),
        output_work=Path(str(outputs["work"])),
        output_result=Path(str(outputs["result"])),
        output_replay=Path(str(outputs["replay"])),
    )


__all__ = ["P1Config", "P1ConfigError", "P1Source", "load_p1_config"]
