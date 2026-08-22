"""Strict configuration authority for the registered MGMR M0 experiment."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml
from yaml.nodes import MappingNode


class MGMRProtocolError(ValueError):
    """Raised when the frozen M0 protocol is incomplete or has drifted."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects silent mapping-key replacement."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise MGMRProtocolError("config YAML mapping key is not hashable") from error
        if duplicate:
            raise MGMRProtocolError(f"config contains duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


_TOP_KEYS = {
    "schema_version",
    "scope",
    "sources",
    "cohort",
    "representation",
    "estimator",
    "residual",
    "specificity",
    "bootstrap",
    "gate",
    "runtime",
    "outputs",
}
_SOURCE_KEYS = {"path", "sha256"}
_COHORT_KEYS = {
    "specimen_count",
    "domain_order",
    "response",
    "response_unit",
    "outer_split",
    "inner_split",
}
_REPRESENTATION_KEYS = {
    "coarse_density",
    "coarse_operator",
    "spatial_layer",
    "spatial_shape",
    "secondary_layer",
    "secondary_shape",
    "wavelet",
    "wavelet_sensitivity",
    "wavelet_mode",
    "wavelet_level",
    "directional_order",
}
_ESTIMATOR_KEYS = {
    "metadata_features",
    "pca_dimensions",
    "pca_tie_tolerance",
    "preprocessing",
    "regressor",
    "ridge_alpha",
    "device",
    "batch_size",
}
_RESIDUAL_KEYS = {"target", "metadata_in_residual_branch", "correction_scale"}
_SPECIFICITY_KEYS = {"control", "seeds", "grid"}
_BOOTSTRAP_KEYS = {
    "seed",
    "resamples",
    "bit_generator",
    "unit",
    "quantiles",
}
_GATE_KEYS = {
    "required",
    "minimum_positive_domains",
    "baseline_mae",
    "positive_mae",
    "strong_mae",
    "stop_on_no_go",
}
_RUNTIME_KEYS = {"numpy", "pywavelets", "scikit_learn", "torch", "torchvision"}
_OUTPUT_KEYS = {"feature_bank", "formal", "replay"}
_DOMAINS = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)
_SOURCE_PATHS = {
    "controlling_prompt": "../Codex Prompt — MGMR_ Mechanics-Guided Multiscale Morphology Representation for Cross-Configuration CAI Assessment.md",
    "design": "docs/superpowers/specs/2026-08-22-mgmr-m0-design.md",
    "implementation_plan": "docs/superpowers/plans/2026-08-22-mgmr-m0.md",
    "repository_audit": "docs/MGMR_REPOSITORY_AUDIT.md",
    "reference_audit": "docs/MGMR_REFERENCE_METHOD_AUDIT.md",
    "impact_center_audit": "docs/MGMR_IMPACT_CENTER_AUDIT.md",
    "m0_protocol": "docs/MGMR_M0_PROTOCOL.md",
    "claim_matrix": "docs/MGMR_CLAIM_EVIDENCE_MATRIX.md",
    "p1_config": "paper_v3/configs/p1_full_field_oracle.yaml",
    "p1_predictions": "paper_v3/experiments/P1_full_field_oracle/predictions.csv",
    "p1_aggregate": "paper_v3/experiments/P1_full_field_oracle/aggregate_metrics.csv",
    "p3_config": "paper_v3/configs/p3_spatial_specificity.yaml",
    "p3_summary": "results/cpb_spatial/p3_spatial_specificity/summary.json",
    "p5_config": "paper_v3/configs/p5_sampling_retention.yaml",
    "p5_summary": "results/cpb_spatial/p5_sparse_scan/summary.json",
    "msss_config": "paper_v3/configs/msss.yaml",
    "msss_summary": "results/msss/s1_scale_discovery/summary.json",
    "paired_feature_bank": "results/aei_selective_invariance/a2_paired_features/paired_features.npz",
    "paired_feature_summary": "results/aei_selective_invariance/a2_paired_features/summary.json",
    "resnet_weights": "paper_v3/assets/resnet18-f37072fd.pth",
}
_OUTPUTS = {
    "feature_bank": Path("results/mgmr/feature_bank"),
    "formal": Path("results/mgmr/m0_component_gate"),
    "replay": Path("results/mgmr/replay/m0_component_gate"),
}


@dataclass(frozen=True, slots=True)
class SourceAuthority:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class MGMRProtocol:
    config_path: Path
    config_sha256: str
    sources: Mapping[str, SourceAuthority]
    specimen_count: int
    domain_order: tuple[str, ...]
    coarse_density: float
    spatial_layer: str
    spatial_shape: tuple[int, int, int]
    secondary_layer: str
    secondary_shape: tuple[int, int, int]
    wavelet: str
    wavelet_sensitivity: str
    wavelet_mode: str
    pca_dimensions: tuple[int, ...]
    pca_tie_tolerance: float
    ridge_alpha: float
    device: str
    batch_size: int
    specificity_seeds: tuple[int, ...]
    bootstrap_seed: int
    bootstrap_resamples: int
    bootstrap_quantiles: tuple[float, float]
    gate_required: tuple[str, ...]
    minimum_positive_domains: int
    baseline_mae: float
    positive_mae: float
    strong_mae: float
    output_paths: Mapping[str, Path]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise MGMRProtocolError(f"{label} must be a string-keyed mapping")
    return value


def _exact(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise MGMRProtocolError(f"{label} keys changed")


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MGMRProtocolError(f"{label} must be a sequence")
    output = tuple(value)
    if not output or any(type(item) is not str or not item for item in output):
        raise MGMRProtocolError(f"{label} contains invalid text")
    return output


def _integers(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MGMRProtocolError(f"{label} must be a sequence")
    output = tuple(value)
    if not output or any(type(item) is not int or item < 0 for item in output):
        raise MGMRProtocolError(f"{label} contains invalid integers")
    return output


def _numbers(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MGMRProtocolError(f"{label} must be a sequence")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or type(item) not in (int, float):
            raise MGMRProtocolError(f"{label} contains invalid numbers")
        number = float(item)
        if not math.isfinite(number):
            raise MGMRProtocolError(f"{label} contains non-finite numbers")
        output.append(number)
    if not output:
        raise MGMRProtocolError(f"{label} must not be empty")
    return tuple(output)


def _source_path(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or not relative.parts:
        raise MGMRProtocolError("source paths must be repo-relative")
    try:
        resolved = (root / relative).resolve(strict=True)
    except OSError as error:
        raise MGMRProtocolError(f"source is unavailable: {relative}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise MGMRProtocolError(f"source is not a regular file: {relative}")
    allowed = root.parent if ".." in relative.parts else root
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise MGMRProtocolError(f"source escapes its allowed root: {relative}") from error
    return resolved


def _load_sources(raw: object, root: Path) -> Mapping[str, SourceAuthority]:
    values = _mapping(raw, "sources")
    if set(values) != set(_SOURCE_PATHS):
        raise MGMRProtocolError("source registry keys changed")
    output: dict[str, SourceAuthority] = {}
    for name, expected_path in _SOURCE_PATHS.items():
        entry = _mapping(values[name], f"source {name}")
        _exact(entry, _SOURCE_KEYS, f"source {name}")
        path_text, digest = entry["path"], entry["sha256"]
        if path_text != expected_path or type(digest) is not str or len(digest) != 64:
            raise MGMRProtocolError(f"source authority changed: {name}")
        try:
            int(digest, 16)
        except ValueError as error:
            raise MGMRProtocolError(f"source SHA-256 is invalid: {name}") from error
        relative = Path(path_text)
        resolved = _source_path(root, relative)
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != digest:
            raise MGMRProtocolError(f"source SHA-256 mismatch: {name}")
        output[name] = SourceAuthority(name=name, path=relative, sha256=digest)
    return MappingProxyType(output)


def load_protocol(path: str | Path, *, project_root: str | Path) -> MGMRProtocol:
    """Load the exact registered M0 protocol and verify every source byte."""

    root = Path(project_root).resolve(strict=True)
    config_path = Path(path).resolve(strict=True)
    if not config_path.is_file() or config_path.is_symlink():
        raise MGMRProtocolError("config must be a regular file")
    payload = config_path.read_bytes()
    if len(payload) > 256_000:
        raise MGMRProtocolError("config is unexpectedly large")
    try:
        top = _mapping(yaml.load(payload, Loader=_UniqueKeyLoader), "config")
    except yaml.YAMLError as error:
        raise MGMRProtocolError("config YAML is invalid") from error
    if set(top) != _TOP_KEYS:
        raise MGMRProtocolError("top-level keys changed")
    if top["schema_version"] != 1 or top["scope"] != "mgmr_m0_component_gate":
        raise MGMRProtocolError("config identity changed")
    sources = _load_sources(top["sources"], root)

    cohort = _mapping(top["cohort"], "cohort")
    _exact(cohort, _COHORT_KEYS, "cohort")
    domains = _strings(cohort["domain_order"], "domain order")
    if (
        cohort["specimen_count"] != 276
        or domains != _DOMAINS
        or cohort["response"] != "damaged_to_intact_cai_strength_ratio"
        or cohort["response_unit"] != "1"
        or cohort["outer_split"] != "leave_one_dataset_out"
        or cohort["inner_split"] != "leave_one_source_dataset_out"
    ):
        raise MGMRProtocolError("cohort registry changed")

    representation = _mapping(top["representation"], "representation")
    _exact(representation, _REPRESENTATION_KEYS, "representation")
    spatial_shape = _integers(representation["spatial_shape"], "spatial shape")
    secondary_shape = _integers(representation["secondary_shape"], "secondary shape")
    if (
        representation["coarse_density"] != 0.25
        or representation["coarse_operator"]
        != "p5_endpoint_preserving_bilinear_reconstruction"
        or representation["spatial_layer"] != "layer3"
        or spatial_shape != (256, 14, 14)
        or representation["secondary_layer"] != "layer2"
        or secondary_shape != (128, 28, 28)
        or representation["wavelet"] != "db2"
        or representation["wavelet_sensitivity"] != "haar"
        or representation["wavelet_mode"] != "periodization"
        or representation["wavelet_level"] != 1
        or _strings(representation["directional_order"], "directional order")
        != ("cH", "cV", "cD")
    ):
        raise MGMRProtocolError("representation registry changed")

    estimator = _mapping(top["estimator"], "estimator")
    _exact(estimator, _ESTIMATOR_KEYS, "estimator")
    dimensions = _integers(estimator["pca_dimensions"], "PCA dimensions")
    if (
        estimator["metadata_features"] != "metadata13_once"
        or dimensions != (8, 16, 32)
        or estimator["pca_tie_tolerance"] != 1.0e-12
        or estimator["preprocessing"]
        != "fold_local_mean_imputation_standard_scaling"
        or estimator["regressor"] != "ridge"
        or estimator["ridge_alpha"] != 10.0
        or estimator["device"] != "cuda"
        or estimator["batch_size"] != 32
    ):
        raise MGMRProtocolError("estimator registry changed")

    residual = _mapping(top["residual"], "residual")
    _exact(residual, _RESIDUAL_KEYS, "residual")
    if residual != {
        "target": "strict_source_domain_oof_residual",
        "metadata_in_residual_branch": False,
        "correction_scale": 1.0,
    }:
        raise MGMRProtocolError("residual registry changed")

    specificity = _mapping(top["specificity"], "specificity")
    _exact(specificity, _SPECIFICITY_KEYS, "specificity")
    seeds = _integers(specificity["seeds"], "specificity seeds")
    if (
        specificity["control"] != "patch_shuffle_8x8"
        or seeds != (20260831, 20260901, 20260902)
        or _integers(specificity["grid"], "specificity grid") != (8, 8)
    ):
        raise MGMRProtocolError("specificity registry changed")

    bootstrap = _mapping(top["bootstrap"], "bootstrap")
    _exact(bootstrap, _BOOTSTRAP_KEYS, "bootstrap")
    quantiles = _numbers(bootstrap["quantiles"], "bootstrap quantiles")
    if (
        bootstrap["seed"] != 20260822
        or bootstrap["resamples"] != 100000
        or bootstrap["bit_generator"] != "PCG64"
        or bootstrap["unit"] != "held_out_domain"
        or quantiles != (0.025, 0.975)
    ):
        raise MGMRProtocolError("bootstrap registry changed")

    gate = _mapping(top["gate"], "gate")
    _exact(gate, _GATE_KEYS, "gate")
    required = _strings(gate["required"], "required gates")
    if (
        required != ("A", "B", "D")
        or gate["minimum_positive_domains"] != 4
        or gate["baseline_mae"] != 0.08963580465761432
        or gate["positive_mae"] != 0.08515
        or gate["strong_mae"] != 0.08247
        or gate["stop_on_no_go"] is not True
    ):
        raise MGMRProtocolError("gate registry changed")

    runtime = _mapping(top["runtime"], "runtime")
    _exact(runtime, _RUNTIME_KEYS, "runtime")
    if runtime != {
        "numpy": "2.5.1",
        "pywavelets": "1.8.0",
        "scikit_learn": "1.9.0",
        "torch": "2.12.1+cu130",
        "torchvision": "0.27.1+cu130",
    }:
        raise MGMRProtocolError("runtime registry changed")

    outputs = _mapping(top["outputs"], "outputs")
    _exact(outputs, _OUTPUT_KEYS, "outputs")
    if {name: Path(value) for name, value in outputs.items()} != _OUTPUTS:
        raise MGMRProtocolError("output registry changed")

    return MGMRProtocol(
        config_path=config_path,
        config_sha256=hashlib.sha256(payload).hexdigest(),
        sources=sources,
        specimen_count=276,
        domain_order=domains,
        coarse_density=0.25,
        spatial_layer="layer3",
        spatial_shape=(256, 14, 14),
        secondary_layer="layer2",
        secondary_shape=(128, 28, 28),
        wavelet="db2",
        wavelet_sensitivity="haar",
        wavelet_mode="periodization",
        pca_dimensions=dimensions,
        pca_tie_tolerance=1.0e-12,
        ridge_alpha=10.0,
        device="cuda",
        batch_size=32,
        specificity_seeds=seeds,
        bootstrap_seed=20260822,
        bootstrap_resamples=100000,
        bootstrap_quantiles=quantiles,
        gate_required=required,
        minimum_positive_domains=4,
        baseline_mae=0.08963580465761432,
        positive_mae=0.08515,
        strong_mae=0.08247,
        output_paths=MappingProxyType(dict(_OUTPUTS)),
    )


__all__ = [
    "MGMRProtocol",
    "MGMRProtocolError",
    "SourceAuthority",
    "load_protocol",
]
