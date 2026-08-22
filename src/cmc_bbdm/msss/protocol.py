"""Strict configuration authority for the MSSS experiments."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml


class MSSSProtocolError(ValueError):
    """Raised when the frozen MSSS configuration contract is violated."""


_CONFIG_RELATIVE = Path("paper_v3/configs/msss.yaml")
_SOURCE_NAMES = (
    "controlling_prompt",
    "existing_evidence",
    "s1_protocol",
    "s2_protocol",
    "reference_audit",
    "claim_matrix",
    "design",
    "implementation_plan",
    "p1_config",
    "p3_config",
    "p3_summary",
    "p5_config",
    "p5_summary",
    "paired_feature_summary",
    "paired_feature_bank",
    "multiview_e1_summary",
    "multiview_e3_summary",
    "multiview_stress_aggregate",
    "multiview_stress_groups",
    "p6_summary",
)
_DOMAIN_ORDER = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)
_SAMPLING = (1.0, 0.75, 0.625, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625)
_GAUSSIAN = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
_WAVELETS = ("db2", "haar", "db4")
_WAVELET_LEVELS = (0, 1, 2, 3)
_MARGINS = (0.025, 0.05, 0.075)
_SHA_CHARS = frozenset("0123456789abcdef")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise MSSSProtocolError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


@dataclass(frozen=True, slots=True)
class SourceAuthority:
    name: str
    path: Path
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class MSSSProtocol:
    config_path: Path
    config_sha256: str
    sources: tuple[SourceAuthority, ...]
    specimen_count: int
    domain_order: tuple[str, ...]
    ply_counts: tuple[int, ...]
    layup_families: tuple[str, ...]
    sampling_densities: tuple[float, ...]
    gaussian_sigmas: tuple[float, ...]
    wavelet_families: tuple[str, ...]
    wavelet_primary: str
    wavelet_levels: tuple[int, ...]
    wavelet_primary_mode: str
    wavelet_sensitivity_mode: str
    fourier_enabled: bool
    fourier_cutoffs: tuple[float, ...]
    pca_dimensions: tuple[int, ...]
    ridge_alpha: float
    device: str
    noninferiority_margins: tuple[float, ...]
    primary_margin: float
    specificity_seeds: tuple[int, ...]
    bootstrap_seed: int
    bootstrap_resamples: int
    output_paths: Mapping[str, Path]


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise MSSSProtocolError(f"{label} must be a string-key mapping")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise MSSSProtocolError(f"{label} keys changed")


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise MSSSProtocolError(f"{label} must be a sequence")
    return tuple(value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise MSSSProtocolError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise MSSSProtocolError(f"{label} must be finite")
    return output


def _integers(value: object, label: str) -> tuple[int, ...]:
    sequence = _sequence(value, label)
    if any(type(item) is not int or isinstance(item, bool) for item in sequence):
        raise MSSSProtocolError(f"{label} must contain integers")
    return tuple(int(item) for item in sequence)


def _numbers(value: object, label: str) -> tuple[float, ...]:
    return tuple(_number(item, label) for item in _sequence(value, label))


def _strings(value: object, label: str) -> tuple[str, ...]:
    sequence = _sequence(value, label)
    if any(type(item) is not str or not item for item in sequence):
        raise MSSSProtocolError(f"{label} must contain nonempty strings")
    return tuple(str(item) for item in sequence)


def _sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise MSSSProtocolError(f"{label} must be lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_authorities(
    value: object, *, project_root: Path
) -> tuple[SourceAuthority, ...]:
    sources = _mapping(value, "sources")
    _exact(sources, set(_SOURCE_NAMES), "source registry")
    workspace = project_root.parent.resolve(strict=True)
    result: list[SourceAuthority] = []
    for name in _SOURCE_NAMES:
        record = _mapping(sources[name], f"source {name}")
        _exact(record, {"path", "sha256"}, f"source {name}")
        relative = record["path"]
        if type(relative) is not str or not relative or Path(relative).is_absolute():
            raise MSSSProtocolError(f"source {name} path is unsafe")
        path = (project_root / relative).resolve(strict=True)
        if not path.is_file() or (path != workspace and workspace not in path.parents):
            raise MSSSProtocolError(f"source {name} escapes the workspace")
        expected = _sha(record["sha256"], f"source {name} hash")
        if _file_sha256(path) != expected:
            raise MSSSProtocolError(f"source {name} SHA-256 mismatch")
        result.append(SourceAuthority(name, path, relative, expected))
    return tuple(result)


def _output_paths(value: object, *, project_root: Path) -> Mapping[str, Path]:
    outputs = _mapping(value, "outputs")
    expected = {"s1_formal", "s1_smoke", "s1_replay", "s2_formal", "s2_replay"}
    _exact(outputs, expected, "outputs")
    registered_root = (project_root / "results/msss").resolve()
    resolved: dict[str, Path] = {}
    for name in sorted(expected):
        raw = outputs[name]
        if type(raw) is not str or Path(raw).is_absolute():
            raise MSSSProtocolError(f"output {name} path is unsafe")
        path = (project_root / raw).resolve()
        if registered_root not in path.parents:
            raise MSSSProtocolError(f"output {name} escapes results/msss")
        resolved[name] = path
    return MappingProxyType(resolved)


def load_protocol(
    config_path: str | Path, *, project_root: str | Path
) -> MSSSProtocol:
    """Load and validate the one registered MSSS configuration."""

    root = Path(project_root).resolve(strict=True)
    requested = Path(config_path).resolve(strict=True)
    registered = (root / _CONFIG_RELATIVE).resolve(strict=True)
    try:
        payload = yaml.load(requested.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except MSSSProtocolError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise MSSSProtocolError("MSSS config is not valid YAML") from error
    if requested != registered:
        raise MSSSProtocolError("MSSS requires the exact registered config path")
    top = _mapping(payload, "config")
    _exact(
        top,
        {
            "schema_version",
            "scope",
            "sources",
            "cohort",
            "axes",
            "estimator",
            "selection",
            "bootstrap",
            "transfer",
            "runtime",
            "outputs",
        },
        "config",
    )
    if top["schema_version"] != 1 or top["scope"] != "mechanically_sufficient_spatial_scale":
        raise MSSSProtocolError("MSSS config identity changed")

    cohort = _mapping(top["cohort"], "cohort")
    _exact(
        cohort,
        {
            "specimen_count",
            "independent_unit",
            "domain_order",
            "response",
            "response_unit",
            "ply_counts",
            "layup_families",
        },
        "cohort",
    )
    if (
        cohort["specimen_count"] != 276
        or cohort["independent_unit"] != "specimen"
        or cohort["response"] != "damaged_to_intact_cai_strength_ratio"
        or cohort["response_unit"] != "1"
        or _strings(cohort["domain_order"], "domain order") != _DOMAIN_ORDER
        or _integers(cohort["ply_counts"], "ply counts") != (8, 16, 24)
        or _strings(cohort["layup_families"], "layup families")
        != ("cross_ply", "quasi_isotropic")
    ):
        raise MSSSProtocolError("cohort contract changed")

    axes = _mapping(top["axes"], "axes")
    _exact(axes, {"sampling", "gaussian", "wavelet", "fourier"}, "axes")
    sampling = _mapping(axes["sampling"], "sampling")
    _exact(
        sampling,
        {
            "requested_densities",
            "interpolation",
            "rounding",
            "endpoints_included",
            "measured_points_restored_exactly",
            "field_size_mm",
        },
        "sampling",
    )
    if (
        _numbers(sampling["requested_densities"], "sampling densities") != _SAMPLING
        or sampling["interpolation"] != "bilinear"
        or sampling["rounding"] != "ieee754_ties_to_even"
        or sampling["endpoints_included"] is not True
        or sampling["measured_points_restored_exactly"] is not True
        or _numbers(sampling["field_size_mm"], "field size") != (75.0, 75.0)
    ):
        raise MSSSProtocolError("sampling registry changed")

    gaussian = _mapping(axes["gaussian"], "gaussian")
    _exact(gaussian, {"sigma_px", "boundary_mode", "sigma_mm"}, "gaussian")
    if (
        _numbers(gaussian["sigma_px"], "Gaussian sigmas") != _GAUSSIAN
        or gaussian["boundary_mode"] != "reflect"
        or gaussian["sigma_mm"] != "unavailable"
    ):
        raise MSSSProtocolError("Gaussian registry changed")

    wavelet = _mapping(axes["wavelet"], "wavelet")
    _exact(
        wavelet,
        {
            "families",
            "primary_family",
            "levels",
            "primary_mode",
            "sensitivity_mode",
            "boundary_mode",
        },
        "wavelet",
    )
    if (
        _strings(wavelet["families"], "wavelet families") != _WAVELETS
        or wavelet["primary_family"] != "db2"
        or _integers(wavelet["levels"], "wavelet levels") != _WAVELET_LEVELS
        or wavelet["primary_mode"] != "low_only"
        or wavelet["sensitivity_mode"] != "low_plus_boundary_details"
        or wavelet["boundary_mode"] != "periodization"
    ):
        raise MSSSProtocolError("wavelet registry changed")

    fourier = _mapping(axes["fourier"], "fourier")
    _exact(fourier, {"enabled", "normalized_cutoffs"}, "fourier")
    cutoffs = _numbers(fourier["normalized_cutoffs"], "Fourier cutoffs")
    if fourier["enabled"] is not False or cutoffs != (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10):
        raise MSSSProtocolError("Fourier registry changed")

    estimator = _mapping(top["estimator"], "estimator")
    _exact(
        estimator,
        {
            "encoder",
            "embedding_dimension",
            "pca_dimensions",
            "pca_tie_tolerance",
            "features",
            "preprocessing",
            "regressor",
            "ridge_alpha",
            "device",
        },
        "estimator",
    )
    dimensions = _integers(estimator["pca_dimensions"], "PCA dimensions")
    alpha = _number(estimator["ridge_alpha"], "Ridge alpha")
    if (
        estimator["encoder"] != "torchvision_resnet18_imagenet1k_v1_frozen"
        or estimator["embedding_dimension"] != 512
        or dimensions != (8, 16, 32)
        or _number(estimator["pca_tie_tolerance"], "PCA tolerance") != 1.0e-12
        or estimator["features"] != "metadata13_plus_pca"
        or estimator["preprocessing"] != "fold_local_mean_imputation_standard_scaling"
        or estimator["regressor"] != "ridge"
        or alpha != 10.0
        or estimator["device"] != "cuda"
    ):
        raise MSSSProtocolError("estimator registry changed")

    selection = _mapping(top["selection"], "selection")
    _exact(
        selection,
        {
            "relative_margins",
            "primary_margin",
            "minimum_plateau_nonfull_candidates",
            "stability_minimum_outer_folds",
            "stability_window_steps",
            "specificity_minimum_positive_domains",
            "specificity_control",
            "specificity_seeds",
            "s1_go_minimum_axes",
            "s1_strong_go_axes",
        },
        "selection",
    )
    margins = _numbers(selection["relative_margins"], "non-inferiority margins")
    seeds = _integers(selection["specificity_seeds"], "specificity seeds")
    if (
        margins != _MARGINS
        or _number(selection["primary_margin"], "primary margin") != 0.05
        or selection["minimum_plateau_nonfull_candidates"] != 2
        or selection["stability_minimum_outer_folds"] != 4
        or selection["stability_window_steps"] != 1
        or selection["specificity_minimum_positive_domains"] != 4
        or selection["specificity_control"] != "patch_shuffle_8x8"
        or seeds != (20260831, 20260901, 20260902)
        or selection["s1_go_minimum_axes"] != 2
        or selection["s1_strong_go_axes"] != 3
    ):
        raise MSSSProtocolError("selection registry changed")

    bootstrap = _mapping(top["bootstrap"], "bootstrap")
    _exact(
        bootstrap,
        {
            "seed",
            "resamples",
            "bit_generator",
            "sampling_unit",
            "synchronized",
            "ordinary_quantiles",
            "three_axis_familywise_quantiles",
        },
        "bootstrap",
    )
    if (
        bootstrap["seed"] != 20260822
        or bootstrap["resamples"] != 100000
        or bootstrap["bit_generator"] != "PCG64"
        or bootstrap["sampling_unit"] != "specimen_stratified_within_group"
        or bootstrap["synchronized"] is not True
        or _numbers(bootstrap["ordinary_quantiles"], "ordinary quantiles") != (0.025, 0.975)
        or _numbers(bootstrap["three_axis_familywise_quantiles"], "family quantiles")
        != (0.008333333333333333, 0.9916666666666667)
    ):
        raise MSSSProtocolError("bootstrap registry changed")

    # Transfer and runtime are still exact-key validated here; their values are
    # consumed by the conditional S2 implementation.
    transfer = _mapping(top["transfer"], "transfer")
    _exact(
        transfer,
        {
            "operational_axis",
            "comparators",
            "six_domain_support_minimum_nonworse",
            "structured_go_minimum_nonworse",
            "strong_ply_positive_minimum",
            "strong_layup_positive_minimum",
            "impact_shift_enabled",
        },
        "transfer",
    )
    runtime = _mapping(top["runtime"], "runtime")
    _exact(runtime, {"numpy", "scipy", "pywavelets", "scikit_learn", "torch", "torchvision"}, "runtime")

    return MSSSProtocol(
        config_path=requested,
        config_sha256=_file_sha256(requested),
        sources=_source_authorities(top["sources"], project_root=root),
        specimen_count=276,
        domain_order=_DOMAIN_ORDER,
        ply_counts=(8, 16, 24),
        layup_families=("cross_ply", "quasi_isotropic"),
        sampling_densities=_SAMPLING,
        gaussian_sigmas=_GAUSSIAN,
        wavelet_families=_WAVELETS,
        wavelet_primary="db2",
        wavelet_levels=_WAVELET_LEVELS,
        wavelet_primary_mode="low_only",
        wavelet_sensitivity_mode="low_plus_boundary_details",
        fourier_enabled=False,
        fourier_cutoffs=cutoffs,
        pca_dimensions=dimensions,
        ridge_alpha=alpha,
        device="cuda",
        noninferiority_margins=margins,
        primary_margin=0.05,
        specificity_seeds=seeds,
        bootstrap_seed=20260822,
        bootstrap_resamples=100000,
        output_paths=_output_paths(top["outputs"], project_root=root),
    )


__all__ = ["MSSSProtocol", "MSSSProtocolError", "SourceAuthority", "load_protocol"]
