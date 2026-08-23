"""Strict configuration authority for the preregistered MVA A0-A3 run."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


class MVAConfigError(ValueError):
    """Raised when the MVA protocol or one of its bound sources drifts."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    output: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise MVAConfigError(f"duplicate config key: {key}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)

_DOMAIN_ORDER = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)
_METHODS = (
    "uniform",
    "random",
    "appearance_oracle",
    "reconstruction_oracle",
    "mechanical_oracle",
)


@dataclass(frozen=True, slots=True)
class SourceBinding:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class MVAConfig:
    schema_version: int
    scope: str
    seed: int
    sources: Mapping[str, SourceBinding]
    domain_order: tuple[str, ...]
    specimen_count: int
    initial_budgets: tuple[float, ...]
    checkpoints: tuple[float, ...]
    auebc_range: tuple[float, float]
    cell_shape: tuple[int, int]
    pca_dimensions: tuple[int, ...]
    ridge_alpha: float
    full_mae: float
    baseline_tolerance: float
    random_seeds: tuple[int, ...]
    bootstrap_seed: int
    bootstrap_resamples: int
    low_budget: float
    h1_relative_improvement: float
    h1_minimum_domains: int
    h4_relative_auebc: float
    h4_b5_saving: float
    methods: tuple[str, ...]
    output_dir: Path


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise MVAConfigError(f"{label} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise MVAConfigError(f"{label} keys changed")


def _expect(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise MVAConfigError(f"{label} changed")


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MVAConfigError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise MVAConfigError(f"{label} must be finite")
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise MVAConfigError(f"bound source is unavailable: {path}") from error
    return digest.hexdigest()


def _sources(value: object, root: Path) -> Mapping[str, SourceBinding]:
    source_map = _mapping(value, "sources")
    expected = {
        "mgmr_config",
        "p1_predictions",
        "p5_config",
        "p5_sampling",
        "paired_feature_bank",
        "resnet_weights",
        "protocol",
    }
    _exact_keys(source_map, expected, "sources")
    output: dict[str, SourceBinding] = {}
    for name in sorted(expected):
        item = _mapping(source_map[name], f"source {name}")
        _exact_keys(item, {"path", "sha256"}, f"source {name}")
        raw_path = item["path"]
        digest = item["sha256"]
        if type(raw_path) is not str or not raw_path:
            raise MVAConfigError(f"source {name} path is invalid")
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise MVAConfigError(f"source {name} path must be repository relative")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise MVAConfigError(f"source {name} hash is invalid")
        if _sha256_file(root / path) != digest:
            raise MVAConfigError(f"source {name} hash changed")
        output[name] = SourceBinding(path=path, sha256=digest)
    return MappingProxyType(output)


def load_mva_config(config_path: str | Path, *, project_root: str | Path) -> MVAConfig:
    """Load a config only when every A0-A3 constant and source remains frozen."""

    root = Path(project_root).resolve(strict=True)
    try:
        payload = Path(config_path).read_text(encoding="utf-8")
        loaded = yaml.load(payload, Loader=_UniqueLoader)
    except MVAConfigError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise MVAConfigError("MVA config cannot be read") from error
    item = _mapping(loaded, "MVA config")
    _exact_keys(
        item,
        {
            "schema_version",
            "scope",
            "seed",
            "sources",
            "cohort",
            "acquisition",
            "estimator",
            "controls",
            "bootstrap",
            "gate",
            "outputs",
        },
        "MVA config",
    )
    _expect(item["schema_version"], 1, "schema_version")
    _expect(item["scope"], "mva_a0_a3_oracle_headroom", "scope")
    _expect(item["seed"], 20260823, "seed")
    sources = _sources(item["sources"], root)

    cohort = _mapping(item["cohort"], "cohort")
    _expect(
        dict(cohort),
        {
            "specimen_count": 276,
            "domain_order": list(_DOMAIN_ORDER),
            "response": "damaged_to_intact_cai_strength_ratio",
            "outer_split": "leave_one_dataset_out",
            "inner_split": "leave_one_source_dataset_out",
        },
        "cohort",
    )

    acquisition = _mapping(item["acquisition"], "acquisition")
    expected_acquisition = {
        "cell_shape": [8, 8],
        "initial_budgets": [0.015625, 0.03125, 0.0625],
        "checkpoints": [
            0.03125,
            0.0625,
            0.09375,
            0.125,
            0.1875,
            0.25,
            0.5,
            1.0,
        ],
        "auebc_range": [0.0625, 0.25],
        "p5_density": 0.25,
        "primary_interpolation": "bilinear",
        "sensitivity_interpolations": ["nearest", "bicubic"],
        "budget_unit": "unique_native_raster_locations",
    }
    _expect(dict(acquisition), expected_acquisition, "acquisition")

    estimator = _mapping(item["estimator"], "estimator")
    expected_estimator = {
        "metadata": "metadata13",
        "encoder": "frozen_imagenet_resnet18_final_512d",
        "pca_dimensions": [8, 16, 32],
        "preprocessing": "fold_local_mean_imputation_standard_scaling",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "pca_tie_tolerance": 1.0e-12,
        "full_mae": 0.08963580465761432,
        "baseline_tolerance": 1.0e-12,
        "protocols": ["P-A", "P-B"],
        "primary_protocol": "P-B",
    }
    _expect(dict(estimator), expected_estimator, "estimator")

    controls = _mapping(item["controls"], "controls")
    expected_controls = {
        "methods": list(_METHODS),
        "random_seed_start": 2026082300,
        "random_seed_count": 100,
        "random_bit_generator": "PCG64",
        "appearance_name": "full_image_appearance_intensity",
        "center_first": "excluded_no_impact_center_authority",
    }
    _expect(dict(controls), expected_controls, "controls")

    bootstrap = _mapping(item["bootstrap"], "bootstrap")
    expected_bootstrap = {
        "seed": 20260823,
        "resamples": 100000,
        "bit_generator": "PCG64",
        "unit": "held_out_domain",
        "quantiles": [0.025, 0.975],
    }
    _expect(dict(bootstrap), expected_bootstrap, "bootstrap")

    gate = _mapping(item["gate"], "gate")
    expected_gate = {
        "low_budget": 0.125,
        "h1_relative_improvement": 0.05,
        "minimum_improved_domains": 4,
        "h2_bootstrap_lower_positive": True,
        "h3_bootstrap_lower_positive": True,
        "h4_relative_auebc": 0.10,
        "h4_b5_saving": 0.25,
        "all_required": True,
        "statuses": ["MVA_ORACLE_GO", "MVA_ORACLE_NO_GO"],
        "stop_after_decision": True,
    }
    _expect(dict(gate), expected_gate, "gate")

    outputs = _mapping(item["outputs"], "outputs")
    expected_outputs = {
        "root": "results/mva",
        "a0": "a0_acquisition_audit",
        "a1": "a1_simulator",
        "a2": "a2_oracle_value",
        "replay": "replay",
    }
    _expect(dict(outputs), expected_outputs, "outputs")

    return MVAConfig(
        schema_version=1,
        scope="mva_a0_a3_oracle_headroom",
        seed=20260823,
        sources=sources,
        domain_order=_DOMAIN_ORDER,
        specimen_count=276,
        initial_budgets=tuple(
            _float(value, "initial budget") for value in acquisition["initial_budgets"]
        ),
        checkpoints=tuple(
            _float(value, "checkpoint") for value in acquisition["checkpoints"]
        ),
        auebc_range=tuple(acquisition["auebc_range"]),
        cell_shape=tuple(acquisition["cell_shape"]),
        pca_dimensions=tuple(estimator["pca_dimensions"]),
        ridge_alpha=_float(estimator["ridge_alpha"], "ridge_alpha"),
        full_mae=_float(estimator["full_mae"], "full_mae"),
        baseline_tolerance=_float(
            estimator["baseline_tolerance"], "baseline_tolerance"
        ),
        random_seeds=tuple(
            range(
                controls["random_seed_start"],
                controls["random_seed_start"] + controls["random_seed_count"],
            )
        ),
        bootstrap_seed=bootstrap["seed"],
        bootstrap_resamples=bootstrap["resamples"],
        low_budget=_float(gate["low_budget"], "low_budget"),
        h1_relative_improvement=_float(
            gate["h1_relative_improvement"], "h1_relative_improvement"
        ),
        h1_minimum_domains=gate["minimum_improved_domains"],
        h4_relative_auebc=_float(gate["h4_relative_auebc"], "h4_relative_auebc"),
        h4_b5_saving=_float(gate["h4_b5_saving"], "h4_b5_saving"),
        methods=_METHODS,
        output_dir=Path(outputs["root"]),
    )


__all__ = ["MVAConfig", "MVAConfigError", "SourceBinding", "load_mva_config"]
