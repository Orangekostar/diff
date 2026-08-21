"""Exact configuration authority for the D8 exploration study."""

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
import optuna
import PIL
import pywt
import scipy
import sklearn
import torch
import torchvision
import yaml

DOMAIN_ORDER = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)
P1_SCIENTIFIC_DIGEST = (
    "498c17a83c687d32eb504420ed5c8687be05f01f04506eec0d89a4887efabfd1"
)
P5_SCIENTIFIC_DIGEST = (
    "87b1da699b0ee59cf1723339a2150d673998b20004b6e990bb1a1a87d48f2257"
)
P6_SCIENTIFIC_DIGEST = (
    "fc8431597eadc9f1dff9b956d16810b6821888eef0b24c3db4df8a0dff50d505"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_CONFIG_BYTES = 256 * 1024
_MAX_SOURCE_BYTES = 512 * 1024 * 1024
_SOURCE_PATHS = {
    "prompt": (
        "docs/Codex Optimization Prompt — Result-Oriented Diffusion "
        "Marginalization for Cross-Domain CAI.md"
    ),
    "design": (
        "docs/superpowers/specs/2026-08-17-d8-morphology-preserving-"
        "diffusion-marginalization-design.md"
    ),
    "exploration_plan": "docs/D8_RESULT_ORIENTED_EXPLORATION_PLAN.md",
    "implementation_plan": (
        "docs/superpowers/plans/2026-08-17-d8-pilot-search.md"
    ),
    "p1_config": "paper_v3/configs/p1_full_field_oracle.yaml",
    "p1_manifest": (
        "paper_v3/experiments/P1_full_field_oracle/artifact_manifest.json"
    ),
    "p1_predictions": (
        "paper_v3/experiments/P1_full_field_oracle/predictions.csv"
    ),
    "p1_inner_selection": (
        "paper_v3/experiments/P1_full_field_oracle/inner_selection.csv"
    ),
    "p5_config": "paper_v3/configs/p5_sampling_retention.yaml",
    "p5_manifest": "results/cpb_spatial/p5_sparse_scan/artifact_manifest.json",
    "p6_config": "paper_v3/configs/p6_diffusion_reconstruction.yaml",
    "p6_manifest": (
        "results/cpb_spatial/p6_diffusion_reconstruction/artifact_manifest.json"
    ),
    "p6_uncertainty_source": (
        "results/cpb_spatial/p6_diffusion_reconstruction/"
        "uncertainty_source_data.npz"
    ),
    "resnet_weights": "paper_v3/assets/resnet18-f37072fd.pth",
    "runtime_requirements": "environment/requirements-runtime.txt",
    "d8_requirements": "environment/requirements-d8.txt",
    "dockerfile": "environment/Dockerfile",
    "d8_package_init": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/__init__.py"
    ),
    "d8_config_code": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/config.py"
    ),
    "d8_authority_code": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/authority.py"
    ),
    "d8_baseline_code": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/baseline.py"
    ),
    "d8_residuals_code": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/residuals.py"
    ),
    "d8_decomposition_code": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/decomposition.py"
    ),
    "d8_variants_code": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/variants.py"
    ),
    "d8_features_code": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/features.py"
    ),
    "d8_regression_code": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/regression.py"
    ),
    "d8_search_code": "src/cmc_bbdm/cpb_diffusion_marginalization/search.py",
    "d8_tracking_code": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/tracking.py"
    ),
    "d8_selection_code": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/selection.py"
    ),
    "d8_pilot_code": "src/cmc_bbdm/cpb_diffusion_marginalization/pilot.py",
    "d8_artifacts_code": (
        "src/cmc_bbdm/cpb_diffusion_marginalization/artifacts.py"
    ),
    "d8_cli_code": "scripts/run_d8_exploration.py",
    "d8_cli_wrapper": "scripts/run_d8_exploration.sh",
}


class D8ConfigError(ValueError):
    """Raised when a frozen D8 configuration authority drifts."""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class D8Config:
    schema_version: int
    scope: str
    seed: int
    outer_domains: tuple[str, ...]
    baseline_mae: float
    positive_mae: float
    strong_mae: float
    stretch_mae: float
    forced_trials: int
    optuna_trials: int
    rerank_seeds: tuple[int, ...]
    p6_draws: int
    sources: Mapping[str, SourceRecord]
    runtime: Mapping[str, str]
    search_space: Mapping[str, object]
    escalation: Mapping[str, object]
    output_dir: str
    final_output_dir: str
    config_sha256: str


class _UniqueLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise D8ConfigError(f"duplicate YAML key: {key}")
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
        raise D8ConfigError(f"{label} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise D8ConfigError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise D8ConfigError(f"{label} is not a bounded regular file")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise D8ConfigError(f"{label} changed while reading")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise D8ConfigError(f"{label} changed while reading")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise D8ConfigError(f"{label} must be a string mapping")
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
        raise D8ConfigError(f"{label} changed")


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise D8ConfigError(f"{label} must be a path string")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise D8ConfigError(f"{label} path is unsafe")
    return path.as_posix()


def _distribution(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise D8ConfigError(f"runtime package is missing: {name}") from error


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
        "optuna_module": optuna.__version__,
        "optuna_distribution": _distribution("optuna"),
        "pywavelets_module": pywt.__version__,
        "pywavelets_distribution": _distribution("PyWavelets"),
        "pyyaml_module": yaml.__version__,
        "pyyaml_distribution": _distribution("PyYAML"),
    }


def _expected_sections() -> dict[str, object]:
    return {
        "cohort": {
            "specimen_count": 276,
            "domain_order": list(DOMAIN_ORDER),
            "response": "damaged_to_intact_cai_strength_ratio",
            "response_unit": "1",
            "inferential_unit": "held_out_dataset",
            "outer_protocol": "leave_one_dataset_out",
            "inner_protocol": "leave_one_dataset_out_within_outer_training_domains",
            "surface_features_allowed": False,
        },
        "baseline": {
            "name": "I_frozen",
            "equal_domain_mae": 0.08963580465761432,
            "domain_mae": {
                "74t7kcdgkr": 0.052003607090763716,
                "cgtnjyggtm": 0.12486849958988917,
                "w68dtmpfyf": 0.09660573834513045,
                "xcmzfsbd9t": 0.07511512755876239,
                "yfxyg8jm46": 0.12743074387350148,
                "ykhs7s2dck": 0.06179111148763877,
            },
            "pca_dimensions_by_outer_domain": {
                "74t7kcdgkr": 8,
                "cgtnjyggtm": 32,
                "w68dtmpfyf": 8,
                "xcmzfsbd9t": 8,
                "yfxyg8jm46": 8,
                "ykhs7s2dck": 8,
            },
            "reproduction_tolerance": 1.0e-12,
            "positive_mae": 0.085154,
            "strong_mae": 0.082465,
            "stretch_mae": 0.080,
        },
        "posterior": {
            "authority": "P6_cross_fitted_diffusion_draws",
            "p6_draws": 8,
            "residual_definition": "diffusion_draw_minus_measured_internal_field",
            "mean_tolerance": 1.0e-6,
            "variance_tolerance": 1.0e-6,
            "dtype": "float32",
            "shape": [3, 64, 64],
        },
        "decomposition": {
            "families": ["gaussian", "fourier", "wavelet"],
            "bands": ["low", "mid", "mid+high", "high"],
            "gaussian_sigma_pixels": {
                "minimum": 0.5,
                "maximum": 8.0,
                "distribution": "log_uniform",
            },
            "fourier_cutoff_fraction": {
                "minimum": 0.04,
                "maximum": 0.50,
                "distribution": "uniform",
            },
            "fourier_transition_fraction": {
                "minimum": 0.01,
                "maximum": 0.10,
                "distribution": "uniform",
            },
            "wavelets": ["haar", "db2", "db4", "sym4"],
            "wavelet_levels": [1, 2, 3],
            "alpha": {
                "minimum": -0.5,
                "maximum": 1.0,
                "distribution": "uniform",
            },
        },
        "morphology_gate": {
            "area_relative_deviation": [0.025, 0.05, 0.075, 0.10],
            "width_relative_deviation": [0.025, 0.05, 0.075, 0.10],
            "height_relative_deviation": [0.025, 0.05, 0.075, 0.10],
            "centroid_shift_mm": [0.5, 1.0, 2.0],
            "low_frequency_correlation_minimum": [0.95, 0.97, 0.98, 0.99],
            "low_frequency_sigma_pixels": 2.0,
            "radial_spearman_minimum": [0.90, 0.95, 0.98],
            "radial_profile_bins": 16,
            "maximum_proposals": 32,
            "minimum_overall_acceptance": 0.80,
            "minimum_domain_acceptance": 0.60,
            "fallback": "raw_measured_internal_field",
        },
        "controls": {
            "registered": ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"],
            "B0": "I_frozen_raw_field",
            "B1": "morphology_component_only",
            "B2": "matched_gaussian_noise",
            "B3": "phase_randomized_residual",
            "B4": "empirical_cross_fitted_residual",
            "B5": "diffusion_residual_augmentation",
            "B6": "diffusion_plus_consistency",
            "B7": "diffusion_plus_test_time_marginalization",
            "B8": "full_proposed_pipeline",
        },
        "features": {
            "encoder": "frozen_resnet18_imagenet1k_v1",
            "layers": ["global", "layer3", "multi_layer"],
            "marginalization_stage": ["feature", "prediction"],
            "aggregation": ["mean", "median", "trimmed", "mean_variance"],
            "prediction_aggregation": [
                "mean",
                "median",
                "trimmed",
                "morphology_weighted",
            ],
            "K_train": [1, 2, 4, 8],
            "K_test": [1, 2, 4, 8, 16],
            "morphology_weight_beta": {
                "minimum": 0.1,
                "maximum": 100.0,
                "distribution": "log_uniform",
            },
            "consistency": [
                "none",
                "prediction_variance",
                "feature_variance",
                "pairwise_ranking",
            ],
            "consistency_weight": {
                "minimum": 0.0,
                "maximum": 1.0,
                "distribution": "uniform",
            },
            "replicated_variant_specimen_weight": 1.0,
        },
        "regressors": {
            "registered": [
                "ridge",
                "elastic_net",
                "pls",
                "huber",
                "kernel_ridge",
                "svr",
                "hist_gradient_boosting",
                "shallow_mlp",
            ],
            "pca_dimensions": [4, 8, 16, 32, 64],
            "standardize_within_inner_fit": True,
        },
        "search": {
            "forced_trials": 12,
            "optuna_trials": 60,
            "sampler": "TPESampler",
            "study_storage": "sqlite",
            "objective": "mean_mae_plus_0.25_worst_plus_0.10_domain_sd",
            "mean_weight": 1.0,
            "worst_weight": 0.25,
            "domain_sd_weight": 0.10,
            "rerank_top": 12,
            "finalist_top": 4,
            "rerank_seeds": [20260820, 20260821, 20260822],
            "trial_failure_visibility": "required",
        },
        "ensemble": {
            "enabled": True,
            "weight_constraint": "nonnegative_simplex",
            "fit_rows": "inner_oof_only",
            "minimum_objective_gain": 0.0001,
            "fallback": "best_single_candidate",
        },
        "escalation": {
            "p6_candidate_minimum_inner_mae_improvement": 0.01,
            "p6_candidate_minimum_inner_domains": 3,
            "p6_candidate_minimum_outer_studies": 3,
            "low_band_energy_fraction": 0.50,
            "low_acceptance_threshold": 0.50,
            "low_acceptance_alpha": 0.10,
            "mismatch_minimum_outer_studies": 3,
            "pilot_freeze_minimum_objective_gain": 0.0001,
            "pilot_freeze_minimum_diffusion_weight": 0.05,
            "pilot_freeze_minimum_outer_studies": 3,
            "decision_priority": [
                "TRAIN_RESIDUAL_DIFFUSION",
                "FREEZE_PILOT_FOR_OUTER_EVALUATION",
                "CLOSE_DIFFUSION_SPECIFIC_ROUTE",
            ],
        },
        "evaluation": {
            "pilot_outer_evaluation_allowed": False,
            "bootstrap_seed": 20260820,
            "bootstrap_resamples": 100000,
            "ordinary_quantiles": [0.025, 0.975],
            "simultaneous_quantiles": [
                0.008333333333333333,
                0.9916666666666667,
            ],
            "primary_minimum_improved_domains": 4,
            "strong_minimum_improved_domains": 5,
            "diffusion_specific_minimum_domains_vs_best_nondiffusion": 4,
        },
        "outputs": {
            "search_dir": "results/d8_search",
            "final_dir": "results/d8_final",
            "required_search_files": [
                "trial_index.csv",
                "study.db",
                "residual_bank_manifest.json",
                "search_summary.csv",
                "selected_configs.json",
                "escalation_evidence.json",
                "pilot_report.md",
                "artifact_manifest.json",
                "CHECKSUMS.sha256",
            ],
            "required_final_files": [
                "aggregate_metrics.csv",
                "domain_metrics.csv",
                "bootstrap.csv",
                "selected_configs.csv",
                "ablation.csv",
                "morphology_audit.csv",
                "search_summary.csv",
                "REPORT.md",
            ],
        },
    }


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _source(
    name: str, value: object, root: Path
) -> tuple[SourceRecord, bytes]:
    record = _mapping(value, f"sources.{name}")
    if set(record) != {"path", "sha256"}:
        raise D8ConfigError(f"sources.{name} keys changed")
    path = _safe_relative(record["path"], f"sources.{name}")
    if path != _SOURCE_PATHS[name]:
        raise D8ConfigError(f"source path changed: {name}")
    digest = record["sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise D8ConfigError(f"source hash is invalid: {name}")
    payload = _read_regular(root / path, f"source {name}", maximum=_MAX_SOURCE_BYTES)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise D8ConfigError(f"source hash mismatch: {name}")
    return SourceRecord(path, digest), payload


def _json_mapping(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise D8ConfigError(f"{label} is invalid JSON") from error
    return _mapping(value, label)


def _validate_upstream(payloads: Mapping[str, bytes]) -> None:
    p1 = _json_mapping(payloads["p1_manifest"], "P1 manifest")
    if (
        p1.get("scope") != "cpb_v3_p1_full_field_oracle"
        or p1.get("scientific_digest") != P1_SCIENTIFIC_DIGEST
        or p1.get("package_digest")
        != "8c13793c064d78cb17b2201b1856db490d12ec564db810fb1a3968cd3199d297"
    ):
        raise D8ConfigError("P1 authorization changed")
    p5 = _json_mapping(payloads["p5_manifest"], "P5 manifest")
    if (
        p5.get("scope") != "cpb_sparse_scan_p5_retention"
        or p5.get("scientific_digest") != P5_SCIENTIFIC_DIGEST
        or p5.get("output_tree_sha256")
        != "a2c54b7fdc53252767c7788365d254bbcdbe36571d2e553ce56f406dc039c80d"
    ):
        raise D8ConfigError("P5 authorization changed")
    p6 = _json_mapping(payloads["p6_manifest"], "P6 manifest")
    if (
        p6.get("scope") != "cpb_spatial_p6_diffusion_reconstruction"
        or p6.get("mode") != "full"
        or p6.get("test_only") is not False
        or p6.get("specimen_count") != 276
        or p6.get("scientific_digest") != P6_SCIENTIFIC_DIGEST
        or p6.get("output_tree_sha256")
        != "433699f015914c166696d2c19feaf8fe452482ff61200d663afb76fae6f493e9"
    ):
        raise D8ConfigError("P6 authorization changed")


def load_d8_config(config_path: str | Path, *, project_root: str | Path) -> D8Config:
    """Load D8 only after every frozen exploration authority validates."""

    root = Path(project_root).resolve(strict=True)
    payload = _read_regular(Path(config_path), "D8 config", maximum=_MAX_CONFIG_BYTES)
    try:
        loaded = yaml.load(payload.decode("utf-8"), Loader=_UniqueLoader)
    except D8ConfigError:
        raise
    except (UnicodeError, yaml.YAMLError) as error:
        raise D8ConfigError("D8 config is invalid YAML") from error
    item = _mapping(loaded, "D8 config")
    expected_sections = _expected_sections()
    expected_root = {
        "schema_version",
        "scope",
        "seed",
        "output_dir",
        "runtime",
        "sources",
        *expected_sections,
    }
    if set(item) != expected_root:
        raise D8ConfigError("D8 config keys changed")
    _expect(item["schema_version"], 1, "schema_version")
    _expect(
        item["scope"],
        "cpb_d8_morphology_preserving_marginalization",
        "scope",
    )
    _expect(item["seed"], 20260820, "seed")
    _expect(item["output_dir"], "results/d8_search", "output_dir")
    runtime = dict(_mapping(item["runtime"], "runtime"))
    _expect(runtime, _runtime_values(), "runtime")
    source_values = _mapping(item["sources"], "sources")
    if set(source_values) != set(_SOURCE_PATHS):
        raise D8ConfigError("sources keys changed")
    sources: dict[str, SourceRecord] = {}
    source_payloads: dict[str, bytes] = {}
    for name in _SOURCE_PATHS:
        sources[name], source_payloads[name] = _source(
            name, source_values[name], root
        )
    for section, expected in expected_sections.items():
        _expect(dict(_mapping(item[section], section)), expected, section)
    _validate_upstream(source_payloads)
    baseline = expected_sections["baseline"]
    posterior = expected_sections["posterior"]
    search = expected_sections["search"]
    escalation = expected_sections["escalation"]
    outputs = expected_sections["outputs"]
    assert isinstance(baseline, dict)
    assert isinstance(posterior, dict)
    assert isinstance(search, dict)
    assert isinstance(escalation, dict)
    assert isinstance(outputs, dict)
    search_space = {
        name: expected_sections[name]
        for name in (
            "decomposition",
            "morphology_gate",
            "controls",
            "features",
            "regressors",
        )
    }
    return D8Config(
        schema_version=1,
        scope="cpb_d8_morphology_preserving_marginalization",
        seed=20260820,
        outer_domains=DOMAIN_ORDER,
        baseline_mae=float(baseline["equal_domain_mae"]),
        positive_mae=float(baseline["positive_mae"]),
        strong_mae=float(baseline["strong_mae"]),
        stretch_mae=float(baseline["stretch_mae"]),
        forced_trials=int(search["forced_trials"]),
        optuna_trials=int(search["optuna_trials"]),
        rerank_seeds=tuple(search["rerank_seeds"]),
        p6_draws=int(posterior["p6_draws"]),
        sources=MappingProxyType(sources),
        runtime=MappingProxyType(runtime),
        search_space=_freeze(search_space),
        escalation=_freeze(escalation),
        output_dir=str(outputs["search_dir"]),
        final_output_dir=str(outputs["final_dir"]),
        config_sha256=hashlib.sha256(payload).hexdigest(),
    )


__all__ = [
    "DOMAIN_ORDER",
    "P1_SCIENTIFIC_DIGEST",
    "P5_SCIENTIFIC_DIGEST",
    "P6_SCIENTIFIC_DIGEST",
    "D8Config",
    "D8ConfigError",
    "SourceRecord",
    "load_d8_config",
]
