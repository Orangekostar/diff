"""Fail-closed configuration authority for MVA A4."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .config import SourceBinding

_DOMAIN_ORDER = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)
_METHODS = (
    "global_appearance_mask",
    "global_reconstruction_mask",
    "global_mechanical_mask",
)
_A4_STATUSES = ("MVA_A4_GLOBAL_GO", "MVA_A4_GLOBAL_NO_GO")
_A5_STATUSES = ("MVA_A5_AUTHORIZED", "MVA_A5_NOT_AUTHORIZED")


class A4ConfigError(ValueError):
    """Raised when the A4 protocol or a bound authority drifts."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    output: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise A4ConfigError(f"duplicate config key: {key}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


@dataclass(frozen=True, slots=True)
class A4Config:
    schema_version: int
    scope: str
    seed: int
    sources: Mapping[str, SourceBinding]
    a3_status: str
    domain_order: tuple[str, ...]
    specimen_count: int
    cell_shape: tuple[int, int]
    candidate_from_level: int
    candidate_to_level: int
    checkpoints: tuple[float, ...]
    auebc_range: tuple[float, float]
    methods: tuple[str, ...]
    rank_aggregation: str
    pca_dimensions: tuple[int, ...]
    ridge_alpha: float
    primary_prediction_protocol: str
    bootstrap_seed: int
    bootstrap_resamples: int
    minimum_improved_domains: int
    adaptive_gap_threshold: float
    required_a4_comparisons: tuple[str, ...]
    a4_statuses: tuple[str, ...]
    a5_statuses: tuple[str, ...]
    output_dir: Path
    replay_dir: Path


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise A4ConfigError(f"{label} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise A4ConfigError(f"{label} keys changed")


def _expect(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise A4ConfigError(f"{label} changed")


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise A4ConfigError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise A4ConfigError(f"{label} must be finite")
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise A4ConfigError(f"bound source is unavailable: {path}") from error
    return digest.hexdigest()


def _load_sources(value: object, root: Path) -> Mapping[str, SourceBinding]:
    source_map = _mapping(value, "sources")
    expected = {
        "a0_a3_config",
        "a1_summary",
        "a2_checksums",
        "a2_manifest",
        "a2_summary",
        "a4_design",
        "a4_protocol",
    }
    _exact_keys(source_map, expected, "sources")
    output: dict[str, SourceBinding] = {}
    for name in sorted(expected):
        item = _mapping(source_map[name], f"source {name}")
        _exact_keys(item, {"path", "sha256"}, f"source {name}")
        raw_path = item["path"]
        digest = item["sha256"]
        if type(raw_path) is not str or not raw_path:
            raise A4ConfigError(f"source {name} path is invalid")
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise A4ConfigError(f"source {name} path must be repository relative")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise A4ConfigError(f"source {name} hash is invalid")
        if _sha256_file(root / path) != digest:
            raise A4ConfigError(f"source {name} hash changed")
        output[name] = SourceBinding(path=path, sha256=digest)
    return MappingProxyType(output)


def _validate_a3_authorization(root: Path, source: SourceBinding) -> None:
    try:
        summary = json.loads((root / source.path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise A4ConfigError("A3 summary cannot be read") from error
    gate = summary.get("gate")
    if (
        summary.get("status") != "MVA_ORACLE_GO"
        or not isinstance(gate, dict)
        or gate.get("status") != "MVA_ORACLE_GO"
        or any(gate.get(f"h{index}_pass") is not True for index in range(1, 5))
    ):
        raise A4ConfigError("A3 does not authorize A4")


def load_a4_config(
    config_path: str | Path, *, project_root: str | Path
) -> A4Config:
    """Load A4 only when its complete frozen authority remains unchanged."""

    root = Path(project_root).resolve(strict=True)
    try:
        payload = Path(config_path).read_text(encoding="utf-8")
        loaded = yaml.load(payload, Loader=_UniqueLoader)
    except A4ConfigError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise A4ConfigError("A4 config cannot be read") from error
    item = _mapping(loaded, "A4 config")
    _exact_keys(
        item,
        {
            "schema_version",
            "scope",
            "seed",
            "sources",
            "authorization",
            "cohort",
            "acquisition",
            "ranking",
            "estimator",
            "evaluation",
            "bootstrap",
            "gate",
            "outputs",
        },
        "A4 config",
    )
    _expect(item["schema_version"], 1, "schema_version")
    _expect(item["scope"], "mva_a4_global_task_mask", "scope")
    _expect(item["seed"], 20260823, "seed")
    sources = _load_sources(item["sources"], root)

    authorization = _mapping(item["authorization"], "authorization")
    _expect(
        dict(authorization),
        {
            "a3_status": "MVA_ORACLE_GO",
            "source": "a2_summary",
            "stop_scope": "a4_decisions_only",
            "forbidden_stages": ["A5_training", "A6", "A7"],
        },
        "authorization",
    )
    _validate_a3_authorization(root, sources["a2_summary"])

    cohort = _mapping(item["cohort"], "cohort")
    _expect(
        dict(cohort),
        {
            "specimen_count": 276,
            "domain_order": list(_DOMAIN_ORDER),
            "response": "damaged_to_intact_cai_strength_ratio",
            "outer_split": "leave_one_dataset_out",
            "source_label_split": "leave_outer_and_query_domain_out",
        },
        "cohort",
    )

    acquisition = _mapping(item["acquisition"], "acquisition")
    expected_acquisition = {
        "cell_shape": [8, 8],
        "initial_budget_authority": "a1_source_selected_per_outer_domain",
        "candidate_from_level": 0,
        "candidate_to_level": 1,
        "checkpoints": [0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25],
        "auebc_range": [0.0625, 0.25],
        "interpolation": "bilinear",
        "measured_values": "restored_exactly",
        "budget_unit": "unique_native_raster_locations",
    }
    _expect(dict(acquisition), expected_acquisition, "acquisition")

    ranking = _mapping(item["ranking"], "ranking")
    expected_ranking = {
        "methods": list(_METHODS),
        "aggregation": "equal_domain_mean_normalized_rank",
        "specimen_rank_range": [0.0, 1.0],
        "value_order": "descending",
        "tie_break": "lower_cell_index",
        "diagnostic_scores": [
            "mean_raw_value",
            "mean_value_per_new_measurement",
        ],
        "target_information": "forbidden",
    }
    _expect(dict(ranking), expected_ranking, "ranking")

    estimator = _mapping(item["estimator"], "estimator")
    expected_estimator = {
        "metadata": "metadata13",
        "encoder": "frozen_imagenet_resnet18_final_512d",
        "pca_dimensions": [8, 16, 32],
        "preprocessing": "fold_local_mean_imputation_standard_scaling",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "pca_tie_tolerance": 1.0e-12,
        "label_protocol": "strict_source_query_oof_P-A",
        "primary_prediction_protocol": "P-B",
        "p_b_training_states": "source_uniform_same_checkpoint",
        "full_mae": 0.08963580465761432,
    }
    _expect(dict(estimator), expected_estimator, "estimator")

    evaluation = _mapping(item["evaluation"], "evaluation")
    expected_evaluation = {
        "reference_methods": ["uniform", "random_median", "mechanical_oracle"],
        "prediction_metric": "equal_domain_mae",
        "budget_metrics": ["auebc", "B_2.5%", "B_5%", "B_7.5%"],
        "image_metrics": ["normalized_rgb_mse", "ssim"],
        "random_seed_count": 100,
        "random_source": "a2_state_metrics",
    }
    _expect(dict(evaluation), expected_evaluation, "evaluation")

    bootstrap = _mapping(item["bootstrap"], "bootstrap")
    expected_bootstrap = {
        "seed": 20260823,
        "resamples": 100000,
        "bit_generator": "PCG64",
        "unit": "held_out_domain",
        "quantiles": [0.025, 0.975],
        "synchronized_across_effects": True,
    }
    _expect(dict(bootstrap), expected_bootstrap, "bootstrap")

    gate = _mapping(item["gate"], "gate")
    expected_gate = {
        "required_a4_comparisons": [
            "uniform",
            "global_reconstruction_mask",
            "global_appearance_mask",
        ],
        "require_positive_point_effect": True,
        "require_positive_bootstrap_lower": True,
        "minimum_improved_domains": 4,
        "adaptive_gap_relative": 0.03,
        "adaptive_reference": "mechanical_oracle",
        "a4_all_required": True,
        "a4_statuses": list(_A4_STATUSES),
        "a5_all_required": True,
        "a5_statuses": list(_A5_STATUSES),
        "thresholds_frozen_before_results": True,
    }
    _expect(dict(gate), expected_gate, "gate")

    outputs = _mapping(item["outputs"], "outputs")
    expected_outputs = {
        "work": "results/mva/.work/a4",
        "formal": "results/mva/a4_global_task_mask",
        "replay": "results/mva/replay/a4_global_task_mask",
    }
    _expect(dict(outputs), expected_outputs, "outputs")

    return A4Config(
        schema_version=1,
        scope="mva_a4_global_task_mask",
        seed=20260823,
        sources=sources,
        a3_status="MVA_ORACLE_GO",
        domain_order=_DOMAIN_ORDER,
        specimen_count=276,
        cell_shape=(8, 8),
        candidate_from_level=0,
        candidate_to_level=1,
        checkpoints=tuple(_finite(value, "checkpoint") for value in acquisition["checkpoints"]),
        auebc_range=(
            _finite(acquisition["auebc_range"][0], "AUEBC start"),
            _finite(acquisition["auebc_range"][1], "AUEBC stop"),
        ),
        methods=_METHODS,
        rank_aggregation="equal_domain_mean_normalized_rank",
        pca_dimensions=tuple(estimator["pca_dimensions"]),
        ridge_alpha=_finite(estimator["ridge_alpha"], "ridge_alpha"),
        primary_prediction_protocol="P-B",
        bootstrap_seed=bootstrap["seed"],
        bootstrap_resamples=bootstrap["resamples"],
        minimum_improved_domains=gate["minimum_improved_domains"],
        adaptive_gap_threshold=_finite(
            gate["adaptive_gap_relative"], "adaptive gap"
        ),
        required_a4_comparisons=tuple(gate["required_a4_comparisons"]),
        a4_statuses=_A4_STATUSES,
        a5_statuses=_A5_STATUSES,
        output_dir=Path(outputs["formal"]),
        replay_dir=Path(outputs["replay"]),
    )


__all__ = ["A4Config", "A4ConfigError", "load_a4_config"]
