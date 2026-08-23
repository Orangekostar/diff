"""Fail-closed authority loader for MVA A5 oracle imitation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .config import SourceBinding


class A5ConfigError(ValueError):
    """Raised when the A5 authority or frozen protocol drifts."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    output: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise A5ConfigError(f"duplicate config key: {key}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


@dataclass(frozen=True, slots=True)
class A5Config:
    schema_version: int
    scope: str
    seed: int
    sources: Mapping[str, SourceBinding]
    a4_global_status: str
    a5_authorization_status: str
    specimen_count: int
    domain_order: tuple[str, ...]
    checkpoints: tuple[float, ...]
    auebc_range: tuple[float, float]
    teacher_split: str
    teacher_trajectory_batch_size: int
    pca_dimensions: tuple[int, ...]
    ridge_alpha: float
    tie_tolerance: float
    state_dimension: int
    candidate_dimension: int
    hidden_dimensions: tuple[tuple[int, ...], ...]
    parameter_count: int
    maximum_parameters: int
    epochs: int
    batch_states: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    methods: tuple[str, ...]
    references: tuple[str, ...]
    bootstrap_seed: int
    bootstrap_resamples: int
    minimum_improved_domains: int
    minimum_gap_closure: float
    a5_statuses: tuple[str, ...]
    a6_statuses: tuple[str, ...]
    work_dir: Path
    output_dir: Path
    replay_dir: Path


_DOMAINS = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)
_CHECKPOINTS = (0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25)
_METHODS = (
    "center_first",
    "observed_gradient",
    "observed_uncertainty",
    "imitation_policy",
)
_REFERENCES = (
    "uniform",
    "random_median",
    "global_mechanical_mask",
    "mechanical_oracle",
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise A5ConfigError(f"{label} must be a string-keyed mapping")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise A5ConfigError(f"{label} keys changed")


def _expect(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise A5ConfigError(f"{label} changed")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise A5ConfigError(f"bound source is unavailable: {path}") from error
    return digest.hexdigest()


def _sources(value: object, root: Path) -> Mapping[str, SourceBinding]:
    items = _mapping(value, "sources")
    names = {
        "a0_a3_config",
        "a2_manifest",
        "a4_config",
        "a4_manifest",
        "a4_summary",
        "a5_design",
        "a5_protocol",
    }
    _exact(items, names, "sources")
    output: dict[str, SourceBinding] = {}
    for name in sorted(names):
        row = _mapping(items[name], f"source {name}")
        _exact(row, {"path", "sha256"}, f"source {name}")
        raw_path, digest = row["path"], row["sha256"]
        if type(raw_path) is not str or not raw_path:
            raise A5ConfigError(f"source {name} path is invalid")
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise A5ConfigError(f"source {name} path must be repository relative")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or _sha_file(root / path) != digest
        ):
            raise A5ConfigError(f"source {name} hash changed")
        output[name] = SourceBinding(path=path, sha256=digest)
    return MappingProxyType(output)


def _authorization(root: Path, source: SourceBinding) -> None:
    try:
        summary = json.loads((root / source.path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise A5ConfigError("A5 authorization summary cannot be read") from error
    gate = summary.get("gate")
    if (
        summary.get("global_mask_status") != "MVA_A4_GLOBAL_NO_GO"
        or summary.get("a5_status") != "MVA_A5_AUTHORIZED"
        or not isinstance(gate, dict)
        or gate.get("adaptive_gap_pass") is not True
        or float(gate.get("relative_adaptive_gap", -1.0)) < 0.03
    ):
        raise A5ConfigError("A5 authorization is not valid")


def load_a5_config(
    config_path: str | Path, *, project_root: str | Path
) -> A5Config:
    """Load A5 only when its complete preregistered authority is unchanged."""

    root = Path(project_root).resolve(strict=True)
    try:
        payload = Path(config_path).read_text(encoding="utf-8")
        loaded = yaml.load(payload, Loader=_UniqueLoader)
    except A5ConfigError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise A5ConfigError("A5 config cannot be read") from error
    item = _mapping(loaded, "A5 config")
    _exact(
        item,
        {
            "schema_version",
            "scope",
            "seed",
            "sources",
            "authorization",
            "cohort",
            "acquisition",
            "teacher",
            "policy",
            "training",
            "evaluation",
            "bootstrap",
            "gate",
            "outputs",
        },
        "A5 config",
    )
    _expect(item["schema_version"], 1, "schema_version")
    _expect(item["scope"], "mva_a5_oracle_imitation_policy", "scope")
    _expect(item["seed"], 20260823, "seed")
    sources = _sources(item["sources"], root)

    authorization = _mapping(item["authorization"], "authorization")
    _expect(
        dict(authorization),
        {
            "a4_global_status": "MVA_A4_GLOBAL_NO_GO",
            "a5_status": "MVA_A5_AUTHORIZED",
            "source": "a4_summary",
            "forbidden_stages": [
                "A6_training",
                "A7",
                "reinforcement_learning",
                "transformer",
                "gnn",
            ],
        },
        "authorization",
    )
    _authorization(root, sources["a4_summary"])

    _expect(
        dict(_mapping(item["cohort"], "cohort")),
        {
            "specimen_count": 276,
            "domain_order": list(_DOMAINS),
            "response": "damaged_to_intact_cai_strength_ratio",
            "outer_split": "leave_one_dataset_out",
        },
        "cohort",
    )
    _expect(
        dict(_mapping(item["acquisition"], "acquisition")),
        {
            "cell_shape": [8, 8],
            "initial_budget_authority": "a1_source_selected_per_outer_domain",
            "action_levels": [[0, 1], [1, 2]],
            "checkpoints": list(_CHECKPOINTS),
            "auebc_range": [0.0625, 0.25],
            "interpolation": "bilinear",
            "budget_unit": "unique_native_raster_locations",
        },
        "acquisition",
    )
    _expect(
        dict(_mapping(item["teacher"], "teacher")),
        {
            "split": "leave_outer_and_query_domain_out",
            "predictor": "P-A",
            "value": "absolute_cai_error_reduction",
            "trajectory_batch_size": 8,
            "pca_dimensions": [8, 16, 32],
            "ridge_alpha": 10.0,
            "tie_tolerance": 1.0e-12,
            "tie_break": ["lower_cell_index", "lower_to_level"],
            "a2_values_for_training": "forbidden",
        },
        "teacher",
    )

    policy = _mapping(item["policy"], "policy architecture")
    _expect(
        dict(policy),
        {
            "state_dimension": 579,
            "candidate_dimension": 8,
            "state_fields": [
                "embedding_512",
                "levels_64",
                "current_p_a_prediction",
                "used_budget",
                "remaining_budget",
            ],
            "candidate_fields": [
                "normalized_row",
                "normalized_column",
                "current_level",
                "added_fraction",
                "measured_fraction",
                "local_gradient",
                "local_variance",
                "nearest_measured_distance",
            ],
            "global_hidden": [64, 32],
            "candidate_hidden": [32, 16],
            "scorer_hidden": [32],
            "activation": "relu",
            "parameter_count": 41617,
            "maximum_parameters": 50000,
        },
        "policy architecture",
    )
    if policy["parameter_count"] >= policy["maximum_parameters"]:
        raise A5ConfigError("policy architecture exceeds its parameter cap")

    training = _mapping(item["training"], "training")
    _expect(
        dict(training),
        {
            "device": "cpu",
            "dtype": "float64",
            "threads": 1,
            "epochs": 50,
            "batch_states": 128,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip": 5.0,
            "objective": "teacher_vs_all_pairwise_logistic",
            "weighting": "equal_domain_equal_specimen_equal_state",
            "early_stopping": "forbidden",
        },
        "training",
    )
    _expect(
        dict(_mapping(item["evaluation"], "evaluation")),
        {
            "primary_protocol": "P-B",
            "methods": list(_METHODS),
            "references": list(_REFERENCES),
            "metrics": [
                "equal_domain_mae",
                "auebc",
                "B_2.5%",
                "B_5%",
                "B_7.5%",
                "gap_closure",
            ],
            "random_seed_count": 100,
        },
        "evaluation",
    )
    _expect(
        dict(_mapping(item["bootstrap"], "bootstrap")),
        {
            "seed": 20260823,
            "resamples": 100000,
            "bit_generator": "PCG64",
            "unit": "held_out_domain",
            "quantiles": [0.025, 0.975],
            "synchronized_across_effects": True,
        },
        "bootstrap",
    )
    gate = _mapping(item["gate"], "gate")
    _expect(
        dict(gate),
        {
            "required_comparisons": ["global_mechanical_mask", "uniform"],
            "require_positive_point_effect": True,
            "require_positive_bootstrap_lower": True,
            "minimum_improved_domains": 4,
            "minimum_gap_closure": 0.20,
            "b5_required": False,
            "a5_statuses": ["MVA_A5_POLICY_GO", "MVA_A5_POLICY_NO_GO"],
            "a6_statuses": ["MVA_A6_AUTHORIZED", "MVA_A6_NOT_AUTHORIZED"],
        },
        "gate",
    )
    _expect(
        dict(_mapping(item["outputs"], "outputs")),
        {
            "work": "results/mva/.work/a5",
            "formal": "results/mva/a5_imitation_policy",
            "replay": "results/mva/replay/a5_imitation_policy",
        },
        "outputs",
    )
    return A5Config(
        schema_version=1,
        scope="mva_a5_oracle_imitation_policy",
        seed=20260823,
        sources=sources,
        a4_global_status="MVA_A4_GLOBAL_NO_GO",
        a5_authorization_status="MVA_A5_AUTHORIZED",
        specimen_count=276,
        domain_order=_DOMAINS,
        checkpoints=_CHECKPOINTS,
        auebc_range=(0.0625, 0.25),
        teacher_split="leave_outer_and_query_domain_out",
        teacher_trajectory_batch_size=8,
        pca_dimensions=(8, 16, 32),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
        state_dimension=579,
        candidate_dimension=8,
        hidden_dimensions=((64, 32), (32, 16), (32,)),
        parameter_count=41617,
        maximum_parameters=50000,
        epochs=50,
        batch_states=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        gradient_clip=5.0,
        methods=_METHODS,
        references=_REFERENCES,
        bootstrap_seed=20260823,
        bootstrap_resamples=100000,
        minimum_improved_domains=4,
        minimum_gap_closure=0.20,
        a5_statuses=("MVA_A5_POLICY_GO", "MVA_A5_POLICY_NO_GO"),
        a6_statuses=("MVA_A6_AUTHORIZED", "MVA_A6_NOT_AUTHORIZED"),
        work_dir=Path("results/mva/.work/a5"),
        output_dir=Path("results/mva/a5_imitation_policy"),
        replay_dir=Path("results/mva/replay/a5_imitation_policy"),
    )


__all__ = ["A5Config", "A5ConfigError", "load_a5_config"]
