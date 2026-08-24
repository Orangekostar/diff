"""Fail-closed authority loader for the MVD M0/M1 feasibility study."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml
from yaml.nodes import MappingNode


class MVDConfigError(ValueError):
    """Raised when the frozen MVD protocol or one of its sources drifts."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


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
            raise MVDConfigError("config mapping key is not hashable") from error
        if duplicate:
            raise MVDConfigError(f"config contains duplicate key: {key}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


_TOP_KEYS = {
    "schema_version",
    "scope",
    "sources",
    "authority",
    "acquisition",
    "estimator",
    "m0",
    "m1",
    "bootstrap",
    "outputs",
}
_SOURCE_PATHS = {
    "controlling_prompt": Path("docs/MVD_CONTROLLING_PROMPT.md"),
    "repository_audit": Path("docs/MVD_REPOSITORY_AUTHORITY_AUDIT.md"),
    "m0_protocol": Path("docs/MVD_M0_PROTOCOL.md"),
    "m1_protocol": Path("docs/MVD_M1_OBSERVABILITY_PROTOCOL.md"),
    "implementation_plan": Path(
        "docs/superpowers/plans/2026-08-24-mvd-feasibility.md"
    ),
    "mva_config": Path("paper_v3/configs/mva_a0_a3.yaml"),
    "a4_config": Path("paper_v3/configs/mva_a4_global_mask.yaml"),
    "a2_manifest": Path("results/mva/a2_oracle_value/artifact_manifest.json"),
    "a2_state_metrics": Path("results/mva/a2_oracle_value/state_metrics.parquet"),
    "a4_manifest": Path("results/mva/a4_global_task_mask/artifact_manifest.json"),
    "a4_summary": Path("results/mva/a4_global_task_mask/summary.json"),
    "a4_state_metrics": Path("results/mva/a4_global_task_mask/state_metrics.parquet"),
    "a4_source_values": Path("results/mva/a4_global_task_mask/source_values.parquet"),
    "a4_fit_audits": Path("results/mva/a4_global_task_mask/fit_audits.csv"),
    "a4_rankings": Path("results/mva/a4_global_task_mask/rankings.csv"),
    "a5_manifest": Path("results/mva/a5_imitation_policy/artifact_manifest.json"),
    "resnet_weights": Path("paper_v3/assets/resnet18-f37072fd.pth"),
    "candidate_bank_0p015625": Path(
        "artifacts/mvd_authority/a4_candidate_bank_0p015625.npz"
    ),
    "candidate_bank_0p03125": Path(
        "artifacts/mvd_authority/a4_candidate_bank_0p03125.npz"
    ),
    "observed_features_0p015625": Path(
        "artifacts/mvd_authority/observed_candidate_features_0p015625.npz"
    ),
    "observed_features_0p03125": Path(
        "artifacts/mvd_authority/observed_candidate_features_0p03125.npz"
    ),
}
_DOMAINS = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)
_INITIAL_BUDGETS = MappingProxyType(
    {
        "74t7kcdgkr": 0.03125,
        "cgtnjyggtm": 0.015625,
        "w68dtmpfyf": 0.015625,
        "xcmzfsbd9t": 0.015625,
        "yfxyg8jm46": 0.015625,
        "ykhs7s2dck": 0.015625,
    }
)
_CHECKPOINTS = (0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25)
_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
_RANKING_LAMBDAS = (0.1, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class SourceAuthority:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class MVDConfig:
    config_path: Path
    config_sha256: str
    scope: str
    sources: Mapping[str, SourceAuthority]
    baseline_commit: str
    authority_state_sha256: str
    candidate_bank_states: Mapping[float, str]
    observed_feature_bank_states: Mapping[float, str]
    specimen_count: int
    domain_order: tuple[str, ...]
    initial_budgets: Mapping[str, float]
    checkpoints: tuple[float, ...]
    full_mae: float
    pca_dimensions: tuple[int, ...]
    ridge_alpha: float
    random_seed_start: int
    random_seed_count: int
    bootstrap_seed: int
    bootstrap_resamples: int
    m0_minimum_improved_domains: int
    m0_minimum_headroom_retention: float
    m0_strong_headroom_retention: float
    m1_minimum_improved_domains: int
    m1_strong_advantage_capture: float
    observability_ridge_alphas: tuple[float, ...]
    observability_ranking_lambdas: tuple[float, ...]
    observability_epochs: int
    output_work: Path
    output_m0: Path
    output_m1: Path
    output_replay: Path


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise MVDConfigError(f"{label} must be a string-keyed mapping")
    return value


def _exact(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise MVDConfigError(f"{label} keys changed")


def _numbers(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MVDConfigError(f"{label} must be a sequence")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise MVDConfigError(f"{label} contains a non-number")
        number = float(item)
        if not math.isfinite(number):
            raise MVDConfigError(f"{label} contains a non-finite value")
        output.append(number)
    return tuple(output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sources(raw: object, root: Path) -> Mapping[str, SourceAuthority]:
    values = _mapping(raw, "sources")
    if set(values) != set(_SOURCE_PATHS):
        raise MVDConfigError("source registry keys changed")
    output: dict[str, SourceAuthority] = {}
    for name, expected_relative in _SOURCE_PATHS.items():
        entry = _mapping(values[name], f"source {name}")
        _exact(entry, {"path", "sha256"}, f"source {name}")
        if entry["path"] != expected_relative.as_posix():
            raise MVDConfigError(f"source authority changed: {name}")
        digest = entry["sha256"]
        if type(digest) is not str or len(digest) != 64:
            raise MVDConfigError(f"source authority changed: {name}")
        try:
            path = (root / expected_relative).resolve(strict=True)
            path.relative_to(root)
        except (OSError, ValueError) as error:
            raise MVDConfigError(f"source is unavailable: {name}") from error
        if not path.is_file() or path.is_symlink() or _sha256(path) != digest:
            raise MVDConfigError(f"source authority changed: {name}")
        output[name] = SourceAuthority(name=name, path=expected_relative, sha256=digest)
    return MappingProxyType(output)


def _load_yaml(path: Path) -> Mapping[str, object]:
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise MVDConfigError("MVD config cannot be read") from error
    values = _mapping(payload, "config")
    _exact(values, _TOP_KEYS, "top-level")
    return values


def load_mvd_config(path: str | Path, *, project_root: str | Path) -> MVDConfig:
    """Load MVD only when every registered constant and source remains exact."""

    root = Path(project_root).resolve(strict=True)
    config_path = Path(path).resolve(strict=True)
    raw = _load_yaml(config_path)
    if raw["schema_version"] != 1 or raw["scope"] != "mvd_m0_m1_feasibility":
        raise MVDConfigError("MVD config identity changed")
    sources = _load_sources(raw["sources"], root)

    authority = _mapping(raw["authority"], "authority")
    _exact(
        authority,
        {
            "baseline_commit",
            "authority_state_sha256",
            "candidate_bank_states",
            "observed_feature_bank_states",
            "specimen_count",
            "domain_order",
            "initial_budgets",
        },
        "authority",
    )
    if (
        authority["baseline_commit"]
        != "d0e0ebfca1f1de6b04e9cb43a5065de3435aee5b"
        or authority["authority_state_sha256"]
        != "3ef44b3379a4377758443d6fa6ef23d8aae2a83f32b5f1ce86a42b18398c5f3a"
        or authority["specimen_count"] != 276
        or tuple(authority["domain_order"]) != _DOMAINS
        or dict(_mapping(authority["initial_budgets"], "initial budgets"))
        != dict(_INITIAL_BUDGETS)
    ):
        raise MVDConfigError("MVD authority changed")
    bank_states_raw = _mapping(authority["candidate_bank_states"], "bank states")
    expected_bank_states = {
        "0.015625": "2b17097a85fdb41b4413fecfa7f5b141b2e132cc479adb127fceb28c2c444fc4",
        "0.03125": "e44d7b8e1ef1b1f715eeddc3ab9af485d1a993e0acfd792d86086a282b2fa0c0",
    }
    if dict(bank_states_raw) != expected_bank_states:
        raise MVDConfigError("candidate bank states changed")
    observed_states_raw = _mapping(
        authority["observed_feature_bank_states"], "observed feature bank states"
    )
    expected_observed_states = {
        "0.015625": "a91ade889bd87a941a58574d2d06278e908a961836686b984b9b6cf88c24580f",
        "0.03125": "a0e9bc870b69b08041e1bbbae3900ebe2028c84fc5ad95b4a7f22857908e5cd5",
    }
    if dict(observed_states_raw) != expected_observed_states:
        raise MVDConfigError("observed feature bank states changed")

    acquisition = _mapping(raw["acquisition"], "acquisition")
    _exact(
        acquisition,
        {
            "cell_shape",
            "checkpoints",
            "auebc_range",
            "interpolation",
            "budget_unit",
            "selection",
        },
        "acquisition",
    )
    checkpoints = _numbers(acquisition["checkpoints"], "checkpoints")
    if (
        tuple(acquisition["cell_shape"]) != (8, 8)
        or checkpoints != _CHECKPOINTS
        or _numbers(acquisition["auebc_range"], "AUEBC range") != (0.0625, 0.25)
        or acquisition["interpolation"] != "bilinear"
        or acquisition["budget_unit"] != "unique_native_raster_locations"
        or acquisition["selection"] != "exact_cost_frozen_ranking_skip_nonfitting"
    ):
        raise MVDConfigError("acquisition protocol changed")

    estimator = _mapping(raw["estimator"], "estimator")
    _exact(
        estimator,
        {"full_mae", "pca_dimensions", "ridge_alpha", "primary_protocol"},
        "estimator",
    )
    pca_dimensions = tuple(estimator["pca_dimensions"])
    if (
        float(estimator["full_mae"]) != 0.08963580465761432
        or pca_dimensions != (8, 16, 32)
        or float(estimator["ridge_alpha"]) != 10.0
        or estimator["primary_protocol"] != "P-B"
    ):
        raise MVDConfigError("estimator protocol changed")

    m0 = _mapping(raw["m0"], "M0")
    _exact(
        m0,
        {
            "methods",
            "minimum_improved_domains",
            "minimum_headroom_retention",
            "strong_headroom_retention",
            "require_positive_bootstrap_lower",
            "stop_on_no_go",
        },
        "M0",
    )
    if (
        tuple(m0["methods"])
        != (
            "uniform",
            "random_median",
            "one_shot_reconstruction",
            "global_mechanical_mask",
            "one_shot_mechanical_oracle",
            "sequential_mechanical_oracle",
            "FULL",
        )
        or m0["minimum_improved_domains"] != 4
        or float(m0["minimum_headroom_retention"]) != 0.20
        or float(m0["strong_headroom_retention"]) != 0.50
        or m0["require_positive_bootstrap_lower"] is not True
        or m0["stop_on_no_go"] is not True
    ):
        raise MVDConfigError("M0 gate changed")

    m1 = _mapping(raw["m1"], "M1")
    _exact(
        m1,
        {
            "methods",
            "ridge_alphas",
            "ranking_lambdas",
            "epochs",
            "minimum_improved_domains",
            "strong_advantage_capture",
            "require_positive_bootstrap_lower",
            "stop_before_m2",
        },
        "M1",
    )
    if (
        tuple(m1["methods"])
        != (
            "global_mechanical",
            "candidate_only",
            "global_candidate",
            "a5_initial",
            "observed_uncertainty",
            "random_median",
        )
        or _numbers(m1["ridge_alphas"], "Ridge alphas") != _RIDGE_ALPHAS
        or _numbers(m1["ranking_lambdas"], "ranking lambdas")
        != _RANKING_LAMBDAS
        or m1["epochs"] != 50
        or m1["minimum_improved_domains"] != 4
        or float(m1["strong_advantage_capture"]) != 0.35
        or m1["require_positive_bootstrap_lower"] is not True
        or m1["stop_before_m2"] is not True
    ):
        raise MVDConfigError("M1 protocol changed")

    bootstrap = _mapping(raw["bootstrap"], "bootstrap")
    _exact(bootstrap, {"seed", "resamples", "unit", "quantiles"}, "bootstrap")
    if (
        bootstrap["seed"] != 20260824
        or bootstrap["resamples"] != 100_000
        or bootstrap["unit"] != "held_out_domain"
        or _numbers(bootstrap["quantiles"], "bootstrap quantiles") != (0.025, 0.975)
    ):
        raise MVDConfigError("bootstrap protocol changed")

    outputs = _mapping(raw["outputs"], "outputs")
    _exact(outputs, {"work", "m0", "m1", "replay"}, "outputs")
    expected_outputs = {
        "work": "results/mvd/.work",
        "m0": "results/mvd/m0_one_shot_oracle",
        "m1": "results/mvd/m1_observability",
        "replay": "results/mvd/replay",
    }
    if dict(outputs) != expected_outputs:
        raise MVDConfigError("output paths changed")

    return MVDConfig(
        config_path=config_path,
        config_sha256=_sha256(config_path),
        scope="mvd_m0_m1_feasibility",
        sources=sources,
        baseline_commit=str(authority["baseline_commit"]),
        authority_state_sha256=str(authority["authority_state_sha256"]),
        candidate_bank_states=MappingProxyType(
            {float(key): value for key, value in expected_bank_states.items()}
        ),
        observed_feature_bank_states=MappingProxyType(
            {float(key): value for key, value in expected_observed_states.items()}
        ),
        specimen_count=276,
        domain_order=_DOMAINS,
        initial_budgets=_INITIAL_BUDGETS,
        checkpoints=checkpoints,
        full_mae=float(estimator["full_mae"]),
        pca_dimensions=pca_dimensions,
        ridge_alpha=float(estimator["ridge_alpha"]),
        random_seed_start=2026082300,
        random_seed_count=100,
        bootstrap_seed=int(bootstrap["seed"]),
        bootstrap_resamples=int(bootstrap["resamples"]),
        m0_minimum_improved_domains=int(m0["minimum_improved_domains"]),
        m0_minimum_headroom_retention=float(m0["minimum_headroom_retention"]),
        m0_strong_headroom_retention=float(m0["strong_headroom_retention"]),
        m1_minimum_improved_domains=int(m1["minimum_improved_domains"]),
        m1_strong_advantage_capture=float(m1["strong_advantage_capture"]),
        observability_ridge_alphas=_RIDGE_ALPHAS,
        observability_ranking_lambdas=_RANKING_LAMBDAS,
        observability_epochs=int(m1["epochs"]),
        output_work=Path(str(outputs["work"])),
        output_m0=Path(str(outputs["m0"])),
        output_m1=Path(str(outputs["m1"])),
        output_replay=Path(str(outputs["replay"])),
    )


__all__ = ["MVDConfig", "MVDConfigError", "SourceAuthority", "load_mvd_config"]
