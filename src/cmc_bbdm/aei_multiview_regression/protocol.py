"""Hash-bound protocol loader for the AEI multi-view experiment."""

from __future__ import annotations

import hashlib
import math
import stat
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]


class MultiViewProtocolError(ValueError):
    """Raised when the registered multi-view protocol changes."""


_REGISTERED_CONFIG = Path("paper_v3/configs/aei_multiview_regression.yaml")
_VIEWS = ("FULL", "BILINEAR_50", "BILINEAR_25")
_STAGES = ("E1", "E2", "E3", "E4", "E5")
_PCA_DIMENSIONS = (8, 16, 32)
_CONSISTENCY_GRID = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
_TARGET_LOSSES = ("mse", "huber")
_BASELINE_MAE = 0.08963580465761432
_DOMAIN_ORDER = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)
_GATES = {
    "e1": {
        "predictive_equivalence_min_pair_correlation": 0.95,
        "predictive_equivalence_max_relative_view_mae": 1.10,
        "complementarity_min_useful_views": 2,
        "complementarity_max_pair_residual_correlation": 0.90,
        "complementarity_min_oracle_improvement_fraction": 0.10,
    },
    "e2": {
        "maximum_equal_domain_mae": _BASELINE_MAE,
        "minimum_improved_domains": 4,
    },
    "e3": {
        "require_below_best_single": True,
        "maximum_equal_domain_mae": _BASELINE_MAE,
        "minimum_improved_domains": 4,
    },
    "e4": {"require_e3_complementarity": True},
    "e5": {
        "require_e1_nontrivial": True,
        "require_confirmed_complementarity": True,
        "minimum_remaining_oracle_gap_fraction": 0.05,
    },
}


@dataclass(frozen=True, slots=True)
class SourceAuthority:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class MultiViewProtocol:
    config_path: Path
    sources: tuple[SourceAuthority, ...]
    views: tuple[str, str, str]
    stage_order: tuple[str, str, str, str, str]
    baseline_mae: float
    specimen_count: int
    domain_order: tuple[str, ...]
    pca_dimensions: tuple[int, int, int]
    ridge_alpha: float
    consistency_grid: tuple[float, ...]
    target_losses: tuple[str, str]
    huber_delta: float
    complementarity_grid: tuple[float, ...]
    bootstrap_seed: int
    bootstrap_resamples: int
    output_root: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise MultiViewProtocolError(f"{label} must be a mapping")
    if any(type(key) is not str or not key for key in value):
        raise MultiViewProtocolError(f"{label} has invalid keys")
    return value


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise MultiViewProtocolError(f"{label} must be a nonempty sequence")
    return tuple(value)


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise MultiViewProtocolError(f"{label} must be numeric")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise MultiViewProtocolError(f"{label} must be finite")
    return result


def _source_authorities(
    value: object, *, project_root: Path
) -> tuple[SourceAuthority, ...]:
    entries = _mapping(value, "sources")
    if not entries:
        raise MultiViewProtocolError("sources cannot be empty")
    result: list[SourceAuthority] = []
    for name, raw in entries.items():
        item = _mapping(raw, f"source {name}")
        if set(item) != {"path", "sha256"}:
            raise MultiViewProtocolError(f"source {name} schema changed")
        raw_path = item["path"]
        expected = item["sha256"]
        if type(raw_path) is not str or not raw_path or "\x00" in raw_path:
            raise MultiViewProtocolError(f"source {name} path is invalid")
        if (
            type(expected) is not str
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise MultiViewProtocolError(f"source {name} SHA-256 is invalid")
        path = (project_root / raw_path).resolve()
        try:
            info = path.stat(follow_symlinks=False)
        except OSError as error:
            raise MultiViewProtocolError(f"source {name} is unavailable") from error
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise MultiViewProtocolError(f"source {name} is not a regular file")
        if _sha256_file(path) != expected:
            raise MultiViewProtocolError(f"source {name} digest changed")
        result.append(SourceAuthority(name=name, path=path, sha256=expected))
    return tuple(result)


def load_protocol(path: str | Path, *, project_root: str | Path) -> MultiViewProtocol:
    """Load the one registered protocol and verify every source authority."""

    root = Path(project_root).resolve()
    requested = Path(path).resolve()
    registered = (root / _REGISTERED_CONFIG).resolve()
    if requested != registered:
        raise MultiViewProtocolError("protocol does not use the registered path")
    try:
        payload = yaml.safe_load(requested.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise MultiViewProtocolError("protocol is unavailable or invalid") from error
    config = _mapping(payload, "protocol")
    expected_keys = {
        "schema_version",
        "scope",
        "sources",
        "cohort",
        "response",
        "views",
        "stage_order",
        "estimator",
        "cooperative",
        "complementarity",
        "bootstrap",
        "gates",
        "outputs",
    }
    if set(config) != expected_keys or config["schema_version"] != 1:
        raise MultiViewProtocolError("protocol schema changed")
    if config["scope"] != "aei_mechanics_consistent_multiview_regression":
        raise MultiViewProtocolError("protocol scope changed")

    cohort = _mapping(config["cohort"], "cohort")
    response = _mapping(config["response"], "response")
    estimator = _mapping(config["estimator"], "estimator")
    cooperative = _mapping(config["cooperative"], "cooperative")
    complementarity = _mapping(config["complementarity"], "complementarity")
    bootstrap = _mapping(config["bootstrap"], "bootstrap")
    outputs = _mapping(config["outputs"], "outputs")

    views = tuple(_sequence(config["views"], "views"))
    stages = tuple(_sequence(config["stage_order"], "stage_order"))
    domains = tuple(_sequence(cohort.get("domain_order"), "domain_order"))
    dimensions = tuple(_sequence(estimator.get("pca_dimensions"), "pca_dimensions"))
    consistency = tuple(
        _number(item, "consistency strength")
        for item in _sequence(cooperative.get("lambda_grid"), "lambda_grid")
    )
    losses = tuple(_sequence(cooperative.get("target_losses"), "target_losses"))
    complementarity_grid = tuple(
        _number(item, "complementarity strength")
        for item in _sequence(
            complementarity.get("lambda_grid"), "complementarity lambda_grid"
        )
    )
    if views != _VIEWS or stages != _STAGES:
        raise MultiViewProtocolError("view or stage order changed")
    if domains != _DOMAIN_ORDER or dimensions != _PCA_DIMENSIONS:
        raise MultiViewProtocolError("domain or PCA registry changed")
    if consistency != _CONSISTENCY_GRID or losses != _TARGET_LOSSES:
        raise MultiViewProtocolError("cooperative search registry changed")
    specimen_count = cohort.get("specimen_count")
    if type(specimen_count) is not int or specimen_count != 276:
        raise MultiViewProtocolError("cohort size changed")
    if response.get("name") != "damaged_to_intact_cai_strength_ratio":
        raise MultiViewProtocolError("response changed")
    baseline = _number(response.get("baseline_equal_domain_mae"), "baseline MAE")
    if baseline != _BASELINE_MAE:
        raise MultiViewProtocolError("baseline changed")
    ridge_alpha = _number(estimator.get("ridge_alpha"), "ridge alpha")
    huber_delta = _number(cooperative.get("huber_delta"), "Huber delta")
    if ridge_alpha != 10.0 or huber_delta != 0.05:
        raise MultiViewProtocolError("registered estimator changed")
    seed = bootstrap.get("seed")
    resamples = bootstrap.get("resamples")
    if seed != 20260811 or resamples != 100_000:
        raise MultiViewProtocolError("bootstrap registry changed")
    output_value = outputs.get("root")
    if output_value != "results/multiview":
        raise MultiViewProtocolError("output root changed")
    gates = _mapping(config["gates"], "gates")
    if gates != _GATES:
        raise MultiViewProtocolError("registered stage gates changed")

    return MultiViewProtocol(
        config_path=requested,
        sources=_source_authorities(config["sources"], project_root=root),
        views=_VIEWS,
        stage_order=_STAGES,
        baseline_mae=baseline,
        specimen_count=specimen_count,
        domain_order=_DOMAIN_ORDER,
        pca_dimensions=_PCA_DIMENSIONS,
        ridge_alpha=ridge_alpha,
        consistency_grid=_CONSISTENCY_GRID,
        target_losses=_TARGET_LOSSES,
        huber_delta=huber_delta,
        complementarity_grid=complementarity_grid,
        bootstrap_seed=seed,
        bootstrap_resamples=resamples,
        output_root=root / output_value,
    )


__all__ = [
    "MultiViewProtocol",
    "MultiViewProtocolError",
    "SourceAuthority",
    "load_protocol",
]
