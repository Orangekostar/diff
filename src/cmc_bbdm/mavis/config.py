"""Strict configuration and source bindings for MAVIS."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from cmc_bbdm.mva.acquisition_grid import INITIAL_BUDGETS


class MAVISConfigError(ValueError):
    """Raised when a MAVIS protocol or bound source changes."""


@dataclass(frozen=True, slots=True)
class SourceBinding:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class MAVISConfig:
    schema_version: int
    mode: str
    seed: int
    master_prompt_sha256: str
    sources: MappingProxyType
    specimen_count: int
    domain_order: tuple[str, ...]
    context_features: tuple[str, ...]
    outer_split: str
    inner_split: str
    source_authority_sha256: str
    initial_budgets: tuple[float, ...]
    checkpoints: tuple[float, ...]
    budget_unit: str
    scout_policy: str
    mris_hidden_size: int
    mris_dimension: int
    learning_rate: float
    loss_weights: MappingProxyType
    on_policy_rounds: int
    selection_criterion: tuple[str, ...]
    final_configuration_frozen: bool
    development_package_sha256: str | None
    output_root: str
    config_sha256: str

    def require_finalized(self) -> None:
        if (
            self.mode != "final"
            or not self.final_configuration_frozen
            or self.development_package_sha256 is None
        ):
            raise MAVISConfigError("final MAVIS configuration is not frozen")


_TOP_LEVEL = {
    "schema_version",
    "scope",
    "mode",
    "seed",
    "master_prompt_sha256",
    "sources",
    "cohort",
    "authority",
    "acquisition",
    "model",
    "aggregation",
    "selection",
    "finalization",
    "outputs",
}


def _mapping(value: object, label: str, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise MAVISConfigError(f"{label} schema changed")
    return value


def _sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MAVISConfigError(f"{label} hash is invalid")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise MAVISConfigError(f"{label} is invalid")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MAVISConfigError(f"{label} is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise MAVISConfigError(f"{label} is invalid")
    return result


def _tuple_text(value: object, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or not value
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise MAVISConfigError(f"{label} is invalid")
    return tuple(value)


def _tuple_float(value: object, label: str) -> tuple[float, ...]:
    if type(value) is not list or not value:
        raise MAVISConfigError(f"{label} is invalid")
    result = tuple(_float(item, label) for item in value)
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise MAVISConfigError(f"{label} is invalid")
    return result


def _source_bindings(value: object, root: Path) -> MappingProxyType:
    entries = _mapping(
        value,
        "sources",
        {
            "mgmr_config",
            "p0_repo_code_map",
            "p0_data_flow",
            "p0_authority_schema",
            "design_spec",
        },
    )
    result: dict[str, SourceBinding] = {}
    seen: set[str] = set()
    for name, raw in entries.items():
        item = _mapping(raw, f"source {name}", {"path", "sha256"})
        path_value = item["path"]
        if (
            type(path_value) is not str
            or not path_value
            or Path(path_value).is_absolute()
            or ".." in Path(path_value).parts
        ):
            raise MAVISConfigError(f"source {name} path is invalid")
        path = root / path_value
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise MAVISConfigError(f"source {name} is unavailable") from error
        actual = hashlib.sha256(payload).hexdigest()
        expected = _sha(item["sha256"], f"source {name}")
        if actual != expected:
            raise MAVISConfigError(f"source {name} hash changed")
        resolved = str(path.resolve())
        if resolved in seen:
            raise MAVISConfigError("source paths must be unique")
        seen.add(resolved)
        result[name] = SourceBinding(path=path_value, sha256=expected)
    return MappingProxyType(result)


def load_mavis_config(path: str | Path, *, project_root: str | Path) -> MAVISConfig:
    root = Path(project_root).resolve(strict=True)
    config_path = Path(path)
    try:
        payload_bytes = config_path.read_bytes()
        raw = yaml.safe_load(payload_bytes)
    except (OSError, yaml.YAMLError) as error:
        raise MAVISConfigError("MAVIS config is unavailable") from error
    config = _mapping(raw, "MAVIS config", _TOP_LEVEL)
    if config["schema_version"] != 1 or config["scope"] != "mavis_closed_loop":
        raise MAVISConfigError("MAVIS config identity changed")
    mode = config["mode"]
    if mode not in ("development", "final"):
        raise MAVISConfigError("MAVIS mode is invalid")
    seed = _positive_int(config["seed"], "seed")
    prompt_sha = _sha(config["master_prompt_sha256"], "master prompt")
    sources = _source_bindings(config["sources"], root)

    cohort = _mapping(
        config["cohort"],
        "cohort",
        {
            "specimen_count",
            "domain_order",
            "context_features",
            "outer_split",
            "inner_split",
        },
    )
    specimen_count = _positive_int(cohort["specimen_count"], "specimen count")
    domains = _tuple_text(cohort["domain_order"], "domain order")
    context_features = _tuple_text(cohort["context_features"], "context features")
    if (
        specimen_count != 276
        or len(domains) != 6
        or context_features != ("metadata13", "profile_stats21")
        or cohort["outer_split"] != "leave_one_dataset_out"
        or cohort["inner_split"] != "leave_one_source_dataset_out"
    ):
        raise MAVISConfigError("cohort contract changed")

    authority = _mapping(
        config["authority"], "authority", {"source_authority_sha256"}
    )
    source_authority_sha = _sha(
        authority["source_authority_sha256"], "source authority"
    )
    acquisition = _mapping(
        config["acquisition"],
        "acquisition",
        {
            "initial_budgets",
            "checkpoints",
            "budget_unit",
            "scout_policy",
            "cell_shape",
        },
    )
    initial_budgets = _tuple_float(acquisition["initial_budgets"], "initial budgets")
    checkpoints = _tuple_float(acquisition["checkpoints"], "checkpoints")
    if (
        initial_budgets != INITIAL_BUDGETS
        or checkpoints != (0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25)
        or acquisition["budget_unit"] != "unique_native_raster_locations"
        or acquisition["scout_policy"] != "uniform_geometry_neutral"
        or acquisition["cell_shape"] != [8, 8]
    ):
        raise MAVISConfigError("acquisition contract changed")

    model = _mapping(
        config["model"],
        "model",
        {
            "state_encoder",
            "mris_hidden_size",
            "mris_dimension",
            "learning_rate",
            "loss_weights",
        },
    )
    if model["state_encoder"] != "deepsets":
        raise MAVISConfigError("state encoder changed")
    hidden = _positive_int(model["mris_hidden_size"], "MRIS hidden size")
    dimension = _positive_int(model["mris_dimension"], "MRIS dimension")
    learning_rate = _float(model["learning_rate"], "learning rate")
    loss_raw = _mapping(
        model["loss_weights"], "loss weights", {"cai", "pair", "list", "value"}
    )
    loss_weights = MappingProxyType(
        {name: _float(value, f"loss weight {name}") for name, value in loss_raw.items()}
    )
    if hidden != 64 or dimension != 64 or learning_rate != 0.001:
        raise MAVISConfigError("recommended MRIS defaults changed")

    aggregation = _mapping(
        config["aggregation"], "aggregation", {"source_only", "rounds"}
    )
    rounds = _positive_int(aggregation["rounds"], "on-policy rounds")
    if aggregation["source_only"] is not True or rounds != 3:
        raise MAVISConfigError("on-policy aggregation contract changed")
    selection = _mapping(
        config["selection"], "selection", {"criterion", "target_data_forbidden"}
    )
    criterion = _tuple_text(selection["criterion"], "selection criterion")
    expected_criterion = (
        "source_cai_auebc",
        "source_improved_domains",
        "worst_source_domain_auebc",
        "model_simplicity",
    )
    if criterion != expected_criterion or selection["target_data_forbidden"] is not True:
        raise MAVISConfigError("source-only selection contract changed")
    finalization = _mapping(
        config["finalization"],
        "finalization",
        {"configuration_frozen", "development_package_sha256"},
    )
    frozen = finalization["configuration_frozen"]
    development_sha = finalization["development_package_sha256"]
    if type(frozen) is not bool:
        raise MAVISConfigError("finalization flag is invalid")
    if development_sha is not None:
        development_sha = _sha(development_sha, "development package")
    if mode == "development" and (frozen or development_sha is not None):
        raise MAVISConfigError("development configuration cannot be finalized")
    outputs = _mapping(config["outputs"], "outputs", {"root"})
    output_root = outputs["root"]
    if type(output_root) is not str or output_root != "results/mavis":
        raise MAVISConfigError("output root changed")
    return MAVISConfig(
        schema_version=1,
        mode=mode,
        seed=seed,
        master_prompt_sha256=prompt_sha,
        sources=sources,
        specimen_count=specimen_count,
        domain_order=domains,
        context_features=context_features,
        outer_split=cohort["outer_split"],
        inner_split=cohort["inner_split"],
        source_authority_sha256=source_authority_sha,
        initial_budgets=initial_budgets,
        checkpoints=checkpoints,
        budget_unit=acquisition["budget_unit"],
        scout_policy=acquisition["scout_policy"],
        mris_hidden_size=hidden,
        mris_dimension=dimension,
        learning_rate=learning_rate,
        loss_weights=loss_weights,
        on_policy_rounds=rounds,
        selection_criterion=criterion,
        final_configuration_frozen=frozen,
        development_package_sha256=development_sha,
        output_root=output_root,
        config_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


__all__ = ["MAVISConfig", "MAVISConfigError", "SourceBinding", "load_mavis_config"]
