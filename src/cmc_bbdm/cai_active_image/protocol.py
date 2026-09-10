"""Hash-bound loader for the frozen CAI active-image v2 protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from .contracts import ACTOR_STATE_PROTOCOL, INITIAL_PROPOSAL_RULE

_REGISTERED_PATH = Path("paper_v3/configs/cai_active_image_v2.yaml")
EXTERNAL_SOURCE_ROOT_BINDING = "external:cmc_damage_inference"


@dataclass(frozen=True, slots=True)
class SourceBinding:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class CAIActiveImageProtocol:
    config_path: Path
    config_sha256: str
    repository_base_sha: str
    sources: tuple[SourceBinding, ...]
    domain_order: tuple[str, ...]
    vlm_revision: str
    vlm_prompt_sha256: str
    vlm_render_version: str
    initial_proposal_rule: str
    actor_state_protocol: str
    early_budget: float
    endpoint_budget: float
    reporting_budgets: tuple[float, ...]
    seeds: tuple[int, ...]
    predictor_updates: tuple[int, ...]
    predictor_mask_counts: tuple[int, ...]
    predictor_validation_interval: int
    actor_updates: int
    static_updates: int
    predictor_batch_size: int
    actor_batch_size: int
    validation_interval: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    entropy_weight: float
    value_weight: float
    predictor_width: int
    actor_width: int
    device: str
    wall_clock_limit_hours: float
    bootstrap_replicates: int
    bootstrap_seed: int
    preselected_cases: tuple[str, ...]
    results_dir: Path
    artifacts_dir: Path

    @property
    def total_optimizer_updates(self) -> int:
        return sum(self.predictor_updates) + len(self.seeds) * (
            3 * self.actor_updates + self.static_updates
        )

    def source(self, name: str) -> Path:
        matches = tuple(item.path for item in self.sources if item.name == name)
        if len(matches) != 1:
            raise ValueError(f"registered source is missing: {name}")
        return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a mapping")
    return value


def load_protocol(
    path: str | Path, *, project_root: str | Path
) -> CAIActiveImageProtocol:
    root = Path(project_root).resolve(strict=True)
    source = Path(path).resolve(strict=True)
    if source != (root / _REGISTERED_PATH).resolve(strict=True):
        raise ValueError("CAI v2 protocol path is not registered")
    raw = source.read_bytes()
    payload = _mapping(yaml.safe_load(raw), "protocol")
    if (
        payload.get("schema_version") != 2
        or payload.get("stage") != "CAI_ACTIVE_IMAGE_VLM_GUIDED_AGENT"
        or payload.get("configuration_frozen") is not True
        or payload.get("repository_base_sha")
        != "29b3249610c821e8f9c4f58d740588181c443dd5"
    ):
        raise ValueError("CAI v2 protocol identity changed")

    bindings: list[SourceBinding] = []
    for name, raw_binding in _mapping(payload.get("sources"), "sources").items():
        binding = _mapping(raw_binding, f"source {name}")
        if set(binding) != {"path", "sha256"}:
            raise ValueError(f"source binding changed: {name}")
        relative = Path(str(binding["path"]))
        expected = str(binding["sha256"])
        if relative.is_absolute() or ".." in relative.parts or len(expected) != 64:
            raise ValueError(f"source binding is invalid: {name}")
        bound = (root / relative).resolve(strict=True)
        if not bound.is_file() or _sha256(bound) != expected:
            raise ValueError(f"source hash changed: {name}")
        bindings.append(SourceBinding(str(name), bound, expected))

    cohort = _mapping(payload.get("cohort"), "cohort")
    vlm = _mapping(payload.get("vlm"), "vlm")
    acquisition = _mapping(payload.get("acquisition"), "acquisition")
    training = _mapping(payload.get("training"), "training")
    evaluation = _mapping(payload.get("evaluation"), "evaluation")
    outputs = _mapping(payload.get("outputs"), "outputs")
    protocol = CAIActiveImageProtocol(
        config_path=source,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        repository_base_sha=str(payload["repository_base_sha"]),
        sources=tuple(bindings),
        domain_order=tuple(str(item) for item in cohort["domain_order"]),
        vlm_revision=str(vlm["revision"]),
        vlm_prompt_sha256=str(vlm["prompt_sha256"]),
        vlm_render_version=str(vlm["render_version"]),
        initial_proposal_rule=str(acquisition["initial_proposal_rule"]),
        actor_state_protocol=str(training["actor_state_protocol"]),
        early_budget=float(acquisition["early_budget"]),
        endpoint_budget=float(acquisition["endpoint_budget"]),
        reporting_budgets=tuple(float(item) for item in acquisition["reporting_budgets"]),
        seeds=tuple(int(item) for item in training["seeds"]),
        predictor_updates=tuple(int(item) for item in training["predictor_updates"]),
        predictor_mask_counts=tuple(
            int(item) for item in training["predictor_mask_counts"]
        ),
        predictor_validation_interval=int(training["predictor_validation_interval"]),
        actor_updates=int(training["actor_updates_per_seed"]),
        static_updates=int(training["learned_static_updates_per_seed"]),
        predictor_batch_size=int(training["predictor_batch_size"]),
        actor_batch_size=int(training["actor_batch_size"]),
        validation_interval=int(training["validation_interval"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        entropy_weight=float(training["entropy_weight"]),
        value_weight=float(training["value_weight"]),
        predictor_width=int(training["predictor_width"]),
        actor_width=int(training["actor_width"]),
        device=str(training["device"]),
        wall_clock_limit_hours=float(training["wall_clock_limit_hours"]),
        bootstrap_replicates=int(evaluation["bootstrap_replicates"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
        preselected_cases=tuple(str(item) for item in evaluation["preselected_cases"]),
        results_dir=(root / str(outputs["results"])).resolve(),
        artifacts_dir=(root / str(outputs["artifacts"])).resolve(),
    )
    if (
        cohort.get("roster") != "VLM_CSCAN_PILOT_60"
        or (cohort.get("train"), cohort.get("valid"), cohort.get("test")) != (24, 12, 24)
        or len(protocol.domain_order) != 6
        or vlm.get("repository") != "Qwen/Qwen2.5-VL-7B-Instruct"
        or protocol.vlm_revision != "cc594898137f460bfe9f0759e9844b3ce807cfb5"
        or protocol.vlm_prompt_sha256
        != "07ba97c242f8f8ab4ec0b742d64aec231dace2455bb275e285174381c1e79ce0"
        or protocol.vlm_render_version
        != "b341e99fb74d9f815be810b2e8b1d91e59f236540072577c244313c584281f40"
        or vlm.get("schema_version") != 2
        or acquisition.get("action_protocol") != "NATIVE_8X8_FULL_CELL_V1"
        or protocol.initial_proposal_rule != INITIAL_PROPOSAL_RULE
        or protocol.actor_state_protocol != ACTOR_STATE_PROTOCOL
        or acquisition.get("grid_shape") != [8, 8]
        or protocol.early_budget != 0.0625
        or protocol.endpoint_budget != 0.25
        or protocol.predictor_updates != (2000, 2000, 2000, 2000)
        or protocol.predictor_mask_counts != (0, 1, 2, 4, 8, 12, 16, 32, 64)
        or protocol.predictor_validation_interval != 250
        or protocol.seeds != (1, 2, 3)
        or protocol.total_optimizer_updates != 21_500
        or protocol.bootstrap_replicates != 5000
    ):
        raise ValueError("CAI v2 frozen design changed")
    return protocol


__all__ = [
    "EXTERNAL_SOURCE_ROOT_BINDING",
    "CAIActiveImageProtocol",
    "SourceBinding",
    "load_protocol",
]
