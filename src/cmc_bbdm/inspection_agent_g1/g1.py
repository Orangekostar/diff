"""Frozen G1 protocol, component gates, and final decision vocabulary."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

from cmc_bbdm.inspection_agent.cai_assessor import (
    StateCAIAssessor,
    StateFeatureRow,
    fit_state_cai_assessor,
    state_scalars,
)
from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    SourceBackgroundPrior,
    fit_source_background_prior,
    reconstruct_observation,
)
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.mva.encoder_session import MVAEncoderSession

from .contracts import CAIContextMode, TaskTokenMode
from .crossfit import (
    CrossfitCAIAssessorFit,
    CrossfitPriorFit,
    CrossfitRoster,
    fit_crossfit_cai_assessor,
    fit_crossfit_source_prior,
)
from .features import build_policy_state
from .policy_training import G1PolicyTrainingExample
from .statistics import (
    FORMAL_BOOTSTRAP_REPLICATES,
    FORMAL_BOOTSTRAP_SEED,
    G1PairedBootstrap,
)
from .teacher import (
    SourceTeacherAuthorization,
    authorize_source_teacher,
    cai_teacher_label,
    field_teacher_label,
)
from .teacher_bank import (
    G1TeacherBankFile,
    G1TeacherBankRecord,
    materialize_label_independent_states,
    materialize_oracle_checkpoint_states,
    read_teacher_bank,
    write_teacher_bank,
)
from .warm_start import build_deployment_grid

_G1_CONFIG_SHA256 = "aaf216ab9033fffc390f123131bca29952b0d5ca34752434aa9686c1d7ff1f05"
_G1_BASE_SHA = "7a10cd425de582fa158bf6639285731ccd8ff7a7"
_G1_PROMPT_SHA256 = "37b7e4b9860dd338589cf2d9dd8f2cc8d202451a84eeafbc4961b2c5a28fcda5"


class G1ExecutionError(RuntimeError):
    """Raised before execution when the frozen G1 authority cannot be honored."""


@dataclass(frozen=True, slots=True)
class G1Protocol:
    config_path: Path
    config_sha256: str
    specimen_count: int
    domain_order: tuple[str, ...]
    domain_counts: MappingProxyType
    endpoint_budget: float
    deployment_initial_nominal_budget: float
    warm_start_k: int
    warm_start_cells: tuple[int, ...]
    evaluation_checkpoints: tuple[float, ...]
    snapshot_fractions: tuple[float, ...]
    oracle_checkpoints: tuple[float, ...]
    teacher_bank_seed: int
    teacher_temperatures: tuple[float, ...]
    model_candidates: tuple[str, ...]
    learning_rates: tuple[float, ...]
    weight_decays: tuple[float, ...]
    dagger_iterations: tuple[int, ...]
    epochs: int
    patience: int
    batch_specimens: int
    gradient_clip_norm: float
    initialization_seed: int
    stop_thresholds: tuple[float, ...]
    bootstrap_replicates: int
    bootstrap_seed: int
    default_device: str
    encoder_batch_size: int
    teacher_bank_work_path: str
    formal_output: str
    replay_output: str
    work_output: str
    source_bindings: MappingProxyType


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise G1ExecutionError(f"{label} must be a mapping")
    return value


def _numbers(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise G1ExecutionError(f"{label} must be a nonempty list")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise G1ExecutionError(f"{label} is invalid") from error
    if not all(math.isfinite(item) for item in result):
        raise G1ExecutionError(f"{label} is invalid")
    return result


def load_g1_protocol(
    path: str | Path,
    *,
    project_root: str | Path,
) -> G1Protocol:
    root = Path(project_root).resolve(strict=True)
    config_path = Path(path).resolve(strict=True)
    try:
        payload = config_path.read_bytes()
    except OSError as error:
        raise G1ExecutionError("G1 protocol config is unavailable") from error
    config_sha = hashlib.sha256(payload).hexdigest()
    if config_sha != _G1_CONFIG_SHA256:
        raise G1ExecutionError("G1 config SHA-256 changed")
    try:
        loaded = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise G1ExecutionError("G1 protocol config cannot be decoded") from error
    config = _mapping(loaded, "G1 config")
    if (
        config.get("schema_version") != 1
        or config.get("stage") != "INSPECTION_AGENT_G1"
        or config.get("mode") != "formal"
        or config.get("repository_base_sha") != _G1_BASE_SHA
        or config.get("controlling_prompt_sha256") != _G1_PROMPT_SHA256
        or config.get("configuration_frozen") is not True
    ):
        raise G1ExecutionError("G1 protocol identity changed")

    cohort = _mapping(config.get("cohort"), "G1 cohort")
    acquisition = _mapping(config.get("acquisition"), "G1 acquisition")
    teacher_bank = _mapping(config.get("teacher_bank"), "G1 teacher bank")
    distribution = _mapping(
        config.get("teacher_distribution"), "G1 teacher distribution"
    )
    models = _mapping(config.get("models"), "G1 models")
    training = _mapping(config.get("training"), "G1 training")
    stopping = _mapping(config.get("stopping"), "G1 stopping")
    statistics = _mapping(config.get("statistics"), "G1 statistics")
    execution = _mapping(config.get("execution"), "G1 execution")

    domain_order = tuple(str(value) for value in cohort.get("domain_order", ()))
    domain_counts_raw = _mapping(cohort.get("domain_counts"), "G1 domain counts")
    domain_counts = {str(key): int(value) for key, value in domain_counts_raw.items()}
    checkpoints = _numbers(
        acquisition.get("evaluation_checkpoints"), "G1 evaluation checkpoints"
    )
    snapshot_fractions = _numbers(
        teacher_bank.get("continuation_snapshot_fractions"),
        "G1 teacher snapshots",
    )
    oracle_checkpoints = _numbers(
        teacher_bank.get("oracle_checkpoints"), "G1 oracle checkpoints"
    )
    temperatures = _numbers(
        distribution.get("temperatures"), "G1 teacher temperatures"
    )
    learning_rates = _numbers(training.get("learning_rates"), "G1 learning rates")
    weight_decays = _numbers(training.get("weight_decays"), "G1 weight decays")
    stop_thresholds = _numbers(
        stopping.get("threshold_candidates"), "G1 STOP thresholds"
    )
    try:
        dagger_iterations = tuple(int(value) for value in training["dagger_iterations"])
        warm_cells = tuple(int(value) for value in acquisition["warm_start_primary_cells"])
        model_candidates = tuple(str(value) for value in models["candidates"])
    except (KeyError, TypeError, ValueError) as error:
        raise G1ExecutionError("G1 registered roster is invalid") from error
    if (
        cohort.get("specimen_count") != 276
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
        or set(domain_counts) != set(domain_order)
        or sum(domain_counts.values()) != 276
        or checkpoints != (0.0, 0.0625, 0.125, 0.1875, 0.25)
        or snapshot_fractions != (1 / 3, 2 / 3, 1.0)
        or oracle_checkpoints != (0.0625, 0.125, 0.1875, 0.25)
        or temperatures != (0.25, 0.5, 1.0, 2.0)
        or model_candidates != ("SharedActionMLP", "StructuredInspectionPolicy")
        or learning_rates != (0.0001, 0.0003)
        or weight_decays != (0.0001, 0.001)
        or dagger_iterations != (0, 1, 2)
        or stop_thresholds != (0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99)
        or teacher_bank.get("work_path")
        != "results/inspection_agent/g1_work/teacher_banks"
    ):
        raise G1ExecutionError("G1 protocol roster changed")

    sources = _mapping(config.get("sources"), "G1 sources")
    source_bindings: dict[str, tuple[str, str]] = {}
    for name, raw in sorted(sources.items()):
        binding = _mapping(raw, f"G1 source {name}")
        if set(binding) != {"path", "sha256"}:
            raise G1ExecutionError("G1 source binding schema changed")
        relative = Path(str(binding["path"]))
        expected = str(binding["sha256"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(expected) != 64
            or set(expected) - set("0123456789abcdef")
        ):
            raise G1ExecutionError("G1 source binding is invalid")
        try:
            source_payload = (root / relative).read_bytes()
        except OSError as error:
            raise G1ExecutionError(f"G1 source is unavailable: {name}") from error
        if hashlib.sha256(source_payload).hexdigest() != expected:
            raise G1ExecutionError(f"G1 source SHA-256 mismatch: {name}")
        source_bindings[str(name)] = (relative.as_posix(), expected)

    return G1Protocol(
        config_path=config_path,
        config_sha256=config_sha,
        specimen_count=276,
        domain_order=domain_order,
        domain_counts=MappingProxyType(domain_counts),
        endpoint_budget=float(acquisition["endpoint_budget"]),
        deployment_initial_nominal_budget=float(
            acquisition["deployment_initial_nominal_budget"]
        ),
        warm_start_k=int(acquisition["warm_start_primary_k"]),
        warm_start_cells=warm_cells,
        evaluation_checkpoints=checkpoints,
        snapshot_fractions=snapshot_fractions,
        oracle_checkpoints=oracle_checkpoints,
        teacher_bank_seed=int(teacher_bank["random_seed"]),
        teacher_temperatures=temperatures,
        model_candidates=model_candidates,
        learning_rates=learning_rates,
        weight_decays=weight_decays,
        dagger_iterations=dagger_iterations,
        epochs=int(training["epochs"]),
        patience=int(training["early_stopping_patience"]),
        batch_specimens=int(training["batch_specimens"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        initialization_seed=int(training["initialization_seed"]),
        stop_thresholds=stop_thresholds,
        bootstrap_replicates=int(statistics["bootstrap_replicates"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        default_device=str(execution["default_device"]),
        encoder_batch_size=int(execution["encoder_batch_size"]),
        teacher_bank_work_path=str(teacher_bank["work_path"]),
        formal_output=str(execution["formal_output"]),
        replay_output=str(execution["replay_output"]),
        work_output=str(execution["work_output"]),
        source_bindings=MappingProxyType(source_bindings),
    )


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _readonly_image(value: object) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise G1ExecutionError("G1 runtime surface image is invalid")
    if image.flags.c_contiguous and not image.flags.writeable:
        return image
    output = np.frombuffer(
        np.ascontiguousarray(image).tobytes(order="C"), dtype=np.uint8
    ).reshape(image.shape)
    output.setflags(write=False)
    return output


def _surface_carrier(hypothesis: SurfaceHypothesis) -> np.ndarray:
    if type(hypothesis) is not SurfaceHypothesis:
        raise G1ExecutionError("issued surface hypothesis is required")
    value = np.rint(hypothesis.border_median_rgb).clip(0, 255).astype(np.uint8)
    carrier = np.frombuffer(value.tobytes(order="C"), dtype=np.uint8).reshape(1, 1, 3)
    carrier.setflags(write=False)
    return carrier


@dataclass(frozen=True, slots=True)
class G1RuntimeSurface:
    dataset_id: str
    specimen_id: str
    image: np.ndarray
    surface_sha256: str
    hypothesis: SurfaceHypothesis
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        image = _readonly_image(self.image)
        if (
            type(self.dataset_id) is not str
            or not self.dataset_id
            or type(self.specimen_id) is not str
            or not self.specimen_id
            or not _valid_sha256(self.surface_sha256)
            or type(self.hypothesis) is not SurfaceHypothesis
            or not _valid_sha256(self.hypothesis.state_sha256)
        ):
            raise G1ExecutionError("G1 runtime surface identity is invalid")
        object.__setattr__(self, "image", image)
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-runtime-surface",
                    "dataset_id": self.dataset_id,
                    "specimen_id": self.specimen_id,
                    "surface": self.surface_sha256,
                    "hypothesis": self.hypothesis.state_sha256,
                    "image_shape": image.shape,
                    "image_sha256": hashlib.sha256(
                        image.tobytes(order="C")
                    ).hexdigest(),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1Runtime:
    mavis: MAVISAuthority
    surfaces: Mapping[tuple[str, str], G1RuntimeSurface]
    surface_authority_sha256: str
    state_sha256: str = field(init=False)
    _identity_index: MappingProxyType = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.mavis) is not MAVISAuthority or not _valid_sha256(
            self.surface_authority_sha256
        ):
            raise G1ExecutionError("G1 runtime authority is invalid")
        values = dict(self.surfaces)
        expected = tuple(
            zip(self.mavis.dataset_ids, self.mavis.specimen_ids, strict=True)
        )
        if set(values) != set(expected) or any(
            type(surface) is not G1RuntimeSurface
            or key != (surface.dataset_id, surface.specimen_id)
            for key, surface in values.items()
        ):
            raise G1ExecutionError("G1 C-scan and surface rosters differ")
        index = {key: row_index for row_index, key in enumerate(expected)}
        frozen = MappingProxyType(values)
        object.__setattr__(self, "surfaces", frozen)
        object.__setattr__(self, "_identity_index", MappingProxyType(index))
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-runtime-authority",
                    "mavis": self.mavis.state_sha256,
                    "surface_authority": self.surface_authority_sha256,
                    "surfaces": tuple(values[key].state_sha256 for key in expected),
                }
            ),
        )

    @property
    def domain_order(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.mavis.dataset_ids))

    def surface(self, dataset_id: str, specimen_id: str) -> G1RuntimeSurface:
        try:
            return self.surfaces[(dataset_id, specimen_id)]
        except (KeyError, TypeError) as error:
            raise G1ExecutionError("G1 runtime specimen is unavailable") from error

    def specimen_sha256(self, dataset_id: str, specimen_id: str) -> str:
        try:
            index = self._identity_index[(dataset_id, specimen_id)]
        except (KeyError, TypeError) as error:
            raise G1ExecutionError("G1 runtime specimen is unavailable") from error
        surface = self.surface(dataset_id, specimen_id)
        return specimen_integrity_sha256(
            dataset_id=dataset_id,
            specimen_id=specimen_id,
            cscan_source_sha256=self.mavis.source_image_sha256[index],
            cscan_decoded_sha256=self.mavis.decoded_image_sha256[index],
            surface_sha256=surface.surface_sha256,
        )


@dataclass(frozen=True, slots=True)
class G1SourceDependencies:
    roster: CrossfitRoster
    prior_fit: CrossfitPriorFit
    assessor_fit: CrossfitCAIAssessorFit
    authorization: SourceTeacherAuthorization
    assessor_row_count: int
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.roster) is not CrossfitRoster
            or type(self.prior_fit) is not CrossfitPriorFit
            or type(self.assessor_fit) is not CrossfitCAIAssessorFit
            or type(self.authorization) is not SourceTeacherAuthorization
            or self.prior_fit.roster != self.roster
            or self.assessor_fit.roster != self.roster
            or self.authorization.roster_sha256 != self.roster.state_sha256
            or self.prior_fit.prior.source_domains != self.roster.fit_domains
            or self.assessor_fit.assessor.fit_domains != self.roster.fit_domains
            or type(self.assessor_row_count) is not int
            or self.assessor_row_count <= 0
            or len(self.assessor_fit.assessor.fit_sample_ids) != self.assessor_row_count
        ):
            raise G1ExecutionError("G1 source dependency bundle is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-source-dependencies",
                    "roster": self.roster.state_sha256,
                    "prior": self.prior_fit.state_sha256,
                    "assessor": self.assessor_fit.state_sha256,
                    "authorization": self.authorization.state_sha256,
                    "assessor_row_count": self.assessor_row_count,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1FinalDependencies:
    outer_target: str
    fit_domains: tuple[str, ...]
    prior: SourceBackgroundPrior
    assessor: StateCAIAssessor
    assessor_row_count: int
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.fit_domains) is not tuple
            or len(self.fit_domains) != 5
            or len(set(self.fit_domains)) != 5
            or self.outer_target in self.fit_domains
            or type(self.prior) is not SourceBackgroundPrior
            or self.prior.outer_domain != self.outer_target
            or self.prior.source_domains != self.fit_domains
            or type(self.assessor) is not StateCAIAssessor
            or self.assessor.outer_domain != self.outer_target
            or self.assessor.fit_domains != self.fit_domains
            or type(self.assessor_row_count) is not int
            or self.assessor_row_count <= 0
            or len(self.assessor.fit_sample_ids) != self.assessor_row_count
        ):
            raise G1ExecutionError("G1 final dependency bundle is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-final-dependencies",
                    "outer_target": self.outer_target,
                    "fit_domains": self.fit_domains,
                    "prior": self.prior.state_sha256,
                    "assessor": self.assessor.model_state_sha256,
                    "assessor_row_count": self.assessor_row_count,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1TeacherBankBuild:
    path: Path
    outer_target: str
    source_domain: str
    specimen_count: int
    dependency_sha256: str
    bank: G1TeacherBankFile


def specimen_integrity_sha256(
    *,
    dataset_id: str,
    specimen_id: str,
    cscan_source_sha256: str,
    cscan_decoded_sha256: str,
    surface_sha256: str,
) -> str:
    if (
        type(dataset_id) is not str
        or not dataset_id
        or type(specimen_id) is not str
        or not specimen_id
        or not all(
            _valid_sha256(value)
            for value in (
                cscan_source_sha256,
                cscan_decoded_sha256,
                surface_sha256,
            )
        )
    ):
        raise G1ExecutionError("G1 specimen integrity identity is invalid")
    return _json_sha(
        {
            "schema": 1,
            "kind": "g1-physical-specimen-integrity",
            "dataset_id": dataset_id,
            "specimen_id": specimen_id,
            "cscan_source": cscan_source_sha256,
            "cscan_decoded": cscan_decoded_sha256,
            "surface": surface_sha256,
        }
    )


def source_teacher_bank_path(
    work_root: str | Path,
    outer_target: str,
    source_domain: str,
) -> Path:
    if (
        type(outer_target) is not str
        or not outer_target
        or type(source_domain) is not str
        or not source_domain
        or outer_target == source_domain
        or any(
            "/" in value or "\\" in value or value in {".", ".."}
            for value in (outer_target, source_domain)
        )
    ):
        raise G1ExecutionError("teacher-bank fold identity is invalid")
    return Path(work_root) / outer_target / f"{source_domain}.parquet"


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def load_g1_runtime(
    protocol: G1Protocol,
    *,
    project_root: str | Path,
    source_project_root: str | Path,
    progress: Callable[[str], None] | None = None,
) -> G1Runtime:
    if type(protocol) is not G1Protocol:
        raise G1ExecutionError("issued G1 protocol is required")
    root = Path(project_root).resolve(strict=True)
    source_root = Path(source_project_root).resolve(strict=True)
    prompt = (
        source_root.parent / "CODEX_INSPECTION_AGENT_G1_OBSERVABLE_POLICY_PROMPT.md"
    )
    try:
        prompt_sha = hashlib.sha256(prompt.read_bytes()).hexdigest()
    except OSError as error:
        raise G1ExecutionError("controlling G1 prompt is unavailable") from error
    if prompt_sha != _G1_PROMPT_SHA256:
        raise G1ExecutionError("controlling G1 prompt SHA-256 changed")

    from cmc_bbdm.inspection_agent.g0 import (
        _load_runtime_authority,
        load_g0_protocol,
    )

    g0_config = root / protocol.source_bindings["g0_config"][0]
    g0_protocol = load_g0_protocol(g0_config, project_root=root)
    runtime = _load_runtime_authority(
        g0_protocol,
        project_root=root,
        source_project_root=source_root,
        progress=progress,
    )
    surfaces = {
        key: G1RuntimeSurface(
            dataset_id=datum.record.dataset_id,
            specimen_id=datum.record.specimen_id,
            image=_surface_carrier(datum.hypothesis),
            surface_sha256=datum.record.surface_sha256,
            hypothesis=datum.hypothesis,
        )
        for key, datum in runtime.surfaces.items()
    }
    result = G1Runtime(
        mavis=runtime.mavis,
        surfaces=MappingProxyType(surfaces),
        surface_authority_sha256=runtime.surface_authority_sha256,
    )
    counts = {
        domain: result.mavis.dataset_ids.count(domain)
        for domain in protocol.domain_order
    }
    if (
        result.mavis.specimen_count != protocol.specimen_count
        or result.domain_order != protocol.domain_order
        or counts != dict(protocol.domain_counts)
    ):
        raise G1ExecutionError("formal G1 runtime roster changed")
    return result


def load_g1_encoder(
    source_project_root: str | Path,
    *,
    device: str,
) -> MVAEncoderSession:
    source_root = Path(source_project_root).resolve(strict=True)
    if type(device) is not str or not device:
        raise G1ExecutionError("G1 encoder device is invalid")
    from cmc_bbdm.inspection_agent.g0 import _registered_encoder

    return _registered_encoder(source_root, device)


def build_g1_world(
    runtime: G1Runtime,
    *,
    dataset_id: str,
    specimen_id: str,
    task: InspectionTask,
    endpoint_budget: float,
) -> tuple[CausalInspectionWorld, AcquisitionGrid, G1RuntimeSurface]:
    if type(runtime) is not G1Runtime or task not in (
        InspectionTask.FIELD,
        InspectionTask.CAI,
    ):
        raise G1ExecutionError("G1 world request is invalid")
    surface = runtime.surface(dataset_id, specimen_id)
    context = runtime.mavis.policy_context(specimen_id)
    grid = build_deployment_grid(context.native_shape)
    world = CausalInspectionWorld(
        runtime.mavis,
        specimen_id=specimen_id,
        task=task,
        surface_rgb=surface.image,
        surface_sha256=surface.surface_sha256,
        grid=grid,
        endpoint_budget=endpoint_budget,
    )
    return world, grid, surface


def _build_crossfit_assessor_rows(
    runtime: G1Runtime,
    protocol: G1Protocol,
    prior_fit: CrossfitPriorFit,
    *,
    encoder: object,
    progress: Callable[[str], None] | None,
) -> tuple[StateFeatureRow, ...]:
    if not callable(getattr(encoder, "encode", None)):
        raise G1ExecutionError("G1 reconstruction encoder is invalid")
    roster = prior_fit.roster
    rows: list[StateFeatureRow] = []
    selected = tuple(
        (domain, specimen)
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain in roster.fit_domains
    )
    for index, (domain, specimen) in enumerate(selected, start=1):
        world, grid, surface = build_g1_world(
            runtime,
            dataset_id=domain,
            specimen_id=specimen,
            task=InspectionTask.CAI,
            endpoint_budget=protocol.endpoint_budget,
        )
        states = materialize_label_independent_states(
            world,
            grid,
            surface.hypothesis,
            outer_target=roster.outer_target,
            random_seed=protocol.teacher_bank_seed,
            snapshot_fractions=protocol.snapshot_fractions,
        )
        if len(states) != 13:
            raise G1ExecutionError("G1 assessor state roster changed")
        reconstructions = tuple(
            reconstruct_observation(
                state.observation,
                grid,
                prior_fit.prior,
            )
            for state in states
        )
        embeddings = np.asarray(
            encoder.encode(tuple(value.image for value in reconstructions)),
            dtype=np.float64,
        )
        if embeddings.shape != (13, 512) or not np.all(np.isfinite(embeddings)):
            raise G1ExecutionError("G1 assessor embeddings are invalid")
        teacher_view = runtime.mavis.source_teacher_view(specimen)
        for state_index, (state, embedding) in enumerate(
            zip(states, embeddings, strict=True)
        ):
            scalars = state_scalars(state.observation)
            rows.append(
                StateFeatureRow(
                    sample_id=_json_sha(
                        {
                            "schema": 1,
                            "kind": "g1-crossfit-assessor-state",
                            "outer_target": roster.outer_target,
                            "labeled_domain": roster.labeled_domain,
                            "dataset_id": domain,
                            "specimen_id": specimen,
                            "state_source": state.source,
                            "state_index": state_index,
                            "observation": state.observation.state_sha256,
                        }
                    ),
                    specimen_id=specimen,
                    dataset_id=domain,
                    policy=state.source,
                    observation_sha256=state.observation.state_sha256,
                    embedding=embedding,
                    effective_budget=float(scalars[0]),
                    observed_cell_fraction=float(scalars[1]),
                    mean_observed_level=float(scalars[2]),
                    true_cai=teacher_view.true_cai,
                )
            )
        if index % 25 == 0 or index == len(selected):
            _progress(
                progress,
                "G1 crossfit assessor states "
                f"{roster.outer_target}/{roster.labeled_domain}: "
                f"{index}/{len(selected)} specimens",
            )
    return tuple(rows)


def build_g1_source_dependencies(
    runtime: G1Runtime,
    protocol: G1Protocol,
    *,
    outer_target: str,
    labeled_domain: str,
    encoder: object,
    progress: Callable[[str], None] | None = None,
) -> G1SourceDependencies:
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or runtime.domain_order != protocol.domain_order
        or outer_target not in protocol.domain_order
        or labeled_domain not in protocol.domain_order
        or outer_target == labeled_domain
    ):
        raise G1ExecutionError("G1 source dependency request is invalid")
    prior_fit = fit_crossfit_source_prior(
        runtime.mavis,
        outer_target=outer_target,
        labeled_domain=labeled_domain,
    )
    rows = _build_crossfit_assessor_rows(
        runtime,
        protocol,
        prior_fit,
        encoder=encoder,
        progress=progress,
    )
    assessor_fit = fit_crossfit_cai_assessor(
        rows,
        domain_order=protocol.domain_order,
        outer_target=outer_target,
        labeled_domain=labeled_domain,
        pca_dimension=32,
        ridge_alpha=10.0,
    )
    authorization = authorize_source_teacher(
        prior_fit.roster,
        query_domain=labeled_domain,
    )
    result = G1SourceDependencies(
        roster=prior_fit.roster,
        prior_fit=prior_fit,
        assessor_fit=assessor_fit,
        authorization=authorization,
        assessor_row_count=len(rows),
    )
    _progress(
        progress,
        f"G1 dependencies complete {outer_target}/{labeled_domain}: "
        f"{result.state_sha256}",
    )
    return result


def _build_final_assessor_rows(
    runtime: G1Runtime,
    protocol: G1Protocol,
    prior: SourceBackgroundPrior,
    *,
    outer_target: str,
    encoder: object,
    progress: Callable[[str], None] | None,
) -> tuple[StateFeatureRow, ...]:
    if not callable(getattr(encoder, "encode", None)):
        raise G1ExecutionError("G1 reconstruction encoder is invalid")
    fit_domains = tuple(
        domain for domain in protocol.domain_order if domain != outer_target
    )
    selected = tuple(
        (domain, specimen)
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain in fit_domains
    )
    rows: list[StateFeatureRow] = []
    for index, (domain, specimen) in enumerate(selected, start=1):
        world, grid, surface = build_g1_world(
            runtime,
            dataset_id=domain,
            specimen_id=specimen,
            task=InspectionTask.CAI,
            endpoint_budget=protocol.endpoint_budget,
        )
        states = materialize_label_independent_states(
            world,
            grid,
            surface.hypothesis,
            outer_target=outer_target,
            random_seed=protocol.teacher_bank_seed,
            snapshot_fractions=protocol.snapshot_fractions,
        )
        if len(states) != 13:
            raise G1ExecutionError("G1 final assessor state roster changed")
        reconstructions = tuple(
            reconstruct_observation(state.observation, grid, prior) for state in states
        )
        embeddings = np.asarray(
            encoder.encode(tuple(value.image for value in reconstructions)),
            dtype=np.float64,
        )
        if embeddings.shape != (13, 512) or not np.all(np.isfinite(embeddings)):
            raise G1ExecutionError("G1 final assessor embeddings are invalid")
        teacher_view = runtime.mavis.source_teacher_view(specimen)
        for state_index, (state, embedding) in enumerate(
            zip(states, embeddings, strict=True)
        ):
            scalars = state_scalars(state.observation)
            rows.append(
                StateFeatureRow(
                    sample_id=_json_sha(
                        {
                            "schema": 1,
                            "kind": "g1-final-assessor-state",
                            "outer_target": outer_target,
                            "dataset_id": domain,
                            "specimen_id": specimen,
                            "state_source": state.source,
                            "state_index": state_index,
                            "observation": state.observation.state_sha256,
                        }
                    ),
                    specimen_id=specimen,
                    dataset_id=domain,
                    policy=state.source,
                    observation_sha256=state.observation.state_sha256,
                    embedding=embedding,
                    effective_budget=float(scalars[0]),
                    observed_cell_fraction=float(scalars[1]),
                    mean_observed_level=float(scalars[2]),
                    true_cai=teacher_view.true_cai,
                )
            )
        if index % 25 == 0 or index == len(selected):
            _progress(
                progress,
                f"G1 final assessor states {outer_target}: "
                f"{index}/{len(selected)} specimens",
            )
    return tuple(rows)


def build_g1_final_dependencies(
    runtime: G1Runtime,
    protocol: G1Protocol,
    *,
    outer_target: str,
    encoder: object,
    progress: Callable[[str], None] | None = None,
) -> G1FinalDependencies:
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or runtime.domain_order != protocol.domain_order
        or outer_target not in protocol.domain_order
    ):
        raise G1ExecutionError("G1 final dependency request is invalid")
    prior = fit_source_background_prior(
        runtime.mavis,
        outer_domain=outer_target,
    )
    rows = _build_final_assessor_rows(
        runtime,
        protocol,
        prior,
        outer_target=outer_target,
        encoder=encoder,
        progress=progress,
    )
    assessor = fit_state_cai_assessor(
        rows,
        outer_domain=outer_target,
        pca_dimension=32,
        ridge_alpha=10.0,
    )
    result = G1FinalDependencies(
        outer_target=outer_target,
        fit_domains=tuple(
            domain for domain in protocol.domain_order if domain != outer_target
        ),
        prior=prior,
        assessor=assessor,
        assessor_row_count=len(rows),
    )
    _progress(
        progress,
        f"G1 final dependencies complete {outer_target}: {result.state_sha256}",
    )
    return result


def materialize_g1_source_teacher_records(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    specimen_ids: tuple[str, ...],
    progress: Callable[[str], None] | None = None,
) -> tuple[G1TeacherBankRecord, ...]:
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or type(dependencies) is not G1SourceDependencies
        or not callable(getattr(encoder, "encode", None))
        or type(specimen_ids) is not tuple
        or not specimen_ids
        or len(set(specimen_ids)) != len(specimen_ids)
    ):
        raise G1ExecutionError("G1 source teacher-bank request is invalid")
    roster = dependencies.roster
    source_ids = {
        specimen
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == roster.labeled_domain
    }
    if not set(specimen_ids) <= source_ids:
        raise G1ExecutionError("teacher-bank specimen is outside the labeled source")
    records: list[G1TeacherBankRecord] = []
    assessor = dependencies.assessor_fit.assessor
    prior = dependencies.prior_fit.prior
    for specimen_index, specimen in enumerate(specimen_ids, start=1):
        teacher_view = runtime.mavis.source_teacher_view(specimen)
        specimen_sha = runtime.specimen_sha256(roster.labeled_domain, specimen)
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            world, grid, surface = build_g1_world(
                runtime,
                dataset_id=roster.labeled_domain,
                specimen_id=specimen,
                task=task,
                endpoint_budget=protocol.endpoint_budget,
            )
            independent = materialize_label_independent_states(
                world,
                grid,
                surface.hypothesis,
                outer_target=roster.outer_target,
                random_seed=protocol.teacher_bank_seed,
                snapshot_fractions=protocol.snapshot_fractions,
            )
            oracle = materialize_oracle_checkpoint_states(
                world,
                grid,
                surface.hypothesis,
                prior,
                dependencies.authorization,
                full_scan=teacher_view.full_scan,
                checkpoints=protocol.oracle_checkpoints,
                true_cai=(
                    teacher_view.true_cai if task is InspectionTask.CAI else None
                ),
                assessor=(assessor if task is InspectionTask.CAI else None),
                encoder=(encoder if task is InspectionTask.CAI else None),
            )
            state_rows = (
                *(
                    (row.source, row.state_sha256, row.observation)
                    for row in independent
                ),
                *(
                    ("ORACLE_CHECKPOINT", row.state_sha256, row.observation)
                    for row in oracle
                ),
            )
            reconstructions = tuple(
                reconstruct_observation(observation, grid, prior)
                for _source, _state_sha, observation in state_rows
            )
            embeddings = np.asarray(
                encoder.encode(tuple(value.image for value in reconstructions)),
                dtype=np.float64,
            )
            if embeddings.shape != (len(state_rows), 512) or not np.all(
                np.isfinite(embeddings)
            ):
                raise G1ExecutionError("G1 teacher-bank embeddings are invalid")
            scalars = np.asarray(
                [state_scalars(observation) for _, _, observation in state_rows],
                dtype=np.float64,
            )
            estimates = assessor.predict(embeddings, scalars)
            for (
                state_source,
                source_state_sha,
                observation,
            ), reconstruction, embedding, estimate in zip(
                state_rows,
                reconstructions,
                embeddings,
                estimates,
                strict=True,
            ):
                policy_state = build_policy_state(
                    observation,
                    surface.hypothesis,
                    grid,
                    prior,
                    reconstruction,
                    reconstruction_embedding=embedding,
                    cai_estimate=float(estimate),
                    cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
                    task_token_mode=TaskTokenMode.CORRECT,
                )
                if task is InspectionTask.FIELD:
                    label = field_teacher_label(
                        observation,
                        grid,
                        prior,
                        surface.hypothesis,
                        dependencies.authorization,
                        full_scan=teacher_view.full_scan,
                        policy_state_sha256=policy_state.state_sha256,
                    )
                else:
                    label = cai_teacher_label(
                        observation,
                        grid,
                        prior,
                        surface.hypothesis,
                        dependencies.authorization,
                        full_scan=teacher_view.full_scan,
                        true_cai=teacher_view.true_cai,
                        assessor=assessor,
                        encoder=encoder,
                        policy_state_sha256=policy_state.state_sha256,
                    )
                example = G1PolicyTrainingExample(
                    outer_target=roster.outer_target,
                    source_domain=roster.labeled_domain,
                    specimen_sha256=specimen_sha,
                    task=task,
                    dagger_iteration=0,
                    policy_state=policy_state,
                    teacher_label=label,
                )
                records.append(
                    G1TeacherBankRecord(
                        example=example,
                        fit_domains=roster.fit_domains,
                        state_source=state_source,
                        source_state_sha256=source_state_sha,
                        prior_sha256=prior.state_sha256,
                        assessor_sha256=assessor.model_state_sha256,
                    )
                )
        _progress(
            progress,
            f"G1 teacher bank {roster.outer_target}/{roster.labeled_domain}: "
            f"{specimen_index}/{len(specimen_ids)} specimens",
        )
    return tuple(records)


def build_g1_source_teacher_bank(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    work_root: str | Path,
    progress: Callable[[str], None] | None = None,
) -> G1TeacherBankBuild:
    roster = dependencies.roster
    specimen_ids = tuple(
        specimen
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == roster.labeled_domain
    )
    if len(specimen_ids) != int(protocol.domain_counts[roster.labeled_domain]):
        raise G1ExecutionError("formal source teacher-bank roster changed")
    path = source_teacher_bank_path(
        work_root,
        roster.outer_target,
        roster.labeled_domain,
    )
    if path.exists() or path.with_suffix(f"{path.suffix}.manifest.json").exists():
        bank, existing = read_teacher_bank(path)
        if {record.example.specimen_sha256 for record in existing} != {
            runtime.specimen_sha256(roster.labeled_domain, specimen)
            for specimen in specimen_ids
        } or any(
            record.prior_sha256 != dependencies.prior_fit.prior.state_sha256
            or record.assessor_sha256
            != dependencies.assessor_fit.assessor.model_state_sha256
            or record.fit_domains != roster.fit_domains
            for record in existing
        ):
            raise G1ExecutionError("existing G1 teacher bank has a stale dependency")
        _progress(progress, f"G1 teacher bank reused: {path}")
    else:
        records = materialize_g1_source_teacher_records(
            runtime,
            protocol,
            dependencies,
            encoder=encoder,
            specimen_ids=specimen_ids,
            progress=progress,
        )
        bank = write_teacher_bank(path, records)
    return G1TeacherBankBuild(
        path=path,
        outer_target=roster.outer_target,
        source_domain=roster.labeled_domain,
        specimen_count=len(specimen_ids),
        dependency_sha256=dependencies.state_sha256,
        bank=bank,
    )


def build_g1_all_source_teacher_banks(
    runtime: G1Runtime,
    protocol: G1Protocol,
    *,
    encoder: object,
    work_root: str | Path,
    progress: Callable[[str], None] | None = None,
) -> tuple[G1TeacherBankBuild, ...]:
    """Build the exact 30 directed outer-target/source-domain teacher banks."""

    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or runtime.domain_order != protocol.domain_order
        or not callable(getattr(encoder, "encode", None))
    ):
        raise G1ExecutionError("G1 all-bank build request is invalid")
    expected = tuple(
        (outer, source)
        for outer in protocol.domain_order
        for source in protocol.domain_order
        if source != outer
    )
    output: list[G1TeacherBankBuild] = []
    for index, (outer, source) in enumerate(expected, start=1):
        _progress(
            progress,
            f"G1 teacher-bank fold {index}/{len(expected)}: {outer}/{source}",
        )
        dependencies = build_g1_source_dependencies(
            runtime,
            protocol,
            outer_target=outer,
            labeled_domain=source,
            encoder=encoder,
            progress=progress,
        )
        result = build_g1_source_teacher_bank(
            runtime,
            protocol,
            dependencies,
            encoder=encoder,
            work_root=work_root,
            progress=progress,
        )
        if (result.outer_target, result.source_domain) != (outer, source):
            raise G1ExecutionError("G1 all-bank fold identity changed")
        output.append(result)
    if tuple((row.outer_target, row.source_domain) for row in output) != expected:
        raise G1ExecutionError("G1 all-bank directed roster changed")
    return tuple(output)


POLICY_GAP_CLOSURE_MINIMUM = 0.20
POLICY_IMPROVED_DOMAINS_MINIMUM = 4
STOPPING_SAVING_MINIMUM = 0.10
STOPPING_TASK_LOSS_RATIO_MAXIMUM = 1.05
STOPPING_PREMATURE_RATE_MAXIMUM = 0.05

FINAL_G1_STATUSES = (
    "G1_TASK_CONDITIONED_POLICY_GO",
    "G1_ACTIVE_POLICY_GO",
    "G1_CAI_ONLY_POLICY_GO",
    "G1_FIELD_ONLY_POLICY_GO",
    "G1_POLICY_OBSERVABILITY_NO_GO",
    "G1_DEPLOYMENT_BRIDGE_NO_GO",
)
G2_AUTHORIZING_STATUSES = FINAL_G1_STATUSES[:4]


class G1GateError(ValueError):
    """Raised when G1 gate evidence is contradictory or incomplete."""


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _validate_bootstrap(value: G1PairedBootstrap) -> None:
    if (
        type(value) is not G1PairedBootstrap
        or value.replicates != FORMAL_BOOTSTRAP_REPLICATES
        or value.seed != FORMAL_BOOTSTRAP_SEED
        or len(value.domain_effects) != 6
        or len({domain for domain, _effect in value.domain_effects}) != 6
        or not 0 <= value.improved_domains <= 6
        or not all(
            math.isfinite(number)
            for number in (value.point_estimate, value.ci_lower, value.ci_upper)
        )
        or value.ci_lower > value.point_estimate
        or value.point_estimate > value.ci_upper
    ):
        raise G1GateError("formal bootstrap evidence is invalid")


@dataclass(frozen=True, slots=True)
class PolicyGateEvidence:
    task: InspectionTask
    fixed_auebc: float
    learned_auebc: float
    oracle_auebc: float
    baseline_minus_learned: G1PairedBootstrap
    no_target_leakage: bool
    deterministic_replay: bool


@dataclass(frozen=True, slots=True)
class PolicyGateResult:
    task: InspectionTask
    status: str
    passed: bool
    oracle_gap_closure: float | None
    baseline_effect: float
    ci_lower: float
    improved_domains: int
    no_target_leakage: bool
    deterministic_replay: bool
    state_sha256: str


def evaluate_policy_gate(evidence: PolicyGateEvidence) -> PolicyGateResult:
    if type(evidence) is not PolicyGateEvidence:
        raise G1GateError("issued policy-gate evidence is required")
    fixed = float(evidence.fixed_auebc)
    learned = float(evidence.learned_auebc)
    oracle = float(evidence.oracle_auebc)
    _validate_bootstrap(evidence.baseline_minus_learned)
    effect = fixed - learned
    if (
        evidence.task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or not all(math.isfinite(value) and value >= 0.0 for value in (fixed, learned, oracle))
        or not math.isclose(
            effect,
            evidence.baseline_minus_learned.point_estimate,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
        or type(evidence.no_target_leakage) is not bool
        or type(evidence.deterministic_replay) is not bool
    ):
        raise G1GateError("policy-gate evidence is invalid")
    denominator = fixed - oracle
    closure = effect / denominator if denominator > 0.0 else None
    statistical = (
        learned < fixed
        and evidence.baseline_minus_learned.ci_lower > 0.0
        and evidence.baseline_minus_learned.improved_domains
        >= POLICY_IMPROVED_DOMAINS_MINIMUM
        and evidence.no_target_leakage
        and evidence.deterministic_replay
    )
    passed = (
        statistical
        and closure is not None
        and closure >= POLICY_GAP_CLOSURE_MINIMUM
    )
    prefix = f"G1_{evidence.task.value}"
    if passed:
        status = f"{prefix}_POLICY_GO"
    elif statistical and closure is not None and closure < POLICY_GAP_CLOSURE_MINIMUM:
        status = f"{prefix}_DESCRIPTIVE_POLICY_SIGNAL_ONLY"
    else:
        status = f"{prefix}_POLICY_NO_GO"
    payload = {
        "schema": 1,
        "kind": "g1-policy-gate",
        "task": evidence.task.value,
        "status": status,
        "passed": passed,
        "fixed_auebc": fixed,
        "learned_auebc": learned,
        "oracle_auebc": oracle,
        "oracle_gap_closure": closure,
        "bootstrap": evidence.baseline_minus_learned.distribution_sha256,
        "no_target_leakage": evidence.no_target_leakage,
        "deterministic_replay": evidence.deterministic_replay,
    }
    return PolicyGateResult(
        task=evidence.task,
        status=status,
        passed=passed,
        oracle_gap_closure=closure,
        baseline_effect=effect,
        ci_lower=evidence.baseline_minus_learned.ci_lower,
        improved_domains=evidence.baseline_minus_learned.improved_domains,
        no_target_leakage=evidence.no_target_leakage,
        deterministic_replay=evidence.deterministic_replay,
        state_sha256=_json_sha(payload),
    )


@dataclass(frozen=True, slots=True)
class TaskConditioningGateResult:
    status: str
    passed: bool
    state_sha256: str


def evaluate_task_conditioning_gate(
    *,
    field_wrong_minus_correct: G1PairedBootstrap,
    field_no_task_minus_correct: G1PairedBootstrap,
    cai_wrong_minus_correct: G1PairedBootstrap,
    cai_no_task_minus_correct: G1PairedBootstrap,
) -> TaskConditioningGateResult:
    values = (
        field_wrong_minus_correct,
        field_no_task_minus_correct,
        cai_wrong_minus_correct,
        cai_no_task_minus_correct,
    )
    for value in values:
        _validate_bootstrap(value)
    passed = all(
        value.ci_lower > 0.0
        and value.improved_domains >= POLICY_IMPROVED_DOMAINS_MINIMUM
        for value in values
    )
    status = "G1_TASK_CONDITIONING_GO" if passed else "G1_TASK_CONDITIONING_NO_GO"
    payload = {
        "schema": 1,
        "kind": "g1-task-conditioning-gate",
        "status": status,
        "bootstrap": tuple(value.distribution_sha256 for value in values),
    }
    return TaskConditioningGateResult(
        status=status,
        passed=passed,
        state_sha256=_json_sha(payload),
    )


@dataclass(frozen=True, slots=True)
class StopGateEvidence:
    task: InspectionTask
    normalized_measurement_saving: float
    task_loss_ratio: float
    premature_stop_rate: float
    saving_bootstrap: G1PairedBootstrap


@dataclass(frozen=True, slots=True)
class StopGateResult:
    task: InspectionTask
    status: str
    passed: bool
    state_sha256: str


def evaluate_stop_gate(evidence: StopGateEvidence) -> StopGateResult:
    if type(evidence) is not StopGateEvidence:
        raise G1GateError("issued STOP-gate evidence is required")
    _validate_bootstrap(evidence.saving_bootstrap)
    saving = float(evidence.normalized_measurement_saving)
    ratio = float(evidence.task_loss_ratio)
    premature = float(evidence.premature_stop_rate)
    if (
        evidence.task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or not math.isfinite(saving)
        or not 0.0 <= saving <= 1.0
        or not math.isfinite(ratio)
        or ratio < 0.0
        or not math.isfinite(premature)
        or not 0.0 <= premature <= 1.0
        or not math.isclose(
            saving,
            evidence.saving_bootstrap.point_estimate,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
    ):
        raise G1GateError("STOP-gate evidence is invalid")
    passed = (
        saving >= STOPPING_SAVING_MINIMUM
        and ratio <= STOPPING_TASK_LOSS_RATIO_MAXIMUM
        and premature <= STOPPING_PREMATURE_RATE_MAXIMUM
        and evidence.saving_bootstrap.improved_domains
        >= POLICY_IMPROVED_DOMAINS_MINIMUM
        and evidence.saving_bootstrap.ci_lower > 0.0
    )
    status = (
        f"G1_{evidence.task.value}_STOPPING_GO"
        if passed
        else f"G1_{evidence.task.value}_STOPPING_NO_GO"
    )
    payload = {
        "schema": 1,
        "kind": "g1-stop-gate",
        "task": evidence.task.value,
        "status": status,
        "saving": saving,
        "task_loss_ratio": ratio,
        "premature_stop_rate": premature,
        "bootstrap": evidence.saving_bootstrap.distribution_sha256,
    }
    return StopGateResult(
        task=evidence.task,
        status=status,
        passed=passed,
        state_sha256=_json_sha(payload),
    )


@dataclass(frozen=True, slots=True)
class G1FinalDecision:
    status: str
    g2_authorized: bool
    field_policy_go: bool
    cai_policy_go: bool
    task_conditioning_go: bool
    deployment_bridge_valid: bool
    state_sha256: str


def evaluate_final_g1_decision(
    *,
    deployment_bridge_valid: bool,
    field_policy_go: bool,
    cai_policy_go: bool,
    task_conditioning_go: bool,
) -> G1FinalDecision:
    values = (
        deployment_bridge_valid,
        field_policy_go,
        cai_policy_go,
        task_conditioning_go,
    )
    if any(type(value) is not bool for value in values):
        raise G1GateError("final G1 decision flags must be boolean")
    if not deployment_bridge_valid:
        status = "G1_DEPLOYMENT_BRIDGE_NO_GO"
    elif field_policy_go and cai_policy_go and task_conditioning_go:
        status = "G1_TASK_CONDITIONED_POLICY_GO"
    elif field_policy_go and cai_policy_go:
        status = "G1_ACTIVE_POLICY_GO"
    elif field_policy_go:
        status = "G1_FIELD_ONLY_POLICY_GO"
    elif cai_policy_go:
        status = "G1_CAI_ONLY_POLICY_GO"
    else:
        status = "G1_POLICY_OBSERVABILITY_NO_GO"
    payload = {
        "schema": 1,
        "kind": "g1-final-decision",
        "status": status,
        "deployment_bridge_valid": deployment_bridge_valid,
        "field_policy_go": field_policy_go,
        "cai_policy_go": cai_policy_go,
        "task_conditioning_go": task_conditioning_go,
    }
    return G1FinalDecision(
        status=status,
        g2_authorized=status in G2_AUTHORIZING_STATUSES,
        field_policy_go=field_policy_go,
        cai_policy_go=cai_policy_go,
        task_conditioning_go=task_conditioning_go,
        deployment_bridge_valid=deployment_bridge_valid,
        state_sha256=_json_sha(payload),
    )


__all__ = [
    "FINAL_G1_STATUSES",
    "G2_AUTHORIZING_STATUSES",
    "POLICY_GAP_CLOSURE_MINIMUM",
    "POLICY_IMPROVED_DOMAINS_MINIMUM",
    "STOPPING_PREMATURE_RATE_MAXIMUM",
    "STOPPING_SAVING_MINIMUM",
    "STOPPING_TASK_LOSS_RATIO_MAXIMUM",
    "G1ExecutionError",
    "G1FinalDecision",
    "G1FinalDependencies",
    "G1GateError",
    "G1Protocol",
    "G1Runtime",
    "G1RuntimeSurface",
    "G1SourceDependencies",
    "G1TeacherBankBuild",
    "PolicyGateEvidence",
    "PolicyGateResult",
    "StopGateEvidence",
    "StopGateResult",
    "TaskConditioningGateResult",
    "build_g1_all_source_teacher_banks",
    "build_g1_final_dependencies",
    "build_g1_source_dependencies",
    "build_g1_source_teacher_bank",
    "build_g1_world",
    "evaluate_final_g1_decision",
    "evaluate_policy_gate",
    "evaluate_stop_gate",
    "evaluate_task_conditioning_gate",
    "load_g1_encoder",
    "load_g1_protocol",
    "load_g1_runtime",
    "materialize_g1_source_teacher_records",
    "source_teacher_bank_path",
    "specimen_integrity_sha256",
]
