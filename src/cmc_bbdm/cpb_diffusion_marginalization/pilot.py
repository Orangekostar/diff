"""Registered D8 pilot evaluation and orchestration primitives."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.ndimage import gaussian_filter

from cmc_bbdm.cpb_diffusion_reconstruction.reconstruction import (
    build_learning_target,
    build_sparse_observation,
)
from cmc_bbdm.cpb_physical_descriptors import (
    PhysicalCalibration,
    load_physical_calibrations,
)
from cmc_bbdm.cpb_spatial.pipeline import load_registered_inputs
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import V3Data, validate_issued_data_authority
from cmc_bbdm.cpb_v3.morphology import REGISTERED_EXTRACTION_RULE

from .authority import D8InnerFold, issue_search_view, validate_inner_fold
from .config import DOMAIN_ORDER, D8Config
from .decomposition import (
    decompose_residual,
    gaussian_control,
    phase_randomized_control,
)
from .features import create_d8_frozen_encoder
from .regression import CandidatePrediction, fit_marginalized_candidate
from .residuals import P6ResidualBank, ResidualRecord, validate_residual_bank
from .search import D8Candidate, InnerEvaluation, SearchResult, run_outer_search
from .selection import (
    EnsembleResult,
    FinalistResult,
    FrozenOuterSelection,
    RerankResult,
    evaluate_finalists,
    fit_nonnegative_ensemble,
    freeze_outer_selection,
    rerank_candidates,
)
from .variants import VariantRecord, build_variant_batch

_FIELD_SHAPE = (3, 64, 64)
_DIFFUSION_CONTROLS = frozenset({"B5", "B6", "B7", "B8"})
_REGISTERED_VARIANT_WORKERS = 8


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class D8PilotStudyEvidence:
    """One pre-outer study's evidence for the frozen escalation decision."""

    outer_domain: str
    baseline_candidate_sha256: str
    diffusion_candidate_sha256: str
    baseline_objective: float
    diffusion_objective: float
    improved_inner_domains: tuple[str, ...]
    low_band_energy_fraction: float
    maximum_alpha_point_one_acceptance: float
    selected_diffusion_weight: float
    selection_state_sha256: str
    residual_bank_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        inner_domains = tuple(
            domain for domain in DOMAIN_ORDER if domain != self.outer_domain
        )
        if (
            self.outer_domain not in DOMAIN_ORDER
            or not _valid_sha256(self.baseline_candidate_sha256)
            or not _valid_sha256(self.diffusion_candidate_sha256)
            or not _valid_sha256(self.selection_state_sha256)
            or not _valid_sha256(self.residual_bank_sha256)
            or type(self.baseline_objective) is not float
            or type(self.diffusion_objective) is not float
            or not math.isfinite(self.baseline_objective)
            or not math.isfinite(self.diffusion_objective)
            or self.baseline_objective < 0.0
            or self.diffusion_objective < 0.0
            or type(self.improved_inner_domains) is not tuple
            or len(set(self.improved_inner_domains)) != len(
                self.improved_inner_domains
            )
            or any(domain not in inner_domains for domain in self.improved_inner_domains)
        ):
            raise ValueError("pilot study evidence authority changed")
        for value in (
            self.low_band_energy_fraction,
            self.maximum_alpha_point_one_acceptance,
            self.selected_diffusion_weight,
        ):
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("pilot study fraction is invalid")
        payload = {
            "outer_domain": self.outer_domain,
            "baseline_candidate_sha256": self.baseline_candidate_sha256,
            "diffusion_candidate_sha256": self.diffusion_candidate_sha256,
            "baseline_objective": self.baseline_objective,
            "diffusion_objective": self.diffusion_objective,
            "improved_inner_domains": self.improved_inner_domains,
            "low_band_energy_fraction": self.low_band_energy_fraction,
            "maximum_alpha_point_one_acceptance": (
                self.maximum_alpha_point_one_acceptance
            ),
            "selected_diffusion_weight": self.selected_diffusion_weight,
            "selection_state_sha256": self.selection_state_sha256,
            "residual_bank_sha256": self.residual_bank_sha256,
        }
        object.__setattr__(self, "state_sha256", _canonical_sha256(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_domain": self.outer_domain,
            "baseline_candidate_sha256": self.baseline_candidate_sha256,
            "diffusion_candidate_sha256": self.diffusion_candidate_sha256,
            "baseline_objective": self.baseline_objective,
            "diffusion_objective": self.diffusion_objective,
            "improved_inner_domains": list(self.improved_inner_domains),
            "low_band_energy_fraction": self.low_band_energy_fraction,
            "maximum_alpha_point_one_acceptance": (
                self.maximum_alpha_point_one_acceptance
            ),
            "selected_diffusion_weight": self.selected_diffusion_weight,
            "selection_state_sha256": self.selection_state_sha256,
            "residual_bank_sha256": self.residual_bank_sha256,
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True, slots=True)
class D8PilotDecision:
    """Frozen aggregate pilot branch before any outer evaluation is issued."""

    config_sha256: str
    residual_bank_sha256: str
    studies: tuple[D8PilotStudyEvidence, ...]
    trend_outer_studies: tuple[str, ...]
    mismatch_outer_studies: tuple[str, ...]
    freeze_outer_studies: tuple[str, ...]
    decision: str
    state_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "scope": "d8_pilot_escalation_evidence",
            "config_sha256": self.config_sha256,
            "residual_bank_sha256": self.residual_bank_sha256,
            "studies": [study.to_payload() for study in self.studies],
            "trend_outer_studies": list(self.trend_outer_studies),
            "mismatch_outer_studies": list(self.mismatch_outer_studies),
            "freeze_outer_studies": list(self.freeze_outer_studies),
            "decision": self.decision,
            "state_sha256": self.state_sha256,
        }


def decide_pilot_escalation(
    studies: tuple[D8PilotStudyEvidence, ...], *, config: D8Config
) -> D8PilotDecision:
    """Apply the preregistered TRAIN, FREEZE, CLOSE priority without outer data."""

    if (
        type(config) is not D8Config
        or type(studies) is not tuple
        or len(studies) != len(DOMAIN_ORDER)
        or tuple(study.outer_domain for study in studies) != DOMAIN_ORDER
        or any(type(study) is not D8PilotStudyEvidence for study in studies)
    ):
        raise TypeError("registered pilot study evidence is required")
    residual_states = {study.residual_bank_sha256 for study in studies}
    if len(residual_states) != 1:
        raise ValueError("pilot residual bank authority changed")
    residual_state = next(iter(residual_states))
    rule = config.escalation
    trend = tuple(
        study.outer_domain
        for study in studies
        if len(study.improved_inner_domains)
        >= int(rule["p6_candidate_minimum_inner_domains"])
    )
    mismatch = tuple(
        study.outer_domain
        for study in studies
        if study.low_band_energy_fraction >= float(rule["low_band_energy_fraction"])
        or study.maximum_alpha_point_one_acceptance
        < float(rule["low_acceptance_threshold"])
    )
    freeze = tuple(
        study.outer_domain
        for study in studies
        if study.baseline_objective - study.diffusion_objective
        >= float(rule["pilot_freeze_minimum_objective_gain"])
        and study.selected_diffusion_weight
        >= float(rule["pilot_freeze_minimum_diffusion_weight"])
    )
    priority = tuple(rule["decision_priority"])
    if (
        len(trend) >= int(rule["p6_candidate_minimum_outer_studies"])
        or len(mismatch) >= int(rule["mismatch_minimum_outer_studies"])
    ):
        decision = priority[0]
    elif len(freeze) >= int(rule["pilot_freeze_minimum_outer_studies"]):
        decision = priority[1]
    else:
        decision = priority[2]
    payload = {
        "schema_version": 1,
        "scope": "d8_pilot_escalation_evidence",
        "config_sha256": config.config_sha256,
        "residual_bank_sha256": residual_state,
        "study_states": [study.state_sha256 for study in studies],
        "trend_outer_studies": trend,
        "mismatch_outer_studies": mismatch,
        "freeze_outer_studies": freeze,
        "decision": decision,
    }
    return D8PilotDecision(
        config_sha256=config.config_sha256,
        residual_bank_sha256=residual_state,
        studies=studies,
        trend_outer_studies=trend,
        mismatch_outer_studies=mismatch,
        freeze_outer_studies=freeze,
        decision=decision,
        state_sha256=_canonical_sha256(payload),
    )


def _pilot_number(value: object, *, label: str) -> float:
    if type(value) is bool:
        raise ValueError(f"{label} is invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is invalid")
    return result


def _pilot_json_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, str):
        raise TypeError(f"{label} is not a JSON mapping")
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(result, dict):
        raise TypeError(f"{label} is not a JSON mapping")
    return result


def _selected_diffusion_weight(
    selection: Mapping[str, object], *, outer_domain: str, config: D8Config
) -> tuple[float, str]:
    if (
        selection.get("outer_domain") != outer_domain
        or not _valid_sha256(selection.get("state_sha256"))
    ):
        raise ValueError("pilot selection authority changed")
    candidates_value = selection.get("selected_candidates")
    ensemble = selection.get("ensemble")
    if not isinstance(candidates_value, list) or not isinstance(ensemble, Mapping):
        raise TypeError("pilot selection evidence is incomplete")
    candidates = tuple(D8Candidate.from_payload(value) for value in candidates_value)
    if any(candidate.config_sha256 != config.config_sha256 for candidate in candidates):
        raise ValueError("pilot selection config changed")
    candidate_sha256 = ensemble.get("candidate_sha256")
    weights_value = ensemble.get("weights")
    if (
        candidate_sha256 != [candidate.state_sha256 for candidate in candidates]
        or not isinstance(weights_value, list)
        or len(weights_value) != len(candidates)
    ):
        raise ValueError("pilot ensemble candidate roster changed")
    weights = np.asarray(weights_value, dtype=np.float64)
    if (
        weights.shape != (len(candidates),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1.0e-12)
    ):
        raise ValueError("pilot ensemble weights changed")
    diffusion_weight = math.fsum(
        float(weight)
        for candidate, weight in zip(candidates, weights, strict=True)
        if candidate.control_id in _DIFFUSION_CONTROLS
    )
    return diffusion_weight, str(selection["state_sha256"])


def _low_band_energy_fraction(
    bank: P6ResidualBank,
    *,
    outer_domain: str,
    family: str,
    band: str,
    parameters: dict[str, object],
) -> float:
    decomposition_parameters = dict(parameters)
    decomposition_parameters["band"] = band
    low_energy = 0.0
    source_energy = 0.0
    record_count = 0
    for record in bank.records:
        if record.dataset_id == outer_domain:
            continue
        bands = decompose_residual(
            record.residual_64,
            family=family,
            parameters=decomposition_parameters,
        )
        low_energy += float(np.sum(np.square(bands.low, dtype=np.float64)))
        source_energy += float(
            np.sum(np.square(record.residual_64, dtype=np.float64))
        )
        record_count += 1
    if record_count == 0 or source_energy <= 0.0:
        raise ValueError("pilot residual energy authority is empty")
    fraction = low_energy / source_energy
    if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0 + 1.0e-12:
        raise ValueError("pilot residual energy fraction is invalid")
    return min(fraction, 1.0)


def build_pilot_escalation_evidence(
    trial_rows: tuple[Mapping[str, str], ...],
    *,
    selections: tuple[Mapping[str, object], ...],
    bank: P6ResidualBank,
    config: D8Config,
) -> D8PilotDecision:
    """Recompute the frozen branch decision from trials, selections, and P6 draws."""

    if (
        type(config) is not D8Config
        or type(bank) is not P6ResidualBank
        or type(trial_rows) is not tuple
        or type(selections) is not tuple
        or len(selections) != len(DOMAIN_ORDER)
        or any(not isinstance(row, Mapping) for row in trial_rows)
        or any(not isinstance(selection, Mapping) for selection in selections)
    ):
        raise TypeError("registered pilot escalation authorities are required")
    threshold = float(config.escalation["p6_candidate_minimum_inner_mae_improvement"])
    studies: list[D8PilotStudyEvidence] = []
    for outer_domain, selection in zip(DOMAIN_ORDER, selections, strict=True):
        rows = tuple(row for row in trial_rows if row.get("outer_fold") == outer_domain)
        if not rows:
            raise ValueError("pilot trial outer-fold roster is incomplete")
        complete = tuple(row for row in rows if row.get("state") == "COMPLETE")
        baseline_rows = tuple(row for row in complete if row.get("control_id") == "B0")
        diffusion_rows = tuple(
            row for row in complete if row.get("control_id") in _DIFFUSION_CONTROLS
        )
        if not baseline_rows or not diffusion_rows:
            raise ValueError("pilot B0 or diffusion trial evidence is missing")
        baseline = min(
            baseline_rows,
            key=lambda row: (
                _pilot_number(row.get("objective"), label="baseline objective"),
                str(row.get("candidate_sha256")),
            ),
        )
        diffusion = min(
            diffusion_rows,
            key=lambda row: (
                _pilot_number(row.get("objective"), label="diffusion objective"),
                str(row.get("candidate_sha256")),
            ),
        )
        if not _valid_sha256(baseline.get("candidate_sha256")) or not _valid_sha256(
            diffusion.get("candidate_sha256")
        ):
            raise ValueError("pilot candidate identity is invalid")
        inner_domains = tuple(domain for domain in DOMAIN_ORDER if domain != outer_domain)
        improved: list[str] = []
        for domain in inner_domains:
            baseline_mae = _pilot_number(
                baseline.get(f"inner_mae__{domain}"), label="baseline inner MAE"
            )
            diffusion_mae = _pilot_number(
                diffusion.get(f"inner_mae__{domain}"), label="diffusion inner MAE"
            )
            if baseline_mae > 0.0 and baseline_mae - diffusion_mae >= threshold * baseline_mae:
                improved.append(domain)
        alpha_rows = tuple(
            row
            for row in rows
            if row.get("control_id") in _DIFFUSION_CONTROLS
            and row.get("state") in {"COMPLETE", "PRUNED"}
            and _pilot_number(row.get("alpha"), label="pilot alpha")
            == float(config.escalation["low_acceptance_alpha"])
            and row.get("acceptance_rate") not in {None, ""}
        )
        if not alpha_rows:
            raise ValueError("pilot alpha=0.1 acceptance evidence is missing")
        maximum_acceptance = max(
            _pilot_number(row.get("acceptance_rate"), label="pilot acceptance rate")
            for row in alpha_rows
        )
        if not 0.0 <= maximum_acceptance <= 1.0:
            raise ValueError("pilot acceptance rate is invalid")
        parameters = _pilot_json_mapping(
            diffusion.get("decomposition_parameters"),
            label="pilot decomposition parameters",
        )
        diffusion_weight, selection_state = _selected_diffusion_weight(
            selection, outer_domain=outer_domain, config=config
        )
        studies.append(
            D8PilotStudyEvidence(
                outer_domain=outer_domain,
                baseline_candidate_sha256=str(baseline["candidate_sha256"]),
                diffusion_candidate_sha256=str(diffusion["candidate_sha256"]),
                baseline_objective=_pilot_number(
                    baseline.get("objective"), label="baseline objective"
                ),
                diffusion_objective=_pilot_number(
                    diffusion.get("objective"), label="diffusion objective"
                ),
                improved_inner_domains=tuple(improved),
                low_band_energy_fraction=_low_band_energy_fraction(
                    bank,
                    outer_domain=outer_domain,
                    family=str(diffusion.get("decomposition_family")),
                    band=str(diffusion.get("band")),
                    parameters=parameters,
                ),
                maximum_alpha_point_one_acceptance=maximum_acceptance,
                selected_diffusion_weight=diffusion_weight,
                selection_state_sha256=selection_state,
                residual_bank_sha256=bank.state_sha256,
            )
        )
    return decide_pilot_escalation(tuple(studies), config=config)


def _readonly_field(value: object, *, label: str, bounded: bool = False) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if array.shape != _FIELD_SHAPE or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite (3, 64, 64) field")
    if bounded and (np.min(array) < -1.0 or np.max(array) > 1.0):
        raise ValueError(f"{label} must lie in [-1, 1]")
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float32).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _seed(candidate: D8Candidate, specimen_id: str, index: int) -> int:
    digest = hashlib.sha256(
        candidate.state_sha256.encode("ascii")
        + b"\0"
        + specimen_id.encode("utf-8")
        + b"\0"
        + str(index).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _state(
    candidate: D8Candidate,
    *,
    specimen_id: str,
    dataset_id: str,
    origins: tuple[tuple[str, str], ...],
    residuals: tuple[np.ndarray, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "candidate_sha256": candidate.state_sha256,
                "specimen_id": specimen_id,
                "dataset_id": dataset_id,
                "origins": origins,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for value in residuals:
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ResidualProposalSet:
    """One specimen's ordered proposal residuals and donor identities."""

    control_id: str
    specimen_id: str
    dataset_id: str
    residuals: tuple[np.ndarray, ...]
    origin_specimen_ids: tuple[str, ...]
    origin_dataset_ids: tuple[str, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class RegisteredPilotAssets:
    """Registered measured fields, native images, and physical calibration roster."""

    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    measured_fields: np.ndarray
    native_images: tuple[np.ndarray, ...]
    source_sha256: tuple[str, ...]
    calibrations: Mapping[str, PhysicalCalibration]
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.specimen_ids) is not tuple
            or not self.specimen_ids
            or len(set(self.specimen_ids)) != len(self.specimen_ids)
            or type(self.dataset_ids) is not tuple
            or len(self.dataset_ids) != len(self.specimen_ids)
            or any(type(value) is not str or not value for value in self.specimen_ids)
            or any(type(value) is not str or not value for value in self.dataset_ids)
        ):
            raise ValueError("registered pilot asset identities are invalid")
        rows = len(self.specimen_ids)
        fields = _readonly_numeric(
            self.measured_fields,
            dtype=np.dtype(np.float32),
            label="registered measured fields",
        )
        if fields.shape != (rows, 3, 64, 64) or np.any(
            (fields < -1.0) | (fields > 1.0)
        ):
            raise ValueError("registered measured field roster is invalid")
        if type(self.native_images) is not tuple or len(self.native_images) != rows:
            raise ValueError("registered native image roster is invalid")
        images: list[np.ndarray] = []
        for image in self.native_images:
            if (
                type(image) is not np.ndarray
                or image.dtype != np.dtype(np.uint8)
                or image.ndim != 3
                or image.shape[2] != 3
                or min(image.shape[:2]) < 5
            ):
                raise ValueError("registered native image is invalid")
            contiguous = np.ascontiguousarray(image, dtype=np.uint8)
            frozen = np.frombuffer(
                contiguous.tobytes(order="C"), dtype=np.uint8
            ).reshape(contiguous.shape)
            frozen.setflags(write=False)
            images.append(frozen)
        if (
            type(self.source_sha256) is not tuple
            or len(self.source_sha256) != rows
            or any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.source_sha256
            )
        ):
            raise ValueError("registered source hash roster is invalid")
        if not isinstance(self.calibrations, Mapping):
            raise TypeError("registered calibration mapping is invalid")
        calibration_values = dict(self.calibrations)
        if set(self.dataset_ids) != set(calibration_values) or any(
            type(value) is not PhysicalCalibration
            or key != value.dataset_id
            for key, value in calibration_values.items()
        ):
            raise ValueError("registered calibration mapping does not match the data")
        payload = {
            "specimen_ids": self.specimen_ids,
            "dataset_ids": self.dataset_ids,
            "field_sha256": _array_digest(fields),
            "image_sha256": [
                hashlib.sha256(value.tobytes(order="C")).hexdigest()
                for value in images
            ],
            "source_sha256": self.source_sha256,
            "calibrations": {
                key: asdict(value) for key, value in sorted(calibration_values.items())
            },
        }
        object.__setattr__(self, "measured_fields", fields)
        object.__setattr__(self, "native_images", tuple(images))
        object.__setattr__(
            self, "calibrations", MappingProxyType(calibration_values)
        )
        object.__setattr__(
            self,
            "state_sha256",
            hashlib.sha256(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode("ascii")
            ).hexdigest(),
        )


def load_registered_pilot_assets(
    data: object,
    *,
    config: D8Config,
    project_root: str | Path,
) -> RegisteredPilotAssets:
    """Load the exact V3 images and P6-compatible measured fields once."""

    if type(data) is not V3Data or type(config) is not D8Config:
        raise TypeError("registered V3Data and D8Config are required")
    validate_issued_data_authority(data)
    root = Path(project_root).resolve(strict=True)
    specimen_ids = tuple(str(value) for value in data.sample_ids.tolist())
    dataset_ids = tuple(str(value) for value in data.dataset_ids.tolist())
    base_config = load_v3_config(
        root / config.sources["p1_config"].path,
        project_root=root,
    )
    calibration_source = base_config.sources["physical_calibration"]
    registered_calibrations = load_physical_calibrations(
        root / calibration_source.path,
        project_root=root,
        expected_sha256=calibration_source.sha256,
    )
    try:
        calibrations = {
            domain: registered_calibrations[domain]
            for domain in sorted(set(dataset_ids))
        }
    except KeyError as error:
        raise ValueError("registered calibration source does not cover D8 data") from error
    inputs = load_registered_inputs(
        data, project_root=root, calibrations=calibrations
    )
    if inputs.specimen_ids != specimen_ids or inputs.dataset_ids != dataset_ids:
        raise ValueError("registered pilot input roster differs from V3 authority")
    measured: list[np.ndarray] = []
    source_hashes: list[str] = []
    for specimen_id, dataset_id, image in zip(
        specimen_ids, dataset_ids, inputs.images, strict=True
    ):
        observation = build_sparse_observation(
            image, specimen_id=specimen_id, dataset_id=dataset_id
        )
        measured.append(build_learning_target(image, observation))
        source_hashes.append(observation.source_sha256)
    return RegisteredPilotAssets(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        measured_fields=np.stack(measured),
        native_images=inputs.images,
        source_sha256=tuple(source_hashes),
        calibrations=calibrations,
    )


def _readonly_numeric(value: object, *, dtype: np.dtype, label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _array_digest(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_counts(value: object, *, label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("proposal counts must be integers")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("proposal counts must be integers") from error
    if array.dtype.kind not in "iu" or not np.all(np.isfinite(array)):
        raise ValueError("proposal counts must be integers")
    return _readonly_numeric(array, dtype=np.dtype(np.int64), label=label)


@dataclass(frozen=True, slots=True)
class D8FeatureBundle:
    """Immutable candidate features and gate counts for one search view."""

    candidate_sha256: str
    search_view_sha256: str
    specimen_ids: tuple[str, ...]
    train_variant_features: np.ndarray
    query_variant_features: np.ndarray
    morphology_distances: np.ndarray
    accepted_proposals: np.ndarray
    proposed_variants: np.ndarray
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (self.candidate_sha256, self.search_view_sha256)
        ):
            raise ValueError("feature bundle authority hash is invalid")
        if (
            type(self.specimen_ids) is not tuple
            or not self.specimen_ids
            or len(set(self.specimen_ids)) != len(self.specimen_ids)
            or any(type(value) is not str or not value for value in self.specimen_ids)
        ):
            raise ValueError("feature bundle specimen roster is invalid")
        rows = len(self.specimen_ids)
        train = _readonly_numeric(
            self.train_variant_features,
            dtype=np.dtype(np.float64),
            label="train variant features",
        )
        query = _readonly_numeric(
            self.query_variant_features,
            dtype=np.dtype(np.float64),
            label="query variant features",
        )
        if (
            train.ndim != 3
            or query.ndim != 3
            or train.shape[0] != rows
            or query.shape[0] != rows
            or train.shape[1] not in (1, 2, 4, 8, 16)
            or query.shape[1] not in (1, 2, 4, 8, 16)
            or train.shape[2] != query.shape[2]
        ):
            raise ValueError("feature bundle variant matrices are not aligned")
        distances = _readonly_numeric(
            self.morphology_distances,
            dtype=np.dtype(np.float64),
            label="morphology distances",
        )
        if distances.shape != query.shape[:2] or np.any(distances < 0.0):
            raise ValueError("feature bundle morphology distances are not aligned")
        accepted = _readonly_counts(
            self.accepted_proposals,
            label="accepted proposals",
        )
        proposed = _readonly_counts(
            self.proposed_variants,
            label="proposed variants",
        )
        if (
            accepted.shape != (rows,)
            or proposed.shape != (rows,)
            or np.any(proposed < 1)
            or np.any(accepted < 0)
            or np.any(accepted > proposed)
        ):
            raise ValueError("feature bundle proposal counts are invalid")
        payload = {
            "candidate_sha256": self.candidate_sha256,
            "search_view_sha256": self.search_view_sha256,
            "specimen_ids": self.specimen_ids,
            "arrays": [
                _array_digest(value)
                for value in (train, query, distances, accepted, proposed)
            ],
        }
        object.__setattr__(self, "train_variant_features", train)
        object.__setattr__(self, "query_variant_features", query)
        object.__setattr__(self, "morphology_distances", distances)
        object.__setattr__(self, "accepted_proposals", accepted)
        object.__setattr__(self, "proposed_variants", proposed)
        object.__setattr__(
            self,
            "state_sha256",
            hashlib.sha256(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode("ascii")
            ).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class D8BundleEvaluation:
    """Search objective cell and its exact query prediction authority."""

    inner: InnerEvaluation
    prediction: CandidatePrediction


@dataclass(frozen=True, slots=True)
class D8OuterPilotRun:
    """One completed pre-outer search, rerank, and frozen selection chain."""

    outer_domain: str
    search_view_sha256: str
    search: SearchResult
    rerank: RerankResult
    finalists: FinalistResult
    ensemble: EnsembleResult
    selection: FrozenOuterSelection
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.outer_domain not in DOMAIN_ORDER
            or type(self.search) is not SearchResult
            or type(self.rerank) is not RerankResult
            or type(self.finalists) is not FinalistResult
            or type(self.ensemble) is not EnsembleResult
            or type(self.selection) is not FrozenOuterSelection
        ):
            raise TypeError("outer pilot result types are invalid")
        authorities = {
            self.outer_domain,
            self.search.outer_domain,
            self.rerank.outer_domain,
            self.finalists.outer_domain,
            self.selection.outer_domain,
        }
        if len(authorities) != 1 or self.selection.outer_evaluation_started:
            raise ValueError("outer pilot result authority changed")
        if (
            self.rerank.search_view_sha256 != self.search_view_sha256
            or self.finalists.search_view_sha256 != self.search_view_sha256
            or self.selection.search_view_sha256 != self.search_view_sha256
            or len(self.search.selected_candidates) != 12
            or len(self.rerank.rows) != 12
            or len(self.rerank.finalists) != 4
            or len(self.finalists.selected) != 4
        ):
            raise ValueError("outer pilot result roster changed")
        selected = tuple(
            item.candidate.state_sha256 for item in self.finalists.selected
        )
        if (
            selected != self.ensemble.candidate_sha256
            or selected != self.selection.selected_candidate_sha256
            or self.ensemble.state_sha256 != self.selection.ensemble_sha256
        ):
            raise ValueError("outer pilot selected candidates changed")
        payload = {
            "outer_domain": self.outer_domain,
            "search_view_sha256": self.search_view_sha256,
            "search": {
                "initial_trial_count": self.search.initial_trial_count,
                "trial_count": self.search.trial_count,
                "completed_count": self.search.completed_count,
                "pruned_count": self.search.pruned_count,
                "failed_count": self.search.failed_count,
                "candidate_sha256": [
                    item.state_sha256 for item in self.search.selected_candidates
                ],
            },
            "rerank_state_sha256": self.rerank.state_sha256,
            "finalist_state_sha256": self.finalists.state_sha256,
            "ensemble_state_sha256": self.ensemble.state_sha256,
            "selection_state_sha256": self.selection.state_sha256,
        }
        object.__setattr__(
            self,
            "state_sha256",
            hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
            ).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class D8PilotRunResult:
    """All six prospective selections before any outer evaluation is issued."""

    config_sha256: str
    residual_bank_sha256: str
    outer_runs: tuple[D8OuterPilotRun, ...]
    outer_evaluation_count: int = 0
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        hashes = (self.config_sha256, self.residual_bank_sha256)
        if any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("pilot run authority hash is invalid")
        if (
            type(self.outer_runs) is not tuple
            or any(type(item) is not D8OuterPilotRun for item in self.outer_runs)
            or tuple(item.outer_domain for item in self.outer_runs) != DOMAIN_ORDER
            or type(self.outer_evaluation_count) is not int
            or self.outer_evaluation_count != 0
        ):
            raise ValueError("pilot run outer roster changed")
        payload = {
            "config_sha256": self.config_sha256,
            "residual_bank_sha256": self.residual_bank_sha256,
            "outer_run_sha256": [item.state_sha256 for item in self.outer_runs],
            "outer_evaluation_count": self.outer_evaluation_count,
        }
        object.__setattr__(
            self,
            "state_sha256",
            hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
            ).hexdigest(),
        )

    @property
    def outer_domains(self) -> tuple[str, ...]:
        return tuple(item.outer_domain for item in self.outer_runs)


class D8PilotEvaluator:
    """Reuse candidate encodings only where residual donor authority is invariant."""

    def __init__(
        self,
        bundle_builder: Callable[[D8Candidate, D8InnerFold], D8FeatureBundle],
    ) -> None:
        if not callable(bundle_builder):
            raise TypeError("pilot bundle builder must be callable")
        self._bundle_builder = bundle_builder
        self._candidate_token: tuple[str, str] | None = None
        self._bundles: dict[str, D8FeatureBundle] = {}

    def _bundle(self, candidate: D8Candidate, fold: D8InnerFold) -> D8FeatureBundle:
        if type(candidate) is not D8Candidate or type(fold) is not D8InnerFold:
            raise TypeError("exact candidate and inner fold are required")
        validate_inner_fold(fold)
        token = (candidate.state_sha256, fold.search_view.state_sha256)
        if token != self._candidate_token:
            self._candidate_token = token
            self._bundles.clear()
        key = fold.state_sha256 if candidate.control_id == "B4" else "fold-invariant"
        if key not in self._bundles:
            bundle = self._bundle_builder(candidate, fold)
            if type(bundle) is not D8FeatureBundle:
                raise TypeError("pilot bundle builder returned an invalid value")
            self._bundles[key] = bundle
        return self._bundles[key]

    def evaluate(self, candidate: D8Candidate, fold: D8InnerFold) -> InnerEvaluation:
        return evaluate_feature_bundle(
            candidate, fold=fold, bundle=self._bundle(candidate, fold)
        ).inner

    def predict(
        self, candidate: D8Candidate, fold: D8InnerFold
    ) -> CandidatePrediction:
        return evaluate_feature_bundle(
            candidate, fold=fold, bundle=self._bundle(candidate, fold)
        ).prediction


def evaluate_feature_bundle(
    candidate: D8Candidate,
    *,
    fold: D8InnerFold,
    bundle: D8FeatureBundle,
) -> D8BundleEvaluation:
    """Fit and score one bundle without reading any prospective outer response."""

    if type(candidate) is not D8Candidate or type(bundle) is not D8FeatureBundle:
        raise TypeError("exact D8 candidate and feature bundle are required")
    fold_state = validate_inner_fold(fold)
    if (
        candidate.config_sha256 != fold.search_view.config_sha256
        or bundle.candidate_sha256 != candidate.state_sha256
    ):
        raise ValueError("feature bundle candidate authority changed")
    if (
        bundle.search_view_sha256 != fold.search_view.state_sha256
        or bundle.specimen_ids != fold.search_view.specimen_ids
    ):
        raise ValueError("feature bundle search authority changed")
    if (
        bundle.train_variant_features.shape[1] != candidate.K_train
        or bundle.query_variant_features.shape[1] != candidate.K_test
    ):
        raise ValueError("feature bundle K differs from candidate")
    targets = np.asarray(fold.search_view.data_view.cai_ratio, dtype=np.float64)
    prediction = fit_marginalized_candidate(
        candidate.regressor_spec,
        inner_fold=fold,
        specimen_ids=bundle.specimen_ids,
        train_variant_features=bundle.train_variant_features,
        query_variant_features=bundle.query_variant_features,
        targets=targets,
        marginalization_stage=candidate.marginalization_stage,
        feature_aggregation=candidate.feature_aggregation,
        prediction_aggregation=candidate.prediction_aggregation,
        morphology_distances=(
            bundle.morphology_distances
            if candidate.marginalization_stage == "prediction"
            else None
        ),
        morphology_beta=candidate.morphology_beta,
        consistency=candidate.consistency,
        consistency_weight=candidate.consistency_weight,
    )
    query_indices = np.asarray(fold.query_indices, dtype=np.int64)
    mae = float(
        np.mean(
            np.abs(prediction.predictions - prediction.targets), dtype=np.float64
        )
    )
    accepted = int(np.sum(bundle.accepted_proposals[query_indices], dtype=np.int64))
    proposed = int(np.sum(bundle.proposed_variants[query_indices], dtype=np.int64))
    evidence = hashlib.sha256(
        candidate.state_sha256.encode("ascii")
        + fold_state.encode("ascii")
        + bundle.state_sha256.encode("ascii")
        + prediction.state_sha256.encode("ascii")
    ).hexdigest()
    if not math.isfinite(mae):
        raise ValueError("feature bundle produced a nonfinite MAE")
    return D8BundleEvaluation(
        inner=InnerEvaluation(
            query_domain=fold.query_domain,
            mae=mae,
            accepted_proposals=accepted,
            proposed_variants=proposed,
            evidence_sha256=evidence,
        ),
        prediction=prediction,
    )


def _specimen_records(
    bank: P6ResidualBank, *, specimen_id: str, dataset_id: str
) -> tuple[ResidualRecord, ...]:
    if type(bank) is not P6ResidualBank or not isinstance(bank.records, tuple):
        raise TypeError("exact P6ResidualBank is required")
    records = tuple(
        sorted(
            (
                record
                for record in bank.records
                if type(record) is ResidualRecord
                and record.specimen_id == specimen_id
                and record.dataset_id == dataset_id
            ),
            key=lambda record: record.draw_index,
        )
    )
    if (
        len(records) != bank.draw_count
        or tuple(record.draw_index for record in records)
        != tuple(range(bank.draw_count))
    ):
        raise ValueError("specimen P6 residual draw roster is incomplete")
    return records


def _decompose(candidate: D8Candidate, value: np.ndarray) -> np.ndarray:
    parameters = dict(candidate.decomposition_parameters)
    parameters["band"] = candidate.band
    return decompose_residual(
        value,
        family=candidate.decomposition_family,
        parameters=parameters,
    ).selected


def build_candidate_residual_proposals(
    candidate: D8Candidate,
    *,
    bank: P6ResidualBank,
    specimen_id: str,
    dataset_id: str,
    source: np.ndarray,
    fit_domains: tuple[str, ...],
) -> ResidualProposalSet:
    """Build the frozen B0--B8 residual proposal sequence for one specimen."""

    if type(candidate) is not D8Candidate:
        raise TypeError("exact D8Candidate is required")
    if (
        type(specimen_id) is not str
        or not specimen_id
        or type(dataset_id) is not str
        or not dataset_id
    ):
        raise ValueError("proposal specimen identity is invalid")
    measured = _readonly_field(source, label="measured source", bounded=True)
    own = _specimen_records(bank, specimen_id=specimen_id, dataset_id=dataset_id)
    origins: list[tuple[str, str]] = []
    values: list[np.ndarray] = []

    if candidate.control_id == "B0":
        values.append(_readonly_field(np.zeros_like(measured), label="raw residual"))
        origins.append((specimen_id, dataset_id))
    elif candidate.control_id == "B1":
        if candidate.decomposition_family != "gaussian" or candidate.band != "low":
            raise ValueError("B1 requires the registered Gaussian low-pass control")
        sigma = candidate.decomposition_parameters.get("sigma")
        if type(sigma) not in (int, float) or not 0.5 <= float(sigma) <= 8.0:
            raise ValueError("B1 Gaussian sigma is invalid")
        low = gaussian_filter(
            measured.astype(np.float64),
            sigma=(0.0, float(sigma), float(sigma)),
            mode="reflect",
        ).astype(np.float32)
        values.append(_readonly_field(low - measured, label="low-pass residual"))
        origins.append((specimen_id, dataset_id))
    elif candidate.control_id in {"B2", "B3"}:
        for index in range(32):
            record = own[index % len(own)]
            controlled = (
                gaussian_control(record.residual_64, seed=_seed(candidate, specimen_id, index))
                if candidate.control_id == "B2"
                else phase_randomized_control(
                    record.residual_64, seed=_seed(candidate, specimen_id, index)
                )
            )
            values.append(_decompose(candidate, controlled))
            origins.append((record.specimen_id, record.dataset_id))
    elif candidate.control_id == "B4":
        if (
            not isinstance(fit_domains, tuple)
            or not fit_domains
            or len(set(fit_domains)) != len(fit_domains)
            or any(type(value) is not str or not value for value in fit_domains)
        ):
            raise ValueError("B4 requires exact inner-fit domains")
        eligible = tuple(
            record
            for record in bank.records
            if type(record) is ResidualRecord
            and record.dataset_id in fit_domains
            and record.specimen_id != specimen_id
        )
        if not eligible:
            raise ValueError("B4 has no authorized empirical residual donor")
        for index in range(32):
            generator = np.random.Generator(
                np.random.PCG64(_seed(candidate, specimen_id, index))
            )
            record = eligible[int(generator.integers(0, len(eligible)))]
            values.append(_decompose(candidate, record.residual_64))
            origins.append((record.specimen_id, record.dataset_id))
    elif candidate.control_id in _DIFFUSION_CONTROLS:
        for record in own:
            values.append(_decompose(candidate, record.residual_64))
            origins.append((record.specimen_id, record.dataset_id))
    else:
        raise ValueError("D8 control is not registered")

    residuals = tuple(
        _readonly_field(value, label="proposal residual") for value in values
    )
    origin_values = tuple(origins)
    return ResidualProposalSet(
        control_id=candidate.control_id,
        specimen_id=specimen_id,
        dataset_id=dataset_id,
        residuals=residuals,
        origin_specimen_ids=tuple(item[0] for item in origin_values),
        origin_dataset_ids=tuple(item[1] for item in origin_values),
        state_sha256=_state(
            candidate,
            specimen_id=specimen_id,
            dataset_id=dataset_id,
            origins=origin_values,
            residuals=residuals,
        ),
    )


def _morphology_distance(record: VariantRecord, candidate: D8Candidate) -> float:
    thresholds = candidate.thresholds
    components = np.asarray(
        (
            record.area_deviation / thresholds.area_relative_deviation,
            record.width_deviation / thresholds.width_relative_deviation,
            record.height_deviation / thresholds.height_relative_deviation,
            record.centroid_shift_mm / thresholds.centroid_shift_mm,
            max(0.0, 1.0 - record.low_frequency_correlation)
            / max(1.0e-12, 1.0 - thresholds.low_frequency_correlation_minimum),
            max(0.0, 1.0 - record.radial_profile_correlation)
            / max(1.0e-12, 1.0 - thresholds.radial_spearman_minimum),
        ),
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(np.square(components), dtype=np.float64)))


def _build_registered_feature_bundle(
    candidate: D8Candidate,
    *,
    fold: D8InnerFold,
    assets: RegisteredPilotAssets,
    bank: P6ResidualBank,
    encoder: object,
) -> D8FeatureBundle:
    """Build and encode one candidate/view bundle from registered source assets."""

    if (
        type(candidate) is not D8Candidate
        or type(fold) is not D8InnerFold
        or type(assets) is not RegisteredPilotAssets
        or type(bank) is not P6ResidualBank
    ):
        raise TypeError("registered bundle inputs have the wrong type")
    validate_inner_fold(fold)
    if candidate.config_sha256 != fold.search_view.config_sha256:
        raise ValueError("candidate config differs from search authority")
    if not hasattr(encoder, "encode") or not callable(encoder.encode):
        raise TypeError("registered frozen encoder is required")
    positions = {value: index for index, value in enumerate(assets.specimen_ids)}
    if set(fold.search_view.specimen_ids) - set(positions):
        raise ValueError("registered pilot assets do not cover the search view")
    fit_domains = tuple(
        domain
        for domain in DOMAIN_ORDER
        if domain not in {fold.outer_domain, fold.query_domain}
    )
    if set(fold.fit_dataset_ids) != set(fit_domains):
        raise ValueError("registered inner-fit domain roster changed")
    maximum_k = max(candidate.K_train, candidate.K_test)
    def build_row(
        identity: tuple[str, str],
    ) -> tuple[tuple[np.ndarray, ...], tuple[float, ...], int, int]:
        specimen_id, dataset_id = identity
        position = positions[specimen_id]
        if assets.dataset_ids[position] != dataset_id:
            raise ValueError("registered pilot asset domain changed")
        calibration = assets.calibrations[dataset_id]
        proposals = build_candidate_residual_proposals(
            candidate,
            bank=bank,
            specimen_id=specimen_id,
            dataset_id=dataset_id,
            source=assets.measured_fields[position],
            fit_domains=fit_domains,
        )
        batch = build_variant_batch(
            assets.measured_fields[position],
            proposals.residuals,
            native_source=assets.native_images[position],
            alpha=candidate.alpha,
            requested_count=maximum_k,
            rule=REGISTERED_EXTRACTION_RULE,
            calibration=calibration,
            thresholds=candidate.thresholds,
        )
        if len(batch.encoder_images) != maximum_k:
            raise ValueError("variant batch did not produce the registered K")
        accepted_records = tuple(record for record in batch.records if record.accepted)
        row_distances = tuple(
            _morphology_distance(record, candidate)
            for record in accepted_records[:maximum_k]
        ) + tuple(0.0 for _ in range(batch.fallback_count))
        if len(row_distances) != maximum_k:
            raise ValueError("morphology distance roster differs from variant batch")
        return (
            batch.encoder_images,
            row_distances,
            batch.accepted_count,
            batch.proposal_count,
        )

    identities = tuple(
        zip(
            fold.search_view.specimen_ids,
            fold.search_view.dataset_ids,
            strict=True,
        )
    )
    with ThreadPoolExecutor(
        max_workers=_REGISTERED_VARIANT_WORKERS,
        thread_name_prefix="d8-variant",
    ) as workers:
        rows = tuple(workers.map(build_row, identities))
    image_grid = tuple(row[0] for row in rows)
    distances = tuple(row[1] for row in rows)
    accepted = tuple(row[2] for row in rows)
    proposed = tuple(row[3] for row in rows)
    encoded = np.asarray(
        encoder.encode(image_grid, layer=candidate.feature_layer),
        dtype=np.float64,
    )
    expected_prefix = (fold.search_view.specimen_count, maximum_k)
    if encoded.ndim != 3 or encoded.shape[:2] != expected_prefix:
        raise ValueError("frozen encoder returned a misaligned feature grid")
    return D8FeatureBundle(
        candidate_sha256=candidate.state_sha256,
        search_view_sha256=fold.search_view.state_sha256,
        specimen_ids=fold.search_view.specimen_ids,
        train_variant_features=encoded[:, : candidate.K_train],
        query_variant_features=encoded[:, : candidate.K_test],
        morphology_distances=np.asarray(distances, dtype=np.float64)[
            :, : candidate.K_test
        ],
        accepted_proposals=np.asarray(accepted, dtype=np.int64),
        proposed_variants=np.asarray(proposed, dtype=np.int64),
    )


def create_registered_pilot_evaluator(
    data: object,
    *,
    config: D8Config,
    bank: object,
    project_root: str | Path,
    device: str = "cuda:0",
) -> D8PilotEvaluator:
    """Bind the registered source assets, residual bank, and frozen encoder."""

    assets = load_registered_pilot_assets(
        data,
        config=config,
        project_root=project_root,
    )
    validate_residual_bank(
        bank,
        specimen_ids=assets.specimen_ids,
        dataset_ids=assets.dataset_ids,
        source_sha256=assets.source_sha256,
        draw_count=config.p6_draws,
    )
    encoder = create_d8_frozen_encoder(
        project_root=project_root,
        device=device,
    )

    def build(candidate: D8Candidate, fold: D8InnerFold) -> D8FeatureBundle:
        return _build_registered_feature_bundle(
            candidate,
            fold=fold,
            assets=assets,
            bank=bank,
            encoder=encoder,
        )

    return D8PilotEvaluator(build)


def run_registered_pilot(
    data: object,
    *,
    config: D8Config,
    bank: object,
    project_root: str | Path,
    output: str | Path,
    device: str = "cuda:0",
) -> D8PilotRunResult:
    """Execute all six registered pre-outer searches and freeze their selections."""

    if type(config) is not D8Config or tuple(config.outer_domains) != DOMAIN_ORDER:
        raise TypeError("registered D8Config is required")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    selection_root = root / "best_inner_configs"
    selection_root.mkdir(parents=True, exist_ok=True)
    evaluator = create_registered_pilot_evaluator(
        data,
        config=config,
        bank=bank,
        project_root=project_root,
        device=device,
    )
    outer_runs: list[D8OuterPilotRun] = []
    for outer_domain in DOMAIN_ORDER:
        view = issue_search_view(data, outer_domain=outer_domain, config=config)
        search = run_outer_search(
            view,
            config=config,
            output=root,
            evaluator=evaluator.evaluate,
        )
        reranked = rerank_candidates(
            search.selected_candidates,
            view=view,
            seeds=config.rerank_seeds,
            evaluator=evaluator.predict,
        )
        finalist_candidates = tuple(
            item.candidate for item in reranked.finalists
        )
        finalists = evaluate_finalists(
            finalist_candidates,
            view=view,
            seeds=config.rerank_seeds,
            K_test_values=(8, 16),
            evaluator=evaluator.predict,
        )
        candidate_sha256 = tuple(
            item.candidate.state_sha256 for item in finalists.selected
        )
        ensemble = fit_nonnegative_ensemble(
            np.vstack([item.oof_predictions for item in finalists.selected]),
            finalists.selected[0].oof_targets,
            specimen_ids=view.specimen_ids,
            domain_ids=view.dataset_ids,
            candidate_sha256=candidate_sha256,
            minimum_j_gain=0.0001,
        )
        selection = freeze_outer_selection(
            reranked,
            finalists=finalists,
            ensemble=ensemble,
            view=view,
            output=selection_root / f"{outer_domain}.json",
        )
        outer_runs.append(
            D8OuterPilotRun(
                outer_domain=outer_domain,
                search_view_sha256=view.state_sha256,
                search=search,
                rerank=reranked,
                finalists=finalists,
                ensemble=ensemble,
                selection=selection,
            )
        )
    residual_state = getattr(bank, "state_sha256", None)
    if type(residual_state) is not str:
        raise TypeError("registered residual bank state is required")
    return D8PilotRunResult(
        config_sha256=config.config_sha256,
        residual_bank_sha256=residual_state,
        outer_runs=tuple(outer_runs),
    )


__all__ = [
    "D8BundleEvaluation",
    "D8FeatureBundle",
    "D8OuterPilotRun",
    "D8PilotDecision",
    "D8PilotEvaluator",
    "D8PilotRunResult",
    "D8PilotStudyEvidence",
    "RegisteredPilotAssets",
    "ResidualProposalSet",
    "build_candidate_residual_proposals",
    "build_pilot_escalation_evidence",
    "create_registered_pilot_evaluator",
    "decide_pilot_escalation",
    "evaluate_feature_bundle",
    "load_registered_pilot_assets",
    "run_registered_pilot",
]
