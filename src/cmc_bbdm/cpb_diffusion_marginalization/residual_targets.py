"""Measured-field nuisance targets for D8 residual diffusion."""

from __future__ import annotations

import hashlib
import io
import json
import math
import weakref
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import optuna
from PIL import Image

from cmc_bbdm.cpb_diffusion_reconstruction.reconstruction import (
    build_learning_target,
    build_sparse_observation,
)

from .artifacts import validate_d8_search_package
from .authority import (
    D8InnerFold,
    D8SearchView,
    validate_inner_fold,
    validate_search_view,
)
from .config import DOMAIN_ORDER
from .decomposition import decompose_residual
from .residual_config import ResidualDiffusionConfig
from .search import D8Candidate

_FIELD_SHAPE = (3, 64, 64)
_DIFFUSION_CONTROLS = frozenset({"B5", "B6", "B7", "B8"})
_BANK_ISSUER = object()
_BANK_REGISTRY: weakref.WeakKeyDictionary[ResidualFieldBank, str] = (
    weakref.WeakKeyDictionary()
)


class ResidualTargetError(ValueError):
    """Raised when a residual target loses its pre-outer authority."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_fields(value: object, *, label: str, bounded: bool) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ResidualTargetError(f"{label} must be real")
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ResidualTargetError(f"{label} must be numeric") from error
    if (
        array.ndim != 4
        or array.shape[1:] != _FIELD_SHAPE
        or len(array) == 0
        or not np.all(np.isfinite(array))
    ):
        raise ResidualTargetError(
            f"{label} must be a nonempty finite (n, 3, 64, 64) array"
        )
    if bounded and (float(np.min(array)) < -1.0 or float(np.max(array)) > 1.0):
        raise ResidualTargetError(f"{label} must lie in [-1, 1]")
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    output = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.float32
    ).reshape(contiguous.shape)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class PilotDiffusionScaffold:
    """Best eligible Pilot diffusion scaffold for one prospective outer."""

    outer_domain: str
    control_id: str
    decomposition_family: str
    selected_band: str
    decomposition_parameters: Mapping[str, object]
    candidate_sha256: str
    config_sha256: str
    trial_number: int
    objective: float
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.outer_domain not in DOMAIN_ORDER:
            raise ResidualTargetError("scaffold outer domain is not registered")
        if self.control_id not in _DIFFUSION_CONTROLS:
            raise ResidualTargetError("scaffold is not diffusion-specific")
        if self.decomposition_family not in {"gaussian", "fourier", "wavelet"}:
            raise ResidualTargetError("scaffold decomposition is not registered")
        if self.selected_band not in {"low", "mid", "mid+high", "high"}:
            raise ResidualTargetError("scaffold band is not registered")
        parameters = dict(self.decomposition_parameters)
        if parameters.get("band") != self.selected_band:
            raise ResidualTargetError("scaffold band parameters differ")
        if any(type(key) is not str for key in parameters):
            raise ResidualTargetError("scaffold parameters are invalid")
        if not _valid_sha256(self.candidate_sha256) or not _valid_sha256(
            self.config_sha256
        ):
            raise ResidualTargetError("scaffold authority hash is invalid")
        if type(self.trial_number) is not int or self.trial_number < 0:
            raise ResidualTargetError("scaffold trial number is invalid")
        if type(self.objective) is not float or not math.isfinite(self.objective):
            raise ResidualTargetError("scaffold objective is invalid")
        frozen_parameters = MappingProxyType(parameters)
        state = _canonical_sha256(
            {
                "outer_domain": self.outer_domain,
                "control_id": self.control_id,
                "decomposition_family": self.decomposition_family,
                "selected_band": self.selected_band,
                "decomposition_parameters": parameters,
                "candidate_sha256": self.candidate_sha256,
                "config_sha256": self.config_sha256,
                "trial_number": self.trial_number,
                "objective": self.objective,
            }
        )
        object.__setattr__(self, "decomposition_parameters", frozen_parameters)
        object.__setattr__(self, "state_sha256", state)


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class ResidualFieldBank:
    """One process-local five-domain measured-field authority."""

    _token: InitVar[object]
    outer_domain: str
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    source_sha256: tuple[str, ...]
    native_source_sha256: tuple[str, ...]
    measured: np.ndarray
    authority_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self, _token: object) -> None:
        if _token is not _BANK_ISSUER:
            raise TypeError("residual field banks require the loader issuer")
        if (
            self.outer_domain not in DOMAIN_ORDER
            or type(self.specimen_ids) is not tuple
            or not self.specimen_ids
            or len(set(self.specimen_ids)) != len(self.specimen_ids)
            or type(self.dataset_ids) is not tuple
            or len(self.dataset_ids) != len(self.specimen_ids)
            or self.outer_domain in self.dataset_ids
        ):
            raise ResidualTargetError("field bank identity roster is invalid")
        if (
            type(self.source_sha256) is not tuple
            or len(self.source_sha256) != len(self.specimen_ids)
            or any(not _valid_sha256(value) for value in self.source_sha256)
            or type(self.native_source_sha256) is not tuple
            or len(self.native_source_sha256) != len(self.specimen_ids)
            or any(
                not _valid_sha256(value) for value in self.native_source_sha256
            )
            or not _valid_sha256(self.authority_sha256)
        ):
            raise ResidualTargetError("field bank source authority is invalid")
        measured = _readonly_fields(self.measured, label="field bank", bounded=True)
        if measured.shape != (len(self.specimen_ids), *_FIELD_SHAPE):
            raise ResidualTargetError("field bank arrays are not aligned")
        state = _canonical_sha256(
            {
                "outer_domain": self.outer_domain,
                "specimen_ids": self.specimen_ids,
                "dataset_ids": self.dataset_ids,
                "source_sha256": self.source_sha256,
                "native_source_sha256": self.native_source_sha256,
                "measured_sha256": _array_sha256(measured),
                "authority_sha256": self.authority_sha256,
            }
        )
        object.__setattr__(self, "measured", measured)
        object.__setattr__(self, "state_sha256", state)
        _BANK_REGISTRY[self] = state

    def __copy__(self):
        raise TypeError("residual field banks cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("residual field banks cannot be copied")

    def __reduce__(self):
        raise TypeError("residual field banks cannot be pickled")


@dataclass(frozen=True, slots=True)
class ResidualTargetBatch:
    """One immutable fit-only exact decomposition batch."""

    role: str
    outer_domain: str
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    source_sha256: tuple[str, ...]
    measured: np.ndarray
    stable: np.ndarray
    stable_condition: np.ndarray
    residual: np.ndarray
    training_target: np.ndarray
    scaffold_sha256: str
    authority_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.role not in {"inner_fit", "outer_fit"}:
            raise ResidualTargetError("residual batch role is invalid")
        if self.outer_domain not in DOMAIN_ORDER:
            raise ResidualTargetError("residual batch outer domain is invalid")
        if (
            type(self.specimen_ids) is not tuple
            or not self.specimen_ids
            or len(set(self.specimen_ids)) != len(self.specimen_ids)
            or type(self.dataset_ids) is not tuple
            or len(self.dataset_ids) != len(self.specimen_ids)
            or self.outer_domain in self.dataset_ids
        ):
            raise ResidualTargetError("residual batch identity roster is invalid")
        if (
            type(self.source_sha256) is not tuple
            or len(self.source_sha256) != len(self.specimen_ids)
            or any(not _valid_sha256(value) for value in self.source_sha256)
            or not _valid_sha256(self.scaffold_sha256)
            or not _valid_sha256(self.authority_sha256)
        ):
            raise ResidualTargetError("residual batch source authority is invalid")
        measured = _readonly_fields(self.measured, label="measured fields", bounded=True)
        stable = _readonly_fields(self.stable, label="stable fields", bounded=False)
        stable_condition = _readonly_fields(
            self.stable_condition,
            label="stable conditions",
            bounded=True,
        )
        residual = _readonly_fields(self.residual, label="residual fields", bounded=False)
        target = _readonly_fields(
            self.training_target, label="training targets", bounded=True
        )
        expected = (len(self.specimen_ids), *_FIELD_SHAPE)
        if any(
            value.shape != expected
            for value in (measured, stable, stable_condition, residual, target)
        ):
            raise ResidualTargetError("residual batch arrays are not aligned")
        if not np.array_equal(
            stable_condition,
            np.clip(stable, -1.0, 1.0).astype(np.float32),
        ):
            raise ResidualTargetError("stable condition is not the registered clip")
        reconstruction_error = float(
            np.max(
                np.abs(
                    stable.astype(np.float64)
                    + residual.astype(np.float64)
                    - measured.astype(np.float64)
                )
            )
        )
        if reconstruction_error > 2.0e-7 or not np.array_equal(
            target * np.float32(2.0), residual
        ):
            raise ResidualTargetError("residual target decomposition changed")
        state = _canonical_sha256(
            {
                "role": self.role,
                "outer_domain": self.outer_domain,
                "specimen_ids": self.specimen_ids,
                "dataset_ids": self.dataset_ids,
                "source_sha256": self.source_sha256,
                "measured_sha256": _array_sha256(measured),
                "stable_sha256": _array_sha256(stable),
                "stable_condition_sha256": _array_sha256(stable_condition),
                "residual_sha256": _array_sha256(residual),
                "training_target_sha256": _array_sha256(target),
                "scaffold_sha256": self.scaffold_sha256,
                "authority_sha256": self.authority_sha256,
            }
        )
        object.__setattr__(self, "measured", measured)
        object.__setattr__(self, "stable", stable)
        object.__setattr__(self, "stable_condition", stable_condition)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "training_target", target)
        object.__setattr__(self, "state_sha256", state)


def validate_residual_target_batch(value: object) -> str:
    """Recompute the complete state of one immutable residual target batch."""

    if type(value) is not ResidualTargetBatch:
        raise ResidualTargetError("exact ResidualTargetBatch is required")
    arrays = (
        value.measured,
        value.stable,
        value.stable_condition,
        value.residual,
        value.training_target,
    )
    if any(type(array) is not np.ndarray or array.flags.writeable for array in arrays):
        raise ResidualTargetError("residual target arrays lost immutability")
    reconstruction_error = float(
        np.max(
            np.abs(
                value.stable.astype(np.float64)
                + value.residual.astype(np.float64)
                - value.measured.astype(np.float64)
            )
        )
    )
    current = _canonical_sha256(
        {
            "role": value.role,
            "outer_domain": value.outer_domain,
            "specimen_ids": value.specimen_ids,
            "dataset_ids": value.dataset_ids,
            "source_sha256": value.source_sha256,
            "measured_sha256": _array_sha256(value.measured),
            "stable_sha256": _array_sha256(value.stable),
            "stable_condition_sha256": _array_sha256(value.stable_condition),
            "residual_sha256": _array_sha256(value.residual),
            "training_target_sha256": _array_sha256(value.training_target),
            "scaffold_sha256": value.scaffold_sha256,
            "authority_sha256": value.authority_sha256,
        }
    )
    if (
        reconstruction_error > 2.0e-7
        or not np.array_equal(
            value.stable_condition,
            np.clip(value.stable, -1.0, 1.0).astype(np.float32),
        )
        or not np.array_equal(
            value.training_target * np.float32(2.0), value.residual
        )
        or current != value.state_sha256
    ):
        raise ResidualTargetError("residual target batch state changed")
    return current


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResidualTargetError(f"{label} cannot be decoded") from error
    if type(value) is not dict:
        raise ResidualTargetError(f"{label} must be a mapping")
    return value


def load_pilot_diffusion_scaffolds(
    config: ResidualDiffusionConfig, *, project_root: str | Path
) -> Mapping[str, PilotDiffusionScaffold]:
    """Rebuild the six escalation-selected diffusion scaffolds from Pilot DB."""

    if type(config) is not ResidualDiffusionConfig:
        raise TypeError("exact ResidualDiffusionConfig is required")
    root = Path(project_root).resolve(strict=True)
    package = root / "results/d8_search"
    validated = validate_d8_search_package(
        package,
        project_root=root,
        config_path=root / config.sources["exploration_config"].path,
    )
    if (
        validated.outer_evaluation_count != 0
        or validated.escalation_status != "TRAIN_RESIDUAL_DIFFUSION"
        or validated.scientific_digest != config.pilot_scientific_digest
    ):
        raise ResidualTargetError("validated Pilot package does not authorize training")
    evidence = _load_json(package / "escalation_evidence.json", "Pilot escalation")
    studies = evidence.get("studies")
    if type(studies) is not list or len(studies) != len(DOMAIN_ORDER):
        raise ResidualTargetError("Pilot escalation study roster changed")
    by_outer = {
        item.get("outer_domain"): item
        for item in studies
        if type(item) is dict and type(item.get("outer_domain")) is str
    }
    if tuple(by_outer) != DOMAIN_ORDER:
        raise ResidualTargetError("Pilot escalation outer order changed")
    storage = f"sqlite:///{(package / 'study.db').resolve()}"
    scaffolds: dict[str, PilotDiffusionScaffold] = {}
    for outer_domain in DOMAIN_ORDER:
        row = by_outer[outer_domain]
        candidate_sha = row.get("diffusion_candidate_sha256")
        objective = row.get("diffusion_objective")
        if not _valid_sha256(candidate_sha) or type(objective) is not float:
            raise ResidualTargetError("Pilot diffusion evidence is invalid")
        study = optuna.load_study(
            study_name=f"d8::{outer_domain}", storage=storage
        )
        matches = tuple(
            trial
            for trial in study.trials
            if type(trial.user_attrs.get("candidate")) is dict
            and trial.user_attrs["candidate"].get("state_sha256") == candidate_sha
        )
        if len(matches) != 1 or matches[0].value != objective:
            raise ResidualTargetError("Pilot diffusion trial binding changed")
        candidate = D8Candidate.from_payload(matches[0].user_attrs["candidate"])
        if (
            candidate.control_id not in _DIFFUSION_CONTROLS
            or candidate.config_sha256
            != config.sources["exploration_config"].sha256
            or candidate.state_sha256 != candidate_sha
        ):
            raise ResidualTargetError("Pilot diffusion candidate is not authorized")
        parameters = dict(candidate.decomposition_parameters)
        parameters["band"] = candidate.band
        scaffolds[outer_domain] = PilotDiffusionScaffold(
            outer_domain=outer_domain,
            control_id=candidate.control_id,
            decomposition_family=candidate.decomposition_family,
            selected_band=candidate.band,
            decomposition_parameters=parameters,
            candidate_sha256=candidate.state_sha256,
            config_sha256=candidate.config_sha256,
            trial_number=matches[0].number,
            objective=float(objective),
        )
    return MappingProxyType(scaffolds)


def _decode_measured_field(
    record: object, *, project_root: Path
) -> tuple[np.ndarray, str]:
    required = (
        "read_bytes",
        "specimen_id",
        "dataset_id",
        "width",
        "height",
        "sha256",
    )
    if any(not hasattr(record, name) for name in required):
        raise ResidualTargetError("C-scan record is incomplete")
    try:
        payload = record.read_bytes(project_root)
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            if source.mode != "RGB" or source.size != (record.width, record.height):
                raise ResidualTargetError("C-scan image mode or dimensions changed")
            image = np.asarray(source, dtype=np.uint8).copy()
    except ResidualTargetError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise ResidualTargetError("C-scan image cannot be decoded") from error
    observation = build_sparse_observation(
        image,
        specimen_id=str(record.specimen_id),
        dataset_id=str(record.dataset_id),
    )
    return (
        np.asarray(build_learning_target(image, observation), dtype=np.float32),
        observation.source_sha256,
    )


def validate_search_residual_field_bank(
    field_bank: object, *, search_view: D8SearchView
) -> str:
    """Revalidate one loader-issued field bank against its search authority."""

    if type(field_bank) is not ResidualFieldBank:
        raise ResidualTargetError("exact ResidualFieldBank is required")
    registered = _BANK_REGISTRY.get(field_bank)
    if registered is None:
        raise ResidualTargetError("field bank has no process-local authority")
    search_state = validate_search_view(search_view)
    current = _canonical_sha256(
        {
            "outer_domain": field_bank.outer_domain,
            "specimen_ids": field_bank.specimen_ids,
            "dataset_ids": field_bank.dataset_ids,
            "source_sha256": field_bank.source_sha256,
            "native_source_sha256": field_bank.native_source_sha256,
            "measured_sha256": _array_sha256(field_bank.measured),
            "authority_sha256": field_bank.authority_sha256,
        }
    )
    if (
        field_bank.outer_domain != search_view.outer_domain
        or field_bank.specimen_ids != search_view.specimen_ids
        or field_bank.dataset_ids != search_view.dataset_ids
        or field_bank.source_sha256
        != tuple(record.sha256 for record in search_view.data_view.cscan_records)
        or field_bank.authority_sha256 != search_state
        or field_bank.measured.flags.writeable
        or current != registered
        or field_bank.state_sha256 != registered
    ):
        raise ResidualTargetError("field bank state changed")
    return current


def load_search_residual_field_bank(
    search_view: D8SearchView, *, project_root: str | Path
) -> ResidualFieldBank:
    """Load each measured C-scan in one five-domain search view exactly once."""

    authority = validate_search_view(search_view)
    root = Path(project_root).resolve(strict=True)
    records = tuple(search_view.data_view.cscan_records)
    decoded = tuple(
        _decode_measured_field(record, project_root=root) for record in records
    )
    measured = np.stack(tuple(value[0] for value in decoded)).astype(np.float32)
    return ResidualFieldBank(
        _BANK_ISSUER,
        outer_domain=search_view.outer_domain,
        specimen_ids=search_view.specimen_ids,
        dataset_ids=search_view.dataset_ids,
        source_sha256=tuple(record.sha256 for record in records),
        native_source_sha256=tuple(value[1] for value in decoded),
        measured=measured,
        authority_sha256=authority,
    )


def _build_batch(
    *,
    role: str,
    outer_domain: str,
    authority_sha256: str,
    indices: np.ndarray,
    scaffold: PilotDiffusionScaffold,
    field_bank: ResidualFieldBank,
) -> ResidualTargetBatch:
    if type(scaffold) is not PilotDiffusionScaffold:
        raise TypeError("exact PilotDiffusionScaffold is required")
    if scaffold.outer_domain != outer_domain:
        raise ResidualTargetError("scaffold outer domain differs from fold authority")
    selected_indices = np.asarray(indices, dtype=np.int64)
    measured = np.asarray(field_bank.measured[selected_indices], dtype=np.float32)
    specimen_ids = tuple(field_bank.specimen_ids[int(index)] for index in selected_indices)
    dataset_ids = tuple(field_bank.dataset_ids[int(index)] for index in selected_indices)
    source_sha256 = tuple(
        field_bank.source_sha256[int(index)] for index in selected_indices
    )
    selected: list[np.ndarray] = []
    for field_value in measured:
        bands = decompose_residual(
            field_value,
            family=scaffold.decomposition_family,
            parameters=dict(scaffold.decomposition_parameters),
        )
        selected.append(np.asarray(bands.selected, dtype=np.float32))
    residual = np.stack(selected).astype(np.float32)
    stable = measured - residual
    stable_condition = np.clip(stable, -1.0, 1.0).astype(np.float32)
    target = residual / np.float32(2.0)
    if float(np.min(target)) < -1.0 or float(np.max(target)) > 1.0:
        raise ResidualTargetError("registered residual scale does not bound targets")
    return ResidualTargetBatch(
        role=role,
        outer_domain=outer_domain,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        source_sha256=source_sha256,
        measured=measured,
        stable=stable,
        stable_condition=stable_condition,
        residual=residual,
        training_target=target,
        scaffold_sha256=scaffold.state_sha256,
        authority_sha256=authority_sha256,
    )


def residual_replacement_perturbations(
    sampled_targets: object,
    *,
    observed_residual: object,
) -> np.ndarray:
    """Convert sampled ``R/2`` targets into ``R_sample - R_observed`` fields."""

    sampled = _readonly_fields(
        sampled_targets,
        label="sampled residual targets",
        bounded=True,
    )
    if np.iscomplexobj(observed_residual):
        raise ResidualTargetError("observed residual must be real")
    try:
        observed_array = np.asarray(observed_residual, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ResidualTargetError("observed residual must be numeric") from error
    if observed_array.shape != _FIELD_SHAPE:
        raise ResidualTargetError("observed residual must have shape (3, 64, 64)")
    observed = _readonly_fields(
        observed_array[np.newaxis, ...] / np.float32(2.0),
        label="scaled observed residual",
        bounded=True,
    ) * np.float32(2.0)
    return _readonly_fields(
        sampled * np.float32(2.0) - observed,
        label="residual replacement perturbations",
        bounded=False,
    )


def build_fit_residual_target_batch(
    inner_fold: D8InnerFold,
    scaffold: PilotDiffusionScaffold,
    *,
    field_bank: ResidualFieldBank,
) -> ResidualTargetBatch:
    """Build targets from four inner-fit domains; query rows are inaccessible."""

    authority = validate_inner_fold(inner_fold)
    validate_search_residual_field_bank(
        field_bank, search_view=inner_fold.search_view
    )
    return _build_batch(
        role="inner_fit",
        outer_domain=inner_fold.outer_domain,
        authority_sha256=authority,
        indices=inner_fold.fit_indices,
        scaffold=scaffold,
        field_bank=field_bank,
    )


def build_outer_fit_residual_target_batch(
    search_view: D8SearchView,
    scaffold: PilotDiffusionScaffold,
    *,
    field_bank: ResidualFieldBank,
) -> ResidualTargetBatch:
    """Build final-training targets from the five-domain pre-outer view."""

    authority = validate_search_view(search_view)
    validate_search_residual_field_bank(field_bank, search_view=search_view)
    return _build_batch(
        role="outer_fit",
        outer_domain=search_view.outer_domain,
        authority_sha256=authority,
        indices=np.arange(search_view.specimen_count, dtype=np.int64),
        scaffold=scaffold,
        field_bank=field_bank,
    )


__all__ = [
    "PilotDiffusionScaffold",
    "ResidualFieldBank",
    "ResidualTargetBatch",
    "ResidualTargetError",
    "build_fit_residual_target_batch",
    "build_outer_fit_residual_target_batch",
    "load_pilot_diffusion_scaffolds",
    "load_search_residual_field_bank",
    "residual_replacement_perturbations",
    "validate_residual_target_batch",
    "validate_search_residual_field_bank",
]
