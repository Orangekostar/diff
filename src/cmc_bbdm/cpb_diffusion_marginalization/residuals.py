"""Cross-fitted P6 residual-bank construction for D8."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cmc_bbdm.cpb_diffusion_reconstruction.artifacts import validate_p6_package
from cmc_bbdm.cpb_diffusion_reconstruction.config import load_p6_config
from cmc_bbdm.cpb_diffusion_reconstruction.models import (
    load_fold_checkpoint,
    sample_diffusion_fields,
)
from cmc_bbdm.cpb_diffusion_reconstruction.reconstruction import (
    build_learning_target,
    build_sparse_observation,
)
from cmc_bbdm.cpb_physical_descriptors import load_physical_calibrations
from cmc_bbdm.cpb_spatial.pipeline import load_registered_inputs
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import V3Data, validate_issued_data_authority

from .config import DOMAIN_ORDER, D8Config


class P6ResidualError(ValueError):
    """Raised when a P6 residual authority is incomplete or inconsistent."""


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise P6ResidualError(f"{label} SHA-256 is invalid")
    return value


def _identities(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise P6ResidualError(f"{label} must be a nonempty tuple")
    result: list[str] = []
    for item in value:
        if (
            type(item) is not str
            or not item
            or item.strip() != item
            or "\0" in item
        ):
            raise P6ResidualError(f"{label} contains an invalid identity")
        result.append(item)
    return tuple(result)


def _readonly_array(
    value: object, *, label: str, dtype: np.dtype, ndim: int
) -> np.ndarray:
    if np.iscomplexobj(value):
        raise P6ResidualError(f"{label} must be real")
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise P6ResidualError(f"{label} must be numeric") from error
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise P6ResidualError(f"{label} must be a finite {ndim}-D array")
    output = np.array(array, dtype=dtype, copy=True, order="C")
    output.setflags(write=False)
    return output


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ResidualAuthority:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    measured_fields: np.ndarray
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    source_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        specimens = _identities(self.specimen_ids, label="specimen IDs")
        domains = _identities(self.dataset_ids, label="dataset IDs")
        sources = tuple(
            _sha256(value, label="source") for value in self.source_sha256
        )
        if (
            len(specimens) != len(set(specimens))
            or len(domains) != len(specimens)
            or len(sources) != len(specimens)
        ):
            raise P6ResidualError("residual authority identity roster changed")
        measured = _readonly_array(
            self.measured_fields,
            label="measured fields",
            dtype=np.dtype(np.float32),
            ndim=4,
        )
        mean = _readonly_array(
            self.posterior_mean,
            label="posterior mean",
            dtype=np.dtype(np.float32),
            ndim=4,
        )
        variance = _readonly_array(
            self.posterior_variance,
            label="posterior variance",
            dtype=np.dtype(np.float32),
            ndim=4,
        )
        expected = (len(specimens), 3, 64, 64)
        if measured.shape != expected or mean.shape != expected or variance.shape != expected:
            raise P6ResidualError("residual authority field shape changed")
        if np.any(variance < 0.0):
            raise P6ResidualError("posterior variance must be nonnegative")
        object.__setattr__(self, "specimen_ids", specimens)
        object.__setattr__(self, "dataset_ids", domains)
        object.__setattr__(self, "source_sha256", sources)
        object.__setattr__(self, "measured_fields", measured)
        object.__setattr__(self, "posterior_mean", mean)
        object.__setattr__(self, "posterior_variance", variance)


@dataclass(frozen=True, slots=True)
class ResidualFoldDraws:
    heldout_domain: str
    specimen_ids: tuple[str, ...]
    checkpoint_train_ids: tuple[str, ...]
    checkpoint_train_domains: tuple[str, ...]
    checkpoint_scientific_digest: str
    draws: np.ndarray

    def __post_init__(self) -> None:
        if type(self.heldout_domain) is not str or not self.heldout_domain:
            raise P6ResidualError("heldout domain is invalid")
        specimens = _identities(self.specimen_ids, label="fold specimen IDs")
        train_ids = _identities(
            self.checkpoint_train_ids, label="checkpoint train IDs"
        )
        train_domains = _identities(
            self.checkpoint_train_domains, label="checkpoint train domains"
        )
        if len(train_ids) != len(train_domains) or len(set(train_ids)) != len(train_ids):
            raise P6ResidualError("checkpoint training roster changed")
        digest = _sha256(
            self.checkpoint_scientific_digest,
            label="checkpoint scientific digest",
        )
        draws = _readonly_array(
            self.draws,
            label="diffusion draws",
            dtype=np.dtype(np.float32),
            ndim=5,
        )
        if draws.shape[0] != len(specimens) or draws.shape[2:] != (3, 64, 64):
            raise P6ResidualError("diffusion draw shape changed")
        object.__setattr__(self, "specimen_ids", specimens)
        object.__setattr__(self, "checkpoint_train_ids", train_ids)
        object.__setattr__(self, "checkpoint_train_domains", train_domains)
        object.__setattr__(self, "checkpoint_scientific_digest", digest)
        object.__setattr__(self, "draws", draws)


@dataclass(frozen=True, slots=True)
class ResidualRecord:
    specimen_id: str
    dataset_id: str
    heldout_domain: str
    draw_index: int
    residual_64: np.ndarray
    source_sha256: str
    checkpoint_scientific_digest: str
    checkpoint_train_ids: tuple[str, ...]
    checkpoint_train_domains: tuple[str, ...]
    residual_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        residual = _readonly_array(
            self.residual_64,
            label="residual field",
            dtype=np.dtype(np.float32),
            ndim=3,
        )
        if residual.shape != (3, 64, 64):
            raise P6ResidualError("residual field shape changed")
        if self.dataset_id != self.heldout_domain:
            raise P6ResidualError("residual is not held out by complete domain")
        if (
            self.dataset_id in self.checkpoint_train_domains
            or self.specimen_id in self.checkpoint_train_ids
        ):
            raise P6ResidualError("heldout residual leaked into checkpoint training")
        if type(self.draw_index) is not int or self.draw_index < 0:
            raise P6ResidualError("residual draw index is invalid")
        object.__setattr__(self, "residual_64", residual)
        object.__setattr__(self, "residual_sha256", _array_sha(residual))


@dataclass(frozen=True, slots=True)
class P6ResidualBank:
    records: tuple[ResidualRecord, ...]
    specimen_count: int
    draw_count: int
    maximum_mean_error: float
    maximum_variance_error: float
    state_sha256: str


def validate_residual_bank(
    bank: object,
    *,
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    source_sha256: tuple[str, ...],
    draw_count: int,
    mean_tolerance: float = 1.0e-6,
    variance_tolerance: float = 1.0e-6,
) -> str:
    """Recompute a residual bank against its registered source authority."""

    if type(bank) is not P6ResidualBank:
        raise P6ResidualError("exact P6ResidualBank is required")
    specimens = _identities(specimen_ids, label="specimen IDs")
    domains = _identities(dataset_ids, label="dataset IDs")
    if type(source_sha256) is not tuple:
        raise P6ResidualError("source hash roster must be a tuple")
    sources = tuple(_sha256(value, label="source") for value in source_sha256)
    if (
        len(specimens) != len(set(specimens))
        or len(domains) != len(specimens)
        or len(sources) != len(specimens)
        or type(draw_count) is not int
        or draw_count < 1
        or type(mean_tolerance) is not float
        or type(variance_tolerance) is not float
        or not math.isfinite(mean_tolerance)
        or not math.isfinite(variance_tolerance)
        or mean_tolerance < 0.0
        or variance_tolerance < 0.0
    ):
        raise P6ResidualError("residual validation authority is invalid")
    if (
        type(bank.records) is not tuple
        or type(bank.specimen_count) is not int
        or type(bank.draw_count) is not int
        or bank.specimen_count != len(specimens)
        or bank.draw_count != draw_count
        or len(bank.records) != len(specimens) * draw_count
    ):
        raise P6ResidualError("residual bank roster is incomplete")
    if (
        type(bank.maximum_mean_error) is not float
        or type(bank.maximum_variance_error) is not float
        or not math.isfinite(bank.maximum_mean_error)
        or not math.isfinite(bank.maximum_variance_error)
        or bank.maximum_mean_error < 0.0
        or bank.maximum_variance_error < 0.0
        or bank.maximum_mean_error > mean_tolerance
        or bank.maximum_variance_error > variance_tolerance
    ):
        raise P6ResidualError("residual bank posterior error is invalid")
    _sha256(bank.state_sha256, label="residual bank state")
    expected = tuple(
        (specimen, domain, source, index)
        for specimen, domain, source in zip(
            specimens, domains, sources, strict=True
        )
        for index in range(draw_count)
    )
    observed: list[tuple[str, str, str, int]] = []
    for record in bank.records:
        if type(record) is not ResidualRecord:
            raise P6ResidualError("residual bank contains an invalid record")
        observed.append(
            (
                record.specimen_id,
                record.dataset_id,
                record.source_sha256,
                record.draw_index,
            )
        )
        if record.source_sha256 != sources[specimens.index(record.specimen_id)]:
            raise P6ResidualError("residual source roster differs from authority")
        if (
            record.dataset_id != record.heldout_domain
            or record.specimen_id in record.checkpoint_train_ids
            or record.dataset_id in record.checkpoint_train_domains
        ):
            raise P6ResidualError("residual checkpoint isolation changed")
        _sha256(record.source_sha256, label="residual source")
        _sha256(
            record.checkpoint_scientific_digest,
            label="checkpoint scientific digest",
        )
        _identities(record.checkpoint_train_ids, label="checkpoint train IDs")
        _identities(record.checkpoint_train_domains, label="checkpoint train domains")
        residual = _readonly_array(
            record.residual_64,
            label="residual field",
            dtype=np.dtype(np.float32),
            ndim=3,
        )
        if residual.shape != (3, 64, 64) or _array_sha(residual) != record.residual_sha256:
            raise P6ResidualError("residual bytes differ from record authority")
    if tuple(observed) != expected:
        raise P6ResidualError("residual bank ordered roster changed")
    state = _bank_state(
        bank.records,
        bank.specimen_count,
        bank.draw_count,
        bank.maximum_mean_error,
        bank.maximum_variance_error,
    )
    if state != bank.state_sha256:
        raise P6ResidualError("residual bank state digest changed")
    return state


def _bank_state(
    records: tuple[ResidualRecord, ...],
    specimen_count: int,
    draw_count: int,
    maximum_mean_error: float,
    maximum_variance_error: float,
) -> str:
    digest = hashlib.sha256()
    for value in (
        specimen_count,
        draw_count,
        maximum_mean_error,
        maximum_variance_error,
    ):
        digest.update(repr(value).encode("ascii"))
        digest.update(b"\0")
    for row in records:
        for value in (
            row.specimen_id,
            row.dataset_id,
            row.heldout_domain,
            str(row.draw_index),
            row.source_sha256,
            row.checkpoint_scientific_digest,
            row.residual_sha256,
            *row.checkpoint_train_ids,
            *row.checkpoint_train_domains,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def build_residual_bank_from_arrays(
    authority: ResidualAuthority,
    folds: tuple[ResidualFoldDraws, ...],
    *,
    draw_count: int,
    mean_tolerance: float = 1.0e-6,
    variance_tolerance: float = 1.0e-6,
) -> P6ResidualBank:
    """Validate a complete cross-fitted draw roster and subtract measured fields."""

    if type(authority) is not ResidualAuthority:
        raise P6ResidualError("exact residual authority is required")
    if (
        not isinstance(folds, tuple)
        or type(draw_count) is not int
        or draw_count < 1
        or not math.isfinite(mean_tolerance)
        or not math.isfinite(variance_tolerance)
        or mean_tolerance < 0.0
        or variance_tolerance < 0.0
    ):
        raise P6ResidualError("residual-bank parameters are invalid")
    domains = tuple(dict.fromkeys(authority.dataset_ids))
    if (
        len(folds) != len(domains)
        or tuple(fold.heldout_domain for fold in folds) != domains
        or len({fold.heldout_domain for fold in folds}) != len(folds)
    ):
        raise P6ResidualError("fold roster does not match residual authority")
    position = {specimen: index for index, specimen in enumerate(authority.specimen_ids)}
    reconstructed_mean = np.empty_like(authority.posterior_mean)
    reconstructed_variance = np.empty_like(authority.posterior_variance)
    records: list[ResidualRecord] = []
    observed: set[str] = set()
    for fold in folds:
        expected_ids = tuple(
            specimen
            for specimen, domain in zip(
                authority.specimen_ids, authority.dataset_ids, strict=True
            )
            if domain == fold.heldout_domain
        )
        expected_train_domains = set(domains) - {fold.heldout_domain}
        if (
            fold.specimen_ids != expected_ids
            or fold.draws.shape[1] != draw_count
            or fold.heldout_domain in fold.checkpoint_train_domains
            or set(fold.checkpoint_train_domains) != expected_train_domains
            or set(fold.specimen_ids) & set(fold.checkpoint_train_ids)
        ):
            raise P6ResidualError("heldout fold authority changed")
        indices = np.asarray([position[item] for item in fold.specimen_ids], dtype=np.int64)
        reconstructed_mean[indices] = np.mean(
            fold.draws, axis=1, dtype=np.float64
        ).astype(np.float32)
        reconstructed_variance[indices] = np.var(
            fold.draws, axis=1, dtype=np.float64
        ).astype(np.float32)
        for local, specimen_id in enumerate(fold.specimen_ids):
            if specimen_id in observed:
                raise P6ResidualError("fold roster contains duplicate specimens")
            observed.add(specimen_id)
            global_index = position[specimen_id]
            for draw_index in range(draw_count):
                residual = np.asarray(
                    fold.draws[local, draw_index]
                    - authority.measured_fields[global_index],
                    dtype=np.float32,
                )
                records.append(
                    ResidualRecord(
                        specimen_id=specimen_id,
                        dataset_id=authority.dataset_ids[global_index],
                        heldout_domain=fold.heldout_domain,
                        draw_index=draw_index,
                        residual_64=residual,
                        source_sha256=authority.source_sha256[global_index],
                        checkpoint_scientific_digest=(
                            fold.checkpoint_scientific_digest
                        ),
                        checkpoint_train_ids=fold.checkpoint_train_ids,
                        checkpoint_train_domains=fold.checkpoint_train_domains,
                    )
                )
    if observed != set(authority.specimen_ids):
        raise P6ResidualError("fold roster is incomplete")
    maximum_mean_error = float(
        np.max(
            np.abs(
                reconstructed_mean.astype(np.float64)
                - authority.posterior_mean.astype(np.float64)
            )
        )
    )
    maximum_variance_error = float(
        np.max(
            np.abs(
                reconstructed_variance.astype(np.float64)
                - authority.posterior_variance.astype(np.float64)
            )
        )
    )
    if maximum_mean_error > mean_tolerance:
        raise P6ResidualError("regenerated posterior mean differs from P6 authority")
    if maximum_variance_error > variance_tolerance:
        raise P6ResidualError("regenerated posterior variance differs from P6 authority")
    frozen_records = tuple(records)
    state = _bank_state(
        frozen_records,
        len(authority.specimen_ids),
        draw_count,
        maximum_mean_error,
        maximum_variance_error,
    )
    return P6ResidualBank(
        records=frozen_records,
        specimen_count=len(authority.specimen_ids),
        draw_count=draw_count,
        maximum_mean_error=maximum_mean_error,
        maximum_variance_error=maximum_variance_error,
        state_sha256=state,
    )


def _decode_p6_normalized_posterior(
    mean: object, variance: object
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the published [0, 1] P6 moments back to its [-1, 1] target scale."""

    if np.iscomplexobj(mean) or np.iscomplexobj(variance):
        raise P6ResidualError("P6 posterior moments must be real")
    try:
        normalized_mean = np.asarray(mean, dtype=np.float32)
        normalized_variance = np.asarray(variance, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise P6ResidualError("P6 posterior moments must be numeric") from error
    if (
        normalized_mean.shape != normalized_variance.shape
        or not np.all(np.isfinite(normalized_mean))
        or not np.all(np.isfinite(normalized_variance))
        or np.any(normalized_mean < 0.0)
        or np.any(normalized_mean > 1.0)
        or np.any(normalized_variance < 0.0)
    ):
        raise P6ResidualError("P6 normalized posterior moments are invalid")
    raw_mean = np.asarray(
        normalized_mean.astype(np.float64) * 2.0 - 1.0,
        dtype=np.float32,
    )
    raw_variance = np.asarray(
        normalized_variance.astype(np.float64) * 4.0,
        dtype=np.float32,
    )
    return np.ascontiguousarray(raw_mean), np.ascontiguousarray(raw_variance)


def _registered_arrays(
    data: V3Data, *, root: Path, base_config: object
) -> tuple[ResidualAuthority, np.ndarray]:
    calibration = base_config.sources["physical_calibration"]
    calibrations = load_physical_calibrations(
        root / calibration.path,
        project_root=root,
        expected_sha256=calibration.sha256,
    )
    inputs = load_registered_inputs(
        data, project_root=root, calibrations=calibrations
    )
    conditions: list[np.ndarray] = []
    measured: list[np.ndarray] = []
    source_hashes: list[str] = []
    specimen_ids = tuple(str(item) for item in data.sample_ids.tolist())
    dataset_ids = tuple(str(item) for item in data.dataset_ids.tolist())
    for specimen, domain, image in zip(
        specimen_ids, dataset_ids, inputs.images, strict=True
    ):
        observation = build_sparse_observation(
            image, specimen_id=specimen, dataset_id=domain
        )
        conditions.append(observation.condition)
        measured.append(build_learning_target(image, observation))
        source_hashes.append(observation.source_sha256)
    p6_source = root / "results/cpb_spatial/p6_diffusion_reconstruction"
    try:
        with np.load(p6_source / "uncertainty_source_data.npz", allow_pickle=False) as item:
            stored_ids = tuple(str(value) for value in item["specimen_ids"].tolist())
            stored_domains = tuple(str(value) for value in item["dataset_ids"].tolist())
            mean = np.asarray(item["posterior_mean_64"], dtype=np.float32)
            variance = np.asarray(item["posterior_variance_64"], dtype=np.float32)
            draws = np.asarray(item["draw_count"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as error:
        raise P6ResidualError("P6 uncertainty authority cannot be decoded") from error
    if (
        stored_ids != specimen_ids
        or stored_domains != dataset_ids
        or mean.shape != (data.n_samples, 64, 64, 3)
        or variance.shape != mean.shape
        or draws.shape != (data.n_samples,)
        or not np.all(draws == 8)
    ):
        raise P6ResidualError("P6 uncertainty authority roster changed")
    raw_mean, raw_variance = _decode_p6_normalized_posterior(mean, variance)
    authority = ResidualAuthority(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        measured_fields=np.stack(measured),
        posterior_mean=raw_mean.transpose(0, 3, 1, 2),
        posterior_variance=raw_variance.transpose(0, 3, 1, 2),
        source_sha256=tuple(source_hashes),
    )
    return authority, np.stack(conditions).astype(np.float32, copy=False)


def build_cross_fitted_p6_residual_bank(
    data: object,
    *,
    config: D8Config,
    project_root: str | Path,
    device: str = "cuda",
) -> P6ResidualBank:
    """Regenerate each formal P6 draw and build the 2,208-row residual bank."""

    if type(data) is not V3Data:
        raise P6ResidualError("formal residual generation requires exact V3Data")
    try:
        validate_issued_data_authority(data)
    except (TypeError, ValueError) as error:
        raise P6ResidualError("formal residual data lacks loader authority") from error
    if type(config) is not D8Config or tuple(config.outer_domains) != DOMAIN_ORDER:
        raise P6ResidualError("formal residual generation requires exact D8Config")
    if device != "cuda":
        raise P6ResidualError("formal P6 residual generation requires CUDA")
    root = Path(project_root).resolve(strict=True)
    p6_dir = root / "results/cpb_spatial/p6_diffusion_reconstruction"
    p6_config_path = root / config.sources["p6_config"].path
    validate_p6_package(
        p6_dir,
        project_root=root,
        config_path=p6_config_path,
    )
    p6_config = load_p6_config(p6_config_path, project_root=root)
    base_config = load_v3_config(
        root / config.sources["p1_config"].path,
        project_root=root,
    )
    authority, conditions = _registered_arrays(
        data, root=root, base_config=base_config
    )
    domains = np.asarray(authority.dataset_ids)
    folds: list[ResidualFoldDraws] = []
    for domain in DOMAIN_ORDER:
        indices = np.flatnonzero(domains == domain)
        checkpoint = load_fold_checkpoint(
            p6_dir / "models" / f"diffusion__{domain}.safetensors",
            p6_dir / "models" / f"diffusion__{domain}.json",
        )
        specimen_ids = tuple(authority.specimen_ids[index] for index in indices)
        values = sample_diffusion_fields(
            checkpoint,
            conditions[indices],
            specimen_ids=specimen_ids,
            draws=p6_config.posterior_draws,
            ddim_steps=p6_config.ddim_steps,
            eta=p6_config.ddim_eta,
            device=device,
        )
        folds.append(
            ResidualFoldDraws(
                heldout_domain=domain,
                specimen_ids=specimen_ids,
                checkpoint_train_ids=checkpoint.train_sample_ids,
                checkpoint_train_domains=checkpoint.train_domains,
                checkpoint_scientific_digest=checkpoint.scientific_digest,
                draws=values,
            )
        )
    return build_residual_bank_from_arrays(
        authority,
        tuple(folds),
        draw_count=p6_config.posterior_draws,
        mean_tolerance=1.0e-6,
        variance_tolerance=1.0e-6,
    )


__all__ = [
    "P6ResidualBank",
    "P6ResidualError",
    "ResidualAuthority",
    "ResidualFoldDraws",
    "ResidualRecord",
    "build_cross_fitted_p6_residual_bank",
    "build_residual_bank_from_arrays",
    "validate_residual_bank",
]
