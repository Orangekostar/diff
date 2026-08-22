"""Strict source-domain OOF residual and P3 specificity audits for M0."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from cmc_bbdm.cpb_spatial.controls import ControlRecord, patch_shuffle_rgb
from cmc_bbdm.cpb_v3.models import fit_fold_ridge

from .evaluation import FitRecord, _fit_pca, nested_lodo_predictions


class MGMRResidualError(ValueError):
    """Raised when the strict residual audit contract is violated."""


_SPECIFICITY_SEEDS = (20260831, 20260901, 20260902)


def _readonly(value: object, *, vector: bool = False) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MGMRResidualError("residual input must be numeric") from error
    if (vector and array.ndim != 1) or not np.all(np.isfinite(array)):
        raise MGMRResidualError("residual input shape or values are invalid")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(
        contiguous.shape
    )
    output.setflags(write=False)
    return output


def _hash(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(repr(value.shape).encode("ascii"))
            digest.update(value.tobytes(order="C"))
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceResidualRecord:
    branch: str
    outer_domain: str
    specimen_id: str
    dataset_id: str
    target: float
    baseline_prediction: float
    residual: float
    baseline_fit_domains: tuple[str, ...]
    baseline_fit_specimen_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResidualOuterRecord:
    branch: str
    seed: int | None
    outer_domain: str
    specimen_ids: tuple[str, ...]
    targets: np.ndarray
    baseline_predictions: np.ndarray
    corrections: np.ndarray
    corrected_predictions: np.ndarray
    selected_dimension: int
    mean_inner_mae: float
    residual_fit_domains: tuple[str, ...]
    residual_fit_specimen_ids: tuple[str, ...]
    residual_ridge_feature_count: int
    pca_state_sha256: str
    ridge_state_sha256: str


@dataclass(frozen=True, slots=True)
class ResidualBranchAudit:
    branch: str
    seed: int | None
    baseline_predictions: np.ndarray
    corrections: np.ndarray
    corrected_predictions: np.ndarray
    outer_records: tuple[ResidualOuterRecord, ...]
    source_residual_state_sha256: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class ResidualAudit:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    targets: np.ndarray
    coarse: ResidualBranchAudit
    full: ResidualBranchAudit
    shuffles: Mapping[int, ResidualBranchAudit]
    source_residuals: tuple[SourceResidualRecord, ...]
    specificity_seeds: tuple[int, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class _SourceBundle:
    outer_domain: str
    indices: np.ndarray
    residuals: np.ndarray
    records: tuple[SourceResidualRecord, ...]
    state_sha256: str


def patch_shuffle_m0_images(
    images: Sequence[np.ndarray],
    *,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    seed: int,
) -> tuple[tuple[np.ndarray, ...], tuple[ControlRecord, ...]]:
    """Apply the unchanged registered P3 8x8 control in specimen order."""

    image_values = tuple(images)
    specimens = tuple(specimen_ids)
    datasets = tuple(dataset_ids)
    if (
        seed not in _SPECIFICITY_SEEDS
        or not image_values
        or len(image_values) != len(specimens)
        or len(image_values) != len(datasets)
    ):
        raise MGMRResidualError("P3 control inputs are not registered or aligned")
    output: list[np.ndarray] = []
    records: list[ControlRecord] = []
    for image, specimen_id, dataset_id in zip(
        image_values, specimens, datasets, strict=True
    ):
        shuffled, record = patch_shuffle_rgb(
            image,
            specimen_id=specimen_id,
            dataset_id=dataset_id,
            seed=seed,
            grid=(8, 8),
        )
        output.append(shuffled)
        records.append(record)
    return tuple(output), tuple(records)


def _source_bundles(
    *,
    branch: str,
    values: np.ndarray,
    metadata: np.ndarray,
    targets: np.ndarray,
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    domain_order: tuple[str, ...],
    pca_dimensions: tuple[int, ...],
    ridge_alpha: float,
    tie_tolerance: float,
) -> tuple[tuple[_SourceBundle, ...], tuple[SourceResidualRecord, ...]]:
    dataset_array = np.asarray(dataset_ids, dtype=object)
    bundles: list[_SourceBundle] = []
    all_records: list[SourceResidualRecord] = []
    for outer_domain in domain_order:
        source_indices = np.flatnonzero(dataset_array != outer_domain)
        source_domains = tuple(value for value in domain_order if value != outer_domain)
        run = nested_lodo_predictions(
            method=f"{branch}_source_oof",
            metadata=metadata[source_indices],
            blocks={branch: values[source_indices]},
            targets=targets[source_indices],
            specimen_ids=tuple(specimen_ids[index] for index in source_indices),
            dataset_ids=tuple(dataset_ids[index] for index in source_indices),
            domain_order=source_domains,
            pca_dimensions=pca_dimensions,
            ridge_alpha=ridge_alpha,
            tie_tolerance=tie_tolerance,
        )
        residuals = _readonly(targets[source_indices] - run.predictions, vector=True)
        fit_by_domain: dict[str, FitRecord] = {
            row.outer_domain: row for row in run.fit_records if row.stage == "outer"
        }
        records: list[SourceResidualRecord] = []
        for local_index, global_index in enumerate(source_indices):
            domain = dataset_ids[global_index]
            fit = fit_by_domain[domain]
            record = SourceResidualRecord(
                branch=branch,
                outer_domain=outer_domain,
                specimen_id=specimen_ids[global_index],
                dataset_id=domain,
                target=float(targets[global_index]),
                baseline_prediction=float(run.predictions[local_index]),
                residual=float(residuals[local_index]),
                baseline_fit_domains=fit.fit_domains,
                baseline_fit_specimen_ids=fit.fit_specimen_ids,
            )
            records.append(record)
            all_records.append(record)
        state = _hash(
            branch,
            outer_domain,
            tuple(specimen_ids[index] for index in source_indices),
            residuals,
            run.state_sha256,
        )
        bundles.append(
            _SourceBundle(
                outer_domain=outer_domain,
                indices=np.asarray(source_indices, dtype=np.int64),
                residuals=residuals,
                records=tuple(records),
                state_sha256=state,
            )
        )
    return tuple(bundles), tuple(all_records)


def _residual_outer_fit(
    *,
    branch: str,
    seed: int | None,
    bundle: _SourceBundle,
    directional: np.ndarray,
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    domain_order: tuple[str, ...],
    pca_dimensions: tuple[int, ...],
    ridge_alpha: float,
    tie_tolerance: float,
) -> tuple[np.ndarray, ResidualOuterRecord]:
    dataset_array = np.asarray(dataset_ids, dtype=object)
    outer_indices = np.flatnonzero(dataset_array == bundle.outer_domain)
    source_indices = bundle.indices
    source_domains = tuple(value for value in domain_order if value != bundle.outer_domain)
    scores: dict[int, float] = {}
    for dimension in pca_dimensions:
        fold_mae: list[float] = []
        for query_domain in source_domains:
            query = source_indices[dataset_array[source_indices] == query_domain]
            fit = source_indices[dataset_array[source_indices] != query_domain]
            fit_ids = tuple(specimen_ids[index] for index in fit)
            fit_domains = tuple(dataset_ids[index] for index in fit)
            pca = _fit_pca(
                directional[fit],
                dimension,
                fit_ids,
                tuple(dict.fromkeys(fit_domains)),
            )
            ridge = fit_fold_ridge(
                pca.transform(directional[fit]),
                np.asarray(
                    [
                        bundle.residuals[np.flatnonzero(source_indices == index)[0]]
                        for index in fit
                    ],
                    dtype=np.float64,
                ),
                alpha=ridge_alpha,
                fit_sample_ids=fit_ids,
                fit_domain_ids=fit_domains,
            )
            prediction = ridge.predict(pca.transform(directional[query]))
            query_residual = np.asarray(
                [
                    bundle.residuals[np.flatnonzero(source_indices == index)[0]]
                    for index in query
                ],
                dtype=np.float64,
            )
            fold_mae.append(
                float(np.mean(np.abs(query_residual - prediction), dtype=np.float64))
            )
        scores[dimension] = float(math.fsum(fold_mae) / len(fold_mae))
    selected = pca_dimensions[0]
    selected_score = scores[selected]
    for dimension in pca_dimensions[1:]:
        score = scores[dimension]
        if score < selected_score - tie_tolerance:
            selected, selected_score = dimension, score
    fit_ids = tuple(specimen_ids[index] for index in source_indices)
    fit_domains = tuple(dataset_ids[index] for index in source_indices)
    pca = _fit_pca(
        directional[source_indices],
        selected,
        fit_ids,
        tuple(dict.fromkeys(fit_domains)),
    )
    ridge = fit_fold_ridge(
        pca.transform(directional[source_indices]),
        bundle.residuals,
        alpha=ridge_alpha,
        fit_sample_ids=fit_ids,
        fit_domain_ids=fit_domains,
    )
    correction = _readonly(
        ridge.predict(pca.transform(directional[outer_indices])), vector=True
    )
    baseline = _readonly(baseline_predictions[outer_indices], vector=True)
    corrected = _readonly(baseline + correction, vector=True)
    record = ResidualOuterRecord(
        branch=branch,
        seed=seed,
        outer_domain=bundle.outer_domain,
        specimen_ids=tuple(specimen_ids[index] for index in outer_indices),
        targets=_readonly(targets[outer_indices], vector=True),
        baseline_predictions=baseline,
        corrections=correction,
        corrected_predictions=corrected,
        selected_dimension=selected,
        mean_inner_mae=selected_score,
        residual_fit_domains=tuple(dict.fromkeys(fit_domains)),
        residual_fit_specimen_ids=fit_ids,
        residual_ridge_feature_count=int(pca.dimension),
        pca_state_sha256=pca.state_sha256,
        ridge_state_sha256=ridge.state_sha256,
    )
    return correction, record


def _branch_audit(
    *,
    branch: str,
    seed: int | None,
    bundles: tuple[_SourceBundle, ...],
    directional: np.ndarray,
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    domain_order: tuple[str, ...],
    pca_dimensions: tuple[int, ...],
    ridge_alpha: float,
    tie_tolerance: float,
) -> ResidualBranchAudit:
    corrections = np.full(targets.size, np.nan, dtype=np.float64)
    records: list[ResidualOuterRecord] = []
    dataset_array = np.asarray(dataset_ids, dtype=object)
    for bundle in bundles:
        correction, record = _residual_outer_fit(
            branch=branch,
            seed=seed,
            bundle=bundle,
            directional=directional,
            targets=targets,
            baseline_predictions=baseline_predictions,
            specimen_ids=specimen_ids,
            dataset_ids=dataset_ids,
            domain_order=domain_order,
            pca_dimensions=pca_dimensions,
            ridge_alpha=ridge_alpha,
            tie_tolerance=tie_tolerance,
        )
        corrections[dataset_array == bundle.outer_domain] = correction
        records.append(record)
    corrections = _readonly(corrections, vector=True)
    baseline = _readonly(baseline_predictions, vector=True)
    corrected = _readonly(baseline + corrections, vector=True)
    source_state = _hash(tuple(bundle.state_sha256 for bundle in bundles))
    state = _hash(
        branch,
        seed,
        source_state,
        baseline,
        corrections,
        tuple((row.outer_domain, row.selected_dimension) for row in records),
    )
    return ResidualBranchAudit(
        branch=branch,
        seed=seed,
        baseline_predictions=baseline,
        corrections=corrections,
        corrected_predictions=corrected,
        outer_records=tuple(records),
        source_residual_state_sha256=source_state,
        state_sha256=state,
    )


def audit_residual_arrays(
    *,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    domain_order: Sequence[str],
    targets: object,
    metadata: object,
    full: object,
    coarse: object,
    boundary: object,
    coarse_outer_predictions: object,
    full_outer_predictions: object,
    shuffled_boundary: Mapping[int, object],
    pca_dimensions: Sequence[int],
    ridge_alpha: float,
    tie_tolerance: float,
) -> ResidualAudit:
    """Run real and shuffled strict-OOF directional correction audits."""

    samples = tuple(specimen_ids)
    datasets = tuple(dataset_ids)
    domains = tuple(domain_order)
    y = _readonly(targets, vector=True)
    if (
        len(samples) != y.size
        or len(set(samples)) != y.size
        or len(datasets) != y.size
        or set(datasets) != set(domains)
    ):
        raise MGMRResidualError("residual cohort is not aligned")
    meta = _readonly(metadata)
    full_values = _readonly(full)
    coarse_values = _readonly(coarse)
    directional = _readonly(boundary)
    coarse_outer = _readonly(coarse_outer_predictions, vector=True)
    full_outer = _readonly(full_outer_predictions, vector=True)
    matrices = (meta, full_values, coarse_values, directional)
    if any(value.ndim != 2 or value.shape[0] != y.size for value in matrices):
        raise MGMRResidualError("residual feature matrices are not aligned")
    if coarse_outer.shape != y.shape or full_outer.shape != y.shape:
        raise MGMRResidualError("residual baseline predictions are not aligned")
    dimensions = tuple(pca_dimensions)
    if not dimensions or tuple(sorted(set(dimensions))) != dimensions:
        raise MGMRResidualError("residual PCA dimensions are invalid")
    if tuple(shuffled_boundary) != _SPECIFICITY_SEEDS:
        raise MGMRResidualError("P3 specificity seed roster changed")
    shuffled = {seed: _readonly(shuffled_boundary[seed]) for seed in _SPECIFICITY_SEEDS}
    if any(value.shape != directional.shape for value in shuffled.values()):
        raise MGMRResidualError("shuffled directional features are not aligned")

    coarse_bundles, coarse_source_records = _source_bundles(
        branch="coarse",
        values=coarse_values,
        metadata=meta,
        targets=y,
        specimen_ids=samples,
        dataset_ids=datasets,
        domain_order=domains,
        pca_dimensions=dimensions,
        ridge_alpha=ridge_alpha,
        tie_tolerance=tie_tolerance,
    )
    full_bundles, full_source_records = _source_bundles(
        branch="full",
        values=full_values,
        metadata=meta,
        targets=y,
        specimen_ids=samples,
        dataset_ids=datasets,
        domain_order=domains,
        pca_dimensions=dimensions,
        ridge_alpha=ridge_alpha,
        tie_tolerance=tie_tolerance,
    )
    common = {
        "targets": y,
        "specimen_ids": samples,
        "dataset_ids": datasets,
        "domain_order": domains,
        "pca_dimensions": dimensions,
        "ridge_alpha": ridge_alpha,
        "tie_tolerance": tie_tolerance,
    }
    coarse_audit = _branch_audit(
        branch="coarse",
        seed=None,
        bundles=coarse_bundles,
        directional=directional,
        baseline_predictions=coarse_outer,
        **common,
    )
    full_audit = _branch_audit(
        branch="full",
        seed=None,
        bundles=full_bundles,
        directional=directional,
        baseline_predictions=full_outer,
        **common,
    )
    shuffle_audits = {
        seed: _branch_audit(
            branch="coarse_shuffle",
            seed=seed,
            bundles=coarse_bundles,
            directional=shuffled[seed],
            baseline_predictions=coarse_outer,
            **common,
        )
        for seed in _SPECIFICITY_SEEDS
    }
    source_records = coarse_source_records + full_source_records
    state = _hash(
        samples,
        datasets,
        y,
        coarse_audit.state_sha256,
        full_audit.state_sha256,
        tuple((seed, shuffle_audits[seed].state_sha256) for seed in _SPECIFICITY_SEEDS),
    )
    return ResidualAudit(
        specimen_ids=samples,
        dataset_ids=datasets,
        targets=y,
        coarse=coarse_audit,
        full=full_audit,
        shuffles=MappingProxyType(shuffle_audits),
        source_residuals=source_records,
        specificity_seeds=_SPECIFICITY_SEEDS,
        state_sha256=state,
    )


__all__ = [
    "MGMRResidualError",
    "ResidualAudit",
    "ResidualBranchAudit",
    "ResidualOuterRecord",
    "SourceResidualRecord",
    "audit_residual_arrays",
    "patch_shuffle_m0_images",
]
