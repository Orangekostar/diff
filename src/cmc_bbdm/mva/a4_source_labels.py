"""Outer-safe source OOF values and global rankings for MVA A4."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from .a4_candidate_bank import CandidateBank, validate_candidate_bank
from .cai_evaluator import CAIPredictor
from .crossfit import FitAudit, fit_outer_source_predictor
from .global_mask import (
    GlobalMaskRanking,
    SourceDomainStability,
    aggregate_global_ranking,
    leave_one_source_domain_out_stability,
)


class SourceLabelError(ValueError):
    """Raised when source-only A4 labels violate their information barrier."""


METHODS = (
    "global_appearance_mask",
    "global_reconstruction_mask",
    "global_mechanical_mask",
)


@dataclass(frozen=True, slots=True)
class SourcePredictorAudit:
    stage: str
    held_out_target_domain: str
    query_source_domain: str
    query_domains: tuple[str, ...]
    fit_domains: tuple[str, ...]
    query_specimen_ids: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    pca_dimension: int
    predictor_state_sha256: str


@dataclass(frozen=True, slots=True)
class SourceLabelResult:
    outer_domain: str
    domain_order: tuple[str, ...]
    source_domains: tuple[str, ...]
    source_specimen_ids: tuple[str, ...]
    candidate_bank_state_sha256: str
    rows: tuple[dict[str, object], ...]
    fit_audits: tuple[SourcePredictorAudit, ...]
    selection_audits: tuple[SourcePredictorAudit, ...]
    rankings: tuple[GlobalMaskRanking, ...]
    stability: tuple[SourceDomainStability, ...]
    state_sha256: str


_ROW_KEYS = {
    "absolute_error_after",
    "absolute_error_before",
    "added_measurements",
    "candidate_bank_state_sha256",
    "candidate_prediction",
    "cell_index",
    "current_prediction",
    "dataset_id",
    "from_level",
    "method",
    "outer_domain",
    "predictor_state_sha256",
    "primary_value",
    "secondary_value",
    "specimen_id",
    "to_level",
}


def _audit(
    value: FitAudit, *, held_out_target_domain: str, query_source_domain: str
) -> SourcePredictorAudit:
    return SourcePredictorAudit(
        stage=value.stage,
        held_out_target_domain=held_out_target_domain,
        query_source_domain=query_source_domain,
        query_domains=value.query_domains,
        fit_domains=value.fit_domains,
        query_specimen_ids=value.query_specimen_ids,
        fit_specimen_ids=value.fit_specimen_ids,
        pca_dimension=value.pca_dimension,
        predictor_state_sha256=value.predictor_state_sha256,
    )


def _result_state(result: SourceLabelResult) -> str:
    digest = hashlib.sha256()
    header = {
        "candidate_bank_state_sha256": result.candidate_bank_state_sha256,
        "domain_order": result.domain_order,
        "fit_audits": [asdict(value) for value in result.fit_audits],
        "outer_domain": result.outer_domain,
        "rankings": [asdict(value) for value in result.rankings],
        "selection_audits": [asdict(value) for value in result.selection_audits],
        "source_domains": result.source_domains,
        "source_specimen_ids": result.source_specimen_ids,
        "stability": [asdict(value) for value in result.stability],
    }
    digest.update(
        json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    )
    for row in result.rows:
        digest.update(
            json.dumps(
                row, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("ascii")
        )
    return digest.hexdigest()


def _validate_inputs(
    *,
    outer_domain: str,
    domain_order: tuple[str, ...],
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    targets: object,
    metadata: object,
    bank: CandidateBank,
) -> tuple[np.ndarray, np.ndarray]:
    validate_candidate_bank(bank)
    count = len(specimen_ids)
    try:
        response = np.asarray(targets, dtype=np.float64)
        meta = np.asarray(metadata, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SourceLabelError("source label arrays must be numeric") from error
    if (
        type(domain_order) is not tuple
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
        or outer_domain not in domain_order
        or type(specimen_ids) is not tuple
        or type(dataset_ids) is not tuple
        or count < 1
        or len(set(specimen_ids)) != count
        or len(dataset_ids) != count
        or set(dataset_ids) != set(domain_order)
        or response.shape != (count,)
        or meta.ndim != 2
        or meta.shape[0] != count
        or not np.all(np.isfinite(response))
        or np.any(np.isinf(meta))
        or bank.specimen_ids != specimen_ids
        or bank.dataset_ids != dataset_ids
    ):
        raise SourceLabelError("source label authority changed")
    return response, meta


def _predict_candidates(
    model: CAIPredictor, metadata: np.ndarray, embeddings: np.ndarray
) -> np.ndarray:
    rows = embeddings.shape[0]
    flat_embeddings = np.ascontiguousarray(embeddings.reshape(rows * 64, 512))
    repeated_metadata = np.repeat(metadata, 64, axis=0)
    predictions: list[np.ndarray] = []
    for start in range(0, flat_embeddings.shape[0], 4096):
        predictions.append(
            np.asarray(
                model.predict(
                    repeated_metadata[start : start + 4096],
                    flat_embeddings[start : start + 4096],
                ),
                dtype=np.float64,
            )
        )
    output = np.concatenate(predictions).reshape(rows, 64)
    if not np.all(np.isfinite(output)):
        raise SourceLabelError("source candidate predictions are nonfinite")
    return output


def validate_source_label_result(result: SourceLabelResult) -> None:
    """Recompute row, roster, ranking, stability, and digest invariants."""

    if type(result) is not SourceLabelResult:
        raise SourceLabelError("issued source label result is required")
    if (
        len(result.domain_order) != 6
        or result.outer_domain not in result.domain_order
        or result.source_domains
        != tuple(
            domain for domain in result.domain_order if domain != result.outer_domain
        )
        or len(set(result.source_specimen_ids)) != len(result.source_specimen_ids)
        or len(result.fit_audits) != 5
    ):
        raise SourceLabelError("source label roster changed")
    predictor_by_domain: dict[str, str] = {}
    for audit in result.fit_audits:
        if (
            audit.stage != "outer"
            or audit.held_out_target_domain != result.outer_domain
            or audit.query_source_domain not in result.source_domains
            or audit.query_domains != (audit.query_source_domain,)
            or len(audit.fit_domains) != 4
            or result.outer_domain in audit.fit_domains
            or audit.query_source_domain in audit.fit_domains
            or set(audit.query_domains) & set(audit.fit_domains)
        ):
            raise SourceLabelError("source label fit barrier changed")
        predictor_by_domain[audit.query_source_domain] = audit.predictor_state_sha256
    if set(predictor_by_domain) != set(result.source_domains):
        raise SourceLabelError("source label predictor roster changed")
    for audit in result.selection_audits:
        if (
            audit.stage != "inner"
            or audit.held_out_target_domain != result.outer_domain
            or result.outer_domain in audit.fit_domains
            or audit.query_source_domain in audit.fit_domains
            or set(audit.query_domains) & set(audit.fit_domains)
        ):
            raise SourceLabelError("source label selection barrier changed")

    grouped: dict[tuple[str, str], set[int]] = {}
    observed_specimens: set[str] = set()
    for row in result.rows:
        if set(row) != _ROW_KEYS:
            raise SourceLabelError("source label row schema changed")
        specimen_id = row["specimen_id"]
        dataset_id = row["dataset_id"]
        method = row["method"]
        cell_index = row["cell_index"]
        primary = row["primary_value"]
        if (
            type(specimen_id) is not str
            or specimen_id not in result.source_specimen_ids
            or dataset_id not in result.source_domains
            or row["outer_domain"] != result.outer_domain
            or method not in METHODS
            or type(cell_index) is not int
            or not 0 <= cell_index < 64
            or row["from_level"] != 0
            or row["to_level"] != 1
            or type(row["added_measurements"]) is not int
            or row["added_measurements"] <= 0
            or row["candidate_bank_state_sha256"]
            != result.candidate_bank_state_sha256
            or isinstance(primary, bool)
            or not isinstance(primary, (int, float))
            or not math.isfinite(float(primary))
        ):
            raise SourceLabelError("source label row changed")
        key = (str(method), str(specimen_id))
        cells = grouped.setdefault(key, set())
        if cell_index in cells:
            raise SourceLabelError("source label cell is duplicated")
        cells.add(cell_index)
        observed_specimens.add(str(specimen_id))
        if method == "global_mechanical_mask":
            try:
                before = float(row["absolute_error_before"])
                after = float(row["absolute_error_after"])
                diagnostics_finite = all(
                    math.isfinite(float(row[name]))
                    for name in (
                        "current_prediction",
                        "candidate_prediction",
                        "secondary_value",
                    )
                )
            except (TypeError, ValueError) as error:
                raise SourceLabelError("mechanical source value changed") from error
            if (
                row["predictor_state_sha256"] != predictor_by_domain[dataset_id]
                or not diagnostics_finite
                or not math.isfinite(before)
                or not math.isfinite(after)
                or not math.isclose(
                    float(primary), before - after, rel_tol=0.0, abs_tol=1.0e-15
                )
            ):
                raise SourceLabelError("mechanical source value changed")
        elif any(
            row[name] is not None
            for name in (
                "absolute_error_after",
                "absolute_error_before",
                "candidate_prediction",
                "current_prediction",
                "predictor_state_sha256",
                "secondary_value",
            )
        ):
            raise SourceLabelError("comparator source value changed")
    if (
        observed_specimens != set(result.source_specimen_ids)
        or len(grouped) != len(METHODS) * len(result.source_specimen_ids)
        or any(cells != set(range(64)) for cells in grouped.values())
    ):
        raise SourceLabelError("source candidate roster is incomplete")

    expected_rankings = tuple(
        aggregate_global_ranking(
            tuple(row for row in result.rows if row["method"] == method),
            outer_domain=result.outer_domain,
            method=method,
            domain_order=result.domain_order,
        )
        for method in METHODS
    )
    if result.rankings != expected_rankings:
        raise SourceLabelError("source ranking changed")
    expected_stability = tuple(
        value
        for ranking in expected_rankings
        for value in leave_one_source_domain_out_stability(
            tuple(row for row in result.rows if row["method"] == ranking.method),
            primary=ranking,
            domain_order=result.domain_order,
        )
    )
    if result.stability != expected_stability:
        raise SourceLabelError("source ranking stability changed")
    if _result_state(result) != result.state_sha256:
        raise SourceLabelError("source label content digest changed")


def generate_source_labels(
    *,
    outer_domain: str,
    domain_order: tuple[str, ...],
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    targets: object,
    metadata: object,
    bank: CandidateBank,
    pca_dimensions: tuple[int, ...],
    ridge_alpha: float,
    tie_tolerance: float = 1.0e-12,
) -> SourceLabelResult:
    """Generate strict four-domain P-A labels for each of five source domains."""

    response, meta = _validate_inputs(
        outer_domain=outer_domain,
        domain_order=domain_order,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=targets,
        metadata=metadata,
        bank=bank,
    )
    dataset_array = np.asarray(dataset_ids, dtype=object)
    source_indices = np.flatnonzero(dataset_array != outer_domain)
    source_domains = tuple(domain for domain in domain_order if domain != outer_domain)
    source_specimen_ids = tuple(specimen_ids[index] for index in source_indices)
    source_dataset_ids = tuple(dataset_ids[index] for index in source_indices)
    source_targets = response[source_indices]
    source_metadata = meta[source_indices]
    source_initial_embeddings = bank.initial_embeddings[source_indices]
    source_dataset_array = np.asarray(source_dataset_ids, dtype=object)

    mechanical: dict[tuple[int, int], dict[str, object]] = {}
    fit_audits: list[SourcePredictorAudit] = []
    selection_audits: list[SourcePredictorAudit] = []
    for query_domain in source_domains:
        fitted = fit_outer_source_predictor(
            method=f"MVA_A4_P_A_{outer_domain}_{query_domain}",
            outer_domain=query_domain,
            specimen_ids=source_specimen_ids,
            dataset_ids=source_dataset_ids,
            domain_order=source_domains,
            targets=source_targets,
            metadata=source_metadata,
            embeddings=source_initial_embeddings,
            pca_dimensions=pca_dimensions,
            ridge_alpha=ridge_alpha,
            tie_tolerance=tie_tolerance,
        )
        final_audits = tuple(
            value for value in fitted.fit_audits if value.stage == "outer"
        )
        if len(final_audits) != 1:
            raise SourceLabelError("source label final fit audit is incomplete")
        fit_audits.append(
            _audit(
                final_audits[0],
                held_out_target_domain=outer_domain,
                query_source_domain=query_domain,
            )
        )
        selection_audits.extend(
            _audit(
                value,
                held_out_target_domain=outer_domain,
                query_source_domain=query_domain,
            )
            for value in fitted.fit_audits
            if value.stage == "inner"
        )
        query_local = np.flatnonzero(source_dataset_array == query_domain)
        query_global = source_indices[query_local]
        current_predictions = fitted.model.predict(
            meta[query_global], bank.initial_embeddings[query_global]
        )
        candidate_predictions = _predict_candidates(
            fitted.model,
            meta[query_global],
            bank.embeddings[query_global],
        )
        for local_row, global_index in enumerate(query_global):
            target = float(response[global_index])
            current_prediction = float(current_predictions[local_row])
            before = abs(target - current_prediction)
            for cell_index in range(64):
                candidate_prediction = float(
                    candidate_predictions[local_row, cell_index]
                )
                after = abs(target - candidate_prediction)
                mechanical[(int(global_index), cell_index)] = {
                    "absolute_error_after": after,
                    "absolute_error_before": before,
                    "candidate_prediction": candidate_prediction,
                    "current_prediction": current_prediction,
                    "predictor_state_sha256": fitted.model.state_sha256,
                    "primary_value": before - after,
                    "secondary_value": (target - current_prediction) ** 2
                    - (target - candidate_prediction) ** 2,
                }

    rows: list[dict[str, object]] = []
    for method in METHODS:
        for global_index in source_indices:
            for cell_index in range(64):
                diagnostics = (
                    mechanical[(int(global_index), cell_index)]
                    if method == "global_mechanical_mask"
                    else {
                        "absolute_error_after": None,
                        "absolute_error_before": None,
                        "candidate_prediction": None,
                        "current_prediction": None,
                        "predictor_state_sha256": None,
                        "primary_value": float(
                            bank.appearance_values[global_index, cell_index]
                            if method == "global_appearance_mask"
                            else bank.reconstruction_values[
                                global_index, cell_index
                            ]
                        ),
                        "secondary_value": None,
                    }
                )
                rows.append(
                    {
                        "specimen_id": specimen_ids[global_index],
                        "dataset_id": dataset_ids[global_index],
                        "outer_domain": outer_domain,
                        "method": method,
                        "cell_index": cell_index,
                        "from_level": 0,
                        "to_level": 1,
                        "added_measurements": int(
                            bank.added_measurements[global_index, cell_index]
                        ),
                        "candidate_bank_state_sha256": bank.state_sha256,
                        **diagnostics,
                    }
                )

    frozen_rows = tuple(rows)
    rankings = tuple(
        aggregate_global_ranking(
            tuple(row for row in frozen_rows if row["method"] == method),
            outer_domain=outer_domain,
            method=method,
            domain_order=domain_order,
        )
        for method in METHODS
    )
    stability = tuple(
        value
        for ranking in rankings
        for value in leave_one_source_domain_out_stability(
            tuple(row for row in frozen_rows if row["method"] == ranking.method),
            primary=ranking,
            domain_order=domain_order,
        )
    )
    result = SourceLabelResult(
        outer_domain=outer_domain,
        domain_order=domain_order,
        source_domains=source_domains,
        source_specimen_ids=source_specimen_ids,
        candidate_bank_state_sha256=bank.state_sha256,
        rows=frozen_rows,
        fit_audits=tuple(fit_audits),
        selection_audits=tuple(selection_audits),
        rankings=rankings,
        stability=stability,
        state_sha256="",
    )
    output = replace(result, state_sha256=_result_state(result))
    validate_source_label_result(output)
    return output


__all__ = [
    "METHODS",
    "SourceLabelError",
    "SourceLabelResult",
    "SourcePredictorAudit",
    "generate_source_labels",
    "validate_source_label_result",
]
