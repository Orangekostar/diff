"""Strict-OOF conditional mechanical value teachers for source states."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from cmc_bbdm.mva.a4_candidate_bank import load_candidate_bank
from cmc_bbdm.mva.cai_evaluator import CAIPredictor
from cmc_bbdm.mva.crossfit import fit_outer_source_predictor
from cmc_bbdm.mva.measurement_state import RefinementAction


class MAVISTeacherError(ValueError):
    """Raised when a teacher fit or source-only value label is invalid."""


@dataclass(frozen=True, slots=True)
class TeacherFitAudit:
    held_out_target_domain: str
    query_source_domain: str
    query_domains: tuple[str, ...]
    fit_domains: tuple[str, ...]
    query_specimen_ids: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    selected_pca_dimension: int
    predictor_state_sha256: str


@dataclass(frozen=True, slots=True)
class StrictOOFTeacher:
    outer_domain: str
    query_domain: str
    model: CAIPredictor
    audit: TeacherFitAudit
    state_sha256: str


@dataclass(frozen=True, slots=True)
class TeacherActionValue:
    specimen_id: str
    dataset_id: str
    action: RefinementAction
    exact_added_cost: int
    true_cai: float
    current_prediction: float
    candidate_prediction: float
    error_before: float
    error_after: float
    primary_value: float
    secondary_value: float
    predictor_state_sha256: str


@dataclass(frozen=True, slots=True)
class FoldStateLabels:
    outer_domain: str
    query_domain: str
    current_prediction: float
    teacher_state_sha256: str
    predictor_state_sha256: str
    action_values: tuple[TeacherActionValue, ...]


@dataclass(frozen=True, slots=True, eq=False)
class TeacherRegistry:
    domain_order: tuple[str, ...]
    teachers: MappingProxyType
    fit_audits: tuple[TeacherFitAudit, ...]
    initial_embedding_state_sha256: str
    state_sha256: str

    def teacher(self, outer_domain: str, query_domain: str) -> StrictOOFTeacher:
        try:
            return self.teachers[(outer_domain, query_domain)]
        except (KeyError, TypeError) as error:
            raise MAVISTeacherError("strict-OOF teacher is unavailable") from error

    def __eq__(self, other: object) -> bool:
        return type(other) is TeacherRegistry and self.state_sha256 == other.state_sha256


@dataclass(frozen=True, slots=True, eq=False)
class RegisteredInitialEmbeddings:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    initial_budgets: tuple[float, ...]
    embeddings: np.ndarray
    bank_state_sha256: MappingProxyType
    state_sha256: str

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is RegisteredInitialEmbeddings
            and self.state_sha256 == other.state_sha256
        )


def _state(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bound_source(config: object, root: Path, name: str) -> Path:
    try:
        binding = config.sources[name]
        path = root / binding.path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except (AttributeError, KeyError, OSError, TypeError) as error:
        raise MAVISTeacherError(f"teacher source {name} is unavailable") from error
    if actual != binding.sha256:
        raise MAVISTeacherError(f"teacher source {name} hash changed")
    return path


def load_registered_initial_embeddings(
    config: object,
    authority: object,
    *,
    project_root: str | Path,
) -> RegisteredInitialEmbeddings:
    from .authority import MAVISAuthority
    from .config import MAVISConfig

    if type(config) is not MAVISConfig or type(authority) is not MAVISAuthority:
        raise MAVISTeacherError("issued config and authority are required")
    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        raise MAVISTeacherError("teacher project root is unavailable") from error
    candidate_semantic_decoded_hashes = tuple(
        hashlib.sha256(
            authority.source_teacher_view(specimen_id).full_scan.tobytes(order="C")
        ).hexdigest()
        for specimen_id in authority.specimen_ids
    )
    bank_by_budget = {}
    for budget, source_name in (
        (0.015625, "candidate_bank_0p015625"),
        (0.03125, "candidate_bank_0p03125"),
    ):
        bank = load_candidate_bank(
            _bound_source(config, root, source_name),
            expected_specimen_ids=authority.specimen_ids,
            expected_image_sha256=authority.source_image_sha256,
            expected_initial_budget=budget,
        )
        if (
            bank.dataset_ids != authority.dataset_ids
            or bank.decoded_image_sha256 != candidate_semantic_decoded_hashes
            or any(
                shape != authority.policy_context(specimen_id).native_shape
                for specimen_id, shape in zip(
                    authority.specimen_ids, bank.native_shapes, strict=True
                )
            )
        ):
            raise MAVISTeacherError("teacher candidate-bank authority changed")
        bank_by_budget[budget] = bank
    initial_budgets = tuple(
        float(config.initial_budget_by_domain[domain])
        for domain in authority.dataset_ids
    )
    if set(initial_budgets) != set(bank_by_budget):
        raise MAVISTeacherError("teacher initial-budget roster changed")
    embeddings = np.empty((authority.specimen_count, 512), dtype="<f8")
    for index, budget in enumerate(initial_budgets):
        embeddings[index] = bank_by_budget[budget].initial_embeddings[index]
    if not np.all(np.isfinite(embeddings)):
        raise MAVISTeacherError("teacher initial embeddings are invalid")
    frozen = np.frombuffer(
        embeddings.tobytes(order="C"), dtype=embeddings.dtype
    ).reshape(embeddings.shape)
    frozen.setflags(write=False)
    bank_states = MappingProxyType(
        {
            budget: bank_by_budget[budget].state_sha256
            for budget in sorted(bank_by_budget)
        }
    )
    state = _state(
        {
            "schema": 1,
            "specimen_ids": authority.specimen_ids,
            "dataset_ids": authority.dataset_ids,
            "initial_budgets": initial_budgets,
            "bank_state_sha256": tuple(bank_states.items()),
            "embeddings_sha256": hashlib.sha256(
                frozen.tobytes(order="C")
            ).hexdigest(),
        }
    )
    return RegisteredInitialEmbeddings(
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        initial_budgets=initial_budgets,
        embeddings=frozen,
        bank_state_sha256=bank_states,
        state_sha256=state,
    )


def fit_strict_oof_teacher(
    *,
    outer_domain: str,
    query_domain: str,
    domain_order: tuple[str, ...],
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    targets: object,
    metadata: object,
    initial_embeddings: object,
    pca_dimensions: tuple[int, ...],
    ridge_alpha: float,
    tie_tolerance: float = 1.0e-12,
) -> StrictOOFTeacher:
    if (
        type(domain_order) is not tuple
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
        or outer_domain not in domain_order
        or query_domain not in domain_order
        or outer_domain == query_domain
        or type(specimen_ids) is not tuple
        or type(dataset_ids) is not tuple
        or len(specimen_ids) != len(dataset_ids)
    ):
        raise MAVISTeacherError("teacher domain or specimen roster is invalid")
    dataset_array = np.asarray(dataset_ids, dtype=object)
    source_indices = np.flatnonzero(dataset_array != outer_domain)
    source_domains = tuple(domain for domain in domain_order if domain != outer_domain)
    response = np.asarray(targets, dtype=np.float64)
    meta = np.asarray(metadata, dtype=np.float64)
    embeddings = np.asarray(initial_embeddings, dtype=np.float64)
    if (
        response.shape != (len(specimen_ids),)
        or meta.ndim != 2
        or meta.shape[0] != len(specimen_ids)
        or embeddings.ndim != 2
        or embeddings.shape[0] != len(specimen_ids)
        or not np.all(np.isfinite(response))
        or not np.all(np.isfinite(embeddings))
    ):
        raise MAVISTeacherError("teacher arrays are invalid")
    fitted = fit_outer_source_predictor(
        method=f"MAVIS_TEACHER_{outer_domain}_{query_domain}",
        outer_domain=query_domain,
        specimen_ids=tuple(specimen_ids[index] for index in source_indices),
        dataset_ids=tuple(dataset_ids[index] for index in source_indices),
        domain_order=source_domains,
        targets=response[source_indices],
        metadata=meta[source_indices],
        embeddings=embeddings[source_indices],
        pca_dimensions=pca_dimensions,
        ridge_alpha=ridge_alpha,
        tie_tolerance=tie_tolerance,
    )
    final_audits = tuple(audit for audit in fitted.fit_audits if audit.stage == "outer")
    if len(final_audits) != 1:
        raise MAVISTeacherError("teacher final fit audit is incomplete")
    source_audit = final_audits[0]
    audit = TeacherFitAudit(
        held_out_target_domain=outer_domain,
        query_source_domain=query_domain,
        query_domains=source_audit.query_domains,
        fit_domains=source_audit.fit_domains,
        query_specimen_ids=source_audit.query_specimen_ids,
        fit_specimen_ids=source_audit.fit_specimen_ids,
        selected_pca_dimension=fitted.selected_pca_dimension,
        predictor_state_sha256=fitted.model.state_sha256,
    )
    if (
        audit.query_domains != (query_domain,)
        or outer_domain in audit.fit_domains
        or query_domain in audit.fit_domains
        or set(audit.query_specimen_ids) & set(audit.fit_specimen_ids)
        or set(audit.fit_domains) != set(domain_order) - {outer_domain, query_domain}
    ):
        raise MAVISTeacherError("strict-OOF teacher barrier failed")
    state = _state(
        {
            "schema": 1,
            "outer_domain": outer_domain,
            "query_domain": query_domain,
            "fit_domains": audit.fit_domains,
            "query_specimen_ids": audit.query_specimen_ids,
            "fit_specimen_ids": audit.fit_specimen_ids,
            "selected_pca_dimension": audit.selected_pca_dimension,
            "predictor_state_sha256": audit.predictor_state_sha256,
        }
    )
    return StrictOOFTeacher(
        outer_domain=outer_domain,
        query_domain=query_domain,
        model=fitted.model,
        audit=audit,
        state_sha256=state,
    )


def predict_teacher_state(
    teacher: StrictOOFTeacher,
    *,
    specimen_id: str,
    dataset_id: str,
    metadata: object,
    current_embedding: object,
) -> float:
    if (
        type(teacher) is not StrictOOFTeacher
        or dataset_id != teacher.query_domain
        or specimen_id not in teacher.audit.query_specimen_ids
    ):
        raise MAVISTeacherError("teacher query roster is invalid")
    meta = np.asarray(metadata, dtype=np.float64)
    current = np.asarray(current_embedding, dtype=np.float64)
    embedding_dimension = int(teacher.model.pca.mean.size)
    if (
        meta.shape != (teacher.model.metadata_features,)
        or current.shape != (embedding_dimension,)
        or not np.all(np.isfinite(meta))
        or not np.all(np.isfinite(current))
    ):
        raise MAVISTeacherError("teacher query arrays are invalid")
    return float(teacher.model.predict(meta[None, :], current[None, :])[0])


def fit_teacher_registry(
    config: object,
    authority: object,
    initial_embeddings: RegisteredInitialEmbeddings,
) -> TeacherRegistry:
    from .authority import MAVISAuthority
    from .config import MAVISConfig

    if (
        type(config) is not MAVISConfig
        or type(authority) is not MAVISAuthority
        or type(initial_embeddings) is not RegisteredInitialEmbeddings
        or initial_embeddings.specimen_ids != authority.specimen_ids
        or initial_embeddings.dataset_ids != authority.dataset_ids
    ):
        raise MAVISTeacherError("teacher registry inputs are invalid")
    metadata = np.vstack(
        [
            authority.policy_context(specimen_id).context_features[:13]
            for specimen_id in authority.specimen_ids
        ]
    )
    targets = np.asarray(
        [
            authority.source_teacher_view(specimen_id).true_cai
            for specimen_id in authority.specimen_ids
        ],
        dtype=np.float64,
    )
    teachers: dict[tuple[str, str], StrictOOFTeacher] = {}
    audits: list[TeacherFitAudit] = []
    for outer_domain in config.domain_order:
        for query_domain in config.domain_order:
            if query_domain == outer_domain:
                continue
            teacher = fit_strict_oof_teacher(
                outer_domain=outer_domain,
                query_domain=query_domain,
                domain_order=config.domain_order,
                specimen_ids=authority.specimen_ids,
                dataset_ids=authority.dataset_ids,
                targets=targets,
                metadata=metadata,
                initial_embeddings=initial_embeddings.embeddings,
                pca_dimensions=config.teacher_pca_dimensions,
                ridge_alpha=config.teacher_ridge_alpha,
                tie_tolerance=config.teacher_tie_tolerance,
            )
            teachers[(outer_domain, query_domain)] = teacher
            audits.append(teacher.audit)
    if len(teachers) != 30 or len({audit.predictor_state_sha256 for audit in audits}) != 30:
        raise MAVISTeacherError("teacher registry is incomplete")
    state = _state(
        {
            "schema": 1,
            "domain_order": config.domain_order,
            "teachers": tuple(
                (outer, query, teachers[(outer, query)].state_sha256)
                for outer in config.domain_order
                for query in config.domain_order
                if query != outer
            ),
            "initial_embedding_state_sha256": initial_embeddings.state_sha256,
        }
    )
    return TeacherRegistry(
        domain_order=config.domain_order,
        teachers=MappingProxyType(teachers),
        fit_audits=tuple(audits),
        initial_embedding_state_sha256=initial_embeddings.state_sha256,
        state_sha256=state,
    )


def label_teacher_candidates(
    teacher: StrictOOFTeacher,
    *,
    specimen_id: str,
    dataset_id: str,
    true_cai: float,
    metadata: object,
    current_embedding: object,
    candidate_embeddings: object,
    actions: tuple[RefinementAction, ...],
    candidate_costs: tuple[int, ...],
) -> tuple[TeacherActionValue, ...]:
    if (
        type(teacher) is not StrictOOFTeacher
        or dataset_id != teacher.query_domain
        or specimen_id not in teacher.audit.query_specimen_ids
        or type(actions) is not tuple
        or type(candidate_costs) is not tuple
        or len(actions) != len(candidate_costs)
        or any(type(action) is not RefinementAction for action in actions)
        or any(type(cost) is not int or cost <= 0 for cost in candidate_costs)
    ):
        raise MAVISTeacherError("teacher query roster is invalid")
    target = float(true_cai)
    meta = np.asarray(metadata, dtype=np.float64)
    current = np.asarray(current_embedding, dtype=np.float64)
    candidates = np.asarray(candidate_embeddings, dtype=np.float64)
    embedding_dimension = int(teacher.model.pca.mean.size)
    if (
        not math.isfinite(target)
        or meta.shape != (teacher.model.metadata_features,)
        or current.shape != (embedding_dimension,)
        or candidates.shape != (len(actions), embedding_dimension)
        or not np.all(np.isfinite(meta))
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(candidates))
    ):
        raise MAVISTeacherError("teacher query arrays are invalid")
    current_prediction = predict_teacher_state(
        teacher,
        specimen_id=specimen_id,
        dataset_id=dataset_id,
        metadata=meta,
        current_embedding=current,
    )
    if not actions:
        return ()
    candidate_predictions = teacher.model.predict(
        np.repeat(meta[None, :], len(actions), axis=0), candidates
    )
    error_before = abs(target - current_prediction)
    output: list[TeacherActionValue] = []
    for action, cost, candidate_prediction in zip(
        actions, candidate_costs, candidate_predictions, strict=True
    ):
        prediction = float(candidate_prediction)
        error_after = abs(target - prediction)
        output.append(
            TeacherActionValue(
                specimen_id=specimen_id,
                dataset_id=dataset_id,
                action=action,
                exact_added_cost=cost,
                true_cai=target,
                current_prediction=current_prediction,
                candidate_prediction=prediction,
                error_before=error_before,
                error_after=error_after,
                primary_value=error_before - error_after,
                secondary_value=(target - current_prediction) ** 2
                - (target - prediction) ** 2,
                predictor_state_sha256=teacher.model.state_sha256,
            )
        )
    return tuple(output)


def label_teacher_state(
    teacher: StrictOOFTeacher,
    *,
    specimen_id: str,
    dataset_id: str,
    true_cai: float,
    metadata: object,
    current_embedding: object,
    candidate_embeddings: object,
    actions: tuple[RefinementAction, ...],
    candidate_costs: tuple[int, ...],
) -> FoldStateLabels:
    current_prediction = predict_teacher_state(
        teacher,
        specimen_id=specimen_id,
        dataset_id=dataset_id,
        metadata=metadata,
        current_embedding=current_embedding,
    )
    values = label_teacher_candidates(
        teacher,
        specimen_id=specimen_id,
        dataset_id=dataset_id,
        true_cai=true_cai,
        metadata=metadata,
        current_embedding=current_embedding,
        candidate_embeddings=candidate_embeddings,
        actions=actions,
        candidate_costs=candidate_costs,
    )
    if values and any(
        value.current_prediction != current_prediction for value in values
    ):
        raise MAVISTeacherError("teacher state prediction changed across candidates")
    return FoldStateLabels(
        outer_domain=teacher.outer_domain,
        query_domain=teacher.query_domain,
        current_prediction=current_prediction,
        teacher_state_sha256=teacher.state_sha256,
        predictor_state_sha256=teacher.model.state_sha256,
        action_values=values,
    )


__all__ = [
    "FoldStateLabels",
    "MAVISTeacherError",
    "RegisteredInitialEmbeddings",
    "StrictOOFTeacher",
    "TeacherActionValue",
    "TeacherFitAudit",
    "TeacherRegistry",
    "fit_strict_oof_teacher",
    "fit_teacher_registry",
    "label_teacher_candidates",
    "label_teacher_state",
    "load_registered_initial_embeddings",
    "predict_teacher_state",
]
