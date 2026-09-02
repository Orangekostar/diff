"""Source-only engineering selection for conditionally authorized AAWR actors."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engineering_selection_execution import (
    G1EngineeringCandidateRun,
)
from .g1 import G1Protocol, G1Runtime, G1SourceDependencies
from .policy_selection import (
    OuterPolicySelection,
    PolicyCandidateEvaluation,
    outer_selection_payload,
    select_outer_policy,
)
from .policy_training import (
    PolicyTrainingHyperparameters,
    fit_inner_observable_policy,
    rebind_training_example_modes,
)
from .privileged_awr import (
    AAWR_TRAJECTORY_SOURCES,
    REGISTERED_BETAS,
    REGISTERED_EXPECTILES,
    AAWRAuthorization,
    G1AAWRTrainingRecord,
    aawr_policy_hyperparameters,
    fit_inner_aawr_policy,
    read_aawr_fit_evidence,
    write_aawr_fit_evidence,
)
from .source_policy_evaluation import (
    build_g1_learned_source_bank,
    evaluate_inner_policy_bridge,
    learned_source_bank_path,
    read_learned_source_bank,
)
from .teacher_bank import G1TeacherBankRecord


class G1AAWRSelectionExecutionError(ValueError):
    """Raised when AAWR source selection crosses an outer-target boundary."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class G1OuterAAWRSelectionRun:
    outer_target: str
    authorization: AAWRAuthorization
    base_candidate: G1EngineeringCandidateRun
    candidates: tuple[G1EngineeringCandidateRun, ...]
    fit_evidence_sha256s: tuple[tuple[str, ...], ...]
    selection: OuterPolicySelection
    transition_record_count: int
    path: Path
    target_outcomes_opened: bool = False

    def __post_init__(self) -> None:
        expected_hyperparameters = {
            row.candidate.hyperparameters.state_sha256 for row in self.candidates
        }
        selection_hyperparameters = {
            row.hyperparameters.state_sha256 for row in self.selection.candidates
        }
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.authorization) is not AAWRAuthorization
            or self.authorization.outer_target != self.outer_target
            or self.authorization.status != "AUTHORIZED_SOURCE_ONLY"
            or type(self.base_candidate) is not G1EngineeringCandidateRun
            or self.base_candidate.candidate.outer_target != self.outer_target
            or type(self.candidates) is not tuple
            or len(self.candidates) != 4
            or any(
                type(row) is not G1EngineeringCandidateRun
                or row.candidate.outer_target != self.outer_target
                for row in self.candidates
            )
            or len(expected_hyperparameters) != 4
            or type(self.fit_evidence_sha256s) is not tuple
            or len(self.fit_evidence_sha256s) != 4
            or any(
                type(values) is not tuple
                or len(values) != 5
                or not all(_valid_sha256(value) for value in values)
                for values in self.fit_evidence_sha256s
            )
            or type(self.selection) is not OuterPolicySelection
            or self.selection.outer_target != self.outer_target
            or selection_hyperparameters
            != {
                self.base_candidate.candidate.hyperparameters.state_sha256,
                *expected_hyperparameters,
            }
            or type(self.transition_record_count) is not int
            or self.transition_record_count <= 0
            or not isinstance(self.path, Path)
            or self.target_outcomes_opened
        ):
            raise G1AAWRSelectionExecutionError(
                "outer AAWR selection run is invalid"
            )


def materialize_g1_aawr_training_records(
    runtime: G1Runtime,
    protocol: G1Protocol,
    records: tuple[G1TeacherBankRecord, ...],
    *,
    encoder: object,
    base_hyperparameters: PolicyTrainingHyperparameters,
    authorization: AAWRAuthorization,
) -> tuple[G1AAWRTrainingRecord, ...]:
    domain_order = tuple(getattr(protocol, "domain_order", ()))
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or len(domain_order) != 6
        or type(records) is not tuple
        or not records
        or any(type(row) is not G1TeacherBankRecord for row in records)
        or len({row.state_sha256 for row in records}) != len(records)
        or type(base_hyperparameters) is not PolicyTrainingHyperparameters
        or type(authorization) is not AAWRAuthorization
        or authorization.status != "AUTHORIZED_SOURCE_ONLY"
        or authorization.outer_target not in domain_order
        or not callable(getattr(encoder, "encode", None))
    ):
        raise G1AAWRSelectionExecutionError(
            "AAWR transition materialization request is invalid"
        )
    outer_target = authorization.outer_target
    selected = tuple(
        row
        for row in records
        if row.example.dagger_iteration <= base_hyperparameters.dagger_iterations
        and row.example.task in authorization.authorized_tasks
    )
    source_domains = tuple(domain for domain in domain_order if domain != outer_target)
    required_specimens = {row.example.specimen_sha256 for row in selected}
    if (
        not selected
        or {row.example.source_domain for row in selected} != set(source_domains)
        or any(
            row.example.outer_target != outer_target
            or row.example.source_domain == outer_target
            for row in selected
        )
    ):
        raise G1AAWRSelectionExecutionError("AAWR source transition roster changed")
    specimen_payloads = []
    for dataset_id, specimen_id in zip(
        runtime.mavis.dataset_ids,
        runtime.mavis.specimen_ids,
        strict=True,
    ):
        if dataset_id == outer_target:
            continue
        specimen_sha256 = runtime.specimen_sha256(dataset_id, specimen_id)
        if specimen_sha256 not in required_specimens:
            continue
        view = runtime.mavis.source_teacher_view(specimen_id)
        specimen_payloads.append(
            (
                specimen_sha256,
                view.full_scan,
                float(view.true_cai),
            )
        )
    specimen_payloads.sort(key=lambda row: row[0])
    if (
        {row[0] for row in specimen_payloads} != required_specimens
        or len(specimen_payloads) != len(required_specimens)
    ):
        raise G1AAWRSelectionExecutionError(
            "AAWR source full-scan roster is incomplete"
        )
    embeddings: dict[str, np.ndarray] = {}
    true_cai: dict[str, float] = {}
    batch_size = int(protocol.encoder_batch_size)
    for start in range(0, len(specimen_payloads), batch_size):
        batch = specimen_payloads[start : start + batch_size]
        values = np.asarray(
            encoder.encode(tuple(row[1] for row in batch)), dtype=np.float64
        )
        if values.shape != (len(batch), 512) or not np.all(np.isfinite(values)):
            raise G1AAWRSelectionExecutionError(
                "AAWR full-scan embedding batch is invalid"
            )
        for payload, embedding in zip(batch, values, strict=True):
            specimen_sha256, _full_scan, cai = payload
            embeddings[specimen_sha256] = embedding
            true_cai[specimen_sha256] = cai
    output = tuple(
        G1AAWRTrainingRecord(
            example=rebind_training_example_modes(
                row.example,
                cai_context_mode=base_hyperparameters.cai_context_mode,
                task_token_mode=base_hyperparameters.task_token_mode,
            ),
            state_source=row.state_source,
            source_record_sha256=row.state_sha256,
            full_scan_embedding=embeddings[row.example.specimen_sha256],
            true_cai=(
                true_cai[row.example.specimen_sha256]
                if row.example.task.value == "CAI"
                else None
            ),
        )
        for row in selected
    )
    state_sources = {row.state_source for row in output}
    fixed_sources = AAWR_TRAJECTORY_SOURCES - {
        "ORACLE_CHECKPOINT",
        "DAGGER_ACTOR_VISITED",
    }
    if (
        not state_sources & fixed_sources
        or "ORACLE_CHECKPOINT" not in state_sources
        or "DAGGER_ACTOR_VISITED" not in state_sources
    ):
        raise G1AAWRSelectionExecutionError(
            "AAWR requires fixed, oracle, and current-actor source trajectories"
        )
    return output


def _candidate_payload(
    run: G1EngineeringCandidateRun,
    fit_evidence_sha256s: tuple[str, ...],
) -> dict[str, object]:
    candidate = run.candidate
    hyperparameters = candidate.hyperparameters
    return {
        "hyperparameters_sha256": hyperparameters.state_sha256,
        "base_hyperparameters_sha256": (
            hyperparameters.base_hyperparameters_sha256
        ),
        "expectile": hyperparameters.aawr_expectile,
        "beta": hyperparameters.aawr_beta,
        "authorized_tasks": [
            task.value for task in hyperparameters.aawr_authorized_tasks
        ],
        "candidate_state_sha256": candidate.state_sha256,
        "run_state_sha256": run.state_sha256,
        "equal_domain_mean_relative_auebc": (
            candidate.equal_domain_mean_relative_auebc
        ),
        "mean_oracle_gap_closure": candidate.mean_oracle_gap_closure,
        "improved_source_domains": candidate.improved_source_domains,
        "final_refit_epochs": candidate.final_refit_epochs,
        "inner_evaluation_sha256s": list(run.inner_evaluation_sha256s),
        "learned_bank_manifest_sha256s": list(
            run.learned_bank_manifest_sha256s
        ),
        "fit_evidence_sha256s": list(fit_evidence_sha256s),
    }


def run_outer_aawr_selection(
    runtime: G1Runtime,
    protocol: G1Protocol,
    *,
    outer_target: str,
    authorization: AAWRAuthorization,
    base_candidate: G1EngineeringCandidateRun,
    teacher_records: tuple[G1TeacherBankRecord, ...],
    bridge_records: tuple[object, ...],
    dependency_factory: Callable[[str], G1SourceDependencies],
    encoder: object,
    learned_root: str | Path,
    work_root: str | Path,
    device: str,
    progress: Callable[[str], None] | None = None,
) -> G1OuterAAWRSelectionRun:
    domain_order = tuple(getattr(protocol, "domain_order", ()))
    base_hyperparameters = getattr(base_candidate.candidate, "hyperparameters", None)
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or len(domain_order) != 6
        or outer_target not in domain_order
        or type(authorization) is not AAWRAuthorization
        or authorization.outer_target != outer_target
        or authorization.status != "AUTHORIZED_SOURCE_ONLY"
        or type(base_candidate) is not G1EngineeringCandidateRun
        or base_candidate.candidate.outer_target != outer_target
        or type(base_hyperparameters) is not PolicyTrainingHyperparameters
        or type(teacher_records) is not tuple
        or not teacher_records
        or type(bridge_records) is not tuple
        or not bridge_records
        or not callable(dependency_factory)
        or not callable(getattr(encoder, "encode", None))
        or type(device) is not str
        or not device
        or (progress is not None and not callable(progress))
    ):
        raise G1AAWRSelectionExecutionError("outer AAWR selection request is invalid")
    source_domains = tuple(domain for domain in domain_order if domain != outer_target)
    selected_records = tuple(
        row
        for row in teacher_records
        if row.example.dagger_iteration <= base_hyperparameters.dagger_iterations
    )
    base_examples = tuple(
        rebind_training_example_modes(
            row.example,
            cai_context_mode=base_hyperparameters.cai_context_mode,
            task_token_mode=base_hyperparameters.task_token_mode,
        )
        for row in selected_records
    )
    transitions = materialize_g1_aawr_training_records(
        runtime,
        protocol,
        selected_records,
        encoder=encoder,
        base_hyperparameters=base_hyperparameters,
        authorization=authorization,
    )
    candidate_runs = []
    candidate_fit_evidence_sha256s = []
    for expectile in REGISTERED_EXPECTILES:
        for beta in REGISTERED_BETAS:
            hyperparameters = aawr_policy_hyperparameters(
                base_hyperparameters,
                authorization_sha256=authorization.state_sha256,
                authorized_tasks=authorization.authorized_tasks,
                expectile=expectile,
                beta=beta,
            )
            metrics = []
            evaluation_hashes = []
            bank_hashes = []
            fit_evidence_hashes = []
            for source in source_domains:
                path = learned_source_bank_path(
                    learned_root,
                    outer_target,
                    hyperparameters.state_sha256,
                    source,
                )
                fit_evidence_path = (
                    Path(work_root)
                    / outer_target
                    / hyperparameters.state_sha256
                    / f"{source}.fit.json"
                )
                if not path.is_file():
                    if progress is not None:
                        progress(
                            f"G1 AAWR fit {outer_target}/{source}: "
                            f"expectile={expectile}, beta={beta}"
                        )
                    base_actor = fit_inner_observable_policy(
                        base_examples,
                        validation_domain=source,
                        hyperparameters=base_hyperparameters,
                        max_epochs=int(protocol.epochs),
                        patience=int(protocol.patience),
                        device=device,
                    )
                    actor, fit_evidence = fit_inner_aawr_policy(
                        transitions,
                        base_actor=base_actor,
                        hyperparameters=hyperparameters,
                        validation_domain=source,
                        max_epochs=int(protocol.epochs),
                        patience=int(protocol.patience),
                        device=device,
                    )
                    write_aawr_fit_evidence(
                        fit_evidence_path,
                        fit_evidence,
                    )
                    dependencies = dependency_factory(source)
                    build_g1_learned_source_bank(
                        runtime,
                        protocol,
                        dependencies,
                        encoder=encoder,
                        actor=actor,
                        work_root=learned_root,
                        progress=progress,
                    )
                bank, learned_records = read_learned_source_bank(path)
                fit_evidence = read_aawr_fit_evidence(fit_evidence_path)
                model_sha256s = {row.model_state_sha256 for row in learned_records}
                if (
                    fit_evidence.outer_target != outer_target
                    or fit_evidence.validation_domain != source
                    or fit_evidence.hyperparameters_sha256
                    != hyperparameters.state_sha256
                    or model_sha256s != {fit_evidence.actor_model_sha256}
                ):
                    raise G1AAWRSelectionExecutionError(
                        "AAWR fit evidence and learned bank differ"
                    )
                evaluation = evaluate_inner_policy_bridge(
                    learned_records,
                    bridge_records,
                    hyperparameters=hyperparameters,
                )
                if (
                    evaluation.outer_target != outer_target
                    or evaluation.validation_domain != source
                    or evaluation.hyperparameters_sha256
                    != hyperparameters.state_sha256
                    or not _valid_sha256(evaluation.state_sha256)
                ):
                    raise G1AAWRSelectionExecutionError(
                        "AAWR engineering evaluation fold changed"
                    )
                metrics.extend(evaluation.metrics)
                evaluation_hashes.append(evaluation.state_sha256)
                bank_hashes.append(bank.manifest_sha256)
                fit_evidence_hashes.append(fit_evidence.state_sha256)
            candidate_runs.append(
                G1EngineeringCandidateRun(
                    candidate=PolicyCandidateEvaluation(
                        hyperparameters=hyperparameters,
                        inner_metrics=tuple(metrics),
                    ),
                    inner_evaluation_sha256s=tuple(evaluation_hashes),
                    learned_bank_manifest_sha256s=tuple(bank_hashes),
                )
            )
            candidate_fit_evidence_sha256s.append(tuple(fit_evidence_hashes))
    candidates = tuple(candidate_runs)
    fit_evidence_sha256s = tuple(candidate_fit_evidence_sha256s)
    selection = select_outer_policy(
        (base_candidate.candidate, *(row.candidate for row in candidates))
    )
    destination = Path(work_root) / outer_target / "selection.json"
    result = G1OuterAAWRSelectionRun(
        outer_target=outer_target,
        authorization=authorization,
        base_candidate=base_candidate,
        candidates=candidates,
        fit_evidence_sha256s=fit_evidence_sha256s,
        selection=selection,
        transition_record_count=len(transitions),
        path=destination,
        target_outcomes_opened=False,
    )
    _atomic_json(
        destination,
        {
            "schema_version": 1,
            "scope": "inspection_agent_g1_outer_aawr_selection",
            "outer_target": outer_target,
            "authorization_sha256": authorization.state_sha256,
            "authorized_tasks": [
                task.value for task in authorization.authorized_tasks
            ],
            "base_candidate_sha256": base_candidate.state_sha256,
            "transition_record_count": len(transitions),
            "transition_record_sha256s": [
                row.state_sha256 for row in transitions
            ],
            "trajectory_source_counts": {
                source: sum(row.state_source == source for row in transitions)
                for source in sorted({row.state_source for row in transitions})
            },
            "candidates": [
                _candidate_payload(row, evidence)
                for row, evidence in zip(
                    candidates,
                    fit_evidence_sha256s,
                    strict=True,
                )
            ],
            "selection": outer_selection_payload(selection),
            "target_outcomes_opened": False,
        },
    )
    return result


__all__ = [
    "G1AAWRSelectionExecutionError",
    "G1OuterAAWRSelectionRun",
    "materialize_g1_aawr_training_records",
    "run_outer_aawr_selection",
]
