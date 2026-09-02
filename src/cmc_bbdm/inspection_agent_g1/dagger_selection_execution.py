"""Source-only engineering selection across registered G1 DAgger iterations."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .dagger_orchestration import (
    G1OuterDaggerBuild,
    build_g1_outer_dagger_banks,
    with_dagger_iterations,
)
from .engineering_selection_execution import (
    G1EngineeringCandidateRun,
    G1OuterEngineeringSelectionRun,
    run_engineering_candidate,
    run_outer_engineering_selection,
)
from .g1 import (
    G1Protocol,
    G1Runtime,
    G1SourceDependencies,
    build_g1_source_dependencies,
)
from .policy_selection import (
    OuterPolicySelection,
    outer_selection_payload,
    select_outer_policy,
)
from .policy_training import TrainingRoute
from .privileged_awr import (
    AAWRAuthorization,
    AAWRSourceEvidence,
    authorize_conditional_aawr,
)
from .source_bridge import read_source_bridge_bank


class G1DaggerSelectionExecutionError(ValueError):
    """Raised when DAgger iteration selection leaves source-only evidence."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


@dataclass(frozen=True, slots=True)
class G1OuterDaggerSelectionRun:
    outer_target: str
    base_run: G1OuterEngineeringSelectionRun
    dagger_build: G1OuterDaggerBuild
    candidates: tuple[G1EngineeringCandidateRun, ...]
    selection: OuterPolicySelection
    aawr_authorization: AAWRAuthorization
    path: Path
    target_outcomes_opened: bool = False

    def __post_init__(self) -> None:
        iterations = tuple(
            row.candidate.hyperparameters.dagger_iterations for row in self.candidates
        )
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.base_run) is not G1OuterEngineeringSelectionRun
            or self.base_run.outer_target != self.outer_target
            or type(self.dagger_build) is not G1OuterDaggerBuild
            or self.dagger_build.outer_target != self.outer_target
            or type(self.candidates) is not tuple
            or len(self.candidates) != 3
            or iterations != (0, 1, 2)
            or any(
                type(row) is not G1EngineeringCandidateRun
                or row.candidate.outer_target != self.outer_target
                for row in self.candidates
            )
            or type(self.selection) is not OuterPolicySelection
            or self.selection.outer_target != self.outer_target
            or tuple(
                row.state_sha256 for row in self.selection.candidates
            )
            != tuple(
                row.state_sha256
                for row in sorted(
                    (candidate.candidate for candidate in self.candidates),
                    key=lambda value: value.hyperparameters.state_sha256,
                )
            )
            or type(self.aawr_authorization) is not AAWRAuthorization
            or self.aawr_authorization.outer_target != self.outer_target
            or not isinstance(self.path, Path)
            or self.target_outcomes_opened
        ):
            raise G1DaggerSelectionExecutionError(
                "outer DAgger selection run is invalid"
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


def _candidate_payload(run: G1EngineeringCandidateRun) -> dict[str, object]:
    candidate = run.candidate
    return {
        "hyperparameters_sha256": candidate.hyperparameters.state_sha256,
        "dagger_iterations": candidate.hyperparameters.dagger_iterations,
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
    }


def _aawr_evidence(
    selected: G1EngineeringCandidateRun,
) -> tuple[AAWRSourceEvidence, ...]:
    candidate = selected.candidate
    output = []
    for task in (InspectionTask.FIELD, InspectionTask.CAI):
        metrics = tuple(
            sorted(
                (
                    row
                    for row in candidate.inner_metrics
                    if row.task is task
                ),
                key=lambda row: row.validation_domain,
            )
        )
        if len(metrics) != 5:
            raise G1DaggerSelectionExecutionError(
                "DAgger AAWR source evidence is incomplete"
            )
        output.append(
            AAWRSourceEvidence(
                outer_target=candidate.outer_target,
                task=task,
                source_domains=tuple(row.validation_domain for row in metrics),
                fixed_auebc=tuple(row.fixed_auebc for row in metrics),
                policy_auebc=tuple(row.learned_auebc for row in metrics),
                oracle_auebc=tuple(row.oracle_auebc for row in metrics),
            )
        )
    return tuple(output)


def _aawr_payload(authorization: AAWRAuthorization) -> dict[str, object]:
    return {
        "status": authorization.status,
        "prerequisite_satisfied": authorization.prerequisite_satisfied,
        "authorized_tasks": [task.value for task in authorization.authorized_tasks],
        "state_sha256": authorization.state_sha256,
        "evidence": [
            {
                "task": row.task.value,
                "source_domains": list(row.source_domains),
                "equal_domain_effect": row.equal_domain_effect,
                "oracle_gap_available": row.oracle_gap_available,
                "oracle_gap_closure": row.oracle_gap_closure,
                "improved_domains": row.improved_domains,
                "positive_action_observability": row.positive_action_observability,
                "state_sha256": row.state_sha256,
            }
            for row in authorization.evidence
        ],
    }


def run_outer_dagger_selection(
    runtime: G1Runtime,
    protocol: G1Protocol,
    *,
    outer_target: str,
    encoder: object,
    teacher_bank_root: str | Path,
    bridge_root: str | Path,
    supervised_root: str | Path,
    learned_root: str | Path,
    base_work_root: str | Path,
    dagger_bank_root: str | Path,
    work_root: str | Path,
    device: str,
    progress: Callable[[str], None] | None = None,
) -> G1OuterDaggerSelectionRun:
    domain_order = tuple(getattr(protocol, "domain_order", ()))
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or len(domain_order) != 6
        or outer_target not in domain_order
        or not callable(getattr(encoder, "encode", None))
        or type(device) is not str
        or not device
        or (progress is not None and not callable(progress))
    ):
        raise G1DaggerSelectionExecutionError("outer DAgger selection request is invalid")
    source_domains = tuple(domain for domain in domain_order if domain != outer_target)
    base_run = run_outer_engineering_selection(
        runtime,
        protocol,
        outer_target=outer_target,
        encoder=encoder,
        teacher_bank_root=teacher_bank_root,
        bridge_root=bridge_root,
        supervised_root=supervised_root,
        learned_root=learned_root,
        work_root=base_work_root,
        device=device,
        progress=progress,
    )
    base_candidates = tuple(
        row
        for row in base_run.candidates
        if row.candidate.hyperparameters.state_sha256
        == base_run.selection.selected_hyperparameters_sha256
    )
    if (
        len(base_candidates) != 1
        or base_candidates[0].candidate.hyperparameters.dagger_iterations != 0
    ):
        raise G1DaggerSelectionExecutionError(
            "base engineering selection did not issue one iteration-zero policy"
        )
    base_candidate = base_candidates[0]
    base_hyperparameters = base_candidate.candidate.hyperparameters
    dependency_cache: dict[str, G1SourceDependencies] = {}

    def dependency_factory(source: str) -> G1SourceDependencies:
        if source not in source_domains:
            raise G1DaggerSelectionExecutionError(
                "DAgger requested a non-source dependency"
            )
        if source not in dependency_cache:
            dependency_cache[source] = build_g1_source_dependencies(
                runtime,
                protocol,
                outer_target=outer_target,
                labeled_domain=source,
                encoder=encoder,
                progress=progress,
            )
        result = dependency_cache[source]
        roster = getattr(result, "roster", None)
        if (
            type(result) is not G1SourceDependencies
            or getattr(roster, "outer_target", None) != outer_target
            or getattr(roster, "labeled_domain", None) != source
            or tuple(getattr(roster, "fit_domains", ()))
            != tuple(domain for domain in source_domains if domain != source)
        ):
            raise G1DaggerSelectionExecutionError("DAgger dependency fold changed")
        return result

    dagger_build = build_g1_outer_dagger_banks(
        runtime,
        protocol,
        outer_target=outer_target,
        base_hyperparameters=base_hyperparameters,
        encoder=encoder,
        teacher_bank_root=teacher_bank_root,
        work_root=dagger_bank_root,
        device=device,
        dependency_factory=dependency_factory,
        progress=progress,
    )
    teacher_examples = tuple(row.example for row in dagger_build.records)
    bridge_records = []
    bridge_manifests = []
    for source in source_domains:
        bank, records = read_source_bridge_bank(
            Path(bridge_root) / outer_target / f"{source}.parquet"
        )
        if not records or any(
            row.outer_target != outer_target or row.source_domain != source
            for row in records
        ):
            raise G1DaggerSelectionExecutionError("DAgger source bridge changed")
        bridge_records.extend(records)
        bridge_manifests.append(bank.manifest_sha256)
    candidate_runs = [base_candidate]
    for iteration in (1, 2):
        hyperparameters = with_dagger_iterations(base_hyperparameters, iteration)
        if progress is not None:
            progress(f"G1 DAgger engineering candidate {outer_target}: {iteration}")
        candidate_runs.append(
            run_engineering_candidate(
                runtime,
                protocol,
                outer_target=outer_target,
                hyperparameters=hyperparameters,
                teacher_examples=teacher_examples,
                bridge_records=tuple(bridge_records),
                dependency_factory=dependency_factory,
                encoder=encoder,
                learned_root=learned_root,
                device=device,
                progress=progress,
            )
        )
    candidates = tuple(candidate_runs)
    selection = select_outer_policy(
        tuple(candidate.candidate for candidate in candidates)
    )
    selected = next(
        candidate
        for candidate in candidates
        if candidate.candidate.hyperparameters.state_sha256
        == selection.selected_hyperparameters_sha256
    )
    selected_hyperparameters = selected.candidate.hyperparameters
    authorization = authorize_conditional_aawr(
        _aawr_evidence(selected),
        soft_distillation_with_dagger=(
            selected_hyperparameters.route is TrainingRoute.SOFT_UTILITY_DISTILL
            and selected_hyperparameters.dagger_iterations > 0
        ),
    )
    destination = Path(work_root) / outer_target / "selection.json"
    _atomic_json(
        destination,
        {
            "schema_version": 1,
            "scope": "inspection_agent_g1_outer_dagger_selection",
            "outer_target": outer_target,
            "base_engineering_selection_sha256": base_run.selection.state_sha256,
            "base_selected_candidate_sha256": base_candidate.state_sha256,
            "base_teacher_manifest_sha256s": list(
                dagger_build.base_teacher_manifest_sha256s
            ),
            "dagger_bank_manifest_sha256s": [
                row.bank.manifest_sha256 for row in dagger_build.banks
            ],
            "source_bridge_manifest_sha256s": bridge_manifests,
            "candidates": [_candidate_payload(row) for row in candidates],
            "selection": outer_selection_payload(selection),
            "selected_dagger_iterations": (
                selected.candidate.hyperparameters.dagger_iterations
            ),
            "aawr_authorization": _aawr_payload(authorization),
            "target_outcomes_opened": False,
        },
    )
    return G1OuterDaggerSelectionRun(
        outer_target=outer_target,
        base_run=base_run,
        dagger_build=dagger_build,
        candidates=candidates,
        selection=selection,
        aawr_authorization=authorization,
        path=destination,
        target_outcomes_opened=False,
    )


__all__ = [
    "G1DaggerSelectionExecutionError",
    "G1OuterDaggerSelectionRun",
    "run_outer_dagger_selection",
]
