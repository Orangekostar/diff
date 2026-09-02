"""Source-only engineering execution for observable-policy selection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .g1 import (
    G1Protocol,
    G1Runtime,
    G1SourceDependencies,
    build_g1_source_dependencies,
)
from .policy_selection import (
    OuterPolicySelection,
    PolicyCandidateEvaluation,
    outer_selection_payload,
    select_outer_policy,
)
from .policy_training import (
    G1PolicyTrainingExample,
    PolicyTrainingHyperparameters,
    fit_inner_observable_policy,
    rebind_training_example_modes,
)
from .selection_execution import run_outer_supervised_selection
from .source_bridge import read_source_bridge_bank
from .source_policy_evaluation import (
    build_g1_learned_source_bank,
    evaluate_inner_policy_bridge,
    learned_source_bank_path,
    read_learned_source_bank,
)
from .teacher_bank import read_teacher_bank


class G1EngineeringSelectionExecutionError(ValueError):
    """Raised when source engineering selection leaves its outer fold."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class G1EngineeringCandidateRun:
    candidate: PolicyCandidateEvaluation
    inner_evaluation_sha256s: tuple[str, ...]
    learned_bank_manifest_sha256s: tuple[str, ...]
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.candidate) is not PolicyCandidateEvaluation
            or type(self.inner_evaluation_sha256s) is not tuple
            or len(self.inner_evaluation_sha256s) != 5
            or not all(_valid_sha256(value) for value in self.inner_evaluation_sha256s)
            or type(self.learned_bank_manifest_sha256s) is not tuple
            or len(self.learned_bank_manifest_sha256s) != 5
            or not all(
                _valid_sha256(value)
                for value in self.learned_bank_manifest_sha256s
            )
        ):
            raise G1EngineeringSelectionExecutionError(
                "engineering candidate evidence is invalid"
            )
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-engineering-candidate-run",
                    "candidate": self.candidate.state_sha256,
                    "inner_evaluations": self.inner_evaluation_sha256s,
                    "learned_bank_manifests": self.learned_bank_manifest_sha256s,
                    "target_outcomes_opened": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1OuterEngineeringSelectionRun:
    outer_target: str
    example_count: int
    candidates: tuple[G1EngineeringCandidateRun, ...]
    selection: OuterPolicySelection
    path: Path

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.example_count) is not int
            or self.example_count <= 0
            or type(self.candidates) is not tuple
            or not self.candidates
            or any(
                type(value) is not G1EngineeringCandidateRun
                or value.candidate.outer_target != self.outer_target
                for value in self.candidates
            )
            or type(self.selection) is not OuterPolicySelection
            or self.selection.outer_target != self.outer_target
            or self.selection.target_outcomes_opened
            or not isinstance(self.path, Path)
        ):
            raise G1EngineeringSelectionExecutionError(
                "outer engineering selection run is invalid"
            )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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


def _candidate_run_payload(value: G1EngineeringCandidateRun) -> dict[str, object]:
    candidate = value.candidate
    return {
        "schema_version": 1,
        "scope": "inspection_agent_g1_engineering_candidate",
        "outer_target": candidate.outer_target,
        "hyperparameters_sha256": candidate.hyperparameters.state_sha256,
        "candidate_state_sha256": candidate.state_sha256,
        "equal_domain_mean_relative_auebc": (
            candidate.equal_domain_mean_relative_auebc
        ),
        "mean_oracle_gap_closure": candidate.mean_oracle_gap_closure,
        "improved_source_domains": candidate.improved_source_domains,
        "final_refit_epochs": candidate.final_refit_epochs,
        "inner_metric_sha256s": [
            row.state_sha256
            for row in sorted(
                candidate.inner_metrics,
                key=lambda row: (row.validation_domain, row.task.value),
            )
        ],
        "inner_evaluation_sha256s": list(value.inner_evaluation_sha256s),
        "learned_bank_manifest_sha256s": list(
            value.learned_bank_manifest_sha256s
        ),
        "target_outcomes_opened": False,
        "state_sha256": value.state_sha256,
    }


def run_engineering_candidate(
    runtime: G1Runtime,
    protocol: G1Protocol,
    *,
    outer_target: str,
    hyperparameters: PolicyTrainingHyperparameters,
    teacher_examples: tuple[G1PolicyTrainingExample, ...],
    bridge_records: tuple[object, ...],
    dependency_factory: Callable[[str], G1SourceDependencies],
    encoder: object,
    learned_root: str | Path,
    device: str,
    progress: Callable[[str], None] | None = None,
) -> G1EngineeringCandidateRun:
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or type(outer_target) is not str
        or not outer_target
        or type(hyperparameters) is not PolicyTrainingHyperparameters
        or type(teacher_examples) is not tuple
        or not teacher_examples
        or type(bridge_records) is not tuple
        or not bridge_records
        or not callable(dependency_factory)
        or not callable(getattr(encoder, "encode", None))
        or type(device) is not str
        or not device
        or (progress is not None and not callable(progress))
    ):
        raise G1EngineeringSelectionExecutionError(
            "engineering candidate request is invalid"
        )
    source_domains = tuple(
        sorted({str(row.source_domain) for row in teacher_examples})
    )
    if (
        len(source_domains) != 5
        or outer_target in source_domains
        or any(row.outer_target != outer_target for row in teacher_examples)
    ):
        raise G1EngineeringSelectionExecutionError(
            "engineering candidate source roster is invalid"
        )
    rebound = tuple(
        rebind_training_example_modes(
            row,
            cai_context_mode=hyperparameters.cai_context_mode,
            task_token_mode=hyperparameters.task_token_mode,
        )
        for row in teacher_examples
    )
    metrics = []
    evaluation_hashes = []
    bank_hashes = []
    for source in source_domains:
        path = learned_source_bank_path(
            learned_root,
            outer_target,
            hyperparameters.state_sha256,
            source,
        )
        manifest_path = path.with_suffix(f"{path.suffix}.manifest.json")
        if not path.exists() and not manifest_path.exists():
            dependencies = dependency_factory(source)
            if (
                type(dependencies) is not G1SourceDependencies
                or dependencies.roster.outer_target != outer_target
                or dependencies.roster.labeled_domain != source
                or tuple(dependencies.roster.fit_domains)
                != tuple(domain for domain in source_domains if domain != source)
            ):
                raise G1EngineeringSelectionExecutionError(
                    "engineering candidate dependency fold changed"
                )
            if progress is not None:
                progress(
                    f"G1 engineering actor fitting {outer_target}/{source}: "
                    f"{hyperparameters.state_sha256}"
                )
            actor = fit_inner_observable_policy(
                rebound,
                validation_domain=source,
                hyperparameters=hyperparameters,
                max_epochs=int(protocol.epochs),
                patience=int(protocol.patience),
                device=device,
            )
            audit = getattr(actor, "audit", None)
            if (
                getattr(actor, "hyperparameters", None) != hyperparameters
                or not _valid_sha256(getattr(actor, "model_state_sha256", None))
                or getattr(audit, "outer_target", None) != outer_target
                or getattr(audit, "validation_domain", None) != source
                or tuple(getattr(audit, "fit_domains", ()))
                != tuple(domain for domain in source_domains if domain != source)
            ):
                raise G1EngineeringSelectionExecutionError(
                    "engineering candidate actor fold changed"
                )
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
            raise G1EngineeringSelectionExecutionError(
                "engineering candidate evaluation fold changed"
            )
        metrics.extend(evaluation.metrics)
        evaluation_hashes.append(evaluation.state_sha256)
        bank_hashes.append(bank.manifest_sha256)
        if progress is not None:
            progress(
                f"G1 engineering source complete {outer_target}/{source}: "
                f"{hyperparameters.state_sha256}"
            )
    candidate = PolicyCandidateEvaluation(
        hyperparameters=hyperparameters,
        inner_metrics=tuple(metrics),
    )
    return G1EngineeringCandidateRun(
        candidate=candidate,
        inner_evaluation_sha256s=tuple(evaluation_hashes),
        learned_bank_manifest_sha256s=tuple(bank_hashes),
    )


def run_outer_engineering_selection(
    runtime: G1Runtime,
    protocol: G1Protocol,
    *,
    outer_target: str,
    encoder: object,
    teacher_bank_root: str | Path,
    bridge_root: str | Path,
    supervised_root: str | Path,
    learned_root: str | Path,
    work_root: str | Path,
    device: str,
    progress: Callable[[str], None] | None = None,
) -> G1OuterEngineeringSelectionRun:
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
        raise G1EngineeringSelectionExecutionError(
            "outer engineering selection request is invalid"
        )
    source_domains = tuple(domain for domain in domain_order if domain != outer_target)
    supervised = run_outer_supervised_selection(
        protocol,
        outer_target=outer_target,
        bank_root=teacher_bank_root,
        work_root=supervised_root,
        device=device,
        progress=progress,
    )
    candidate_hyperparameters = tuple(
        row.hyperparameters
        for row in (*supervised.core_results, *supervised.tuning_results)
    )
    if (
        not candidate_hyperparameters
        or len({row.state_sha256 for row in candidate_hyperparameters})
        != len(candidate_hyperparameters)
    ):
        raise G1EngineeringSelectionExecutionError(
            "supervised screen candidate roster changed"
        )
    examples = []
    teacher_manifests = []
    bridge_records = []
    bridge_manifests = []
    for source in source_domains:
        teacher_bank, teacher_rows = read_teacher_bank(
            Path(teacher_bank_root) / outer_target / f"{source}.parquet"
        )
        bridge_bank, source_bridges = read_source_bridge_bank(
            Path(bridge_root) / outer_target / f"{source}.parquet"
        )
        if (
            not teacher_rows
            or any(
                row.example.outer_target != outer_target
                or row.example.source_domain != source
                for row in teacher_rows
            )
            or not source_bridges
            or any(
                row.outer_target != outer_target or row.source_domain != source
                for row in source_bridges
            )
        ):
            raise G1EngineeringSelectionExecutionError(
                "outer engineering source evidence changed"
            )
        examples.extend(row.example for row in teacher_rows)
        bridge_records.extend(source_bridges)
        teacher_manifests.append(teacher_bank.manifest_sha256)
        bridge_manifests.append(bridge_bank.manifest_sha256)
    dependency_cache: dict[str, G1SourceDependencies] = {}

    def dependency_factory(source: str) -> G1SourceDependencies:
        if source not in source_domains:
            raise G1EngineeringSelectionExecutionError(
                "engineering dependency requested a non-source domain"
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
        return dependency_cache[source]

    destination = Path(work_root) / outer_target
    candidate_runs = []
    for hyperparameters in candidate_hyperparameters:
        if progress is not None:
            progress(
                f"G1 engineering candidate {outer_target}: "
                f"{hyperparameters.state_sha256}"
            )
        run = run_engineering_candidate(
            runtime,
            protocol,
            outer_target=outer_target,
            hyperparameters=hyperparameters,
            teacher_examples=tuple(examples),
            bridge_records=tuple(bridge_records),
            dependency_factory=dependency_factory,
            encoder=encoder,
            learned_root=learned_root,
            device=device,
            progress=progress,
        )
        _atomic_json(
            destination / f"{hyperparameters.state_sha256}.json",
            _candidate_run_payload(run),
        )
        candidate_runs.append(run)
    selection = select_outer_policy(
        tuple(run.candidate for run in candidate_runs)
    )
    selection_path = destination / "selection.json"
    _atomic_json(
        selection_path,
        {
            "schema_version": 1,
            "scope": "inspection_agent_g1_outer_engineering_selection",
            "outer_target": outer_target,
            "example_count": len(examples),
            "teacher_bank_manifest_sha256s": teacher_manifests,
            "source_bridge_manifest_sha256s": bridge_manifests,
            "supervised_selection_sha256": supervised.selection.state_sha256,
            "candidate_run_sha256s": [run.state_sha256 for run in candidate_runs],
            "selection": outer_selection_payload(selection),
        },
    )
    return G1OuterEngineeringSelectionRun(
        outer_target=outer_target,
        example_count=len(examples),
        candidates=tuple(candidate_runs),
        selection=selection,
        path=selection_path,
    )


__all__ = [
    "G1EngineeringCandidateRun",
    "G1EngineeringSelectionExecutionError",
    "G1OuterEngineeringSelectionRun",
    "run_engineering_candidate",
    "run_outer_engineering_selection",
]
