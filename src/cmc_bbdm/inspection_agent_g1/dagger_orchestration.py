"""Recoverable cross-fitted source-bank orchestration for G1 DAgger."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .dagger import REGISTERED_DAGGER_ITERATIONS, trajectory_quantile_indices
from .dagger_execution import (
    G1DaggerRelabelBatch,
    materialize_g1_dagger_relabels_for_world,
)
from .g1 import (
    G1Protocol,
    G1Runtime,
    G1SourceDependencies,
    build_g1_source_dependencies,
    build_g1_world,
)
from .policy_training import (
    PolicyTrainingHyperparameters,
    fit_inner_observable_policy,
    rebind_training_example_modes,
)
from .teacher_bank import (
    G1TeacherBankRecord,
    read_teacher_bank,
    write_teacher_bank,
)


class G1DaggerOrchestrationError(ValueError):
    """Raised when cross-fitted DAgger evidence is incomplete or stale."""


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


def with_dagger_iterations(
    hyperparameters: PolicyTrainingHyperparameters,
    iterations: int,
) -> PolicyTrainingHyperparameters:
    if (
        type(hyperparameters) is not PolicyTrainingHyperparameters
        or iterations not in REGISTERED_DAGGER_ITERATIONS
    ):
        raise G1DaggerOrchestrationError("DAgger hyperparameter request is invalid")
    return PolicyTrainingHyperparameters(
        model_name=hyperparameters.model_name,
        route=hyperparameters.route,
        cai_context_mode=hyperparameters.cai_context_mode,
        task_token_mode=hyperparameters.task_token_mode,
        tau=hyperparameters.tau,
        learning_rate=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
        dagger_iterations=iterations,
    )


def dagger_source_bank_path(
    work_root: str | Path,
    outer_target: str,
    iteration: int,
    source_domain: str,
) -> Path:
    if (
        type(outer_target) is not str
        or not outer_target
        or type(source_domain) is not str
        or not source_domain
        or outer_target == source_domain
        or iteration not in REGISTERED_DAGGER_ITERATIONS[1:]
        or any(
            "/" in value or "\\" in value or value in {".", ".."}
            for value in (outer_target, source_domain)
        )
    ):
        raise G1DaggerOrchestrationError("DAgger bank identity is invalid")
    return (
        Path(work_root)
        / outer_target
        / f"iteration_{iteration}"
        / f"{source_domain}.parquet"
    )


def dagger_manifest_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_suffix(f"{source.suffix}.dagger.json")


@dataclass(frozen=True, slots=True)
class G1DaggerSourceBankFile:
    outer_target: str
    source_domain: str
    fit_domains: tuple[str, ...]
    iteration: int
    specimen_count: int
    row_count: int
    dependency_sha256: str
    actor_model_sha256: str
    actor_fit_audit_sha256: str
    generation_hyperparameters_sha256: str
    teacher_bank_manifest_sha256: str
    batch_state_sha256s: tuple[str, ...]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class G1DaggerSourceBankBuild:
    path: Path
    outer_target: str
    source_domain: str
    iteration: int
    specimen_count: int
    dependency_sha256: str
    actor_model_sha256: str
    generation_hyperparameters_sha256: str
    bank: object
    records: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class G1OuterDaggerBuild:
    outer_target: str
    base_hyperparameters_sha256: str
    base_teacher_manifest_sha256s: tuple[str, ...]
    banks: tuple[G1DaggerSourceBankBuild, ...]
    records: tuple[object, ...]
    target_outcomes_opened: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or not _valid_sha256(self.base_hyperparameters_sha256)
            or len(self.base_teacher_manifest_sha256s) != 5
            or not all(
                _valid_sha256(value)
                for value in self.base_teacher_manifest_sha256s
            )
            or len(self.banks) != 10
            or not self.records
            or self.target_outcomes_opened
        ):
            raise G1DaggerOrchestrationError("outer DAgger build is invalid")


def _batch_payload(batch: G1DaggerRelabelBatch) -> dict[str, object]:
    return {
        "specimen_sha256": batch.specimen_sha256,
        "task": batch.task.value,
        "trajectory_sha256": batch.trajectory_sha256,
        "trajectory_length": batch.trajectory_length,
        "record_sha256s": [row.state_sha256 for row in batch.records],
        "visited_state_sha256s": [row.state_sha256 for row in batch.visited_states],
        "batch_state_sha256": batch.state_sha256,
    }


def _batch_state_sha(
    *,
    outer_target: str,
    source_domain: str,
    iteration: int,
    actor_model_sha256: str,
    payload: dict[str, object],
) -> str:
    return _json_sha(
        {
            "schema": 1,
            "kind": "g1-dagger-relabel-batch",
            "outer_target": outer_target,
            "source_domain": source_domain,
            "specimen_sha256": payload["specimen_sha256"],
            "task": payload["task"],
            "iteration": iteration,
            "actor_model_sha256": actor_model_sha256,
            "trajectory_sha256": payload["trajectory_sha256"],
            "trajectory_length": payload["trajectory_length"],
            "records": tuple(payload["record_sha256s"]),
            "visited": tuple(payload["visited_state_sha256s"]),
        }
    )


def _atomic_json(path: Path, payload: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return encoded


def write_g1_dagger_source_bank(
    path: str | Path,
    batches: tuple[G1DaggerRelabelBatch, ...],
    *,
    dependency_sha256: str,
    actor_fit_audit_sha256: str,
    generation_hyperparameters_sha256: str,
) -> G1DaggerSourceBankFile:
    if (
        type(batches) is not tuple
        or not batches
        or any(type(row) is not G1DaggerRelabelBatch for row in batches)
        or len({row.state_sha256 for row in batches}) != len(batches)
        or not all(
            _valid_sha256(value)
            for value in (
                dependency_sha256,
                actor_fit_audit_sha256,
                generation_hyperparameters_sha256,
            )
        )
    ):
        raise G1DaggerOrchestrationError("DAgger bank write request is invalid")
    first = batches[0]
    identity = (
        first.outer_target,
        first.source_domain,
        first.iteration,
        first.actor_model_sha256,
    )
    if any(
        (row.outer_target, row.source_domain, row.iteration, row.actor_model_sha256)
        != identity
        for row in batches
    ):
        raise G1DaggerOrchestrationError("DAgger bank batches mix identities")
    specimen_tasks: dict[str, set[InspectionTask]] = {}
    for batch in batches:
        specimen_tasks.setdefault(batch.specimen_sha256, set()).add(batch.task)
    if any(
        tasks != {InspectionTask.FIELD, InspectionTask.CAI}
        for tasks in specimen_tasks.values()
    ) or len(batches) != 2 * len(specimen_tasks):
        raise G1DaggerOrchestrationError("DAgger bank specimen/task roster is incomplete")
    records = tuple(record for batch in batches for record in batch.records)
    teacher_bank = write_teacher_bank(path, records)
    batch_payloads = [_batch_payload(batch) for batch in batches]
    payload = {
        "schema_version": 1,
        "scope": "inspection_agent_g1_dagger_source_bank",
        "outer_target": first.outer_target,
        "source_domain": first.source_domain,
        "fit_domains": list(first.records[0].fit_domains),
        "iteration": first.iteration,
        "specimen_count": len(specimen_tasks),
        "row_count": len(records),
        "dependency_sha256": dependency_sha256,
        "actor_model_sha256": first.actor_model_sha256,
        "actor_fit_audit_sha256": actor_fit_audit_sha256,
        "generation_hyperparameters_sha256": generation_hyperparameters_sha256,
        "teacher_bank_manifest_sha256": teacher_bank.manifest_sha256,
        "batches": batch_payloads,
    }
    manifest_payload = _atomic_json(dagger_manifest_path(path), payload)
    identity_file, replayed = read_g1_dagger_source_bank(path)
    if tuple(row.state_sha256 for row in replayed) != tuple(
        row.state_sha256 for row in read_teacher_bank(path)[1]
    ):
        raise G1DaggerOrchestrationError("written DAgger bank did not replay exactly")
    if identity_file.manifest_sha256 != hashlib.sha256(manifest_payload).hexdigest():
        raise G1DaggerOrchestrationError("written DAgger manifest changed")
    return identity_file


def read_g1_dagger_source_bank(
    path: str | Path,
) -> tuple[G1DaggerSourceBankFile, tuple[G1TeacherBankRecord, ...]]:
    source = Path(path)
    try:
        manifest_payload = dagger_manifest_path(source).read_bytes()
        manifest = json.loads(manifest_payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise G1DaggerOrchestrationError("DAgger bank manifest cannot be read") from error
    expected = {
        "schema_version",
        "scope",
        "outer_target",
        "source_domain",
        "fit_domains",
        "iteration",
        "specimen_count",
        "row_count",
        "dependency_sha256",
        "actor_model_sha256",
        "actor_fit_audit_sha256",
        "generation_hyperparameters_sha256",
        "teacher_bank_manifest_sha256",
        "batches",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected
        or manifest.get("schema_version") != 1
        or manifest.get("scope") != "inspection_agent_g1_dagger_source_bank"
        or manifest.get("iteration") not in REGISTERED_DAGGER_ITERATIONS[1:]
        or not all(
            _valid_sha256(manifest.get(key))
            for key in (
                "dependency_sha256",
                "actor_model_sha256",
                "actor_fit_audit_sha256",
                "generation_hyperparameters_sha256",
                "teacher_bank_manifest_sha256",
            )
        )
        or not isinstance(manifest.get("batches"), list)
    ):
        raise G1DaggerOrchestrationError("DAgger bank manifest is invalid")
    teacher_bank, records = read_teacher_bank(source)
    outer = str(manifest["outer_target"])
    source_domain = str(manifest["source_domain"])
    iteration = int(manifest["iteration"])
    actor_model = str(manifest["actor_model_sha256"])
    if (
        teacher_bank.manifest_sha256 != manifest["teacher_bank_manifest_sha256"]
        or manifest["row_count"] != len(records)
        or not records
        or any(
            row.example.outer_target != outer
            or row.example.source_domain != source_domain
            or row.example.dagger_iteration != iteration
            or row.state_source != "DAGGER_ACTOR_VISITED"
            for row in records
        )
        or list(records[0].fit_domains) != manifest["fit_domains"]
    ):
        raise G1DaggerOrchestrationError("DAgger bank teacher evidence changed")
    records_by_sha = {row.state_sha256: row for row in records}
    used: list[str] = []
    batch_hashes: list[str] = []
    specimen_tasks: dict[str, set[str]] = {}
    for value in manifest["batches"]:
        if not isinstance(value, dict) or set(value) != {
            "specimen_sha256",
            "task",
            "trajectory_sha256",
            "trajectory_length",
            "record_sha256s",
            "visited_state_sha256s",
            "batch_state_sha256",
        }:
            raise G1DaggerOrchestrationError("DAgger batch manifest is invalid")
        record_shas = value["record_sha256s"]
        visited_shas = value["visited_state_sha256s"]
        length = value["trajectory_length"]
        if (
            not _valid_sha256(value["specimen_sha256"])
            or value["task"] not in {InspectionTask.FIELD.value, InspectionTask.CAI.value}
            or not _valid_sha256(value["trajectory_sha256"])
            or type(length) is not int
            or length < 1
            or not isinstance(record_shas, list)
            or not isinstance(visited_shas, list)
            or len(record_shas) != len(trajectory_quantile_indices(length))
            or len(visited_shas) != len(record_shas)
            or not all(_valid_sha256(item) for item in (*record_shas, *visited_shas))
            or any(item not in records_by_sha for item in record_shas)
            or any(
                records_by_sha[item].source_state_sha256 != visited
                for item, visited in zip(record_shas, visited_shas, strict=True)
            )
            or any(
                records_by_sha[item].example.specimen_sha256
                != value["specimen_sha256"]
                or records_by_sha[item].example.task.value != value["task"]
                for item in record_shas
            )
        ):
            raise G1DaggerOrchestrationError("DAgger batch evidence changed")
        computed = _batch_state_sha(
            outer_target=outer,
            source_domain=source_domain,
            iteration=iteration,
            actor_model_sha256=actor_model,
            payload=value,
        )
        if value["batch_state_sha256"] != computed:
            raise G1DaggerOrchestrationError("DAgger batch hash changed")
        used.extend(record_shas)
        batch_hashes.append(computed)
        specimen_tasks.setdefault(str(value["specimen_sha256"]), set()).add(
            str(value["task"])
        )
    if (
        len(used) != len(set(used))
        or set(used) != set(records_by_sha)
        or any(
            tasks != {InspectionTask.FIELD.value, InspectionTask.CAI.value}
            for tasks in specimen_tasks.values()
        )
        or manifest["specimen_count"] != len(specimen_tasks)
    ):
        raise G1DaggerOrchestrationError("DAgger bank batch roster changed")
    identity = G1DaggerSourceBankFile(
        outer_target=outer,
        source_domain=source_domain,
        fit_domains=tuple(str(value) for value in manifest["fit_domains"]),
        iteration=iteration,
        specimen_count=len(specimen_tasks),
        row_count=len(records),
        dependency_sha256=str(manifest["dependency_sha256"]),
        actor_model_sha256=actor_model,
        actor_fit_audit_sha256=str(manifest["actor_fit_audit_sha256"]),
        generation_hyperparameters_sha256=str(
            manifest["generation_hyperparameters_sha256"]
        ),
        teacher_bank_manifest_sha256=teacher_bank.manifest_sha256,
        batch_state_sha256s=tuple(batch_hashes),
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
    )
    return identity, records


def materialize_g1_dagger_source_batches(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    actor: object,
    iteration: int,
    specimen_ids: tuple[str, ...],
    progress: Callable[[str], None] | None = None,
) -> tuple[G1DaggerRelabelBatch, ...]:
    audit = getattr(actor, "audit", None)
    hyperparameters = getattr(actor, "hyperparameters", None)
    roster = getattr(dependencies, "roster", None)
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or type(dependencies) is not G1SourceDependencies
        or not callable(getattr(encoder, "encode", None))
        or not callable(actor)
        or not _valid_sha256(getattr(actor, "model_state_sha256", None))
        or type(hyperparameters) is not PolicyTrainingHyperparameters
        or iteration not in REGISTERED_DAGGER_ITERATIONS[1:]
        or hyperparameters.dagger_iterations != iteration - 1
        or type(specimen_ids) is not tuple
        or not specimen_ids
        or len(set(specimen_ids)) != len(specimen_ids)
        or (progress is not None and not callable(progress))
        or getattr(audit, "outer_target", None) != roster.outer_target
        or getattr(audit, "validation_domain", None) != roster.labeled_domain
        or tuple(getattr(audit, "fit_domains", ())) != roster.fit_domains
    ):
        raise G1DaggerOrchestrationError("DAgger source materialization is invalid")
    available = {
        specimen
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == roster.labeled_domain
    }
    if not set(specimen_ids) <= available:
        raise G1DaggerOrchestrationError("DAgger specimen is outside its source fold")
    output = []
    for index, specimen in enumerate(specimen_ids, start=1):
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
            output.append(
                materialize_g1_dagger_relabels_for_world(
                    world,
                    grid,
                    surface.hypothesis,
                    dependencies.prior_fit.prior,
                    dependencies.authorization,
                    assessor=dependencies.assessor_fit.assessor,
                    encoder=encoder,
                    actor=actor,
                    outer_target=roster.outer_target,
                    source_domain=roster.labeled_domain,
                    specimen_sha256=specimen_sha,
                    iteration=iteration,
                    full_scan=teacher_view.full_scan,
                    true_cai=teacher_view.true_cai,
                )
            )
        if progress is not None and (index % 10 == 0 or index == len(specimen_ids)):
            progress(
                f"G1 DAgger relabel {roster.outer_target}/{roster.labeled_domain} "
                f"iteration {iteration}: {index}/{len(specimen_ids)} specimens"
            )
    batches = tuple(output)
    if len(batches) != 2 * len(specimen_ids):
        raise G1DaggerOrchestrationError("DAgger source batch roster changed")
    return batches


def build_g1_dagger_source_bank(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    actor: object,
    iteration: int,
    work_root: str | Path,
    progress: Callable[[str], None] | None = None,
) -> G1DaggerSourceBankBuild:
    roster = dependencies.roster
    audit = getattr(actor, "audit", None)
    hyperparameters = getattr(actor, "hyperparameters", None)
    path = dagger_source_bank_path(
        work_root, roster.outer_target, iteration, roster.labeled_domain
    )
    specimen_ids = tuple(
        specimen
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == roster.labeled_domain
    )
    if (
        len(specimen_ids) != int(protocol.domain_counts[roster.labeled_domain])
        or type(hyperparameters) is not PolicyTrainingHyperparameters
        or hyperparameters.dagger_iterations != iteration - 1
        or not _valid_sha256(getattr(audit, "state_sha256", None))
    ):
        raise G1DaggerOrchestrationError("formal DAgger source request changed")
    if path.exists() or dagger_manifest_path(path).exists():
        if not path.exists() or not dagger_manifest_path(path).exists():
            raise G1DaggerOrchestrationError("partial DAgger bank cache exists")
        bank, records = read_g1_dagger_source_bank(path)
        if (
            bank.dependency_sha256 != dependencies.state_sha256
            or bank.fit_domains != roster.fit_domains
            or bank.actor_model_sha256 != actor.model_state_sha256
            or bank.actor_fit_audit_sha256 != audit.state_sha256
            or bank.generation_hyperparameters_sha256
            != hyperparameters.state_sha256
            or bank.specimen_count != len(specimen_ids)
        ):
            raise G1DaggerOrchestrationError("cached DAgger source bank is stale")
    else:
        batches = materialize_g1_dagger_source_batches(
            runtime,
            protocol,
            dependencies,
            encoder=encoder,
            actor=actor,
            iteration=iteration,
            specimen_ids=specimen_ids,
            progress=progress,
        )
        bank = write_g1_dagger_source_bank(
            path,
            batches,
            dependency_sha256=dependencies.state_sha256,
            actor_fit_audit_sha256=audit.state_sha256,
            generation_hyperparameters_sha256=hyperparameters.state_sha256,
        )
        _identity, records = read_g1_dagger_source_bank(path)
    return G1DaggerSourceBankBuild(
        path=path,
        outer_target=roster.outer_target,
        source_domain=roster.labeled_domain,
        iteration=iteration,
        specimen_count=len(specimen_ids),
        dependency_sha256=dependencies.state_sha256,
        actor_model_sha256=actor.model_state_sha256,
        generation_hyperparameters_sha256=hyperparameters.state_sha256,
        bank=bank,
        records=records,
    )


def build_g1_outer_dagger_banks(
    runtime: G1Runtime,
    protocol: G1Protocol,
    *,
    outer_target: str,
    base_hyperparameters: PolicyTrainingHyperparameters,
    encoder: object,
    teacher_bank_root: str | Path,
    work_root: str | Path,
    device: str,
    progress: Callable[[str], None] | None = None,
) -> G1OuterDaggerBuild:
    domain_order = tuple(getattr(protocol, "domain_order", ()))
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or len(domain_order) != 6
        or outer_target not in domain_order
        or type(base_hyperparameters) is not PolicyTrainingHyperparameters
        or base_hyperparameters.dagger_iterations != 0
        or not callable(getattr(encoder, "encode", None))
        or type(device) is not str
        or not device
        or (progress is not None and not callable(progress))
    ):
        raise G1DaggerOrchestrationError("outer DAgger request is invalid")
    source_domains = tuple(domain for domain in domain_order if domain != outer_target)
    records: list[object] = []
    base_manifests = []
    for source in source_domains:
        bank, source_records = read_teacher_bank(
            Path(teacher_bank_root) / outer_target / f"{source}.parquet"
        )
        if not source_records or any(
            row.example.outer_target != outer_target
            or row.example.source_domain != source
            or row.example.dagger_iteration != 0
            for row in source_records
        ):
            raise G1DaggerOrchestrationError("base teacher-bank fold changed")
        records.extend(source_records)
        base_manifests.append(bank.manifest_sha256)
    builds: list[G1DaggerSourceBankBuild] = []
    dependency_cache: dict[str, G1SourceDependencies] = {}
    for iteration in REGISTERED_DAGGER_ITERATIONS[1:]:
        generation_hyperparameters = with_dagger_iterations(
            base_hyperparameters, iteration - 1
        )
        iteration_records: list[object] = []
        for source in source_domains:
            path = dagger_source_bank_path(work_root, outer_target, iteration, source)
            present = (path.exists(), dagger_manifest_path(path).exists())
            if any(present):
                if not all(present):
                    raise G1DaggerOrchestrationError("partial DAgger bank cache exists")
                bank, cached_records = read_g1_dagger_source_bank(path)
                if (
                    bank.outer_target != outer_target
                    or bank.source_domain != source
                    or bank.fit_domains
                    != tuple(
                        domain
                        for domain in source_domains
                        if domain != source
                    )
                    or bank.iteration != iteration
                    or bank.generation_hyperparameters_sha256
                    != generation_hyperparameters.state_sha256
                ):
                    raise G1DaggerOrchestrationError("cached DAgger fold changed")
                build = G1DaggerSourceBankBuild(
                    path=path,
                    outer_target=outer_target,
                    source_domain=source,
                    iteration=iteration,
                    specimen_count=bank.specimen_count,
                    dependency_sha256=bank.dependency_sha256,
                    actor_model_sha256=bank.actor_model_sha256,
                    generation_hyperparameters_sha256=(
                        bank.generation_hyperparameters_sha256
                    ),
                    bank=bank,
                    records=cached_records,
                )
                if progress is not None:
                    progress(f"G1 DAgger bank reused: {path}")
            else:
                rebound = tuple(
                    rebind_training_example_modes(
                        row.example,
                        cai_context_mode=generation_hyperparameters.cai_context_mode,
                        task_token_mode=generation_hyperparameters.task_token_mode,
                    )
                    for row in records
                )
                actor = fit_inner_observable_policy(
                    rebound,
                    validation_domain=source,
                    hyperparameters=generation_hyperparameters,
                    max_epochs=int(protocol.epochs),
                    patience=int(protocol.patience),
                    device=device,
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
                build = build_g1_dagger_source_bank(
                    runtime,
                    protocol,
                    dependency_cache[source],
                    encoder=encoder,
                    actor=actor,
                    iteration=iteration,
                    work_root=work_root,
                    progress=progress,
                )
            builds.append(build)
            iteration_records.extend(build.records)
        records.extend(iteration_records)
    return G1OuterDaggerBuild(
        outer_target=outer_target,
        base_hyperparameters_sha256=base_hyperparameters.state_sha256,
        base_teacher_manifest_sha256s=tuple(base_manifests),
        banks=tuple(builds),
        records=tuple(records),
        target_outcomes_opened=False,
    )


__all__ = [
    "G1DaggerOrchestrationError",
    "G1DaggerSourceBankBuild",
    "G1DaggerSourceBankFile",
    "G1OuterDaggerBuild",
    "build_g1_dagger_source_bank",
    "build_g1_outer_dagger_banks",
    "dagger_manifest_path",
    "dagger_source_bank_path",
    "materialize_g1_dagger_source_batches",
    "read_g1_dagger_source_bank",
    "with_dagger_iterations",
    "write_g1_dagger_source_bank",
]
