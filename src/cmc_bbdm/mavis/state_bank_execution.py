"""Resumable execution workers for the MAVIS P1 state bank."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import polars as pl

from cmc_bbdm.mva.encoder_session import MVAEncoderSession
from cmc_bbdm.mva.pipeline import _encoder

from .authority import MAVISAuthority, load_mavis_authority
from .config import MAVISConfig, load_mavis_config
from .state_bank import materialize_action_plan
from .state_bank_artifacts import StateBankSample, state_bank_rows
from .state_candidates import build_source_candidate_batch_from_view
from .teacher import (
    TeacherRegistry,
    fit_teacher_registry,
    label_teacher_state,
    load_registered_initial_embeddings,
)
from .trajectory_sources import FrozenActionPlanSource


class MAVISStateBankExecutionError(RuntimeError):
    """Raised when a P1 worker cannot produce a complete deterministic shard."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(path: Path, rows: list[dict[str, object]], sort: list[str]) -> None:
    if not rows:
        raise MAVISStateBankExecutionError("state-bank shard is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pl.DataFrame(rows, infer_schema_length=None).sort(sort).write_parquet(
            temporary,
            compression="zstd",
            statistics=True,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fit_audit_rows(registry: TeacherRegistry) -> list[dict[str, object]]:
    return [
        {
            "held_out_target_domain": audit.held_out_target_domain,
            "query_source_domain": audit.query_source_domain,
            "query_domains": list(audit.query_domains),
            "fit_domains": list(audit.fit_domains),
            "query_specimen_ids": list(audit.query_specimen_ids),
            "fit_specimen_ids": list(audit.fit_specimen_ids),
            "selected_pca_dimension": audit.selected_pca_dimension,
            "predictor_state_sha256": audit.predictor_state_sha256,
            "teacher_registry_state_sha256": registry.state_sha256,
            "initial_embedding_state_sha256": (
                registry.initial_embedding_state_sha256
            ),
        }
        for audit in registry.fit_audits
    ]


def _validate_existing_shard(
    state_path: Path,
    action_path: Path,
    *,
    specimen_id: str,
    dataset_id: str,
    config: MAVISConfig,
    authority: MAVISAuthority,
) -> bool:
    if not state_path.is_file() or not action_path.is_file():
        return False
    try:
        states = pl.read_parquet(state_path)
        actions = pl.read_parquet(action_path)
    except (OSError, TypeError, ValueError, pl.exceptions.PolarsError):
        return False
    expected_states = 5 * len(config.checkpoints)
    return bool(
        states.height == expected_states
        and states.get_column("specimen_id").unique().to_list() == [specimen_id]
        and states.get_column("domain_id").unique().to_list() == [dataset_id]
        and states.get_column("authority_state_sha256").unique().to_list()
        == [authority.state_sha256]
        and states.get_column("state_id").n_unique() == expected_states
        and actions.height > 0
        and actions.get_column("specimen_id").unique().to_list() == [specimen_id]
        and actions.get_column("domain_id").unique().to_list() == [dataset_id]
        and actions.get_column("authority_state_sha256").unique().to_list()
        == [authority.state_sha256]
        and actions.get_column("outer_domain").n_unique() == 5
    )


def _specimen_rows(
    *,
    config: MAVISConfig,
    authority: MAVISAuthority,
    registry: TeacherRegistry,
    plan_source: FrozenActionPlanSource,
    encoder: MVAEncoderSession,
    specimen_id: str,
    dataset_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_view = authority.source_teacher_view(specimen_id)
    if source_view.dataset_id != dataset_id:
        raise MAVISStateBankExecutionError("source specimen domain changed")
    plans = plan_source.plans(specimen_id, dataset_id)
    state_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    for method, actions in plans.items():
        trajectory = materialize_action_plan(
            authority,
            specimen_id=specimen_id,
            method=method,
            seed=config.trajectory_random_seed if method == "random" else None,
            initial_budget=config.initial_budget_by_domain[dataset_id],
            checkpoints=config.checkpoints,
            actions=actions,
        )
        for snapshot in trajectory.snapshots:
            candidates = build_source_candidate_batch_from_view(
                source_view,
                snapshot.inspection_state,
                dataset_id=dataset_id,
                endpoint_budget=config.checkpoints[-1],
                encoder=encoder,
                interpolation=config.teacher_interpolation,
            )
            fold_labels = tuple(
                label_teacher_state(
                    registry.teacher(outer_domain, dataset_id),
                    specimen_id=specimen_id,
                    dataset_id=dataset_id,
                    true_cai=source_view.true_cai,
                    metadata=snapshot.inspection_state.context_features[:13],
                    current_embedding=candidates.current_embedding,
                    candidate_embeddings=candidates.candidate_embeddings,
                    actions=candidates.actions,
                    candidate_costs=candidates.candidate_costs,
                )
                for outer_domain in config.domain_order
                if outer_domain != dataset_id
            )
            sample = StateBankSample(
                dataset_id=dataset_id,
                trajectory=trajectory,
                snapshot=snapshot,
                candidates=candidates,
                fold_labels=fold_labels,
            )
            state_row, candidate_rows = state_bank_rows(
                sample,
                authority_state_sha256=authority.state_sha256,
                endpoint_budget=config.checkpoints[-1],
            )
            state_rows.append(state_row)
            action_rows.extend(candidate_rows)
    expected_states = 5 * len(config.checkpoints)
    if len(state_rows) != expected_states or not action_rows:
        raise MAVISStateBankExecutionError("specimen state bank is incomplete")
    return state_rows, action_rows


def _run_loaded_domain(
    *,
    root: Path,
    config: MAVISConfig,
    authority: MAVISAuthority,
    registry: TeacherRegistry,
    plan_source: FrozenActionPlanSource,
    encoder: MVAEncoderSession,
    domain_id: str,
    device: str,
    max_specimens: int | None,
) -> Path:
    work = root / config.output_root / ".work/p1_state_bank"
    state_directory = work / "states"
    action_directory = work / "state_action_pairs"
    specimens = tuple(
        specimen_id
        for specimen_id, dataset in zip(
            authority.specimen_ids,
            authority.dataset_ids,
            strict=True,
        )
        if dataset == domain_id
    )
    if max_specimens is not None:
        specimens = specimens[:max_specimens]
    completed: list[dict[str, object]] = []
    for specimen_id in specimens:
        token = specimen_id.replace("/", "_")
        state_path = state_directory / f"{domain_id}__{token}.parquet"
        action_path = action_directory / f"{domain_id}__{token}.parquet"
        if not _validate_existing_shard(
            state_path,
            action_path,
            specimen_id=specimen_id,
            dataset_id=domain_id,
            config=config,
            authority=authority,
        ):
            state_rows, action_rows = _specimen_rows(
                config=config,
                authority=authority,
                registry=registry,
                plan_source=plan_source,
                encoder=encoder,
                specimen_id=specimen_id,
                dataset_id=domain_id,
            )
            _atomic_parquet(
                state_path,
                state_rows,
                ["specimen_id", "method", "nominal_checkpoint"],
            )
            _atomic_parquet(
                action_path,
                action_rows,
                [
                    "specimen_id",
                    "method",
                    "nominal_checkpoint",
                    "outer_domain",
                    "candidate_index",
                ],
            )
        completed.append(
            {
                "specimen_id": specimen_id,
                "state_file": state_path.name,
                "state_sha256": _sha256(state_path),
                "action_file": action_path.name,
                "action_sha256": _sha256(action_path),
            }
        )
    audit_path = work / f"teacher_fit_audits__{domain_id}.parquet"
    _atomic_parquet(
        audit_path,
        _fit_audit_rows(registry),
        ["held_out_target_domain", "query_source_domain"],
    )
    summary_path = work / f"worker__{domain_id}.json"
    _atomic_json(
        summary_path,
        {
            "schema_version": 1,
            "domain_id": domain_id,
            "device": device,
            "config_sha256": config.config_sha256,
            "authority_state_sha256": authority.state_sha256,
            "teacher_registry_state_sha256": registry.state_sha256,
            "specimen_count": len(specimens),
            "specimens": completed,
        },
    )
    return summary_path


def run_state_bank_worker_group(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_project_root: str | Path,
    domain_ids: tuple[str, ...],
    device: str,
    max_specimens_per_domain: int | None = None,
) -> tuple[Path, ...]:
    root = Path(project_root).resolve(strict=True)
    config = load_mavis_config(config_path, project_root=root)
    if (
        type(domain_ids) is not tuple
        or not domain_ids
        or len(set(domain_ids)) != len(domain_ids)
        or any(domain_id not in config.domain_order for domain_id in domain_ids)
    ):
        raise MAVISStateBankExecutionError("worker domain group is invalid")
    if max_specimens_per_domain is not None and (
        type(max_specimens_per_domain) is not int
        or max_specimens_per_domain <= 0
    ):
        raise MAVISStateBankExecutionError("worker specimen limit is invalid")
    authority = load_mavis_authority(
        config,
        source_project_root=source_project_root,
    )
    try:
        source_root = Path(source_project_root).resolve(strict=True)
    except OSError as error:
        raise MAVISStateBankExecutionError(
            "source project root is unavailable"
        ) from error
    encoder = MVAEncoderSession(_encoder(source_root, device))
    initial_embeddings = load_registered_initial_embeddings(
        config,
        authority,
        project_root=root,
    )
    registry = fit_teacher_registry(config, authority, initial_embeddings)
    plan_source = FrozenActionPlanSource(config, project_root=root)
    outputs = tuple(
        _run_loaded_domain(
            root=root,
            config=config,
            authority=authority,
            registry=registry,
            plan_source=plan_source,
            encoder=encoder,
            domain_id=domain_id,
            device=device,
            max_specimens=max_specimens_per_domain,
        )
        for domain_id in domain_ids
    )
    encoder.validate()
    return outputs


def run_state_bank_domain_worker(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_project_root: str | Path,
    domain_id: str,
    device: str,
    max_specimens: int | None = None,
) -> Path:
    return run_state_bank_worker_group(
        config_path,
        project_root=project_root,
        source_project_root=source_project_root,
        domain_ids=(domain_id,),
        device=device,
        max_specimens_per_domain=max_specimens,
    )[0]


__all__ = [
    "MAVISStateBankExecutionError",
    "run_state_bank_domain_worker",
    "run_state_bank_worker_group",
]
