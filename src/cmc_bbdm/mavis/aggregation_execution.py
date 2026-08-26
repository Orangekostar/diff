"""Resumable source-only on-policy aggregation for MAVIS P5 development."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.mva.encoder_session import MVAEncoderSession
from cmc_bbdm.mva.pipeline import _encoder

from .aggregation import (
    AggregationModel,
    SourceAggregationResult,
    build_on_policy_group,
    run_source_only_aggregation,
)
from .authority import MAVISAuthority
from .config import MAVISConfig
from .contracts import InspectionState
from .dynamic_data import DynamicStateGroup, build_dynamic_training_groups
from .dynamic_training import (
    FittedDynamicVoI,
    fit_final_dynamic_voi,
    load_fitted_dynamic_checkpoint,
    save_fitted_dynamic_checkpoint,
)
from .mris_data import MRISFeatureBank
from .mris_training import load_fitted_mris_checkpoint
from .policy import DeployedDynamicScorer
from .rollout import ScoutAndFocusCurve, rollout_scout_and_focus_curve
from .state_candidates import build_source_candidate_batch
from .teacher import (
    fit_teacher_registry,
    label_teacher_state,
    load_registered_initial_embeddings,
)


class MAVISAggregationExecutionError(RuntimeError):
    """Raised when a P5 aggregation worker violates its outer fold."""


def checkpoint_decision_states(curve: ScoutAndFocusCurve) -> tuple[InspectionState, ...]:
    if (
        type(curve) is not ScoutAndFocusCurve
        or not curve.feedback
        or len(curve.scoring_states) != len(curve.steps)
    ):
        raise MAVISAggregationExecutionError("aggregation rollout states are invalid")
    first_by_checkpoint = {}
    for state, step in zip(curve.scoring_states, curve.steps, strict=True):
        first_by_checkpoint.setdefault(step.nominal_checkpoint, state)
    selected_checkpoints = tuple(
        checkpoint
        for checkpoint in curve.checkpoints
        if checkpoint in first_by_checkpoint
    )
    if not selected_checkpoints or tuple(first_by_checkpoint) != selected_checkpoints or any(
        state.checkpoint != checkpoint
        for checkpoint, state in first_by_checkpoint.items()
    ):
        raise MAVISAggregationExecutionError(
            "aggregation checkpoint decision roster is incomplete"
        )
    return tuple(first_by_checkpoint.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _checkpoint(root: str | Path, outer_domain: str, mode: str) -> Path:
    base = Path(root)
    formal = base / f"{outer_domain}__{mode}.npz"
    local = base / f"{mode}.npz"
    return formal if formal.is_file() else local


def build_registered_encoder_session(
    encoder_project_root: str | Path,
    device: str,
) -> MVAEncoderSession:
    if type(device) is not str or not device:
        raise MAVISAggregationExecutionError("aggregation encoder device is invalid")
    try:
        root = Path(encoder_project_root).resolve(strict=True)
    except OSError as error:
        raise MAVISAggregationExecutionError(
            "aggregation encoder project root is unavailable"
        ) from error
    return MVAEncoderSession(_encoder(root, device))


def _embedding_matrix(
    groups: tuple[DynamicStateGroup, ...],
    lookup: dict[str, np.ndarray],
) -> np.ndarray:
    if any(group.state_id not in lookup for group in groups):
        raise MAVISAggregationExecutionError("aggregation embedding roster is incomplete")
    matrix = np.ascontiguousarray(
        np.stack([lookup[group.state_id] for group in groups]),
        dtype="<f8",
    )
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise MAVISAggregationExecutionError("aggregation embedding matrix is invalid")
    return matrix


def _group_rows(result: SourceAggregationResult) -> list[dict[str, object]]:
    return [
        {
            "outer_domain": group.outer_domain,
            "domain_id": group.domain_id,
            "specimen_id": group.specimen_id,
            "state_id": group.state_id,
            "state_sha256": group.state_sha256,
            "candidate_count": len(group.candidates),
            "teacher_fold_count": group.teacher_fold_count,
            "teacher_outer_domains": list(group.teacher_outer_domains),
            "is_on_policy": group.state_id.startswith("on_policy::"),
        }
        for group in result.groups
    ]


def run_aggregation_outer_domain(
    authority: MAVISAuthority,
    config: MAVISConfig,
    bank: MRISFeatureBank,
    *,
    states: pl.DataFrame,
    actions: pl.DataFrame,
    outer_domain: str,
    p2_checkpoint_root: str | Path,
    p3_checkpoint_root: str | Path,
    output_root: str | Path,
    project_root: str | Path,
    encoder_project_root: str | Path,
    device: str,
    batch_size: int,
) -> Path:
    if (
        type(authority) is not MAVISAuthority
        or type(config) is not MAVISConfig
        or type(bank) is not MRISFeatureBank
        or outer_domain not in config.domain_order
        or bank.domain_order != config.domain_order
        or not isinstance(states, pl.DataFrame)
        or not isinstance(actions, pl.DataFrame)
        or type(device) is not str
        or not device
        or type(batch_size) is not int
        or batch_size <= 0
    ):
        raise MAVISAggregationExecutionError("aggregation worker request is invalid")
    initial_groups = build_dynamic_training_groups(
        states,
        actions,
        outer_domain=outer_domain,
    )
    p2 = load_fitted_mris_checkpoint(
        _checkpoint(p2_checkpoint_root, outer_domain, "real")
    )
    initial_model = load_fitted_dynamic_checkpoint(
        _checkpoint(p3_checkpoint_root, outer_domain, "real")
    )
    if (
        p2.mode != "real"
        or p2.outer_domain != outer_domain
        or initial_model.outer_domain != outer_domain
        or p2.mris_dimension != initial_model.mris_dimension
        or set(initial_model.audit.fit_domains) != set(config.domain_order) - {outer_domain}
    ):
        raise MAVISAggregationExecutionError("aggregation initial model fold changed")
    initial_matrix = p2.encode(
        bank,
        state_ids=tuple(group.state_id for group in initial_groups),
        batch_size=batch_size,
        device=device,
    )
    embedding_lookup = {
        group.state_id: initial_matrix[index]
        for index, group in enumerate(initial_groups)
    }
    root = Path(project_root).resolve(strict=True)
    registered = load_registered_initial_embeddings(
        config,
        authority,
        project_root=root,
    )
    registry = fit_teacher_registry(config, authority, registered)
    encoder = build_registered_encoder_session(encoder_project_root, device)
    source_specimens = tuple(
        sorted(
            specimen_id
            for specimen_id, domain in zip(
                authority.specimen_ids,
                authority.dataset_ids,
                strict=True,
            )
            if domain != outer_domain
        )
    )
    selected_epochs = initial_model.audit.selected_epoch
    loss_weights = dict(initial_model.loss_weights)
    trajectory_rows: list[dict[str, object]] = []

    def train_model(
        groups: tuple[DynamicStateGroup, ...],
        round_index: int,
    ) -> AggregationModel:
        if round_index == 0:
            fitted = initial_model
        else:
            fitted = fit_final_dynamic_voi(
                groups,
                _embedding_matrix(groups, embedding_lookup),
                hidden_dimension=initial_model.hidden_dimension,
                learning_rate=config.learning_rate,
                selected_epochs=selected_epochs,
                batch_size=batch_size,
                seed=config.seed + 500_000 + 10_000 * config.domain_order.index(outer_domain) + round_index,
                device=device,
                loss_weights=loss_weights,
            )
        return AggregationModel(
            model=fitted,
            model_state_sha256=fitted.model_state_sha256,
        )

    def collect_source_groups(
        model: object,
        specimen_ids: tuple[str, ...],
        round_index: int,
    ) -> tuple[DynamicStateGroup, ...]:
        if type(model) is not FittedDynamicVoI or specimen_ids != source_specimens:
            raise MAVISAggregationExecutionError(
                "aggregation collector source roster changed"
            )
        scorer = DeployedDynamicScorer(
            mris_model=p2,
            dynamic_model=model,
            device=device,
        )
        visited_groups: list[DynamicStateGroup] = []
        visited_states = []
        for specimen_id in specimen_ids:
            domain_id = authority.source_teacher_view(specimen_id).dataset_id
            curve = rollout_scout_and_focus_curve(
                authority,
                specimen_id=specimen_id,
                initial_budget=config.initial_budget_by_domain[domain_id],
                checkpoints=config.checkpoints,
                scorer=scorer,
                objective="direct_cost_aware",
                feedback=True,
            )
            trajectory_rows.extend(
                {
                    "outer_domain": outer_domain,
                    "round_index": round_index,
                    "domain_id": domain_id,
                    "specimen_id": specimen_id,
                    "step": step.step,
                    "nominal_checkpoint": step.nominal_checkpoint,
                    "cell_index": step.action.cell_index,
                    "from_level": step.action.from_level,
                    "to_level": step.action.to_level,
                    "exact_cost_before": step.exact_cost_before,
                    "exact_cost_after": step.exact_cost_after,
                    "state_sha256_before": step.state_sha256_before,
                    "state_sha256_after": step.state_sha256_after,
                    "decision_confidence": step.decision_confidence,
                    "model_state_sha256": model.model_state_sha256,
                }
                for step in curve.steps
            )
            for state in checkpoint_decision_states(curve):
                candidate_batch = build_source_candidate_batch(
                    authority,
                    state,
                    dataset_id=domain_id,
                    endpoint_budget=config.checkpoints[-1],
                    action_budget=state.checkpoint,
                    encoder=encoder,
                    interpolation=config.teacher_interpolation,
                )
                labels = label_teacher_state(
                    registry.teacher(outer_domain, domain_id),
                    specimen_id=specimen_id,
                    dataset_id=domain_id,
                    true_cai=authority.source_teacher_view(specimen_id).true_cai,
                    metadata=state.context_features[:13],
                    current_embedding=candidate_batch.current_embedding,
                    candidate_embeddings=candidate_batch.candidate_embeddings,
                    actions=candidate_batch.actions,
                    candidate_costs=candidate_batch.candidate_costs,
                )
                visited_groups.append(
                    build_on_policy_group(
                        state,
                        candidate_batch,
                        labels,
                        outer_domain=outer_domain,
                    )
                )
                visited_states.append(state)
        encoded = p2.encode_inspection_states(
            tuple(visited_states),
            batch_size=batch_size,
            device=device,
        )
        for group, embedding in zip(visited_groups, encoded, strict=True):
            previous = embedding_lookup.get(group.state_id)
            if previous is not None and not np.allclose(
                previous,
                embedding,
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise MAVISAggregationExecutionError(
                    "aggregation duplicate MRIS embedding changed"
                )
            embedding_lookup[group.state_id] = embedding
        return tuple(visited_groups)

    result = run_source_only_aggregation(
        initial_groups,
        outer_domain=outer_domain,
        rounds=config.on_policy_rounds,
        train_model=train_model,
        collect_source_groups=collect_source_groups,
    )
    if (
        result.final_model.model.audit.outer_domain != outer_domain
        or set(result.final_model.model.audit.fit_domains)
        != set(config.domain_order) - {outer_domain}
    ):
        raise MAVISAggregationExecutionError("aggregation final model used target data")
    destination_root = Path(output_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / outer_domain
    if destination.exists():
        raise MAVISAggregationExecutionError("aggregation worker output already exists")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{outer_domain}.", dir=destination_root)
    )
    try:
        checkpoint = temporary / "checkpoint.npz"
        save_fitted_dynamic_checkpoint(result.final_model.model, checkpoint)
        pl.DataFrame(
            [asdict(audit) for audit in result.audits],
            infer_schema_length=None,
        ).sort("round_index").write_parquet(
            temporary / "round_audit.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        pl.DataFrame(_group_rows(result), infer_schema_length=None).sort(
            ["domain_id", "specimen_id", "state_id"]
        ).write_parquet(
            temporary / "aggregated_states.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        pl.DataFrame(trajectory_rows, infer_schema_length=None).sort(
            ["round_index", "domain_id", "specimen_id", "step"]
        ).write_parquet(
            temporary / "source_rollout_trajectories.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        complete = temporary / "complete.json"
        _write_json(
            complete,
            {
                "schema_version": 1,
                "outer_domain": outer_domain,
                "config_sha256": config.config_sha256,
                "feature_bank_input_state_sha256": bank.input_state_sha256,
                "feature_bank_target_state_sha256": bank.target_state_sha256,
                "round_count": config.on_policy_rounds,
                "source_domains": list(result.audits[0].source_domains),
                "source_specimen_count": len(source_specimens),
                "initial_state_count": result.audits[0].state_count_before,
                "final_state_count": len(result.groups),
                "appended_state_count": sum(
                    audit.appended_state_count for audit in result.audits
                ),
                "target_state_count": sum(
                    audit.target_state_count for audit in result.audits
                ),
                "selected_epochs": selected_epochs,
                "p2_model_state_sha256": p2.model_state_sha256,
                "initial_p3_model_state_sha256": initial_model.model_state_sha256,
                "final_model_state_sha256": result.final_model.model_state_sha256,
                "aggregation_state_sha256": result.state_sha256,
                "teacher_registry_state_sha256": registry.state_sha256,
                "target_data_used_for_training": False,
                "files": {
                    path.relative_to(temporary).as_posix(): _sha256(path)
                    for path in files
                },
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "complete.json"


__all__ = [
    "MAVISAggregationExecutionError",
    "build_registered_encoder_session",
    "checkpoint_decision_states",
    "run_aggregation_outer_domain",
]
