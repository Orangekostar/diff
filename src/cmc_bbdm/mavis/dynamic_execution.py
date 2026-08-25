"""Resumable outer-domain execution for MAVIS dynamic mechanical VoI."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .dynamic_data import (
    DynamicStateGroup,
    build_dynamic_training_groups,
    build_target_evaluation_groups,
)
from .dynamic_metrics import evaluate_dynamic_scores
from .dynamic_training import (
    FittedDynamicVoI,
    fit_final_dynamic_voi,
    fit_inner_dynamic_voi,
    save_fitted_dynamic_checkpoint,
)
from .mris_data import MRISFeatureBank
from .mris_training import load_fitted_mris_checkpoint


class MAVISDynamicExecutionError(RuntimeError):
    """Raised when a P3 outer worker is incomplete or inconsistent."""


_MODES = ("static", "positions_only", "real", "shuffled")


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


def _p2_checkpoint(
    root: str | Path,
    *,
    outer_domain: str,
    mode: str,
    validation_domain: str | None = None,
) -> Path:
    base = Path(root)
    if validation_domain is None:
        candidates = (base / f"{mode}.npz", base / f"{outer_domain}__{mode}.npz")
    else:
        candidates = (
            base / "inner" / f"{validation_domain}__{mode}.npz",
            base / "inner" / f"{outer_domain}__{validation_domain}__{mode}.npz",
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise MAVISDynamicExecutionError("P3 P2 checkpoint is unavailable")


def _audit_row(
    fitted: FittedDynamicVoI,
    *,
    mode: str,
    record_type: str,
    selected_final_epochs: int,
    p2_model_state_sha256: str,
) -> dict[str, object]:
    return {
        "record_type": record_type,
        "mode": mode,
        "outer_domain": fitted.outer_domain,
        "validation_domain": fitted.audit.validation_domain,
        "fit_domains": list(fitted.audit.fit_domains),
        "fit_specimen_ids": list(fitted.audit.fit_specimen_ids),
        "validation_specimen_ids": list(fitted.audit.validation_specimen_ids),
        "epochs_run": fitted.audit.epochs_run,
        "selected_epoch": fitted.audit.selected_epoch,
        "selected_final_epochs": selected_final_epochs,
        "best_validation_regret": fitted.audit.best_validation_regret,
        "p2_model_state_sha256": p2_model_state_sha256,
        "dynamic_model_state_sha256": fitted.model_state_sha256,
        "target_data_used_for_selection": False,
    }


def _score_groups(
    fitted: FittedDynamicVoI,
    groups: tuple[DynamicStateGroup, ...],
    embeddings: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, ...]:
    if embeddings.shape != (len(groups), fitted.mris_dimension):
        raise MAVISDynamicExecutionError("P3 target embedding roster is invalid")
    fitted.model.to(device).eval()
    parameter = next(fitted.model.parameters())
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for group, embedding in zip(groups, embeddings, strict=True):
            result = fitted.score_actions(
                torch.tensor(embedding, dtype=parameter.dtype, device=device),
                group.candidates,
            )
            scores = result.scores.detach().cpu().numpy().astype(np.float64)
            if scores.shape != group.teacher_values.shape or not np.all(
                np.isfinite(scores)
            ):
                raise MAVISDynamicExecutionError("P3 target scores are invalid")
            outputs.append(scores)
    fitted.model.cpu()
    return tuple(outputs)


def _action_score_rows(
    groups: tuple[DynamicStateGroup, ...],
    scores: tuple[np.ndarray, ...],
    *,
    mode: str,
    model_state_sha256: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, state_scores in zip(groups, scores, strict=True):
        for candidate_index, (candidate, score, teacher, prediction) in enumerate(
            zip(
                group.candidates,
                state_scores,
                group.teacher_values,
                group.candidate_predictions,
                strict=True,
            )
        ):
            rows.append(
                {
                    "outer_domain": group.outer_domain,
                    "domain_id": group.domain_id,
                    "specimen_id": group.specimen_id,
                    "state_id": group.state_id,
                    "mode": mode,
                    "candidate_index": candidate_index,
                    "cell_index": candidate.cell_index,
                    "from_level": candidate.from_level,
                    "to_level": candidate.to_level,
                    "exact_added_cost": candidate.exact_added_cost,
                    "predicted_score": float(score),
                    "teacher_value": float(teacher),
                    "current_prediction": group.current_prediction,
                    "candidate_prediction": float(prediction),
                    "evaluation_true_cai": group.true_cai,
                    "teacher_fold_count": group.teacher_fold_count,
                    "dynamic_model_state_sha256": model_state_sha256,
                }
            )
    return rows


def run_dynamic_outer_domain(
    bank: MRISFeatureBank,
    *,
    states: pl.DataFrame,
    actions: pl.DataFrame,
    outer_domain: str,
    p2_checkpoint_root: str | Path,
    output_root: str | Path,
    modes: tuple[str, ...] = _MODES,
    hidden_dimension: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
    loss_weights: dict[str, float],
    recall_k: int,
) -> Path:
    if (
        type(bank) is not MRISFeatureBank
        or outer_domain not in bank.domain_order
        or type(modes) is not tuple
        or not modes
        or len(set(modes)) != len(modes)
        or any(mode not in _MODES for mode in modes)
    ):
        raise MAVISDynamicExecutionError("P3 outer worker request is invalid")
    training_groups = build_dynamic_training_groups(
        states,
        actions,
        outer_domain=outer_domain,
    )
    target_groups = build_target_evaluation_groups(
        states,
        actions,
        target_domain=outer_domain,
    )
    source_domains = tuple(
        domain for domain in bank.domain_order if domain != outer_domain
    )
    if {group.domain_id for group in training_groups} != set(source_domains):
        raise MAVISDynamicExecutionError("P3 source group roster is incomplete")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / outer_domain
    if destination.exists():
        raise MAVISDynamicExecutionError("P3 outer worker output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{outer_domain}.", dir=root))
    try:
        checkpoint_root = temporary / "checkpoints"
        checkpoint_root.mkdir()
        inner_checkpoint_root = checkpoint_root / "inner"
        inner_checkpoint_root.mkdir()
        action_rows: list[dict[str, object]] = []
        metric_parts: list[pl.DataFrame] = []
        audit_rows: list[dict[str, object]] = []
        model_states: dict[str, str] = {}
        p2_states: dict[str, str] = {}
        inner_model_states: dict[str, str] = {}
        inner_p2_states: dict[str, str] = {}
        outer_index = bank.domain_order.index(outer_domain)
        for mode_index, mode in enumerate(modes):
            p2 = load_fitted_mris_checkpoint(
                _p2_checkpoint(
                    p2_checkpoint_root,
                    outer_domain=outer_domain,
                    mode=mode,
                )
            )
            if p2.outer_domain != outer_domain or p2.mode != mode:
                raise MAVISDynamicExecutionError("P3 P2 checkpoint fold changed")
            p2_states[mode] = p2.model_state_sha256
            inner_models: list[FittedDynamicVoI] = []
            for validation_index, validation_domain in enumerate(source_domains):
                inner_p2 = load_fitted_mris_checkpoint(
                    _p2_checkpoint(
                        p2_checkpoint_root,
                        outer_domain=outer_domain,
                        validation_domain=validation_domain,
                        mode=mode,
                    )
                )
                if (
                    inner_p2.outer_domain != outer_domain
                    or inner_p2.mode != mode
                    or inner_p2.audit.validation_domains != (validation_domain,)
                    or validation_domain in inner_p2.audit.fit_domains
                ):
                    raise MAVISDynamicExecutionError(
                        "P3 inner P2 checkpoint fold changed"
                    )
                inner_embeddings = inner_p2.encode(
                    bank,
                    state_ids=tuple(group.state_id for group in training_groups),
                    batch_size=batch_size,
                    device=device,
                )
                inner = fit_inner_dynamic_voi(
                    training_groups,
                    inner_embeddings,
                    validation_domain=validation_domain,
                    hidden_dimension=hidden_dimension,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=batch_size,
                    seed=(
                        seed
                        + 100_000 * outer_index
                        + 1_000 * mode_index
                        + validation_index
                    ),
                    device=device,
                    loss_weights=loss_weights,
                )
                inner_models.append(inner)
                save_fitted_dynamic_checkpoint(
                    inner,
                    inner_checkpoint_root / f"{validation_domain}__{mode}.npz",
                )
                key = f"{validation_domain}::{mode}"
                inner_model_states[key] = inner.model_state_sha256
                inner_p2_states[key] = inner_p2.model_state_sha256
            selected_epochs = max(
                1,
                math.floor(
                    float(
                        np.median(
                            [model.audit.selected_epoch for model in inner_models]
                        )
                    )
                    + 0.5
                ),
            )
            audit_rows.extend(
                _audit_row(
                    model,
                    mode=mode,
                    record_type="inner_fold",
                    selected_final_epochs=selected_epochs,
                    p2_model_state_sha256=inner_p2_states[
                        f"{model.audit.validation_domain}::{mode}"
                    ],
                )
                for model in inner_models
            )
            training_embeddings = p2.encode(
                bank,
                state_ids=tuple(group.state_id for group in training_groups),
                batch_size=batch_size,
                device=device,
            )
            final = fit_final_dynamic_voi(
                training_groups,
                training_embeddings,
                hidden_dimension=hidden_dimension,
                learning_rate=learning_rate,
                selected_epochs=selected_epochs,
                batch_size=batch_size,
                seed=seed + 900_000 + 10_000 * outer_index + mode_index,
                device=device,
                loss_weights=loss_weights,
            )
            audit_rows.append(
                _audit_row(
                    final,
                    mode=mode,
                    record_type="final_refit",
                    selected_final_epochs=selected_epochs,
                    p2_model_state_sha256=p2.model_state_sha256,
                )
            )
            save_fitted_dynamic_checkpoint(
                final,
                checkpoint_root / f"{mode}.npz",
            )
            model_states[mode] = final.model_state_sha256
            target_embeddings = p2.encode(
                bank,
                state_ids=tuple(group.state_id for group in target_groups),
                batch_size=batch_size,
                device=device,
            )
            scores = _score_groups(final, target_groups, target_embeddings, device=device)
            action_rows.extend(
                _action_score_rows(
                    target_groups,
                    scores,
                    mode=mode,
                    model_state_sha256=final.model_state_sha256,
                )
            )
            metric_parts.append(
                evaluate_dynamic_scores(
                    target_groups,
                    scores,
                    mode=mode,
                    recall_k=recall_k,
                )
            )
        pl.DataFrame(action_rows, infer_schema_length=None).sort(
            ["specimen_id", "state_id", "mode", "candidate_index"]
        ).write_parquet(
            temporary / "action_scores.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        pl.concat(metric_parts, how="vertical_relaxed").sort(
            ["specimen_id", "state_id", "mode"]
        ).write_parquet(
            temporary / "state_metrics.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        pl.DataFrame(audit_rows, infer_schema_length=None).sort(
            ["mode", "record_type", "validation_domain"],
            nulls_last=True,
        ).write_parquet(
            temporary / "model_selection_audit.parquet",
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
                "feature_bank_input_state_sha256": bank.input_state_sha256,
                "feature_bank_target_state_sha256": bank.target_state_sha256,
                "modes": list(modes),
                "p2_model_state_sha256": p2_states,
                "inner_p2_model_state_sha256": inner_p2_states,
                "dynamic_model_state_sha256": model_states,
                "inner_dynamic_model_state_sha256": inner_model_states,
                "source_group_count": len(training_groups),
                "target_group_count": len(target_groups),
                "inner_fold_count": len(source_domains) * len(modes),
                "target_data_used_for_selection": False,
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


__all__ = ["MAVISDynamicExecutionError", "run_dynamic_outer_domain"]
