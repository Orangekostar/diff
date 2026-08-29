"""Resumable outer-domain P2 execution for the spatial neural probe."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from ..dynamic_data import (
    build_dynamic_training_groups,
    build_target_evaluation_groups,
)
from ..dynamic_execution import (
    _action_score_rows,
    _p2_checkpoint,
    _score_groups,
)
from ..dynamic_execution import _audit_row as _dynamic_audit_row
from ..dynamic_metrics import evaluate_dynamic_scores
from ..dynamic_training import (
    FittedDynamicVoI,
    fit_final_dynamic_voi,
    fit_inner_dynamic_voi,
    save_fitted_dynamic_checkpoint,
)
from ..mris_data import MRISFeatureBank
from ..mris_execution import (
    _control_hash,
    _donor_rows,
    _prediction_rows,
    _sha256,
    _write_json,
)
from .training import (
    FittedSpatialMRISModel,
    fit_final_spatial_mris_model,
    fit_inner_spatial_mris_fold,
    load_fitted_spatial_mris_checkpoint,
    save_fitted_spatial_mris_checkpoint,
)


class SpatialMRISExecutionError(RuntimeError):
    """Raised when a spatial P2 outer worker violates its contract."""


_TRAINABLE_MODES = ("static", "positions_only", "real", "shuffled")


def _audit_row(
    fitted: FittedSpatialMRISModel,
    *,
    record_type: str,
    validation_domain: str | None,
    selected_final_epochs: int,
) -> dict[str, object]:
    return {
        "architecture_name": fitted.architecture_name,
        "base_commit": fitted.base_commit,
        "best_validation_mae": fitted.audit.best_validation_mae,
        "config_sha256": fitted.config_sha256,
        "epochs_run": fitted.audit.epochs_run,
        "feature_bank_input_sha256": fitted.feature_bank_input_sha256,
        "feature_bank_target_sha256": fitted.feature_bank_target_sha256,
        "fit_domains": list(fitted.audit.fit_domains),
        "fit_specimen_ids": list(fitted.audit.fit_specimen_ids),
        "mode": fitted.mode,
        "model_state_sha256": fitted.model_state_sha256,
        "normalizer_state_sha256": fitted.normalizer.state_sha256,
        "outer_domain": fitted.outer_domain,
        "record_type": record_type,
        "selected_epoch": fitted.audit.selected_epoch,
        "selected_final_epochs": selected_final_epochs,
        "selection_candidate": "spatial_grid_cnn_v1_fixed",
        "selection_criterion": "source_validation_specimen_mae",
        "state_dict_sha256": fitted.state_dict_sha256,
        "target_data_used_for_selection": False,
        "validation_domain": validation_domain,
        "validation_domains": list(fitted.audit.validation_domains),
        "validation_specimen_ids": list(fitted.audit.validation_specimen_ids),
    }


def run_spatial_mris_outer_domain(
    bank: MRISFeatureBank,
    *,
    outer_domain: str,
    output_root: str | Path,
    trainable_modes: tuple[str, ...] = _TRAINABLE_MODES,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
    base_commit: str,
    config_sha256: str,
) -> Path:
    if (
        type(bank) is not MRISFeatureBank
        or outer_domain not in bank.domain_order
        or type(trainable_modes) is not tuple
        or not trainable_modes
        or len(set(trainable_modes)) != len(trainable_modes)
        or any(mode not in _TRAINABLE_MODES for mode in trainable_modes)
    ):
        raise SpatialMRISExecutionError("spatial P2 outer worker request is invalid")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / outer_domain
    if destination.exists():
        raise SpatialMRISExecutionError("spatial P2 outer output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{outer_domain}.", dir=root))
    try:
        checkpoint_root = temporary / "checkpoints"
        inner_checkpoint_root = checkpoint_root / "inner"
        inner_checkpoint_root.mkdir(parents=True)
        prediction_rows: list[dict[str, object]] = []
        audit_rows: list[dict[str, object]] = []
        model_states: dict[str, str] = {}
        inner_model_states: dict[str, str] = {}
        source_domains = tuple(
            domain for domain in bank.domain_order if domain != outer_domain
        )
        outer_index = bank.domain_order.index(outer_domain)
        for mode_index, mode in enumerate(trainable_modes):
            inner_models: list[FittedSpatialMRISModel] = []
            for validation_index, validation_domain in enumerate(source_domains):
                fitted = fit_inner_spatial_mris_fold(
                    bank,
                    mode=mode,
                    outer_domain=outer_domain,
                    validation_domain=validation_domain,
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
                    base_commit=base_commit,
                    config_sha256=config_sha256,
                )
                inner_models.append(fitted)
                checkpoint = inner_checkpoint_root / f"{validation_domain}__{mode}.npz"
                save_fitted_spatial_mris_checkpoint(fitted, checkpoint)
                inner_model_states[f"{validation_domain}::{mode}"] = (
                    fitted.model_state_sha256
                )
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
                    fitted,
                    record_type="inner_fold",
                    validation_domain=fitted.audit.validation_domains[0],
                    selected_final_epochs=selected_epochs,
                )
                for fitted in inner_models
            )
            final = fit_final_spatial_mris_model(
                bank,
                mode=mode,
                outer_domain=outer_domain,
                learning_rate=learning_rate,
                selected_epochs=selected_epochs,
                batch_size=batch_size,
                seed=seed + 900_000 + 10_000 * outer_index + mode_index,
                device=device,
                base_commit=base_commit,
                config_sha256=config_sha256,
            )
            audit_rows.append(
                _audit_row(
                    final,
                    record_type="final_refit",
                    validation_domain=None,
                    selected_final_epochs=selected_epochs,
                )
            )
            save_fitted_spatial_mris_checkpoint(
                final, checkpoint_root / f"{mode}.npz"
            )
            model_states[mode] = final.model_state_sha256
            prediction_rows.extend(
                _prediction_rows(
                    bank,
                    outer_domain=outer_domain,
                    mode=mode,
                    predictions=final.predict(
                        bank,
                        domain=outer_domain,
                        batch_size=batch_size,
                        device=device,
                    ),
                    model_state_sha256=final.model_state_sha256,
                )
            )

        target_indices = np.flatnonzero(
            np.asarray(bank.domain_ids, dtype=object) == outer_domain
        )
        reconstruction_state = _control_hash(bank, outer_domain)
        model_states["reconstruction"] = reconstruction_state
        prediction_rows.extend(
            _prediction_rows(
                bank,
                outer_domain=outer_domain,
                mode="reconstruction",
                predictions=bank.reconstruction_predictions[
                    outer_index, target_indices
                ],
                model_state_sha256=reconstruction_state,
            )
        )

        predictions_path = temporary / "predictions.parquet"
        audit_path = temporary / "model_selection_audit.parquet"
        donor_path = temporary / "donor_mapping.parquet"
        pl.DataFrame(prediction_rows, infer_schema_length=None).sort(
            ["specimen_id", "mode", "method", "nominal_checkpoint"]
        ).write_parquet(
            predictions_path,
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        pl.DataFrame(audit_rows, infer_schema_length=None).sort(
            ["mode", "record_type", "validation_domain"], nulls_last=True
        ).write_parquet(
            audit_path,
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        pl.DataFrame(_donor_rows(bank, outer_domain)).write_parquet(
            donor_path,
            compression="zstd",
            statistics=True,
        )
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        _write_json(
            temporary / "complete.json",
            {
                "architecture_name": "spatial_grid_cnn_v1",
                "base_commit": base_commit,
                "config_sha256": config_sha256,
                "feature_bank_input_state_sha256": bank.input_state_sha256,
                "feature_bank_target_state_sha256": bank.target_state_sha256,
                "files": {
                    path.relative_to(temporary).as_posix(): _sha256(path)
                    for path in files
                },
                "inner_fold_count": len(source_domains) * len(trainable_modes),
                "inner_model_state_sha256": inner_model_states,
                "model_state_sha256": model_states,
                "outer_domain": outer_domain,
                "prediction_count": len(prediction_rows),
                "schema_version": 1,
                "target_data_used_for_selection": False,
                "trainable_modes": list(trainable_modes),
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "complete.json"


def _spatial_dynamic_audit_row(
    fitted: FittedDynamicVoI,
    *,
    mode: str,
    record_type: str,
    selected_final_epochs: int,
    p2_model_state_sha256: str,
    base_commit: str,
    config_sha256: str,
) -> dict[str, object]:
    row = _dynamic_audit_row(
        fitted,
        mode=mode,
        record_type=record_type,
        selected_final_epochs=selected_final_epochs,
        p2_model_state_sha256=p2_model_state_sha256,
    )
    row.update(
        {
            "architecture_name": "spatial_grid_cnn_v1",
            "base_commit": base_commit,
            "config_sha256": config_sha256,
            "dynamic_scorer": "DynamicActionScorer",
        }
    )
    return row


def run_spatial_dynamic_outer_domain(
    bank: MRISFeatureBank,
    *,
    states: pl.DataFrame,
    actions: pl.DataFrame,
    outer_domain: str,
    p2_checkpoint_root: str | Path,
    output_root: str | Path,
    modes: tuple[str, ...] = _TRAINABLE_MODES,
    hidden_dimension: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
    loss_weights: dict[str, float],
    recall_k: int,
    base_commit: str,
    config_sha256: str,
) -> Path:
    if (
        type(bank) is not MRISFeatureBank
        or outer_domain not in bank.domain_order
        or type(modes) is not tuple
        or not modes
        or len(set(modes)) != len(modes)
        or any(mode not in _TRAINABLE_MODES for mode in modes)
    ):
        raise SpatialMRISExecutionError("spatial P3 outer worker request is invalid")
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
        raise SpatialMRISExecutionError("spatial P3 source group roster is incomplete")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / outer_domain
    if destination.exists():
        raise SpatialMRISExecutionError("spatial P3 outer output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{outer_domain}.", dir=root))
    try:
        checkpoint_root = temporary / "checkpoints"
        inner_checkpoint_root = checkpoint_root / "inner"
        inner_checkpoint_root.mkdir(parents=True)
        action_rows: list[dict[str, object]] = []
        metric_parts: list[pl.DataFrame] = []
        audit_rows: list[dict[str, object]] = []
        model_states: dict[str, str] = {}
        p2_states: dict[str, str] = {}
        inner_model_states: dict[str, str] = {}
        inner_p2_states: dict[str, str] = {}
        outer_index = bank.domain_order.index(outer_domain)
        for mode_index, mode in enumerate(modes):
            p2 = load_fitted_spatial_mris_checkpoint(
                _p2_checkpoint(
                    p2_checkpoint_root,
                    outer_domain=outer_domain,
                    mode=mode,
                ),
                expected_base_commit=base_commit,
                expected_feature_bank_input_sha256=bank.input_state_sha256,
                expected_feature_bank_target_sha256=bank.target_state_sha256,
                expected_config_sha256=config_sha256,
            )
            if p2.outer_domain != outer_domain or p2.mode != mode:
                raise SpatialMRISExecutionError("spatial P3 P2 fold changed")
            p2_states[mode] = p2.model_state_sha256
            inner_models: list[FittedDynamicVoI] = []
            for validation_index, validation_domain in enumerate(source_domains):
                inner_p2 = load_fitted_spatial_mris_checkpoint(
                    _p2_checkpoint(
                        p2_checkpoint_root,
                        outer_domain=outer_domain,
                        validation_domain=validation_domain,
                        mode=mode,
                    ),
                    expected_base_commit=base_commit,
                    expected_feature_bank_input_sha256=bank.input_state_sha256,
                    expected_feature_bank_target_sha256=bank.target_state_sha256,
                    expected_config_sha256=config_sha256,
                )
                if (
                    inner_p2.outer_domain != outer_domain
                    or inner_p2.mode != mode
                    or inner_p2.audit.validation_domains != (validation_domain,)
                    or validation_domain in inner_p2.audit.fit_domains
                ):
                    raise SpatialMRISExecutionError(
                        "spatial P3 inner P2 fold changed"
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
                _spatial_dynamic_audit_row(
                    fitted,
                    mode=mode,
                    record_type="inner_fold",
                    selected_final_epochs=selected_epochs,
                    p2_model_state_sha256=inner_p2_states[
                        f"{fitted.audit.validation_domain}::{mode}"
                    ],
                    base_commit=base_commit,
                    config_sha256=config_sha256,
                )
                for fitted in inner_models
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
                _spatial_dynamic_audit_row(
                    final,
                    mode=mode,
                    record_type="final_refit",
                    selected_final_epochs=selected_epochs,
                    p2_model_state_sha256=p2.model_state_sha256,
                    base_commit=base_commit,
                    config_sha256=config_sha256,
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
            ["mode", "record_type", "validation_domain"], nulls_last=True
        ).write_parquet(
            temporary / "model_selection_audit.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        _write_json(
            temporary / "complete.json",
            {
                "architecture_name": "spatial_grid_cnn_v1",
                "base_commit": base_commit,
                "config_sha256": config_sha256,
                "dynamic_model_state_sha256": model_states,
                "dynamic_scorer": "DynamicActionScorer",
                "feature_bank_input_state_sha256": bank.input_state_sha256,
                "feature_bank_target_state_sha256": bank.target_state_sha256,
                "files": {
                    path.relative_to(temporary).as_posix(): _sha256(path)
                    for path in files
                },
                "inner_dynamic_model_state_sha256": inner_model_states,
                "inner_fold_count": len(source_domains) * len(modes),
                "inner_p2_model_state_sha256": inner_p2_states,
                "modes": list(modes),
                "outer_domain": outer_domain,
                "p2_model_state_sha256": p2_states,
                "schema_version": 1,
                "source_group_count": len(training_groups),
                "target_data_used_for_selection": False,
                "target_group_count": len(target_groups),
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "complete.json"


__all__ = [
    "SpatialMRISExecutionError",
    "run_spatial_dynamic_outer_domain",
    "run_spatial_mris_outer_domain",
]
