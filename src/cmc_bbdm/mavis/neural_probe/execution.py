"""Resumable outer-domain P2 execution for the spatial neural probe."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

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


__all__ = ["SpatialMRISExecutionError", "run_spatial_mris_outer_domain"]
