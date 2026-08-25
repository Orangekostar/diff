"""Resumable outer-domain execution for MAVIS P2 MRIS evaluation."""

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

from .mris_data import MRISFeatureBank
from .mris_training import (
    FittedMRISModel,
    fit_final_mris_model,
    fit_inner_mris_fold,
    save_fitted_mris_checkpoint,
)


class MAVISMRISExecutionError(RuntimeError):
    """Raised when a P2 outer-domain worker is incomplete or inconsistent."""


_TRAINABLE_MODES = ("static", "positions_only", "real", "shuffled")


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


def _audit_row(
    fitted: FittedMRISModel,
    *,
    record_type: str,
    validation_domain: str | None,
    selected_final_epochs: int,
) -> dict[str, object]:
    return {
        "record_type": record_type,
        "mode": fitted.mode,
        "outer_domain": fitted.outer_domain,
        "validation_domain": validation_domain,
        "fit_domains": list(fitted.audit.fit_domains),
        "validation_domains": list(fitted.audit.validation_domains),
        "fit_specimen_ids": list(fitted.audit.fit_specimen_ids),
        "validation_specimen_ids": list(fitted.audit.validation_specimen_ids),
        "epochs_run": fitted.audit.epochs_run,
        "selected_epoch": fitted.audit.selected_epoch,
        "selected_final_epochs": selected_final_epochs,
        "best_validation_mae": fitted.audit.best_validation_mae,
        "normalizer_state_sha256": fitted.normalizer.state_sha256,
        "model_state_sha256": fitted.model_state_sha256,
        "selection_candidate": "recommended_deepsets_64_or_registered_override",
        "selection_criterion": [
            "source_cai_auebc",
            "source_improved_domains",
            "worst_source_domain_auebc",
            "model_simplicity",
        ],
        "target_data_used_for_selection": False,
    }


def _prediction_rows(
    bank: MRISFeatureBank,
    *,
    outer_domain: str,
    mode: str,
    predictions: np.ndarray,
    model_state_sha256: str,
) -> list[dict[str, object]]:
    indices = np.flatnonzero(
        np.asarray(bank.domain_ids, dtype=object) == outer_domain
    )
    if predictions.shape != (indices.size,) or not np.all(np.isfinite(predictions)):
        raise MAVISMRISExecutionError("P2 target predictions are invalid")
    rows: list[dict[str, object]] = []
    for prediction, index in zip(predictions, indices, strict=True):
        target = float(bank.targets[index])
        error = abs(target - float(prediction))
        rows.append(
            {
                "outer_domain": outer_domain,
                "state_id": bank.state_ids[index],
                "specimen_id": bank.specimen_ids[index],
                "trajectory_id": bank.trajectory_ids[index],
                "method": bank.methods[index],
                "seed": bank.seeds[index],
                "nominal_checkpoint": float(bank.nominal_checkpoints[index]),
                "exact_acquired_cost": int(bank.exact_acquired_costs[index]),
                "native_count": int(bank.native_counts[index]),
                "effective_budget": float(bank.effective_budgets[index]),
                "mode": mode,
                "target": target,
                "prediction": float(prediction),
                "absolute_error": error,
                "model_state_sha256": model_state_sha256,
            }
        )
    return rows


def _control_hash(bank: MRISFeatureBank, outer_domain: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "schema": 1,
                "control": "strict_oof_reconstruction_prediction",
                "outer_domain": outer_domain,
                "feature_bank_input_state_sha256": bank.input_state_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _donor_rows(bank: MRISFeatureBank, outer_domain: str) -> list[dict[str, object]]:
    donors = bank.donor_specimen_ids[outer_domain]
    relaxations = bank.donor_relaxations[outer_domain]
    rows: dict[str, dict[str, object]] = {}
    for index, recipient_id in enumerate(bank.specimen_ids):
        row = {
            "outer_domain": outer_domain,
            "recipient_id": recipient_id,
            "recipient_domain": bank.domain_ids[index],
            "recipient_pool": (
                "target" if bank.domain_ids[index] == outer_domain else "source"
            ),
            "donor_id": donors[index],
            "relaxation": relaxations[index],
        }
        previous = rows.setdefault(recipient_id, row)
        if previous != row or recipient_id == donors[index]:
            raise MAVISMRISExecutionError("P2 donor mapping is inconsistent")
    return [rows[key] for key in sorted(rows)]


def run_mris_outer_domain(
    bank: MRISFeatureBank,
    *,
    outer_domain: str,
    output_root: str | Path,
    trainable_modes: tuple[str, ...] = _TRAINABLE_MODES,
    hidden_dimension: int,
    mris_dimension: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
) -> Path:
    if (
        type(bank) is not MRISFeatureBank
        or outer_domain not in bank.domain_order
        or type(trainable_modes) is not tuple
        or not trainable_modes
        or len(set(trainable_modes)) != len(trainable_modes)
        or any(mode not in _TRAINABLE_MODES for mode in trainable_modes)
    ):
        raise MAVISMRISExecutionError("P2 outer worker request is invalid")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / outer_domain
    if destination.exists():
        raise MAVISMRISExecutionError("P2 outer worker output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{outer_domain}.", dir=root))
    try:
        checkpoint_root = temporary / "checkpoints"
        checkpoint_root.mkdir()
        inner_checkpoint_root = checkpoint_root / "inner"
        inner_checkpoint_root.mkdir()
        prediction_rows: list[dict[str, object]] = []
        audit_rows: list[dict[str, object]] = []
        model_states: dict[str, str] = {}
        inner_model_states: dict[str, str] = {}
        source_domains = tuple(
            domain for domain in bank.domain_order if domain != outer_domain
        )
        outer_index = bank.domain_order.index(outer_domain)
        for mode_index, mode in enumerate(trainable_modes):
            inner_models: list[FittedMRISModel] = []
            for validation_index, validation_domain in enumerate(source_domains):
                inner = fit_inner_mris_fold(
                    bank,
                    mode=mode,
                    outer_domain=outer_domain,
                    validation_domain=validation_domain,
                    hidden_dimension=hidden_dimension,
                    mris_dimension=mris_dimension,
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
                )
                inner_models.append(inner)
                save_fitted_mris_checkpoint(
                    inner,
                    inner_checkpoint_root / f"{validation_domain}__{mode}.npz",
                )
                inner_model_states[f"{validation_domain}::{mode}"] = (
                    inner.model_state_sha256
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
            final = fit_final_mris_model(
                bank,
                mode=mode,
                outer_domain=outer_domain,
                hidden_dimension=hidden_dimension,
                mris_dimension=mris_dimension,
                learning_rate=learning_rate,
                selected_epochs=selected_epochs,
                batch_size=batch_size,
                seed=seed + 900_000 + 10_000 * outer_index + mode_index,
                device=device,
            )
            audit_rows.append(
                _audit_row(
                    final,
                    record_type="final_refit",
                    validation_domain=None,
                    selected_final_epochs=selected_epochs,
                )
            )
            checkpoint = checkpoint_root / f"{mode}.npz"
            save_fitted_mris_checkpoint(final, checkpoint)
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
        complete = temporary / "complete.json"
        _write_json(
            complete,
            {
                "schema_version": 1,
                "outer_domain": outer_domain,
                "feature_bank_input_state_sha256": bank.input_state_sha256,
                "feature_bank_target_state_sha256": bank.target_state_sha256,
                "trainable_modes": list(trainable_modes),
                "model_state_sha256": model_states,
                "inner_model_state_sha256": inner_model_states,
                "prediction_count": len(prediction_rows),
                "inner_fold_count": len(source_domains) * len(trainable_modes),
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
    "MAVISMRISExecutionError",
    "run_mris_outer_domain",
]
