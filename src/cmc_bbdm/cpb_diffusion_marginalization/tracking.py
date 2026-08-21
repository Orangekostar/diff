"""Exact CSV tracking for D8 Optuna studies."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import optuna

from .config import DOMAIN_ORDER

TRIAL_INDEX_FIELDS = (
    "study_name",
    "trial_id",
    "outer_fold",
    "state",
    "failure_reason",
    *(f"inner_mae__{domain}" for domain in DOMAIN_ORDER),
    "mean_mae",
    "worst_mae",
    "domain_sd",
    "objective",
    "accepted_proposals",
    "proposed_variants",
    "acceptance_rate",
    "acceptance_by_domain",
    "control_id",
    "decomposition_family",
    "band",
    "decomposition_parameters",
    "alpha",
    "K_train",
    "K_test",
    "marginalization_stage",
    "feature_layer",
    "feature_aggregation",
    "prediction_aggregation",
    "morphology_beta",
    "consistency",
    "consistency_weight",
    "pca_dimension",
    "regressor",
    "regressor_parameters",
    "seed",
    "runtime_seconds",
    "config_sha256",
    "search_view_sha256",
    "candidate_sha256",
    "evidence_sha256",
)


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    return value


def _trial_row(
    study: optuna.Study, trial: optuna.trial.FrozenTrial
) -> dict[str, object]:
    attributes = trial.user_attrs
    candidate = attributes.get("candidate")
    if not isinstance(candidate, dict):
        candidate = {}
    regressor = candidate.get("regressor_spec")
    if not isinstance(regressor, dict):
        regressor = {}
    inner = attributes.get("inner_mae")
    if not isinstance(inner, dict):
        inner = {}
    decomposition = candidate.get("decomposition_parameters")
    row: dict[str, object] = {
        "study_name": study.study_name,
        "trial_id": trial.number,
        "outer_fold": attributes.get("outer_fold", ""),
        "state": trial.state.name,
        "failure_reason": attributes.get("failure_reason", ""),
        "mean_mae": attributes.get("mean_mae", ""),
        "worst_mae": attributes.get("worst_mae", ""),
        "domain_sd": attributes.get("domain_sd", ""),
        "objective": trial.value,
        "accepted_proposals": attributes.get("accepted_proposals", ""),
        "proposed_variants": attributes.get("proposed_variants", ""),
        "acceptance_rate": attributes.get("acceptance_rate", ""),
        "acceptance_by_domain": attributes.get("acceptance_by_domain", {}),
        "control_id": candidate.get("control_id", ""),
        "decomposition_family": candidate.get("decomposition_family", ""),
        "band": candidate.get("band", ""),
        "decomposition_parameters": decomposition,
        "alpha": candidate.get("alpha", ""),
        "K_train": candidate.get("K_train", ""),
        "K_test": candidate.get("K_test", ""),
        "marginalization_stage": candidate.get("marginalization_stage", ""),
        "feature_layer": candidate.get("feature_layer", ""),
        "feature_aggregation": candidate.get("feature_aggregation", ""),
        "prediction_aggregation": candidate.get("prediction_aggregation", ""),
        "morphology_beta": candidate.get("morphology_beta", ""),
        "consistency": candidate.get("consistency", ""),
        "consistency_weight": candidate.get("consistency_weight", ""),
        "pca_dimension": regressor.get("pca_dimension", ""),
        "regressor": regressor.get("regressor", ""),
        "regressor_parameters": regressor.get("parameters", {}),
        "seed": candidate.get("seed", ""),
        "runtime_seconds": attributes.get("runtime_seconds", ""),
        "config_sha256": attributes.get("config_sha256", ""),
        "search_view_sha256": attributes.get("search_view_sha256", ""),
        "candidate_sha256": candidate.get("state_sha256", ""),
        "evidence_sha256": attributes.get("evidence_sha256", ""),
    }
    for domain in DOMAIN_ORDER:
        row[f"inner_mae__{domain}"] = inner.get(domain, "")
    return {field: _cell(row.get(field, "")) for field in TRIAL_INDEX_FIELDS}


def write_trial_index(studies: tuple[optuna.Study, ...], path: Path) -> None:
    """Atomically write every visible trial from all D8 outer studies."""

    if not studies or any(type(study) is not optuna.Study for study in studies):
        raise ValueError("trial index requires exact Optuna studies")
    rows = [
        _trial_row(study, trial)
        for study in sorted(studies, key=lambda item: item.study_name)
        for trial in sorted(study.trials, key=lambda item: item.number)
        if trial.state
        in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.PRUNED,
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with staged.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TRIAL_INDEX_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        staged.replace(path)
    finally:
        if staged.exists():
            staged.unlink()


__all__ = ["TRIAL_INDEX_FIELDS", "write_trial_index"]
