"""Deterministic assembly of the complete formal G1 result package."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .artifacts import G1PackageValidation, publish_g1_manifest, validate_g1_package
from .decision_diagnostics import (
    G1SourceDecisionDiagnosticBankFile,
    G1SourceDecisionDiagnosticRecord,
    G1SourceDecisionDiagnosticSummary,
    summarize_g1_source_decision_diagnostics,
)
from .formal_selection import G1OuterFormalSelection
from .statistics import (
    FORMAL_BOOTSTRAP_SEED,
    G1PairedBootstrap,
    formal_synchronized_bootstrap,
)
from .surface_strata import (
    G1SurfaceStratumAuthority,
    G1SurfaceStratumRecord,
    surface_stratum_records_sha256,
)
from .target_analysis import G1TargetCurveAnalysis, G1TaskCurveAnalysis
from .target_evaluation import G1TargetCurveRecord
from .target_execution import G1TargetTrajectoryRecord, TargetPolicyVariant
from .target_reference import G1TargetReferenceCurveRecord
from .target_stopping import (
    G1TargetStopOutcome,
    G1TargetStoppingAnalysis,
    G1TaskStopAnalysis,
)


class G1FormalPackageError(ValueError):
    """Raised when formal G1 package evidence is incomplete or cannot publish."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


@dataclass(frozen=True, slots=True)
class G1TeacherBankManifestRow:
    outer_target: str
    source_domain: str
    row_count: int
    parquet_sha256: str
    records_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domain) is not str
            or not self.source_domain
            or self.source_domain == self.outer_target
            or type(self.row_count) is not int
            or self.row_count <= 0
            or not all(
                _valid_sha256(value)
                for value in (
                    self.parquet_sha256,
                    self.records_sha256,
                    self.manifest_sha256,
                )
            )
        ):
            raise G1FormalPackageError("teacher-bank manifest row is invalid")


def _json_text(payload: object) -> str:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def _cell(value: object) -> object:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return value


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(tuple(_cell(value) for value in row) for row in rows)
    path.write_text(buffer.getvalue(), encoding="ascii", newline="")


def _write_parquet(
    path: Path,
    rows: list[dict[str, object]],
    *,
    schema: dict[str, pl.DataType] | None = None,
) -> None:
    frame = (
        pl.DataFrame(rows, infer_schema_length=None)
        if rows
        else pl.DataFrame(schema=schema or {})
    )
    frame.write_parquet(
        path,
        compression="zstd",
        statistics=False,
        row_group_size=256,
    )


def _selection_payload(selection: G1OuterFormalSelection) -> dict[str, object]:
    hyperparameters = selection.action_hyperparameters
    return {
        "outer_target": selection.outer_target,
        "source_domains": list(selection.source_domains),
        "fixed_selections": [
            {
                "task": row.task.value,
                "method": row.method,
                "equal_domain_auebc": row.equal_domain_auebc,
                "domain_auebc": [list(value) for value in row.domain_auebc],
                "evidence_sha256": row.evidence_sha256,
                "state_sha256": row.state_sha256,
            }
            for row in selection.fixed_selections
        ],
        "stop_thresholds": [
            {
                "task": row.task.value,
                "status": row.status,
                "threshold": row.threshold,
                "source_evidence_sha256": row.source_evidence_sha256,
                "state_sha256": row.state_sha256,
            }
            for row in selection.stop_thresholds
        ],
        "action_hyperparameters": {
            "model_name": hyperparameters.model_name.value,
            "route": hyperparameters.route.value,
            "cai_context_mode": hyperparameters.cai_context_mode.value,
            "task_token_mode": hyperparameters.task_token_mode.value,
            "tau": hyperparameters.tau,
            "learning_rate": hyperparameters.learning_rate,
            "weight_decay": hyperparameters.weight_decay,
            "dagger_iterations": hyperparameters.dagger_iterations,
            "aawr_expectile": hyperparameters.aawr_expectile,
            "aawr_beta": hyperparameters.aawr_beta,
            "aawr_authorized_tasks": [
                task.value for task in hyperparameters.aawr_authorized_tasks
            ],
            "aawr_authorization_sha256": (
                hyperparameters.aawr_authorization_sha256
            ),
            "base_hyperparameters_sha256": (
                hyperparameters.base_hyperparameters_sha256
            ),
            "state_sha256": hyperparameters.state_sha256,
        },
        "action_selection_sha256": selection.action_selection_sha256,
        "action_model_sha256": selection.action_model_sha256,
        "stop_model_sha256": selection.stop_model_sha256,
        "decision_diagnostic_manifest_sha256": (
            selection.decision_diagnostic_manifest_sha256
        ),
        "target_outcomes_opened": False,
        "state_sha256": selection.state_sha256,
    }


def _curve_records(
    learned: tuple[G1TargetCurveRecord, ...],
    references: tuple[G1TargetReferenceCurveRecord, ...],
) -> list[tuple[str, str, str, str, object, str]]:
    output = [
        (
            "LEARNED",
            row.outer_target,
            row.specimen_id,
            row.variant.value,
            row.curve,
            row.state_sha256,
        )
        for row in learned
    ]
    output.extend(
        (
            "REFERENCE",
            row.outer_target,
            row.specimen_id,
            row.method,
            row.curve,
            row.state_sha256,
        )
        for row in references
    )
    return sorted(output, key=lambda row: (row[1], row[2], row[4].task.value, row[3]))


def _bootstrap_map(
    curves: G1TargetCurveAnalysis,
    stopping: G1TargetStoppingAnalysis,
    misleading: tuple[tuple[str, G1PairedBootstrap], ...],
) -> tuple[tuple[str, G1PairedBootstrap], ...]:
    return (
        ("field_policy", curves.field.baseline_minus_learned),
        ("cai_policy", curves.cai.baseline_minus_learned),
        (
            "field_wrong_task",
            curves.task_conditioning.field_wrong_minus_correct,
        ),
        (
            "field_no_task",
            curves.task_conditioning.field_no_task_minus_correct,
        ),
        ("cai_wrong_task", curves.task_conditioning.cai_wrong_minus_correct),
        ("cai_no_task", curves.task_conditioning.cai_no_task_minus_correct),
        (
            "field_no_surface",
            curves.surface_robustness.field_no_surface_minus_correct,
        ),
        (
            "field_shuffled_surface",
            curves.surface_robustness.field_shuffled_surface_minus_correct,
        ),
        (
            "cai_no_surface",
            curves.surface_robustness.cai_no_surface_minus_correct,
        ),
        (
            "cai_shuffled_surface",
            curves.surface_robustness.cai_shuffled_surface_minus_correct,
        ),
        ("field_stopping_saving", stopping.field.saving_bootstrap),
        ("cai_stopping_saving", stopping.cai.saving_bootstrap),
        *misleading,
    )


def _misleading_surface_bootstraps(
    learned: tuple[G1TargetCurveRecord, ...],
    strata: tuple[G1SurfaceStratumRecord, ...],
) -> tuple[tuple[str, G1PairedBootstrap], ...]:
    stratum_by_key = {
        (row.outer_target, row.specimen_id): row.stratum for row in strata
    }
    keys = tuple(
        sorted(
            key
            for key, stratum in stratum_by_key.items()
            if stratum == "SURFACE_INTERNAL_MISLEADING"
        )
    )
    if len({domain for domain, _specimen in keys}) != 6:
        raise G1FormalPackageError(
            "misleading surface stratum does not span six domains"
        )
    values = {
        (row.outer_target, row.specimen_id, row.task, row.variant): row.curve.auebc
        for row in learned
    }
    output = []
    for task in (InspectionTask.FIELD, InspectionTask.CAI):
        for variant in ("NO_SURFACE", "SHUFFLED_SURFACE"):
            variant_value = TargetPolicyVariant(variant)
            try:
                baseline = tuple(
                    values[(domain, specimen, task, variant_value)]
                    for domain, specimen in keys
                )
                correct = tuple(
                    values[
                        (
                            domain,
                            specimen,
                            task,
                            TargetPolicyVariant.PROPOSED,
                        )
                    ]
                    for domain, specimen in keys
                )
            except KeyError as error:
                raise G1FormalPackageError(
                    "misleading surface curve roster is incomplete"
                ) from error
            output.append(
                (
                    f"{task.value.lower()}_{variant.lower()}_misleading",
                    formal_synchronized_bootstrap(
                        dataset_ids=tuple(domain for domain, _specimen in keys),
                        specimen_ids=tuple(specimen for _domain, specimen in keys),
                        baseline_values=baseline,
                        learned_values=correct,
                        seed=FORMAL_BOOTSTRAP_SEED,
                    ),
                )
            )
    return tuple(output)


def _write_outer_selection(path: Path, selections: tuple[G1OuterFormalSelection, ...]) -> None:
    path.write_text(
        _json_text(
            {
                "schema_version": 1,
                "scope": "inspection_agent_g1_six_outer_selections",
                "target_outcomes_opened_during_selection": False,
                "outer_selections": [_selection_payload(row) for row in selections],
            }
        ),
        encoding="ascii",
    )


def _write_curve_tables(
    root: Path,
    learned: tuple[G1TargetCurveRecord, ...],
    references: tuple[G1TargetReferenceCurveRecord, ...],
    diagnostics: tuple[G1SourceDecisionDiagnosticRecord, ...],
    strata: tuple[G1SurfaceStratumRecord, ...],
) -> None:
    records = _curve_records(learned, references)
    stratum_by_key = {
        (row.outer_target, row.specimen_id): row.stratum for row in strata
    }
    state_rows = []
    specimen_rows = []
    by_domain: dict[tuple[str, str, str], list[float]] = {}
    for source, domain, specimen, method, curve, record_sha in records:
        specimen_rows.append(
            (
                domain,
                specimen,
                curve.specimen_sha256,
                curve.task.value,
                source,
                method,
                stratum_by_key[(domain, specimen)],
                curve.auebc,
                float(curve.task_losses[-1]),
                curve.state_sha256,
                record_sha,
            )
        )
        by_domain.setdefault((domain, curve.task.value, method), []).append(curve.auebc)
        for index, (nominal, exact, loss, state_sha) in enumerate(
            zip(
                curve.nominal_budgets,
                curve.exact_budgets,
                curve.task_losses,
                curve.projected_state_sha256,
                strict=True,
            )
        ):
            state_rows.append(
                (
                    "TARGET_ENGINEERING_CHECKPOINT",
                    domain,
                    "",
                    specimen,
                    curve.specimen_sha256,
                    curve.task.value,
                    source,
                    method,
                    index,
                    float(nominal),
                    float(exact),
                    float(loss),
                    state_sha,
                    curve.state_sha256,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    record_sha,
                )
            )
    state_rows.extend(
        (
            "SOURCE_DECISION_DIAGNOSTIC",
            row.outer_target,
            row.source_domain,
            "",
            row.specimen_sha256,
            row.task.value,
            "SOURCE",
            "PROPOSED",
            "",
            "",
            "",
            "",
            "",
            "",
            row.dagger_iteration,
            row.state_origin,
            row.policy_state_sha256,
            row.teacher_label_sha256,
            row.action_model_sha256,
            row.score_sha256,
            row.candidate_count,
            row.teacher_selected_slot,
            row.predicted_selected_slot,
            row.teacher_decision.value,
            row.predicted_decision.value,
            row.high_level_decision_accuracy,
            row.primitive_top1_match,
            row.top5_utility_recall,
            row.expected_teacher_regret,
            row.candidate_utility_ndcg,
            row.state_sha256,
        )
        for row in diagnostics
    )
    _write_csv(
        root / "state_level_metrics.csv",
        (
            "record_type",
            "outer_target",
            "source_domain",
            "specimen_id",
            "specimen_sha256",
            "task",
            "source",
            "method",
            "checkpoint_index",
            "nominal_budget",
            "exact_budget",
            "task_loss",
            "projected_state_sha256",
            "curve_sha256",
            "dagger_iteration",
            "state_origin",
            "policy_state_sha256",
            "teacher_label_sha256",
            "action_model_sha256",
            "score_sha256",
            "candidate_count",
            "teacher_selected_slot",
            "predicted_selected_slot",
            "teacher_decision",
            "predicted_decision",
            "high_level_decision_accuracy",
            "primitive_top1_match",
            "top5_utility_recall",
            "expected_teacher_regret",
            "candidate_utility_ndcg",
            "record_sha256",
        ),
        state_rows,
    )
    _write_csv(
        root / "per_specimen_metrics.csv",
        (
            "outer_target",
            "specimen_id",
            "specimen_sha256",
            "task",
            "source",
            "method",
            "surface_internal_stratum",
            "auebc",
            "endpoint_task_loss",
            "curve_sha256",
            "record_sha256",
        ),
        specimen_rows,
    )
    _write_csv(
        root / "domain_metrics.csv",
        ("outer_target", "task", "method", "specimen_count", "mean_auebc"),
        [
            (domain, task, method, len(values), float(np.mean(values)))
            for (domain, task, method), values in sorted(by_domain.items())
        ],
    )


def _write_trajectory_tables(
    root: Path,
    trajectories: tuple[G1TargetTrajectoryRecord, ...],
) -> None:
    trajectory_rows = []
    score_rows = []
    for row in sorted(
        trajectories,
        key=lambda value: (
            value.outer_target,
            value.specimen_id,
            value.task.value,
            value.variant.value,
        ),
    ):
        trajectory = row.trajectory
        actions = [
            [action.cell_index, action.from_level, action.to_level]
            for action in trajectory.action_history
        ]
        trajectory_rows.append(
            {
                "outer_target": row.outer_target,
                "specimen_id": row.specimen_id,
                "specimen_sha256": row.specimen_sha256,
                "task": row.task.value,
                "variant": row.variant.value,
                "stop_threshold": trajectory.stop_threshold,
                "stopped": trajectory.stopped,
                "termination_reason": trajectory.termination_reason,
                "effective_budget": trajectory.effective_budget,
                "native_count": trajectory.native_count,
                "action_history_json": json.dumps(actions, separators=(",", ":")),
                "final_observation_sha256": trajectory.final_observation_sha256,
                "acquired_positions_sha256": trajectory.acquired_positions_sha256,
                "acquired_values_sha256": trajectory.acquired_values_sha256,
                "trajectory_sha256": trajectory.state_sha256,
                "target_record_sha256": row.state_sha256,
            }
        )
        for step in trajectory.steps:
            score_rows.append(
                {
                    "outer_target": row.outer_target,
                    "specimen_id": row.specimen_id,
                    "task": row.task.value,
                    "variant": row.variant.value,
                    "step_index": step.step_index,
                    "observation_sha256": step.observation_sha256,
                    "policy_state_sha256": step.policy_state_sha256,
                    "model_sha256": step.scores.model_sha256,
                    "stop_probability": step.scores.stop_probability,
                    "selected_slot": step.selected_slot,
                    "action_logits_le_f64": np.ascontiguousarray(
                        step.scores.action_logits, dtype="<f8"
                    ).tobytes(order="C"),
                    "scores_sha256": step.scores.state_sha256,
                    "step_sha256": step.state_sha256,
                    "trajectory_sha256": trajectory.state_sha256,
                }
            )
    _write_parquet(root / "action_trajectories.parquet", trajectory_rows)
    _write_parquet(
        root / "action_score_audit.parquet",
        score_rows,
        schema={
            "outer_target": pl.String,
            "specimen_id": pl.String,
            "task": pl.String,
            "variant": pl.String,
            "step_index": pl.Int64,
            "observation_sha256": pl.String,
            "policy_state_sha256": pl.String,
            "model_sha256": pl.String,
            "stop_probability": pl.Float64,
            "selected_slot": pl.Int64,
            "action_logits_le_f64": pl.Binary,
            "scores_sha256": pl.String,
            "step_sha256": pl.String,
            "trajectory_sha256": pl.String,
        },
    )


def _effect_rows(values: tuple[tuple[str, G1PairedBootstrap], ...]) -> list[tuple[object, ...]]:
    return [
        (
            name,
            value.point_estimate,
            value.ci_lower,
            value.ci_upper,
            value.improved_domains,
            value.replicates,
            value.seed,
            value.distribution_sha256,
            json.dumps(value.domain_effects, separators=(",", ":")),
        )
        for name, value in values
    ]


def _write_inference_tables(
    root: Path,
    curves: G1TargetCurveAnalysis,
    stopping: G1TargetStoppingAnalysis,
    misleading: tuple[tuple[str, G1PairedBootstrap], ...],
) -> None:
    task_values = (
        ("field_wrong_task", curves.task_conditioning.field_wrong_minus_correct),
        ("field_no_task", curves.task_conditioning.field_no_task_minus_correct),
        ("cai_wrong_task", curves.task_conditioning.cai_wrong_minus_correct),
        ("cai_no_task", curves.task_conditioning.cai_no_task_minus_correct),
    )
    surface_values = (
        ("field_no_surface", curves.surface_robustness.field_no_surface_minus_correct),
        (
            "field_shuffled_surface",
            curves.surface_robustness.field_shuffled_surface_minus_correct,
        ),
        ("cai_no_surface", curves.surface_robustness.cai_no_surface_minus_correct),
        (
            "cai_shuffled_surface",
            curves.surface_robustness.cai_shuffled_surface_minus_correct,
        ),
    )
    task_header = (
        "comparison",
        "point_estimate",
        "ci_lower",
        "ci_upper",
        "improved_domains",
        "replicates",
        "seed",
        "distribution_sha256",
        "domain_effects_json",
    )
    _write_csv(
        root / "task_conditioning.csv",
        task_header,
        _effect_rows(task_values),
    )
    surface_rows = [
        (row[0], "ALL", *row[1:]) for row in _effect_rows(surface_values)
    ]
    surface_rows.extend(
        (row[0], "SURFACE_INTERNAL_MISLEADING", *row[1:])
        for row in _effect_rows(misleading)
    )
    _write_csv(
        root / "surface_robustness.csv",
        ("comparison", "stratum", *task_header[1:]),
        surface_rows,
    )
    bootstraps = _bootstrap_map(curves, stopping, misleading)
    replicate_count = bootstraps[0][1].replicates
    if any(value.replicates != replicate_count for _name, value in bootstraps):
        raise G1FormalPackageError("formal bootstrap replicate roster changed")
    _write_csv(
        root / "bootstrap.csv",
        ("replicate", *(name for name, _value in bootstraps)),
        [
            (
                index,
                *(float(value.distribution[index]) for _name, value in bootstraps),
            )
            for index in range(replicate_count)
        ],
    )


def _write_stopping(
    path: Path,
    outcomes: tuple[G1TargetStopOutcome, ...],
) -> None:
    _write_csv(
        path,
        (
            "outer_target",
            "specimen_id",
            "specimen_sha256",
            "task",
            "stop_authorized",
            "threshold",
            "stopped",
            "budget",
            "normalized_measurement_saving",
            "task_loss",
            "reference_task_loss",
            "task_loss_ratio",
            "premature_stop",
            "false_continue",
            "state_sha256",
        ),
        [
            (
                row.outer_target,
                row.specimen_id,
                row.specimen_sha256,
                row.task.value,
                row.stop_authorized,
                row.threshold,
                row.stopped,
                row.budget,
                row.normalized_measurement_saving,
                row.task_loss,
                row.reference_task_loss,
                row.task_loss_ratio,
                row.premature_stop,
                row.false_continue,
                row.state_sha256,
            )
            for row in sorted(
                outcomes,
                key=lambda value: (
                    value.outer_target,
                    value.specimen_id,
                    value.task.value,
                ),
            )
        ],
    )


def _diagnostic_payload(
    value: G1SourceDecisionDiagnosticSummary,
) -> dict[str, object]:
    return {
        "scope": "SOURCE_ONLY_DIAGNOSTIC_NOT_A_GATE",
        "record_count": value.record_count,
        "high_level_decision_accuracy": value.high_level_decision_accuracy,
        "primitive_top1_match": value.primitive_top1_match,
        "top5_utility_recall": value.top5_utility_recall,
        "expected_teacher_regret": value.expected_teacher_regret,
        "candidate_utility_ndcg": value.candidate_utility_ndcg,
        "predicted_decision_proportions": [
            {"decision": decision, "proportion": proportion}
            for decision, proportion in value.predicted_decision_proportions
        ],
        "teacher_to_predicted_transition_proportions": [
            {
                "teacher_decision": teacher,
                "predicted_decision": predicted,
                "proportion": proportion,
            }
            for teacher, predicted, proportion in value.transition_proportions
        ],
        "state_sha256": value.state_sha256,
    }


def _bootstrap_payload(value: G1PairedBootstrap) -> dict[str, object]:
    return {
        "point_estimate": value.point_estimate,
        "ci_lower": value.ci_lower,
        "ci_upper": value.ci_upper,
        "improved_domains": value.improved_domains,
        "replicates": value.replicates,
        "seed": value.seed,
        "distribution_sha256": value.distribution_sha256,
        "domain_effects": [list(row) for row in value.domain_effects],
    }


def _policy_payload(value: G1TaskCurveAnalysis) -> dict[str, object]:
    return {
        "status": value.policy_gate.status,
        "fixed_methods": [list(row) for row in value.fixed_methods],
        "fixed_auebc": value.fixed_auebc,
        "learned_auebc": value.learned_auebc,
        "oracle_auebc": value.oracle_auebc,
        "oracle_gap_closure": value.oracle_gap_closure,
        "baseline_minus_learned": _bootstrap_payload(
            value.baseline_minus_learned
        ),
        "gate_sha256": value.policy_gate.state_sha256,
        "state_sha256": value.state_sha256,
    }


def _stop_analysis_payload(value: G1TaskStopAnalysis) -> dict[str, object]:
    return {
        "status": value.status,
        "authorized_domains": value.authorized_domains,
        "normalized_measurement_saving": value.normalized_measurement_saving,
        "task_loss_ratio": value.task_loss_ratio,
        "premature_stop_rate": value.premature_stop_rate,
        "false_continue_rate": value.false_continue_rate,
        "fraction_stopped": value.fraction_stopped,
        "fraction_never_stopped": value.fraction_never_stopped,
        "saving_bootstrap": _bootstrap_payload(value.saving_bootstrap),
        "gate_sha256": None if value.gate is None else value.gate.state_sha256,
        "state_sha256": value.state_sha256,
    }


def _write_decision(
    path: Path,
    curves: G1TargetCurveAnalysis,
    stopping: G1TargetStoppingAnalysis,
    diagnostics: G1SourceDecisionDiagnosticSummary,
    surface_authority: G1SurfaceStratumAuthority,
    misleading: tuple[tuple[str, G1PairedBootstrap], ...],
) -> None:
    path.write_text(
        _json_text(
            {
                "schema_version": 2,
                "scope": "inspection_agent_g1_final_decision",
                "status": curves.final_decision.status,
                "g2_authorized": curves.final_decision.g2_authorized,
                "field_policy": _policy_payload(curves.field),
                "cai_policy": _policy_payload(curves.cai),
                "task_conditioning": {
                    "status": curves.task_conditioning.gate.status,
                    "field_wrong_task": _bootstrap_payload(
                        curves.task_conditioning.field_wrong_minus_correct
                    ),
                    "field_no_task": _bootstrap_payload(
                        curves.task_conditioning.field_no_task_minus_correct
                    ),
                    "cai_wrong_task": _bootstrap_payload(
                        curves.task_conditioning.cai_wrong_minus_correct
                    ),
                    "cai_no_task": _bootstrap_payload(
                        curves.task_conditioning.cai_no_task_minus_correct
                    ),
                    "gate_sha256": curves.task_conditioning.gate.state_sha256,
                    "state_sha256": curves.task_conditioning.state_sha256,
                },
                "surface_robustness": {
                    "field_no_surface": _bootstrap_payload(
                        curves.surface_robustness.field_no_surface_minus_correct
                    ),
                    "field_shuffled_surface": _bootstrap_payload(
                        curves.surface_robustness.field_shuffled_surface_minus_correct
                    ),
                    "cai_no_surface": _bootstrap_payload(
                        curves.surface_robustness.cai_no_surface_minus_correct
                    ),
                    "cai_shuffled_surface": _bootstrap_payload(
                        curves.surface_robustness.cai_shuffled_surface_minus_correct
                    ),
                    "state_sha256": curves.surface_robustness.state_sha256,
                },
                "stopping": {
                    "field": _stop_analysis_payload(stopping.field),
                    "cai": _stop_analysis_payload(stopping.cai),
                },
                "source_decision_diagnostics": _diagnostic_payload(diagnostics),
                "surface_strata_authority": {
                    "scope": "FROZEN_G0_DIAGNOSTIC_NOT_A_GATE",
                    "source": surface_authority.source,
                    "file_sha256": surface_authority.file_sha256,
                    "record_count": surface_authority.record_count,
                    "stratum_counts": [
                        list(row) for row in surface_authority.stratum_counts
                    ],
                    "records_sha256": surface_authority.records_sha256,
                    "state_sha256": surface_authority.state_sha256,
                },
                "misleading_surface_diagnostics": {
                    name: _bootstrap_payload(value)
                    for name, value in misleading
                },
                "no_target_leakage": curves.no_target_leakage,
                "deterministic_replay": curves.deterministic_replay,
                "deployment_bridge_valid": curves.deployment_bridge_valid,
                "curve_analysis_sha256": curves.state_sha256,
                "stopping_analysis_sha256": stopping.state_sha256,
                "decision_sha256": curves.final_decision.state_sha256,
            }
        ),
        encoding="ascii",
    )


def _write_report(
    path: Path,
    curves: G1TargetCurveAnalysis,
    stopping: G1TargetStoppingAnalysis,
    diagnostics: G1SourceDecisionDiagnosticSummary,
    surface_authority: G1SurfaceStratumAuthority,
    misleading: tuple[tuple[str, G1PairedBootstrap], ...],
) -> None:
    path.write_text(
        "\n".join(
            (
                "# G1 Observable Task-Conditioned Inspection Policy",
                "",
                f"Final status: `{curves.final_decision.status}`",
                "",
                "## Engineering policy",
                "",
                (
                    f"- FIELD: `{curves.field.policy_gate.status}`; "
                    f"fixed/learned/oracle AUEBC "
                    f"{curves.field.fixed_auebc!r}/"
                    f"{curves.field.learned_auebc!r}/"
                    f"{curves.field.oracle_auebc!r}; effect "
                    f"{curves.field.baseline_minus_learned.point_estimate!r}, "
                    f"95% CI [{curves.field.baseline_minus_learned.ci_lower!r}, "
                    f"{curves.field.baseline_minus_learned.ci_upper!r}], "
                    f"domains {curves.field.baseline_minus_learned.improved_domains}/6; "
                    f"oracle-gap closure {curves.field.oracle_gap_closure!r}"
                ),
                (
                    f"- CAI: `{curves.cai.policy_gate.status}`; "
                    f"fixed/learned/oracle AUEBC "
                    f"{curves.cai.fixed_auebc!r}/"
                    f"{curves.cai.learned_auebc!r}/"
                    f"{curves.cai.oracle_auebc!r}; effect "
                    f"{curves.cai.baseline_minus_learned.point_estimate!r}, "
                    f"95% CI [{curves.cai.baseline_minus_learned.ci_lower!r}, "
                    f"{curves.cai.baseline_minus_learned.ci_upper!r}], "
                    f"domains {curves.cai.baseline_minus_learned.improved_domains}/6; "
                    f"oracle-gap closure {curves.cai.oracle_gap_closure!r}"
                ),
                f"- Task conditioning: `{curves.task_conditioning.gate.status}`",
                "",
                "## Task and surface controls",
                "",
                *(
                    f"- {name}: effect {value.point_estimate!r}, "
                    f"95% CI [{value.ci_lower!r}, {value.ci_upper!r}], "
                    f"domains {value.improved_domains}/6"
                    for name, value in (
                        (
                            "FIELD WRONG_TASK minus correct",
                            curves.task_conditioning.field_wrong_minus_correct,
                        ),
                        (
                            "FIELD NO_TASK minus correct",
                            curves.task_conditioning.field_no_task_minus_correct,
                        ),
                        (
                            "CAI WRONG_TASK minus correct",
                            curves.task_conditioning.cai_wrong_minus_correct,
                        ),
                        (
                            "CAI NO_TASK minus correct",
                            curves.task_conditioning.cai_no_task_minus_correct,
                        ),
                        (
                            "FIELD NO_SURFACE minus correct",
                            curves.surface_robustness.field_no_surface_minus_correct,
                        ),
                        (
                            "FIELD SHUFFLED_SURFACE minus correct",
                            curves.surface_robustness.field_shuffled_surface_minus_correct,
                        ),
                        (
                            "CAI NO_SURFACE minus correct",
                            curves.surface_robustness.cai_no_surface_minus_correct,
                        ),
                        (
                            "CAI SHUFFLED_SURFACE minus correct",
                            curves.surface_robustness.cai_shuffled_surface_minus_correct,
                        ),
                    )
                ),
                "",
                "## Source-only decision diagnostics",
                "",
                (
                    "- High-level decision accuracy: "
                    f"{diagnostics.high_level_decision_accuracy!r}"
                ),
                (
                    "- Primitive top-1 match: "
                    f"{diagnostics.primitive_top1_match!r}"
                ),
                (
                    "- Top-5 utility recall: "
                    f"{diagnostics.top5_utility_recall!r}"
                ),
                (
                    "- Expected teacher regret: "
                    f"{diagnostics.expected_teacher_regret!r}"
                ),
                (
                    "- Candidate-utility NDCG: "
                    f"{diagnostics.candidate_utility_ndcg!r}"
                ),
                (
                    "- FOCUS/BROADEN/REFINE proportions: "
                    + ", ".join(
                        f"{name}={value!r}"
                        for name, value in (
                            diagnostics.predicted_decision_proportions
                        )
                    )
                ),
                "- Teacher-to-predicted transition matrix:",
                *(
                    f"  - {teacher}->{predicted}: {value!r}"
                    for teacher, predicted, value in (
                        diagnostics.transition_proportions
                    )
                ),
                "",
                "## Frozen misleading-surface diagnostic",
                "",
                (
                    f"- G0 authority SHA-256: `{surface_authority.file_sha256}`; "
                    f"strata {surface_authority.stratum_counts!r}"
                ),
                *(
                    f"- {name}: effect {value.point_estimate!r}, "
                    f"95% CI [{value.ci_lower!r}, {value.ci_upper!r}], "
                    f"domains {value.improved_domains}/6"
                    for name, value in misleading
                ),
                "- This frozen subset is diagnostic only and does not alter the G1 gate.",
                "",
                "## Stopping",
                "",
                (
                    f"- FIELD: `{stopping.field.status}`; normalized saving "
                    f"{stopping.field.normalized_measurement_saving!r}, "
                    f"95% CI [{stopping.field.saving_bootstrap.ci_lower!r}, "
                    f"{stopping.field.saving_bootstrap.ci_upper!r}], "
                    f"domains {stopping.field.saving_bootstrap.improved_domains}/6; "
                    f"loss ratio {stopping.field.task_loss_ratio!r}; premature "
                    f"rate {stopping.field.premature_stop_rate!r}; authorized "
                    f"domains {stopping.field.authorized_domains}/6"
                ),
                (
                    f"- CAI: `{stopping.cai.status}`; normalized saving "
                    f"{stopping.cai.normalized_measurement_saving!r}, "
                    f"95% CI [{stopping.cai.saving_bootstrap.ci_lower!r}, "
                    f"{stopping.cai.saving_bootstrap.ci_upper!r}], "
                    f"domains {stopping.cai.saving_bootstrap.improved_domains}/6; "
                    f"loss ratio {stopping.cai.task_loss_ratio!r}; premature "
                    f"rate {stopping.cai.premature_stop_rate!r}; authorized "
                    f"domains {stopping.cai.authorized_domains}/6"
                ),
                "",
                "All target policy trajectories were frozen before hidden target truth was opened.",
                "Inference uses 100,000 synchronized physical-specimen bootstrap replicates with equal held-out-domain weighting.",
                "",
            )
        ),
        encoding="ascii",
    )


def _validate_evidence(
    selections: tuple[G1OuterFormalSelection, ...],
    trajectories: tuple[G1TargetTrajectoryRecord, ...],
    learned: tuple[G1TargetCurveRecord, ...],
    references: tuple[G1TargetReferenceCurveRecord, ...],
    outcomes: tuple[G1TargetStopOutcome, ...],
    curves: G1TargetCurveAnalysis,
    stopping: G1TargetStoppingAnalysis,
    teacher_banks: tuple[G1TeacherBankManifestRow, ...],
    diagnostic_banks: tuple[G1SourceDecisionDiagnosticBankFile, ...],
    diagnostics: tuple[G1SourceDecisionDiagnosticRecord, ...],
    surface_authority: G1SurfaceStratumAuthority,
    strata: tuple[G1SurfaceStratumRecord, ...],
) -> tuple[str, ...]:
    domains = tuple(sorted(row.outer_target for row in selections))
    if (
        type(selections) is not tuple
        or len(selections) != 6
        or len(set(domains)) != 6
        or any(type(row) is not G1OuterFormalSelection for row in selections)
        or type(trajectories) is not tuple
        or not trajectories
        or any(type(row) is not G1TargetTrajectoryRecord for row in trajectories)
        or {row.outer_target for row in trajectories} != set(domains)
        or type(learned) is not tuple
        or not learned
        or any(type(row) is not G1TargetCurveRecord for row in learned)
        or {row.outer_target for row in learned} != set(domains)
        or type(references) is not tuple
        or not references
        or any(type(row) is not G1TargetReferenceCurveRecord for row in references)
        or {row.outer_target for row in references} != set(domains)
        or type(outcomes) is not tuple
        or not outcomes
        or any(type(row) is not G1TargetStopOutcome for row in outcomes)
        or {row.outer_target for row in outcomes} != set(domains)
        or type(curves) is not G1TargetCurveAnalysis
        or type(stopping) is not G1TargetStoppingAnalysis
        or type(teacher_banks) is not tuple
        or len(teacher_banks) != 30
        or any(type(row) is not G1TeacherBankManifestRow for row in teacher_banks)
        or {
            (row.outer_target, row.source_domain) for row in teacher_banks
        }
        != {
            (outer, source) for outer in domains for source in domains if source != outer
        }
        or type(diagnostic_banks) is not tuple
        or len(diagnostic_banks) != 6
        or any(
            type(row) is not G1SourceDecisionDiagnosticBankFile
            for row in diagnostic_banks
        )
        or {row.outer_target for row in diagnostic_banks} != set(domains)
        or type(diagnostics) is not tuple
        or not diagnostics
        or any(
            type(row) is not G1SourceDecisionDiagnosticRecord
            for row in diagnostics
        )
        or {row.outer_target for row in diagnostics} != set(domains)
        or type(surface_authority) is not G1SurfaceStratumAuthority
        or type(strata) is not tuple
        or not strata
        or any(type(row) is not G1SurfaceStratumRecord for row in strata)
        or len({(row.outer_target, row.specimen_id) for row in strata})
        != len(strata)
        or len(strata) != surface_authority.record_count
        or surface_stratum_records_sha256(strata)
        != surface_authority.records_sha256
        or tuple(
            (
                name,
                sum(row.stratum == name for row in strata),
            )
            for name, _count in surface_authority.stratum_counts
        )
        != surface_authority.stratum_counts
    ):
        raise G1FormalPackageError("formal package evidence roster changed")
    selections_by_domain = {row.outer_target: row for row in selections}
    banks_by_domain = {row.outer_target: row for row in diagnostic_banks}
    if any(
        bank.source_domains != tuple(sorted(selection.source_domains))
        or bank.action_model_sha256 != selection.action_model_sha256
        or bank.manifest_sha256
        != selection.decision_diagnostic_manifest_sha256
        or bank.row_count
        != sum(row.outer_target == domain for row in diagnostics)
        or any(
            row.action_model_sha256 != selection.action_model_sha256
            or row.source_domain not in selection.source_domains
            for row in diagnostics
            if row.outer_target == domain
        )
        for domain, selection in selections_by_domain.items()
        for bank in (banks_by_domain[domain],)
    ):
        raise G1FormalPackageError("formal decision diagnostics changed after freeze")
    proposed_roster = {
        (row.outer_target, row.specimen_id)
        for row in learned
        if row.variant is TargetPolicyVariant.PROPOSED
    }
    if {(row.outer_target, row.specimen_id) for row in strata} != proposed_roster:
        raise G1FormalPackageError("frozen surface-stratum roster changed")
    return domains


def _write_package_files(
    root: Path,
    selections: tuple[G1OuterFormalSelection, ...],
    trajectories: tuple[G1TargetTrajectoryRecord, ...],
    learned: tuple[G1TargetCurveRecord, ...],
    references: tuple[G1TargetReferenceCurveRecord, ...],
    outcomes: tuple[G1TargetStopOutcome, ...],
    curves: G1TargetCurveAnalysis,
    stopping: G1TargetStoppingAnalysis,
    teacher_banks: tuple[G1TeacherBankManifestRow, ...],
    diagnostic_banks: tuple[G1SourceDecisionDiagnosticBankFile, ...],
    diagnostics: tuple[G1SourceDecisionDiagnosticRecord, ...],
    surface_authority: G1SurfaceStratumAuthority,
    strata: tuple[G1SurfaceStratumRecord, ...],
    config_path: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_bytes(config_path.read_bytes())
    proposed = tuple(
        row for row in learned if row.variant is TargetPolicyVariant.PROPOSED
    )
    roster = sorted(
        {
            (row.outer_target, row.specimen_id, row.specimen_sha256)
            for row in proposed
        }
    )
    _write_csv(
        root / "authorized_roster.csv",
        ("outer_target", "specimen_id", "specimen_sha256", "role"),
        [(domain, specimen, sha, "HELD_OUT_TARGET") for domain, specimen, sha in roster],
    )
    _write_csv(
        root / "deployment_geometry_audit.csv",
        (
            "outer_target",
            "specimen_id",
            "task",
            "grid_sha256",
            "warm_start_sha256",
            "evaluator_sha256",
            "warm_start_cells",
            "endpoint_budget",
            "nominal_checkpoints_json",
        ),
        [
            (
                row.outer_target,
                row.specimen_id,
                row.task.value,
                row.curve.grid_sha256,
                row.curve.warm_start_sha256,
                row.curve.evaluator_sha256,
                8,
                0.25,
                json.dumps(
                    [float(value) for value in row.curve.nominal_budgets],
                    separators=(",", ":"),
                ),
            )
            for row in sorted(
                proposed,
                key=lambda value: (
                    value.outer_target,
                    value.specimen_id,
                    value.task.value,
                ),
            )
        ],
    )
    _write_outer_selection(root / "outer_selection.json", selections)
    _write_csv(
        root / "teacher_bank_manifest.csv",
        (
            "outer_target",
            "source_domain",
            "row_count",
            "parquet_sha256",
            "records_sha256",
            "manifest_sha256",
        ),
        [
            (
                row.outer_target,
                row.source_domain,
                row.row_count,
                row.parquet_sha256,
                row.records_sha256,
                row.manifest_sha256,
            )
            for row in sorted(
                teacher_banks,
                key=lambda value: (value.outer_target, value.source_domain),
            )
        ],
    )
    dependencies = {
        row.outer_target: row.final_dependency_sha256 for row in trajectories
    }
    _write_csv(
        root / "model_manifest.csv",
        (
            "outer_target",
            "formal_selection_sha256",
            "selected_hyperparameters_sha256",
            "training_route",
            "model_name",
            "cai_context_mode",
            "task_token_mode",
            "tau",
            "learning_rate",
            "weight_decay",
            "dagger_iterations",
            "aawr_expectile",
            "aawr_beta",
            "aawr_authorized_tasks_json",
            "aawr_authorization_sha256",
            "base_hyperparameters_sha256",
            "action_selection_sha256",
            "action_model_sha256",
            "stop_model_sha256",
            "decision_diagnostic_manifest_sha256",
            "final_dependency_sha256",
        ),
        [
            (
                row.outer_target,
                row.state_sha256,
                row.action_hyperparameters.state_sha256,
                row.action_hyperparameters.route.value,
                row.action_hyperparameters.model_name.value,
                row.action_hyperparameters.cai_context_mode.value,
                row.action_hyperparameters.task_token_mode.value,
                row.action_hyperparameters.tau,
                row.action_hyperparameters.learning_rate,
                row.action_hyperparameters.weight_decay,
                row.action_hyperparameters.dagger_iterations,
                row.action_hyperparameters.aawr_expectile,
                row.action_hyperparameters.aawr_beta,
                json.dumps(
                    [
                        task.value
                        for task in row.action_hyperparameters.aawr_authorized_tasks
                    ],
                    separators=(",", ":"),
                ),
                row.action_hyperparameters.aawr_authorization_sha256,
                row.action_hyperparameters.base_hyperparameters_sha256,
                row.action_selection_sha256,
                row.action_model_sha256,
                row.stop_model_sha256,
                row.decision_diagnostic_manifest_sha256,
                dependencies[row.outer_target],
            )
            for row in selections
        ],
    )
    misleading = _misleading_surface_bootstraps(learned, strata)
    _write_curve_tables(root, learned, references, diagnostics, strata)
    _write_trajectory_tables(root, trajectories)
    _write_stopping(root / "stopping_results.csv", outcomes)
    _write_inference_tables(root, curves, stopping, misleading)
    diagnostic_summary = summarize_g1_source_decision_diagnostics(diagnostics)
    _write_decision(
        root / "decision_summary.json",
        curves,
        stopping,
        diagnostic_summary,
        surface_authority,
        misleading,
    )
    _write_report(
        root / "REPORT.md",
        curves,
        stopping,
        diagnostic_summary,
        surface_authority,
        misleading,
    )


def write_g1_formal_package(
    output_dir: str | Path,
    selections: tuple[G1OuterFormalSelection, ...],
    trajectories: tuple[G1TargetTrajectoryRecord, ...],
    learned: tuple[G1TargetCurveRecord, ...],
    references: tuple[G1TargetReferenceCurveRecord, ...],
    outcomes: tuple[G1TargetStopOutcome, ...],
    curves: G1TargetCurveAnalysis,
    stopping: G1TargetStoppingAnalysis,
    teacher_banks: tuple[G1TeacherBankManifestRow, ...],
    diagnostic_banks: tuple[G1SourceDecisionDiagnosticBankFile, ...],
    diagnostics: tuple[G1SourceDecisionDiagnosticRecord, ...],
    surface_authority: G1SurfaceStratumAuthority,
    strata: tuple[G1SurfaceStratumRecord, ...],
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> G1PackageValidation:
    _validate_evidence(
        selections,
        trajectories,
        learned,
        references,
        outcomes,
        curves,
        stopping,
        teacher_banks,
        diagnostic_banks,
        diagnostics,
        surface_authority,
        strata,
    )
    destination = Path(output_dir)
    if destination.exists():
        raise G1FormalPackageError("formal package destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _write_package_files(
            temporary,
            selections,
            trajectories,
            learned,
            references,
            outcomes,
            curves,
            stopping,
            teacher_banks,
            diagnostic_banks,
            diagnostics,
            surface_authority,
            strata,
            Path(config_path),
        )
        publish_g1_manifest(
            temporary,
            project_root=project_root,
            config_path=config_path,
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_g1_package(
        destination,
        project_root=project_root,
        config_path=config_path,
    )


__all__ = [
    "G1FormalPackageError",
    "G1TeacherBankManifestRow",
    "write_g1_formal_package",
]
