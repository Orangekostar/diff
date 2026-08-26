"""Outer-domain execution helpers for exact-cost closed-loop MAVIS."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

import polars as pl

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.interpolation import reconstruct_measurement_state
from cmc_bbdm.mva.measurement_state import RefinementAction
from cmc_bbdm.mva.reconstruction_value import normalized_rgb_mse

from .authority import MAVISAuthority
from .config import MAVISConfig
from .contracts import InspectionState
from .dynamic_training import load_fitted_dynamic_checkpoint
from .historical_sources import HistoricalPolicySource
from .mris_training import load_fitted_mris_checkpoint
from .policy import DeployedDynamicScorer, FrozenCellScorer, ShuffledControlScorer
from .reveal import reveal_action_history
from .rollout import ScoutAndFocusCurve, rollout_scout_and_focus_curve
from .state_bank import PlannedAction, materialize_action_plan


class MAVISClosedLoopExecutionError(RuntimeError):
    """Raised when a P4 curve or evaluator violates the frozen outer fold."""


class CAIStateEvaluator(Protocol):
    model_state_sha256: str

    def predict_inspection_state(
        self,
        state: object,
        *,
        device: str,
    ) -> float: ...


def evaluate_inspection_curve(
    authority: MAVISAuthority,
    *,
    outer_domain: str,
    method: str,
    checkpoints: tuple[float, ...],
    states: tuple[InspectionState, ...],
    cai_evaluator: CAIStateEvaluator,
    device: str,
    interpolation: str = "bilinear",
) -> tuple[dict[str, object], ...]:
    if (
        type(authority) is not MAVISAuthority
        or type(outer_domain) is not str
        or not outer_domain
        or type(method) is not str
        or not method
        or type(checkpoints) is not tuple
        or not checkpoints
        or type(states) is not tuple
        or len(states) != len(checkpoints)
        or any(type(state) is not InspectionState for state in states)
        or not hasattr(cai_evaluator, "predict_inspection_state")
        or type(device) is not str
        or not device
    ):
        raise MAVISClosedLoopExecutionError("closed-loop curve request is invalid")
    specimen_ids = {state.specimen_id for state in states}
    if len(specimen_ids) != 1:
        raise MAVISClosedLoopExecutionError("closed-loop curve specimen changed")
    specimen_id = next(iter(specimen_ids))
    evaluation = authority.evaluation_view(specimen_id)
    if evaluation.dataset_id != outer_domain:
        raise MAVISClosedLoopExecutionError("closed-loop curve is not an outer target")
    model_hash = getattr(cai_evaluator, "model_state_sha256", None)
    if type(model_hash) is not str or len(model_hash) != 64:
        raise MAVISClosedLoopExecutionError("closed-loop CAI endpoint hash is invalid")
    rows: list[dict[str, object]] = []
    if hasattr(cai_evaluator, "predict_inspection_states"):
        predictions = cai_evaluator.predict_inspection_states(
            states,
            batch_size=len(states),
            device=device,
        )
    else:
        predictions = tuple(
            cai_evaluator.predict_inspection_state(state, device=device)
            for state in states
        )
    previous_cost = 0
    for checkpoint, state, prediction_raw in zip(
        checkpoints,
        states,
        predictions,
        strict=True,
    ):
        cap = float(checkpoint)
        quantized_initial_scout = not state.action_history and math.isclose(
            cap,
            state.initial_budget,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        if (
            not math.isfinite(cap)
            or not 0.0 < cap <= 1.0
            or state.exact_acquired_count < previous_cost
            or (
                state.effective_budget > cap + 1.0e-15
                and not quantized_initial_scout
            )
        ):
            raise MAVISClosedLoopExecutionError(
                "closed-loop exact checkpoint changed"
            )
        grid = build_acquisition_grid(
            *state.native_shape,
            initial_budget=state.initial_budget,
        )
        reconstruction = reconstruct_measurement_state(
            evaluation.full_scan,
            grid,
            state.measurement_state,
            interpolation=interpolation,
            specimen_id=specimen_id,
            dataset_id=outer_domain,
        )
        prediction = float(prediction_raw)
        error = abs(evaluation.true_cai - prediction)
        reconstruction_mse = normalized_rgb_mse(
            evaluation.full_scan,
            reconstruction.image,
        )
        if not all(math.isfinite(value) for value in (prediction, error, reconstruction_mse)):
            raise MAVISClosedLoopExecutionError("closed-loop endpoint value is invalid")
        rows.append(
            {
                "outer_domain": outer_domain,
                "specimen_id": specimen_id,
                "method": method,
                "nominal_checkpoint": cap,
                "initial_budget": state.initial_budget,
                "action_count": len(state.action_history),
                "exact_acquired_cost": state.exact_acquired_count,
                "native_count": state.native_count,
                "effective_budget": state.effective_budget,
                "target": evaluation.true_cai,
                "prediction": prediction,
                "absolute_error": error,
                "reconstruction_mse": reconstruction_mse,
                "inspection_state_sha256": state.state_sha256,
                "cai_model_state_sha256": model_hash,
                "reconstruction_state_sha256": reconstruction.output_sha256,
            }
        )
        previous_cost = state.exact_acquired_count
    return tuple(rows)


_P1_METHODS = (
    "uniform",
    "random",
    "reconstruction_driven",
    "one_shot_mechanical_oracle",
    "sequential_mechanical_oracle",
)
PRIMARY_CLOSED_LOOP_METHODS = (
    *_P1_METHODS,
    "global_mechanical",
    "mva_a5",
    "mvd_m1_o2",
    "mavis_no_feedback",
    "mavis_positions_only",
    "mavis_shuffled_content",
    "mavis_full",
    "mavis_raw_value",
    "mavis_value_per_cost",
)


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


def replay_state_manifest_row(
    authority: MAVISAuthority,
    row: dict[str, object],
) -> InspectionState:
    actions = tuple(
        RefinementAction(int(cell), int(source), int(target))
        for cell, source, target in zip(
            row["acquired_action_cell_indices"],
            row["acquired_action_from_levels"],
            row["acquired_action_to_levels"],
            strict=True,
        )
    )
    state = reveal_action_history(
        authority,
        authority.policy_context(str(row["specimen_id"])),
        initial_budget=float(row["initial_budget"]),
        checkpoint=float(row["endpoint_budget"]),
        actions=actions,
    )
    if (
        state.state_sha256 != row["inspection_state_sha256"]
        or state.exact_acquired_count != int(row["exact_acquired_cost"])
    ):
        raise MAVISClosedLoopExecutionError("P1 baseline state replay changed")
    return state


def _p1_action_rows(
    table: pl.DataFrame,
    *,
    outer_domain: str,
    specimen_id: str,
    method: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous = 0
    for state in table.sort("nominal_checkpoint").iter_rows(named=True):
        cells = state["acquired_action_cell_indices"]
        sources = state["acquired_action_from_levels"]
        targets = state["acquired_action_to_levels"]
        for index in range(previous, len(cells)):
            rows.append(
                {
                    "outer_domain": outer_domain,
                    "specimen_id": specimen_id,
                    "method": method,
                    "step": len(rows),
                    "nominal_checkpoint": float(state["nominal_checkpoint"]),
                    "cell_index": int(cells[index]),
                    "from_level": int(sources[index]),
                    "to_level": int(targets[index]),
                    "raw_score": None,
                    "objective_score": None,
                    "decision_confidence": None,
                    "exact_cost_before": None,
                    "exact_cost_after": None,
                    "state_sha256_before": None,
                    "state_sha256_after": None,
                    "feedback_used": method == "sequential_mechanical_oracle",
                    "source": "p1_frozen_state_bank",
                }
            )
        previous = len(cells)
    return rows


def _plan_action_rows(
    plan: tuple[PlannedAction, ...],
    *,
    outer_domain: str,
    specimen_id: str,
    method: str,
) -> list[dict[str, object]]:
    return [
        {
            "outer_domain": outer_domain,
            "specimen_id": specimen_id,
            "method": method,
            "step": index,
            "nominal_checkpoint": item.nominal_checkpoint,
            "cell_index": item.action.cell_index,
            "from_level": item.action.from_level,
            "to_level": item.action.to_level,
            "raw_score": None,
            "objective_score": None,
            "decision_confidence": None,
            "exact_cost_before": None,
            "exact_cost_after": None,
            "state_sha256_before": None,
            "state_sha256_after": None,
            "feedback_used": method == "mva_a5",
            "source": "frozen_historical_trajectory",
        }
        for index, item in enumerate(plan)
    ]


def _rollout_action_rows(
    curve: ScoutAndFocusCurve,
    *,
    outer_domain: str,
    method: str,
) -> list[dict[str, object]]:
    return [
        {
            "outer_domain": outer_domain,
            "specimen_id": curve.specimen_id,
            "method": method,
            "step": step.step,
            "nominal_checkpoint": step.nominal_checkpoint,
            "cell_index": step.action.cell_index,
            "from_level": step.action.from_level,
            "to_level": step.action.to_level,
            "raw_score": step.raw_score,
            "objective_score": step.objective_score,
            "decision_confidence": step.decision_confidence,
            "exact_cost_before": step.exact_cost_before,
            "exact_cost_after": step.exact_cost_after,
            "state_sha256_before": step.state_sha256_before,
            "state_sha256_after": step.state_sha256_after,
            "feedback_used": step.feedback_used,
            "source": "mavis_causal_rollout",
        }
        for step in curve.steps
    ]


def _full_scan_state(
    authority: MAVISAuthority,
    *,
    specimen_id: str,
    initial_budget: float,
) -> InspectionState:
    actions = tuple(
        RefinementAction(cell, 0, 1) for cell in range(64)
    ) + tuple(RefinementAction(cell, 1, 2) for cell in range(64))
    state = reveal_action_history(
        authority,
        authority.policy_context(specimen_id),
        initial_budget=initial_budget,
        checkpoint=1.0,
        actions=actions,
    )
    if state.exact_acquired_count != state.native_count:
        raise MAVISClosedLoopExecutionError("full-scan anchor is incomplete")
    return state


def _target_donor_lookup(
    donor_mapping: pl.DataFrame,
    *,
    outer_domain: str,
    target_ids: tuple[str, ...],
) -> dict[str, str]:
    required = {
        "outer_domain",
        "recipient_id",
        "recipient_domain",
        "recipient_pool",
        "donor_id",
    }
    if (
        not isinstance(donor_mapping, pl.DataFrame)
        or not required <= set(donor_mapping.columns)
        or type(outer_domain) is not str
        or not outer_domain
        or type(target_ids) is not tuple
        or not target_ids
        or len(set(target_ids)) != len(target_ids)
    ):
        raise MAVISClosedLoopExecutionError("P4 donor mapping request is invalid")
    selected = donor_mapping.filter(
        (pl.col("outer_domain") == outer_domain)
        & (pl.col("recipient_domain") == outer_domain)
        & (pl.col("recipient_pool") == "target")
    )
    if (
        selected.height != len(target_ids)
        or selected.get_column("recipient_id").n_unique() != selected.height
        or set(selected.get_column("recipient_id")) != set(target_ids)
        or selected.filter(pl.col("recipient_id") == pl.col("donor_id")).height
    ):
        raise MAVISClosedLoopExecutionError("P4 shuffled donor roster is incomplete")
    return {
        str(row["recipient_id"]): str(row["donor_id"])
        for row in selected.iter_rows(named=True)
    }


def run_closed_loop_outer_domain(
    authority: MAVISAuthority,
    config: MAVISConfig,
    *,
    outer_domain: str,
    p1_states: pl.DataFrame,
    donor_mapping: pl.DataFrame,
    historical_source: HistoricalPolicySource,
    p2_checkpoint_root: str | Path,
    p3_checkpoint_root: str | Path,
    output_root: str | Path,
    device: str,
) -> Path:
    if (
        type(authority) is not MAVISAuthority
        or type(config) is not MAVISConfig
        or outer_domain not in config.domain_order
        or not isinstance(p1_states, pl.DataFrame)
        or not isinstance(donor_mapping, pl.DataFrame)
        or type(historical_source) is not HistoricalPolicySource
        or type(device) is not str
        or not device
    ):
        raise MAVISClosedLoopExecutionError("P4 outer worker request is invalid")
    target_ids = tuple(
        specimen
        for specimen, domain in zip(
            authority.specimen_ids,
            authority.dataset_ids,
            strict=True,
        )
        if domain == outer_domain
    )
    target_states = p1_states.filter(pl.col("domain_id") == outer_domain)
    expected_p1_rows = len(target_ids) * len(_P1_METHODS) * len(config.checkpoints)
    if (
        not target_ids
        or target_states.height != expected_p1_rows
        or set(target_states.get_column("method").unique()) != set(_P1_METHODS)
    ):
        raise MAVISClosedLoopExecutionError("P4 P1 target roster is incomplete")
    p2 = {
        mode: load_fitted_mris_checkpoint(
            _checkpoint(p2_checkpoint_root, outer_domain, mode)
        )
        for mode in ("real", "positions_only", "shuffled")
    }
    p3 = {
        mode: load_fitted_dynamic_checkpoint(
            _checkpoint(p3_checkpoint_root, outer_domain, mode)
        )
        for mode in ("real", "positions_only", "shuffled")
    }
    if any(
        model.outer_domain != outer_domain or model.mode != mode
        for mode, model in p2.items()
    ) or any(model.outer_domain != outer_domain for model in p3.values()):
        raise MAVISClosedLoopExecutionError("P4 model outer fold changed")
    donor_lookup = _target_donor_lookup(
        donor_mapping,
        outer_domain=outer_domain,
        target_ids=target_ids,
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / outer_domain
    if destination.exists():
        raise MAVISClosedLoopExecutionError("P4 outer worker output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{outer_domain}.", dir=root))
    try:
        prediction_rows: list[dict[str, object]] = []
        trajectory_rows: list[dict[str, object]] = []
        full_scan_rows: list[dict[str, object]] = []
        for specimen_id in target_ids:
            initial_budget = float(config.initial_budget_by_domain[outer_domain])
            specimen_table = target_states.filter(
                pl.col("specimen_id") == specimen_id
            )
            for method in _P1_METHODS:
                table = specimen_table.filter(pl.col("method") == method).sort(
                    "nominal_checkpoint"
                )
                states = tuple(
                    replay_state_manifest_row(authority, row)
                    for row in table.iter_rows(named=True)
                )
                prediction_rows.extend(
                    evaluate_inspection_curve(
                        authority,
                        outer_domain=outer_domain,
                        method=method,
                        checkpoints=config.checkpoints,
                        states=states,
                        cai_evaluator=p2["real"],
                        device=device,
                    )
                )
                trajectory_rows.extend(
                    _p1_action_rows(
                        table,
                        outer_domain=outer_domain,
                        specimen_id=specimen_id,
                        method=method,
                    )
                )
            historical = historical_source.action_plans(
                specimen_id=specimen_id,
                dataset_id=outer_domain,
                outer_domain=outer_domain,
            )
            for method, plan in historical.items():
                trajectory = materialize_action_plan(
                    authority,
                    specimen_id=specimen_id,
                    method=method,
                    seed=None,
                    initial_budget=initial_budget,
                    checkpoints=config.checkpoints,
                    actions=plan,
                )
                prediction_rows.extend(
                    evaluate_inspection_curve(
                        authority,
                        outer_domain=outer_domain,
                        method=method,
                        checkpoints=config.checkpoints,
                        states=tuple(
                            snapshot.inspection_state
                            for snapshot in trajectory.snapshots
                        ),
                        cai_evaluator=p2["real"],
                        device=device,
                    )
                )
                trajectory_rows.extend(
                    _plan_action_rows(
                        plan,
                        outer_domain=outer_domain,
                        specimen_id=specimen_id,
                        method=method,
                    )
                )
            o2_curve = rollout_scout_and_focus_curve(
                authority,
                specimen_id=specimen_id,
                initial_budget=initial_budget,
                checkpoints=config.checkpoints,
                scorer=FrozenCellScorer(
                    historical_source.o2_scores(
                        specimen_id=specimen_id,
                        dataset_id=outer_domain,
                        outer_domain=outer_domain,
                    )
                ),
                objective="direct_cost_aware",
                feedback=False,
            )
            prediction_rows.extend(
                evaluate_inspection_curve(
                    authority,
                    outer_domain=outer_domain,
                    method="mvd_m1_o2",
                    checkpoints=config.checkpoints,
                    states=o2_curve.checkpoint_states,
                    cai_evaluator=p2["real"],
                    device=device,
                )
            )
            trajectory_rows.extend(
                _rollout_action_rows(
                    o2_curve,
                    outer_domain=outer_domain,
                    method="mvd_m1_o2",
                )
            )
            real_scorer = DeployedDynamicScorer(
                mris_model=p2["real"],
                dynamic_model=p3["real"],
                device=device,
            )
            positions_scorer = DeployedDynamicScorer(
                mris_model=p2["positions_only"],
                dynamic_model=p3["positions_only"],
                device=device,
            )
            shuffled_scorer = ShuffledControlScorer(
                mris_model=p2["shuffled"],
                dynamic_model=p3["shuffled"],
                authority=authority,
                donor_specimen_id=donor_lookup[specimen_id],
                device=device,
            )
            mavis_cases = (
                ("mavis_no_feedback", real_scorer, "direct_cost_aware", False),
                ("mavis_positions_only", positions_scorer, "direct_cost_aware", True),
                ("mavis_shuffled_content", shuffled_scorer, "direct_cost_aware", True),
                ("mavis_full", real_scorer, "direct_cost_aware", True),
                ("mavis_raw_value", real_scorer, "raw_score", True),
                ("mavis_value_per_cost", real_scorer, "value_per_exact_cost", True),
            )
            for method, scorer, objective, feedback in mavis_cases:
                curve = rollout_scout_and_focus_curve(
                    authority,
                    specimen_id=specimen_id,
                    initial_budget=initial_budget,
                    checkpoints=config.checkpoints,
                    scorer=scorer,
                    objective=objective,
                    feedback=feedback,
                )
                prediction_rows.extend(
                    evaluate_inspection_curve(
                        authority,
                        outer_domain=outer_domain,
                        method=method,
                        checkpoints=config.checkpoints,
                        states=curve.checkpoint_states,
                        cai_evaluator=p2["real"],
                        device=device,
                    )
                )
                trajectory_rows.extend(
                    _rollout_action_rows(
                        curve,
                        outer_domain=outer_domain,
                        method=method,
                    )
                )
            full_scan_rows.extend(
                evaluate_inspection_curve(
                    authority,
                    outer_domain=outer_domain,
                    method="full_scan",
                    checkpoints=(1.0,),
                    states=(
                        _full_scan_state(
                            authority,
                            specimen_id=specimen_id,
                            initial_budget=initial_budget,
                        ),
                    ),
                    cai_evaluator=p2["real"],
                    device=device,
                )
            )
        predictions = pl.DataFrame(prediction_rows, infer_schema_length=None).sort(
            ["specimen_id", "method", "nominal_checkpoint"]
        )
        trajectories = pl.DataFrame(trajectory_rows, infer_schema_length=None).sort(
            ["specimen_id", "method", "step"]
        )
        full_scan = pl.DataFrame(full_scan_rows, infer_schema_length=None).sort(
            "specimen_id"
        )
        if (
            predictions.height
            != len(target_ids)
            * len(PRIMARY_CLOSED_LOOP_METHODS)
            * len(config.checkpoints)
            or set(predictions.get_column("method").unique())
            != set(PRIMARY_CLOSED_LOOP_METHODS)
            or full_scan.height != len(target_ids)
        ):
            raise MAVISClosedLoopExecutionError("P4 output method roster is incomplete")
        predictions.write_parquet(
            temporary / "predictions.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        trajectories.write_parquet(
            temporary / "trajectories.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        full_scan.write_parquet(
            temporary / "full_scan_anchors.parquet",
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
                "methods": list(PRIMARY_CLOSED_LOOP_METHODS),
                "target_specimen_count": len(target_ids),
                "prediction_count": predictions.height,
                "trajectory_row_count": trajectories.height,
                "full_scan_anchor_count": full_scan.height,
                "p2_model_state_sha256": {
                    mode: model.model_state_sha256 for mode, model in p2.items()
                },
                "p3_model_state_sha256": {
                    mode: model.model_state_sha256 for mode, model in p3.items()
                },
                "historical_source_state_sha256": historical_source.state_sha256,
                "target_true_cai_used_by_policy": False,
                "future_target_content_used_by_policy": False,
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
    "PRIMARY_CLOSED_LOOP_METHODS",
    "CAIStateEvaluator",
    "MAVISClosedLoopExecutionError",
    "evaluate_inspection_curve",
    "replay_state_manifest_row",
    "run_closed_loop_outer_domain",
]
