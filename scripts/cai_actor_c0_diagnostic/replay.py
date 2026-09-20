"""Native trajectory replay, physical-state caching, and mask-only C0 intervention."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cmc_bbdm.cai_agent_v3.actor_training import _actor_forward
from cmc_bbdm.cai_agent_v3.metrics import left_error_area_mpa
from cmc_bbdm.cai_agent_v3.policy import vlm_first_action_mask

from .context import ResourceLedger, TaskContext, atomic_json
from .models import FrozenInputs, load_frozen_inputs, state_dict_sha256


@dataclass(frozen=True)
class PhysicalState:
    surface: np.ndarray
    observed_cscan: np.ndarray
    measured: np.ndarray
    history: np.ndarray
    exact_cost: float
    actor_cost: np.float32
    remaining_cost: np.float32
    current_prediction_mpa: np.float32
    environment_legal: np.ndarray
    policy_prior: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class RolloutResult:
    condition: str
    actions: list[int]
    costs: list[float]
    predictions_mpa: list[float]
    traces: list[dict[str, Any]]
    pre_states: list[PhysicalState]
    terminal_state: PhysicalState
    target_mpa: float
    metrics: dict[str, float]


def _hash_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def physical_state_sha256(state: PhysicalState) -> str:
    digest = hashlib.sha256()
    for name in ("surface", "observed_cscan", "measured", "history", "environment_legal"):
        _hash_array(digest, name, np.asarray(getattr(state, name)))
    _hash_array(digest, "exact_cost", np.asarray([state.exact_cost], dtype=np.float64))
    _hash_array(digest, "actor_cost", np.asarray([state.actor_cost], dtype=np.float32))
    _hash_array(
        digest, "remaining_cost", np.asarray([state.remaining_cost], dtype=np.float32)
    )
    _hash_array(
        digest,
        "current_prediction_mpa",
        np.asarray([state.current_prediction_mpa], dtype=np.float32),
    )
    return digest.hexdigest()


def snapshot_indices(action_count: int) -> tuple[tuple[int, ...], int]:
    if action_count < 1:
        raise ValueError("a completed native trajectory must contain an action")
    pre = tuple(sorted({value for value in (0, 1, 2, 8, action_count - 1) if value < action_count}))
    return pre, action_count


def choose_action(logits: np.ndarray, mask: np.ndarray) -> int:
    values = np.asarray(logits)
    legal = np.asarray(mask, dtype=bool)
    if values.shape != (64,) or legal.shape != (64,) or not legal.any():
        raise ValueError("action selection requires 64 logits and a nonempty mask")
    return int(np.where(legal, values, -np.inf).argmax())


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError("cannot write empty replay table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _tensor(value: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(value).to(device)


def _make_state(
    inputs: FrozenInputs,
    specimen: int,
    measured: torch.Tensor,
    history: torch.Tensor,
    exact_cost: float,
    prediction: float,
) -> PhysicalState:
    measured_np = measured[0].detach().cpu().numpy().astype(bool, copy=True)
    legal = (~measured_np) & (
        exact_cost + inputs.cell_costs[specimen] <= 0.25 + 1e-12
    )
    cscan = inputs.bank.cscan_tokens[specimen]
    observed = np.where(measured_np[:, None], cscan, np.float32(0.0))
    return PhysicalState(
        surface=np.asarray(inputs.bank.surface_tokens[specimen], dtype=np.float32),
        observed_cscan=np.asarray(observed, dtype=np.float32),
        measured=measured_np,
        history=history[0].detach().cpu().numpy().astype(np.float32, copy=True),
        exact_cost=float(exact_cost),
        actor_cost=np.float32(exact_cost),
        remaining_cost=np.float32(0.25 - exact_cost),
        current_prediction_mpa=np.float32(prediction),
        environment_legal=np.asarray(legal, dtype=bool),
    )


def rollout(
    inputs: FrozenInputs,
    ledger: ResourceLedger,
    specimen: int,
    *,
    model_name: str,
    disable_external_c0: bool = False,
) -> RolloutResult:
    if model_name not in {"C", "N"}:
        raise ValueError("unknown frozen Actor")
    actor = inputs.actor_c if model_name == "C" else inputs.actor_n
    method = "VLM_SPATIAL_FEEDBACK" if model_name == "C" else "NO_VLM_SPATIAL_FEEDBACK"
    device = inputs.device
    surface = _tensor(inputs.bank.surface_tokens[specimen : specimen + 1], device)
    cscan = _tensor(inputs.bank.cscan_tokens[specimen : specimen + 1], device)
    measured = torch.zeros((1, 64), dtype=torch.bool, device=device)
    history = torch.zeros((1, 64), dtype=torch.float32, device=device)
    indicator = _tensor(inputs.features.indicator[specimen : specimen + 1], device)
    confidence = _tensor(inputs.features.confidence[specimen : specimen + 1], device)
    available = _tensor(inputs.features.available[specimen : specimen + 1], device)
    no_reliable = _tensor(inputs.features.no_reliable[specimen : specimen + 1], device)
    ledger.charge("predictor_forward_examples", 1)
    with torch.inference_mode():
        initial = float(
            inputs.predictor(surface, cscan, measured, cost=torch.zeros(1, device=device))[0]
        )
    costs = [0.0]
    predictions = [initial]
    actions: list[int] = []
    traces: list[dict[str, Any]] = []
    states: list[PhysicalState] = []
    while True:
        current = _make_state(inputs, specimen, measured, history, costs[-1], predictions[-1])
        if not current.environment_legal.any():
            terminal = current
            break
        states.append(current)
        legal = _tensor(current.environment_legal[None, :], device)
        proposal, reasons = vlm_first_action_mask(
            legal=legal,
            use_vlm=model_name == "C",
            action_count=torch.tensor([len(actions)], device=device),
            indicator=indicator,
            confidence=confidence,
            available=available,
            no_reliable=no_reliable,
        )
        original_proposal = proposal[0].detach().cpu().numpy().astype(bool)
        if disable_external_c0 and not actions:
            proposal = legal
            reason = "EXTERNAL_C0_DISABLED_C_PRIOR_UNCHANGED"
        else:
            reason = reasons[0]
        ledger.charge("actor_forward_examples", 1)
        with torch.inference_mode():
            scores, _ = _actor_forward(
                actor,
                method,
                surface=surface,
                cscan=cscan,
                measured=measured,
                history=history,
                indicator=indicator,
                confidence=confidence,
                available=available,
                no_reliable=no_reliable,
                current_prediction=torch.tensor([predictions[-1]], device=device),
                costs=torch.tensor([costs[-1]], device=device),
            )
        logits = scores[0].detach().cpu().numpy().astype(np.float32)
        proposal_np = proposal[0].detach().cpu().numpy().astype(bool)
        cell = choose_action(logits, proposal_np)
        before = costs[-1]
        measured[0, cell] = True
        history[0, cell] = (len(actions) + 1) / 64.0
        actions.append(cell)
        next_cost = float(np.sum(inputs.cell_costs[specimen, actions], dtype=np.float64))
        costs.append(next_cost)
        ledger.charge("predictor_forward_examples", 1)
        with torch.inference_mode():
            prediction = float(
                inputs.predictor(
                    surface,
                    cscan,
                    measured,
                    cost=torch.tensor([next_cost], dtype=torch.float32, device=device),
                )[0]
            )
        predictions.append(prediction)
        traces.append(
            {
                "action_index": len(actions),
                "cell": cell,
                "environment_legal": "".join("1" if value else "0" for value in current.environment_legal),
                "proposal_legal": "".join("1" if value else "0" for value in proposal_np),
                "original_c0_legal": "".join("1" if value else "0" for value in original_proposal),
                "c0_reason": reason,
                "before_cost": before,
                "after_cost": next_cost,
                "before_prediction_mpa": predictions[-2],
                "after_prediction_mpa": prediction,
                "raw_logits": [float(value) for value in logits],
            }
        )
    target = float(inputs.bank.targets_mpa[specimen])
    metrics = {
        "left_error_area_mpa": left_error_area_mpa(costs, predictions, target, end=0.25),
        "early_left_error_area_mpa": left_error_area_mpa(costs, predictions, target, end=0.0625),
        "final_error_mpa": abs(predictions[-1] - target),
        "final_cost": costs[-1],
        "action_count": float(len(actions)),
    }
    condition = "C_NO_C0" if disable_external_c0 else f"{model_name}_NATIVE"
    return RolloutResult(
        condition=condition,
        actions=actions,
        costs=costs,
        predictions_mpa=predictions,
        traces=traces,
        pre_states=states,
        terminal_state=terminal,
        target_mpa=target,
        metrics=metrics,
    )


def _expected_row(context: TaskContext, key: str, model_name: str) -> dict[str, str]:
    if model_name == "C":
        path = (
            context.path("c_release")
            / "w3/selected_episodes/vlm_spatial_feedback_seed1.csv.gz"
        )
    else:
        path = (
            context.path("historical_w3")
            / "candidate_episodes/no_vlm_spatial_feedback_seed1/update_001000.csv.gz"
        )
    return next(row for row in _read_gzip_csv(path) if row["specimen_key"] == key)


def _verify_reproduction(
    result: RolloutResult, expected: dict[str, str], acceptance: dict[str, Any]
) -> dict[str, Any]:
    expected_actions = [int(value) for value in expected["cells"].split(";") if value]
    expected_costs = np.asarray([float(value) for value in expected["costs"].split(";")])
    expected_predictions = np.asarray(
        [float(value) for value in expected["predictions_mpa"].split(";")]
    )
    actions_match = result.actions == expected_actions
    cost_difference = float(np.max(np.abs(np.asarray(result.costs) - expected_costs)))
    prediction_difference = float(
        np.max(np.abs(np.asarray(result.predictions_mpa) - expected_predictions))
    )
    passed = (
        actions_match
        and cost_difference <= acceptance["cost_abs_tolerance"]
        and prediction_difference <= acceptance["historical_prediction_abs_tolerance_mpa"]
    )
    if not passed:
        raise ValueError(
            "native reproduction failed: "
            f"actions={actions_match}, cost={cost_difference}, prediction={prediction_difference}"
        )
    return {
        "status": "PASS",
        "actions_match": actions_match,
        "maximum_cost_difference": cost_difference,
        "maximum_prediction_difference_mpa": prediction_difference,
    }


def _slug(key: str) -> str:
    return key.replace(":", "_")


def _trajectory_payload(key: str, result: RolloutResult, reproduction: Any) -> dict[str, Any]:
    return {
        "specimen_key": key,
        "condition": result.condition,
        "target_mpa": result.target_mpa,
        "actions": result.actions,
        "costs": result.costs,
        "predictions_mpa": result.predictions_mpa,
        "metrics": result.metrics,
        "trace": [dict(row) for row in result.traces],
        "reproduction": reproduction,
    }


def _save_state(path: Path, state: PhysicalState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        surface=state.surface,
        observed_cscan=state.observed_cscan,
        measured=state.measured,
        history=state.history,
        exact_cost=np.asarray(state.exact_cost, dtype=np.float64),
        actor_cost=np.asarray(state.actor_cost, dtype=np.float32),
        remaining_cost=np.asarray(state.remaining_cost, dtype=np.float32),
        current_prediction_mpa=np.asarray(state.current_prediction_mpa, dtype=np.float32),
        environment_legal=state.environment_legal,
    )


def _finalize_cached_replay(
    context: TaskContext,
    inputs: FrozenInputs,
    ledger: ResourceLedger,
    signature: str,
) -> dict[str, Any] | None:
    output = context.path("output")
    trajectories = sorted((output / "trajectories").glob("*/*.json"))
    state_manifest = output / "state_manifest.csv"
    interventions = output / "c0_intervention_results.csv"
    if len(trajectories) != 18 or not state_manifest.exists() or not interventions.exists():
        return None
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in trajectories]
    native = [row for row in payloads if row["condition"] in {"C_NATIVE", "N_NATIVE"}]
    no_c0 = [row for row in payloads if row["condition"] == "C_NO_C0"]
    if len(native) != 12 or len(no_c0) != 6:
        raise ValueError("cached replay inventory is incomplete")
    if any(row["reproduction"].get("status") != "PASS" for row in native):
        raise ValueError("cached native replay lacks a passing reproduction audit")
    full_no_c0 = [
        row for row in no_c0 if row.get("intervention_execution") == "FULL_MASK_ONLY_REPLAY"
    ]
    actor_examples = sum(len(row["actions"]) for row in native + full_no_c0)
    predictor_examples = sum(len(row["actions"]) + 1 for row in native + full_no_c0)
    ledger.reconcile_minimum(
        "actor_forward_examples", actor_examples, "DERIVED_FROM_CACHED_REPLAY_ACTION_COUNTS"
    )
    ledger.reconcile_minimum(
        "predictor_forward_examples",
        predictor_examples,
        "DERIVED_FROM_CACHED_REPLAY_STATE_COUNTS",
    )
    ledger.reconcile_minimum("native_full_episode_runs", 12, "TWELVE_PASSING_NATIVE_FILES")
    ledger.reconcile_minimum(
        "c_no_c0_full_episode_runs", len(full_no_c0), "FULL_MASK_ONLY_REPLAY_FILES"
    )
    ledger.reconcile_minimum(
        "full_episode_runs", 12 + len(full_no_c0), "NATIVE_PLUS_FULL_MASK_ONLY_FILES"
    )
    state_rows = _read_csv(state_manifest)
    if len(state_rows) > int(context.scope["states"]["maximum_common_states"]):
        raise ValueError("cached common-state cap exceeded")
    after_hashes = {
        "C": state_dict_sha256(inputs.actor_c),
        "N": state_dict_sha256(inputs.actor_n),
        "P_all": state_dict_sha256(inputs.predictor),
    }
    if after_hashes != inputs.parameter_hashes:
        raise ValueError("frozen model parameters changed during replay finalization")
    result = {
        "status": "REPLAY_COMPLETE",
        "native_reproductions": 12,
        "c_no_c0_full_runs": len(full_no_c0),
        "common_state_count": len(state_rows),
        "parameter_hashes_unchanged": True,
        "recovered_from_complete_cache": True,
    }
    context.complete_stage("replay", signature, result)
    return result


def replay_stage(context: TaskContext, *, device: str) -> dict[str, Any]:
    output = context.path("output")
    selected_path = output / "selected_cases.csv"
    signature = context.phase_signature(
        "replay",
        (
            context.config_path,
            selected_path,
            context.root / context.scope["models"]["C"]["checkpoint"],
            context.root / context.scope["models"]["N"]["checkpoint"],
            context.path("predictor"),
        ),
    )
    if context.stage_complete("replay", signature):
        return {"status": "REPLAY_REUSED"}
    inputs = load_frozen_inputs(context, device=device)
    ledger = ResourceLedger(output / "resource_usage.json", context.scope["limits"])
    cached = _finalize_cached_replay(context, inputs, ledger, signature)
    if cached is not None:
        return cached
    selected = _read_csv(selected_path)
    state_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    native_reproductions = 0
    intervention_runs = 0
    for case in selected:
        key = case["specimen_key"]
        specimen = int(case["feature_bank_index"])
        by_model: dict[str, RolloutResult] = {}
        for model_name in ("C", "N"):
            ledger.charge("native_full_episode_runs", 1)
            ledger.charge("full_episode_runs", 1)
            result = rollout(inputs, ledger, specimen, model_name=model_name)
            reproduction = _verify_reproduction(
                result, _expected_row(context, key, model_name), context.scope["acceptance"]
            )
            native_reproductions += 1
            by_model[model_name] = result
            destination = output / "trajectories" / _slug(key) / f"{model_name}_NATIVE.json"
            atomic_json(destination, _trajectory_payload(key, result, reproduction))
            intervention_rows.append(
                {"specimen_key": key, "condition": result.condition, **result.metrics,
                 "execution": "FULL_NATIVE_REPLAY", "c_native_minus_c_no_c0_area_mpa": ""}
            )

        c_native = by_model["C"]
        free_action = choose_action(
            np.asarray(c_native.traces[0]["raw_logits"], dtype=np.float32),
            c_native.pre_states[0].environment_legal,
        )
        if free_action == c_native.actions[0]:
            c_no_c0 = c_native
            reuse_reason = "DETERMINISTIC_SAME_FIRST_ACTION_REUSED_C_NATIVE"
            execution = "EXACT_NATIVE_REUSE"
        else:
            ledger.charge("c_no_c0_full_episode_runs", 1)
            ledger.charge("full_episode_runs", 1)
            c_no_c0 = rollout(
                inputs, ledger, specimen, model_name="C", disable_external_c0=True
            )
            intervention_runs += 1
            reuse_reason = "FIRST_ACTION_CHANGED_FULL_CLOSED_LOOP_REPLAY"
            execution = "FULL_MASK_ONLY_REPLAY"
        no_c0_payload = _trajectory_payload(key, c_no_c0, {"status": "NOT_NATIVE"})
        no_c0_payload["condition"] = "C_NO_C0"
        if execution == "EXACT_NATIVE_REUSE":
            no_c0_payload["trace"][0] = {
                **no_c0_payload["trace"][0],
                "proposal_legal": no_c0_payload["trace"][0]["environment_legal"],
                "c0_reason": "EXTERNAL_C0_DISABLED_C_PRIOR_UNCHANGED",
            }
        no_c0_payload["intervention_execution"] = execution
        no_c0_payload["intervention_reason"] = reuse_reason
        atomic_json(output / "trajectories" / _slug(key) / "C_NO_C0.json", no_c0_payload)
        area_effect = c_native.metrics["left_error_area_mpa"] - c_no_c0.metrics["left_error_area_mpa"]
        intervention_rows.append(
            {"specimen_key": key, "condition": "C_NO_C0", **c_no_c0.metrics,
             "execution": execution, "c_native_minus_c_no_c0_area_mpa": area_effect}
        )
        for row in intervention_rows[-3:-2]:
            row["c_native_minus_c_no_c0_area_mpa"] = area_effect

        unique: dict[str, dict[str, Any]] = {}
        for model_name, result in by_model.items():
            indices, _terminal = snapshot_indices(len(result.actions))
            for prefix in indices:
                state = result.pre_states[prefix]
                digest = physical_state_sha256(state)
                record = unique.setdefault(
                    digest,
                    {"state": state, "sources": [], "prefixes": [], "source_models": []},
                )
                record["sources"].append(f"{model_name}_NATIVE_PREFIX")
                record["prefixes"].append(prefix)
                record["source_models"].append(model_name)
        if len(unique) > 9:
            raise ValueError(f"more than nine common states for {key}")
        for order, (digest, record) in enumerate(unique.items()):
            state_id = f"s{order:02d}_{digest[:12]}"
            state_root = output / "states" / _slug(key) / state_id
            _save_state(state_root / "physical_state.npz", record["state"])
            metadata = {
                "specimen_key": key,
                "state_id": state_id,
                "physical_state_sha256": digest,
                "sources": record["sources"],
                "source_models": record["source_models"],
                "prefix_lengths": record["prefixes"],
                "action_count": int(np.count_nonzero(record["state"].measured)),
                "exact_cost": record["state"].exact_cost,
                "current_prediction_mpa": float(record["state"].current_prediction_mpa),
                "terminal_view_only": False,
            }
            atomic_json(state_root / "metadata.json", metadata)
            state_rows.append(metadata)
        for model_name, result in by_model.items():
            terminal_id = f"terminal_{model_name.lower()}"
            terminal_root = output / "states" / _slug(key) / terminal_id
            _save_state(terminal_root / "physical_state.npz", result.terminal_state)
            atomic_json(
                terminal_root / "metadata.json",
                {"specimen_key": key, "state_id": terminal_id,
                 "physical_state_sha256": physical_state_sha256(result.terminal_state),
                 "sources": [f"{model_name}_NATIVE_TERMINAL"],
                 "source_models": [model_name], "prefix_lengths": [len(result.actions)],
                 "action_count": len(result.actions), "exact_cost": result.costs[-1],
                 "current_prediction_mpa": result.predictions_mpa[-1],
                 "terminal_view_only": True},
            )
    _write_csv(output / "state_manifest.csv", state_rows)
    _write_csv(output / "c0_intervention_results.csv", intervention_rows)
    after_hashes = {
        "C": state_dict_sha256(inputs.actor_c),
        "N": state_dict_sha256(inputs.actor_n),
        "P_all": state_dict_sha256(inputs.predictor),
    }
    if after_hashes != inputs.parameter_hashes:
        raise ValueError("frozen model parameters changed during replay")
    result = {
        "status": "REPLAY_COMPLETE",
        "native_reproductions": native_reproductions,
        "c_no_c0_full_runs": intervention_runs,
        "common_state_count": len(state_rows),
        "parameter_hashes_unchanged": True,
    }
    context.complete_stage("replay", signature, result)
    return result


__all__ = [
    "PhysicalState",
    "RolloutResult",
    "choose_action",
    "physical_state_sha256",
    "replay_stage",
    "rollout",
    "snapshot_indices",
]
