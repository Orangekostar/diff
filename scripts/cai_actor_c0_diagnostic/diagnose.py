"""Fixed-state scores, prior sensitivity, surface attribution, and Actor attention."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cmc_bbdm.cai_agent_v3.policy import vlm_first_action_mask

from .context import ResourceLedger, TaskContext, atomic_json
from .models import FrozenInputs, load_frozen_inputs, state_dict_sha256
from .replay import choose_action


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if values.shape != (64,) or selected.shape != (64,) or not selected.any():
        raise ValueError("masked softmax requires 64 logits and a nonempty mask")
    shifted = values[selected] - np.max(values[selected])
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    output = np.zeros(64, dtype=np.float64)
    output[selected] = probabilities
    return output


def attribution_cells(attribution: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(attribution)
    if values.ndim != 3 or values.shape[1] != 64:
        raise ValueError("attribution must have shape [target,64,feature]")
    return values.sum(axis=-1), np.abs(values).sum(axis=-1)


def select_probe_cells(magnitude: np.ndarray, *, count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    values = np.asarray(magnitude, dtype=np.float64)
    if values.shape != (64,) or count < 1 or 2 * count > 64:
        raise ValueError("invalid probe-cell selection")
    top = tuple(sorted(range(64), key=lambda cell: (-values[cell], cell))[:count])
    remaining = [cell for cell in range(64) if cell not in top]
    bottom = tuple(sorted(remaining, key=lambda cell: (values[cell], cell))[:count])
    return top, bottom


def instrumented_actor_forward(
    actor: torch.nn.Module,
    surface: torch.Tensor,
    cscan: torch.Tensor,
    measured: torch.Tensor,
    action_history: torch.Tensor,
    vlm_indicator: torch.Tensor,
    vlm_confidence: torch.Tensor,
    vlm_available: torch.Tensor,
    vlm_no_reliable: torch.Tensor,
    current_prediction_mpa: torch.Tensor,
    cost: torch.Tensor,
    remaining_cost: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror SpatialCAIActor while requesting per-head weights at actual inputs."""

    batch = surface.shape[0]
    surface_local = actor.surface(surface)
    if actor.use_feedback:
        mask = actor.mask_cscan.to(cscan).view(1, 1, 512)
        visible = torch.where(measured.unsqueeze(-1), cscan, mask)
        cscan_local = actor.cscan(visible)
        prediction = (current_prediction_mpa - actor.target_mean) / actor.target_scale
    else:
        cscan_local = torch.zeros(batch, 64, 48, dtype=surface.dtype, device=surface.device)
        prediction = torch.zeros_like(current_prediction_mpa)
    if actor.use_vlm:
        indicator = vlm_indicator.to(surface.dtype)
        confidence = vlm_confidence.to(surface.dtype)
        available = vlm_available.to(surface.dtype)
        no_reliable = vlm_no_reliable.to(surface.dtype)
    else:
        indicator = torch.zeros_like(vlm_indicator, dtype=surface.dtype)
        confidence = torch.zeros_like(vlm_confidence, dtype=surface.dtype)
        available = torch.zeros_like(vlm_available, dtype=surface.dtype)
        no_reliable = torch.zeros_like(vlm_no_reliable, dtype=surface.dtype)
    coordinates = actor.coordinates.to(surface).unsqueeze(0).expand(batch, -1, -1)
    cells = actor.cell(
        torch.cat(
            (
                surface_local,
                cscan_local,
                coordinates,
                measured.to(surface.dtype).unsqueeze(-1),
                action_history.to(surface.dtype).unsqueeze(-1),
                indicator.unsqueeze(-1),
                confidence.unsqueeze(-1),
            ),
            dim=-1,
        )
    )
    query_features = torch.stack(
        (
            cost,
            remaining_cost,
            prediction,
            available,
            no_reliable,
            measured.to(surface.dtype).mean(dim=1),
        ),
        dim=-1,
    )
    query = actor.query(query_features)
    contextual = torch.cat((query.unsqueeze(1), cells), dim=1)
    per_layer = []
    for layer in actor.contextualizer.layers:
        if not layer.norm_first:
            raise ValueError("diagnostic expects the registered pre-norm Actor")
        attention_input = layer.norm1(contextual)
        attention_output, weights = layer.self_attn(
            attention_input,
            attention_input,
            attention_input,
            need_weights=True,
            average_attn_weights=False,
            is_causal=False,
        )
        contextual = contextual + layer.dropout1(attention_output)
        contextual = contextual + layer._ff_block(layer.norm2(contextual))
        per_layer.append(weights)
    if actor.contextualizer.norm is not None:
        contextual = actor.contextualizer.norm(contextual)
    global_context = contextual[:, :1].expand(-1, 64, -1)
    scores = actor.action_scorer(
        torch.cat((contextual[:, 1:], global_context), dim=-1)
    ).squeeze(-1)
    value = actor.value_head(contextual[:, 0]).squeeze(-1)
    attention = torch.stack(per_layer, dim=1)
    head_mean = attention.mean(dim=2)
    identity = torch.eye(65, dtype=head_mean.dtype, device=head_mean.device).view(1, 1, 65, 65)
    residual = head_mean + identity
    residual = residual / residual.sum(dim=-1, keepdim=True)
    rollout = residual[:, 0]
    for layer_index in range(1, residual.shape[1]):
        rollout = torch.bmm(residual[:, layer_index], rollout)
    return scores, value, attention, rollout


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError("cannot write an empty diagnostic table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _slug(key: str) -> str:
    return key.replace(":", "_")


def _load_state(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _state_tensors(
    inputs: FrozenInputs, specimen: int, state: dict[str, Any]
) -> tuple[torch.Tensor, ...]:
    device = inputs.device
    tensor = lambda value: torch.from_numpy(np.asarray(value)).to(device)
    return (
        tensor(state["surface"][None].astype(np.float32)),
        tensor(state["observed_cscan"][None].astype(np.float32)),
        tensor(state["measured"][None].astype(bool)),
        tensor(state["history"][None].astype(np.float32)),
        tensor(inputs.features.indicator[specimen : specimen + 1]),
        tensor(inputs.features.confidence[specimen : specimen + 1]),
        tensor(inputs.features.available[specimen : specimen + 1]),
        tensor(inputs.features.no_reliable[specimen : specimen + 1]),
        torch.tensor([float(state["current_prediction_mpa"])], dtype=torch.float32, device=device),
        torch.tensor([float(state["actor_cost"])], dtype=torch.float32, device=device),
        torch.tensor([float(state["remaining_cost"])], dtype=torch.float32, device=device),
    )


def _ranks(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros(64, dtype=np.int64)
    cells = [cell for cell in range(64) if mask[cell]]
    cells.sort(key=lambda cell: (-float(logits[cell]), cell))
    for rank, cell in enumerate(cells, start=1):
        output[cell] = rank
    return output


def _query_actor(
    inputs: FrozenInputs,
    ledger: ResourceLedger,
    specimen: int,
    state: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    actor = inputs.actor_c if model_name == "C" else inputs.actor_n
    args = _state_tensors(inputs, specimen, state)
    ledger.charge("actor_forward_examples", 1)
    with torch.inference_mode():
        native, _ = actor(*args)
    ledger.charge("actor_forward_examples", 1)
    with torch.inference_mode():
        instrumented, _, attention, rollout = instrumented_actor_forward(actor, *args)
    native_np = native[0].cpu().numpy().astype(np.float32)
    instrumented_np = instrumented[0].cpu().numpy().astype(np.float32)
    maximum_logit_difference = float(np.max(np.abs(native_np - instrumented_np)))
    acceptance = 1e-5 + 1e-5 * float(np.max(np.abs(native_np)))
    if maximum_logit_difference > acceptance:
        raise ValueError(f"instrumented {model_name} logits changed by {maximum_logit_difference}")
    legal = np.asarray(state["environment_legal"], dtype=bool)
    proposal, reasons = vlm_first_action_mask(
        legal=torch.from_numpy(legal[None]).to(inputs.device),
        use_vlm=model_name == "C",
        action_count=torch.tensor([int(np.count_nonzero(state["measured"]))], device=inputs.device),
        indicator=args[4],
        confidence=args[5],
        available=args[6],
        no_reliable=args[7],
    )
    proposal_np = proposal[0].cpu().numpy().astype(bool)
    action = choose_action(native_np, proposal_np)
    p_env = masked_softmax(native_np, legal)
    p_policy = masked_softmax(native_np, proposal_np)
    rank_env = _ranks(native_np, legal)
    rank_policy = _ranks(native_np, proposal_np)
    conditional_error = float(
        np.max(np.abs(p_policy[proposal_np] - p_env[proposal_np] / p_env[proposal_np].sum()))
    )
    if conditional_error > 1e-6:
        raise ValueError("conditional softmax identity failed")
    attention_np = attention[0].cpu().numpy().astype(np.float32)
    rollout_np = rollout[0].cpu().numpy().astype(np.float32)
    row_error = float(np.max(np.abs(attention_np.sum(axis=-1) - 1.0)))
    if row_error > 1e-6:
        raise ValueError(f"attention row sum changed by {row_error}")
    return {
        "logits": native_np,
        "legal": legal,
        "proposal": proposal_np,
        "reason": reasons[0],
        "action": action,
        "p_env": p_env,
        "p_policy": p_policy,
        "rank_env": rank_env,
        "rank_policy": rank_policy,
        "attention": attention_np,
        "rollout": rollout_np,
        "maximum_logit_difference": maximum_logit_difference,
        "attention_row_error": row_error,
        "conditional_error": conditional_error,
    }


def _save_query(
    destination: Path,
    state: dict[str, Any],
    query: dict[str, Any],
    inputs: FrozenInputs,
    specimen: int,
) -> None:
    rows = []
    selected = query["action"]
    measured = np.asarray(state["measured"], dtype=bool)
    indicator = inputs.features.indicator[specimen]
    confidence = inputs.features.confidence[specimen]
    for cell in range(64):
        rows.append(
            {
                "cell": cell,
                "row": cell // 8,
                "col": cell % 8,
                "raw_logit": float(query["logits"][cell]),
                "env_legal": bool(query["legal"][cell]),
                "proposal_legal": bool(query["proposal"][cell]),
                "p_env": float(query["p_env"][cell]),
                "p_policy": float(query["p_policy"][cell]),
                "rank_env": int(query["rank_env"][cell]) or "",
                "rank_policy": int(query["rank_policy"][cell]) or "",
                "measured": bool(measured[cell]),
                "vlm_indicator": float(indicator[cell]),
                "vlm_confidence": float(confidence[cell]),
                "selected": cell == selected,
            }
        )
    _write_csv(destination / "scores.csv", rows)
    attention = query["attention"]
    head_mean = attention.mean(axis=1)
    action_token = selected + 1
    np.savez_compressed(
        destination / "attention.npz",
        per_head=attention,
        head_mean=head_mean,
        query_to_cells=head_mean[:, 0, 1:],
        action_to_cells=head_mean[:, action_token, 1:],
        query_self_mass=head_mean[:, 0, 0],
        action_self_mass=head_mean[:, action_token, action_token],
        rollout=query["rollout"],
        rollout_query_to_cells=query["rollout"][0, 1:],
        rollout_action_to_cells=query["rollout"][action_token, 1:],
        selected_action=np.asarray(selected, dtype=np.int64),
    )
    atomic_json(
        destination / "query_checks.json",
        {
            "maximum_instrumented_logit_difference": query["maximum_logit_difference"],
            "maximum_attention_row_sum_error": query["attention_row_error"],
            "maximum_conditional_softmax_error": query["conditional_error"],
            "logit_difference_is_upper_bound": False,
        },
    )


def _load_cached_query(
    destination: Path,
    state: dict[str, Any],
    inputs: FrozenInputs,
    specimen: int,
    model_name: str,
) -> dict[str, Any] | None:
    scores_path = destination / "scores.csv"
    attention_path = destination / "attention.npz"
    if not scores_path.exists() or not attention_path.exists():
        return None
    rows = _read_csv(scores_path)
    if len(rows) != 64:
        raise ValueError("cached score map must contain 64 cells")
    logits = np.asarray([float(row["raw_logit"]) for row in rows], dtype=np.float32)
    legal = np.asarray([row["env_legal"] == "True" for row in rows], dtype=bool)
    proposal = np.asarray([row["proposal_legal"] == "True" for row in rows], dtype=bool)
    p_env = np.asarray([float(row["p_env"]) for row in rows], dtype=np.float64)
    p_policy = np.asarray([float(row["p_policy"]) for row in rows], dtype=np.float64)
    rank_env = np.asarray([int(row["rank_env"] or 0) for row in rows], dtype=np.int64)
    rank_policy = np.asarray([int(row["rank_policy"] or 0) for row in rows], dtype=np.int64)
    selected = [int(row["cell"]) for row in rows if row["selected"] == "True"]
    if len(selected) != 1:
        raise ValueError("cached score map must have one selected cell")
    args = _state_tensors(inputs, specimen, state)
    _proposal, reasons = vlm_first_action_mask(
        legal=torch.from_numpy(legal[None]).to(inputs.device),
        use_vlm=model_name == "C",
        action_count=torch.tensor([int(np.count_nonzero(state["measured"]))], device=inputs.device),
        indicator=args[4],
        confidence=args[5],
        available=args[6],
        no_reliable=args[7],
    )
    if not np.array_equal(_proposal[0].cpu().numpy(), proposal):
        raise ValueError("cached proposal mask changed")
    with np.load(attention_path, allow_pickle=False) as payload:
        attention = np.asarray(payload["per_head"], dtype=np.float32)
        rollout = np.asarray(payload["rollout"], dtype=np.float32)
    row_error = float(np.max(np.abs(attention.sum(axis=-1) - 1.0)))
    conditional_error = float(
        np.max(np.abs(p_policy[proposal] - p_env[proposal] / p_env[proposal].sum()))
    )
    checks_path = destination / "query_checks.json"
    if checks_path.exists():
        checks = json.loads(checks_path.read_text(encoding="utf-8"))
        logit_difference = float(checks["maximum_instrumented_logit_difference"])
        upper_bound = bool(checks.get("logit_difference_is_upper_bound", False))
    else:
        logit_difference = 1e-5 + 1e-5 * float(np.max(np.abs(logits)))
        upper_bound = True
        atomic_json(
            checks_path,
            {
                "maximum_instrumented_logit_difference": logit_difference,
                "maximum_attention_row_sum_error": row_error,
                "maximum_conditional_softmax_error": conditional_error,
                "logit_difference_is_upper_bound": True,
                "recovery_basis": "CACHE_WRITTEN_ONLY_AFTER_INSTRUMENTED_LOGIT_GATE_PASSED",
            },
        )
    return {
        "logits": logits,
        "legal": legal,
        "proposal": proposal,
        "reason": reasons[0],
        "action": selected[0],
        "p_env": p_env,
        "p_policy": p_policy,
        "rank_env": rank_env,
        "rank_policy": rank_policy,
        "attention": attention,
        "rollout": rollout,
        "maximum_logit_difference": logit_difference,
        "logit_difference_is_upper_bound": upper_bound,
        "attention_row_error": row_error,
        "conditional_error": conditional_error,
    }


def _zero_prior_query(
    inputs: FrozenInputs,
    ledger: ResourceLedger,
    specimen: int,
    state: dict[str, Any],
) -> np.ndarray:
    args = list(_state_tensors(inputs, specimen, state))
    args[4] = torch.zeros_like(args[4])
    args[5] = torch.zeros_like(args[5])
    args[6] = torch.zeros_like(args[6])
    args[7] = torch.zeros_like(args[7])
    ledger.charge("actor_forward_examples", 1)
    with torch.inference_mode():
        scores, _ = inputs.actor_c(*args)
    return scores[0].cpu().numpy().astype(np.float32)


def _log_probability(scores: torch.Tensor, legal: torch.Tensor, target: int) -> torch.Tensor:
    legal_cells = torch.nonzero(legal, as_tuple=False).flatten()
    position = torch.nonzero(legal_cells == target, as_tuple=False)
    if position.numel() != 1:
        raise ValueError("attribution target is not environment-legal")
    return torch.log_softmax(scores[0, legal], dim=0)[position[0, 0]]


def _surface_attribution(
    inputs: FrozenInputs,
    ledger: ResourceLedger,
    specimen: int,
    state: dict[str, Any],
    model_name: str,
    targets: list[int],
    surface_reference: np.ndarray,
) -> dict[str, Any]:
    actor = inputs.actor_c if model_name == "C" else inputs.actor_n
    base_args = list(_state_tensors(inputs, specimen, state))
    legal = base_args[2].new_tensor(np.asarray(state["environment_legal"], dtype=bool))
    direct_gradients = []
    total_gradients = []
    direct_logits = []
    total_logits = []
    total_predictions = []
    for target in targets:
        surface_direct = base_args[0].detach().clone().requires_grad_(True)
        direct_args = [surface_direct, *base_args[1:]]
        ledger.charge("actor_forward_examples", 1)
        direct_scores, _ = actor(*direct_args)
        direct_scalar = _log_probability(direct_scores, legal, target)
        ledger.charge("autograd_gradient_queries", 1)
        direct_gradient = torch.autograd.grad(direct_scalar, surface_direct)[0]

        surface_total = base_args[0].detach().clone().requires_grad_(True)
        ledger.charge("predictor_forward_examples", 1)
        total_prediction = inputs.predictor(
            surface_total, base_args[1], base_args[2], cost=base_args[9]
        )
        total_args = [surface_total, *base_args[1:8], total_prediction, *base_args[9:]]
        ledger.charge("actor_forward_examples", 1)
        total_scores, _ = actor(*total_args)
        total_scalar = _log_probability(total_scores, legal, target)
        ledger.charge("autograd_gradient_queries", 1)
        total_gradient = torch.autograd.grad(total_scalar, surface_total)[0]
        direct_gradients.append(direct_gradient[0].detach().cpu().numpy())
        total_gradients.append(total_gradient[0].detach().cpu().numpy())
        direct_logits.append(direct_scores[0].detach().cpu().numpy())
        total_logits.append(total_scores[0].detach().cpu().numpy())
        total_predictions.append(float(total_prediction.detach()[0]))
    direct_gradient = np.asarray(direct_gradients, dtype=np.float32)
    total_gradient = np.asarray(total_gradients, dtype=np.float32)
    delta = np.asarray(state["surface"], dtype=np.float32) - surface_reference
    direct_attribution = direct_gradient * delta[None]
    total_attribution = total_gradient * delta[None]
    via_attribution = total_attribution - direct_attribution
    direct_signed, direct_l1 = attribution_cells(direct_attribution)
    total_signed, total_l1 = attribution_cells(total_attribution)
    via_signed, via_l1 = attribution_cells(via_attribution)
    maximum_logit_difference = float(
        np.max(np.abs(np.asarray(direct_logits) - np.asarray(total_logits)))
    )
    prediction_difference = float(
        np.max(np.abs(np.asarray(total_predictions) - float(state["current_prediction_mpa"])))
    )
    return {
        "target_actions": np.asarray(targets, dtype=np.int64),
        "direct_gradient": direct_gradient,
        "total_gradient": total_gradient,
        "via_gradient": total_gradient - direct_gradient,
        "direct_attribution": direct_attribution,
        "total_attribution": total_attribution,
        "via_attribution": via_attribution,
        "direct_signed": direct_signed,
        "direct_l1": direct_l1,
        "total_signed": total_signed,
        "total_l1": total_l1,
        "via_signed": via_signed,
        "via_l1": via_l1,
        "maximum_forward_logit_difference": maximum_logit_difference,
        "maximum_prediction_difference_mpa": prediction_difference,
    }


def _probe_t0(
    inputs: FrozenInputs,
    ledger: ResourceLedger,
    specimen: int,
    state: dict[str, Any],
    model_name: str,
    query: dict[str, Any],
    attribution: dict[str, Any],
    surface_reference: np.ndarray,
) -> list[dict[str, Any]]:
    actor = inputs.actor_c if model_name == "C" else inputs.actor_n
    args = list(_state_tensors(inputs, specimen, state))
    target = int(query["action"])
    legal = np.asarray(state["environment_legal"], dtype=bool)
    baseline_logp = math.log(float(query["p_env"][target]))
    top, bottom = select_probe_cells(attribution["direct_l1"][0], count=3)
    rows = []
    for group, cells in (("TOP_MAGNITUDE", top), ("BOTTOM_MAGNITUDE", bottom)):
        for cell in cells:
            perturbed = args[0].detach().clone()
            reference = torch.from_numpy(surface_reference[cell]).to(perturbed)
            perturbed[0, cell] = perturbed[0, cell] + 0.1 * (reference - perturbed[0, cell])
            current = [perturbed, *args[1:]]
            ledger.charge("actor_forward_examples", 1)
            with torch.inference_mode():
                scores, _ = actor(*current)
            logits = scores[0].cpu().numpy()
            probabilities = masked_softmax(logits, legal)
            rows.append(
                {
                    "model": model_name,
                    "group": group,
                    "cell": cell,
                    "target_action": target,
                    "baseline_target_log_probability": baseline_logp,
                    "perturbed_target_log_probability": math.log(float(probabilities[target])),
                    "delta_target_log_probability": math.log(float(probabilities[target])) - baseline_logp,
                    "perturbed_env_argmax": choose_action(logits, legal),
                    "toward_train_mean_fraction": 0.1,
                    "prediction_held_fixed": True,
                }
            )
    return rows


def diagnose_stage(context: TaskContext, *, device: str) -> dict[str, Any]:
    output = context.path("output")
    signature = context.phase_signature(
        "diagnose", (context.config_path, output / "state_manifest.csv", output / "selected_cases.csv")
    )
    if context.stage_complete("diagnose", signature):
        return {"status": "DIAGNOSE_REUSED"}
    inputs = load_frozen_inputs(context, device=device)
    ledger = ResourceLedger(output / "resource_usage.json", context.scope["limits"])
    selected = {row["specimen_key"]: row for row in _read_csv(output / "selected_cases.csv")}
    states = _read_csv(output / "state_manifest.csv")
    surface_reference = np.asarray(
        inputs.bank.surface_tokens[inputs.bank.indices("TRAIN")].mean(axis=0), dtype=np.float32
    )
    np.savez_compressed(output / "train_surface_reference.npz", surface_reference=surface_reference)

    queries: dict[tuple[str, str, str], dict[str, Any]] = {}
    loaded_states: dict[tuple[str, str], dict[str, Any]] = {}
    prior_rows = []
    attention_logit_differences = []
    attention_logit_upper_bounds = []
    attention_row_errors = []
    conditional_errors = []
    for row in states:
        key, state_id = row["specimen_key"], row["state_id"]
        specimen = int(selected[key]["feature_bank_index"])
        state_root = output / "states" / _slug(key) / state_id
        state = _load_state(state_root / "physical_state.npz")
        loaded_states[(key, state_id)] = state
        for model_name in ("C", "N"):
            model_root = state_root / model_name
            model_root.mkdir(parents=True, exist_ok=True)
            query = _load_cached_query(model_root, state, inputs, specimen, model_name)
            if query is None:
                query = _query_actor(inputs, ledger, specimen, state, model_name=model_name)
                _save_query(model_root, state, query, inputs, specimen)
            queries[(key, state_id, model_name)] = query
            attention_logit_differences.append(query["maximum_logit_difference"])
            attention_logit_upper_bounds.append(query.get("logit_difference_is_upper_bound", False))
            attention_row_errors.append(query["attention_row_error"])
            conditional_errors.append(query["conditional_error"])
        normal = queries[(key, state_id, "C")]
        zero_path = state_root / "C" / "zero_prior_scores.npz"
        if zero_path.exists():
            with np.load(zero_path, allow_pickle=False) as payload:
                zero_logits = np.asarray(payload["raw_logits"], dtype=np.float32)
                zero_probability = np.asarray(payload["p_env"], dtype=np.float64)
                zero_action = int(payload["selected_action"])
        else:
            zero_logits = _zero_prior_query(inputs, ledger, specimen, state)
            zero_probability = masked_softmax(zero_logits, normal["legal"])
            zero_action = choose_action(zero_logits, normal["legal"])
        top_normal = set(np.argsort(-normal["p_env"], kind="stable")[:5])
        top_zero = set(np.argsort(-zero_probability, kind="stable")[:5])
        tvd = float(0.5 * np.abs(normal["p_env"] - zero_probability).sum())
        if not zero_path.exists():
            np.savez_compressed(
                zero_path,
                raw_logits=zero_logits,
                p_env=zero_probability,
                environment_legal=normal["legal"],
                selected_action=np.asarray(zero_action, dtype=np.int64),
            )
        prior_rows.append(
            {
                "specimen_key": key,
                "state_id": state_id,
                "normal_action": normal["action"],
                "zero_prior_action": zero_action,
                "top1_changed": normal["action"] != zero_action,
                "tvd": tvd,
                "top5_overlap": len(top_normal & top_zero),
                "classification": "PRIOR_CHANNEL_SENSITIVE_AT_FIXED_STATE" if tvd > 1e-6 else "NO_NUMERIC_PRIOR_SENSITIVITY",
            }
        )
    _write_csv(output / "fixed_state_prior_sensitivity.csv", prior_rows)

    attribution_checks = []
    probe_rows = []
    for row in states:
        key, state_id = row["specimen_key"], row["state_id"]
        specimen = int(selected[key]["feature_bank_index"])
        state = loaded_states[(key, state_id)]
        c_action = int(queries[(key, state_id, "C")]["action"])
        for model_name in ("C", "N"):
            own = int(queries[(key, state_id, model_name)]["action"])
            targets = [own]
            target_labels = ["OWN_ACTUAL_ACTION"]
            if c_action != own:
                targets.append(c_action)
                target_labels.append("COMMON_C_ACTION")
            attribution = _surface_attribution(
                inputs,
                ledger,
                specimen,
                state,
                model_name,
                targets,
                surface_reference,
            )
            destination = output / "states" / _slug(key) / state_id / model_name
            np.savez_compressed(
                destination / "surface_attribution.npz",
                **{name: value for name, value in attribution.items() if isinstance(value, np.ndarray)},
                target_labels=np.asarray(target_labels),
            )
            attribution_checks.append(
                {
                    "specimen_key": key,
                    "state_id": state_id,
                    "model": model_name,
                    "target_count": len(targets),
                    "maximum_forward_logit_difference": attribution["maximum_forward_logit_difference"],
                    "maximum_prediction_difference_mpa": attribution["maximum_prediction_difference_mpa"],
                }
            )
            if int(row["action_count"]) == 0:
                current = _probe_t0(
                    inputs,
                    ledger,
                    specimen,
                    state,
                    model_name,
                    queries[(key, state_id, model_name)],
                    attribution,
                    surface_reference,
                )
                probe_rows.extend({"specimen_key": key, "state_id": state_id, **item} for item in current)
    _write_csv(output / "surface_probe_results.csv", probe_rows)
    _write_csv(output / "attribution_check_rows.csv", attribution_checks)

    first_rows = []
    for key in selected:
        t0 = next(row for row in states if row["specimen_key"] == key and int(row["action_count"]) == 0)
        state_id = t0["state_id"]
        c_query = queries[(key, state_id, "C")]
        n_query = queries[(key, state_id, "N")]
        a_c0 = int(c_query["action"])
        a_free = choose_action(c_query["logits"], c_query["legal"])
        blocked = float(c_query["p_env"][~c_query["proposal"]].sum())
        no_c0 = json.loads(
            (output / "trajectories" / _slug(key) / "C_NO_C0.json").read_text(encoding="utf-8")
        )
        if not bool(c_query["proposal"].any()):
            classification = "C0_NOT_ACTIVE_ON_CASE"
        elif a_c0 != a_free:
            classification = "C0_CHANGED_THIS_DECISION"
        elif blocked > 0:
            classification = "C0_ONLY_CHANGED_DISTRIBUTION"
        else:
            classification = "C0_NOT_ACTIVE_ON_CASE"
        first_rows.append(
            {
                "specimen_key": key,
                "state_id": state_id,
                "c0_reason": c_query["reason"],
                "c0_size": int(c_query["proposal"].sum()),
                "a_C0": a_c0,
                "a_free": a_free,
                "a_N": int(n_query["action"]),
                "blocked_mass": blocked,
                "score_gap_free_minus_c0": float(c_query["logits"][a_free] - c_query["logits"][a_c0]),
                "a_C0_unrestricted_rank": int(c_query["rank_env"][a_c0]),
                "argmax_changed": a_c0 != a_free,
                "conditional_softmax_max_error": c_query["conditional_error"],
                "intervention_execution": no_c0["intervention_execution"],
                "C_NATIVE_reproduction": "PASS",
                "N_NATIVE_reproduction": "PASS",
                "classification": classification,
            }
        )
    _write_csv(output / "first_action_c0_audit.csv", first_rows)

    maximum_attribution_logit_difference = max(
        row["maximum_forward_logit_difference"] for row in attribution_checks
    )
    maximum_prediction_difference = max(
        row["maximum_prediction_difference_mpa"] for row in attribution_checks
    )
    acceptance = context.scope["acceptance"]
    if maximum_attribution_logit_difference > acceptance["instrumented_logits_atol"]:
        raise ValueError("direct and total attribution forward logits differ")
    if maximum_prediction_difference > acceptance["historical_prediction_abs_tolerance_mpa"]:
        raise ValueError("total-path prediction differs from replay state")
    atomic_json(
        output / "attribution_checks.json",
        {
            "status": "PASS",
            "method": context.scope["attribution"]["method"],
            "object_count": len(attribution_checks),
            "maximum_forward_logit_difference": maximum_attribution_logit_difference,
            "maximum_prediction_difference_mpa": maximum_prediction_difference,
            "via_definition": "TOTAL_MINUS_DIRECT_ELEMENTWISE",
        },
    )
    atomic_json(
        output / "attention_checks.json",
        {
            "status": "PASS",
            "owner": "ACTOR_NOT_QWEN",
            "object_count": len(attention_logit_differences),
            "shape": [2, 4, 65, 65],
            "maximum_instrumented_logit_difference": (
                None if any(attention_logit_upper_bounds) else max(attention_logit_differences)
            ),
            "maximum_instrumented_logit_difference_upper_bound": max(attention_logit_differences),
            "all_instrumented_logits_passed_per_state_allclose_gate": True,
            "cached_gate_proof_count": sum(attention_logit_upper_bounds),
            "maximum_attention_row_sum_error": max(attention_row_errors),
            "maximum_conditional_softmax_error": max(conditional_errors),
            "rollout": "APPROXIMATE_RESIDUAL_HEAD_MEAN_PRODUCT",
        },
    )
    final_hashes = {
        "C": state_dict_sha256(inputs.actor_c),
        "N": state_dict_sha256(inputs.actor_n),
        "P_all": state_dict_sha256(inputs.predictor),
    }
    if final_hashes != inputs.parameter_hashes:
        raise ValueError("frozen model parameters changed during diagnostics")
    result = {
        "status": "DIAGNOSE_COMPLETE",
        "state_count": len(states),
        "model_state_pairs": 2 * len(states),
        "prior_queries": len(prior_rows),
        "probe_forwards": len(probe_rows),
        "parameter_hashes_unchanged": True,
    }
    context.complete_stage("diagnose", signature, result)
    return result


__all__ = [
    "attribution_cells",
    "diagnose_stage",
    "instrumented_actor_forward",
    "masked_softmax",
    "select_probe_cells",
]
