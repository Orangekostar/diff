"""Visible-state evaluation episodes with exact acquisition accounting."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import torch

from .contracts import Method
from .environment import NativeCellGrid
from .features import FeatureBank
from .models import CAIActor, CommonCAIPredictor
from .perception import (
    VLMActorFeatures,
    highest_reliable_candidate_mask,
    proposal_decision_for_method,
)
from .policies import fixed_action_order
from .statistics import normalized_error_area_mpa


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    specimen_key: str
    method: Method
    seed: int
    cells: tuple[int, ...]
    costs: tuple[float, ...]
    predictions_mpa: tuple[float, ...]
    rows: tuple[dict[str, object], ...]
    normalized_error_area_mpa: float
    final_absolute_error_mpa: float


def _state_id(
    measured: np.ndarray, cscan_tokens: np.ndarray, *, action_count: int
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(measured, dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(cscan_tokens[measured]).tobytes())
    return f"state:{action_count}:{digest.hexdigest()}"


def _tensor(value: np.ndarray, *, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).to(device)


def _mask_string(mask: np.ndarray) -> str:
    values = np.asarray(mask, dtype=bool)
    if values.shape != (64,):
        raise ValueError("logged action mask is invalid")
    return "".join("1" if value else "0" for value in values)


def run_episode(
    bank: FeatureBank,
    *,
    specimen_index: int,
    method: Method,
    predictor: CommonCAIPredictor,
    actor: CAIActor | None,
    seed: int,
    endpoint_budget: float,
    device: str,
    sample_actions: bool,
) -> EpisodeResult:
    if not 0 <= specimen_index < len(bank.specimen_keys) or endpoint_budget <= 0.0:
        raise ValueError("episode request is invalid")
    if method.learned != (actor is not None):
        raise ValueError("learned method and actor identity differ")
    predictor.eval()
    if actor is not None:
        actor.eval()
    surface = _tensor(bank.surface_tokens[specimen_index : specimen_index + 1], device=device)
    cscan = _tensor(bank.cscan_tokens[specimen_index : specimen_index + 1], device=device)
    measured_np = np.zeros(64, dtype=bool)
    action_history_np = np.zeros(64, dtype=np.float32)
    grid = NativeCellGrid.from_shape(
        tuple(int(item) for item in bank.native_shapes[specimen_index])
    )
    features = VLMActorFeatures(
        region_indicator=bank.vlm_indicators[specimen_index].copy(),
        confidence=bank.vlm_confidences[specimen_index].copy(),
        reliable_candidates=highest_reliable_candidate_mask(
            bank.vlm_indicators[specimen_index],
            bank.vlm_confidences[specimen_index],
        ),
        available=bool(bank.vlm_available[specimen_index]),
        no_reliable_cue=bool(bank.vlm_no_reliable[specimen_index]),
    )
    order = (
        fixed_action_order(method, seed=seed)
        if method in {
            Method.SERPENTINE,
            Method.CENTER_FIRST,
            Method.GEOMETRY_SPREAD,
            Method.RANDOM,
        }
        else ()
    )
    order_cursor = 0
    costs = [0.0]
    predictor_started = time.perf_counter()
    with torch.inference_mode():
        measured = _tensor(measured_np[None, :], device=device)
        prediction = float(
            predictor(
                surface,
                cscan,
                measured,
                cost=torch.zeros(1, device=device),
            )[0].cpu()
        )
    current_prediction_latency = time.perf_counter() - predictor_started
    predictions = [prediction]
    cells: list[int] = []
    rows: list[dict[str, object]] = []
    initial_candidates = tuple(int(item) for item in np.flatnonzero(features.reliable_candidates))
    initial_action = -1
    initial_proposal_mask = ""
    initial_proposal_reason = ""
    if sample_actions:
        torch.manual_seed(seed)

    for call_index in range(64):
        before_cost = grid.measured_cost(set(cells))
        legal = grid.legal_mask(set(cells), endpoint_budget=endpoint_budget)
        if not bool(legal.any()):
            break
        proposal_decision = proposal_decision_for_method(
            method,
            features=features,
            legal_mask=legal,
            action_count=call_index,
        )
        proposal = proposal_decision.mask
        if call_index == 0:
            initial_proposal_mask = _mask_string(proposal)
            initial_proposal_reason = proposal_decision.reason
        restricted = bool(call_index == 0 and not np.array_equal(proposal, legal))
        observed_state_id = _state_id(
            measured_np, bank.cscan_tokens[specimen_index], action_count=call_index
        )

        if actor is None:
            actor_latency = 0.0
            while order_cursor < len(order) and not proposal[order[order_cursor]]:
                order_cursor += 1
            if order_cursor == len(order):
                raise ValueError("fixed policy exhausted before endpoint")
            action = order[order_cursor]
            order_cursor += 1
        else:
            actor_started = time.perf_counter()
            with torch.inference_mode():
                measured = _tensor(measured_np[None, :], device=device)
                scores, _ = actor(
                    surface,
                    cscan,
                    measured,
                    _tensor(action_history_np[None, :], device=device),
                    _tensor(bank.vlm_indicators[specimen_index : specimen_index + 1], device=device),
                    _tensor(bank.vlm_confidences[specimen_index : specimen_index + 1], device=device),
                    torch.tensor([bank.vlm_available[specimen_index]], device=device),
                    torch.tensor([bank.vlm_no_reliable[specimen_index]], device=device),
                    torch.tensor([predictions[-1]], dtype=torch.float32, device=device),
                    torch.tensor([before_cost], dtype=torch.float32, device=device),
                    torch.tensor(
                        [endpoint_budget - before_cost],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                masked = scores[0].masked_fill(
                    ~torch.from_numpy(proposal).to(device), -torch.inf
                )
                if sample_actions:
                    action = int(torch.distributions.Categorical(logits=masked).sample().cpu())
                else:
                    action = int(torch.argmax(masked).cpu())
            actor_latency = time.perf_counter() - actor_started
        if call_index == 0:
            initial_action = action
        cell = grid.cells[action]
        action_history_cells = ";".join(str(item) for item in cells)
        measured_np[action] = True
        action_history_np[action] = (call_index + 1) / 64.0
        cells.append(action)
        after_cost = grid.measured_cost(set(cells))
        predictor_started = time.perf_counter()
        with torch.inference_mode():
            measured = _tensor(measured_np[None, :], device=device)
            next_prediction = float(
                predictor(
                    surface,
                    cscan,
                    measured,
                    cost=torch.tensor([after_cost], dtype=torch.float32, device=device),
                )[0].cpu()
            )
        predictor_latency = time.perf_counter() - predictor_started
        costs.append(after_cost)
        predictions.append(next_prediction)
        rows.append(
            {
                "specimen_key": bank.specimen_keys[specimen_index],
                "dataset_id": bank.dataset_ids[specimen_index],
                "split": bank.splits[specimen_index],
                "method": method.value,
                "seed": seed,
                "vlm_cache_key": bank.vlm_cache_keys[specimen_index],
                "vlm_available": bool(bank.vlm_available[specimen_index]),
                "initial_candidates": ";".join(str(item) for item in initial_candidates),
                "initial_action": initial_action,
                "proposal_restricted": restricted,
                "initial_proposal_reason": initial_proposal_reason,
                "environment_legal_mask": _mask_string(legal),
                "initial_proposal_mask": initial_proposal_mask,
                "actor_call_index": call_index,
                "actor_latency_seconds": actor_latency,
                "prediction_before_latency_seconds": current_prediction_latency,
                "predictor_latency_seconds": predictor_latency,
                "observed_state_id": observed_state_id,
                "action_history_cells": action_history_cells,
                "before_cost": before_cost,
                "remaining_cost": endpoint_budget - before_cost,
                "after_cost": after_cost,
                "next_cell": action,
                "pixel_row_start": cell.row_start,
                "pixel_row_stop": cell.row_stop,
                "pixel_col_start": cell.col_start,
                "pixel_col_stop": cell.col_stop,
                "normalized_center_x": (cell.col_start + cell.col_stop) / (2.0 * grid.native_shape[1]),
                "normalized_center_y": (cell.row_start + cell.row_stop) / (2.0 * grid.native_shape[0]),
                "actual_revealed_pixels": cell.pixel_count,
                "billed_cost": cell.cost,
                "prediction_before_mpa": predictions[-2],
                "prediction_mpa": next_prediction,
            }
        )
        current_prediction_latency = predictor_latency
    target = float(bank.targets_mpa[specimen_index])
    area = normalized_error_area_mpa(
        costs=np.asarray(costs),
        predictions_mpa=np.asarray(predictions),
        target_mpa=target,
        endpoint_budget=endpoint_budget,
    )
    return EpisodeResult(
        specimen_key=bank.specimen_keys[specimen_index],
        method=method,
        seed=seed,
        cells=tuple(cells),
        costs=tuple(costs),
        predictions_mpa=tuple(predictions),
        rows=tuple(rows),
        normalized_error_area_mpa=area,
        final_absolute_error_mpa=abs(predictions[-1] - target),
    )


__all__ = ["EpisodeResult", "run_episode"]
