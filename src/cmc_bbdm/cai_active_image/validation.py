"""Focused scientific-contract validators for CAI v2 artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .contracts import Method


def validate_common_predictor_identity(hashes: Mapping[Method, str]) -> None:
    if not hashes or any(type(method) is not Method for method in hashes):
        raise ValueError("common predictor identity map is invalid")
    values = tuple(hashes.values())
    if any(type(value) is not str or len(value) != 64 for value in values):
        raise ValueError("common predictor identity hash is invalid")
    if len(set(values)) != 1:
        raise ValueError("common predictor differs across methods")


def validate_trajectory_rows(rows: Sequence[Mapping[str, object]]) -> None:
    previous_after = 0.0
    endpoint_budget: float | None = None
    observed_states: set[str] = set()
    for expected_index, row in enumerate(rows):
        required = {
            "vlm_cache_key",
            "vlm_available",
            "initial_candidates",
            "initial_action",
            "proposal_restricted",
            "initial_proposal_reason",
            "environment_legal_mask",
            "initial_proposal_mask",
            "actor_call_index",
            "actor_latency_seconds",
            "prediction_before_latency_seconds",
            "predictor_latency_seconds",
            "observed_state_id",
            "action_history_cells",
            "before_cost",
            "remaining_cost",
            "after_cost",
            "next_cell",
            "prediction_mpa",
        }
        if not required <= set(row):
            raise ValueError("trajectory row is incomplete")
        if row["actor_call_index"] != expected_index:
            raise ValueError("actor call index is not one-per-observation")
        before = float(row["before_cost"])
        after = float(row["after_cost"])
        if abs(before - previous_after) > 1e-12 or after <= before:
            raise ValueError("trajectory cost is not contiguous")
        if type(row["next_cell"]) is not int or not 0 <= int(row["next_cell"]) < 64:
            raise ValueError("trajectory cell is invalid")
        legal = str(row["environment_legal_mask"])
        initial = str(row["initial_proposal_mask"])
        if (
            len(legal) != 64
            or set(legal) > {"0", "1"}
            or len(initial) != 64
            or set(initial) > {"0", "1"}
            or legal[int(row["next_cell"])] != "1"
            or (expected_index == 0 and initial[int(row["next_cell"])] != "1")
        ):
            raise ValueError("trajectory action masks are invalid")
        state = str(row["observed_state_id"])
        actor_latency = float(row["actor_latency_seconds"])
        predictor_latency = float(row["predictor_latency_seconds"])
        prediction_before_latency = float(row["prediction_before_latency_seconds"])
        history = str(row["action_history_cells"])
        expected_history = ";".join(str(rows[index]["next_cell"]) for index in range(expected_index))
        remaining = float(row["remaining_cost"])
        state_budget = remaining + before
        if endpoint_budget is None:
            endpoint_budget = state_budget
        if (
            not state
            or state in observed_states
            or not str(row["initial_proposal_reason"])
            or history != expected_history
            or not math.isfinite(remaining)
            or remaining < 0.0
            or not math.isclose(state_budget, endpoint_budget, abs_tol=1e-8)
            or not math.isfinite(actor_latency)
            or actor_latency < 0.0
            or not math.isfinite(prediction_before_latency)
            or prediction_before_latency < 0.0
            or not math.isfinite(predictor_latency)
            or predictor_latency < 0.0
            or not math.isfinite(float(row["prediction_mpa"]))
        ):
            raise ValueError("trajectory observability fields are invalid")
        observed_states.add(state)
        previous_after = after


__all__ = ["validate_common_predictor_identity", "validate_trajectory_rows"]
