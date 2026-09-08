"""Episode-first STOP calibration for ordered visible trajectories."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from statistics import NormalDist

import numpy as np

CANDIDATE_THRESHOLDS = (0.90, 0.95, 0.99)


@dataclass(frozen=True, slots=True)
class EpisodeStopOutcome:
    stopped: bool
    completed: bool
    false_stop: bool
    exhausted: bool
    stop_step: int | None
    stop_cost: float | None
    autonomous_ausc: float
    failure_penalized_cost: float


@dataclass(frozen=True, slots=True)
class EpisodeRisk:
    episode_count: int
    stop_count: int
    completion_count: int
    false_stop_count: int
    exhaustion_count: int
    completion_rate: float
    false_stop_episode_rate: float
    wrong_among_stops: float | None
    wrong_ci_lower: float | None
    wrong_ci_upper: float | None
    exhaustion_rate: float
    autonomous_ausc: float
    failure_penalized_cost: float


@dataclass(frozen=True, slots=True)
class PlannerThresholdResult:
    planner: str
    threshold: float
    risk: EpisodeRisk
    rule_completion_rate: float
    qualified: bool
    failed_requirements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ThresholdCandidateResult:
    threshold: float
    planner_results: tuple[PlannerThresholdResult, ...]
    equal_planner_failure_cost: float
    qualified: bool


@dataclass(frozen=True, slots=True)
class EpisodeStopCalibration:
    task: str
    candidates: tuple[ThresholdCandidateResult, ...]
    qualified: bool
    selected_threshold: float | None
    status: str


def _number(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"trajectory {field} is invalid")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"trajectory {field} is invalid")
    return converted


def first_stop_outcome(
    rows: tuple[Mapping[str, object], ...],
    *,
    probability_field: str,
    eligibility_field: str,
    threshold: float,
) -> EpisodeStopOutcome:
    """Apply one threshold to the first eligible crossing in row order."""

    value = float(threshold)
    if (
        type(rows) is not tuple
        or not rows
        or any(not isinstance(row, Mapping) for row in rows)
        or not probability_field
        or not eligibility_field
        or isinstance(threshold, bool)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError("episode first-stop request is invalid")
    steps: list[int] = []
    costs: list[float] = []
    checked: list[tuple[Mapping[str, object], int, float, float]] = []
    for row in rows:
        step_raw = row.get("step")
        success = row.get("success")
        eligible = row.get(eligibility_field)
        if (
            type(step_raw) is not int
            or step_raw < 0
            or type(success) is not bool
            or type(eligible) is not bool
        ):
            raise ValueError("ordered episode trajectory is invalid")
        cost = _number(row, "cost")
        probability = _number(row, probability_field)
        if not 0.0 <= cost <= 1.0 or not 0.0 <= probability <= 1.0:
            raise ValueError("ordered episode trajectory is invalid")
        steps.append(step_raw)
        costs.append(cost)
        checked.append((row, step_raw, cost, probability))
    if (
        len(set(steps)) != len(steps)
        or any(right <= left for left, right in pairwise(steps))
        or any(right < left - 1e-15 for left, right in pairwise(costs))
    ):
        raise ValueError("episode trajectory rows are not ordered")
    for row, step, cost, probability in checked:
        if bool(row[eligibility_field]) and probability >= value:
            completed = bool(row["success"])
            return EpisodeStopOutcome(
                stopped=True,
                completed=completed,
                false_stop=not completed,
                exhausted=False,
                stop_step=step,
                stop_cost=cost,
                autonomous_ausc=(1.0 - cost if completed else 0.0),
                failure_penalized_cost=(cost if completed else 1.0),
            )
    return EpisodeStopOutcome(
        stopped=False,
        completed=False,
        false_stop=False,
        exhausted=True,
        stop_step=None,
        stop_cost=None,
        autonomous_ausc=0.0,
        failure_penalized_cost=1.0,
    )


def _wilson_interval(
    successes: int, total: int, *, confidence_level: float
) -> tuple[float, float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Wilson interval counts are invalid")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    fraction = successes / total
    denominator = 1.0 + z * z / total
    center = (fraction + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            fraction * (1.0 - fraction) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_episode_risk(
    outcomes: tuple[EpisodeStopOutcome, ...], *, confidence_level: float = 0.95
) -> EpisodeRisk:
    """Summarize risk with physical episodes, not trajectory states, as N."""

    confidence = float(confidence_level)
    if (
        type(outcomes) is not tuple
        or not outcomes
        or any(type(row) is not EpisodeStopOutcome for row in outcomes)
        or isinstance(confidence_level, bool)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("episode risk request is invalid")
    episodes = len(outcomes)
    stop_count = sum(row.stopped for row in outcomes)
    completed = sum(row.completed for row in outcomes)
    false_stops = sum(row.false_stop for row in outcomes)
    exhausted = sum(row.exhausted for row in outcomes)
    if stop_count != completed + false_stops or episodes != stop_count + exhausted:
        raise RuntimeError("episode STOP outcomes are inconsistent")
    wrong = false_stops / stop_count if stop_count else None
    interval = (
        _wilson_interval(false_stops, stop_count, confidence_level=confidence)
        if stop_count
        else (None, None)
    )
    return EpisodeRisk(
        episode_count=episodes,
        stop_count=stop_count,
        completion_count=completed,
        false_stop_count=false_stops,
        exhaustion_count=exhausted,
        completion_rate=completed / episodes,
        false_stop_episode_rate=false_stops / episodes,
        wrong_among_stops=wrong,
        wrong_ci_lower=interval[0],
        wrong_ci_upper=interval[1],
        exhaustion_rate=exhausted / episodes,
        autonomous_ausc=float(
            np.mean([row.autonomous_ausc for row in outcomes])
        ),
        failure_penalized_cost=float(
            np.mean([row.failure_penalized_cost for row in outcomes])
        ),
    )


def calibrate_episode_stop(
    rows: tuple[Mapping[str, object], ...],
    *,
    task: str,
    planners: tuple[str, ...],
    rule_completion: Mapping[str, float],
    probability_field: str = "stop_probability",
    eligibility_field: str = "mechanically_eligible",
) -> EpisodeStopCalibration:
    """Select one shared task threshold using the preregistered three values."""

    if (
        type(rows) is not tuple
        or not rows
        or task not in {"LOCATE", "CHARACTERIZE"}
        or type(planners) is not tuple
        or len(planners) < 2
        or any(type(planner) is not str or not planner for planner in planners)
        or len(set(planners)) != len(planners)
        or set(rule_completion) != set(planners)
    ):
        raise ValueError("episode STOP calibration request is invalid")
    rules = {planner: float(rule_completion[planner]) for planner in planners}
    if any(not 0.0 <= value <= 1.0 for value in rules.values()):
        raise ValueError("rule completion rates are invalid")
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("episode STOP trajectory rows are invalid")
        specimen = row.get("specimen_key")
        row_task = row.get("task")
        planner = row.get("planner")
        if (
            type(specimen) is not str
            or not specimen
            or row_task != task
            or planner not in planners
        ):
            raise ValueError("episode STOP trajectory rows are invalid")
        grouped.setdefault((str(planner), specimen), []).append(row)
    specimens_by_planner = {
        planner: {specimen for candidate, specimen in grouped if candidate == planner}
        for planner in planners
    }
    first_specimens = specimens_by_planner[planners[0]]
    if not first_specimens or any(
        specimens != first_specimens
        for specimens in specimens_by_planner.values()
    ):
        raise ValueError("calibration planner episodes are not paired")

    candidates: list[ThresholdCandidateResult] = []
    for threshold in CANDIDATE_THRESHOLDS:
        planner_results: list[PlannerThresholdResult] = []
        for planner in planners:
            outcomes = tuple(
                first_stop_outcome(
                    tuple(grouped[(planner, specimen)]),
                    probability_field=probability_field,
                    eligibility_field=eligibility_field,
                    threshold=threshold,
                )
                for specimen in sorted(first_specimens)
            )
            risk = summarize_episode_risk(outcomes)
            failed = []
            if (
                risk.wrong_among_stops is None
                or risk.wrong_among_stops > 0.05 + 1e-15
            ):
                failed.append("WRONG_AMONG_STOPS")
            if risk.completion_rate < 0.50 - 1e-15:
                failed.append("MINIMUM_COMPLETION")
            if risk.completion_rate < rules[planner] - 0.05 - 1e-15:
                failed.append("RULE_COMPLETION_MARGIN")
            if risk.stop_count < 6:
                failed.append("MINIMUM_STOP_COUNT")
            planner_results.append(
                PlannerThresholdResult(
                    planner=planner,
                    threshold=threshold,
                    risk=risk,
                    rule_completion_rate=rules[planner],
                    qualified=not failed,
                    failed_requirements=tuple(failed),
                )
            )
        candidates.append(
            ThresholdCandidateResult(
                threshold=threshold,
                planner_results=tuple(planner_results),
                equal_planner_failure_cost=float(
                    np.mean(
                        [
                            result.risk.failure_penalized_cost
                            for result in planner_results
                        ]
                    )
                ),
                qualified=all(result.qualified for result in planner_results),
            )
        )
    qualified = [candidate for candidate in candidates if candidate.qualified]
    selected = (
        min(
            qualified,
            key=lambda candidate: (
                round(candidate.equal_planner_failure_cost, 15),
                -candidate.threshold,
            ),
        )
        if qualified
        else None
    )
    return EpisodeStopCalibration(
        task=task,
        candidates=tuple(candidates),
        qualified=selected is not None,
        selected_threshold=(None if selected is None else selected.threshold),
        status=("S_EP_QUALIFIED" if selected is not None else "S_EP_NOT_QUALIFIED"),
    )


__all__ = [
    "CANDIDATE_THRESHOLDS",
    "EpisodeRisk",
    "EpisodeStopCalibration",
    "EpisodeStopOutcome",
    "PlannerThresholdResult",
    "ThresholdCandidateResult",
    "calibrate_episode_stop",
    "first_stop_outcome",
    "summarize_episode_risk",
]
