#!/usr/bin/env python3
"""Independent numeric reference for the v3 prompt, not a research runner.

Only Python standard library. Does not import production code, read research
labels, train a model, access a network, or mutate the user's repository.
"""
from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Sequence


def _validate(costs: Sequence[float], predictions: Sequence[float], target: float) -> None:
    if not costs or len(costs) != len(predictions):
        raise ValueError("nonempty aligned states are required")
    if not all(math.isfinite(float(v)) for v in [*costs, *predictions, target]):
        raise ValueError("all values must be finite")
    if float(costs[0]) != 0.0 or any(float(b) <= float(a) for a, b in zip(costs, costs[1:])):
        raise ValueError("costs must start at zero and be strictly increasing")


def prediction_at(costs: Sequence[float], predictions: Sequence[float], budget: float) -> float:
    _validate(costs, predictions, 0.0)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    return float(predictions[bisect_right(costs, budget) - 1])


def left_error_area(
    costs: Sequence[float], predictions: Sequence[float], target: float,
    start: float = 0.0, end: float = 0.25,
) -> float:
    """Normalized left-constant integral on [start,end], with last-state tail.

States after 'end' may exist but never contribute their predictions early.
"""
    _validate(costs, predictions, target)
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError("invalid integration interval")
    points = [start, *(float(x) for x in costs if start < x < end), end]
    area = 0.0
    for left, right in zip(points, points[1:]):
        pred = prediction_at(costs, predictions, left)
        area += (right - left) * abs(pred - target)
    return area / (end - start)


def trajectory_terms(
    costs: Sequence[float], predictions: Sequence[float], target: float,
    budget: float = 0.25, terminal_weight: float = 0.25,
) -> dict[str, object]:
    _validate(costs, predictions, target)
    if not math.isfinite(budget) or budget <= 0 or costs[-1] > budget + 1e-12:
        raise ValueError("trajectory exceeds valid endpoint")
    if not math.isfinite(terminal_weight) or terminal_weight < 0:
        raise ValueError("invalid terminal weight")
    errors = [abs(float(p) - target) for p in predictions]
    local = [(float(costs[t + 1]) - float(costs[t])) * errors[t] / budget
             for t in range(len(costs) - 1)]
    tail = max(0.0, budget - float(costs[-1])) * errors[-1] / budget
    terminal = tail + terminal_weight * errors[-1]
    running = terminal
    reverse_returns = []
    for value in reversed(local):
        running += value
        reverse_returns.append(running)
    return {
        "errors": errors,
        "local_costs": local,
        "tail_area": tail,
        "terminal_cost": terminal,
        "cost_to_go": list(reversed(reverse_returns)),
        "area": sum(local) + tail,
        "objective": sum(local) + terminal,
    }


def independent_seed_mae(target: float, predictions: Sequence[float]) -> float:
    if not predictions or not all(math.isfinite(float(v)) for v in [target, *predictions]):
        raise ValueError("finite predictions are required")
    return sum(abs(float(v) - target) for v in predictions) / len(predictions)


def evaluate_fixtures(path: Path) -> dict[str, object]:
    fixtures = json.loads(path.read_text(encoding="utf-8"))
    results = []
    def eq(actual: float, expected: float) -> None:
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise AssertionError(f"actual={actual!r}, expected={expected!r}")
    for case in fixtures["trajectory_cases"]:
        terms = trajectory_terms(case["costs"], case["predictions"], case["target"],
                                 case["budget"], case["terminal_weight"])
        eq(float(terms["area"]), case["expected_area"])
        eq(float(terms["objective"]), case["expected_objective"])
        eq(left_error_area(case["costs"], case["predictions"], case["target"],
                           end=case["budget"]), case["expected_area"])
        for actual, expected in zip(terms["cost_to_go"], case["expected_cost_to_go"]):
            eq(float(actual), expected)
        if len(terms["cost_to_go"]) != len(case["expected_cost_to_go"]):
            raise AssertionError("cost-to-go lengths differ")
        results.append({"id": case["id"], "reference_check": "PASS"})
    for case in fixtures["prefix_cases"]:
        eq(prediction_at(case["costs"], case["predictions"], case["budget"]),
           case["expected_prediction"])
        results.append({"id": case["id"], "reference_check": "PASS"})
    for case in fixtures["seed_cases"]:
        independent = independent_seed_mae(case["target"], case["predictions"])
        ensemble = abs(sum(case["predictions"]) / len(case["predictions"]) - case["target"])
        eq(independent, case["expected_independent_mae"])
        eq(ensemble, case["expected_ensemble_mae"])
        results.append({"id": case["id"], "reference_check": "PASS"})
    # Expected-policy-loss gradient: grad(z_j) = pi_j (cost_j-E[cost]).
    toy = fixtures["policy_gradient_case"]
    p = toy["probabilities"]
    c = toy["costs"]
    expected_cost = sum(a * b for a, b in zip(p, c))
    gradient = [a * (b - expected_cost) for a, b in zip(p, c)]
    for actual, expected in zip(gradient, toy["expected_logit_gradient"]):
        eq(actual, expected)
    logits = [-toy["learning_rate"] * g for g in gradient]
    exponentials = [math.exp(v - max(logits)) for v in logits]
    next_p = [v / sum(exponentials) for v in exponentials]
    if not next_p[0] > p[0]:
        raise AssertionError("gradient descent did not favor the low-cost action")
    results.append({"id": toy["id"], "reference_check": "PASS"})
    return {
        "status": "INDEPENDENT_REFERENCE_SELF_CHECK_PASS",
        "fixture_checks": len(results),
        "checks": results,
        "scope": "Checks only this supplied numeric oracle; NOT the repository implementation or any model performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).with_name("golden_cases.json"))
    parser.add_argument("--self-test", action="store_true", help="Check numeric fixtures (no training).")
    args = parser.parse_args()
    if not args.self_test:
        parser.print_help()
        return
    print(json.dumps(evaluate_fixtures(args.fixtures), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
