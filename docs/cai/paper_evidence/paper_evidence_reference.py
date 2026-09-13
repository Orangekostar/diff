#!/usr/bin/env python3
"""Independent numeric examples for a frozen-result paper-evidence task.

No repository imports, models, network, research data, or GPU use.
This is an acceptance reference, not the study implementation or new evidence.
"""
from __future__ import annotations

import argparse
import bisect
import math
from collections.abc import Mapping, Sequence


def _states(costs: Sequence[float], values: Sequence[float]) -> None:
    if not costs or len(costs) != len(values) or costs[0] != 0:
        raise ValueError("Aligned nonempty states must begin at zero")
    if any(not math.isfinite(float(x)) for x in (*costs, *values)):
        raise ValueError("Finite values required")
    if any(b <= a for a, b in zip(costs[:-1], costs[1:])):
        raise ValueError("Strictly increasing costs required")


def held_value(
    costs: Sequence[float], values: Sequence[float], budget: float,
    *, observed_limit: float = 0.25,
) -> float:
    _states(costs, values)
    if not math.isfinite(budget) or not 0 <= budget <= observed_limit:
        raise ValueError("Budget outside the observed evaluation horizon")
    if costs[-1] > observed_limit:
        raise ValueError("State outside the declared horizon")
    return float(values[bisect.bisect_right(costs, budget) - 1])


def left_area(
    costs: Sequence[float], predictions: Sequence[float], target: float,
    *, end: float = 0.25,
) -> float:
    _states(costs, predictions)
    if not 0 < end <= 0.25 or not math.isfinite(target):
        raise ValueError("Invalid integral range or target")
    points = [0.0, *[float(c) for c in costs if 0 < c < end], end]
    return sum(
        (b - a) * abs(held_value(costs, predictions, a) - target)
        for a, b in zip(points[:-1], points[1:])
    ) / end


def repeated_errors(residuals_by_specimen: Mapping[str, Sequence[float]]) -> dict:
    if not residuals_by_specimen or any(not r for r in residuals_by_specimen.values()):
        raise ValueError("Each physical specimen needs observations")
    mae = sum(sum(abs(x) for x in r) / len(r)
              for r in residuals_by_specimen.values()) / len(residuals_by_specimen)
    mse = sum(sum(x * x for x in r) / len(r)
              for r in residuals_by_specimen.values()) / len(residuals_by_specimen)
    return {"mae": mae, "mse": mse, "rmse": math.sqrt(mse)}


def first_quality_cost(
    costs: Sequence[float], maes: Sequence[float], target_mae: float,
) -> dict:
    if len(costs) != len(maes) or not costs:
        raise ValueError("Aligned observed grid required")
    if any(b <= a for a, b in zip(costs[:-1], costs[1:])):
        raise ValueError("Ascending observed grid required")
    if any(not math.isfinite(x) for x in (*costs, *maes, target_mae)):
        raise ValueError("Finite input required")
    hit = next((i for i, m in enumerate(maes) if m <= target_mae), None)
    if hit is None:
        return {"status": "NOT_REACHED_WITHIN_OBSERVED_RANGE", "cost": None,
                "mae_at_cost": None, "later_recrosses_target": None}
    return {"status": "REACHED_ON_OBSERVED_GRID", "cost": float(costs[hit]),
            "mae_at_cost": float(maes[hit]),
            "later_recrosses_target": any(m > target_mae for m in maes[hit + 1:])}


def relative_saving(main_cost: float | None, control_cost: float | None) -> float | None:
    if main_cost is None or control_cost is None or control_cost <= 0:
        return None
    return 1.0 - main_cost / control_cost


def grouped_weighted_mean(
    specimen_values: Mapping[str, float], specimen_group: Mapping[str, str],
    group_multiplicity: Mapping[str, int],
) -> float:
    weights = {key: group_multiplicity.get(specimen_group[key], 0)
               for key in specimen_values}
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError("Empty bootstrap sample")
    return sum(weights[k] * v for k, v in specimen_values.items()) / denominator


def map_full_predictions(
    original_key_order: Sequence[str], full_indices: Sequence[int],
    full_predictions: Sequence[float], expected_keys: set[str],
) -> dict[str, float]:
    if len(full_indices) != len(full_predictions):
        raise ValueError("Full-input arrays not aligned")
    keys = [original_key_order[i] for i in full_indices]
    if len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise ValueError("Full-input keys differ from frozen evaluation keys")
    return dict(zip(keys, map(float, full_predictions)))


def self_test() -> list[str]:
    passed: list[str] = []
    costs, preds = [0.0, 0.125, 0.25], [190.0, 194.0, 198.0]
    assert left_area(costs, preds, 200.0) == 8.0
    assert held_value(costs, preds, 0.0625) == 190.0
    assert held_value(costs, preds, 0.125) == 194.0
    passed.append("left step and no future prediction")

    try:
        held_value(costs, preds, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Unobserved 0.5 budget must be rejected")
    assert held_value([0, 0.125], [190, 198], 0.25) == 198
    assert left_area([0, 0.125], [190, 198], 200) == 6
    passed.append("valid tail and no 0.25-to-1 extrapolation")

    assert repeated_errors({"sample": [-10, 10]}) == {"mae": 10.0, "mse": 100.0, "rmse": 10.0}
    assert repeated_errors({"a": [-3, 3], "b": [4]})["mae"] == 3.5
    passed.append("repeat losses, not ensemble predictions")

    weighted = grouped_weighted_mean(
        {"a": 0, "b": 10, "c": 100},
        {"a": "g1", "b": "g1", "c": "g2"}, {"g1": 2, "g2": 1})
    assert weighted == 24.0  # (0*2 + 10*2 + 100) / 5; groups are not single specimens.
    passed.append("cluster multiplicity preserves member weights")

    grid = [0, 0.0625, 0.125, 0.1875, 0.25]
    main = [58.55099456787109, 45.51743743896484, 44.11356689453125,
            44.68454376220703, 44.285798950195314]
    geom = [58.55099456787109, 50.22089508056641, 46.188976745605466,
            46.55507232666016, 46.9099169921875]
    static = [58.55099456787109, 46.46498809814453, 48.06415130615235,
              46.784012145996094, 46.99311798095703]
    q = 46.9099169921875
    a = first_quality_cost(grid, main, q)
    b = first_quality_cost(grid, geom, q)
    s = first_quality_cost(grid, static, q)
    assert (a["cost"], b["cost"], s["cost"]) == (0.0625, 0.125, 0.0625)
    assert relative_saving(a["cost"], b["cost"]) == 0.5
    assert relative_saving(a["cost"], s["cost"]) == 0.0
    passed.append("both methods use their earliest empirical quality point")

    original = [10.0, 4.0, 6.0, 3.0]
    result = first_quality_cost([0, 0.0625, 0.125, 0.25], original, 5)
    assert result["cost"] == 0.0625 and result["later_recrosses_target"] is True
    assert original == [10.0, 4.0, 6.0, 3.0]
    assert relative_saving(0.0, 0.0) is None
    assert relative_saving(None, 0.25) is None
    passed.append("nonmonotonic curves and undefined savings retained")

    full_mae = 41.69001007080078
    assert first_quality_cost([1.0], [full_mae], 42.0)["cost"] == 1.0
    assert first_quality_cost([1.0], [full_mae], 40.0)["cost"] is None
    assert first_quality_cost(grid, main, full_mae)["cost"] is None
    passed.append("full scan is one point; unattained quality remains null")

    # Two specimens with errors [1, 9] and [9, 1]. Their separate best is 1,
    # but the cohort mean is 5 at each shared budget. Do not use per-item oracle STOP.
    cohort_maes = [(1 + 9) / 2, (9 + 1) / 2]
    assert first_quality_cost([0.0625, 0.125], cohort_maes, 2)["cost"] is None
    passed.append("aggregate at a common budget before inverting quality")

    mapped = map_full_predictions(["train", "vB", "vA"], [2, 1], [30, 40], {"vA", "vB"})
    assert mapped == {"vA": 30.0, "vB": 40.0}
    passed.append("full indices map through the original index order")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test; this reference does not run the research workflow")
    result = self_test()
    for index, description in enumerate(result, 1):
        print(f"PASS {index}: {description}")
    print(f"{len(result)} independent numeric examples passed; no models or research runs executed.")


if __name__ == "__main__":
    main()
