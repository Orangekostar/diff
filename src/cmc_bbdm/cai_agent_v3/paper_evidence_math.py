"""Array-only estimands for frozen VALID; no models, fitting or GPU access."""

from __future__ import annotations

import numpy as np

from .metrics import left_error_area_mpa


def held(costs, predictions, budgets):
    x = np.asarray(costs, dtype=float)
    p = np.asarray(predictions, dtype=float)
    b = np.asarray(budgets, dtype=float)
    if (
        x.ndim != 1
        or len(x) == 0
        or x.shape != p.shape
        or x[0] != 0
        or not np.isfinite(x).all()
        or not np.isfinite(p).all()
        or np.any(np.diff(x) <= 0)
        or x[-1] > 0.25 + 1e-12
        or not np.isfinite(b).all()
        or np.any(b < 0)
        or np.any(b > 0.25)
    ):
        raise ValueError("Invalid states or budget outside observed partial horizon")
    return p[np.searchsorted(x, b, side="right") - 1]


def area(costs, predictions, target, end=0.25):
    held(costs, predictions, [end])
    return left_error_area_mpa(costs, predictions, target, end=end)


def aggregate(residuals, targets):
    """Residual axes: repeat, physical specimen, common budget."""
    e = np.asarray(residuals, dtype=float)
    y = np.asarray(targets, dtype=float)
    abs_loss = np.mean(np.abs(e), axis=0)
    sq_loss = np.mean(e**2, axis=0)
    mse = np.mean(sq_loss, axis=0)
    den = np.sum((y - y.mean()) ** 2)
    return {
        "mae": np.mean(abs_loss, axis=0),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": np.mean(1 - np.sum(e**2, axis=1) / den, axis=0)
        if den > 0
        else np.full(e.shape[-1], np.nan),
        "physical_abs": abs_loss,
        "physical_sq": sq_loss,
    }


def bootstrap_weights(domains, groups, *, replicates=5000, seed=2026091401):
    domains = np.asarray(domains)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    if any(len(set(domains[groups == g])) != 1 for g in set(groups)):
        raise ValueError("Capture group crosses fixed domains")
    weights = np.zeros((replicates, len(groups)), dtype=np.int16)
    for d in sorted(set(domains)):
        names = sorted(set(groups[domains == d]))
        draw = rng.integers(0, len(names), size=(replicates, len(names)))
        for j, g in enumerate(names):
            weights[:, groups == g] = np.sum(draw == j, axis=1)[:, None]
    return weights


def weighted(values, weights, domains=None):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if domains is not None:
        domains = np.asarray(domains)
        return np.mean(
            [
                weighted(values[domains == d], weights[:, domains == d])
                for d in sorted(set(domains))
            ],
            axis=0,
        )
    den = weights.sum(axis=1)
    if np.any(den <= 0):
        raise ValueError("Empty bootstrap sample")
    return (weights @ values) / (den if values.ndim == 1 else den[:, None])


def first_quality(costs, maes, target):
    costs = np.asarray(costs, dtype=float)
    maes = np.asarray(maes, dtype=float)
    if (
        costs.shape != maes.shape
        or not np.isfinite(maes).all()
        or not np.isfinite(target)
        or np.any(np.diff(costs) <= 0)
    ):
        raise ValueError("Invalid common quality curve")
    indices = np.flatnonzero(maes <= target)
    if not len(indices):
        return {
            "cost": None,
            "mae": None,
            "index": None,
            "later_recrosses_target": None,
            "status": "NOT_REACHED_WITHIN_OBSERVED_RANGE",
        }
    i = int(indices[0])
    return {
        "cost": float(costs[i]),
        "mae": float(maes[i]),
        "index": i,
        "later_recrosses_target": bool(np.any(maes[i + 1 :] > target)),
        "status": "REACHED",
    }


def saving(main, control):
    if main is None or control is None:
        return {
            "absolute_saving": None,
            "relative_saving": None,
            "saving_status": "ONE_OR_BOTH_NOT_REACHED",
        }
    return {
        "absolute_saving": control - main,
        "relative_saving": None if control == 0 else 1 - main / control,
        "saving_status": "ZERO_CONTROL_COST" if control == 0 else "DEFINED",
    }


def map_full(index_rows, indices, predictions, targets, expected):
    if not len(indices) == len(predictions) == len(targets):
        raise ValueError("Unaligned full arrays")
    out = []
    for row, i, p, y in zip(range(len(indices)), indices, predictions, targets):
        i = int(i)
        if not 0 <= i < len(index_rows):
            raise ValueError("Invalid source index")
        source = index_rows[i]
        key = source["specimen_key"]
        if (
            source["split"] != "VALID"
            or key not in expected
            or float(y) != expected[key]
        ):
            raise ValueError("Full key/target mismatch")
        if not np.isfinite([p, y]).all():
            raise ValueError("Nonfinite full prediction")
        out.append(
            {
                "specimen_key": key,
                "source_row": row,
                "source_index": i,
                "full_prediction_mpa": float(p),
                "target_mpa": float(y),
            }
        )
    if len(out) != len(expected) or {r["specimen_key"] for r in out} != set(expected):
        raise ValueError("Full keys differ")
    return out
