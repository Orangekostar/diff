from __future__ import annotations

import hashlib

import polars as pl
import pytest

from cmc_bbdm.mvd.statistics import aggregate_m0_tables

DOMAINS = tuple(f"d{index}" for index in range(6))
CHECKPOINTS = (0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25)
SEEDS = tuple(range(100, 200))


def _head(domain: str, checkpoint: float) -> str:
    return hashlib.sha256(f"{domain}:{checkpoint}".encode("ascii")).hexdigest()


def _curve(method: str, domain_index: int, checkpoint: float) -> float:
    baseline = 0.22 + 0.003 * domain_index - 0.28 * checkpoint
    offset = {
        "uniform": 0.0,
        "one_shot_reconstruction": -0.010,
        "global_mechanical_mask": -0.012,
        "one_shot_mechanical_oracle": -0.035,
        "sequential_mechanical_oracle": -0.050,
    }[method]
    return baseline + offset


def _tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    one_shot: list[dict[str, object]] = []
    a2: list[dict[str, object]] = []
    a4: list[dict[str, object]] = []
    for domain_index, domain in enumerate(DOMAINS):
        for specimen_index in range(domain_index + 1):
            specimen_id = f"{domain}-{specimen_index}"
            for checkpoint in CHECKPOINTS:
                head = _head(domain, checkpoint)
                common = {
                    "specimen_id": specimen_id,
                    "dataset_id": domain,
                    "nominal_checkpoint": checkpoint,
                    "effective_budget": checkpoint - 0.0001,
                    "p_b_predictor_state_sha256": head,
                }
                for method in (
                    "one_shot_reconstruction",
                    "one_shot_mechanical_oracle",
                ):
                    error = _curve(method, domain_index, checkpoint)
                    one_shot.append(
                        {
                            **common,
                            "method": method,
                            "p_a_absolute_error": error + 0.01,
                            "p_b_absolute_error": error,
                        }
                    )
                for method in ("uniform", "sequential_mechanical_oracle"):
                    error = _curve(method, domain_index, checkpoint)
                    a2.append(
                        {
                            **common,
                            "method": method,
                            "seed": None,
                            "p_a_absolute_error": error + 0.01,
                            "p_b_absolute_error": error,
                        }
                    )
                for seed_index, seed in enumerate(SEEDS):
                    error = _curve("uniform", domain_index, checkpoint)
                    error += 0.00002 * (seed_index - 49.5)
                    a2.append(
                        {
                            **common,
                            "method": "random",
                            "seed": seed,
                            "p_a_absolute_error": error + 0.01,
                            "p_b_absolute_error": error,
                        }
                    )
                error = _curve("global_mechanical_mask", domain_index, checkpoint)
                a4.append(
                    {
                        **common,
                        "method": "global_mechanical_mask",
                        "p_a_absolute_error": error + 0.01,
                        "p_b_absolute_error": error,
                    }
                )
    return (
        pl.DataFrame(one_shot),
        pl.DataFrame(a2, infer_schema_length=None),
        pl.DataFrame(a4),
    )


def test_m0_aggregation_uses_equal_domain_curves_and_frozen_gate() -> None:
    one_shot, a2, a4 = _tables()
    result = aggregate_m0_tables(
        one_shot,
        a2,
        a4,
        domain_order=DOMAINS,
        checkpoints=CHECKPOINTS,
        random_seeds=SEEDS,
        full_mae=0.08963580465761432,
        bootstrap_seed=20260824,
        bootstrap_resamples=2000,
        minimum_improved_domains=4,
        minimum_headroom_retention=0.20,
        strong_headroom_retention=0.50,
    )

    assert result.gate.status == "MVD_ONE_SHOT_STRONG_GO"
    assert result.gate.go
    assert result.gate.uniform_pass
    assert result.gate.reconstruction_pass
    assert result.gate.headroom_pass
    assert result.gate.headroom_retention == pytest.approx(0.625)
    assert len(result.domain_metrics) == 6 * 7
    assert len(result.budget_metrics) == 7
    assert len(result.curves) == 2 * 7 * 6
    assert len(result.bootstrap_effects) == 2
    assert len({item.indices_sha256 for item in result.bootstrap_effects}) == 1
    oracle = next(
        row
        for row in result.curves
        if row["method"] == "one_shot_mechanical_oracle"
        and row["protocol"] == "P-B"
        and row["nominal_checkpoint"] == 0.125
    )
    expected = sum(
        _curve("one_shot_mechanical_oracle", index, 0.125)
        for index in range(6)
    ) / 6.0
    assert oracle["equal_domain_mae"] == pytest.approx(expected)


def test_m0_aggregation_rejects_roster_and_checkpoint_head_drift() -> None:
    one_shot, a2, a4 = _tables()
    arguments = {
        "domain_order": DOMAINS,
        "checkpoints": CHECKPOINTS,
        "random_seeds": SEEDS,
        "full_mae": 0.08963580465761432,
        "bootstrap_seed": 20260824,
        "bootstrap_resamples": 100,
        "minimum_improved_domains": 4,
        "minimum_headroom_retention": 0.20,
        "strong_headroom_retention": 0.50,
    }
    with pytest.raises(ValueError, match="roster"):
        aggregate_m0_tables(one_shot.slice(1), a2, a4, **arguments)

    tampered = one_shot.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("f" * 64))
        .otherwise(pl.col("p_b_predictor_state_sha256"))
        .alias("p_b_predictor_state_sha256")
    )
    with pytest.raises(ValueError, match="P-B"):
        aggregate_m0_tables(tampered, a2, a4, **arguments)


def test_m0_aggregation_is_independent_of_input_row_order() -> None:
    one_shot, a2, a4 = _tables()
    arguments = {
        "domain_order": DOMAINS,
        "checkpoints": CHECKPOINTS,
        "random_seeds": SEEDS,
        "full_mae": 0.08963580465761432,
        "bootstrap_seed": 20260824,
        "bootstrap_resamples": 200,
        "minimum_improved_domains": 4,
        "minimum_headroom_retention": 0.20,
        "strong_headroom_retention": 0.50,
    }
    first = aggregate_m0_tables(one_shot, a2, a4, **arguments)
    second = aggregate_m0_tables(
        one_shot.reverse(), a2.reverse(), a4.reverse(), **arguments
    )
    assert first.state_sha256 == second.state_sha256
    assert first.curves == second.curves
