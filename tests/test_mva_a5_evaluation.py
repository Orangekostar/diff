from __future__ import annotations

import polars as pl
import pytest

from cmc_bbdm.mva.a5_evaluation import (
    DEPLOYABLE_METHODS,
    A5GateInputs,
    aggregate_a5_tables,
    evaluate_a5_gate,
)
from cmc_bbdm.mva.statistics import BootstrapEffect


def _effect(
    effect_id: str,
    point: float,
    lower: float,
    improved: int,
) -> BootstrapEffect:
    domain_effects = tuple(
        point + 0.0001 * (index - 2.5) for index in range(6)
    )
    return BootstrapEffect(
        effect_id=effect_id,
        point_estimate=sum(domain_effects) / 6,
        lower=lower,
        upper=point + 0.001,
        improved_domains=improved,
        domain_effects=domain_effects,
        indices_sha256="a" * 64,
    )


def test_a5_gate_requires_both_baselines_and_twenty_percent_gap_closure() -> None:
    result = evaluate_a5_gate(
        A5GateInputs(
            global_effect=_effect("global_minus_policy_auebc", 0.003, 0.001, 6),
            uniform_effect=_effect("uniform_minus_policy_auebc", 0.002, 0.0005, 6),
            policy_oracle_effect=_effect(
                "policy_minus_oracle_auebc", 0.007, 0.004, 6
            ),
            global_auebc=0.020,
            policy_auebc=0.017,
            oracle_auebc=0.010,
        )
    )

    assert result.a5_status == "MVA_A5_POLICY_GO"
    assert result.a6_status == "MVA_A6_AUTHORIZED"
    assert result.global_pass
    assert result.uniform_pass
    assert result.gap_closure == pytest.approx(0.3)


def test_a5_gate_fails_closed_when_gap_closure_is_insufficient() -> None:
    result = evaluate_a5_gate(
        A5GateInputs(
            global_effect=_effect("global_minus_policy_auebc", 0.001, 0.0001, 6),
            uniform_effect=_effect("uniform_minus_policy_auebc", 0.001, 0.0001, 6),
            policy_oracle_effect=_effect(
                "policy_minus_oracle_auebc", 0.009, 0.004, 6
            ),
            global_auebc=0.020,
            policy_auebc=0.019,
            oracle_auebc=0.010,
        )
    )

    assert result.gap_closure == pytest.approx(0.1)
    assert result.a5_status == "MVA_A5_POLICY_NO_GO"
    assert result.a6_status == "MVA_A6_NOT_AUTHORIZED"


def _aggregation_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    domains = tuple(f"d{index}" for index in range(6))
    checkpoints = (0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25)
    rows: dict[str, list[dict[str, object]]] = {"a5": [], "a4": [], "a2": []}
    errors = {
        "center_first": 0.12,
        "observed_gradient": 0.11,
        "observed_uncertainty": 0.105,
        "imitation_policy": 0.09,
        "global_mechanical_mask": 0.10,
        "uniform": 0.11,
        "mechanical_oracle": 0.07,
    }
    for domain_index, domain in enumerate(domains):
        specimen = f"s{domain_index}"
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            predictor = f"predictor-{domain}-{checkpoint}"
            common = {
                "dataset_id": domain,
                "effective_budget": checkpoint,
                "nominal_checkpoint": checkpoint,
                "normalized_rgb_mse": 0.2 - checkpoint,
                "p_b_predictor_state_sha256": predictor,
                "specimen_id": specimen,
                "ssim": 0.8 + checkpoint,
            }
            for method in DEPLOYABLE_METHODS:
                error = errors[method] - 0.01 * checkpoint_index
                rows["a5"].append(
                    {**common, "method": method, "p_a_absolute_error": error, "p_b_absolute_error": error}
                )
            error = errors["global_mechanical_mask"] - 0.01 * checkpoint_index
            rows["a4"].append(
                {**common, "method": "global_mechanical_mask", "p_a_absolute_error": error, "p_b_absolute_error": error}
            )
            for method in ("uniform", "mechanical_oracle"):
                error = errors[method] - 0.01 * checkpoint_index
                rows["a2"].append(
                    {**common, "method": method, "seed": None, "p_a_absolute_error": error, "p_b_absolute_error": error}
                )
            for seed in (0, 1):
                error = 0.115 + 0.001 * seed - 0.01 * checkpoint_index
                rows["a2"].append(
                    {**common, "method": "random", "seed": seed, "p_a_absolute_error": error, "p_b_absolute_error": error}
                )
    return tuple(pl.DataFrame(rows[name]) for name in ("a5", "a4", "a2"))


def test_a5_aggregation_hash_and_specimen_rows_are_order_invariant() -> None:
    a5, a4, a2 = _aggregation_tables()
    arguments = {
        "domain_order": tuple(f"d{index}" for index in range(6)),
        "checkpoints": (0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25),
        "random_seeds": (0, 1),
        "full_mae": 0.05,
        "bootstrap_seed": 17,
        "bootstrap_resamples": 100,
    }

    first = aggregate_a5_tables(a5, a4, a2, **arguments)
    second = aggregate_a5_tables(a5.reverse(), a4.reverse(), a2.reverse(), **arguments)
    keys = [(row["specimen_id"], row["method"]) for row in first.specimen_metrics]

    assert keys == sorted(keys)
    assert first.specimen_metrics == second.specimen_metrics
    assert first.state_sha256 == second.state_sha256
