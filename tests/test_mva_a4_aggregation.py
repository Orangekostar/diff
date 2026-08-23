from __future__ import annotations

import hashlib

import polars as pl
import pytest

from cmc_bbdm.mva.a4_evaluation import aggregate_a4_tables

DOMAINS = tuple(f"d{index}" for index in range(6))
CHECKPOINTS = (0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25)
GLOBAL_METHODS = (
    "global_appearance_mask",
    "global_reconstruction_mask",
    "global_mechanical_mask",
)
RANDOM_SEEDS = (11, 13, 17)


def _head(domain: str, checkpoint: float) -> str:
    return hashlib.sha256(f"{domain}:{checkpoint}".encode("ascii")).hexdigest()


def _error(domain_index: int, checkpoint: float, method: str) -> float:
    uniform = 0.22 + 0.004 * domain_index - 0.30 * checkpoint
    offsets = {
        "uniform": 0.0,
        "global_appearance_mask": -0.005,
        "global_reconstruction_mask": -0.01,
        "global_mechanical_mask": -0.03,
        "mechanical_oracle": -0.04,
    }
    return uniform + offsets[method]


def _tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    a4_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    counts = (1, 2, 3, 1, 2, 3)
    for domain_index, (domain, count) in enumerate(zip(DOMAINS, counts, strict=True)):
        for specimen_index in range(count):
            specimen_id = f"{domain}-{specimen_index}"
            for checkpoint in CHECKPOINTS:
                head = _head(domain, checkpoint)
                for method in GLOBAL_METHODS:
                    error = _error(domain_index, checkpoint, method)
                    a4_rows.append(
                        {
                            "specimen_id": specimen_id,
                            "dataset_id": domain,
                            "method": method,
                            "nominal_checkpoint": checkpoint,
                            "effective_budget": checkpoint - 0.0001,
                            "p_a_absolute_error": error + 0.01,
                            "p_b_absolute_error": error,
                            "normalized_rgb_mse": 0.3 - checkpoint,
                            "ssim": 0.6 + checkpoint,
                            "p_b_predictor_state_sha256": head,
                        }
                    )
                for method in ("uniform", "mechanical_oracle"):
                    error = _error(domain_index, checkpoint, method)
                    reference_rows.append(
                        {
                            "specimen_id": specimen_id,
                            "dataset_id": domain,
                            "method": method,
                            "seed": None,
                            "nominal_checkpoint": checkpoint,
                            "effective_budget": checkpoint - 0.0001,
                            "p_a_absolute_error": error + 0.01,
                            "p_b_absolute_error": error,
                            "normalized_rgb_mse": 0.3 - checkpoint,
                            "ssim": 0.6 + checkpoint,
                            "p_b_predictor_state_sha256": head,
                        }
                    )
                for seed_index, seed in enumerate(RANDOM_SEEDS):
                    error = _error(domain_index, checkpoint, "uniform") + 0.002 * (
                        seed_index - 1
                    )
                    reference_rows.append(
                        {
                            "specimen_id": specimen_id,
                            "dataset_id": domain,
                            "method": "random",
                            "seed": seed,
                            "nominal_checkpoint": checkpoint,
                            "effective_budget": checkpoint - 0.0001,
                            "p_a_absolute_error": error + 0.01,
                            "p_b_absolute_error": error,
                            "normalized_rgb_mse": None,
                            "ssim": None,
                            "p_b_predictor_state_sha256": head,
                        }
                    )
    return pl.DataFrame(a4_rows), pl.DataFrame(reference_rows, infer_schema_length=None)


def test_a4_aggregation_uses_equal_domain_curves_and_synchronized_effects() -> None:
    a4, reference = _tables()
    result = aggregate_a4_tables(
        a4,
        reference,
        domain_order=DOMAINS,
        checkpoints=CHECKPOINTS,
        random_seeds=RANDOM_SEEDS,
        full_mae=0.08963580465761432,
        bootstrap_seed=20260823,
        bootstrap_resamples=2000,
    )

    assert result.gate.global_mask_status == "MVA_A4_GLOBAL_GO"
    assert result.gate.a5_status == "MVA_A5_AUTHORIZED"
    assert len(result.domain_metrics) == 6 * 6
    assert len(result.budget_metrics) == 6
    assert len(result.bootstrap_effects) == 4
    assert len({effect.indices_sha256 for effect in result.bootstrap_effects}) == 1
    mechanical = next(
        row
        for row in result.curves
        if row["method"] == "global_mechanical_mask"
        and row["protocol"] == "P-B"
        and row["nominal_checkpoint"] == 0.125
    )
    expected = sum(
        _error(index, 0.125, "global_mechanical_mask") for index in range(6)
    ) / 6.0
    assert mechanical["equal_domain_mae"] == pytest.approx(expected)


def test_a4_aggregation_rejects_missing_rows_and_pb_head_drift() -> None:
    a4, reference = _tables()
    with pytest.raises(ValueError, match="roster"):
        aggregate_a4_tables(
            a4.slice(1),
            reference,
            domain_order=DOMAINS,
            checkpoints=CHECKPOINTS,
            random_seeds=RANDOM_SEEDS,
            full_mae=0.08963580465761432,
            bootstrap_seed=20260823,
            bootstrap_resamples=100,
        )

    tampered = a4.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("f" * 64))
        .otherwise(pl.col("p_b_predictor_state_sha256"))
        .alias("p_b_predictor_state_sha256")
    )
    with pytest.raises(ValueError, match="P-B"):
        aggregate_a4_tables(
            tampered,
            reference,
            domain_order=DOMAINS,
            checkpoints=CHECKPOINTS,
            random_seeds=RANDOM_SEEDS,
            full_mae=0.08963580465761432,
            bootstrap_seed=20260823,
            bootstrap_resamples=100,
        )


def test_a4_aggregation_digest_is_independent_of_input_row_order() -> None:
    a4, reference = _tables()
    arguments = {
        "domain_order": DOMAINS,
        "checkpoints": CHECKPOINTS,
        "random_seeds": RANDOM_SEEDS,
        "full_mae": 0.08963580465761432,
        "bootstrap_seed": 20260823,
        "bootstrap_resamples": 200,
    }

    first = aggregate_a4_tables(a4, reference, **arguments)
    second = aggregate_a4_tables(a4.reverse(), reference.reverse(), **arguments)

    assert first.state_sha256 == second.state_sha256
    assert first.curves == second.curves
