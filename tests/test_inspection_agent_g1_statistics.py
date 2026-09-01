from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.metrics import (
    NOMINAL_CHECKPOINTS,
    build_engineering_curve,
    oracle_gap_closure,
)
from cmc_bbdm.inspection_agent_g1.statistics import (
    FORMAL_BOOTSTRAP_REPLICATES,
    G1StatisticsError,
    formal_synchronized_bootstrap,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _curve(method: str, losses: tuple[float, ...]):
    return build_engineering_curve(
        method=method,
        target_domain="d1",
        specimen_sha256=_sha("sample"),
        task=InspectionTask.FIELD,
        grid_sha256=_sha("grid"),
        evaluator_sha256=_sha("evaluator"),
        warm_start_sha256=_sha("warm"),
        state_budgets=(0.0, 0.01, 0.07, 0.20),
        state_losses=losses,
        state_sha256=tuple(_sha(f"{method}-{index}") for index in range(4)),
    )


def test_engineering_curve_uses_registered_checkpoints_and_carry_forward() -> None:
    curve = _curve("LEARNED", (4.0, 3.0, 2.0, 1.0))
    np.testing.assert_array_equal(curve.nominal_budgets, NOMINAL_CHECKPOINTS)
    np.testing.assert_allclose(curve.exact_budgets, (0.0, 0.01, 0.07, 0.07, 0.20))
    np.testing.assert_allclose(curve.task_losses, (4.0, 3.0, 2.0, 2.0, 1.0))
    assert curve.auebc == pytest.approx(
        np.trapezoid(curve.task_losses, curve.nominal_budgets)
    )


def test_oracle_gap_closure_is_same_geometry_and_unclamped() -> None:
    fixed = _curve("FIXED", (2.0, 2.0, 2.0, 2.0))
    learned = _curve("LEARNED", (1.5, 1.5, 1.5, 1.5))
    oracle = _curve("ORACLE", (1.0, 1.0, 1.0, 1.0))
    result = oracle_gap_closure(fixed=fixed, learned=learned, oracle=oracle)
    assert result.available is True
    assert result.value == pytest.approx(0.5)
    unavailable = oracle_gap_closure(
        fixed=oracle,
        learned=learned,
        oracle=fixed,
    )
    assert unavailable.available is False
    assert unavailable.value is None


def test_formal_bootstrap_is_specimen_paired_equal_domain_and_deterministic() -> None:
    domains = tuple(f"d{domain}" for domain in range(1, 7) for _ in range(2))
    specimens = tuple(f"s{index}" for index in range(12))
    baseline = np.arange(12, dtype=np.float64) + 2.0
    learned = baseline - 1.0
    first = formal_synchronized_bootstrap(
        dataset_ids=domains,
        specimen_ids=specimens,
        baseline_values=baseline,
        learned_values=learned,
        seed=2026090104,
    )
    second = formal_synchronized_bootstrap(
        dataset_ids=domains,
        specimen_ids=specimens,
        baseline_values=baseline,
        learned_values=learned,
        seed=2026090104,
    )
    assert first.replicates == FORMAL_BOOTSTRAP_REPLICATES == 100_000
    assert first.point_estimate == pytest.approx(1.0)
    assert first.ci_lower == pytest.approx(1.0)
    assert first.ci_upper == pytest.approx(1.0)
    assert first.improved_domains == 6
    assert first.distribution_sha256 == second.distribution_sha256
    np.testing.assert_array_equal(first.distribution, second.distribution)


def test_formal_bootstrap_rejects_nonformal_replication_or_domain_roster() -> None:
    with pytest.raises(G1StatisticsError):
        formal_synchronized_bootstrap(
            dataset_ids=("d1",) * 2,
            specimen_ids=("s1", "s2"),
            baseline_values=(2.0, 2.0),
            learned_values=(1.0, 1.0),
            seed=2026090104,
        )
