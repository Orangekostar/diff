from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.formal import FIXED_BASELINE_METHODS
from cmc_bbdm.inspection_agent_g1.metrics import build_engineering_curve
from cmc_bbdm.inspection_agent_g1.source_bridge import (
    G1SourceBridgeError,
    G1SourceBridgeRecord,
    read_source_bridge_bank,
    select_source_fixed_bridge,
    write_source_bridge_bank,
)

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _record(
    source: str,
    method: str,
    loss: float,
    *,
    task: InspectionTask = InspectionTask.FIELD,
    specimen_token: str | None = None,
) -> G1SourceBridgeRecord:
    specimen = f"{source}-specimen"
    fit_domains = tuple(
        domain for domain in DOMAINS if domain not in {"d6", source}
    )
    state_hashes = tuple(_sha(f"{source}-{method}-{index}") for index in range(5))
    curve = build_engineering_curve(
        method=method,
        target_domain=source,
        specimen_sha256=_sha(specimen if specimen_token is None else specimen_token),
        task=task,
        grid_sha256=_sha(f"grid-{source}"),
        evaluator_sha256=_sha(f"evaluator-{source}"),
        warm_start_sha256=_sha(f"warm-{source}-{task.value}"),
        state_budgets=(0.0, 0.05, 0.1, 0.18, 0.24),
        state_losses=(loss,) * 5,
        state_sha256=state_hashes,
    )
    return G1SourceBridgeRecord(
        outer_target="d6",
        source_domain=source,
        specimen_id=specimen,
        fit_domains=fit_domains,
        dependency_sha256=_sha(f"dependencies-{source}"),
        action_history_sha256=_sha(f"actions-{source}-{method}-{task.value}"),
        curve=curve,
    )


def test_source_fixed_bridge_selection_excludes_validation_domain() -> None:
    records = []
    for source in DOMAINS[:-1]:
        for method_index, method in enumerate(FIXED_BASELINE_METHODS):
            loss = 1.0 + method_index
            if method == "SURFACE_FOCUS":
                loss = 0.5 if source != "d1" else 100.0
            records.append(_record(source, method, loss))

    selection = select_source_fixed_bridge(
        tuple(records),
        validation_domain="d1",
        task=InspectionTask.FIELD,
    )

    assert selection.method == "SURFACE_FOCUS"
    assert selection.fit_domains == ("d2", "d3", "d4", "d5")
    assert tuple(domain for domain, _value in selection.domain_auebc) == (
        "d2",
        "d3",
        "d4",
        "d5",
    )
    assert selection.equal_domain_auebc == 0.125


def test_source_bridge_bank_round_trips_curve_evidence(tmp_path: Path) -> None:
    records = tuple(
        _record("d1", method, 1.0 + index, task=task)
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
        for index, method in enumerate(
            (
                *FIXED_BASELINE_METHODS,
                f"ORACLE_{task.value}",
            )
        )
    )
    path = tmp_path / "bridge.parquet"

    identity = write_source_bridge_bank(path, records)
    replay_identity, replay = read_source_bridge_bank(path)

    assert replay_identity == identity
    ordered = tuple(
        sorted(
            records,
            key=lambda row: (
                row.outer_target,
                row.source_domain,
                row.specimen_id,
                row.curve.task.value,
                row.curve.method,
            ),
        )
    )
    assert tuple(row.state_sha256 for row in replay) == tuple(
        row.state_sha256 for row in ordered
    )
    for expected, actual in zip(ordered, replay, strict=True):
        assert actual.curve.state_sha256 == expected.curve.state_sha256
        assert np.array_equal(actual.curve.exact_budgets, expected.curve.exact_budgets)
        assert np.array_equal(actual.curve.task_losses, expected.curve.task_losses)


def test_source_bridge_bank_rejects_cross_method_specimen_mixing(
    tmp_path: Path,
) -> None:
    records = [
        _record("d1", method, 1.0 + index, task=task)
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
        for index, method in enumerate(
            (*FIXED_BASELINE_METHODS, f"ORACLE_{task.value}")
        )
    ]
    records[0] = _record(
        "d1",
        records[0].curve.method,
        1.0,
        task=records[0].curve.task,
        specimen_token="other-physical-specimen",
    )

    with pytest.raises(G1SourceBridgeError, match="specimen identity"):
        write_source_bridge_bank(tmp_path / "mixed.parquet", tuple(records))
