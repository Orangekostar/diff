from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.stopping import ReferenceEndpoint
from cmc_bbdm.inspection_agent_g1.crossfit import build_crossfit_roster
from cmc_bbdm.inspection_agent_g1.stop_bank import (
    G1StopBankError,
    G1StopBankRecord,
    read_stop_bank,
    write_stop_bank,
)
from cmc_bbdm.inspection_agent_g1.stopping_policy import (
    GATE_ELIGIBLE_FIXED_METHODS,
    build_source_stop_label,
    select_source_fixed_reference,
)
from cmc_bbdm.inspection_agent_g1.teacher import authorize_source_teacher

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _record(token: str, *, current_loss: float) -> G1StopBankRecord:
    authorization = authorize_source_teacher(
        build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1"),
        query_domain="d1",
    )
    rows = tuple(
        ReferenceEndpoint(
            method=method,
            dataset_id=domain,
            specimen_id=f"{domain}-sample",
            task_loss=0.5 if method == "CENTER_FIRST" else 1.0,
        )
        for domain in ("d2", "d3", "d4", "d5")
        for method in GATE_ELIGIBLE_FIXED_METHODS
    )
    reference = select_source_fixed_reference(authorization, rows)
    policy_state_sha = _sha(f"policy-{token}")
    specimen_sha = _sha(f"specimen-{token}")
    label = build_source_stop_label(
        authorization,
        reference,
        source_domain="d1",
        specimen_sha256=specimen_sha,
        task=InspectionTask.FIELD,
        policy_state_sha256=policy_state_sha,
        current_true_loss=current_loss,
        reference_true_loss=1.0,
    )
    return G1StopBankRecord(
        outer_target="d6",
        source_domain="d1",
        specimen_sha256=specimen_sha,
        task=InspectionTask.FIELD,
        fit_domains=("d2", "d3", "d4", "d5"),
        state_source="WARM_START",
        source_state_sha256=_sha(f"source-state-{token}"),
        action_example_sha256=_sha(f"action-example-{token}"),
        policy_state_sha256=policy_state_sha,
        label=label,
    )


def test_stop_bank_round_trip_preserves_privileged_labels_and_join_keys(
    tmp_path: Path,
) -> None:
    records = (_record("a", current_loss=1.04), _record("b", current_loss=1.06))
    path = tmp_path / "stop.parquet"

    identity = write_stop_bank(path, records)
    loaded_identity, loaded = read_stop_bank(path)

    assert loaded_identity == identity
    assert loaded == records
    assert identity.row_count == 2
    assert tuple(row.label.is_sufficient for row in loaded) == (True, False)
    assert all(row.policy_state_sha256 == row.label.policy_state_sha256 for row in loaded)


def test_stop_bank_rejects_duplicate_policy_state_join_keys(tmp_path: Path) -> None:
    row = _record("duplicate", current_loss=1.0)
    with pytest.raises(G1StopBankError, match="join key"):
        write_stop_bank(tmp_path / "stop.parquet", (row, row))
