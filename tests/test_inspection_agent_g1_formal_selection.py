from __future__ import annotations

import json

import pytest
from test_inspection_agent_g1_source_bridge import DOMAINS, _record, _sha

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.formal import FIXED_BASELINE_METHODS
from cmc_bbdm.inspection_agent_g1.formal_selection import (
    G1FormalSelectionError,
    freeze_g1_outer_formal_selection,
    read_g1_outer_formal_selection,
)
from cmc_bbdm.inspection_agent_g1.stopping_policy import StopThresholdSelection


def _bridges():
    return tuple(
        _record(
            source,
            method,
            1.0 + method_index,
            task=task,
        )
        for source in DOMAINS[:-1]
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
        for method_index, method in enumerate(FIXED_BASELINE_METHODS)
    )


def _thresholds():
    return tuple(
        StopThresholdSelection(
            outer_target="d6",
            task=task,
            status="STOP_NOT_AUTHORIZED",
            threshold=None,
            candidates=(),
            trajectory_sha256=(),
            state_sha256=_sha(f"threshold-{task.value}"),
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    )


def test_outer_formal_selection_freezes_only_source_evidence(tmp_path) -> None:
    path = tmp_path / "selection.json"

    first = freeze_g1_outer_formal_selection(
        _bridges(),
        _thresholds(),
        outer_target="d6",
        action_selection_sha256=_sha("action-selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=_sha("stop-model"),
        decision_diagnostic_manifest_sha256=_sha("decision-diagnostics"),
        path=path,
    )
    replay = freeze_g1_outer_formal_selection(
        _bridges(),
        _thresholds(),
        outer_target="d6",
        action_selection_sha256=_sha("action-selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=_sha("stop-model"),
        decision_diagnostic_manifest_sha256=_sha("decision-diagnostics"),
        path=path,
    )

    assert replay == first == read_g1_outer_formal_selection(path)
    assert tuple(row.method for row in first.fixed_selections) == (
        "RANDOM",
        "RANDOM",
    )
    assert first.target_outcomes_opened is False
    assert first.decision_diagnostic_manifest_sha256 == _sha(
        "decision-diagnostics"
    )


def test_outer_formal_selection_rejects_tampering(tmp_path) -> None:
    path = tmp_path / "selection.json"
    freeze_g1_outer_formal_selection(
        _bridges(),
        _thresholds(),
        outer_target="d6",
        action_selection_sha256=_sha("action-selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=_sha("stop-model"),
        decision_diagnostic_manifest_sha256=_sha("decision-diagnostics"),
        path=path,
    )
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["target_outcomes_opened"] = True
    path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(G1FormalSelectionError, match="changed|invalid"):
        read_g1_outer_formal_selection(path)
