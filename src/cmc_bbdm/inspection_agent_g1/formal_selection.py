"""Source-only outer selections frozen before held-out target evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .source_bridge import (
    G1SourceBridgeRecord,
    OuterFixedBridgeSelection,
    select_outer_fixed_bridge,
)
from .stopping_policy import REGISTERED_STOP_THRESHOLDS, StopThresholdSelection


class G1FormalSelectionError(ValueError):
    """Raised when a source-only outer selection changes after freeze."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class G1FrozenStopThreshold:
    outer_target: str
    task: InspectionTask
    status: str
    threshold: float | None
    source_evidence_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or self.status not in {
                "STOP_AUTHORIZED_SOURCE_ONLY",
                "STOP_NOT_AUTHORIZED",
            }
            or (
                self.status == "STOP_AUTHORIZED_SOURCE_ONLY"
                and self.threshold not in REGISTERED_STOP_THRESHOLDS
            )
            or (
                self.status == "STOP_NOT_AUTHORIZED" and self.threshold is not None
            )
            or not _valid_sha256(self.source_evidence_sha256)
        ):
            raise G1FormalSelectionError("frozen STOP threshold is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-frozen-stop-threshold",
                    "outer_target": self.outer_target,
                    "task": self.task.value,
                    "status": self.status,
                    "threshold": self.threshold,
                    "source_evidence": self.source_evidence_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1OuterFormalSelection:
    outer_target: str
    source_domains: tuple[str, ...]
    fixed_selections: tuple[OuterFixedBridgeSelection, ...]
    stop_thresholds: tuple[G1FrozenStopThreshold, ...]
    action_selection_sha256: str
    action_model_sha256: str
    stop_model_sha256: str
    decision_diagnostic_manifest_sha256: str
    target_outcomes_opened: bool = False
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domains) is not tuple
            or len(self.source_domains) != 5
            or len(set(self.source_domains)) != 5
            or self.outer_target in self.source_domains
            or type(self.fixed_selections) is not tuple
            or tuple(row.task for row in self.fixed_selections)
            != (InspectionTask.FIELD, InspectionTask.CAI)
            or any(
                type(row) is not OuterFixedBridgeSelection
                or row.outer_target != self.outer_target
                or row.source_domains != self.source_domains
                for row in self.fixed_selections
            )
            or type(self.stop_thresholds) is not tuple
            or tuple(row.task for row in self.stop_thresholds)
            != (InspectionTask.FIELD, InspectionTask.CAI)
            or any(
                type(row) is not G1FrozenStopThreshold
                or row.outer_target != self.outer_target
                for row in self.stop_thresholds
            )
            or not all(
                _valid_sha256(value)
                for value in (
                    self.action_selection_sha256,
                    self.action_model_sha256,
                    self.stop_model_sha256,
                    self.decision_diagnostic_manifest_sha256,
                )
            )
            or type(self.target_outcomes_opened) is not bool
            or self.target_outcomes_opened
        ):
            raise G1FormalSelectionError("outer formal selection is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 2,
                    "kind": "g1-outer-formal-selection",
                    "outer_target": self.outer_target,
                    "source_domains": self.source_domains,
                    "fixed_selections": tuple(
                        row.state_sha256 for row in self.fixed_selections
                    ),
                    "stop_thresholds": tuple(
                        row.state_sha256 for row in self.stop_thresholds
                    ),
                    "action_selection": self.action_selection_sha256,
                    "action_model": self.action_model_sha256,
                    "stop_model": self.stop_model_sha256,
                    "decision_diagnostics": (
                        self.decision_diagnostic_manifest_sha256
                    ),
                    "target_outcomes_opened": False,
                }
            ),
        )


def _fixed_payload(row: OuterFixedBridgeSelection) -> dict[str, object]:
    return {
        "task": row.task.value,
        "method": row.method,
        "source_domains": list(row.source_domains),
        "equal_domain_auebc": row.equal_domain_auebc,
        "domain_auebc": [list(value) for value in row.domain_auebc],
        "evidence_sha256": row.evidence_sha256,
        "state_sha256": row.state_sha256,
    }


def _stop_payload(row: G1FrozenStopThreshold) -> dict[str, object]:
    return {
        "task": row.task.value,
        "status": row.status,
        "threshold": row.threshold,
        "source_evidence_sha256": row.source_evidence_sha256,
        "state_sha256": row.state_sha256,
    }


def _payload(selection: G1OuterFormalSelection) -> dict[str, object]:
    return {
        "schema_version": 2,
        "scope": "inspection_agent_g1_outer_formal_selection",
        "outer_target": selection.outer_target,
        "source_domains": list(selection.source_domains),
        "fixed_selections": [
            _fixed_payload(row) for row in selection.fixed_selections
        ],
        "stop_thresholds": [
            _stop_payload(row) for row in selection.stop_thresholds
        ],
        "action_selection_sha256": selection.action_selection_sha256,
        "action_model_sha256": selection.action_model_sha256,
        "stop_model_sha256": selection.stop_model_sha256,
        "decision_diagnostic_manifest_sha256": (
            selection.decision_diagnostic_manifest_sha256
        ),
        "target_outcomes_opened": False,
        "state_sha256": selection.state_sha256,
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def outer_formal_selection_path(
    work_root: str | Path,
    outer_target: str,
) -> Path:
    if (
        type(outer_target) is not str
        or not outer_target
        or "/" in outer_target
        or "\\" in outer_target
        or outer_target in {".", ".."}
    ):
        raise G1FormalSelectionError("outer formal selection path is invalid")
    return Path(work_root) / outer_target / "selection.json"


def freeze_g1_outer_formal_selection(
    bridge_records: tuple[G1SourceBridgeRecord, ...],
    stop_thresholds: tuple[StopThresholdSelection, ...],
    *,
    outer_target: str,
    action_selection_sha256: str,
    action_model_sha256: str,
    stop_model_sha256: str,
    decision_diagnostic_manifest_sha256: str,
    path: str | Path,
) -> G1OuterFormalSelection:
    if (
        type(stop_thresholds) is not tuple
        or tuple(row.task for row in stop_thresholds)
        != (InspectionTask.FIELD, InspectionTask.CAI)
        or any(
            type(row) is not StopThresholdSelection
            or row.outer_target != outer_target
            or not _valid_sha256(row.state_sha256)
            for row in stop_thresholds
        )
    ):
        raise G1FormalSelectionError("source STOP selection roster is invalid")
    fixed = tuple(
        select_outer_fixed_bridge(
            bridge_records,
            outer_target=outer_target,
            task=task,
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    )
    sources = fixed[0].source_domains
    frozen_stop = tuple(
        G1FrozenStopThreshold(
            outer_target=outer_target,
            task=row.task,
            status=row.status,
            threshold=row.threshold,
            source_evidence_sha256=row.state_sha256,
        )
        for row in stop_thresholds
    )
    selection = G1OuterFormalSelection(
        outer_target=outer_target,
        source_domains=sources,
        fixed_selections=fixed,
        stop_thresholds=frozen_stop,
        action_selection_sha256=action_selection_sha256,
        action_model_sha256=action_model_sha256,
        stop_model_sha256=stop_model_sha256,
        decision_diagnostic_manifest_sha256=(
            decision_diagnostic_manifest_sha256
        ),
    )
    destination = Path(path)
    if destination.exists():
        replay = read_g1_outer_formal_selection(destination)
        if replay.state_sha256 != selection.state_sha256:
            raise G1FormalSelectionError("outer formal selection changed after freeze")
        return replay
    _atomic_json(destination, _payload(selection))
    replay = read_g1_outer_formal_selection(destination)
    if replay.state_sha256 != selection.state_sha256:
        raise G1FormalSelectionError("outer formal selection did not replay")
    return replay


def read_g1_outer_formal_selection(path: str | Path) -> G1OuterFormalSelection:
    try:
        payload = json.loads(Path(path).read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise G1FormalSelectionError("outer formal selection cannot be read") from error
    expected_keys = {
        "schema_version",
        "scope",
        "outer_target",
        "source_domains",
        "fixed_selections",
        "stop_thresholds",
        "action_selection_sha256",
        "action_model_sha256",
        "stop_model_sha256",
        "decision_diagnostic_manifest_sha256",
        "target_outcomes_opened",
        "state_sha256",
    }
    try:
        if (
            not isinstance(payload, dict)
            or set(payload) != expected_keys
            or payload["schema_version"] != 2
            or payload["scope"] != "inspection_agent_g1_outer_formal_selection"
            or payload["target_outcomes_opened"] is not False
            or not isinstance(payload["fixed_selections"], list)
            or not isinstance(payload["stop_thresholds"], list)
        ):
            raise G1FormalSelectionError("outer formal selection schema changed")
        outer = str(payload["outer_target"])
        fixed = tuple(
            OuterFixedBridgeSelection(
                outer_target=outer,
                task=InspectionTask(str(row["task"])),
                method=str(row["method"]),
                source_domains=tuple(str(value) for value in row["source_domains"]),
                equal_domain_auebc=float(row["equal_domain_auebc"]),
                domain_auebc=tuple(
                    (str(value[0]), float(value[1]))
                    for value in row["domain_auebc"]
                ),
                evidence_sha256=str(row["evidence_sha256"]),
            )
            for row in payload["fixed_selections"]
        )
        if any(
            row.state_sha256 != str(raw["state_sha256"])
            for row, raw in zip(fixed, payload["fixed_selections"], strict=True)
        ):
            raise G1FormalSelectionError("frozen fixed selection changed")
        frozen_stop = tuple(
            G1FrozenStopThreshold(
                outer_target=outer,
                task=InspectionTask(str(row["task"])),
                status=str(row["status"]),
                threshold=(
                    None if row["threshold"] is None else float(row["threshold"])
                ),
                source_evidence_sha256=str(row["source_evidence_sha256"]),
            )
            for row in payload["stop_thresholds"]
        )
        if any(
            row.state_sha256 != str(raw["state_sha256"])
            for row, raw in zip(
                frozen_stop, payload["stop_thresholds"], strict=True
            )
        ):
            raise G1FormalSelectionError("frozen STOP threshold changed")
        selection = G1OuterFormalSelection(
            outer_target=outer,
            source_domains=tuple(str(value) for value in payload["source_domains"]),
            fixed_selections=fixed,
            stop_thresholds=frozen_stop,
            action_selection_sha256=str(payload["action_selection_sha256"]),
            action_model_sha256=str(payload["action_model_sha256"]),
            stop_model_sha256=str(payload["stop_model_sha256"]),
            decision_diagnostic_manifest_sha256=str(
                payload["decision_diagnostic_manifest_sha256"]
            ),
        )
        if selection.state_sha256 != str(payload["state_sha256"]):
            raise G1FormalSelectionError("outer formal selection state changed")
        return selection
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, G1FormalSelectionError):
            raise
        raise G1FormalSelectionError("outer formal selection is invalid") from error


__all__ = [
    "G1FormalSelectionError",
    "G1FrozenStopThreshold",
    "G1OuterFormalSelection",
    "freeze_g1_outer_formal_selection",
    "outer_formal_selection_path",
    "read_g1_outer_formal_selection",
]
