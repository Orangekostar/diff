"""Source-only action-policy diagnostics over frozen privileged utility labels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.inspection_agent.contracts import InspectionDecision, InspectionTask

from .contracts import ACTION_SLOT_COUNT
from .policy_training import G1PolicyTrainingExample
from .rollout import ObservablePolicyScores


class G1DecisionDiagnosticError(ValueError):
    """Raised when source-only action diagnostics lose their frozen lineage."""


_DECISIONS = (
    InspectionDecision.FOCUS,
    InspectionDecision.BROADEN,
    InspectionDecision.REFINE,
)


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
class G1SourceDecisionDiagnosticRecord:
    outer_target: str
    source_domain: str
    specimen_sha256: str
    task: InspectionTask
    dagger_iteration: int
    state_origin: str
    training_example_sha256: str
    policy_state_sha256: str
    teacher_label_sha256: str
    action_model_sha256: str
    score_sha256: str
    candidate_count: int
    teacher_selected_slot: int
    predicted_selected_slot: int
    teacher_decision: InspectionDecision
    predicted_decision: InspectionDecision
    high_level_decision_accuracy: float
    primitive_top1_match: float
    top5_utility_recall: float
    expected_teacher_regret: float
    candidate_utility_ndcg: float
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        metrics = (
            float(self.high_level_decision_accuracy),
            float(self.primitive_top1_match),
            float(self.top5_utility_recall),
            float(self.expected_teacher_regret),
            float(self.candidate_utility_ndcg),
        )
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domain) is not str
            or not self.source_domain
            or self.source_domain == self.outer_target
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or type(self.dagger_iteration) is not int
            or self.dagger_iteration not in (0, 1, 2)
            or self.state_origin
            != (
                "TEACHER_BANK"
                if self.dagger_iteration == 0
                else "DAGGER_ACTOR_VISITED"
            )
            or not all(
                _valid_sha256(value)
                for value in (
                    self.training_example_sha256,
                    self.policy_state_sha256,
                    self.teacher_label_sha256,
                    self.action_model_sha256,
                    self.score_sha256,
                )
            )
            or type(self.candidate_count) is not int
            or not 1 <= self.candidate_count <= ACTION_SLOT_COUNT
            or type(self.teacher_selected_slot) is not int
            or type(self.predicted_selected_slot) is not int
            or not 0 <= self.teacher_selected_slot < ACTION_SLOT_COUNT
            or not 0 <= self.predicted_selected_slot < ACTION_SLOT_COUNT
            or self.teacher_decision not in _DECISIONS
            or self.predicted_decision not in _DECISIONS
            or any(not math.isfinite(value) for value in metrics)
            or metrics[0] not in (0.0, 1.0)
            or metrics[1] not in (0.0, 1.0)
            or not 0.0 <= metrics[2] <= 1.0
            or metrics[3] < 0.0
            or not 0.0 <= metrics[4] <= 1.0 + 1.0e-12
        ):
            raise G1DecisionDiagnosticError("source decision diagnostic is invalid")
        object.__setattr__(self, "high_level_decision_accuracy", metrics[0])
        object.__setattr__(self, "primitive_top1_match", metrics[1])
        object.__setattr__(self, "top5_utility_recall", metrics[2])
        object.__setattr__(self, "expected_teacher_regret", metrics[3])
        object.__setattr__(self, "candidate_utility_ndcg", min(metrics[4], 1.0))
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-source-decision-diagnostic",
                    "outer_target": self.outer_target,
                    "source_domain": self.source_domain,
                    "specimen": self.specimen_sha256,
                    "task": self.task.value,
                    "dagger_iteration": self.dagger_iteration,
                    "state_origin": self.state_origin,
                    "training_example": self.training_example_sha256,
                    "policy_state": self.policy_state_sha256,
                    "teacher_label": self.teacher_label_sha256,
                    "action_model": self.action_model_sha256,
                    "scores": self.score_sha256,
                    "candidate_count": self.candidate_count,
                    "teacher_selected_slot": self.teacher_selected_slot,
                    "predicted_selected_slot": self.predicted_selected_slot,
                    "teacher_decision": self.teacher_decision.value,
                    "predicted_decision": self.predicted_decision.value,
                    "metrics": metrics,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1SourceDecisionDiagnosticBankFile:
    row_count: int
    outer_target: str
    source_domains: tuple[str, ...]
    action_model_sha256: str
    parquet_sha256: str
    records_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.row_count) is not int
            or self.row_count <= 0
            or type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domains) is not tuple
            or len(self.source_domains) != 5
            or self.source_domains != tuple(sorted(set(self.source_domains)))
            or self.outer_target in self.source_domains
            or not all(
                _valid_sha256(value)
                for value in (
                    self.action_model_sha256,
                    self.parquet_sha256,
                    self.records_sha256,
                    self.manifest_sha256,
                )
            )
        ):
            raise G1DecisionDiagnosticError(
                "source decision diagnostic bank identity is invalid"
            )


@dataclass(frozen=True, slots=True)
class G1SourceDecisionDiagnosticSummary:
    record_count: int
    high_level_decision_accuracy: float
    primitive_top1_match: float
    top5_utility_recall: float
    expected_teacher_regret: float
    candidate_utility_ndcg: float
    predicted_decision_proportions: tuple[tuple[str, float], ...]
    transition_proportions: tuple[tuple[str, str, float], ...]
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        metrics = (
            float(self.high_level_decision_accuracy),
            float(self.primitive_top1_match),
            float(self.top5_utility_recall),
            float(self.expected_teacher_regret),
            float(self.candidate_utility_ndcg),
        )
        if (
            type(self.record_count) is not int
            or self.record_count <= 0
            or any(not math.isfinite(value) for value in metrics)
            or any(not 0.0 <= value <= 1.0 for value in (*metrics[:3], metrics[4]))
            or metrics[3] < 0.0
            or self.predicted_decision_proportions
            != tuple(
                (decision.value, value)
                for decision, (_name, value) in zip(
                    _DECISIONS,
                    self.predicted_decision_proportions,
                    strict=True,
                )
            )
            or not math.isclose(
                sum(value for _name, value in self.predicted_decision_proportions),
                1.0,
                abs_tol=1.0e-12,
            )
            or not self.transition_proportions
            or not math.isclose(
                sum(value for _teacher, _predicted, value in self.transition_proportions),
                1.0,
                abs_tol=1.0e-12,
            )
        ):
            raise G1DecisionDiagnosticError("source decision summary is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-source-decision-summary",
                    "record_count": self.record_count,
                    "metrics": metrics,
                    "predicted_decisions": self.predicted_decision_proportions,
                    "transitions": self.transition_proportions,
                }
            ),
        )


def _order(values: np.ndarray, slots: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sorted(range(len(slots)), key=lambda index: (-float(values[index]), slots[index]))
    )


def _metric_record(
    example: G1PolicyTrainingExample,
    scores: ObservablePolicyScores,
) -> G1SourceDecisionDiagnosticRecord:
    label = example.teacher_label
    candidates = label.candidates
    slots = tuple(row.slot for row in candidates)
    utilities = np.asarray(
        [row.objective_value for row in candidates], dtype=np.float64
    )
    logits = np.asarray([scores.action_logits[slot] for slot in slots], dtype=np.float64)
    if (
        scores.policy_state_sha256 != example.policy_state.state_sha256
        or not np.all(np.isfinite(logits))
        or not np.all(np.isfinite(utilities))
        or tuple(np.flatnonzero(example.policy_state.legal_action_mask)) != slots
    ):
        raise G1DecisionDiagnosticError("source decision scores changed")
    predicted_order = _order(logits, slots)
    teacher_order = _order(utilities, slots)
    predicted_index = predicted_order[0]
    teacher_index = teacher_order[0]
    if slots[teacher_index] != label.selected_slot:
        raise G1DecisionDiagnosticError("teacher selected action is not utility-optimal")
    count = min(5, len(slots))
    recall = len(set(predicted_order[:count]) & set(teacher_order[:count])) / count
    shifted = logits - float(np.max(logits))
    probabilities = np.exp(shifted)
    probabilities /= float(np.sum(probabilities, dtype=np.float64))
    best = float(np.max(utilities))
    expected_regret = float(
        np.sum(probabilities * (best - utilities), dtype=np.float64)
    )
    gains = utilities - float(np.min(utilities))
    if float(np.max(gains)) <= np.finfo(np.float64).eps:
        ndcg = 1.0
    else:
        discounts = 1.0 / np.log2(
            np.arange(2, len(slots) + 2, dtype=np.float64)
        )
        observed = float(np.sum(gains[list(predicted_order)] * discounts))
        ideal = float(np.sum(gains[list(teacher_order)] * discounts))
        ndcg = observed / ideal
    teacher = candidates[teacher_index]
    predicted = candidates[predicted_index]
    return G1SourceDecisionDiagnosticRecord(
        outer_target=example.outer_target,
        source_domain=example.source_domain,
        specimen_sha256=example.specimen_sha256,
        task=example.task,
        dagger_iteration=example.dagger_iteration,
        state_origin=(
            "TEACHER_BANK"
            if example.dagger_iteration == 0
            else "DAGGER_ACTOR_VISITED"
        ),
        training_example_sha256=example.state_sha256,
        policy_state_sha256=example.policy_state.state_sha256,
        teacher_label_sha256=label.state_sha256,
        action_model_sha256=scores.model_sha256,
        score_sha256=scores.state_sha256,
        candidate_count=len(candidates),
        teacher_selected_slot=teacher.slot,
        predicted_selected_slot=predicted.slot,
        teacher_decision=teacher.decision,
        predicted_decision=predicted.decision,
        high_level_decision_accuracy=float(teacher.decision is predicted.decision),
        primitive_top1_match=float(teacher.slot == predicted.slot),
        top5_utility_recall=float(recall),
        expected_teacher_regret=max(expected_regret, 0.0),
        candidate_utility_ndcg=float(ndcg),
    )


def _record_key(row: G1SourceDecisionDiagnosticRecord) -> tuple[object, ...]:
    return (
        row.outer_target,
        row.source_domain,
        row.specimen_sha256,
        row.task.value,
        row.dagger_iteration,
        row.training_example_sha256,
    )


def _ordered_records(
    rows: tuple[G1SourceDecisionDiagnosticRecord, ...],
) -> tuple[G1SourceDecisionDiagnosticRecord, ...]:
    if (
        type(rows) is not tuple
        or not rows
        or any(type(row) is not G1SourceDecisionDiagnosticRecord for row in rows)
        or len({_record_key(row) for row in rows}) != len(rows)
        or len({row.state_sha256 for row in rows}) != len(rows)
        or len({row.outer_target for row in rows}) != 1
        or len({row.action_model_sha256 for row in rows}) != 1
        or len({row.source_domain for row in rows}) != 5
    ):
        raise G1DecisionDiagnosticError("source decision diagnostic roster changed")
    return tuple(sorted(rows, key=_record_key))


def materialize_g1_source_decision_diagnostics(
    examples: tuple[G1PolicyTrainingExample, ...],
    actor: object,
) -> tuple[G1SourceDecisionDiagnosticRecord, ...]:
    if (
        type(examples) is not tuple
        or not examples
        or any(type(row) is not G1PolicyTrainingExample for row in examples)
        or not callable(getattr(actor, "score_batch", None))
        or not _valid_sha256(getattr(actor, "model_state_sha256", None))
    ):
        raise G1DecisionDiagnosticError("source decision diagnostic request is invalid")
    output = []
    for start in range(0, len(examples), 256):
        batch = examples[start : start + 256]
        scores = actor.score_batch(tuple(row.policy_state for row in batch))
        if (
            type(scores) is not tuple
            or len(scores) != len(batch)
            or any(
                type(row) is not ObservablePolicyScores
                or row.model_sha256 != actor.model_state_sha256
                for row in scores
            )
        ):
            raise G1DecisionDiagnosticError("source decision actor output is invalid")
        output.extend(
            _metric_record(example, score)
            for example, score in zip(batch, scores, strict=True)
        )
    return _ordered_records(tuple(output))


def _equal_fold_mean(
    rows: tuple[G1SourceDecisionDiagnosticRecord, ...],
    value: object,
) -> float:
    outer_values = []
    for outer in sorted({row.outer_target for row in rows}):
        source_values = []
        for source in sorted(
            {row.source_domain for row in rows if row.outer_target == outer}
        ):
            specimen_values = []
            for specimen in sorted(
                {
                    row.specimen_sha256
                    for row in rows
                    if row.outer_target == outer and row.source_domain == source
                }
            ):
                selected = tuple(
                    row
                    for row in rows
                    if row.outer_target == outer
                    and row.source_domain == source
                    and row.specimen_sha256 == specimen
                )
                specimen_values.append(
                    float(np.mean([value(row) for row in selected], dtype=np.float64))
                )
            source_values.append(float(np.mean(specimen_values, dtype=np.float64)))
        outer_values.append(float(np.mean(source_values, dtype=np.float64)))
    return float(np.mean(outer_values, dtype=np.float64))


def summarize_g1_source_decision_diagnostics(
    rows: tuple[G1SourceDecisionDiagnosticRecord, ...],
) -> G1SourceDecisionDiagnosticSummary:
    if (
        type(rows) is not tuple
        or not rows
        or any(type(row) is not G1SourceDecisionDiagnosticRecord for row in rows)
        or len({_record_key(row) for row in rows}) != len(rows)
        or len({row.state_sha256 for row in rows}) != len(rows)
        or any(
            len(
                {
                    row.source_domain
                    for row in rows
                    if row.outer_target == outer
                }
            )
            != 5
            or len(
                {
                    row.action_model_sha256
                    for row in rows
                    if row.outer_target == outer
                }
            )
            != 1
            for outer in {row.outer_target for row in rows}
        )
    ):
        raise G1DecisionDiagnosticError("source decision summary roster changed")
    ordered = tuple(sorted(rows, key=_record_key))
    proportions = tuple(
        (
            decision.value,
            _equal_fold_mean(
                ordered,
                lambda row, expected=decision: float(
                    row.predicted_decision is expected
                ),
            ),
        )
        for decision in _DECISIONS
    )
    transitions = tuple(
        (
            teacher.value,
            predicted.value,
            _equal_fold_mean(
                ordered,
                lambda row, expected_teacher=teacher, expected_predicted=predicted: float(
                    row.teacher_decision is expected_teacher
                    and row.predicted_decision is expected_predicted
                ),
            ),
        )
        for teacher in _DECISIONS
        for predicted in _DECISIONS
    )
    nonzero_transitions = tuple(row for row in transitions if row[2] > 0.0)
    return G1SourceDecisionDiagnosticSummary(
        record_count=len(ordered),
        high_level_decision_accuracy=_equal_fold_mean(
            ordered, lambda row: row.high_level_decision_accuracy
        ),
        primitive_top1_match=_equal_fold_mean(
            ordered, lambda row: row.primitive_top1_match
        ),
        top5_utility_recall=_equal_fold_mean(
            ordered, lambda row: row.top5_utility_recall
        ),
        expected_teacher_regret=_equal_fold_mean(
            ordered, lambda row: row.expected_teacher_regret
        ),
        candidate_utility_ndcg=_equal_fold_mean(
            ordered, lambda row: row.candidate_utility_ndcg
        ),
        predicted_decision_proportions=proportions,
        transition_proportions=nonzero_transitions,
    )


def source_decision_diagnostic_bank_path(
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
        raise G1DecisionDiagnosticError("source decision diagnostic path is invalid")
    return Path(work_root) / outer_target / "decision_diagnostics.parquet"


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.manifest.json")


def _row_payload(row: G1SourceDecisionDiagnosticRecord) -> dict[str, object]:
    return {
        "outer_target": row.outer_target,
        "source_domain": row.source_domain,
        "specimen_sha256": row.specimen_sha256,
        "task": row.task.value,
        "dagger_iteration": row.dagger_iteration,
        "state_origin": row.state_origin,
        "training_example_sha256": row.training_example_sha256,
        "policy_state_sha256": row.policy_state_sha256,
        "teacher_label_sha256": row.teacher_label_sha256,
        "action_model_sha256": row.action_model_sha256,
        "score_sha256": row.score_sha256,
        "candidate_count": row.candidate_count,
        "teacher_selected_slot": row.teacher_selected_slot,
        "predicted_selected_slot": row.predicted_selected_slot,
        "teacher_decision": row.teacher_decision.value,
        "predicted_decision": row.predicted_decision.value,
        "high_level_decision_accuracy": row.high_level_decision_accuracy,
        "primitive_top1_match": row.primitive_top1_match,
        "top5_utility_recall": row.top5_utility_recall,
        "expected_teacher_regret": row.expected_teacher_regret,
        "candidate_utility_ndcg": row.candidate_utility_ndcg,
        "record_sha256": row.state_sha256,
    }


def _record_from_payload(row: dict[str, object]) -> G1SourceDecisionDiagnosticRecord:
    try:
        record = G1SourceDecisionDiagnosticRecord(
            outer_target=str(row["outer_target"]),
            source_domain=str(row["source_domain"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            dagger_iteration=int(row["dagger_iteration"]),
            state_origin=str(row["state_origin"]),
            training_example_sha256=str(row["training_example_sha256"]),
            policy_state_sha256=str(row["policy_state_sha256"]),
            teacher_label_sha256=str(row["teacher_label_sha256"]),
            action_model_sha256=str(row["action_model_sha256"]),
            score_sha256=str(row["score_sha256"]),
            candidate_count=int(row["candidate_count"]),
            teacher_selected_slot=int(row["teacher_selected_slot"]),
            predicted_selected_slot=int(row["predicted_selected_slot"]),
            teacher_decision=InspectionDecision(str(row["teacher_decision"])),
            predicted_decision=InspectionDecision(str(row["predicted_decision"])),
            high_level_decision_accuracy=float(
                row["high_level_decision_accuracy"]
            ),
            primitive_top1_match=float(row["primitive_top1_match"]),
            top5_utility_recall=float(row["top5_utility_recall"]),
            expected_teacher_regret=float(row["expected_teacher_regret"]),
            candidate_utility_ndcg=float(row["candidate_utility_ndcg"]),
        )
        if record.state_sha256 != str(row["record_sha256"]):
            raise G1DecisionDiagnosticError("source decision record hash changed")
        return record
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, G1DecisionDiagnosticError):
            raise
        raise G1DecisionDiagnosticError(
            "source decision diagnostic row cannot be reconstructed"
        ) from error


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_g1_source_decision_diagnostic_bank(
    path: str | Path,
    rows: tuple[G1SourceDecisionDiagnosticRecord, ...],
) -> G1SourceDecisionDiagnosticBankFile:
    destination = Path(path)
    ordered = _ordered_records(rows)
    manifest_path = _manifest_path(destination)
    present = destination.exists(), manifest_path.exists()
    if present == (True, True):
        identity, replay = read_g1_source_decision_diagnostic_bank(destination)
        if tuple(row.state_sha256 for row in replay) != tuple(
            row.state_sha256 for row in ordered
        ):
            raise G1DecisionDiagnosticError(
                "source decision diagnostic changed after freeze"
            )
        return identity
    if present != (False, False):
        raise G1DecisionDiagnosticError("source decision diagnostic bank is partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        pl.DataFrame(
            [_row_payload(row) for row in ordered], infer_schema_length=None
        ).write_parquet(
            temporary,
            compression="zstd",
            statistics=False,
            row_group_size=256,
        )
        parquet_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
        records_sha = _json_sha(tuple(row.state_sha256 for row in ordered))
        manifest = {
            "schema_version": 1,
            "scope": "inspection_agent_g1_source_decision_diagnostics",
            "row_count": len(ordered),
            "outer_target": ordered[0].outer_target,
            "source_domains": sorted({row.source_domain for row in ordered}),
            "action_model_sha256": ordered[0].action_model_sha256,
            "parquet_sha256": parquet_sha,
            "records_sha256": records_sha,
        }
        manifest_payload = (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("ascii")
        os.replace(temporary, destination)
        _atomic_bytes(manifest_path, manifest_payload)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    identity, replay = read_g1_source_decision_diagnostic_bank(destination)
    if tuple(row.state_sha256 for row in replay) != tuple(
        row.state_sha256 for row in ordered
    ):
        raise G1DecisionDiagnosticError(
            "written source decision diagnostics did not replay"
        )
    return identity


def read_g1_source_decision_diagnostic_bank(
    path: str | Path,
) -> tuple[
    G1SourceDecisionDiagnosticBankFile,
    tuple[G1SourceDecisionDiagnosticRecord, ...],
]:
    source = Path(path)
    try:
        parquet_payload = source.read_bytes()
        manifest_payload = _manifest_path(source).read_bytes()
        manifest = json.loads(manifest_payload)
        rows = pl.read_parquet(source).to_dicts()
    except (OSError, UnicodeError, json.JSONDecodeError, pl.exceptions.PolarsError) as error:
        raise G1DecisionDiagnosticError(
            "source decision diagnostic bank cannot be read"
        ) from error
    expected_keys = {
        "schema_version",
        "scope",
        "row_count",
        "outer_target",
        "source_domains",
        "action_model_sha256",
        "parquet_sha256",
        "records_sha256",
    }
    parquet_sha = hashlib.sha256(parquet_payload).hexdigest()
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("scope")
        != "inspection_agent_g1_source_decision_diagnostics"
        or manifest.get("parquet_sha256") != parquet_sha
    ):
        raise G1DecisionDiagnosticError(
            "source decision diagnostic manifest changed"
        )
    records = tuple(_record_from_payload(row) for row in rows)
    ordered = _ordered_records(records)
    records_sha = _json_sha(tuple(row.state_sha256 for row in ordered))
    first = ordered[0]
    sources = tuple(sorted({row.source_domain for row in ordered}))
    if (
        records != ordered
        or manifest.get("row_count") != len(records)
        or manifest.get("outer_target") != first.outer_target
        or manifest.get("source_domains") != list(sources)
        or manifest.get("action_model_sha256") != first.action_model_sha256
        or manifest.get("records_sha256") != records_sha
    ):
        raise G1DecisionDiagnosticError(
            "source decision diagnostic manifest or roster changed"
        )
    return (
        G1SourceDecisionDiagnosticBankFile(
            row_count=len(records),
            outer_target=first.outer_target,
            source_domains=sources,
            action_model_sha256=first.action_model_sha256,
            parquet_sha256=parquet_sha,
            records_sha256=records_sha,
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        ),
        records,
    )


__all__ = [
    "G1DecisionDiagnosticError",
    "G1SourceDecisionDiagnosticBankFile",
    "G1SourceDecisionDiagnosticRecord",
    "G1SourceDecisionDiagnosticSummary",
    "materialize_g1_source_decision_diagnostics",
    "read_g1_source_decision_diagnostic_bank",
    "source_decision_diagnostic_bank_path",
    "summarize_g1_source_decision_diagnostics",
    "write_g1_source_decision_diagnostic_bank",
]
