"""Deterministic post-warm source states for the fold-safe G1 teacher bank."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.inspection_agent.contracts import (
    InspectionDecision,
    InspectionObservation,
    InspectionTask,
)
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.oracle import (
    CAIStatePredictor,
    ReconstructionEncoder,
    choose_cai_action,
    choose_field_action,
)
from cmc_bbdm.inspection_agent.state import (
    GeneralizedMeasurementState,
    InspectionCellAction,
    action_added_positions_from_mask,
    apply_action,
    legal_actions,
    measurement_mask,
)
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.mva.oracle import uniform_cell_order

from .contracts import CAIContextMode, G1PolicyState, TaskTokenMode
from .features import canonical_action_from_slot
from .policy_training import G1PolicyTrainingExample
from .teacher import (
    PrivilegedTeacherLabel,
    SourceTeacherAuthorization,
    TeacherCandidateRecord,
    validate_source_teacher_dependencies,
)
from .warm_start import PRIMARY_WARM_START_CELLS, apply_warm_start


class G1TeacherBankError(ValueError):
    """Raised when source-state generation violates the frozen G1 roster."""


class ContinuationPolicy(str, Enum):
    UNIFORM_CONTINUE = "UNIFORM_CONTINUE"
    SURFACE_FOCUS_CONTINUE = "SURFACE_FOCUS_CONTINUE"
    RANDOM_CONTINUE = "RANDOM_CONTINUE"
    ALTERNATE_BROADEN_REFINE = "ALTERNATE_BROADEN_REFINE"


@dataclass(frozen=True, slots=True)
class TeacherBankState:
    outer_target: str
    source: str
    snapshot_index: int
    progress_fraction: float
    label_independent: bool
    observation: InspectionObservation
    state_sha256: str


@dataclass(frozen=True, slots=True)
class OracleCheckpointState:
    checkpoints: tuple[float, ...]
    label_independent: bool
    observation: InspectionObservation
    state_sha256: str


_STATE_SOURCES = frozenset(
    (
        "WARM_START",
        "UNIFORM_CONTINUE",
        "SURFACE_FOCUS_CONTINUE",
        "RANDOM_CONTINUE",
        "ALTERNATE_BROADEN_REFINE",
        "ORACLE_CHECKPOINT",
    )
)


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class G1TeacherBankRecord:
    example: G1PolicyTrainingExample
    fit_domains: tuple[str, ...]
    state_source: str
    source_state_sha256: str
    prior_sha256: str
    assessor_sha256: str
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.example) is not G1PolicyTrainingExample
            or type(self.fit_domains) is not tuple
            or len(self.fit_domains) != 4
            or len(set(self.fit_domains)) != 4
            or self.example.outer_target in self.fit_domains
            or self.example.source_domain in self.fit_domains
            or self.state_source not in _STATE_SOURCES
            or not all(
                _valid_sha256(value)
                for value in (
                    self.source_state_sha256,
                    self.prior_sha256,
                    self.assessor_sha256,
                )
            )
            or self.example.dagger_iteration != 0
            or self.example.policy_state.cai_context_mode
            is not CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT
            or self.example.policy_state.task_token_mode is not TaskTokenMode.CORRECT
        ):
            raise G1TeacherBankError("teacher-bank record is invalid")
        state = _json_digest(
            {
                "schema": 1,
                "kind": "g1-teacher-bank-record",
                "example": self.example.state_sha256,
                "fit_domains": self.fit_domains,
                "state_source": self.state_source,
                "source_state": self.source_state_sha256,
                "prior": self.prior_sha256,
                "assessor": self.assessor_sha256,
            }
        )
        if self.state_sha256 not in ("", state):
            raise G1TeacherBankError("teacher-bank record hash changed")
        object.__setattr__(self, "state_sha256", state)


@dataclass(frozen=True, slots=True)
class G1TeacherBankFile:
    row_count: int
    parquet_sha256: str
    records_sha256: str
    manifest_sha256: str


def _state_hash(kind: str, values: tuple[object, ...]) -> str:
    return hashlib.sha256(
        ("|".join(("inspection-agent-g1-teacher-bank-v1", kind, *(str(v) for v in values)))).encode(
            "ascii"
        )
    ).hexdigest()


def _ordered_action(
    actions: tuple[InspectionCellAction, ...],
    order: tuple[int, ...],
    *,
    minimum_level: bool,
) -> InspectionCellAction:
    eligible = actions
    if minimum_level:
        level = min(action.from_level for action in actions)
        eligible = tuple(action for action in actions if action.from_level == level)
    by_cell = {action.cell_index: action for action in eligible}
    for cell in order:
        if cell in by_cell:
            return by_cell[cell]
    raise G1TeacherBankError("continuation order has no eligible action")


def _positive_cost_actions(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
    endpoint_budget: float,
) -> tuple[InspectionCellAction, ...]:
    current_mask = measurement_mask(grid, state)
    current_count = int(np.count_nonzero(current_mask))
    native_count = int(current_mask.size)
    output = []
    for action in legal_actions(grid, state):
        added = action_added_positions_from_mask(
            grid,
            state,
            action,
            current_mask,
        )
        if (
            len(added) > 0
            and (current_count + len(added)) / native_count <= endpoint_budget + 1.0e-15
        ):
            output.append(action)
    return tuple(output)


def plan_continuation_actions(
    grid: AcquisitionGrid,
    warm_state: GeneralizedMeasurementState,
    policy: ContinuationPolicy,
    surface_hypothesis: SurfaceHypothesis,
    *,
    outer_target: str,
    task: InspectionTask,
    surface_sha256: str,
    random_seed: int,
    endpoint_budget: float,
) -> tuple[InspectionCellAction, ...]:
    endpoint = float(endpoint_budget)
    if (
        type(grid) is not AcquisitionGrid
        or type(warm_state) is not GeneralizedMeasurementState
        or warm_state != apply_warm_start(grid, k=8)
        or type(policy) is not ContinuationPolicy
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(outer_target) is not str
        or not outer_target
        or task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or type(surface_sha256) is not str
        or len(surface_sha256) != 64
        or type(random_seed) is not int
        or isinstance(endpoint_budget, bool)
        or not math.isfinite(endpoint)
        or not 0.0 < endpoint <= 1.0
    ):
        raise G1TeacherBankError("continuation policy request is invalid")
    uniform_order = uniform_cell_order()
    surface_order = tuple(
        sorted(
            range(64),
            key=lambda cell: (-float(surface_hypothesis.scores[cell]), cell),
        )
    )
    token = hashlib.sha256(
        (
            f"g1-continuation|{random_seed}|{outer_target}|"
            f"{surface_sha256}|{task.value}"
        ).encode("ascii")
    ).hexdigest()
    generator = np.random.Generator(np.random.PCG64(int(token[:16], 16)))
    state = warm_state
    output: list[InspectionCellAction] = []
    pending_refine: int | None = None
    for _ in range(192):
        actions = _positive_cost_actions(grid, state, endpoint)
        if not actions:
            break
        if policy is ContinuationPolicy.UNIFORM_CONTINUE:
            selected = _ordered_action(actions, uniform_order, minimum_level=True)
        elif policy is ContinuationPolicy.SURFACE_FOCUS_CONTINUE:
            selected = _ordered_action(actions, surface_order, minimum_level=True)
        elif policy is ContinuationPolicy.RANDOM_CONTINUE:
            selected = actions[int(generator.integers(0, len(actions)))]
        else:
            immediate = tuple(
                action
                for action in actions
                if pending_refine is not None
                and action.cell_index == pending_refine
                and action.from_level == 0
            )
            if immediate:
                selected = immediate[0]
                pending_refine = None
            else:
                broaden = tuple(action for action in actions if action.from_level == -1)
                selected = _ordered_action(
                    broaden if broaden else actions,
                    uniform_order,
                    minimum_level=not broaden,
                )
                pending_refine = selected.cell_index if selected.from_level == -1 else None
        output.append(selected)
        state = apply_action(grid, state, selected)
    else:
        raise G1TeacherBankError("continuation policy exceeded the finite action roster")
    if not output:
        raise G1TeacherBankError("continuation policy produced no action")
    return tuple(output)


def _bank_state(
    outer_target: str,
    source: str,
    snapshot_index: int,
    fraction: float,
    label_independent: bool,
    observation: InspectionObservation,
) -> TeacherBankState:
    return TeacherBankState(
        outer_target=outer_target,
        source=source,
        snapshot_index=snapshot_index,
        progress_fraction=fraction,
        label_independent=label_independent,
        observation=observation,
        state_sha256=_state_hash(
            "state",
            (
                source,
                outer_target,
                snapshot_index,
                f"{fraction:.17g}",
                label_independent,
                observation.state_sha256,
            ),
        ),
    )


def materialize_label_independent_states(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    surface_hypothesis: SurfaceHypothesis,
    *,
    outer_target: str,
    random_seed: int,
    snapshot_fractions: tuple[float, ...],
) -> tuple[TeacherBankState, ...]:
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(outer_target) is not str
        or not outer_target
        or type(snapshot_fractions) is not tuple
        or len(snapshot_fractions) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= 1.0
            for value in snapshot_fractions
        )
        or any(float(right) <= float(left) for left, right in pairwise(snapshot_fractions))
        or float(snapshot_fractions[-1]) != 1.0
    ):
        raise G1TeacherBankError("label-independent state request is invalid")
    warm_actions = tuple(
        canonical_action_from_slot(cell) for cell in PRIMARY_WARM_START_CELLS
    )
    warm = world.replay(warm_actions)
    rows = [_bank_state(outer_target, "WARM_START", 0, 0.0, True, warm)]
    for policy in ContinuationPolicy:
        actions = plan_continuation_actions(
            grid,
            warm.measurement_state,
            policy,
            surface_hypothesis,
            outer_target=outer_target,
            task=warm.task,
            surface_sha256=warm.surface_sha256,
            random_seed=random_seed,
            endpoint_budget=warm.endpoint_budget,
        )
        action_counts = tuple(
            min(len(actions) - 1, math.ceil(float(fraction) * len(actions)))
            for fraction in snapshot_fractions
        )
        if min(action_counts) < 1 or len(set(action_counts)) != 3:
            raise G1TeacherBankError("continuation snapshots are not unique")
        for snapshot_index, action_count in enumerate(action_counts):
            observation = world.replay(
                (*warm_actions, *actions[:action_count])
            )
            rows.append(
                _bank_state(
                    outer_target,
                    policy.value,
                    snapshot_index,
                    float(snapshot_fractions[snapshot_index]),
                    True,
                    observation,
                )
            )
    if len(rows) != 13:
        raise G1TeacherBankError("label-independent state count changed")
    return tuple(rows)


def materialize_oracle_checkpoint_states(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    surface_hypothesis: SurfaceHypothesis,
    prior: SourceBackgroundPrior,
    authorization: SourceTeacherAuthorization,
    *,
    full_scan: object,
    checkpoints: tuple[float, ...],
    true_cai: float | None = None,
    assessor: CAIStatePredictor | None = None,
    encoder: ReconstructionEncoder | None = None,
) -> tuple[OracleCheckpointState, ...]:
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(checkpoints) is not tuple
        or len(checkpoints) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= 0.25
            for value in checkpoints
        )
        or any(float(right) <= float(left) for left, right in pairwise(checkpoints))
    ):
        raise G1TeacherBankError("oracle checkpoint request is invalid")
    validate_source_teacher_dependencies(
        authorization,
        prior,
        assessor=assessor,
    )
    warm_actions = tuple(
        canonical_action_from_slot(cell) for cell in PRIMARY_WARM_START_CELLS
    )
    current = world.replay(warm_actions)
    trajectory = [current]
    for _ in range(192):
        if not _positive_cost_actions(
            grid,
            current.measurement_state,
            current.endpoint_budget,
        ):
            break
        if current.task is InspectionTask.FIELD:
            selection = choose_field_action(
                current,
                grid,
                prior,
                full_scan=full_scan,
                checkpoint=current.endpoint_budget,
            )
        elif current.task is InspectionTask.CAI:
            if true_cai is None or assessor is None or encoder is None:
                raise G1TeacherBankError("CAI oracle dependencies are incomplete")
            selection = choose_cai_action(
                current,
                grid,
                prior,
                full_scan=full_scan,
                true_cai=true_cai,
                assessor=assessor,
                encoder=encoder,
                checkpoint=current.endpoint_budget,
            )
        else:
            raise G1TeacherBankError("G1 teacher bank supports FIELD and CAI only")
        current = world.step(current, selection.action)
        trajectory.append(current)
    else:
        raise G1TeacherBankError("oracle trajectory exceeded the finite action roster")
    grouped: dict[str, tuple[InspectionObservation, list[float]]] = {}
    for checkpoint in checkpoints:
        eligible = [
            observation
            for observation in trajectory
            if observation.effective_budget <= float(checkpoint) + 1.0e-15
            and _positive_cost_actions(
                grid,
                observation.measurement_state,
                observation.endpoint_budget,
            )
        ]
        if not eligible:
            raise G1TeacherBankError("no oracle state fits a checkpoint")
        selected = eligible[-1]
        if selected.state_sha256 not in grouped:
            grouped[selected.state_sha256] = (selected, [])
        grouped[selected.state_sha256][1].append(float(checkpoint))
    output = []
    for observation, grouped_checkpoints in grouped.values():
        checkpoints_tuple = tuple(grouped_checkpoints)
        output.append(
            OracleCheckpointState(
                checkpoints=checkpoints_tuple,
                label_independent=False,
                observation=observation,
                state_sha256=_state_hash(
                    "oracle-checkpoint",
                    (
                        authorization.state_sha256,
                        checkpoints_tuple,
                        observation.state_sha256,
                    ),
                ),
            )
        )
    return tuple(output)


def _record_row(record: G1TeacherBankRecord) -> dict[str, object]:
    example = record.example
    state = example.policy_state
    label = example.teacher_label
    candidates = label.candidates
    return {
        "integrity_outer_target": example.outer_target,
        "integrity_source_domain": example.source_domain,
        "integrity_specimen_sha256": example.specimen_sha256,
        "integrity_fit_domains_json": json.dumps(
            record.fit_domains, separators=(",", ":")
        ),
        "integrity_state_source": record.state_source,
        "integrity_source_state_sha256": record.source_state_sha256,
        "integrity_prior_sha256": record.prior_sha256,
        "integrity_assessor_sha256": record.assessor_sha256,
        "integrity_example_sha256": example.state_sha256,
        "integrity_record_sha256": record.state_sha256,
        "policy_visible_task": state.task.value,
        "policy_visible_task_token_mode": state.task_token_mode.value,
        "policy_visible_cai_context_mode": state.cai_context_mode.value,
        "policy_visible_observation_sha256": state.observation_sha256,
        "policy_visible_reconstruction_sha256": state.reconstruction_sha256,
        "policy_visible_surface_hypothesis_sha256": state.surface_hypothesis_sha256,
        "policy_visible_grid_sha256": state.grid_sha256,
        "policy_visible_state_sha256": state.state_sha256,
        "policy_visible_reconstruction_embedding": state.reconstruction_embedding.tolist(),
        "policy_visible_global_scalars": state.global_scalars.tolist(),
        "policy_visible_task_token": state.task_token.tolist(),
        "policy_visible_cell_features": state.cell_features.reshape(-1).tolist(),
        "policy_visible_candidate_features": state.candidate_features.reshape(-1).tolist(),
        "policy_visible_legal_action_mask": state.legal_action_mask.tolist(),
        "privileged_teacher_authorization_sha256": label.authorization_sha256,
        "privileged_teacher_selected_slot": label.selected_slot,
        "privileged_teacher_state_sha256": label.state_sha256,
        "privileged_teacher_candidate_slots": [value.slot for value in candidates],
        "privileged_teacher_candidate_cells": [
            value.action.cell_index for value in candidates
        ],
        "privileged_teacher_candidate_from_levels": [
            value.action.from_level for value in candidates
        ],
        "privileged_teacher_candidate_to_levels": [
            value.action.to_level for value in candidates
        ],
        "privileged_teacher_candidate_decisions": [
            value.decision.value for value in candidates
        ],
        "privileged_teacher_candidate_exact_added_costs": [
            value.exact_added_cost for value in candidates
        ],
        "privileged_teacher_candidate_raw_values": [
            value.raw_value for value in candidates
        ],
        "privileged_teacher_candidate_objective_values": [
            value.objective_value for value in candidates
        ],
        "privileged_teacher_candidate_task_losses_after": [
            value.task_loss_after for value in candidates
        ],
        "privileged_teacher_candidate_state_sha256s": [
            value.candidate_state_sha256 for value in candidates
        ],
        "privileged_teacher_candidate_selected": [
            value.selected for value in candidates
        ],
    }


def _ordered_records(
    records: tuple[G1TeacherBankRecord, ...],
) -> tuple[G1TeacherBankRecord, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(record) is not G1TeacherBankRecord for record in records)
        or len({record.state_sha256 for record in records}) != len(records)
        or len({record.example.outer_target for record in records}) != 1
        or len({record.example.source_domain for record in records}) != 1
        or len({record.fit_domains for record in records}) != 1
    ):
        raise G1TeacherBankError("teacher-bank record roster is invalid")
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.example.specimen_sha256,
                record.example.task.value,
                record.state_source,
                record.example.policy_state.state_sha256,
            ),
        )
    )


def _records_sha(records: tuple[G1TeacherBankRecord, ...]) -> str:
    return _json_digest(tuple(record.state_sha256 for record in records))


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.manifest.json")


def write_teacher_bank(
    path: str | Path,
    records: tuple[G1TeacherBankRecord, ...],
) -> G1TeacherBankFile:
    destination = Path(path)
    ordered = _ordered_records(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pl.DataFrame([_record_row(record) for record in ordered], infer_schema_length=None).write_parquet(
            temporary,
            compression="zstd",
            statistics=False,
            row_group_size=512,
        )
        payload = temporary.read_bytes()
        parquet_sha = hashlib.sha256(payload).hexdigest()
        records_sha = _records_sha(ordered)
        manifest = {
            "schema_version": 1,
            "scope": "inspection_agent_g1_source_teacher_bank",
            "row_count": len(ordered),
            "outer_target": ordered[0].example.outer_target,
            "source_domain": ordered[0].example.source_domain,
            "fit_domains": list(ordered[0].fit_domains),
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
        manifest_destination = _manifest_path(destination)
        manifest_descriptor, manifest_name = tempfile.mkstemp(
            prefix=f".{manifest_destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(manifest_descriptor)
        manifest_temporary = Path(manifest_name)
        manifest_temporary.write_bytes(manifest_payload)
        os.replace(temporary, destination)
        os.replace(manifest_temporary, manifest_destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    identity, loaded = read_teacher_bank(destination)
    if tuple(record.state_sha256 for record in loaded) != tuple(
        record.state_sha256 for record in ordered
    ):
        raise G1TeacherBankError("written teacher bank did not replay exactly")
    return identity


def _list(row: dict[str, object], key: str) -> list[object]:
    value = row.get(key)
    if not isinstance(value, list):
        raise G1TeacherBankError(f"teacher-bank list column is invalid: {key}")
    return value


def _record_from_row(row: dict[str, object]) -> G1TeacherBankRecord:
    try:
        task = InspectionTask(str(row["policy_visible_task"]))
        state = G1PolicyState(
            task=task,
            task_token_mode=TaskTokenMode(
                str(row["policy_visible_task_token_mode"])
            ),
            cai_context_mode=CAIContextMode(
                str(row["policy_visible_cai_context_mode"])
            ),
            observation_sha256=str(row["policy_visible_observation_sha256"]),
            reconstruction_sha256=str(row["policy_visible_reconstruction_sha256"]),
            surface_hypothesis_sha256=str(
                row["policy_visible_surface_hypothesis_sha256"]
            ),
            grid_sha256=str(row["policy_visible_grid_sha256"]),
            reconstruction_embedding=np.asarray(
                _list(row, "policy_visible_reconstruction_embedding"),
                dtype=np.float64,
            ),
            global_scalars=np.asarray(
                _list(row, "policy_visible_global_scalars"), dtype=np.float64
            ),
            task_token=np.asarray(
                _list(row, "policy_visible_task_token"), dtype=np.float64
            ),
            cell_features=np.asarray(
                _list(row, "policy_visible_cell_features"), dtype=np.float64
            ).reshape(64, 18),
            candidate_features=np.asarray(
                _list(row, "policy_visible_candidate_features"), dtype=np.float64
            ).reshape(192, 12),
            legal_action_mask=np.asarray(
                _list(row, "policy_visible_legal_action_mask"), dtype=np.bool_
            ),
            state_sha256=str(row["policy_visible_state_sha256"]),
        )
        candidate_columns = (
            _list(row, "privileged_teacher_candidate_slots"),
            _list(row, "privileged_teacher_candidate_cells"),
            _list(row, "privileged_teacher_candidate_from_levels"),
            _list(row, "privileged_teacher_candidate_to_levels"),
            _list(row, "privileged_teacher_candidate_decisions"),
            _list(row, "privileged_teacher_candidate_exact_added_costs"),
            _list(row, "privileged_teacher_candidate_raw_values"),
            _list(row, "privileged_teacher_candidate_objective_values"),
            _list(row, "privileged_teacher_candidate_task_losses_after"),
            _list(row, "privileged_teacher_candidate_state_sha256s"),
            _list(row, "privileged_teacher_candidate_selected"),
        )
        if len({len(value) for value in candidate_columns}) != 1:
            raise G1TeacherBankError("teacher candidate columns have unequal lengths")
        candidates = tuple(
            TeacherCandidateRecord(
                slot=int(slot),
                action=InspectionCellAction(int(cell), int(source), int(target)),
                decision=InspectionDecision(str(decision)),
                exact_added_cost=int(cost),
                raw_value=float(raw),
                objective_value=float(objective),
                task_loss_after=float(loss),
                candidate_state_sha256=str(candidate_sha),
                selected=bool(selected),
            )
            for (
                slot,
                cell,
                source,
                target,
                decision,
                cost,
                raw,
                objective,
                loss,
                candidate_sha,
                selected,
            ) in zip(*candidate_columns, strict=True)
        )
        label = PrivilegedTeacherLabel(
            task=task,
            authorization_sha256=str(
                row["privileged_teacher_authorization_sha256"]
            ),
            observation_sha256=state.observation_sha256,
            policy_state_sha256=state.state_sha256,
            selected_slot=int(row["privileged_teacher_selected_slot"]),
            candidates=candidates,
            state_sha256=str(row["privileged_teacher_state_sha256"]),
        )
        example = G1PolicyTrainingExample(
            outer_target=str(row["integrity_outer_target"]),
            source_domain=str(row["integrity_source_domain"]),
            specimen_sha256=str(row["integrity_specimen_sha256"]),
            task=task,
            dagger_iteration=0,
            policy_state=state,
            teacher_label=label,
        )
        if example.state_sha256 != str(row["integrity_example_sha256"]):
            raise G1TeacherBankError("teacher-bank example hash changed")
        fit_domains_raw = json.loads(str(row["integrity_fit_domains_json"]))
        if not isinstance(fit_domains_raw, list):
            raise G1TeacherBankError("teacher-bank fit domains are invalid")
        return G1TeacherBankRecord(
            example=example,
            fit_domains=tuple(str(value) for value in fit_domains_raw),
            state_source=str(row["integrity_state_source"]),
            source_state_sha256=str(row["integrity_source_state_sha256"]),
            prior_sha256=str(row["integrity_prior_sha256"]),
            assessor_sha256=str(row["integrity_assessor_sha256"]),
            state_sha256=str(row["integrity_record_sha256"]),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ) as error:
        if isinstance(error, G1TeacherBankError):
            raise
        raise G1TeacherBankError("teacher-bank row cannot be reconstructed") from error


def read_teacher_bank(
    path: str | Path,
) -> tuple[G1TeacherBankFile, tuple[G1TeacherBankRecord, ...]]:
    source = Path(path)
    manifest_path = _manifest_path(source)
    try:
        parquet_payload = source.read_bytes()
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise G1TeacherBankError("teacher-bank package cannot be read") from error
    parquet_sha = hashlib.sha256(parquet_payload).hexdigest()
    expected_keys = {
        "schema_version",
        "scope",
        "row_count",
        "outer_target",
        "source_domain",
        "fit_domains",
        "parquet_sha256",
        "records_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("scope") != "inspection_agent_g1_source_teacher_bank"
        or manifest.get("parquet_sha256") != parquet_sha
    ):
        raise G1TeacherBankError("teacher-bank Parquet SHA-256 mismatch")
    try:
        rows = pl.read_parquet(source).to_dicts()
    except (OSError, pl.exceptions.PolarsError) as error:
        raise G1TeacherBankError("teacher-bank Parquet cannot be decoded") from error
    records = tuple(_record_from_row(row) for row in rows)
    ordered = _ordered_records(records)
    records_sha = _records_sha(ordered)
    if (
        records != ordered
        or manifest.get("row_count") != len(records)
        or manifest.get("outer_target") != records[0].example.outer_target
        or manifest.get("source_domain") != records[0].example.source_domain
        or manifest.get("fit_domains") != list(records[0].fit_domains)
        or manifest.get("records_sha256") != records_sha
    ):
        raise G1TeacherBankError("teacher-bank manifest or row order changed")
    identity = G1TeacherBankFile(
        row_count=len(records),
        parquet_sha256=parquet_sha,
        records_sha256=records_sha,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
    )
    return identity, records


__all__ = [
    "ContinuationPolicy",
    "G1TeacherBankError",
    "G1TeacherBankFile",
    "G1TeacherBankRecord",
    "OracleCheckpointState",
    "TeacherBankState",
    "materialize_label_independent_states",
    "materialize_oracle_checkpoint_states",
    "plan_continuation_actions",
    "read_teacher_bank",
    "write_teacher_bank",
]
