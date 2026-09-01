from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    SourceBackgroundPrior,
    reconstruct_observation,
)
from cmc_bbdm.inspection_agent.state import action_added_positions, fitting_actions
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1 import teacher_bank as teacher_bank_module
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.crossfit import build_crossfit_roster
from cmc_bbdm.inspection_agent_g1.features import build_policy_state
from cmc_bbdm.inspection_agent_g1.policy_training import G1PolicyTrainingExample
from cmc_bbdm.inspection_agent_g1.teacher import (
    authorize_source_teacher,
    field_teacher_label,
)
from cmc_bbdm.inspection_agent_g1.teacher_bank import (
    ContinuationPolicy,
    G1TeacherBankError,
    G1TeacherBankRecord,
    _positive_cost_actions,
    materialize_label_independent_states,
    materialize_oracle_checkpoint_states,
    read_teacher_bank,
    write_teacher_bank,
)
from cmc_bbdm.inspection_agent_g1.warm_start import (
    PRIMARY_WARM_START_CELLS,
    build_deployment_grid,
)
from cmc_bbdm.mavis.authority import MAVISAuthority

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _hypothesis() -> SurfaceHypothesis:
    scores = np.linspace(0.0, 1.0, 64)
    scores.setflags(write=False)
    median = np.zeros(3)
    median.setflags(write=False)
    return SurfaceHypothesis(scores, tuple(range(63, 55, -1)), median, "b" * 64)


def _prior() -> SourceBackgroundPrior:
    return SourceBackgroundPrior(
        outer_domain="d6",
        source_domains=("d2", "d3", "d4", "d5"),
        fit_specimen_ids=("s2", "s3", "s4", "s5"),
        source_authority_sha256="a" * 64,
        domain_border_medians=np.full((4, 3), 90.0),
        background_rgb=np.full(3, 90, dtype=np.uint8),
    )


def _world(*, invert: bool, true_cai: float = 0.4):
    rows, columns = np.indices((41, 43))
    image = np.stack((3 * rows, 4 * columns, rows + columns), axis=2).astype(np.uint8)
    if invert:
        image = 255 - image
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("d1-sample",),
        dataset_ids=("d1",),
        images=(image,),
        targets=np.asarray([true_cai]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )
    surface = np.zeros((80, 80, 3), dtype=np.uint8)
    grid = build_deployment_grid(image.shape[:2])
    world = CausalInspectionWorld(
        authority,
        specimen_id="d1-sample",
        task=InspectionTask.FIELD,
        surface_rgb=surface,
        surface_sha256=hashlib.sha256(surface.tobytes()).hexdigest(),
        grid=grid,
        endpoint_budget=0.25,
    )
    return world, grid, image


def test_label_independent_bank_has_one_warm_and_twelve_continuation_states() -> None:
    world, grid, _image = _world(invert=False)
    states = materialize_label_independent_states(
        world,
        grid,
        _hypothesis(),
        outer_target="d6",
        random_seed=2026090101,
        snapshot_fractions=(1 / 3, 2 / 3, 1.0),
    )
    assert len(states) == 13
    assert states[0].source == "WARM_START"
    assert tuple(
        action.cell_index for action in states[0].observation.action_history
    ) == (PRIMARY_WARM_START_CELLS)
    for policy in ContinuationPolicy:
        selected = [state for state in states if state.source == policy.value]
        assert len(selected) == 3
        assert [state.snapshot_index for state in selected] == [0, 1, 2]
        assert [len(state.observation.action_history) for state in selected] == sorted(
            len(state.observation.action_history) for state in selected
        )
        assert all(state.label_independent for state in selected)
    for state in states:
        assert any(
            len(
                action_added_positions(
                    grid, state.observation.measurement_state, action
                )
            )
            for action in fitting_actions(
                grid,
                state.observation.measurement_state,
                state.observation.endpoint_budget,
            )
        )


def test_fast_positive_cost_roster_matches_registered_state_semantics() -> None:
    world, grid, _image = _world(invert=False)
    state = materialize_label_independent_states(
        world,
        grid,
        _hypothesis(),
        outer_target="d6",
        random_seed=2026090101,
        snapshot_fractions=(1 / 3, 2 / 3, 1.0),
    )[7].observation.measurement_state
    expected = tuple(
        action
        for action in fitting_actions(grid, state, 0.25)
        if len(action_added_positions(grid, state, action)) > 0
    )
    assert not hasattr(teacher_bank_module, "action_added_positions")
    assert _positive_cost_actions(grid, state, 0.25) == expected


def test_continuation_histories_do_not_depend_on_hidden_scan_or_cai() -> None:
    first_world, grid, _ = _world(invert=False, true_cai=0.1)
    second_world, _, _ = _world(invert=True, true_cai=99.0)
    kwargs = {
        "outer_target": "d6",
        "random_seed": 2026090101,
        "snapshot_fractions": (1 / 3, 2 / 3, 1.0),
    }
    first = materialize_label_independent_states(
        first_world, grid, _hypothesis(), **kwargs
    )
    second = materialize_label_independent_states(
        second_world, grid, _hypothesis(), **kwargs
    )
    assert [(row.source, row.observation.action_history) for row in first] == [
        (row.source, row.observation.action_history) for row in second
    ]


def test_field_oracle_checkpoint_states_are_fold_safe_and_at_or_below_checkpoints() -> (
    None
):
    world, grid, image = _world(invert=False)
    authorization = authorize_source_teacher(
        build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1"),
        query_domain="d1",
    )
    rows = materialize_oracle_checkpoint_states(
        world,
        grid,
        _hypothesis(),
        _prior(),
        authorization,
        full_scan=image,
        checkpoints=(0.0625, 0.125, 0.1875, 0.25),
    )
    assert 1 <= len(rows) <= 4
    assert len({row.observation.state_sha256 for row in rows}) == len(rows)
    assert sum(len(row.checkpoints) for row in rows) == 4
    for row in rows:
        assert row.label_independent is False
        assert all(
            row.observation.effective_budget <= checkpoint
            for checkpoint in row.checkpoints
        )
        assert (
            tuple(action.cell_index for action in row.observation.action_history[:8])
            == PRIMARY_WARM_START_CELLS
        )
        assert any(
            len(action_added_positions(grid, row.observation.measurement_state, action))
            for action in fitting_actions(
                grid,
                row.observation.measurement_state,
                row.observation.endpoint_budget,
            )
        )


def test_teacher_bank_parquet_round_trip_revalidates_all_three_namespaces(
    tmp_path: Path,
) -> None:
    world, grid, image = _world(invert=False)
    source_state = materialize_label_independent_states(
        world,
        grid,
        _hypothesis(),
        outer_target="d6",
        random_seed=2026090101,
        snapshot_fractions=(1 / 3, 2 / 3, 1.0),
    )[0]
    authorization = authorize_source_teacher(
        build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1"),
        query_domain="d1",
    )
    reconstruction = reconstruct_observation(source_state.observation, grid, _prior())
    policy_state = build_policy_state(
        source_state.observation,
        _hypothesis(),
        grid,
        _prior(),
        reconstruction,
        reconstruction_embedding=np.linspace(0.0, 1.0, 512),
        cai_estimate=0.4,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
    )
    label = field_teacher_label(
        source_state.observation,
        grid,
        _prior(),
        _hypothesis(),
        authorization,
        full_scan=image,
        policy_state_sha256=policy_state.state_sha256,
    )
    example = G1PolicyTrainingExample(
        outer_target="d6",
        source_domain="d1",
        specimen_sha256=hashlib.sha256(b"d1-sample").hexdigest(),
        task=InspectionTask.FIELD,
        dagger_iteration=0,
        policy_state=policy_state,
        teacher_label=label,
    )
    record = G1TeacherBankRecord(
        example=example,
        fit_domains=("d2", "d3", "d4", "d5"),
        state_source="WARM_START",
        source_state_sha256=source_state.state_sha256,
        prior_sha256=_prior().state_sha256,
        assessor_sha256="c" * 64,
    )
    path = tmp_path / "teacher.parquet"
    identity = write_teacher_bank(path, (record,))
    loaded_identity, loaded = read_teacher_bank(path)
    assert loaded_identity == identity
    assert loaded == (record,)
    assert loaded[0].example.policy_state.state_sha256 == policy_state.state_sha256
    assert loaded[0].example.teacher_label.state_sha256 == label.state_sha256

    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(G1TeacherBankError, match="SHA-256"):
        read_teacher_bank(path)
