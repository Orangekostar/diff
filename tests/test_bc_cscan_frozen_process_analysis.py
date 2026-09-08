from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cmc_bbdm.inspection_agent.state import (
    InspectionCellAction,
    apply_action,
    measurement_mask,
    zero_state,
)
from cmc_bbdm.learned_cscan.contracts import Split, Task
from cmc_bbdm.learned_cscan.frozen_evidence_finalize import (
    evaluate_path_b_decision,
    match_human_sessions,
    rescore_frozen_episode,
    summarize_blind_reviews,
)
from cmc_bbdm.learned_cscan.frozen_process_analysis import (
    analyze_trajectory_tables,
    derive_action_events,
    load_frozen_process_config,
    overlap_metrics,
    planning_stop_accounting,
    report_stability,
    stop_decomposition,
    surface_cell_partition,
    verify_frozen_input,
)
from cmc_bbdm.learned_cscan.frozen_process_recovery import (
    FrozenVisibleReportReader,
    action_spatial_allocation,
    cell_partition_masks,
    recover_reviewed_report_scores,
    support_spatial_allocation,
)
from cmc_bbdm.learned_cscan.readout import (
    BackgroundPrior,
    read_visible_task_report,
)
from cmc_bbdm.learned_cscan.runtime import load_study_config, load_study_context
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.vlm_cscan.contracts import (
    CScanReference,
    ReferenceType,
    ReviewState,
)
from scripts.analyze_bc_cscan_frozen_process import build_parser

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/bc_cscan_frozen_process_analysis.yaml"


def _rows(
    costs: Iterable[float],
    successes: Iterable[bool],
    *,
    calibrated_stop_step: int | None = None,
    rule_stop_step: int | None = None,
) -> tuple[dict[str, object], ...]:
    costs_tuple = tuple(costs)
    successes_tuple = tuple(successes)
    assert len(costs_tuple) == len(successes_tuple)
    rows = []
    for step, (cost, success) in enumerate(
        zip(costs_tuple, successes_tuple, strict=True)
    ):
        terminal = step == len(costs_tuple) - 1
        rows.append(
            {
                "dataset_id": "domain",
                "specimen_id": "s1",
                "specimen_key": "domain:s1",
                "task": "LOCATE",
                "method": "BC_S1",
                "seed": 1,
                "step": step,
                "cost": cost,
                "success": success,
                "task_loss": float(not success),
                "iou": float(success),
                "recall": float(success),
                "relative_area_error": 0.0,
                "report_sha256": f"report-{step}",
                "support_count": step,
                "action_cell": -1 if terminal else 0,
                "action_from_level": -2 if terminal else step - 1,
                "action_to_level": -2 if terminal else step,
                "rule_stop": rule_stop_step is not None and step >= rule_stop_step,
                "calibrated_stop_trigger": step == calibrated_stop_step,
                "learned_stop_probability": (
                    0.99 if step == calibrated_stop_step else 0.50
                ),
                "learned_stop_eligible": step > 0,
                "cumulative_route_cost": float(step) / 10.0,
                "cumulative_route_turns": step,
                "action_inference_seconds": 0.01,
                "reference_version": "PROXY_LEGACY",
                "stop_model": "S_BC_CAL",
                "calibrated_threshold": 0.99,
            }
        )
    return tuple(rows)


def test_stop_prefix_excludes_action_recorded_on_stop_row() -> None:
    rows = _rows((0.0, 0.1, 0.1), (False, True, True), calibrated_stop_step=1)

    stop = stop_decomposition(rows, "S_BC_CAL")
    events = derive_action_events(rows, {"S_BC_CAL": 1, "S_RULE": None})

    assert stop["stop_step"] == 1
    assert stop["completion"] is True
    assert stop["prefix_action_count"] == 1
    assert len(events) == 2
    assert events[0]["in_s_bc_cal_prefix"] is True
    assert events[1]["in_s_bc_cal_prefix"] is False
    assert events[1]["added_measurement_fraction"] == 0.0
    assert all(event["action_cell"] >= 0 for event in events)


def test_report_stability_preserves_nonmonotone_current_reports() -> None:
    rows = _rows((0.0, 0.2, 0.4, 1.0), (False, True, False, True))

    stability = report_stability(rows, stop_step=1)

    assert stability["first_success_step"] == 1
    assert stability["first_success_cost"] == 0.2
    assert stability["sustained_success_step"] == 3
    assert stability["sustained_success_cost"] == 1.0
    assert stability["full_input_only"] is True
    assert stability["success_to_failure_count"] == 1
    assert stability["failure_to_success_count"] == 2
    assert stability["regressed_after_first_success"] is True
    assert stability["outcome_category"] == "SUCCESS_BEFORE_LATER_REGRESSION"


@pytest.mark.parametrize(
    ("costs", "successes", "stop_step", "expected"),
    (
        (
            (0.0, 0.2, 0.6, 1.0),
            (False, True, True, True),
            2,
            {"planner_ausc": 0.8, "autonomous_ausc": 0.4, "gap": 0.4},
        ),
        (
            (0.0, 0.2, 0.4, 1.0),
            (False, True, False, True),
            1,
            {"planner_ausc": 0.2, "autonomous_ausc": 0.8, "gap": -0.6},
        ),
        (
            (0.0, 0.1, 0.3, 1.0),
            (False, False, True, True),
            1,
            {"planner_ausc": 0.7, "autonomous_ausc": 0.0, "gap": 0.7},
        ),
    ),
)
def test_planning_to_stop_accounting_is_signed_and_exact(
    costs: tuple[float, ...],
    successes: tuple[bool, ...],
    stop_step: int,
    expected: dict[str, float],
) -> None:
    accounting = planning_stop_accounting(
        _rows(costs, successes), stop_step=stop_step
    )

    for field, value in expected.items():
        assert accounting[field] == pytest.approx(value)
    assert accounting["gap"] == pytest.approx(
        accounting["planner_ausc"] - accounting["autonomous_ausc"]
    )
    if accounting["completion"]:
        assert accounting["gap"] == pytest.approx(
            accounting["pre_stop_success_area"]
            - accounting["post_stop_failure_area"]
        )


def test_duplicate_cost_uses_the_last_report_at_that_cost() -> None:
    accounting = planning_stop_accounting(
        _rows((0.0, 0.2, 0.2, 1.0), (False, True, False, True)),
        stop_step=None,
    )

    assert accounting["planner_ausc"] == 0.0
    assert accounting["autonomous_ausc"] == 0.0


def test_surface_cell_partition_is_mutually_exclusive() -> None:
    partition = surface_cell_partition((0,))

    assert partition["V"] == (0,)
    assert partition["R"] == (1, 8, 9)
    assert len(partition["O"]) == 60
    assert set(partition["V"]).isdisjoint(partition["R"])
    assert set(partition["V"]) | set(partition["R"]) | set(
        partition["O"]
    ) == set(range(64))

    empty = surface_cell_partition(())
    assert empty == {"V": (), "R": (), "O": tuple(range(64))}


def test_overlap_metrics_keep_empty_targets_and_no_cues_explicit() -> None:
    empty = np.zeros((3, 4), dtype=np.bool_)
    target = np.zeros_like(empty)
    target[1, 1] = True
    cue = np.zeros_like(empty)
    cue[1, 2] = True

    empty_target = overlap_metrics(empty, cue)
    no_cue = overlap_metrics(target, empty)
    disjoint = overlap_metrics(target, cue)
    overlap = overlap_metrics(target, target)

    assert empty_target["overlap_group"] == "EMPTY_TARGET"
    assert empty_target["target_coverage_by_surface"] is None
    assert no_cue["overlap_group"] == "NO_CUE"
    assert no_cue["surface_precision_proxy"] is None
    assert disjoint["overlap_group"] == "CUE_DISJOINT"
    assert disjoint["surface_target_iou"] == 0.0
    assert overlap["overlap_group"] == "CUE_OVERLAP"
    assert overlap["target_coverage_by_surface"] == 1.0


def test_config_binds_frozen_inputs_and_zero_model_calls() -> None:
    config = load_frozen_process_config(CONFIG, project_root=ROOT)

    assert config.repository_base_sha == "fe58de39298c412d0829580cfda40b7c7c4c53e9"
    assert config.source_result_root == ROOT / "results/bc_cscan_path_b_supplement"
    assert config.output_root == ROOT / "results/bc_cscan_frozen_process_analysis"
    assert config.source_result_root != config.output_root
    assert config.methods == (
        "R_BALANCED_P4",
        "R_BALANCED_P8",
        "BC_S1",
        "BC_S2",
        "BC_S3",
        "BC_NO_VLM_S1",
        "BC_NO_US_FEEDBACK_S1",
    )
    assert config.tasks == ("LOCATE", "CHARACTERIZE")
    assert config.stop_thresholds == {"LOCATE": 0.99, "CHARACTERIZE": 0.99}
    assert config.training_updates == 0
    assert config.vlm_calls == 0
    assert config.actor_stop_forward_calls == 0
    assert config.base_recovery_step_cap == 40_000
    assert config.reviewed_recovery_step_cap == 60_000


def test_changed_frozen_input_is_rejected(tmp_path: Path) -> None:
    changed = tmp_path / "trajectory.parquet"
    changed.write_bytes(b"changed")

    with pytest.raises(ValueError, match="trajectory hash changed"):
        verify_frozen_input(changed, "0" * 64, label="trajectory")


def test_real_frozen_matrix_contract_and_table_counts() -> None:
    config = load_frozen_process_config(CONFIG, project_root=ROOT)
    trajectories = pl.read_parquet(
        config.source_result_root / "trajectories.parquet"
    )

    tables = analyze_trajectory_tables(config, trajectories=trajectories)

    assert tables["input_manifest"]["matrix"] == {
        "physical_specimens": 24,
        "domains": 6,
        "methods": 7,
        "tasks": 2,
        "episodes": 336,
        "trajectory_rows": 64848,
        "states_per_episode": 193,
        "actions_per_episode": 192,
    }
    assert len(tables["first_stop_decomposition"]) == 672
    assert len(tables["report_stability_and_delay"]) == 672
    assert len(tables["planning_to_stop_accounting"]) == 672
    assert len(tables["stop_wait_components"]) == 672
    assert len(tables["action_events"]) == 64512
    assert len(tables["action_cost_allocation"]) == 1008
    assert len(tables["prefix_divergence"]) == 144
    assert len(tables["paired_diagnostic_effects"]) == 24
    anchors = tables["input_manifest"]["planner_ausc_anchors"]
    assert anchors["LOCATE"]["BC_3SEED_MEAN"] == pytest.approx(
        0.7268777738167383
    )
    assert anchors["LOCATE"]["R_BALANCED_P8"] == pytest.approx(
        0.6298188483710804
    )
    assert anchors["CHARACTERIZE"]["BC_3SEED_MEAN"] == pytest.approx(
        0.5082537144730087
    )
    assert anchors["CHARACTERIZE"]["R_BALANCED_P8"] == pytest.approx(
        0.467930098731157
    )


def test_spatial_allocation_keeps_action_measurement_and_support_distinct() -> None:
    grid = build_acquisition_grid(65, 65, initial_budget=0.015625)
    partition = surface_cell_partition((0,))
    region_masks = cell_partition_masks(grid, partition)

    assert np.all(
        region_masks["V"].astype(np.int8)
        + region_masks["R"].astype(np.int8)
        + region_masks["O"].astype(np.int8)
        == 1
    )

    state = zero_state(grid)
    current = np.zeros(grid.native_shape, dtype=np.bool_)
    action = InspectionCellAction(0, -1, 0)
    target = np.zeros(grid.native_shape, dtype=np.bool_)
    target[8, 8] = True

    allocation = action_spatial_allocation(
        grid,
        state=state,
        action=action,
        current_mask=current,
        target_mask=target,
        region_masks=region_masks,
    )
    support = support_spatial_allocation(
        np.asarray([[0, 0]], dtype=np.int64),
        target_mask=target,
        region_masks=region_masks,
    )

    assert allocation["action_cell_region"] == "V"
    assert allocation["added_pixel_count"] == (
        allocation["added_v_pixel_count"]
        + allocation["added_r_pixel_count"]
        + allocation["added_o_pixel_count"]
    )
    assert allocation["added_target_pixel_count"] == 1
    assert support["support_target_pixel_count"] == 0


def test_cached_frozen_reader_is_exact_for_mixed_cell_levels() -> None:
    grid = build_acquisition_grid(40, 42, initial_budget=0.0625)
    random = np.random.default_rng(20260908)
    full_scan = random.integers(0, 256, (*grid.native_shape, 3), dtype=np.uint8)
    prior = BackgroundPrior(
        rgb=np.asarray((121, 132, 143), dtype=np.uint8),
        fit_specimen_keys=("train:s1",),
        fit_split="TRAIN",
    )
    fast = FrozenVisibleReportReader(
        grid=grid,
        full_scan=full_scan,
        prior=prior,
        distance_threshold=0.21,
    )
    state = zero_state(grid)
    actions = (
        InspectionCellAction(0, -1, 0),
        InspectionCellAction(1, -1, 0),
        InspectionCellAction(0, 0, 1),
        InspectionCellAction(8, -1, 0),
        InspectionCellAction(1, 0, 1),
        InspectionCellAction(0, 1, 2),
    )
    states = [state]
    for action in actions:
        state = apply_action(grid, state, action)
        states.append(state)

    for state in states:
        measured = measurement_mask(grid, state)
        positions = np.argwhere(measured)
        values = full_scan[positions[:, 0], positions[:, 1]]
        for task in (Task.LOCATE, Task.CHARACTERIZE):
            expected = read_visible_task_report(
                grid=grid,
                positions=positions,
                values=values,
                cell_levels=state.levels,
                prior=prior,
                distance_threshold=0.21,
                task=task,
            ).report
            actual = fast.report(state=state, measured_mask=measured, task=task)

            assert np.array_equal(actual.predicted_mask, expected.predicted_mask)
            assert np.array_equal(actual.support_positions, expected.support_positions)
            assert actual.candidate_cells == expected.candidate_cells
            assert (
                actual.unverified_boundary_cells
                == expected.unverified_boundary_cells
            )
            assert actual.signal_strength == pytest.approx(
                expected.signal_strength, abs=1e-15
            )
            assert actual.reason_code == expected.reason_code


def test_reviewed_recovery_executes_one_specimen_without_model_calls() -> None:
    config = load_frozen_process_config(CONFIG, project_root=ROOT)
    trajectories = pl.read_parquet(
        config.source_result_root / "trajectories.parquet"
    )
    specimen_key = str(trajectories["specimen_key"].sort().item(0))
    parent = load_study_config(config.parent_config_path, project_root=ROOT)
    context = load_study_context(
        parent, source_root=Path("/home/ww/paper3/cmc_damage_inference")
    )
    record = next(
        assignment.record
        for assignment in context.roster.assignments
        if assignment.split is Split.TEST
        and assignment.record.specimen_key == specimen_key
    )
    target = np.zeros(record.native_shape, dtype=np.bool_)
    target[record.native_shape[0] // 2, record.native_shape[1] // 2] = True
    reference = CScanReference(
        specimen_key=specimen_key,
        source_image_sha256=record.cscan_sha256,
        reference_type=ReferenceType.AUTHOR_PROVIDED,
        review_state=ReviewState.REVIEWED,
        reviewer_alias="integration-test",
        certain_mask=target,
        uncertain_mask=np.zeros_like(target),
    )

    recovered = recover_reviewed_report_scores(
        config,
        source_root=Path("/home/ww/paper3/cmc_damage_inference"),
        trajectories=trajectories,
        references={specimen_key: reference},
        reference_version="REVIEWED_INTEGRATION_TEST",
    )

    assert len(recovered["report_scores"]) == 2 * 6 * 193
    assert len(recovered["full_input_readout"]) == 2
    assert recovered["recovery"]["episode_count"] == 12
    assert recovered["recovery"]["stored_action_transition_count"] == 2 * 6 * 192
    assert recovered["recovery"]["world_step_count"] == 0
    assert recovered["recovery"]["training_updates"] == 0
    assert recovered["recovery"]["vlm_calls"] == 0
    assert recovered["recovery"]["actor_forward_calls"] == 0
    assert recovered["recovery"]["stop_forward_calls"] == 0


def test_reviewed_cache_merge_reuses_only_prior_specimens_and_updates_version() -> None:
    from cmc_bbdm.learned_cscan.frozen_evidence_finalize import (
        _merge_reviewed_recoveries,
    )

    config = load_frozen_process_config(CONFIG, project_root=ROOT)
    table_names = (
        "report_scores",
        "full_input_readout",
        "surface_reference_agreement",
        "spatial_action_statistics",
        "spatial_episode_summary",
    )
    cached = {
        name: (
            {
                "specimen_key": "domain:cached",
                "reference_version": "REVIEWED_OLD",
            },
        )
        for name in table_names
    }
    fresh = {
        name: (
            {
                "specimen_key": "domain:fresh",
                "reference_version": "REVIEWED_NEW",
            },
        )
        for name in table_names
    }
    fresh["recovery"] = {
        "stored_action_transition_count": 2 * 6 * config.actions_per_episode,
    }

    merged = _merge_reviewed_recoveries(
        config,
        reference_version="REVIEWED_NEW",
        cached=cached,
        fresh=fresh,
        total_specimen_count=2,
        cached_specimen_count=1,
    )

    assert {
        row["reference_version"] for row in merged["report_scores"]
    } == {"REVIEWED_NEW"}
    assert merged["recovery"]["episode_count"] == 24
    assert merged["recovery"]["cache_hit_episode_count"] == 12
    assert merged["recovery"]["stored_action_transition_count"] == 2304
    assert merged["recovery"]["cached_action_transition_count"] == 2304
    assert merged["recovery"]["source_action_transition_count"] == 4608
    assert merged["recovery"]["training_updates"] == 0


def test_reviewed_cache_loads_only_identity_matched_specimens(tmp_path: Path) -> None:
    from cmc_bbdm.learned_cscan.frozen_evidence_finalize import (
        REVIEWED_METHODS,
        _load_cached_reviewed_recovery,
        _reviewed_cache_identity,
    )

    config = replace(
        load_frozen_process_config(CONFIG, project_root=ROOT), output_root=tmp_path
    )
    reviewed = tmp_path / "reviewed"
    reviewed.mkdir()
    specimen_key = "domain:cached"
    reference_sha = "a" * 64
    coverage = (
        {
            "specimen_key": specimen_key,
            "status": "REVIEWED_REFERENCE_MATCHED",
            "reference_sha256": reference_sha,
        },
    )
    pl.DataFrame(coverage).write_csv(reviewed / "reference_coverage.csv")
    report_scores = [
        {
            "specimen_key": specimen_key,
            "task": task,
            "method": method,
            "seed": 8 if method == "R_BALANCED_P8" else 1,
            "step": step,
            "reference_version": "REVIEWED_OLD",
        }
        for task in config.tasks
        for method in REVIEWED_METHODS
        for step in range(config.states_per_episode)
    ]
    pl.DataFrame(report_scores).write_parquet(reviewed / "report_scores.parquet")
    for name, rows in (
        (
            "full_input_readout.csv",
            [
                {
                    "specimen_key": specimen_key,
                    "task": task,
                    "reference_version": "REVIEWED_OLD",
                }
                for task in config.tasks
            ],
        ),
        (
            "surface_reference_agreement.csv",
            [
                {
                    "specimen_key": specimen_key,
                    "task": task,
                    "reference_version": "REVIEWED_OLD",
                }
                for task in config.tasks
            ],
        ),
        (
            "spatial_episode_summary.csv",
            [
                {
                    "specimen_key": specimen_key,
                    "task": task,
                    "method": method,
                    "stop_system": stop_system,
                    "reference_version": "REVIEWED_OLD",
                }
                for task in config.tasks
                for method in config.spatial_methods
                for stop_system in ("NONE_FULL_TRAJECTORY", "S_BC_CAL", "S_RULE")
            ],
        ),
    ):
        pl.DataFrame(rows).write_csv(reviewed / name)
    spatial_actions = [
        {
            "specimen_key": specimen_key,
            "task": task,
            "method": method,
            "seed": 8 if method == "R_BALANCED_P8" else 1,
            "step": step,
            "reference_version": "REVIEWED_OLD",
        }
        for task in config.tasks
        for method in config.spatial_methods
        for step in range(config.actions_per_episode)
    ]
    pl.DataFrame(spatial_actions).write_parquet(
        reviewed / "spatial_action_statistics.parquet"
    )
    cache_manifest = {
        **_reviewed_cache_identity(config),
        "reference_files": {specimen_key: reference_sha},
    }
    (reviewed / "recovery_manifest.json").write_text(
        json.dumps(cache_manifest), encoding="utf-8"
    )

    cached = _load_cached_reviewed_recovery(
        config,
        current_coverage=coverage,
    )
    mismatched = _load_cached_reviewed_recovery(
        config,
        current_coverage=({**coverage[0], "reference_sha256": "b" * 64},),
    )

    assert cached["cached_specimen_keys"] == (specimen_key,)
    assert len(cached["report_scores"]) == 2 * 6 * 193
    assert len(cached["spatial_action_statistics"]) == 2 * 4 * 192
    assert mismatched["cached_specimen_keys"] == ()


def test_reviewed_rescore_changes_success_but_keeps_frozen_stop_and_actions() -> None:
    rows = _rows(
        (0.0, 0.2, 1.0),
        (False, True, True),
        calibrated_stop_step=1,
    )
    reviewed = {"report-0": False, "report-1": False, "report-2": True}

    result = rescore_frozen_episode(
        rows,
        formal_success_by_report=reviewed,
        stop_system="S_BC_CAL",
    )

    assert result["stop_step"] == 1
    assert result["stop_cost"] == 0.2
    assert result["prefix_action_count"] == 1
    assert result["stop_report_sha256"] == "report-1"
    assert result["completion"] is False
    assert result["false_stop"] is True
    assert result["planner_ausc"] == 0.0


def test_path_b_requires_both_bounds_and_complete_reference_coverage() -> None:
    completion = {"estimate": 0.01, "ci_lower": -0.05, "ci_upper": 0.08}
    cost = {"estimate": 0.03, "ci_lower": 0.001, "ci_upper": 0.05}

    supported = evaluate_path_b_decision(
        completion, cost, physical_n=24, expected_physical_n=24
    )
    failed = evaluate_path_b_decision(
        completion,
        {**cost, "ci_lower": 0.0},
        physical_n=24,
        expected_physical_n=24,
    )
    partial = evaluate_path_b_decision(
        completion, cost, physical_n=6, expected_physical_n=24
    )

    assert supported["status"] == "SUPPORTED"
    assert supported["joint_path_b_pass"] is True
    assert failed["status"] == "NOT_SUPPORTED"
    assert failed["joint_path_b_pass"] is False
    assert partial["status"] == "INSUFFICIENT_PRECISION"
    assert partial["joint_path_b_pass"] is None


def test_two_blind_reviewers_do_not_become_two_physical_specimens() -> None:
    reports = {
        "r1": {
            "report_id": "r1",
            "specimen_key": "domain:s1",
            "task": "LOCATE",
            "method": "BC_S1",
            "seed": 1,
            "stop_system": "S_BC_CAL",
            "report_sha256": "digest",
            "objective_success": True,
            "reference_version": "REVIEWED_X",
        }
    }
    reviews = (
        {"report_id": "r1", "reviewer_id": "A", "decision": "DELIVERABLE"},
        {
            "report_id": "r1",
            "reviewer_id": "B",
            "decision": "NEEDS_FURTHER_INSPECTION",
        },
    )

    result = summarize_blind_reviews(reviews, report_index=reports)

    assert len(result["report_decisions"]) == 2
    assert result["method_summary"][0]["review_count"] == 2
    assert result["method_summary"][0]["report_count"] == 1
    assert result["method_summary"][0]["physical_specimen_count"] == 1
    assert result["method_summary"][0]["consensus"] == "NOT_COMPUTED"
    assert result["method_summary"][0]["reference_version"] == "REVIEWED_X"


def test_blind_reviews_pair_p8_and_bc_by_specimen_task_stop_and_reviewer() -> None:
    reports = {
        "p8": {
            "report_id": "p8",
            "specimen_key": "domain:s1",
            "task": "LOCATE",
            "method": "R_BALANCED_P8",
            "seed": 8,
            "stop_system": "S_BC_CAL",
            "report_sha256": "p8-digest",
            "objective_success": False,
            "reference_version": "REVIEWED_X",
        },
        "bc1": {
            "report_id": "bc1",
            "specimen_key": "domain:s1",
            "task": "LOCATE",
            "method": "BC_S1",
            "seed": 1,
            "stop_system": "S_BC_CAL",
            "report_sha256": "bc1-digest",
            "objective_success": True,
            "reference_version": "REVIEWED_X",
        },
        "bc2": {
            "report_id": "bc2",
            "specimen_key": "domain:s1",
            "task": "LOCATE",
            "method": "BC_S2",
            "seed": 2,
            "stop_system": "S_BC_CAL",
            "report_sha256": "bc2-digest",
            "objective_success": True,
            "reference_version": "REVIEWED_X",
        },
    }
    reviews = (
        {"report_id": "p8", "reviewer_id": "A", "decision": "NEEDS_FURTHER_INSPECTION"},
        {"report_id": "bc1", "reviewer_id": "A", "decision": "DELIVERABLE"},
        {"report_id": "bc2", "reviewer_id": "A", "decision": "UNABLE_TO_JUDGE"},
    )

    paired = summarize_blind_reviews(reviews, report_index=reports)[
        "paired_specimen_differences"
    ]

    assert len(paired) == 1
    assert paired[0]["bc_report_count"] == 1
    assert paired[0]["p8_report_count"] == 1
    assert paired[0]["bc_minus_p8_deliverable_fraction"] == 1.0


def test_blind_reviews_do_not_pair_across_reference_versions() -> None:
    common = {
        "specimen_key": "domain:s1",
        "task": "LOCATE",
        "seed": 1,
        "stop_system": "S_BC_CAL",
        "objective_success": True,
    }
    reports = {
        "p8": {
            **common,
            "report_id": "p8",
            "method": "R_BALANCED_P8",
            "report_sha256": "p8-digest",
            "reference_version": "REVIEWED_A",
        },
        "bc": {
            **common,
            "report_id": "bc",
            "method": "BC_S1",
            "report_sha256": "bc-digest",
            "reference_version": "REVIEWED_B",
        },
    }
    reviews = (
        {"report_id": "p8", "reviewer_id": "A", "decision": "DELIVERABLE"},
        {"report_id": "bc", "reviewer_id": "A", "decision": "DELIVERABLE"},
    )

    result = summarize_blind_reviews(reviews, report_index=reports)

    assert result["paired_specimen_differences"] == ()


def test_reviewed_reference_without_certain_region_is_not_formally_scored(
    tmp_path: Path,
) -> None:
    from cmc_bbdm.learned_cscan.frozen_evidence_finalize import _reference_inputs

    config = load_frozen_process_config(CONFIG, project_root=ROOT)
    parent = load_study_config(config.parent_config_path, project_root=ROOT)
    context = load_study_context(
        parent, source_root=Path("/home/ww/paper3/cmc_damage_inference")
    )
    record = next(
        assignment.record
        for assignment in context.roster.assignments
        if assignment.split is Split.TEST
    )
    payload = {
        "specimen_key": record.specimen_key,
        "source_image_sha256": record.cscan_sha256,
        "reference_type": "AUTHOR_PROVIDED",
        "review_state": "reviewed",
        "reviewer_alias": "integration-test",
        "frame": "registered_cscan",
        "regions": [],
        "uncertain_regions": [],
    }
    (tmp_path / "empty-reference.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = _reference_inputs(
        config,
        source_root=Path("/home/ww/paper3/cmc_damage_inference"),
        references_path=tmp_path,
    )

    row = next(
        item for item in result["coverage"] if item["specimen_key"] == record.specimen_key
    )
    assert result["references"] == {}
    assert result["input_status"] == "NO_EVALUABLE_REVIEWED_REFERENCE"
    assert row["status"] == "REVIEWED_NO_CERTAIN_REGION"
    assert row["formal_eligible"] is False


def test_unmatched_human_session_does_not_create_a_fake_pair() -> None:
    sessions = (
        {
            "session_id": "h1",
            "specimen_key": "domain:s1",
            "task": "LOCATE",
            "operator_id": "anonymous-1",
            "geometry_version": "OTHER_GEOMETRY",
            "reference_version": "PROXY_LEGACY",
            "completion": True,
            "false_stop": False,
            "measurement_cost": 0.3,
            "route_cost": 0.5,
        },
    )
    model_rows = (
        {
            "specimen_key": "domain:s1",
            "task": "LOCATE",
            "method": "BC_S1",
            "seed": 1,
            "stop_system": "S_BC_CAL",
            "reference_version": "PROXY_LEGACY",
            "completion": True,
            "false_stop": False,
            "failure_penalized_cost": 0.2,
            "prefix_route_cost": 0.4,
        },
    )

    result = match_human_sessions(
        sessions,
        model_rows=model_rows,
        expected_geometry="FROZEN_NATIVE_8X8_THREE_LEVEL",
    )

    assert result["comparability_manifest"][0]["status"] == "UNMATCHED_DATA"
    assert result["matched_method_comparison"] == ()


def test_human_model_pool_retains_proxy_rows_with_partial_reviewed_coverage() -> None:
    from cmc_bbdm.learned_cscan.frozen_evidence_finalize import _human_model_rows

    proxy = (
        {
            "specimen_key": "domain:proxy",
            "reference_version": "PROXY_LEGACY",
            "report_id": "proxy-report",
        },
    )
    reviewed = (
        {
            "specimen_key": "domain:reviewed",
            "reference_version": "REVIEWED_X",
            "report_id": "reviewed-report",
        },
    )

    rows = _human_model_rows(proxy, reviewed)

    assert rows == proxy + reviewed


def test_human_matching_excludes_ablations_and_keeps_cost_units_separate() -> None:
    sessions = (
        {
            "session_id": "h1",
            "specimen_key": "domain:s1",
            "task": "LOCATE",
            "operator_id": "anonymous-1",
            "geometry_version": "FROZEN_NATIVE_8X8_THREE_LEVEL",
            "reference_version": "REVIEWED_X",
            "completion": True,
            "false_stop": False,
            "measurement_cost": 0.3,
            "cost_unit": "NATIVE_RASTER_FRACTION",
            "route_cost": 0.5,
        },
    )
    common = {
        "specimen_key": "domain:s1",
        "task": "LOCATE",
        "seed": 1,
        "stop_system": "S_BC_CAL",
        "reference_version": "REVIEWED_X",
        "completion": True,
        "false_stop": False,
        "prefix_measurement_cost": 0.2,
        "failure_penalized_cost": 0.2,
        "prefix_route_cost": 0.4,
    }
    model_rows = (
        {**common, "method": "R_BALANCED_P8", "seed": 8},
        {**common, "method": "BC_S1"},
        {**common, "method": "BC_NO_VLM_S1"},
        {**common, "method": "BC_NO_US_FEEDBACK_S1"},
    )

    result = match_human_sessions(
        sessions,
        model_rows=model_rows,
        expected_geometry="FROZEN_NATIVE_8X8_THREE_LEVEL",
    )

    assert result["comparability_manifest"][0]["cost_comparable"] is True
    assert result["comparability_manifest"][0]["route_cost_comparable"] is False
    assert result["comparability_manifest"][0]["information_permission"] == (
        "NOT_RECORDED"
    )
    assert result["comparability_manifest"][0]["report_production"] == "NOT_RECORDED"
    assert result["comparability_manifest"][0]["action_sequence_recorded"] is False
    assert result["comparability_manifest"][0]["first_stop_or_handoff_recorded"] is False
    assert result["comparability_manifest"][0]["final_report_recorded"] is False
    assert result["comparability_manifest"][0]["process_trace_comparable"] is False
    assert result["comparability_manifest"][0]["pure_planner_comparable"] is False
    assert {row["model_method"] for row in result["matched_method_comparison"]} == {
        "R_BALANCED_P8",
        "BC_S1",
    }
    assert all(
        row["comparison_scope"] == "OUTCOME_AND_MEASUREMENT_COST"
        and row["analysis_scope"] == "MATCHED_HUMAN_FULL_SYSTEM_COMPARISON"
        and row["pure_planner_comparable"] is False
        and row["human_measurement_cost"] == 0.3
        and row["model_measurement_cost"] == 0.2
        and row["model_failure_penalized_cost"] == 0.2
        and row["human_route_cost"] is None
        and row["model_route_cost"] is None
        for row in result["matched_method_comparison"]
    )


def test_cli_has_only_analyze_and_optional_input_finalize_commands() -> None:
    parser = build_parser()

    analyze = parser.parse_args(
        ["analyze", "--config", str(CONFIG), "--source-root", "/source"]
    )
    finalize = parser.parse_args(
        ["finalize", "--config", str(CONFIG), "--source-root", "/source"]
    )

    assert analyze.command == "analyze"
    assert finalize.command == "finalize"
    assert finalize.references is None
    assert finalize.blind_reviews is None
    assert finalize.human_sessions is None
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])


def test_finalize_without_external_inputs_is_explicitly_pending(tmp_path: Path) -> None:
    config = load_frozen_process_config(CONFIG, project_root=ROOT)
    trajectories = pl.read_parquet(
        config.source_result_root / "trajectories.parquet"
    )
    tables = analyze_trajectory_tables(config, trajectories=trajectories)

    from cmc_bbdm.learned_cscan.frozen_evidence_finalize import (
        finalize_frozen_evidence,
    )
    from cmc_bbdm.learned_cscan.frozen_process_analysis import (
        _write_analysis_documents,
        _write_finalize_documents,
    )

    finalized = finalize_frozen_evidence(
        config,
        source_root=Path("/home/ww/paper3/cmc_damage_inference"),
        trajectories=trajectories,
        proxy_stop_rows=tables["first_stop_decomposition"],
    )

    assert finalized["reference_input_status"] == "PENDING_USER_INPUT"
    assert len(finalized["reference_coverage"]) == 24
    assert {row["status"] for row in finalized["reference_coverage"]} == {
        "PENDING_USER_INPUT"
    }
    assert finalized["reviewed_task_decisions"] == {
        "LOCATE": {"status": "PENDING_INPUT"},
        "CHARACTERIZE": {"status": "PENDING_INPUT"},
    }
    assert (
        finalized["final_evidence_manifest"]["stage"]
        == "COMPUTATION_COMPLETE_INPUT_PENDING"
    )
    scope_claim = next(
        row for row in finalized["claim_rows"] if row["claim_id"] == "C8_SCOPE_BOUNDARIES"
    )
    assert scope_claim["exact_claim"].startswith("Current evidence does not establish")
    document_config = replace(config, artifact_root=tmp_path)
    _write_finalize_documents(document_config, finalized)
    handoff = (tmp_path / "CODEX_HANDOFF_FROZEN_PROCESS_ANALYSIS.md").read_text()
    assert "## External Input Identities" in handoff
    assert "## Main Frozen Recovery" in handoff
    assert "## Planning-to-Stop Findings" in handoff
    assert "## Report Stability and STOP Delay" in handoff
    assert "## Action-Cost Allocation" in handoff
    assert "## Output Inventory" in handoff
    assert "36,864" in handoff
    summary = json.loads(
        (config.output_root / "analysis_summary.json").read_text(encoding="utf-8")
    )
    _write_analysis_documents(document_config, summary=summary, figure_checks=())
    bindings = (tmp_path / "INPUT_AND_CODE_BINDINGS.md").read_text()
    assert "## Trajectory Columns and Dtypes" in bindings
    assert "`report_sha256`: `String`" in bindings
    assert "## Function Bindings" in bindings
    assert "`recover_reviewed_report_scores`" in bindings
    assert "`finalize_frozen_evidence`" in bindings
