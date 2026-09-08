from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.learned_cscan.bc_supplement import (
    FROZEN_BC_SHA256,
    load_supplement_config,
    planned_checkpoint_paths,
    verify_frozen_file,
)
from cmc_bbdm.learned_cscan.contracts import Task
from cmc_bbdm.learned_cscan.episode_stop_calibration import (
    calibrate_episode_stop,
    first_stop_outcome,
    summarize_episode_risk,
)
from cmc_bbdm.learned_cscan.metrics import MetricRecord
from cmc_bbdm.learned_cscan.readout import TaskReportV2
from cmc_bbdm.learned_cscan.supplement_adapters import (
    ActorInputMode,
    adapt_task_report_v2,
    mask_actor_tensors,
)
from cmc_bbdm.learned_cscan.supplement_analysis import (
    cost_at_success_rate,
    make_domain_bootstrap_draws,
    paired_domain_bootstrap,
)
from cmc_bbdm.vlm_cscan.contracts import BenchmarkTask, TaskReport

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/bc_cscan_path_b_supplement.yaml"
PARENT_CONFIG_SHA256 = (
    "12268dcaf470f769c326007702f7b0b6f4a13cf326447eef652a6dda972b5662"
)


def test_config_separates_frozen_source_and_destination() -> None:
    config = load_supplement_config(CONFIG, project_root=ROOT)

    assert config.source_result_root != config.output_root
    assert config.parent_config_sha256 == PARENT_CONFIG_SHA256
    assert config.actor_update_cap == 16_000
    assert config.source_result_root not in config.output_root.parents
    assert config.output_root not in config.source_result_root.parents


def test_changed_frozen_bc_is_rejected(tmp_path: Path) -> None:
    changed = tmp_path / "l_bc.pt"
    changed.write_bytes(b"changed")

    with pytest.raises(ValueError, match="frozen BC"):
        verify_frozen_file(changed, FROZEN_BC_SHA256, label="frozen BC")


def test_no_us_mask_preserves_geometry_and_removes_content() -> None:
    tensors = {
        "cell_features": np.ones((64, 17), dtype=np.float32),
        "subblock_features": np.ones((64, 16, 10), dtype=np.float32),
        "global_features": np.ones(9, dtype=np.float32),
        "history_features": np.ones((4, 5), dtype=np.float32),
        "legal_mask": np.ones(64, dtype=np.bool_),
    }

    masked = mask_actor_tensors(tensors, ActorInputMode.NO_US_FEEDBACK)

    np.testing.assert_array_equal(
        masked["cell_features"][:, :4], tensors["cell_features"][:, :4]
    )
    assert not masked["cell_features"][:, 4:15].any()
    np.testing.assert_array_equal(
        masked["cell_features"][:, 15:17],
        tensors["cell_features"][:, 15:17],
    )
    assert not masked["subblock_features"][:, :, 2:10].any()
    assert masked["global_features"][7] == 0.0
    assert tensors["cell_features"][0, 4] == 1.0
    assert not np.shares_memory(
        masked["cell_features"], tensors["cell_features"]
    )


def test_review_adapter_preserves_mask_support_and_formal_scope() -> None:
    prediction = np.zeros((5, 6), dtype=np.bool_)
    prediction[1:3, 2:4] = True
    support = np.asarray([[1, 2], [2, 3]], dtype=np.int64)
    report = TaskReportV2(
        task=Task.CHARACTERIZE,
        predicted_mask=prediction,
        support_positions=support,
        candidate_cells=(10,),
        unverified_boundary_cells=(),
        signal_strength=0.75,
        reason_code="VISIBLE_CANDIDATE",
    )

    adapted = adapt_task_report_v2(report)

    assert type(adapted) is TaskReport
    assert adapted.task is BenchmarkTask.CHARACTERIZE
    np.testing.assert_array_equal(adapted.predicted_mask, prediction)
    np.testing.assert_array_equal(adapted.support_positions, support)
    assert adapted.confidence == 0.75


def test_first_false_stop_is_not_replaced_by_later_success() -> None:
    rows = (
        {
            "step": 0,
            "cost": 0.0,
            "success": False,
            "eligible": False,
            "p": 0.99,
        },
        {
            "step": 1,
            "cost": 0.2,
            "success": False,
            "eligible": True,
            "p": 0.91,
        },
        {
            "step": 2,
            "cost": 0.4,
            "success": True,
            "eligible": True,
            "p": 0.99,
        },
    )

    outcome = first_stop_outcome(
        rows,
        probability_field="p",
        eligibility_field="eligible",
        threshold=0.90,
    )

    assert outcome.false_stop and not outcome.completed
    assert outcome.stop_cost == 0.2
    assert outcome.failure_penalized_cost == 1.0


def test_never_stopping_is_exhaustion() -> None:
    rows = (
        {
            "step": 0,
            "cost": 0.0,
            "success": False,
            "eligible": False,
            "p": 0.2,
        },
        {
            "step": 1,
            "cost": 1.0,
            "success": True,
            "eligible": True,
            "p": 0.98,
        },
    )

    outcome = first_stop_outcome(
        rows,
        probability_field="p",
        eligibility_field="eligible",
        threshold=0.99,
    )

    assert outcome.exhausted and not outcome.stopped
    assert outcome.failure_penalized_cost == 1.0


def test_wrong_among_stops_uses_actual_stops_and_shared_threshold() -> None:
    false = first_stop_outcome(
        (
            {"step": 0, "cost": 0.2, "success": False, "eligible": True, "p": 1.0},
        ),
        probability_field="p",
        eligibility_field="eligible",
        threshold=0.9,
    )
    correct = first_stop_outcome(
        (
            {"step": 0, "cost": 0.2, "success": True, "eligible": True, "p": 1.0},
        ),
        probability_field="p",
        eligibility_field="eligible",
        threshold=0.9,
    )
    exhausted = first_stop_outcome(
        (
            {"step": 0, "cost": 1.0, "success": True, "eligible": True, "p": 0.0},
        ),
        probability_field="p",
        eligibility_field="eligible",
        threshold=0.9,
    )
    risk = summarize_episode_risk((false, correct, exhausted))
    assert risk.wrong_among_stops == 0.5
    assert risk.false_stop_episode_rate == pytest.approx(1 / 3)

    rows = []
    for planner in ("BC_S1", "R_BALANCED_P8"):
        for index in range(6):
            rows.extend(
                (
                    {
                        "specimen_key": f"domain:s{index}",
                        "task": "LOCATE",
                        "planner": planner,
                        "step": 0,
                        "cost": 0.2,
                        "success": index != 0,
                        "eligible": True,
                        "p": 0.91,
                    },
                    {
                        "specimen_key": f"domain:s{index}",
                        "task": "LOCATE",
                        "planner": planner,
                        "step": 1,
                        "cost": 0.4,
                        "success": True,
                        "eligible": True,
                        "p": 0.96,
                    },
                )
            )
    calibration = calibrate_episode_stop(
        tuple(rows),
        task="LOCATE",
        planners=("BC_S1", "R_BALANCED_P8"),
        rule_completion={"BC_S1": 1.0, "R_BALANCED_P8": 1.0},
        probability_field="p",
        eligibility_field="eligible",
    )
    assert calibration.qualified
    assert calibration.selected_threshold == 0.95


def test_cost_at_success_rate_keeps_nonmonotone_current_reports() -> None:
    rows = (
        {"dataset_id": "d", "specimen_key": "d:a", "seed": 1, "step": 0, "cost": 0.0, "success": False},
        {"dataset_id": "d", "specimen_key": "d:b", "seed": 1, "step": 0, "cost": 0.0, "success": False},
        {"dataset_id": "d", "specimen_key": "d:a", "seed": 1, "step": 1, "cost": 0.2, "success": True},
        {"dataset_id": "d", "specimen_key": "d:b", "seed": 1, "step": 1, "cost": 0.2, "success": True},
        {"dataset_id": "d", "specimen_key": "d:a", "seed": 1, "step": 2, "cost": 0.4, "success": False},
        {"dataset_id": "d", "specimen_key": "d:b", "seed": 1, "step": 2, "cost": 0.4, "success": False},
    )

    assert cost_at_success_rate(rows, target=1.0) == 0.2


def test_paired_bootstrap_averages_seeds_within_physical_specimen() -> None:
    records = []
    for domain, values in {"d1": (1.0, 3.0), "d2": (2.0, 4.0)}.items():
        for index, effect in enumerate(values):
            specimen = f"{domain}:s{index}"
            records.extend(
                (
                    MetricRecord(specimen, domain, "BC", Task.LOCATE, 1, 100.0 + effect),
                    MetricRecord(specimen, domain, "BC", Task.LOCATE, 2, 102.0 + effect),
                    MetricRecord(specimen, domain, "RULE", Task.LOCATE, 1, 101.0),
                )
            )
    domains = {row.specimen_key: row.domain for row in records}
    draws = make_domain_bootstrap_draws(domains, replicates=128, seed=7)

    effect = paired_domain_bootstrap(
        tuple(records),
        treatment="BC",
        comparator="RULE",
        confidence_level=0.95,
        draws=draws,
    )

    assert effect.estimate == pytest.approx(2.5)
    assert effect.physical_specimen_count == 4
    assert effect.domain_count == 2


def test_supplement_checkpoint_paths_cannot_overwrite_frozen_models() -> None:
    config = load_supplement_config(CONFIG, project_root=ROOT)

    for path in planned_checkpoint_paths(config):
        assert config.output_root in path.parents
        assert config.source_result_root not in path.parents
        assert path != config.frozen_bc_path
