from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.learned_cscan.bc_supplement import (
    FROZEN_BC_SHA256,
    load_supplement_config,
    verify_frozen_file,
)
from cmc_bbdm.learned_cscan.contracts import Task
from cmc_bbdm.learned_cscan.readout import TaskReportV2
from cmc_bbdm.learned_cscan.supplement_adapters import (
    ActorInputMode,
    adapt_task_report_v2,
    mask_actor_tensors,
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
