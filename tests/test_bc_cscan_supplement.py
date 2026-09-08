from __future__ import annotations

from pathlib import Path

import pytest

from cmc_bbdm.learned_cscan.bc_supplement import (
    FROZEN_BC_SHA256,
    load_supplement_config,
    verify_frozen_file,
)

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
