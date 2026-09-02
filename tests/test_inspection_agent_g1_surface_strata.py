from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cmc_bbdm.inspection_agent_g1.surface_strata import (
    FROZEN_G0_INITIALIZATION_CURVES_SHA256,
    G1SurfaceStratumError,
    load_g1_frozen_surface_strata,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "results/inspection_agent/g0/initialization_curves.csv"


def test_frozen_g0_surface_strata_replay_exactly() -> None:
    authority, rows = load_g1_frozen_surface_strata(AUTHORITY)

    assert len(rows) == authority.record_count == 276
    assert authority.file_sha256 == FROZEN_G0_INITIALIZATION_CURVES_SHA256
    assert authority.stratum_counts == (
        ("SURFACE_INTERNAL_AGREE", 1),
        ("SURFACE_INTERNAL_PARTIAL", 114),
        ("SURFACE_INTERNAL_MISLEADING", 161),
    )
    assert len({(row.outer_target, row.specimen_id) for row in rows}) == 276
    assert hashlib.sha256(AUTHORITY.read_bytes()).hexdigest() == authority.file_sha256


def test_frozen_g0_surface_strata_rejects_changed_authority(tmp_path: Path) -> None:
    changed = tmp_path / "initialization_curves.csv"
    changed.write_bytes(AUTHORITY.read_bytes() + b"\n")

    with pytest.raises(G1SurfaceStratumError, match="SHA"):
        load_g1_frozen_surface_strata(changed)
