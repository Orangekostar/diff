from __future__ import annotations

from pathlib import Path

from cmc_bbdm.inspection_agent_g1.warm_start import audit_deployment_geometry

ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT / "results/inspection_agent/g0/authorized_roster.csv"


def test_authorized_roster_requires_a_geometry_only_bridge() -> None:
    audit = audit_deployment_geometry(ROSTER)
    assert audit.status == "G1_DEPLOYMENT_GEOMETRY_GO"
    assert audit.specimen_count == 276
    assert audit.native_shapes == ((338, 340), (338, 352), (674, 675))
    assert audit.domain_dependent_g0_budget is True
    assert audit.selected_nominal_budget == 0.015625
    assert len(audit.preset_records) == 10
    assert sum(record.count for record in audit.preset_records) == 276


def test_every_registered_candidate_is_valid_for_every_geometry() -> None:
    audit = audit_deployment_geometry(ROSTER)
    assert len(audit.grid_candidates) == 9
    assert all(candidate.valid for candidate in audit.grid_candidates)
    assert {
        candidate.nominal_budget for candidate in audit.grid_candidates
    } == {0.015625, 0.03125, 0.0625}
    assert len({candidate.grid_sha256 for candidate in audit.grid_candidates}) == 9


def test_same_visible_geometry_maps_to_both_hidden_g0_presets() -> None:
    audit = audit_deployment_geometry(ROSTER)
    by_shape: dict[tuple[int, int], set[float]] = {}
    for record in audit.preset_records:
        by_shape.setdefault(record.native_shape, set()).add(record.g0_nominal_budget)
    assert by_shape == {
        (338, 340): {0.015625, 0.03125},
        (338, 352): {0.015625, 0.03125},
        (674, 675): {0.015625, 0.03125},
    }
    assert len(audit.state_sha256) == 64
