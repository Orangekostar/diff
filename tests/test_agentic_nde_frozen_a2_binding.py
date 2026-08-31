from __future__ import annotations

from pathlib import Path

from cmc_bbdm.agentic_nde.frozen_bindings import A2_SHA256, bind_frozen_a2

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_a2_initial_mechanical_rows_are_exact() -> None:
    binding = bind_frozen_a2(ROOT)
    assert binding.sha256 == A2_SHA256
    assert binding.specimen_count == 276
    assert binding.initial_row_count == 276 * 64
    assert binding.cell_ids == tuple(range(64))
    assert binding.columns[:4] == ("specimen_id", "dataset_id", "method", "step")


def test_binding_exposes_identity_not_target_values() -> None:
    binding = bind_frozen_a2(ROOT)
    assert not hasattr(binding, "values")
    assert not hasattr(binding, "targets")


def test_binding_proves_post_scout_state_identity_without_values() -> None:
    binding = bind_frozen_a2(ROOT)
    assert binding.from_level == 0
    assert binding.to_level == 1
    assert binding.nominal_checkpoint == 0.0625
    assert len(binding.predictor_state_hashes) == 6
    assert all(len(value) == 64 for value in binding.predictor_state_hashes)
    assert len(binding.identity_state_sha256) == 64
