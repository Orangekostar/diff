from __future__ import annotations

import numpy as np
from test_mvd_config import CONFIG, ROOT

from cmc_bbdm.mvd.authority import load_compact_mvd_authority
from cmc_bbdm.mvd.config import load_mvd_config
from cmc_bbdm.mvd.initial_value_dataset import build_source_initial_value_dataset


def test_mvd_binds_both_readonly_candidate_banks() -> None:
    config = load_mvd_config(CONFIG, project_root=ROOT)
    authority = load_compact_mvd_authority(config, project_root=ROOT)

    assert authority.specimen_count == 276
    assert set(authority.candidate_banks) == {0.015625, 0.03125}
    for budget, bank in authority.candidate_banks.items():
        assert bank.state_sha256 == config.candidate_bank_states[budget]
        assert bank.authority_state_sha256 == config.authority_state_sha256
        assert bank.initial_embeddings.shape == (276, 512)
        assert bank.embeddings.shape == (276, 64, 512)
        assert bank.added_measurements.shape == (276, 64)
        assert not bank.initial_embeddings.flags.writeable
        assert not bank.embeddings.flags.writeable
        assert not bank.added_measurements.flags.writeable
        assert np.all(bank.added_measurements > 0)


def test_source_initial_value_dataset_aligns_complete_values_and_embeddings() -> None:
    config = load_mvd_config(CONFIG, project_root=ROOT)
    authority = load_compact_mvd_authority(config, project_root=ROOT)
    dataset = build_source_initial_value_dataset(
        authority, outer_domain="74t7kcdgkr"
    )

    assert dataset.outer_domain == "74t7kcdgkr"
    assert dataset.initial_budget == 0.03125
    assert dataset.specimen_count == 231
    assert "74t7kcdgkr" not in dataset.dataset_ids
    assert dataset.initial_embeddings.shape == (231, 512)
    assert dataset.candidate_embeddings.shape == (231, 64, 512)
    assert dataset.mechanical_values.shape == (231, 64)
    assert dataset.candidate_costs.shape == (231, 64)
    assert dataset.current_predictions.shape == (231,)
    assert len(dataset.predictor_state_sha256) == 231
    assert not dataset.initial_embeddings.flags.writeable
    assert not dataset.candidate_embeddings.flags.writeable
    assert not dataset.mechanical_values.flags.writeable
    assert not dataset.candidate_costs.flags.writeable
    assert not dataset.current_predictions.flags.writeable

    first = authority.source_values.filter(
        (authority.source_values["outer_domain"] == dataset.outer_domain)
        & (authority.source_values["specimen_id"] == dataset.specimen_ids[0])
        & (authority.source_values["method"] == "global_mechanical_mask")
    ).sort("cell_index")
    assert np.array_equal(
        dataset.mechanical_values[0], first["primary_value"].to_numpy()
    )
    assert np.array_equal(
        dataset.candidate_costs[0], first["added_measurements"].to_numpy()
    )
