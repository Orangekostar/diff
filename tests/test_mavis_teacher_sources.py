from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.mavis.authority import load_mavis_authority
from cmc_bbdm.mavis.config import load_mavis_config
from cmc_bbdm.mavis.teacher import load_registered_initial_embeddings
from cmc_bbdm.mva.a4_candidate_bank import load_candidate_bank

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/home/ww/paper3/cmc_damage_inference")
CONFIG = ROOT / "paper_v3/configs/mavis_development.yaml"


@pytest.mark.slow
def test_mavis_initial_teacher_embeddings_use_registered_domain_scouts() -> None:
    config = load_mavis_config(CONFIG, project_root=ROOT)
    authority = load_mavis_authority(config, source_project_root=SOURCE_ROOT)

    roster = load_registered_initial_embeddings(
        config,
        authority,
        project_root=ROOT,
    )

    assert roster.specimen_ids == authority.specimen_ids
    assert roster.dataset_ids == authority.dataset_ids
    assert roster.embeddings.shape == (276, 512)
    assert not roster.embeddings.flags.writeable
    assert set(roster.bank_state_sha256) == {0.015625, 0.03125}
    for domain, budget in config.initial_budget_by_domain.items():
        assert {
            roster.initial_budgets[index]
            for index, item in enumerate(roster.dataset_ids)
            if item == domain
        } == {budget}

    low_bank = load_candidate_bank(
        ROOT / config.sources["candidate_bank_0p015625"].path
    )
    high_bank = load_candidate_bank(
        ROOT / config.sources["candidate_bank_0p03125"].path
    )
    low_index = next(
        index
        for index, domain in enumerate(authority.dataset_ids)
        if domain != "74t7kcdgkr"
    )
    high_index = authority.dataset_ids.index("74t7kcdgkr")
    np.testing.assert_array_equal(
        roster.embeddings[low_index], low_bank.initial_embeddings[low_index]
    )
    np.testing.assert_array_equal(
        roster.embeddings[high_index], high_bank.initial_embeddings[high_index]
    )
