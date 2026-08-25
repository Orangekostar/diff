from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.mavis.authority import load_mavis_authority
from cmc_bbdm.mavis.config import load_mavis_config

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/home/ww/paper3/cmc_damage_inference")
CONFIG = ROOT / "paper_v3/configs/mavis_development.yaml"


@pytest.mark.slow
def test_registered_mavis_authority_loads_exact_causal_cohort() -> None:
    config = load_mavis_config(CONFIG, project_root=ROOT)
    authority = load_mavis_authority(config, source_project_root=SOURCE_ROOT)

    assert authority.specimen_count == 276
    assert tuple(dict.fromkeys(authority.dataset_ids)) == config.domain_order
    assert authority.source_authority_sha256 == (
        "ae03bf35f5b32665007e7f928ee4e2ed098083a4d0982c73100307886acee394"
    )
    assert len(authority.source_image_sha256) == 276
    assert len(set(authority.source_image_sha256)) == 276
    context = authority.policy_context(authority.specimen_ids[0])
    teacher = authority.source_teacher_view(authority.specimen_ids[0])
    assert context.context_features.shape == (34,)
    np.testing.assert_array_equal(
        context.context_features[:13], teacher.policy_context.context_features[:13]
    )
    assert teacher.source_image_sha256 == authority.source_image_sha256[0]
    assert teacher.source_image_sha256 != authority.decoded_image_sha256[0]
