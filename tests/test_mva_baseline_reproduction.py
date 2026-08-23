from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.mva.authority import load_mva_authority, reproduce_full_baseline
from cmc_bbdm.mva.config import load_mva_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mva_a0_a3.yaml"


@pytest.fixture(scope="module")
def authority_and_baseline() -> tuple[object, object]:
    config = load_mva_config(CONFIG, project_root=ROOT)
    authority = load_mva_authority(config, project_root=ROOT)
    baseline = reproduce_full_baseline(config, authority)
    return authority, baseline


def test_mva_authority_binds_exact_p1_cohort(
    authority_and_baseline: tuple[object, object],
) -> None:
    authority, _baseline = authority_and_baseline

    assert authority.specimen_count == 276
    assert len(set(authority.specimen_ids)) == 276
    assert tuple(dict.fromkeys(authority.dataset_ids)) == (
        "74t7kcdgkr",
        "cgtnjyggtm",
        "w68dtmpfyf",
        "xcmzfsbd9t",
        "yfxyg8jm46",
        "ykhs7s2dck",
    )
    assert authority.targets.shape == (276,)
    assert authority.metadata13.shape == (276, 13)
    assert authority.full_embeddings.shape == (276, 512)
    assert len(authority.images) == 276
    assert {image.shape for image in authority.images} == {
        (674, 675, 3),
        (338, 352, 3),
        (338, 340, 3),
    }
    assert all(image.dtype == np.uint8 for image in authority.images)
    assert not authority.targets.flags.writeable
    assert not authority.metadata13.flags.writeable
    assert not authority.full_embeddings.flags.writeable
    assert all(not image.flags.writeable for image in authority.images)


def test_fresh_full_nested_lodo_exactly_reproduces_registered_p1(
    authority_and_baseline: tuple[object, object],
) -> None:
    authority, baseline = authority_and_baseline

    assert baseline.specimen_ids == authority.specimen_ids
    assert baseline.dataset_ids == authority.dataset_ids
    assert baseline.selected_pca_dimensions == (8, 32, 8, 8, 8, 8)
    assert baseline.equal_domain_mae == pytest.approx(0.08963580465761432, abs=1.0e-12)
    assert baseline.maximum_prediction_delta <= 1.0e-12
    assert np.array_equal(baseline.targets, authority.targets)
    assert not baseline.predictions.flags.writeable
    assert len(baseline.fit_records) == 96


def test_every_outer_and_inner_fit_excludes_its_query_domain(
    authority_and_baseline: tuple[object, object],
) -> None:
    _authority, baseline = authority_and_baseline

    for record in baseline.fit_records:
        assert set(record.query_domains).isdisjoint(record.fit_domains)
        assert set(record.query_specimen_ids).isdisjoint(record.fit_specimen_ids)
        if record.stage == "outer":
            assert len(set(record.fit_domains)) == 5
        else:
            assert len(set(record.fit_domains)) == 4
