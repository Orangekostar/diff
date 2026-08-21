from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.cpb_diffusion_marginalization.authority import (
    issue_inner_fold,
    issue_search_view,
)
from cmc_bbdm.cpb_diffusion_marginalization.config import load_d8_config
from cmc_bbdm.cpb_diffusion_marginalization.residual_config import (
    load_residual_diffusion_config,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_targets import (
    ResidualTargetError,
    build_fit_residual_target_batch,
    build_outer_fit_residual_target_batch,
    load_pilot_diffusion_scaffolds,
    load_search_residual_field_bank,
    residual_replacement_perturbations,
)
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLORATION_CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_exploration.yaml"
RESIDUAL_CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_residual_diffusion.yaml"
OUTER = "74t7kcdgkr"
QUERY = "cgtnjyggtm"


@lru_cache(maxsize=1)
def _registered_authorities():
    exploration = load_d8_config(EXPLORATION_CONFIG, project_root=PROJECT_ROOT)
    residual = load_residual_diffusion_config(
        RESIDUAL_CONFIG, project_root=PROJECT_ROOT
    )
    v3_config = load_v3_config(
        PROJECT_ROOT / exploration.sources["p1_config"].path,
        project_root=PROJECT_ROOT,
    )
    data = load_data(v3_config, PROJECT_ROOT)
    search = issue_search_view(data, outer_domain=OUTER, config=exploration)
    inner = issue_inner_fold(search, query_domain=QUERY)
    scaffolds = load_pilot_diffusion_scaffolds(
        residual, project_root=PROJECT_ROOT
    )
    field_bank = load_search_residual_field_bank(
        search, project_root=PROJECT_ROOT
    )
    return residual, search, inner, scaffolds, field_bank


def test_pilot_diffusion_scaffolds_cover_six_outers_and_escalation_rows() -> None:
    residual, _search, _inner, scaffolds, _field_bank = _registered_authorities()
    assert tuple(scaffolds) == (
        "74t7kcdgkr",
        "cgtnjyggtm",
        "w68dtmpfyf",
        "xcmzfsbd9t",
        "yfxyg8jm46",
        "ykhs7s2dck",
    )
    assert all(item.control_id in {"B5", "B6", "B7", "B8"} for item in scaffolds.values())
    assert scaffolds["cgtnjyggtm"].candidate_sha256 == (
        "fd9d02b05231e1f217668bf5edf6d01c85d8fd507cd5172345ac4dd501957573"
    )
    assert scaffolds[OUTER].decomposition_family == "gaussian"
    assert scaffolds[OUTER].selected_band == "high"
    assert scaffolds[OUTER].decomposition_parameters == {
        "band": "high",
        "sigma": 2.824543350191041,
    }
    assert all(item.config_sha256 == residual.sources["exploration_config"].sha256 for item in scaffolds.values())


def test_fit_residual_target_round_trip_and_scale_are_exact() -> None:
    _residual, _search, inner, scaffolds, field_bank = _registered_authorities()
    batch = build_fit_residual_target_batch(
        inner,
        scaffolds[OUTER],
        field_bank=field_bank,
    )
    np.testing.assert_allclose(
        batch.stable + batch.residual,
        batch.measured,
        rtol=0.0,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        batch.training_target * np.float32(2.0),
        batch.residual,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        batch.stable_condition,
        np.clip(batch.stable, -1.0, 1.0).astype(np.float32),
    )
    assert batch.measured.shape == (len(inner.fit_indices), 3, 64, 64)
    assert batch.training_target.min() >= -1.0
    assert batch.training_target.max() <= 1.0
    assert batch.stable_condition.min() >= -1.0
    assert batch.stable_condition.max() <= 1.0
    assert all(not value.flags.writeable for value in (
        batch.measured,
        batch.stable,
        batch.stable_condition,
        batch.residual,
        batch.training_target,
    ))


def test_fit_target_contains_only_four_fit_domains_and_no_query_or_outer() -> None:
    _residual, _search, inner, scaffolds, field_bank = _registered_authorities()
    batch = build_fit_residual_target_batch(
        inner,
        scaffolds[OUTER],
        field_bank=field_bank,
    )
    assert set(batch.specimen_ids) == set(inner.fit_specimen_ids)
    assert not set(batch.specimen_ids) & set(inner.query_specimen_ids)
    assert set(batch.dataset_ids) == {
        "w68dtmpfyf",
        "xcmzfsbd9t",
        "yfxyg8jm46",
        "ykhs7s2dck",
    }
    assert OUTER not in batch.dataset_ids
    assert QUERY not in batch.dataset_ids


def test_outer_fit_target_contains_exactly_five_search_domains() -> None:
    _residual, search, _inner, scaffolds, field_bank = _registered_authorities()
    batch = build_outer_fit_residual_target_batch(
        search,
        scaffolds[OUTER],
        field_bank=field_bank,
    )
    assert batch.specimen_ids == search.specimen_ids
    assert set(batch.dataset_ids) == set(search.dataset_ids)
    assert OUTER not in batch.dataset_ids
    assert batch.measured.shape == (search.specimen_count, 3, 64, 64)


def test_target_api_has_no_response_or_query_target_entrypoint() -> None:
    for function in (
        build_fit_residual_target_batch,
        build_outer_fit_residual_target_batch,
    ):
        parameters = inspect.signature(function).parameters
        assert "response" not in parameters
        assert "target" not in parameters
        assert "cai" not in parameters
    import cmc_bbdm.cpb_diffusion_marginalization.residual_targets as module

    assert not hasattr(module, "build_query_residual_target_batch")


def test_sampled_targets_become_exact_residual_replacement_perturbations() -> None:
    observed = np.full((3, 64, 64), 0.25, dtype=np.float32)
    sampled_targets = np.stack(
        (
            np.full_like(observed, 0.20),
            np.full_like(observed, -0.10),
        )
    )

    perturbations = residual_replacement_perturbations(
        sampled_targets,
        observed_residual=observed,
    )

    np.testing.assert_array_equal(
        perturbations,
        sampled_targets * np.float32(2.0) - observed,
    )
    assert perturbations.shape == (2, 3, 64, 64)
    assert not perturbations.flags.writeable


def test_target_builder_rejects_wrong_fold_scaffold_pair() -> None:
    _residual, _search, inner, scaffolds, field_bank = _registered_authorities()
    with pytest.raises(ResidualTargetError, match="outer"):
        build_fit_residual_target_batch(
            inner,
            scaffolds["cgtnjyggtm"],
            field_bank=field_bank,
        )


def test_search_field_bank_is_exactly_the_five_domain_authority() -> None:
    _residual, search, _inner, _scaffolds, field_bank = _registered_authorities()
    assert field_bank.outer_domain == OUTER
    assert field_bank.specimen_ids == search.specimen_ids
    assert field_bank.dataset_ids == search.dataset_ids
    assert field_bank.measured.shape == (search.specimen_count, 3, 64, 64)
    assert OUTER not in field_bank.dataset_ids
    assert not field_bank.measured.flags.writeable
