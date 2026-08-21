from __future__ import annotations

import inspect
import math
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from cmc_bbdm.cpb_diffusion_marginalization.authority import (
    issue_inner_fold,
    issue_search_view,
)
from cmc_bbdm.cpb_diffusion_marginalization.config import load_d8_config
from cmc_bbdm.cpb_diffusion_marginalization.residual_config import (
    load_residual_diffusion_config,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_targets import (
    build_fit_residual_target_batch,
    build_outer_fit_residual_target_batch,
    load_pilot_diffusion_scaffolds,
    load_search_residual_field_bank,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_training import (
    ResidualFinalTrainingResult,
    ResidualTrainingError,
    train_inner_residual_model,
    train_outer_fit_residual_model,
)
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLORATION_CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_exploration.yaml"
RESIDUAL_CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_residual_diffusion.yaml"
OUTER = "74t7kcdgkr"
QUERY = "cgtnjyggtm"


@lru_cache(maxsize=1)
def _training_authority():
    exploration = load_d8_config(EXPLORATION_CONFIG, project_root=PROJECT_ROOT)
    residual = load_residual_diffusion_config(
        RESIDUAL_CONFIG,
        project_root=PROJECT_ROOT,
    )
    v3_config = load_v3_config(
        PROJECT_ROOT / exploration.sources["p1_config"].path,
        project_root=PROJECT_ROOT,
    )
    data = load_data(v3_config, PROJECT_ROOT)
    search = issue_search_view(data, outer_domain=OUTER, config=exploration)
    inner = issue_inner_fold(search, query_domain=QUERY)
    field_bank = load_search_residual_field_bank(search, project_root=PROJECT_ROOT)
    scaffold = load_pilot_diffusion_scaffolds(
        residual,
        project_root=PROJECT_ROOT,
    )[OUTER]
    target_batch = build_fit_residual_target_batch(
        inner,
        scaffold,
        field_bank=field_bank,
    )
    return residual, search, inner, scaffold, field_bank, target_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="registered training uses CUDA")
def test_one_epoch_training_is_fit_only_and_byte_replayable() -> None:
    residual, _search, inner, _scaffold, _field_bank, target_batch = (
        _training_authority()
    )
    candidate = residual.candidate("RD0")
    kwargs = {
        "config": residual,
        "candidate": candidate,
        "epochs": 1,
        "seed": 20260823,
        "device": "cuda:0",
        "test_scale_override": True,
    }

    first = train_inner_residual_model(inner, target_batch, **kwargs)
    second = train_inner_residual_model(inner, target_batch, **kwargs)

    assert first.outer_domain == OUTER
    assert first.query_domain == QUERY
    assert first.fit_specimen_ids == inner.fit_specimen_ids
    assert first.fit_dataset_ids == inner.fit_dataset_ids
    assert not set(first.fit_specimen_ids) & set(inner.query_specimen_ids)
    assert OUTER not in first.fit_dataset_ids
    assert QUERY not in first.fit_dataset_ids
    assert first.response_read_count == 0
    assert first.sample_count == len(inner.fit_indices)
    assert first.batch_count == math.ceil(first.sample_count / residual.batch_size)
    assert len(first.epoch_losses) == 1
    assert all(
        math.isfinite(value) and value >= 0.0
        for value in (
            first.epoch_losses[0].total,
            first.epoch_losses[0].diffusion,
            first.epoch_losses[0].spectral,
            first.epoch_losses[0].low_pass,
        )
    )
    assert first.checkpoint.split_sha256 == inner.state_sha256
    assert first.checkpoint.config_sha256 == residual.config_sha256
    assert first.checkpoint.state_dict_sha256 == second.checkpoint.state_dict_sha256
    assert first.epoch_losses == second.epoch_losses
    assert first.state_sha256 == second.state_sha256


def test_training_api_has_no_query_array_or_response_input() -> None:
    parameters = inspect.signature(train_inner_residual_model).parameters
    assert "response" not in parameters
    assert "cai" not in parameters
    assert "query" not in parameters
    assert "query_targets" not in parameters


def test_outer_fit_training_api_has_no_query_or_response_input() -> None:
    parameters = inspect.signature(train_outer_fit_residual_model).parameters
    assert "response" not in parameters
    assert "cai" not in parameters
    assert "query" not in parameters
    assert "query_targets" not in parameters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="registered training uses CUDA")
def test_outer_fit_training_consumes_all_five_search_domains() -> None:
    residual, search, _inner, scaffold, field_bank, _target_batch = (
        _training_authority()
    )
    target_batch = build_outer_fit_residual_target_batch(
        search,
        scaffold,
        field_bank=field_bank,
    )

    result = train_outer_fit_residual_model(
        search,
        target_batch,
        config=residual,
        candidate=residual.candidate("RD0"),
        epochs=1,
        seed=residual.training_seeds[0],
        device="cuda",
        test_scale_override=True,
    )

    assert type(result) is ResidualFinalTrainingResult
    assert result.outer_domain == search.outer_domain
    assert result.fit_specimen_ids == search.specimen_ids
    assert result.fit_dataset_ids == search.dataset_ids
    assert tuple(dict.fromkeys(result.fit_dataset_ids)) == tuple(
        domain for domain in load_d8_config(
            EXPLORATION_CONFIG,
            project_root=PROJECT_ROOT,
        ).outer_domains
        if domain != search.outer_domain
    )
    assert result.response_read_count == 0
    assert result.checkpoint.split_sha256 == search.state_sha256


def test_training_rejects_a_target_batch_from_another_inner_fold() -> None:
    residual, search, inner, scaffold, field_bank, _target_batch = (
        _training_authority()
    )
    other = issue_inner_fold(search, query_domain="w68dtmpfyf")
    other_batch = build_fit_residual_target_batch(
        other,
        scaffold,
        field_bank=field_bank,
    )

    with pytest.raises(ResidualTrainingError, match="authority"):
        train_inner_residual_model(
            inner,
            other_batch,
            config=residual,
            candidate=residual.candidate("RD0"),
            epochs=1,
            seed=20260823,
            device="cuda" if torch.cuda.is_available() else "cpu",
            test_scale_override=True,
        )
