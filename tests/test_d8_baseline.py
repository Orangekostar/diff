from __future__ import annotations

import copy
import pickle
from pathlib import Path

import pytest

from cmc_bbdm.cpb_diffusion_marginalization.authority import (
    D8AuthorityError,
    D8SearchView,
    issue_evaluation_view,
    issue_inner_fold,
    issue_search_view,
    validate_inner_fold,
    validate_search_view,
)
from cmc_bbdm.cpb_diffusion_marginalization.baseline import (
    reproduce_internal_only_baseline,
)
from cmc_bbdm.cpb_diffusion_marginalization.config import load_d8_config
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
D8_CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_exploration.yaml"
P1_CONFIG = PROJECT_ROOT / "paper_v3/configs/p1_full_field_oracle.yaml"


@pytest.fixture(scope="module")
def config():
    return load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def data():
    p1 = load_v3_config(P1_CONFIG, project_root=PROJECT_ROOT)
    return load_data(p1, PROJECT_ROOT)


def test_search_view_excludes_outer_domain_and_cannot_issue_evaluation(
    data, config
) -> None:
    view = issue_search_view(data, outer_domain="74t7kcdgkr", config=config)
    assert validate_search_view(view) == view.state_sha256
    assert view.outer_domain == "74t7kcdgkr"
    assert view.specimen_count == 231
    assert set(view.dataset_ids) == set(config.outer_domains) - {"74t7kcdgkr"}
    assert "74t7kcdgkr" not in view.dataset_ids
    with pytest.raises(D8AuthorityError):
        issue_evaluation_view(
            data,
            selection=None,
            outer_domain="74t7kcdgkr",
            config=config,
        )


def test_inner_view_excludes_query_and_outer_domains_from_fit(data, config) -> None:
    search = issue_search_view(data, outer_domain="74t7kcdgkr", config=config)
    inner = issue_inner_fold(search, query_domain="cgtnjyggtm")
    assert validate_inner_fold(inner) == inner.state_sha256
    assert inner.outer_domain == "74t7kcdgkr"
    assert inner.query_domain == "cgtnjyggtm"
    assert set(inner.fit_dataset_ids) == set(config.outer_domains) - {
        "74t7kcdgkr",
        "cgtnjyggtm",
    }
    assert set(inner.query_dataset_ids) == {"cgtnjyggtm"}
    assert set(inner.fit_specimen_ids).isdisjoint(inner.query_specimen_ids)
    assert set(inner.fit_indices).isdisjoint(inner.query_indices)


def test_search_authorities_are_process_local_and_tamper_evident(data, config) -> None:
    view = issue_search_view(data, outer_domain="74t7kcdgkr", config=config)
    with pytest.raises(TypeError):
        D8SearchView(
            None,
            outer_domain=view.outer_domain,
            data_view=view.data_view,
            config_sha256=view.config_sha256,
        )
    with pytest.raises(TypeError):
        copy.copy(view)
    with pytest.raises(TypeError):
        copy.deepcopy(view)
    with pytest.raises(TypeError):
        pickle.dumps(view)
    object.__setattr__(view, "outer_domain", "cgtnjyggtm")
    with pytest.raises(D8AuthorityError):
        validate_search_view(view)


@pytest.mark.slow
def test_d8_baseline_reproduces_all_p1_internal_only_predictions(
    data, config
) -> None:
    result = reproduce_internal_only_baseline(
        data,
        config=config,
        project_root=PROJECT_ROOT,
        device="cuda:0",
    )
    assert result.specimen_count == 276
    assert result.pca_dimensions == (8, 32, 8, 8, 8, 8)
    assert result.equal_domain_mae == pytest.approx(
        0.08963580465761432, abs=1.0e-12
    )
    assert result.maximum_prediction_error <= 1.0e-12
    assert result.maximum_target_error <= 1.0e-12
    assert result.predictions.flags.writeable is False
    assert result.targets.flags.writeable is False
