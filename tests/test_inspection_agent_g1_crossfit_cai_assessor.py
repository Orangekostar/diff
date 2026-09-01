from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.cai_assessor import StateFeatureRow
from cmc_bbdm.inspection_agent_g1.crossfit import (
    G1CrossfitError,
    fit_crossfit_cai_assessor,
)

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _fit_rows() -> tuple[StateFeatureRow, ...]:
    generator = np.random.Generator(np.random.PCG64(20260901))
    output = []
    for domain_index, domain in enumerate(("d2", "d3", "d4", "d5")):
        for specimen_index in range(2):
            specimen = f"{domain}-s{specimen_index}"
            target = domain_index / 4 + specimen_index / 20
            for state_index in range(13):
                embedding = generator.normal(size=512)
                embedding[0] += 10 * target
                output.append(
                    StateFeatureRow(
                        sample_id=f"{specimen}|state-{state_index:02d}",
                        specimen_id=specimen,
                        dataset_id=domain,
                        policy="WARM" if state_index == 0 else "CONTINUATION",
                        observation_sha256=f"{len(output) + 1:064x}",
                        embedding=embedding,
                        effective_budget=state_index / 52,
                        observed_cell_fraction=(state_index + 8) / 64,
                        mean_observed_level=min(2.0, state_index / 12),
                        true_cai=target,
                    )
                )
    return tuple(output)


def test_cai_assessor_fit_is_dual_exclusion_crossfit() -> None:
    fit = fit_crossfit_cai_assessor(
        _fit_rows(),
        domain_order=DOMAINS,
        outer_target="d6",
        labeled_domain="d1",
        pca_dimension=32,
        ridge_alpha=10.0,
    )
    assert fit.roster.fit_domains == ("d2", "d3", "d4", "d5")
    assert fit.assessor.fit_domains == fit.roster.fit_domains
    assert len(fit.assessor.fit_sample_ids) == 4 * 2 * 13
    assert set(fit.assessor.fit_domains).isdisjoint({"d1", "d6"})
    assert len(fit.state_sha256) == 64


@pytest.mark.parametrize("forbidden_domain", ("d1", "d6"))
def test_cai_assessor_rejects_outer_or_labeled_rows(forbidden_domain: str) -> None:
    rows = _fit_rows()
    contaminated = (
        *rows,
        replace(
            rows[0],
            sample_id=f"{forbidden_domain}|contaminated",
            specimen_id=f"{forbidden_domain}|specimen",
            dataset_id=forbidden_domain,
        ),
    )
    with pytest.raises(G1CrossfitError, match="exact fit-domain roster"):
        fit_crossfit_cai_assessor(
            contaminated,
            domain_order=DOMAINS,
            outer_target="d6",
            labeled_domain="d1",
            pca_dimension=32,
            ridge_alpha=10.0,
        )


def test_cai_assessor_rejects_a_missing_fit_domain() -> None:
    rows = tuple(row for row in _fit_rows() if row.dataset_id != "d5")
    with pytest.raises(G1CrossfitError, match="exact fit-domain roster"):
        fit_crossfit_cai_assessor(
            rows,
            domain_order=DOMAINS,
            outer_target="d6",
            labeled_domain="d1",
            pca_dimension=32,
            ridge_alpha=10.0,
        )
