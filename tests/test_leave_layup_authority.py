from __future__ import annotations

from pathlib import Path

import numpy as np

from cmc_bbdm.msss.authority import load_authority
from cmc_bbdm.msss.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]


def test_leave_layup_authority_is_exact_and_domain_bound() -> None:
    protocol = load_protocol(ROOT / "paper_v3/configs/msss.yaml", project_root=ROOT)
    authority = load_authority(protocol, project_root=ROOT)
    groups = np.asarray(authority.layup_family)

    assert tuple(
        (group, int(np.count_nonzero(groups == group)))
        for group in protocol.layup_families
    ) == (("cross_ply", 146), ("quasi_isotropic", 130))
    assert {
        group: tuple(
            domain
            for domain in protocol.domain_order
            if np.any((groups == group) & (np.asarray(authority.dataset_ids) == domain))
        )
        for group in protocol.layup_families
    } == {
        "cross_ply": ("74t7kcdgkr", "xcmzfsbd9t", "yfxyg8jm46"),
        "quasi_isotropic": ("cgtnjyggtm", "w68dtmpfyf", "ykhs7s2dck"),
    }


def test_leave_layup_splits_are_specimen_disjoint() -> None:
    authority = load_authority(
        load_protocol(ROOT / "paper_v3/configs/msss.yaml", project_root=ROOT),
        project_root=ROOT,
    )
    groups = np.asarray(authority.layup_family)
    all_query: list[int] = []
    for group in ("cross_ply", "quasi_isotropic"):
        query = np.flatnonzero(groups == group)
        fit = np.flatnonzero(groups != group)
        assert set(query).isdisjoint(fit)
        assert len(query) and len(fit)
        all_query.extend(int(index) for index in query)
    assert sorted(all_query) == list(range(276))
