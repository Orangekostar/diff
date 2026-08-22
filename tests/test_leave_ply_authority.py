from __future__ import annotations

from pathlib import Path

import numpy as np

from cmc_bbdm.msss.authority import load_authority
from cmc_bbdm.msss.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]


def test_leave_ply_authority_is_exact_and_domain_bound() -> None:
    protocol = load_protocol(ROOT / "paper_v3/configs/msss.yaml", project_root=ROOT)
    authority = load_authority(protocol, project_root=ROOT)

    assert tuple(
        (ply, int(np.count_nonzero(authority.ply_count == ply)))
        for ply in protocol.ply_counts
    ) == ((8, 83), (16, 85), (24, 108))
    assert {
        ply: tuple(
            domain
            for domain in protocol.domain_order
            if np.any(
                (authority.ply_count == ply)
                & (np.asarray(authority.dataset_ids) == domain)
            )
        )
        for ply in protocol.ply_counts
    } == {
        8: ("74t7kcdgkr", "ykhs7s2dck"),
        16: ("w68dtmpfyf", "yfxyg8jm46"),
        24: ("cgtnjyggtm", "xcmzfsbd9t"),
    }


def test_leave_ply_splits_are_specimen_disjoint() -> None:
    authority = load_authority(
        load_protocol(ROOT / "paper_v3/configs/msss.yaml", project_root=ROOT),
        project_root=ROOT,
    )
    all_query: list[int] = []
    for ply in (8, 16, 24):
        query = np.flatnonzero(authority.ply_count == ply)
        fit = np.flatnonzero(authority.ply_count != ply)
        assert set(query).isdisjoint(fit)
        assert len(query) and len(fit)
        all_query.extend(int(index) for index in query)
    assert sorted(all_query) == list(range(276))
