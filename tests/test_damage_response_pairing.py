from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS
from cmc_bbdm.damage_response.pairing import (
    FeatureIdentity,
    PairingError,
    TraceIdentity,
    load_feature_identities,
    pair_exact,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_BANK = (
    ROOT / "results/aei_selective_invariance/a2_paired_features/paired_features.npz"
)
FEATURE_BANK_SHA256 = (
    "f2a69f0da75e20880202d7fc4a6a92f979978406ec21f9d83e4bc8db07fb72a8"
)


def test_pairing_rejects_order_only_match() -> None:
    features = (FeatureIdentity("c8-2", "74t7kcdgkr"),)
    traces = (TraceIdentity("c8-3", "74t7kcdgkr", "a" * 64),)

    with pytest.raises(PairingError, match="exact identity"):
        pair_exact(features, traces)


def test_pairing_rejects_duplicate_specimen_domain() -> None:
    item = TraceIdentity("c8-2", "74t7kcdgkr", "a" * 64)

    with pytest.raises(PairingError, match="duplicate"):
        pair_exact((FeatureIdentity("c8-2", "74t7kcdgkr"),), (item, item))


def test_pairing_rejects_same_specimen_assigned_to_two_domains() -> None:
    traces = (
        TraceIdentity("c8-2", "74t7kcdgkr", "a" * 64),
        TraceIdentity("c8-2", "cgtnjyggtm", "b" * 64),
    )
    features = (
        FeatureIdentity("c8-2", "74t7kcdgkr"),
        FeatureIdentity("c8-2", "cgtnjyggtm"),
    )

    with pytest.raises(PairingError, match="multiple domains"):
        pair_exact(features, traces)


def test_pairing_normalizes_only_case_and_surrounding_whitespace() -> None:
    pairs = pair_exact(
        (FeatureIdentity(" C8-2 ", "74T7KCDGKR"),),
        (TraceIdentity("c8-2", " 74t7kcdgkr ", "A" * 64),),
    )

    assert len(pairs) == 1
    assert pairs[0].specimen_id == "c8-2"
    assert pairs[0].domain_id == "74t7kcdgkr"
    assert pairs[0].raw_trace_sha256 == "a" * 64


def test_pairing_does_not_equate_zero_padded_specimen_names() -> None:
    with pytest.raises(PairingError, match="exact identity"):
        pair_exact(
            (FeatureIdentity("c8-2", "74t7kcdgkr"),),
            (TraceIdentity("c8-002", "74t7kcdgkr", "a" * 64),),
        )


def test_pairing_rejects_invalid_trace_hash() -> None:
    with pytest.raises(PairingError, match="SHA-256"):
        pair_exact(
            (FeatureIdentity("c8-2", "74t7kcdgkr"),),
            (TraceIdentity("c8-2", "74t7kcdgkr", "not-a-hash"),),
        )


def test_real_feature_bank_is_exact_frozen_primary_roster() -> None:
    identities = load_feature_identities(
        FEATURE_BANK,
        expected_sha256=FEATURE_BANK_SHA256,
    )

    assert len(identities) == 276
    assert len({(item.specimen_id, item.domain_id) for item in identities}) == 276
    assert Counter(item.domain_id for item in identities) == Counter(PRIMARY_COUNTS)


def test_real_feature_bank_rejects_wrong_authority_hash() -> None:
    with pytest.raises(PairingError, match="feature-bank SHA-256"):
        load_feature_identities(FEATURE_BANK, expected_sha256="0" * 64)
