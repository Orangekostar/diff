from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmc_bbdm.damage_response.authority import snapshot_file
from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FEATURE_BANK_RELATIVE_PATH = (
    "results/aei_selective_invariance/a2_paired_features/paired_features.npz"
)


class PairingError(RuntimeError):
    """Raised when specimen identity cannot be joined exactly."""


@dataclass(frozen=True)
class FeatureIdentity:
    specimen_id: str
    domain_id: str


@dataclass(frozen=True)
class TraceIdentity:
    specimen_id: str
    domain_id: str
    raw_trace_sha256: str


@dataclass(frozen=True)
class PairedIdentity:
    specimen_id: str
    domain_id: str
    raw_trace_sha256: str


def _canonical_identity(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise PairingError(f"{field} must be a string")
    normalized = value.strip().casefold()
    if not normalized:
        raise PairingError(f"{field} must be nonempty")
    return normalized


def _feature_key(item: FeatureIdentity) -> tuple[str, str]:
    return (
        _canonical_identity(item.specimen_id, field="specimen_id"),
        _canonical_identity(item.domain_id, field="domain_id"),
    )


def _trace_key(item: TraceIdentity) -> tuple[str, str]:
    return (
        _canonical_identity(item.specimen_id, field="specimen_id"),
        _canonical_identity(item.domain_id, field="domain_id"),
    )


def _reject_duplicate_keys(
    keyed_items: Iterable[tuple[tuple[str, str], object]], *, source: str
) -> dict[tuple[str, str], object]:
    by_key: dict[tuple[str, str], object] = {}
    specimen_domains: dict[str, set[str]] = {}
    for key, item in keyed_items:
        if key in by_key:
            raise PairingError(f"duplicate {source} specimen/domain identity: {key}")
        by_key[key] = item
        specimen_domains.setdefault(key[0], set()).add(key[1])
    reused = sorted(
        specimen for specimen, domains in specimen_domains.items() if len(domains) > 1
    )
    if reused:
        raise PairingError(
            f"{source} specimen assigned to multiple domains: {', '.join(reused)}"
        )
    return by_key


def pair_exact(
    features: Iterable[FeatureIdentity], traces: Iterable[TraceIdentity]
) -> tuple[PairedIdentity, ...]:
    """Join feature and raw-trace identities without fuzzy or order matching."""

    feature_items = tuple(features)
    trace_items = tuple(traces)
    feature_by_key = _reject_duplicate_keys(
        ((_feature_key(item), item) for item in feature_items), source="feature"
    )
    trace_by_key = _reject_duplicate_keys(
        ((_trace_key(item), item) for item in trace_items), source="trace"
    )

    for item in trace_items:
        digest = item.raw_trace_sha256.strip().casefold()
        if _SHA256_RE.fullmatch(digest) is None:
            raise PairingError(
                f"raw trace requires a valid SHA-256: {item.specimen_id!r}"
            )

    feature_keys = set(feature_by_key)
    trace_keys = set(trace_by_key)
    if feature_keys != trace_keys:
        missing = sorted(feature_keys - trace_keys)
        unexpected = sorted(trace_keys - feature_keys)
        raise PairingError(
            "exact identity sets differ; "
            f"missing traces={missing!r}; unexpected traces={unexpected!r}"
        )

    pairs: list[PairedIdentity] = []
    for feature in feature_items:
        key = _feature_key(feature)
        trace = trace_by_key[key]
        if not isinstance(trace, TraceIdentity):
            raise PairingError("internal trace identity type mismatch")
        pairs.append(
            PairedIdentity(
                specimen_id=key[0],
                domain_id=key[1],
                raw_trace_sha256=trace.raw_trace_sha256.strip().casefold(),
            )
        )
    return tuple(pairs)


def load_feature_identities(
    path: Path, *, expected_sha256: str
) -> tuple[FeatureIdentity, ...]:
    """Load the frozen primary identity vectors after checking their exact hash."""

    expected_digest = expected_sha256.strip().casefold()
    if _SHA256_RE.fullmatch(expected_digest) is None:
        raise PairingError("expected feature-bank SHA-256 is invalid")
    snapshot = snapshot_file(
        path,
        max_bytes=1024 * 1024 * 1024,
        logical_source="git:frozen_feature_bank",
        relative_path=_FEATURE_BANK_RELATIVE_PATH,
    )
    if snapshot.sha256 != expected_digest:
        raise PairingError(
            "feature-bank SHA-256 mismatch: "
            f"expected {expected_digest}, observed {snapshot.sha256}"
        )

    try:
        with np.load(path, allow_pickle=False) as archive:
            if not {"specimen_ids", "dataset_ids"}.issubset(archive.files):
                raise PairingError("feature bank lacks required identity vectors")
            specimen_ids = archive["specimen_ids"]
            domain_ids = archive["dataset_ids"]
    except PairingError:
        raise
    except Exception as error:
        raise PairingError("feature bank could not be decoded safely") from error

    if specimen_ids.ndim != 1 or domain_ids.ndim != 1:
        raise PairingError("feature-bank identity vectors must be one-dimensional")
    if specimen_ids.shape != domain_ids.shape:
        raise PairingError("feature-bank identity vector lengths differ")
    if specimen_ids.dtype.kind != "U" or domain_ids.dtype.kind != "U":
        raise PairingError("feature-bank identity vectors must contain Unicode strings")

    identities = tuple(
        FeatureIdentity(
            _canonical_identity(specimen, field="specimen_id"),
            _canonical_identity(domain, field="domain_id"),
        )
        for specimen, domain in zip(specimen_ids.tolist(), domain_ids.tolist(), strict=True)
    )
    by_key = _reject_duplicate_keys(
        ((_feature_key(item), item) for item in identities), source="feature-bank"
    )
    observed_counts = Counter(item.domain_id for item in identities)
    if len(by_key) != sum(PRIMARY_COUNTS.values()) or dict(observed_counts) != dict(
        PRIMARY_COUNTS
    ):
        raise PairingError(
            "feature bank does not match the frozen 276-specimen primary roster"
        )
    return identities
