"""Capture-group construction and deterministic v3 cohort splitting."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

_IDENTITY_FIELDS = (
    "cscan_source_sha256",
    "cscan_source_path",
    "surface_sha256",
    "impacted_surface_path",
    "registered_cscan_crop_sha256",
)


class _UnionFind:
    def __init__(self, keys: Sequence[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            self.parent[high] = low


def _specimen_key(row: Mapping[str, str]) -> str:
    key = row.get("specimen_key", "").strip()
    if key:
        return key
    dataset_id = row.get("dataset_id", "").strip()
    specimen_id = row.get("specimen_id", "").strip()
    if not dataset_id or not specimen_id:
        raise ValueError("candidate row lacks exact specimen identity")
    return f"{dataset_id}:{specimen_id}"


def build_capture_groups(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Union records sharing any registered physical/source image identity."""

    keys = tuple(_specimen_key(row) for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("candidate specimen identity is duplicated")
    groups = _UnionFind(keys)
    seen: dict[tuple[str, str], str] = {}
    for row, key in zip(rows, keys):
        for field in _IDENTITY_FIELDS:
            value = row.get(field, "").strip()
            if not value:
                continue
            identity = (field, value)
            previous = seen.setdefault(identity, key)
            groups.union(previous, key)
    members: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        members[groups.find(key)].append(key)
    group_ids: dict[str, str] = {}
    for values in members.values():
        payload = "|".join(sorted(values))
        group_id = (
            "cg_"
            + hashlib.sha256(f"cai-agent-v3-capture|{payload}".encode()).hexdigest()[
                :16
            ]
        )
        for key in values:
            group_ids[key] = group_id
    return group_ids


def assign_capture_group_splits(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Apply the fixed per-domain 60/20/remainder capture-group split."""

    group_domains: dict[str, set[str]] = defaultdict(set)
    group_members: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = _specimen_key(row)
        domain = row.get("dataset_id", "").strip()
        group = row.get("capture_group_id", "").strip()
        if not domain or not group:
            raise ValueError("split row lacks domain or capture group")
        group_domains[group].add(domain)
        group_members[group].append(key)
    cross_domain = sorted(
        group for group, domains in group_domains.items() if len(domains) != 1
    )
    if cross_domain:
        raise ValueError(f"capture groups cross domains: {cross_domain}")
    domain_groups: dict[str, list[str]] = defaultdict(list)
    for group, domains in group_domains.items():
        domain_groups[next(iter(domains))].append(group)
    assignments: dict[str, str] = {}
    for domain, groups in sorted(domain_groups.items()):
        ordered = sorted(
            groups,
            key=lambda group: (
                hashlib.sha256(f"cai-agent-v3|{group}".encode()).hexdigest(),
                group,
            ),
        )
        count = len(ordered)
        train_stop = math.floor(0.6 * count)
        valid_stop = train_stop + math.floor(0.2 * count)
        if train_stop == 0 or valid_stop == train_stop or valid_stop == count:
            raise ValueError(f"domain {domain} cannot populate all three splits")
        for index, group in enumerate(ordered):
            split = (
                "TRAIN"
                if index < train_stop
                else "VALID"
                if index < valid_stop
                else "TEST"
            )
            for key in group_members[group]:
                assignments[key] = split
    return dict(sorted(assignments.items()))


__all__ = ["assign_capture_group_splits", "build_capture_groups"]
