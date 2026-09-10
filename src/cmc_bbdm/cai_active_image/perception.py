"""Frozen surface-percept adapter and first-action proposal contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmc_bbdm.learned_cscan.perception import SurfacePercept, parse_surface_percept

from .contracts import Method

_CONFIDENCE = {"unknown": 0.0, "low": 1.0 / 3.0, "medium": 2.0 / 3.0, "high": 1.0}


@dataclass(frozen=True, slots=True)
class VLMActorFeatures:
    region_indicator: np.ndarray
    confidence: np.ndarray
    reliable_candidates: np.ndarray
    available: bool
    no_reliable_cue: bool

    def __post_init__(self) -> None:
        if (
            self.region_indicator.shape != (64,)
            or self.confidence.shape != (64,)
            or self.reliable_candidates.shape != (64,)
            or self.region_indicator.dtype.kind != "f"
            or self.confidence.dtype.kind != "f"
            or self.reliable_candidates.dtype != np.dtype(bool)
        ):
            raise ValueError("VLM actor features are invalid")


@dataclass(frozen=True, slots=True)
class CachedVLMPercept:
    specimen_key: str
    cache_key: str
    percept: SurfacePercept
    features: VLMActorFeatures
    cache_hit: bool
    actual_call_count: int
    original_call_count: int
    repaired: bool
    latency_seconds: float
    raw_text: str


@dataclass(frozen=True, slots=True)
class ProposalDecision:
    mask: np.ndarray
    reason: str

    def __post_init__(self) -> None:
        if self.mask.shape != (64,) or self.mask.dtype != np.dtype(bool) or not self.reason:
            raise ValueError("proposal decision is invalid")


def build_vlm_actor_features(
    percept: SurfacePercept,
    *,
    cache_available: bool,
    clockwise_quarter_turns: int,
) -> VLMActorFeatures:
    """Use cells from the already registered display grid without a second rotation."""

    if type(percept) is not SurfacePercept or type(cache_available) is not bool:
        raise TypeError("issued percept and cache availability are required")
    if clockwise_quarter_turns != 1:
        raise ValueError("surface percept must use the once-registered C-scan frame")
    indicator = np.zeros(64, dtype=np.float32)
    confidence = np.zeros(64, dtype=np.float32)
    if cache_available:
        for region in percept.regions:
            value = _CONFIDENCE[region.confidence]
            for cell in region.cells:
                indicator[cell] = 1.0
                confidence[cell] = max(confidence[cell], value)
    reliable = highest_reliable_candidate_mask(indicator, confidence)
    return VLMActorFeatures(
        region_indicator=indicator,
        confidence=confidence,
        reliable_candidates=reliable,
        available=cache_available,
        no_reliable_cue=bool(cache_available and percept.no_reliable_cue),
    )


def highest_reliable_candidate_mask(
    region_indicator: np.ndarray, confidence: np.ndarray
) -> np.ndarray:
    indicator = np.asarray(region_indicator)
    values = np.asarray(confidence)
    if (
        indicator.shape != values.shape
        or indicator.ndim not in {1, 2}
        or indicator.shape[-1] != 64
        or not np.all(np.isfinite(indicator))
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("VLM confidence fields are invalid")
    eligible = (indicator > 0.0) & (values >= 2.0 / 3.0)
    highest = np.max(np.where(eligible, values, -1.0), axis=-1, keepdims=True)
    return eligible & (values == highest)


def proposal_decision_for_method(
    method: Method,
    *,
    features: VLMActorFeatures,
    legal_mask: np.ndarray,
    action_count: int,
) -> ProposalDecision:
    legal = np.asarray(legal_mask, dtype=bool)
    if (
        type(method) is not Method
        or type(features) is not VLMActorFeatures
        or legal.shape != (64,)
        or type(action_count) is not int
        or action_count < 0
    ):
        raise ValueError("proposal request is invalid")
    if action_count > 0:
        return ProposalDecision(legal.copy(), "C0_RELEASED_AFTER_FIRST_ACTION")
    if not method.uses_vlm:
        return ProposalDecision(legal.copy(), "METHOD_DOES_NOT_USE_VLM_C0")
    if not features.available:
        return ProposalDecision(legal.copy(), "VLM_UNAVAILABLE")
    if features.no_reliable_cue:
        return ProposalDecision(legal.copy(), "VLM_NO_RELIABLE_CUE")
    if not bool(features.reliable_candidates.any()):
        reason = (
            "NO_MEDIUM_OR_HIGH_CONFIDENCE"
            if bool((features.region_indicator > 0.0).any())
            else "NO_VLM_REGION_CANDIDATES"
        )
        return ProposalDecision(legal.copy(), reason)
    restricted = legal & features.reliable_candidates
    if not bool(restricted.any()):
        return ProposalDecision(legal.copy(), "C0_NO_AFFORDABLE_LEGAL_CELL")
    return ProposalDecision(restricted, "HIGHEST_RELIABLE_CONFIDENCE_C0")


def proposal_mask_for_method(
    method: Method,
    *,
    features: VLMActorFeatures,
    legal_mask: np.ndarray,
    action_count: int,
) -> np.ndarray:
    return proposal_decision_for_method(
        method,
        features=features,
        legal_mask=legal_mask,
        action_count=action_count,
    ).mask


def load_frozen_vlm_cache(
    cache_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, CachedVLMPercept]:
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        cache_rows = [
            json.loads(line)
            for line in Path(cache_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("frozen VLM cache is unreadable") from error
    if (
        type(manifest) is not dict
        or manifest.get("specimen_count") != 60
        or manifest.get("model_revision")
        != "cc594898137f460bfe9f0759e9844b3ce807cfb5"
        or manifest.get("schema_version_surface_percept") != 2
        or type(manifest.get("records")) is not list
        or len(cache_rows) != 60
    ):
        raise ValueError("frozen VLM cache identity changed")
    by_key: dict[str, dict[str, object]] = {}
    for row in cache_rows:
        key = row.get("cache_key")
        if type(key) is not str or key in by_key:
            raise ValueError("frozen VLM cache key is invalid")
        by_key[key] = row
    output: dict[str, CachedVLMPercept] = {}
    for entry in manifest["records"]:
        if type(entry) is not dict:
            raise ValueError("frozen VLM manifest row is invalid")
        specimen_key = entry.get("specimen_key")
        cache_key = entry.get("cache_key")
        if type(specimen_key) is not str or specimen_key in output or cache_key not in by_key:
            raise ValueError("frozen VLM specimen binding is invalid")
        row = by_key[str(cache_key)]
        request = row.get("request")
        call_count = row.get("call_count")
        repaired = row.get("repaired")
        if (
            type(request) is not dict
            or row.get("schema_version") != 2
            or row.get("record_type") != "surface_percept"
            or type(call_count) is not int
            or call_count not in {1, 2}
            or type(repaired) is not bool
            or repaired != (call_count == 2)
            or type(row.get("raw_text")) is not str
            or not math.isfinite(float(row.get("latency_seconds", -1.0)))
            or float(row.get("latency_seconds", -1.0)) < 0.0
            or request.get("model_revision") != manifest.get("model_revision")
            or request.get("prompt_sha256") != manifest.get("prompt_sha256")
            or request.get("render_version") != manifest.get("render_version")
            or request.get("clean_image_sha256") != entry.get("clean_image_sha256")
            or request.get("gridded_image_sha256") != entry.get("gridded_image_sha256")
            or entry.get("cache_hit") is not True
        ):
            raise ValueError("frozen VLM request binding or call or repair count changed")
        percept = parse_surface_percept(
            json.dumps(row.get("percept"), ensure_ascii=False, separators=(",", ":"))
        )
        features = build_vlm_actor_features(
            percept,
            cache_available=True,
            clockwise_quarter_turns=1,
        )
        output[specimen_key] = CachedVLMPercept(
            specimen_key=specimen_key,
            cache_key=str(cache_key),
            percept=percept,
            features=features,
            cache_hit=True,
            actual_call_count=0,
            original_call_count=call_count,
            repaired=repaired,
            latency_seconds=float(row["latency_seconds"]),
            raw_text=str(row["raw_text"]),
        )
    if len(output) != 60 or set(by_key) != {
        item.cache_key for item in output.values()
    }:
        raise ValueError("frozen VLM cache coverage changed")
    return output


__all__ = [
    "CachedVLMPercept",
    "ProposalDecision",
    "VLMActorFeatures",
    "build_vlm_actor_features",
    "highest_reliable_candidate_mask",
    "load_frozen_vlm_cache",
    "proposal_decision_for_method",
    "proposal_mask_for_method",
]
