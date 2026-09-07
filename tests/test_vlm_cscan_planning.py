from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.vlm_cscan.contracts import MethodId, SurfacePlan
from cmc_bbdm.vlm_cscan.planning import geometry_cell_order, initial_cell_order
from cmc_bbdm.vlm_cscan.vlm import (
    ReplanChoiceCache,
    ReplanRequest,
    SurfacePlanCache,
    SurfacePlanRequest,
    VLMRawResponse,
    parse_replan_choice,
    parse_surface_plan,
)

VALID_PLAN = """{
  "regions": [{
    "cells": [27, 28],
    "visible_cue": "indentation_like",
    "alternative": "illumination_possible",
    "confidence": "low"
  }],
  "priority_cells": [27, 28],
  "initial_skill": "SURVEY_ROI",
  "no_reliable_surface_cue": false
}"""


def test_vlm_json_enforces_cell_and_menu_bounds() -> None:
    """Catches accepting free coordinates, duplicate cells, or illegal menu IDs."""

    plan = parse_surface_plan(VALID_PLAN)
    assert plan.priority_cells == (27, 28)
    with pytest.raises(ValueError, match="surface plan"):
        parse_surface_plan(VALID_PLAN.replace("[27, 28]", "[27, 64]", 1))
    with pytest.raises(ValueError, match="surface plan"):
        parse_surface_plan(VALID_PLAN.replace("[27, 28]", "[27, 27]", 1))
    assert parse_replan_choice(
        '{"menu_id":"m2","reason_code":"NEW_EVIDENCE"}',
        legal_menu_ids=("m1", "m2"),
    ).menu_id == "m2"
    with pytest.raises(ValueError, match="legal menu"):
        parse_replan_choice(
            '{"menu_id":"free_coordinate","reason_code":"TRY"}',
            legal_menu_ids=("m1", "m2"),
        )


def test_vlm_open_and_feedback_methods_share_exact_initial_order() -> None:
    """Catches changing B5/B6 initial conditions relative to B3."""

    plan = parse_surface_plan(VALID_PLAN)
    saliency = np.linspace(0.0, 1.0, 64, dtype=np.float64)

    orders = {
        method: initial_cell_order(method, surface_plan=plan, saliency_scores=saliency)
        for method in (MethodId.B3, MethodId.B5, MethodId.B6)
    }

    assert orders[MethodId.B3] == orders[MethodId.B5] == orders[MethodId.B6]
    assert orders[MethodId.B3][:2] == (27, 28)
    assert len(orders[MethodId.B3]) == len(set(orders[MethodId.B3])) == 64
    abstention = SurfacePlan(
        regions=(),
        priority_cells=(),
        initial_skill="BROADEN_SEARCH",
        no_reliable_surface_cue=True,
    )
    assert initial_cell_order(
        MethodId.B3,
        surface_plan=abstention,
        saliency_scores=saliency,
    ) == geometry_cell_order()


def test_surface_plan_cache_reuses_original_response_and_latency(tmp_path: Path) -> None:
    """Catches duplicate inference or reporting cache-hit latency as zero deployment cost."""

    request = SurfacePlanRequest(
        model_repository="Qwen/Qwen2.5-VL-7B-Instruct",
        model_revision="c" * 40,
        prompt_sha256="a" * 64,
        clean_image_sha256="b" * 64,
        gridded_image_sha256="d" * 64,
        preprocessing_sha256="e" * 64,
    )
    cache_path = tmp_path / "plans.jsonl"

    first = SurfacePlanCache(cache_path).resolve(
        request,
        lambda _prompt: VLMRawResponse(
            text=VALID_PLAN,
            latency_seconds=0.25,
            input_tokens=100,
            output_tokens=40,
        ),
    )

    def forbidden_call(_prompt: str) -> VLMRawResponse:
        raise AssertionError("cache miss repeated external inference")

    second = SurfacePlanCache(cache_path).resolve(request, forbidden_call)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.plan == first.plan
    assert second.original_latency_seconds == 0.25
    assert second.actual_call_count == 0
    assert cache_path.read_text(encoding="utf-8").count("\n") == 1


def test_replan_cache_shares_jsonl_without_repeating_inference(tmp_path: Path) -> None:
    """Catches repeated tool calls or cross-record cache corruption on resume."""

    cache_path = tmp_path / "plans.jsonl"
    prompt = "visible menu"
    request = ReplanRequest(
        model_repository="Qwen/Qwen2.5-VL-7B-Instruct",
        model_revision="c" * 40,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        clean_image_sha256="b" * 64,
        evidence_image_sha256="d" * 64,
        evidence_state_sha256="e" * 64,
        menu_sha256="f" * 64,
        task="LOCATE",
        event_code="INITIAL_ROI_UNSUPPORTED",
    )
    cache = ReplanChoiceCache(cache_path)
    first = cache.resolve(
        request,
        prompt=prompt,
        legal_menu_ids=("m1", "m2"),
        infer=lambda _prompt: VLMRawResponse(
            text='{"menu_id":"m2","reason_code":"BROADEN"}',
            latency_seconds=0.4,
            input_tokens=80,
            output_tokens=12,
        ),
    )

    second = cache.resolve(
        request,
        prompt=prompt,
        legal_menu_ids=("m1", "m2"),
        infer=lambda _prompt: (_ for _ in ()).throw(
            AssertionError("cache miss repeated external inference")
        ),
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.choice.menu_id == "m2"
    assert second.original_latency_seconds == 0.4
    assert second.actual_call_count == 0
    assert cache_path.read_text(encoding="utf-8").count("\n") == 1
