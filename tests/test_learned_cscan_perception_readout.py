from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionObservation, InspectionTask
from cmc_bbdm.inspection_agent.state import (
    InspectionCellAction,
    action_added_positions,
    apply_action,
    zero_state,
)
from cmc_bbdm.learned_cscan.contracts import Task
from cmc_bbdm.learned_cscan.observation import build_observation_packet
from cmc_bbdm.learned_cscan.perception import (
    FORMAT_REPAIR_PROMPT,
    SURFACE_PERCEPT_PROMPT,
    SURFACE_PERCEPT_SCHEMA,
    SurfacePerceptCache,
    SurfacePerceptRequest,
    parse_surface_percept,
)
from cmc_bbdm.learned_cscan.policies import RuleMethod, select_rule_action
from cmc_bbdm.learned_cscan.readout import BackgroundPrior, read_visible_evidence
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.vlm_cscan.vlm import VLMRawResponse


def test_surface_percept_contract_is_action_free_and_strict() -> None:
    contract = (
        SURFACE_PERCEPT_PROMPT
        + json.dumps(SURFACE_PERCEPT_SCHEMA, sort_keys=True, separators=(",", ":"))
    ).lower()
    for forbidden in (
        "next_action",
        "menu_id",
        "initial_scan_skill",
        '"cells":[27]',
        "indentation",
        '"m1"',
    ):
        assert forbidden not in contract

    percept = parse_surface_percept(
        json.dumps(
            {
                "regions": [
                    {
                        "cells": [3, 4],
                        "cue": "surface_shape_change",
                        "alternative": "illumination_or_texture",
                        "confidence": "medium",
                    }
                ],
                "no_reliable_cue": False,
            }
        )
    )

    assert percept.regions[0].cells == (3, 4)
    assert percept.regions[0].confidence == "medium"
    assert percept.no_reliable_cue is False
    overlapping = parse_surface_percept(
        json.dumps(
            {
                "regions": [
                    {
                        "cells": [35, 36, 37],
                        "cue": "circle_with_lines",
                        "alternative": "scratch_or_tool_mark",
                        "confidence": "medium",
                    },
                    {
                        "cells": [36],
                        "cue": "circle_with_lines",
                        "alternative": "scratch_or_tool_mark",
                        "confidence": "medium",
                    },
                ],
                "no_reliable_cue": False,
            }
        )
    )
    assert overlapping.regions[0].cells == (35, 36, 37)
    assert overlapping.regions[1].cells == (36,)
    with pytest.raises(ValueError, match="schema"):
        parse_surface_percept(
            '{"regions":[],"no_reliable_cue":true,"next_action":"scan"}'
        )


def test_surface_percept_cache_identity_is_complete_and_task_independent(
    tmp_path,
) -> None:
    request = SurfacePerceptRequest(
        model_revision="a" * 40,
        clean_image_sha256="b" * 64,
        gridded_image_sha256="c" * 64,
        render_version="surface-grid-v2",
        prompt_sha256="d" * 64,
        schema_version=2,
    )
    assert "task" not in request.__dataclass_fields__
    variants = (
        replace(request, model_revision="e" * 40),
        replace(request, clean_image_sha256="e" * 64),
        replace(request, gridded_image_sha256="e" * 64),
        replace(request, render_version="surface-grid-v3"),
        replace(request, prompt_sha256="e" * 64),
        replace(request, schema_version=3),
    )
    assert len({request.cache_key, *(variant.cache_key for variant in variants)}) == 7

    response = VLMRawResponse(
        text='{"regions":[],"no_reliable_cue":true}',
        latency_seconds=0.25,
        input_tokens=10,
        output_tokens=4,
    )
    cache = SurfacePerceptCache(tmp_path / "surface_percepts.jsonl")
    first = cache.resolve(request, lambda _prompt: response)
    second = cache.resolve(
        request,
        lambda _prompt: (_ for _ in ()).throw(
            AssertionError("cached percept repeated inference")
        ),
    )

    assert first.cache_hit is False
    assert first.actual_call_count == 1
    assert second.cache_hit is True
    assert second.actual_call_count == 0
    assert second.percept == first.percept

    repair_prompts: list[str] = []
    responses = iter(
        (
            VLMRawResponse(
                text=(
                    '{"regions":[{"cells":[1],"cue":"x",'
                    '"alternative":"y","confidence":"low"}],'
                    '"no_reliable_cue":true}'
                ),
                latency_seconds=0.1,
                input_tokens=5,
                output_tokens=5,
            ),
            response,
        )
    )

    def infer_repair(prompt: str) -> VLMRawResponse:
        repair_prompts.append(prompt)
        return next(responses)

    repaired = SurfacePerceptCache(tmp_path / "repair.jsonl").resolve(
        replace(request, prompt_sha256="f" * 64), infer_repair
    )
    assert repaired.repaired is True
    assert len(repair_prompts) == 2
    assert '"no_reliable_cue":true' in repair_prompts[1]


def test_surface_percept_repair_prompt_restates_the_cross_region_cell_cap() -> None:
    assert "所有 regions 的 cells 合计不得超过 8" in FORMAT_REPAIR_PROMPT
    region = SURFACE_PERCEPT_SCHEMA["properties"]["regions"]["items"]
    assert SURFACE_PERCEPT_SCHEMA["properties"]["regions"]["maxItems"] == 2
    assert region["properties"]["cells"]["maxItems"] == 4
    assert region["properties"]["cue"]["minLength"] == 1
    assert region["properties"]["alternative"]["minLength"] == 1


def test_reader_v2_preserves_measurements_without_filling_unmeasured_cells() -> None:
    grid = build_acquisition_grid(33, 33, initial_budget=0.015625)
    positions = np.asarray([[0, 0]], dtype=np.int64)
    values = np.asarray([[255, 0, 0]], dtype=np.uint8)
    levels = (0,) + (-1,) * 63

    readout = read_visible_evidence(
        grid=grid,
        positions=positions,
        values=values,
        cell_levels=levels,
        prior=BackgroundPrior(
            rgb=np.asarray([0, 0, 0], dtype=np.uint8),
            fit_specimen_keys=("train:one",),
            fit_split="TRAIN",
        ),
        distance_threshold=0.18,
    )

    assert readout.measured_mask[0, 0]
    assert readout.estimate_valid[0, 0]
    assert np.array_equal(readout.estimated_rgb[0, 0], values[0])
    assert np.count_nonzero(readout.estimate_valid) == 1
    assert np.count_nonzero(readout.candidate_mask) == 1
    assert readout.cells[1].measured_count == 0
    assert readout.cells[1].estimate_valid is False
    assert readout.cells[1].candidate is False


def test_packet_depends_only_on_visible_history_and_perception() -> None:
    grid = build_acquisition_grid(33, 33, initial_budget=0.015625)
    initial = zero_state(grid)
    action = InspectionCellAction(0, -1, 0)
    state = apply_action(grid, initial, action)
    positions = action_added_positions(grid, initial, action)
    first_hidden = np.arange(33 * 33 * 3, dtype=np.uint16).reshape(33, 33, 3)
    first_hidden = (first_hidden % 256).astype(np.uint8)
    second_hidden = first_hidden.copy()
    second_hidden[-1, -1] = (255, 255, 255)
    values = first_hidden[positions[:, 0], positions[:, 1]]
    assert np.array_equal(values, second_hidden[positions[:, 0], positions[:, 1]])

    def observation(hidden: np.ndarray) -> InspectionObservation:
        visible_values = hidden[positions[:, 0], positions[:, 1]]
        return InspectionObservation(
            surface_rgb=np.full((20, 22, 3), 127, dtype=np.uint8),
            surface_sha256="a" * 64,
            task=InspectionTask.FIELD,
            native_shape=grid.native_shape,
            native_count=33 * 33,
            grid_sha256=grid.state_sha256,
            measurement_state=state,
            acquired_positions=positions,
            measurement_values=visible_values,
            exact_acquired_count=len(positions),
            endpoint_budget=1.0,
            action_history=(action,),
        )

    percept = parse_surface_percept(
        '{"regions":[{"cells":[0],"cue":"shape_change",'
        '"alternative":"lighting","confidence":"low"}],'
        '"no_reliable_cue":false}'
    )
    prior = BackgroundPrior(
        rgb=np.asarray([120, 120, 120], dtype=np.uint8),
        fit_specimen_keys=("train:one",),
        fit_split="TRAIN",
    )
    first = build_observation_packet(
        observation(first_hidden),
        grid=grid,
        percept=percept,
        task=Task.LOCATE,
        prior=prior,
        distance_threshold=0.18,
        probe_position=(0.0, 0.0),
        route_cost=0.0,
    )
    second = build_observation_packet(
        observation(second_hidden),
        grid=grid,
        percept=percept,
        task=Task.LOCATE,
        prior=prior,
        distance_threshold=0.18,
        probe_position=(0.0, 0.0),
        route_cost=0.0,
    )

    assert first.feature_sha256 == second.feature_sha256
    assert first.cell_features.tobytes() == second.cell_features.tobytes()
    assert first.subblock_features.tobytes() == second.subblock_features.tobytes()
    rule_only = build_observation_packet(
        observation(first_hidden),
        grid=grid,
        percept=percept,
        task=Task.LOCATE,
        prior=prior,
        distance_threshold=0.18,
        probe_position=(0.0, 0.0),
        route_cost=0.0,
        include_actor_subblocks=False,
    )
    assert not np.any(rule_only.subblock_features)
    assert select_rule_action(
        RuleMethod.R_BALANCED, rule_only, grid=grid, coverage_period=4
    ) == select_rule_action(
        RuleMethod.R_BALANCED, first, grid=grid, coverage_period=4
    )
    assert set(first.actor_tensors()) == {
        "cell_features",
        "subblock_features",
        "global_features",
        "history_features",
        "legal_mask",
    }
