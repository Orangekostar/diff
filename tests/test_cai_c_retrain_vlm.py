from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GROUNDING = ROOT / "results/cai_agent_v3/vlm_grounding_pilot/r1_e8d9ef01"


class FakeBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def infer(self, images, prompt, directory, number):
        self.calls.append((tuple(image.size for image in images), prompt, number))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return {
            "text": response,
            "generated_token_ids": [number, 99],
            "input_tokens": 17,
            "output_tokens": 2,
            "forward_calls": 3,
            "chat_text": "chat",
            "input_metadata": {"image_order": ["clean", "R1"]},
        }


def _case():
    from scripts.cai_c_retrain.vlm import CaseInput

    clean = Image.new("RGB", (32, 32), "black")
    numbered = Image.new("RGB", (32, 32), "white")
    return CaseInput(
        specimen_key="domain:sample",
        dataset_id="domain",
        split="TRAIN",
        capture_group_id="group",
        impacted_surface_path="data/source.png",
        source_sha256="0" * 64,
        clean=clean,
        numbered=numbered,
        clean_sha256="1" * 64,
        numbered_sha256="2" * 64,
        render_config={"font_sha256": "3" * 64},
        signature="4" * 64,
        signature_data={"prior_version": "C_P0_R1_GLOBAL_V1"},
    )


def _valid(regions=None, no_reliable=False):
    return json.dumps(
        {
            "regions": regions
            if regions is not None
            else [
                {
                    "cells": [7, 8],
                    "cue": "mark",
                    "alternative": "stain",
                    "confidence": "medium",
                }
            ],
            "no_reliable_cue": no_reliable,
        },
        ensure_ascii=False,
    )


def test_roster_has_only_211_train_valid_in_source_order():
    from scripts.cai_c_retrain.vlm import load_roster

    rows = load_roster(ROOT / "results/cai_agent_v3/new_protocol/candidate_queue.csv")

    assert len(rows) == 211
    assert [row["split"] for row in rows].count("TRAIN") == 161
    assert [row["split"] for row in rows].count("VALID") == 50
    assert all(row["split"] != "TEST" for row in rows)
    with (
        ROOT / "results/cai_agent_v3/new_protocol/candidate_queue.csv"
    ).open() as handle:
        expected = [
            row["specimen_key"]
            for row in csv.DictReader(handle)
            if row["split"] in {"TRAIN", "VALID"}
        ]
    assert [row["specimen_key"] for row in rows] == expected


def test_exact_renderer_matches_all_six_pilot_c_hashes():
    from scripts.cai_c_retrain.vlm import render_c_inputs

    lock = json.loads((GROUNDING / "experiment_lock.json").read_text())
    jobs = [job for job in lock["jobs"] if job["variant"] == "C_P0_R1"]
    manifest = json.loads(
        (
            ROOT / "results/cai_agent_v3/new_protocol/feature_bank_manifest.json"
        ).read_text()
    )
    source_root = Path(manifest["encoder_execution_root"])
    queue = {}
    with (
        ROOT / "results/cai_agent_v3/new_protocol/candidate_queue.csv"
    ).open() as handle:
        queue = {row["specimen_key"]: row for row in csv.DictReader(handle)}

    for job in jobs:
        row = queue[job["specimen_key"]]
        with Image.open(source_root / row["impacted_surface_path"]) as source:
            rendered = render_c_inputs(source)
        assert rendered.clean_sha256 == job["signature_data"]["clean_sha256"]
        assert rendered.numbered_sha256 == job["signature_data"]["numbered_sha256"]
        assert rendered.render_config == job["signature_data"]["render_config"]


def test_no_cue_is_terminal_without_repair(tmp_path):
    from scripts.cai_c_retrain.vlm import execute_case

    backend = FakeBackend([_valid(regions=[], no_reliable=True)])
    state = execute_case(tmp_path, _case(), "P0", backend)

    assert state["status"] == "VALID_NO_RELIABLE_CUE"
    assert state["available"] is True
    assert state["no_reliable_cue"] is True
    assert len(backend.calls) == 1
    assert state["attempts"][0]["parser_valid"] is True
    assert state["attempts"][0]["contract_valid"] is True


def test_duplicate_cells_repairs_once_then_schema_failure(tmp_path):
    from scripts.cai_c_retrain.vlm import execute_case

    duplicate = _valid(
        regions=[
            {
                "cells": [7],
                "cue": "mark",
                "alternative": "stain",
                "confidence": "medium",
            },
            {
                "cells": [7],
                "cue": "line",
                "alternative": "scratch",
                "confidence": "low",
            },
        ]
    )
    backend = FakeBackend([duplicate, "still invalid"])
    state = execute_case(tmp_path, _case(), "P0", backend)

    assert state["status"] == "SCHEMA_INVALID_AFTER_ONE_REPAIR"
    assert state["available"] is False
    assert len(backend.calls) == 2
    assert state["attempts"][0]["parser_valid"] is True
    assert state["attempts"][0]["contract_valid"] is False
    assert state["attempts"][1]["kind"] == "FORMAT_REPAIR"
    execute_case(tmp_path, _case(), "P0", FakeBackend([]))


def test_interrupted_primary_gets_one_retry_and_never_a_third_call(tmp_path):
    from scripts.cai_c_retrain.vlm import execute_case

    first = FakeBackend([RuntimeError("interrupted")])
    state = execute_case(tmp_path, _case(), "P0", first)
    assert state["status"] == "INCOMPLETE"
    assert state["attempts"][0]["kind"] == "PRIMARY"

    second = FakeBackend([_valid()])
    state = execute_case(tmp_path, _case(), "P0", second)
    assert state["status"] == "VALID_AFTER_INTERRUPTED_RETRY"
    assert state["attempts"][1]["kind"] == "INTERRUPTED_RETRY"
    assert len(second.calls) == 1

    third = FakeBackend([])
    assert execute_case(tmp_path, _case(), "P0", third) == state
    assert third.calls == []


def test_changed_signature_never_reuses_terminal_state(tmp_path):
    from scripts.cai_c_retrain.vlm import execute_case

    execute_case(tmp_path, _case(), "P0", FakeBackend([_valid()]))
    with pytest.raises(ValueError, match="signature"):
        execute_case(
            tmp_path,
            replace(_case(), signature="5" * 64),
            "P0",
            FakeBackend([]),
        )


def test_feature_rows_use_loader_booleans_and_64_float32_values(tmp_path):
    from scripts.cai_c_retrain.vlm import build_feature_row, execute_case

    state = execute_case(tmp_path, _case(), "P0", FakeBackend([_valid()]))
    row = build_feature_row(_case(), state, reused_from_pilot=False)

    assert row["vlm_available"] == "True"
    assert row["no_reliable_cue"] == "False"
    indicator = np.asarray(row["region_indicator"].split(";"), dtype=np.float32)
    confidence = np.asarray(row["confidence"].split(";"), dtype=np.float32)
    assert indicator.shape == confidence.shape == (64,)
    assert indicator[[7, 8]].tolist() == [1.0, 1.0]
    assert confidence[[7, 8]].tolist() == pytest.approx(
        [np.float32(2 / 3), np.float32(2 / 3)]
    )


def test_model_config_rejects_chat_or_generation_drift():
    from scripts.cai_c_retrain.vlm import validate_model_config

    config = json.loads((GROUNDING / "experiment_lock.json").read_text())["config"]
    validate_model_config(config)

    changed = json.loads(json.dumps(config))
    changed["generation"]["max_new_tokens"] = 501
    with pytest.raises(ValueError, match="model configuration"):
        validate_model_config(changed)


def test_vlm_contract_exports_input_manifest_and_config_lock(tmp_path):
    from scripts.cai_c_retrain.vlm import _write_vlm_contract

    output = tmp_path / "output"
    vlm = output / "vlm"
    context = SimpleNamespace(
        task_id="task",
        scope={"prior_version": "C_P0_R1_GLOBAL_V1"},
        path=lambda name: vlm if name == "vlm" else output,
    )
    protocol = {
        "prompt_sha256": "5" * 64,
        "repair_sha256": "6" * 64,
        "model_config": {"image_order": ["clean", "numbered"]},
    }
    record = {
        "case": _case(),
        "state": {"status": "VALID_FIRST_PASS", "attempts": [{"kind": "PRIMARY"}]},
        "reused_from_pilot": False,
    }

    exported = _write_vlm_contract(context, protocol, [record])

    with (vlm / "input_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    config = json.loads((vlm / "config_lock.json").read_text(encoding="utf-8"))
    assert rows[0]["clean_sha256"] == "1" * 64
    assert rows[0]["numbered_sha256"] == "2" * 64
    assert rows[0]["status"] == "VALID_FIRST_PASS"
    assert config["prompt_sha256"] == "5" * 64
    assert config["render_config"]["font_sha256"] == "3" * 64
    assert config["image_order"] == ["clean", "numbered"]
    assert exported["input_manifest_sha256"]
    assert exported["config_lock_sha256"]
