import json
from pathlib import Path

import pytest

from scripts.cai_actor_c0_diagnostic.context import (
    BRANCH,
    SOURCE_COMMIT,
    TASK_ID,
    ResourceLedger,
    TaskContext,
    canonical_json,
)


def scope(tmp_path: Path) -> dict:
    return {
        "schema_version": 2,
        "task_id": TASK_ID,
        "branch": BRANCH,
        "source_commit": SOURCE_COMMIT,
        "scope": "FROZEN_POLICY_DIAGNOSTICS_NOT_TRAINING_OR_PAPER_REVISION",
        "roots": {
            "data": "results/data",
            "c_release": "results/c",
            "historical_w3": "results/n",
            "predictor": "results/p.pt",
            "code": "scripts/cai_actor_c0_diagnostic",
            "output": "results/out",
            "artifacts": "artifacts/out",
            "config": "docs/cai/scope.json",
        },
        "models": {
            "C": {"method": "VLM_SPATIAL_FEEDBACK"},
            "N": {"method": "NO_VLM_SPATIAL_FEEDBACK"},
            "P_all_sha256": "a" * 64,
        },
        "cases": {"split": "VALID", "max_specimens": 6},
        "limits": {
            "optimizer_updates": 0,
            "new_qwen_generations": 0,
            "new_qwen_forwards": 0,
            "cnn_forwards": 0,
            "oof_predictor_forwards": 0,
            "test_access": 0,
            "paper_writes": 0,
            "bootstrap_draws": 0,
            "actor_forward_examples_including_failures_max": 4,
            "predictor_forward_examples_including_failures_max": 3,
            "autograd_gradient_queries_max": 2,
            "native_full_episode_runs": 12,
            "c_no_c0_full_episode_runs": 6,
            "all_full_episode_runs_max": 18,
            "cpu_threads_max": 4,
            "gpu_devices_max": 1,
        },
        "delivery": {"actual_commit_push": True, "same_branch": True},
    }


def test_canonical_json_is_stable():
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_context_rejects_wrong_identity_and_escaping_root(tmp_path):
    value = scope(tmp_path)
    with pytest.raises(ValueError, match="task identity"):
        TaskContext.from_mapping(
            tmp_path,
            {**value, "task_id": "wrong"},
            tmp_path / "scope.json",
            branch=BRANCH,
            head=SOURCE_COMMIT,
        )
    value["roots"]["output"] = "../escape"
    with pytest.raises(ValueError, match="repository-relative"):
        TaskContext.from_mapping(
            tmp_path,
            value,
            tmp_path / "scope.json",
            branch=BRANCH,
            head=SOURCE_COMMIT,
        )


def test_context_rejects_test_or_nonzero_forbidden_work(tmp_path):
    value = scope(tmp_path)
    value["cases"]["split"] = "TEST"
    with pytest.raises(ValueError, match="VALID"):
        TaskContext.from_mapping(
            tmp_path,
            value,
            tmp_path / "scope.json",
            branch=BRANCH,
            head=SOURCE_COMMIT,
        )
    value = scope(tmp_path)
    value["limits"]["optimizer_updates"] = 1
    with pytest.raises(ValueError, match="forbidden resource"):
        TaskContext.from_mapping(
            tmp_path,
            value,
            tmp_path / "scope.json",
            branch=BRANCH,
            head=SOURCE_COMMIT,
        )


def test_resource_ledger_counts_examples_and_never_resets(tmp_path):
    path = tmp_path / "usage.json"
    ledger = ResourceLedger(path, scope(tmp_path)["limits"])
    ledger.charge("actor_forward_examples", 3)
    ResourceLedger(path, scope(tmp_path)["limits"]).charge(
        "actor_forward_examples", 1
    )
    saved = json.loads(path.read_text())
    assert saved["counts"]["actor_forward_examples"] == 4
    with pytest.raises(ValueError, match="actor_forward_examples"):
        ResourceLedger(path, scope(tmp_path)["limits"]).charge(
            "actor_forward_examples", 1
        )


def test_stage_completion_is_signature_bound(tmp_path):
    value = scope(tmp_path)
    context = TaskContext.from_mapping(
        tmp_path,
        value,
        tmp_path / "scope.json",
        branch=BRANCH,
        head=SOURCE_COMMIT,
    )
    context.complete_stage("prepare", "abc", {"case_count": 6})
    assert context.stage_complete("prepare", "abc")
    assert not context.stage_complete("prepare", "changed")

