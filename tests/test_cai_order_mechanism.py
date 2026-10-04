"""Focused synthetic acceptance tests for the order-mechanism pipeline.

No test in this module loads a research checkpoint or performs a model forward.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SOURCE = "37b3c40414c00c6633b64656c4bbb2b178ef9848"
BRANCH = "research/cai-vlm-agent-v3-controlled-reuse"


def _scope() -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_id": "CAI_ORDER_MECHANISM_R1_37B3C404",
        "source_commit": SOURCE,
        "branch": BRANCH,
        "task_type": "FROZEN_TRAJECTORY_ANALYSIS_AND_FIXED_FINAL_SET_ORDER_INTERVENTION",
        "roots": {
            "code": "scripts/cai_order_mechanism",
            "tests": "tests/test_cai_order_mechanism.py",
            "scope": "docs/cai/order_mechanism/ORDER_MECHANISM_SCOPE.json",
            "results": "results/cai_agent_v3/order_mechanism/r1_37b3c404",
            "artifacts": "artifacts/cai_agent_v3/order_mechanism/r1_37b3c404",
        },
        "cohort": {
            "split": "VALID",
            "physical_n": 50,
            "capture_groups": 48,
            "domains": 6,
            "test_inference_or_label_join": False,
        },
        "frozen_models": {
            "actor": {"forward_allowed": False},
            "predictor": {
                "model": "MEAN_SC",
                "selected_update": 1750,
                "forward_allowed": True,
            },
            "selection": "LOCK_EXISTING_SELECTION_NO_RESELECTION",
        },
        "resource_limits": {
            "optimizer_updates": 0,
            "actor_forwards": 0,
            "qwen_forwards": 0,
            "cnn_forwards": 0,
            "oof_predictor_forwards": 0,
            "autograd": 0,
            "test_forwards": 0,
            "predictor_evaluated_state_rows_including_padding_retries_qa": 7000,
            "max_visible_gpus": 1,
            "max_cpu_threads": 4,
        },
    }


def test_q1_scope_context_rejects_test_wrong_branch_and_path_escape(tmp_path: Path):
    """Catches accepting TEST, another branch, or a root outside the repository."""
    from scripts.cai_order_mechanism.inputs import TaskContext

    valid = _scope()
    context = TaskContext.from_mapping(tmp_path, valid, branch=BRANCH, head=SOURCE)
    assert context.results_root == tmp_path / valid["roots"]["results"]

    test_scope = copy.deepcopy(valid)
    test_scope["cohort"]["split"] = "TEST"
    with pytest.raises(ValueError, match="VALID"):
        TaskContext.from_mapping(tmp_path, test_scope, branch=BRANCH, head=SOURCE)

    with pytest.raises(ValueError, match="branch"):
        TaskContext.from_mapping(tmp_path, valid, branch="main", head=SOURCE)

    escaping = copy.deepcopy(valid)
    escaping["roots"]["results"] = "../escaped"
    with pytest.raises(ValueError, match="repository-relative"):
        TaskContext.from_mapping(tmp_path, escaping, branch=BRANCH, head=SOURCE)


def test_q2_episode_parser_enforces_action_and_trajectory_contract():
    """Catches duplicate cells and trajectories not aligned to action completions."""
    from scripts.cai_order_mechanism.analysis import parse_episode

    row = {
        "specimen_key": "domain:sample",
        "dataset_id": "domain",
        "capture_group_id": "group",
        "method": "NO_VLM_SPATIAL_FEEDBACK",
        "run": "0",
        "target_mpa": "20",
        "cells": "1;3",
        "costs": "0;0.1;0.25",
        "predictions_mpa": "10;15;18",
    }
    episode = parse_episode(row)
    assert episode.cells == (1, 3)
    np.testing.assert_array_equal(episode.costs, [0.0, 0.1, 0.25])

    duplicate = dict(row, cells="1;1")
    with pytest.raises(ValueError, match="unique"):
        parse_episode(duplicate)
    short = dict(row, predictions_mpa="10;15")
    with pytest.raises(ValueError, match="actions plus one"):
        parse_episode(short)


def test_q3_stage_boundary_negative_terms_and_identities():
    """Catches left-closed staging or loss of negative event contributions."""
    from scripts.cai_order_mechanism.analysis import timing_decomposition

    result = timing_decomposition(
        np.array([0.0, 0.0625, 0.125, 0.25]),
        np.array([10.0, 8.0, 9.0, 4.0]),
        0.0,
    )
    np.testing.assert_array_equal(result.stage_index, [0, 1, 3])
    assert result.raw_stage.sum() == pytest.approx(6.0, abs=1e-12)
    assert result.raw_negative[1] == pytest.approx(-1.0)
    assert result.weighted_stage.sum() == pytest.approx(
        result.initial_error - result.area_mpa, abs=1e-12
    )
    assert result.area_mpa == pytest.approx(9.0)


def test_q4_fixed_permutation_and_plan_are_label_free_and_keep_duplicates():
    """Catches unstable hashing, label leakage, redraws, or changed terminal sets."""
    from scripts.cai_order_mechanism.analysis import parse_episode
    from scripts.cai_order_mechanism.orders import (
        NAMESPACE,
        build_order_plan,
        fixed_permutation,
    )

    cells = [2, 7, 9]
    assert fixed_permutation("domain:sample", 0, cells, namespace=NAMESPACE) == (9, 7, 2)
    assert fixed_permutation("domain:sample", 1, cells, namespace=NAMESPACE) == (9, 7, 2)
    episode = parse_episode(
        {
            "specimen_key": "domain:sample",
            "dataset_id": "domain",
            "capture_group_id": "group",
            "method": "NO_VLM_SPATIAL_FEEDBACK",
            "run": "0",
            "target_mpa": "999",
            "cells": "2;7;9",
            "costs": "0;0.05;0.12;0.2",
            "predictions_mpa": "1;2;3;4",
        }
    )
    plan = build_order_plan([episode], namespace=NAMESPACE)
    assert [(row.variant, row.repeat) for row in plan] == [
        ("NATIVE_REPLAY", 0),
        ("REVERSE", 0),
        ("PERMUTED", 0),
        ("PERMUTED", 1),
        ("PERMUTED", 2),
        ("PERMUTED", 3),
        ("PERMUTED", 4),
    ]
    assert all(set(row.order) == {2, 7, 9} for row in plan)
    assert plan[2].order == plan[3].order


def test_q5_prefix_clock_uses_integer_native_pixels_and_fixed_positions():
    """Catches replacing unequal native cell areas with a uniform 1/64 clock."""
    from cmc_bbdm.cai_active_image.environment import NativeCellGrid
    from scripts.cai_order_mechanism.orders import prefix_states

    grid = NativeCellGrid.from_shape((15, 17))
    states = prefix_states((0, 4, 32), grid)
    assert [row.pixels for row in states] == [0, 4, 10, 12]
    np.testing.assert_allclose(
        [row.cost64 for row in states], np.array([0, 4, 10, 12]) / 255.0, rtol=0, atol=0
    )
    assert [row.mask_bits for row in states] == [0, 1, 17, 1 + 16 + (1 << 32)]
    assert all(type(row.cost32) is np.float32 for row in states)


def test_q6_cache_key_binds_every_scientific_input():
    """Catches cache reuse across protocols, models, shards, engines, specimens, masks, or costs."""
    from scripts.cai_order_mechanism.orders import CacheBinding, prefix_cache_key

    base = CacheBinding("protocol", "predictor", "shard", "engine")
    variants = [
        (base, "sample", 3, np.float32(0.1)),
        (CacheBinding("other", "predictor", "shard", "engine"), "sample", 3, np.float32(0.1)),
        (CacheBinding("protocol", "other", "shard", "engine"), "sample", 3, np.float32(0.1)),
        (CacheBinding("protocol", "predictor", "other", "engine"), "sample", 3, np.float32(0.1)),
        (CacheBinding("protocol", "predictor", "shard", "other"), "sample", 3, np.float32(0.1)),
        (base, "other", 3, np.float32(0.1)),
        (base, "sample", 4, np.float32(0.1)),
        (base, "sample", 3, np.nextafter(np.float32(0.1), np.float32(1.0))),
    ]
    keys = [prefix_cache_key(*args) for args in variants]
    assert len(set(keys)) == len(keys)


def test_q7_replay_precharges_batches_and_reuses_atomic_cache(tmp_path: Path):
    """Catches forwarding before accounting or re-evaluating a cached scientific state."""
    from scripts.cai_order_mechanism.replay import (
        PredictorCache,
        PrefixRequest,
        ReplayEngine,
        ResourceLedger,
        read_resource_events,
    )

    class SyntheticPredictor(torch.nn.Module):
        def forward(self, surface, cscan, measured, *, cost):
            signal = (surface[..., 0] * measured.float()).sum(dim=1)
            return signal + 10.0 * cost

    requests = []
    for index in range(3):
        surface = np.zeros((64, 512), dtype=np.float32)
        surface[index, 0] = index + 1
        measured = np.zeros(64, dtype=bool)
        measured[index] = True
        requests.append(
            PrefixRequest(
                cache_key=f"key-{index}",
                specimen_key="domain:sample",
                feature_shard_sha256="shard",
                surface_tokens=surface,
                cscan_tokens=np.zeros((64, 512), dtype=np.float32),
                measured_mask=measured,
                mask_bits=1 << index,
                cost32=np.float32(index / 10),
            )
        )

    cache = PredictorCache(tmp_path / "cache.npz")
    ledger = ResourceLedger(tmp_path / "events.jsonl", evaluated_row_cap=10)
    engine = ReplayEngine(
        SyntheticPredictor(), cache, ledger, device=torch.device("cpu"), batch_size=2
    )
    first = engine.evaluate(requests, kind="SYNTHETIC_TEST")
    np.testing.assert_allclose(first, [1.0, 3.0, 5.0], rtol=0, atol=1e-6)
    events = read_resource_events(ledger.path)
    assert [(row["state"], row["evaluated_rows"]) for row in events] == [
        ("INTENDED", 2),
        ("COMPLETE", 2),
        ("INTENDED", 1),
        ("COMPLETE", 1),
    ]
    assert cache.path.exists()
    before = cache.path.read_bytes()
    second = engine.evaluate(list(reversed(requests)), kind="SYNTHETIC_TEST")
    np.testing.assert_allclose(second, [5.0, 3.0, 1.0], rtol=0, atol=1e-6)
    assert cache.path.read_bytes() == before
    assert read_resource_events(ledger.path) == events


def test_q8_estimands_bootstrap_and_quality_crossings_are_not_pooled_or_oracle():
    """Catches pooled-domain A, prediction ensembling, and monotone/oracle quality assumptions."""
    from scripts.cai_order_mechanism.analysis import (
        bootstrap_domain_equal,
        compare_quality,
        domain_equal,
        endpoint_metrics,
        first_quality_crossing,
    )

    values = np.array([1.0, 3.0, 5.0, 10.0])
    domains = np.array(["a", "a", "a", "b"])
    assert domain_equal(values, domains) == pytest.approx(6.5)
    weights = np.ones((2, 4), dtype=np.int16)
    np.testing.assert_allclose(bootstrap_domain_equal(values, weights, domains), [6.5, 6.5])

    predictions = np.array([[0.0, 4.0], [4.0, 4.0]])
    targets = np.array([0.0, 0.0])
    metrics = endpoint_metrics(predictions, targets)
    assert metrics["mae_mpa"] == pytest.approx(3.0)
    assert metrics["rmse_mpa"] == pytest.approx(np.sqrt(12.0))
    assert metrics["r2"] is None

    crossing = first_quality_crossing(
        np.array([0.0, 0.1, 0.2, 0.25]), np.array([50.0, 40.0, 45.0, 39.0]), 41.0
    )
    assert crossing == {
        "cost": 0.1,
        "value": 40.0,
        "status": "REACHED",
        "later_recrosses_target": True,
    }
    comparison = compare_quality(
        np.array([0.0, 0.1, 0.2]),
        main_values=np.array([50.0, 45.0, 40.0]),
        comparator_values=np.array([50.0, 40.0, 39.0]),
        target=41.0,
    )
    assert comparison["absolute_saving"] == pytest.approx(-0.1)
    assert comparison["relative_saving"] == pytest.approx(-1.0)


def test_q9_phase_state_checks_hashes_and_limits_resume(tmp_path: Path):
    """Catches stale phase reuse, phase reordering, or more than one recovery attempt."""
    from scripts.cai_order_mechanism.run import PHASES, PhaseStore, ordered_phases

    assert ordered_phases("all") == PHASES
    assert ordered_phases("infer") == ("infer",)
    output = tmp_path / "result.csv"
    output.write_text("a\n1\n", encoding="utf-8")
    store = PhaseStore(tmp_path / "task_state.json")
    store.complete("derive", "signature", [output])
    assert store.reusable("derive", "signature")
    output.write_text("a\n2\n", encoding="utf-8")
    assert not store.reusable("derive", "signature")

    stable = tmp_path / "predictor_cache.npz"
    stable.write_bytes(b"cache")
    mutable = tmp_path / "resource_usage.json"
    mutable.write_text("{}\n", encoding="utf-8")
    store.complete("infer", "infer-signature", [stable, mutable])
    mutable.write_text('{"report_render_count": 1}\n', encoding="utf-8")
    assert store.reusable("infer", "infer-signature")
    stable.write_bytes(b"changed")
    assert not store.reusable("infer", "infer-signature")

    assert store.claim_resume() == 1
    with pytest.raises(ValueError, match="resume"):
        store.claim_resume()
    state = json.loads(store.path.read_text(encoding="utf-8"))
    assert state["resume_attempts"] == 1


def test_q10_delivery_inventory_is_complete_offline_and_fixed_case_only(tmp_path: Path):
    """Catches a locally incomplete report, remote dependencies, or cherry-picked cases."""
    from scripts.cai_order_mechanism.verify import (
        REQUIRED_ARTIFACT_FILES,
        REQUIRED_RESULT_FILES,
        verify_delivery_inventory,
    )

    results = tmp_path / "results"
    artifacts = tmp_path / "artifacts"
    results.mkdir()
    artifacts.mkdir()
    for relative in REQUIRED_RESULT_FILES:
        path = results / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    for relative in REQUIRED_ARTIFACT_FILES:
        path = artifacts / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("证" * 650 if path.name == "FINDINGS_ZH.md" else "evidence\n", encoding="utf-8")
    (results / "index.html").write_text(
        "<!doctype html><html><body><a href='stage_summary.csv'>table</a></body></html>",
        encoding="utf-8",
    )
    figures = results / "figures"
    figures.mkdir()
    for stem in ("F1_archived_stages", "F2_order_curves"):
        (figures / f"{stem}.png").write_bytes(b"png")
        (figures / f"{stem}.svg").write_text("<svg/>", encoding="utf-8")
    cases = results / "cases"
    cases.mkdir()
    for stem in ("74t7kcdgkr_c8-16", "cgtnjyggtm_q24-48", "w68dtmpfyf_q16-29"):
        (cases / f"{stem}.png").write_bytes(b"png")
        (cases / f"{stem}.svg").write_text("<svg/>", encoding="utf-8")

    report = verify_delivery_inventory(results, artifacts)
    assert report["figure_families"] == 2
    assert report["case_figures"] == 3
    (results / "index.html").write_text("<script src='https://cdn.example/x.js'></script>")
    with pytest.raises(ValueError, match="offline"):
        verify_delivery_inventory(results, artifacts)
