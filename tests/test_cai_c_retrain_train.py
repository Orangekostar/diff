from __future__ import annotations

import random

import numpy as np
import pytest
import torch


def _model_and_optimizer():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    return model, optimizer


def test_tie_within_tolerance_keeps_earliest_candidate():
    from scripts.cai_c_retrain.train import Candidate, select_candidate

    candidates = [
        Candidate(update=250, score=7.0),
        Candidate(update=500, score=7.0 - 5e-13),
        Candidate(update=750, score=6.9),
        Candidate(update=1000, score=6.9),
    ]

    assert select_candidate(candidates).update == 750


def test_snapshot_resume_restores_all_rng_and_optimizer_state(tmp_path):
    from scripts.cai_c_retrain.train import restore_snapshot, save_snapshot

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    rng = np.random.default_rng(14)
    model, optimizer = _model_and_optimizer()
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    path = tmp_path / "snapshot.pt"
    save_snapshot(
        path,
        model,
        optimizer,
        rng,
        logical_update=250,
        best_score=4.5,
        best_update=250,
        stale=0,
        progress=[{"update": 250}],
        environment_identity="env",
        input_signature="input",
        run_id="run",
    )
    expected = (
        random.random(),
        float(np.random.random()),
        float(rng.random()),
        torch.rand(3),
    )
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    rng = np.random.default_rng(94)
    restored = restore_snapshot(
        path,
        model,
        optimizer,
        rng,
        expected_environment="env",
        expected_signature="input",
    )

    observed = (
        random.random(),
        float(np.random.random()),
        float(rng.random()),
        torch.rand(3),
    )
    assert restored["logical_update"] == 250
    assert observed[:3] == pytest.approx(expected[:3])
    torch.testing.assert_close(observed[3], expected[3], rtol=0, atol=0)
    assert optimizer.state_dict()["state"]


def test_changed_input_signature_is_not_resumable(tmp_path):
    from scripts.cai_c_retrain.train import restore_snapshot, save_snapshot

    model, optimizer = _model_and_optimizer()
    rng = np.random.default_rng(3)
    path = tmp_path / "snapshot.pt"
    save_snapshot(
        path,
        model,
        optimizer,
        rng,
        logical_update=0,
        best_score=float("inf"),
        best_update=0,
        stale=0,
        progress=[],
        environment_identity="env",
        input_signature="one",
        run_id="run",
    )

    with pytest.raises(ValueError, match="signature"):
        restore_snapshot(
            path,
            model,
            optimizer,
            rng,
            expected_environment="env",
            expected_signature="two",
        )


def test_lost_segment_and_replay_are_each_charged_once(tmp_path):
    from scripts.cai_c_retrain.train import SegmentLedger

    local = tmp_path / "segments.jsonl"
    global_ledger = tmp_path / "global.jsonl"
    ledger = SegmentLedger(local, global_ledger, task_id="task", method_cap=1500)
    first = ledger.reserve("METHOD", 0, 250, "run")
    ledger.complete(first, actual_updates=250)
    lost = ledger.reserve("METHOD", 250, 500, "run")
    assert ledger.resume_open("METHOD", "run") == (250, 500)
    replay = ledger.reserve("METHOD", 250, 500, "run")
    ledger.complete(replay, actual_updates=250)

    summary = ledger.summary("METHOD")
    assert summary["logical_update"] == 500
    assert summary["actual_optimizer_updates"] == 500
    assert summary["lost_updates_upper_bound"] == 250
    assert summary["charged_upper_bound"] == 750
    assert summary["resume_count"] == 1
    global_rows = global_ledger.read_text().splitlines()
    assert len(global_rows) == 3
    assert len(set(global_rows)) == 3
    assert lost != replay


def test_second_resume_is_rejected(tmp_path):
    from scripts.cai_c_retrain.train import SegmentLedger

    ledger = SegmentLedger(
        tmp_path / "segments.jsonl",
        tmp_path / "global.jsonl",
        task_id="task",
        method_cap=1500,
    )
    ledger.reserve("METHOD", 0, 250, "run")
    ledger.resume_open("METHOD", "run")
    ledger.reserve("METHOD", 0, 250, "run")

    with pytest.raises(ValueError, match="resume"):
        ledger.resume_open("METHOD", "run")


def test_disk_reloaded_weights_reproduce_outputs(tmp_path):
    from scripts.cai_c_retrain.train import load_weight_state, save_weight_state

    torch.manual_seed(8)
    model, _ = _model_and_optimizer()
    inputs = torch.randn(5, 3)
    expected = model(inputs).detach()
    path = tmp_path / "candidate.pt"
    save_weight_state(
        path,
        model,
        method="METHOD",
        update=750,
        environment_identity="env",
        input_signature="input",
    )
    replacement, _ = _model_and_optimizer()
    metadata = load_weight_state(
        path,
        replacement,
        expected_method="METHOD",
        expected_environment="env",
        expected_signature="input",
    )

    assert metadata["update"] == 750
    torch.testing.assert_close(replacement(inputs), expected, rtol=0, atol=0)


def test_global_resume_only_resumes_existing_method_directories(tmp_path):
    from scripts.cai_c_retrain.train import _resume_mode

    existing = tmp_path / "existing"
    existing.mkdir()

    assert _resume_mode(True, explicit_method=False, method_dir=existing) is True
    assert (
        _resume_mode(True, explicit_method=False, method_dir=tmp_path / "new") is False
    )
    with pytest.raises(ValueError, match="no incomplete"):
        _resume_mode(True, explicit_method=True, method_dir=tmp_path / "new")
