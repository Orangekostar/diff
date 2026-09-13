"""W3 bounded checks; synthetic data, no real optimizer updates."""

import numpy as np
import pytest
import torch


def test_seed_before_construction_is_independent_of_prior_rng():
    from cmc_bbdm.cai_agent_v3.actor_training import _state_dict_sha256, seeded_actor

    for name in ("VLM_SPATIAL_FEEDBACK", "VLM_MEAN_FEEDBACK", "LEARNED_STATIC_TRUE"):
        a, _ = seeded_actor(
            name, target_mean=200.0, target_scale=10.0, seed=71, device="cpu"
        )
        torch.randn(321)
        b, _ = seeded_actor(
            name, target_mean=200.0, target_scale=10.0, seed=71, device="cpu"
        )
        assert _state_dict_sha256(a.state_dict()) == _state_dict_sha256(b.state_dict())


def test_task_cap_does_not_spend_w2_remainder_or_allow_method_overrun():
    from cmc_bbdm.cai_agent_v3.w3_pilot import check_budget

    check_budget(
        global_used=33264, task_used=0, method_used=0, required=1250, method_cap=1250
    )
    with pytest.raises(ValueError, match="RESOURCE_LIMITED"):
        check_budget(
            global_used=39014,
            task_used=5750,
            method_used=0,
            required=1,
            method_cap=1250,
        )
    with pytest.raises(ValueError, match="RESOURCE_LIMITED"):
        check_budget(
            global_used=33264,
            task_used=0,
            method_used=1250,
            required=1,
            method_cap=1250,
        )


def test_actor_archive_allows_different_paths_and_keeps_loser(tmp_path):
    from cmc_bbdm.cai_agent_v3.actor_selection import (
        ActorArchive,
        inspect_actor_archive,
    )

    archive = ActorArchive(
        tmp_path / "run",
        method="LEARNED_STATIC_TRUE",
        seed=4,
        max_updates=500,
        environment={"identity": "toy"},
        run_id="toy",
    )

    def episode(cells, costs, predictions):
        return {
            "specimen_key": "a",
            "dataset_id": "d",
            "capture_group_id": "g",
            "method": "LEARNED_STATIC_TRUE",
            "run": 0,
            "target_mpa": 200.0,
            "cells": cells,
            "costs": costs,
            "predictions_mpa": predictions,
        }

    archive.record(
        250, {"x": torch.tensor([1.0])}, [episode("1;2", "0;.125;.25", "190;194;198")]
    )
    archive.record(500, {"x": torch.tensor([2.0])}, [episode("3", "0;.25", "198;198")])
    selected = archive.finish(500)
    assert selected["update"] == 500
    assert (
        inspect_actor_archive(tmp_path / "run", environment={"identity": "toy"})[
            "selected_update"
        ]
        == 500
    )
    (tmp_path / "run/update_000250.pt").unlink()
    with pytest.raises(ValueError):
        inspect_actor_archive(tmp_path / "run", environment={"identity": "toy"})


def test_repeat_loss_and_domain_mean_are_not_ensemble_or_pooled():
    from cmc_bbdm.cai_agent_v3.w3_results import pooled_errors

    rows = [
        {
            "specimen_key": "a",
            "run": r,
            "target_mpa": 200.0,
            "predictions_mpa": str(p),
            "costs": "0",
        }
        for r, p in enumerate([190.0, 210.0])
    ]
    assert pooled_errors(rows, budget=0.25)["mae_mpa"] == 10.0
    from cmc_bbdm.cai_agent_v3.actor_training import _domain_equal_episode_score

    rows = [
        {"specimen_key": str(i), "dataset_id": "a" if i < 4 else "b", "error": v}
        for i, v in enumerate([1, 3, 5, 10, 14])
    ]
    assert _domain_equal_episode_score(rows, "error") == 9.375


def test_hand_computed_cost_to_go_and_early_no_future():
    from cmc_bbdm.cai_agent_v3.metrics import (
        left_error_area_mpa,
        torch_policy_cost_to_go,
    )

    for costs, preds, expected in [
        ([0, 0.125, 0.25], [190.0, 194.0, 198.0], [8.5, 3.5]),
        ([0, 0.1, 0.2], [190.0, 196.0, 198.0], [6.5, 2.5]),
    ]:
        returns = torch_policy_cost_to_go(
            torch.tensor(costs, dtype=torch.float64),
            torch.tensor(preds, dtype=torch.float64),
            torch.tensor(200.0),
            budget=0.25,
        )
        np.testing.assert_allclose(returns.numpy(), expected, rtol=0, atol=1e-12)
    assert (
        left_error_area_mpa([0, 0.125, 0.25], [190, 194, 198], 200, end=0.0625) == 10.0
    )


def test_w3_training_wiring_and_completed_reuse_without_optimizer(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    from test_cai_agent_v3_w2_recovery import _bank

    from cmc_bbdm.cai_agent_v3 import actor_training as t
    from cmc_bbdm.cai_agent_v3.actor_selection import read_episodes
    from cmc_bbdm.cai_agent_v3.w3_pilot import ART, RUN, TASK_ID, PilotContext

    class CountPredictor(torch.nn.Module):
        def forward(self, surface, cscan, measured, *, cost):
            return 200.0 + measured.float().sum(1)

    bank = _bank()
    out = tmp_path / RUN
    out.mkdir(parents=True)
    (tmp_path / ART).mkdir(parents=True)
    (out / "protocol_snapshot.json").write_text(json.dumps({"identity": "toy"}))
    (out / "pilot_authorization.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "budget": {"new_W3_updates_cap_including_failures": 5750},
                "models": [
                    {
                        "method": "LEARNED_STATIC_TRUE",
                        "max_updates": 1,
                        "training_seed": 42,
                    }
                ],
            }
        )
    )
    ledger = tmp_path / "results/cai_agent_v3/compute_ledger.jsonl"
    ledger.write_text('{"actual_optimizer_updates":33264}\n')
    import subprocess

    original = subprocess.check_output
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: (
            "toy\n"
            if args[0] == ["git", "rev-parse", "HEAD"]
            else original(*args, **kwargs)
        ),
    )
    context = PilotContext(tmp_path)
    monkeypatch.setattr(context, "preflight", lambda: None)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self: None)
    monkeypatch.setattr(t, "_BATCH_SIZE", 2)
    features = SimpleNamespace(
        indicator=np.zeros((3, 64), np.float32),
        confidence=np.zeros((3, 64), np.float32),
        available=np.zeros(3, bool),
        no_reliable=np.zeros(3, bool),
    )
    args = (
        "LEARNED_STATIC_TRUE",
        1,
        bank,
        features,
        CountPredictor(),
        {0: CountPredictor()},
        np.zeros(3, np.int64),
        t._cell_costs(bank),
    )
    _, manifest, _, _episodes = t._train_actor(
        *args, seed_panel=1, training_seed=42, device="cpu", run_context=context
    )
    stored = read_episodes(tmp_path / manifest["selected_episodes"])
    assert len(stored) == 2
    for row in stored:
        trace = json.loads(row["execution_trace"])
        cells = [int(c) for c in row["cells"].split(";")]
        assert [t["cell"] for t in trace] == cells
        assert [t["actor_call_index"] for t in trace] == list(range(1, len(cells) + 1))
        assert sum(t["new_pixels"] for t in trace) <= 24
        assert all(t["environment_legal"][t["cell"]] == "1" for t in trace)
    before = ledger.read_bytes()
    rng_before = torch.get_rng_state().clone()
    _, again, _, _ = t._train_actor(
        *args, seed_panel=1, training_seed=42, device="cpu", run_context=context
    )
    assert again["selected_update"] == manifest["selected_update"]
    assert ledger.read_bytes() == before and torch.equal(
        rng_before, torch.get_rng_state()
    )
    assert not (tmp_path / "results/cai_agent_v3/new_protocol").exists()


def test_cost_minimization_gradient_and_visible_content_location_path():
    logits = torch.zeros(2, requires_grad=True)
    probabilities = logits.softmax(0)
    loss = (
        probabilities.detach() * logits.log_softmax(0) * torch.tensor([2.0, 8.0])
    ).sum()
    loss.backward()
    assert logits.grad.tolist() == pytest.approx([-1.5, 1.5])
    from test_cai_agent_v3 import _actor_inputs

    from cmc_bbdm.cai_agent_v3.models import SpatialCAIActor

    torch.manual_seed(91)
    model = SpatialCAIActor(use_vlm=True, use_feedback=True).eval()
    inputs = _actor_inputs()
    inputs["measured"][0, 1] = True
    inputs["cscan"].requires_grad_(True)
    scores, _ = model(**inputs)
    scores[0, 2].backward()
    assert inputs["cscan"].grad[0, :2].abs().sum() > 0
    with torch.no_grad():
        swapped = inputs["cscan"].detach().clone()
        swapped[:, [0, 1]] = swapped[:, [1, 0]]
        changed = dict(inputs, cscan=swapped)
        assert not torch.equal(scores[:, 2:], model(**changed)[0][:, 2:])


def test_old_predictor_source_is_rejected(tmp_path):
    from cmc_bbdm.cai_agent_v3.w3_pilot import source_root

    with pytest.raises(ValueError, match="valid W2"):
        source_root(tmp_path, "results/cai_agent_v3/new_protocol")


def test_evaluation_label_is_scoring_only_and_fixed_trace_has_no_actor_call():
    from types import SimpleNamespace

    from test_cai_agent_v3_w2_recovery import _bank

    from cmc_bbdm.cai_agent_v3 import actor_training as t
    from cmc_bbdm.cai_agent_v3.models import MeanSCPredictor

    bank = _bank()
    features = SimpleNamespace(
        indicator=np.zeros((3, 64), np.float32),
        confidence=np.zeros((3, 64), np.float32),
        available=np.zeros(3, bool),
        no_reliable=np.zeros(3, bool),
    )
    predictor = MeanSCPredictor().eval().requires_grad_(False)
    trace = []
    args = (None, "CENTER_FIRST", predictor, bank, features, t._cell_costs(bank), 1)
    first = t._evaluate_one(
        *args, device="cpu", target_mpa=200.0, action_callback=trace.append
    )
    second = t._evaluate_one(*args, device="cpu", target_mpa=400.0)
    assert first[1:] == second[1:]
    assert all(
        row["actor_call_index"] is None and row["c0_reason"] == "FIXED_ORDER"
        for row in trace
    )
