"""Only new W2 replay wiring; no real optimization or Actor forward."""

import pytest


def test_replay_stage_b_allocation_after_a_is_not_twelve_thousand_again():
    from cmc_bbdm.cai_agent_v3.w2_replay import check_allocation

    # Historical 22264 + full A6000 leaves exactly B6000 under34264.
    check_allocation(global_used=28264, replay_used=6000, stage_used=0, required=6000)
    with pytest.raises(ValueError, match="RESOURCE_LIMITED"):
        check_allocation(
            global_used=28265, replay_used=6001, stage_used=0, required=6000
        )


def test_output_root_cannot_alias_historical_results(tmp_path):
    from cmc_bbdm.cai_agent_v3.w2_replay import resolve_output

    with pytest.raises(ValueError):
        resolve_output(tmp_path, "results/cai_agent_v3/new_protocol")
    path = resolve_output(tmp_path, "results/cai_agent_v3/w2_replay/r1_292b1c74")
    assert path.is_relative_to(tmp_path)


def test_replay_usage_counts_each_reservation_once():
    from cmc_bbdm.cai_agent_v3.w2_replay import update_usage

    rows = [
        {
            "run_id": "a",
            "actual_optimizer_updates": 0,
            "optimizer_update_reservation": 2000,
        },
        {"run_id": "a", "status": "COMPLETED", "actual_optimizer_updates": 1500},
    ]
    assert update_usage(rows) == 1500
    rows.append(
        {
            "run_id": "b",
            "actual_optimizer_updates": 0,
            "optimizer_update_reservation": 2000,
        }
    )
    assert update_usage(rows) == 3500


def test_replay_training_wiring_saves_same_forward_details_and_reuses_job(
    tmp_path, monkeypatch
):
    import json

    import torch
    from test_cai_agent_v3_w2_recovery import _bank

    from cmc_bbdm.cai_agent_v3 import predictor_training as t
    from cmc_bbdm.cai_agent_v3.checkpoint_selection import validation_identity
    from cmc_bbdm.cai_agent_v3.w2_replay import ART, REPLAY_ID, RUN, ReplayContext

    bank = _bank()
    library = t.build_validation_library(bank, t._cell_costs(bank))
    output = tmp_path / RUN
    output.mkdir(parents=True)
    (tmp_path / ART).mkdir(parents=True)
    ledger = tmp_path / "results/cai_agent_v3/compute_ledger.jsonl"
    ledger.write_text('{"actual_optimizer_updates":22264}\n')
    (output / "replay_authorization.json").write_text(
        json.dumps({"replay_id": REPLAY_ID, "budget": {"new_cumulative_cap": 34264}})
    )
    (output / "input_reuse_manifest.json").write_text(
        json.dumps({"validation_identity": validation_identity(bank, library)})
    )
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self: None)
    monkeypatch.setattr(t, "_BATCH_SIZE", 2)
    context = ReplayContext(tmp_path, "A")
    saved_updates = []
    original_save = context.save_details

    def save_once(key, update, details):
        assert update not in saved_updates, "winner reload must not overwrite candidate predictions"
        saved_updates.append(update)
        original_save(key, update, details)

    monkeypatch.setattr(context, "save_details", save_once)
    kwargs = {
        "device": "cpu",
        "max_updates": 2,
        "validation_interval": 1,
        "archive_dir": output / "models/selection_history/toy",
        "run_context": context,
        "job_key": "MEAN_SC",
    }
    _model, manifest, _progress = t._train_candidate(
        "MEAN_SC",
        bank,
        library,
        t._cell_costs(bank),
        t._constant_metrics(bank),
        **kwargs,
    )
    assert saved_updates == [1, 2]
    assert (
        len(list((output / "candidate_state_predictions/A_MEAN_SC").glob("*.npz"))) == 2
    )
    saved = torch.load(
        output / "models/selection_history/toy/latest_training_state.pt",
        weights_only=False,
    )
    assert {
        "optimizer",
        "numpy_generator_state",
        "torch_cpu_rng",
        "torch_cuda_rng",
        "best_update",
        "stale",
        "run_id",
    }.issubset(saved)
    before = ledger.read_bytes()
    _, again, _ = t._train_candidate(
        "MEAN_SC",
        bank,
        library,
        t._cell_costs(bank),
        t._constant_metrics(bank),
        **kwargs,
    )
    assert again["selected_update"] == manifest["selected_update"]
    assert ledger.read_bytes() == before
    assert not (tmp_path / "results/cai_agent_v3/new_protocol").exists()
