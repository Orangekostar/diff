"""Zero-optimization recovery checks; all checkpoints live under tmp_path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cmc_bbdm.cai_agent_v3 import predictor_training as training


def _bank():
    return SimpleNamespace(
        specimen_keys=("train:a", "valid:b", "valid:c"),
        dataset_ids=("d", "d", "d"),
        capture_group_ids=("a", "b", "c"),
        splits=("TRAIN", "VALID", "VALID"),
        targets_mpa=np.array([200.0, 200.0, 200.0], dtype=np.float32),
        native_shapes=np.array([[9, 11]] * 3),
        surface_tokens=np.zeros((3, 64, 512), dtype=np.float32),
        cscan_tokens=np.zeros((3, 64, 512), dtype=np.float32),
        indices=lambda split: np.array([0] if split == "TRAIN" else [1, 2]),
    )


def _selection():
    from cmc_bbdm.cai_agent_v3 import checkpoint_selection

    return checkpoint_selection


def test_archive_preserves_losing_weights_and_replays_all_candidates(tmp_path):
    selection = _selection()
    bank = _bank()
    library = training.build_validation_library(bank, training._cell_costs(bank))
    binding = selection.validation_identity(bank, library)
    archive = selection.CheckpointArchive(
        tmp_path / "run",
        model_name="MEAN_SC",
        validation=binding,
        max_updates=750,
        validation_interval=250,
        validation_patience=4,
        run_metadata={"seed": 7, "fit_specimen_keys": ["train:a"]},
    )
    for update, prediction in [(250, 190.0), (500, 198.0), (750, 194.0)]:
        archive.record(
            update,
            {"value": torch.tensor(prediction)},
            {"valid_area_mpa": abs(prediction - 200.0)},
        )
    summary = archive.finish(updates_completed=750)
    assert summary["selected_update"] == 500
    seen = []

    def score(payload):
        seen.append(payload["manifest"]["update"])
        return {"valid_area_mpa": abs(float(payload["state_dict"]["value"]) - 200.0)}

    result = selection.rescore_archive(
        tmp_path / "run", validation=binding, score=score
    )
    assert seen == [250, 500, 750]
    assert result["selected_update"] == 500
    assert [r["metrics"]["valid_area_mpa"] for r in result["candidates"]] == [
        10.0,
        2.0,
        6.0,
    ]
    assert len(list((tmp_path / "run").glob("update_*.pt"))) == 3
    (tmp_path / "run" / "update_000250.pt").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        selection.rescore_archive(tmp_path / "run", validation=binding, score=score)


def test_identity_binds_masks_costs_targets_tokens_and_specimen_order():
    selection = _selection()
    bank = _bank()
    library = training.build_validation_library(bank, training._cell_costs(bank))
    original = selection.validation_identity(bank, library)
    bank.cscan_tokens[1, 0, 0] = 1
    assert selection.validation_identity(bank, library) != original
    bank.cscan_tokens[1, 0, 0] = 0
    bank.targets_mpa[1] += 1
    assert selection.validation_identity(bank, library) != original
    bank.targets_mpa[1] -= 1
    library.masks[1, 0] = ~library.masks[1, 0]
    assert selection.validation_identity(bank, library) != original


def test_archive_rejects_changed_validation_and_incomplete_history(tmp_path):
    selection = _selection()
    archive = selection.CheckpointArchive(
        tmp_path / "run",
        model_name="MEAN_SC",
        validation={"identity": "one"},
        max_updates=750,
        validation_interval=250,
        validation_patience=4,
        run_metadata={"seed": 7},
    )
    archive.record(250, {"x": torch.tensor(1.0)}, {"valid_area_mpa": 1.0})
    with pytest.raises(ValueError, match="incomplete|schedule"):
        archive.finish(updates_completed=750)
    with pytest.raises(ValueError):
        selection.rescore_archive(
            tmp_path / "run",
            validation={"identity": "two"},
            score=lambda _: {"valid_area_mpa": 0.0},
        )


def test_native_nondivisible_prefix_matches_integer_pixels():
    bank = _bank()
    costs = training._cell_costs(bank)
    library = training.build_validation_library(bank, costs)
    # Frozen nearest-even grid widths (not floor partitioning).
    # 9x11 -> 99 pixels; hard quarter budget can purchase at most 24 pixels.
    for index, mask, cost in zip(
        library.specimen_indices, library.masks, library.costs
    ):
        pixels = sum(
            ([1, 1, 1, 1, 2, 1, 1, 1][r]) * ([1, 2, 1, 2, 1, 1, 2, 1][c])
            for r in range(8)
            for c in range(8)
            if mask[r * 8 + c]
        )
        assert cost == pytest.approx(pixels / 99, abs=1e-15)
        assert pixels <= 24
        assert np.sum(costs[index] * mask) == pytest.approx(cost, abs=1e-15)


def test_training_loop_archives_each_synthetic_selection_point_without_updates(
    tmp_path, monkeypatch
):
    # Exercise production loop wiring with optimizer.step explicitly disabled.
    # These counters simulate selection times, not real optimization updates.
    bank = _bank()
    costs = training._cell_costs(bank)
    library = training.build_validation_library(bank, costs)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self: None)
    monkeypatch.setattr(training, "_BATCH_SIZE", 2)
    model, manifest, progress = training._train_candidate(
        "MEAN_SC",
        bank,
        library,
        costs,
        training._constant_metrics(bank),
        device="cpu",
        max_updates=3,
        validation_interval=1,
        archive_dir=tmp_path / "synthetic",
    )
    assert len(progress) == 3
    assert manifest["selected_update"] == 1
    replay = training.rescore_predictor_archive(
        tmp_path / "synthetic", bank, library, training._constant_metrics(bank)
    )
    assert replay["selected_update"] == 1
    assert [r["update"] for r in replay["candidates"]] == [1, 2, 3]
    restored, saved = training.load_predictor_checkpoint(
        tmp_path / "synthetic" / "update_000001.pt", device="cpu"
    )
    for key in model.state_dict():
        assert torch.equal(model.state_dict()[key], restored.state_dict()[key])
    assert saved["validation"] == manifest["validation_identity"]
    manifest["checkpoint_path"] = str(tmp_path / "synthetic" / "update_000001.pt")
    manifest["checkpoint_sha256"] = training.sha256_file(manifest["checkpoint_path"])
    assert training._manifest_selection_verified(tmp_path, manifest, bank, library)
    library.costs[1] += 0.0001
    assert not training._manifest_selection_verified(tmp_path, manifest, bank, library)


def test_real_model_checkpoint_scoring_has_independent_constant_error(tmp_path):
    selection = _selection()
    bank = _bank()
    library = training.build_validation_library(bank, training._cell_costs(bank))
    constants = training._constant_metrics(bank)
    archive = selection.CheckpointArchive(
        tmp_path / "constants",
        model_name="MEAN_SC",
        validation=selection.validation_identity(bank, library),
        max_updates=500,
        validation_interval=250,
        validation_patience=4,
        run_metadata={"seed": 0},
    )
    for update, prediction in [(250, 190.0), (500, 198.0)]:
        model = training._model("MEAN_SC", constants)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            model.target_mean.fill_(prediction)
        metrics = training.evaluate_predictor(
            model, bank, library, constants, device="cpu"
        )
        assert metrics["valid_area_mpa"] == pytest.approx(200.0 - prediction)
        archive.record(update, model.state_dict(), metrics)
    archive.finish(updates_completed=500)
    replay = training.rescore_predictor_archive(
        tmp_path / "constants", bank, library, constants
    )
    assert replay["selected_update"] == 500
    assert [
        r["metrics"]["valid_area_mpa"] for r in replay["candidates"]
    ] == pytest.approx([10.0, 2.0])


def test_insufficient_cumulative_budget_stops_w2_before_loading_data(
    tmp_path, monkeypatch
):
    path = tmp_path / "results/cai_agent_v3/compute_ledger.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"actual_optimizer_updates":21514}\n'
        '{"actual_optimizer_updates":null,"actual_optimizer_updates_upper_bound":750}\n'
    )

    def forbidden(**kwargs):
        raise AssertionError("must not load training data")

    monkeypatch.setattr(training, "load_feature_bank", forbidden)
    with pytest.raises(ValueError, match="RESOURCE_LIMITED"):
        training.run_predictor_candidates(project_root=tmp_path, device="cpu")
    assert not (path.parent / "new_protocol").exists()


def test_unverified_real_stage_gates_stop_before_data_or_optimization(
    tmp_path, monkeypatch
):
    import json

    from cmc_bbdm.cai_agent_v3 import actor_training, gdfs_training

    output = tmp_path / "results/cai_agent_v3/new_protocol"
    output.mkdir(parents=True)
    payloads = {
        "predictor_gate.json": {
            "status": "PREDICTOR_NOT_READY",
            "selected_p_all": None,
        },
        "oof_readiness.json": {"status": "REWARD_MODELS_NOT_READY"},
        "vlm_manifest_fit.json": {"status": "REAL_FROZEN_VLM_PERCEPTION_COMPLETE"},
        "policy_pilot_gate.json": {
            "status": "INVALIDATED_UPSTREAM_PREDICTOR_CHECKPOINT_SELECTION"
        },
    }
    for name, payload in payloads.items():
        (output / name).write_text(json.dumps(payload))

    def forbidden(**kwargs):
        raise AssertionError("upstream blocker must stop before data loading")

    for module in (training, actor_training, gdfs_training):
        monkeypatch.setattr(module, "load_feature_bank", forbidden)
    result = training.run_oof_reward_predictors(project_root=tmp_path, device="cpu")
    assert result["actual_optimizer_updates"] == 0
    with pytest.raises(ValueError, match="not ready"):
        actor_training.precheck_policy_training(project_root=tmp_path, device="cpu")
    result = actor_training.run_policy_pilots(project_root=tmp_path, device="cpu")
    assert result["status"] == "POLICY_PILOT_BLOCKED_INPUT"
    with pytest.raises(ValueError, match="completed W3"):
        gdfs_training._load_inputs(tmp_path, device="cpu")
    assert not (output / "models").exists()
    assert not (output.parent / "compute_ledger.jsonl").exists()


def test_archive_early_stopping_keeps_all_four_nonimproving_points(tmp_path):
    selection = _selection()
    archive = selection.CheckpointArchive(
        tmp_path / "early",
        model_name="MEAN_SC",
        validation={"identity": "fixed"},
        max_updates=2000,
        validation_interval=250,
        validation_patience=4,
        run_metadata={"seed": 1},
    )
    for update in [250, 500, 750, 1000, 1250]:
        archive.record(update, {"x": torch.tensor(update)}, {"valid_area_mpa": 10.0})
    result = archive.finish(updates_completed=1250)
    assert result["selected_update"] == 250
    assert (
        len(
            selection.inspect_archive(
                tmp_path / "early", validation={"identity": "fixed"}
            )["candidates"]
        )
        == 5
    )


def test_ready_label_without_candidate_evidence_cannot_load_reward_models(
    tmp_path, monkeypatch
):
    import json

    from cmc_bbdm.cai_agent_v3 import actor_training

    output = tmp_path / "results/cai_agent_v3/new_protocol"
    output.mkdir(parents=True)
    metrics = {
        "full_mae_mpa": 1.0,
        "train_median_full_mae_mpa": 10.0,
        "full_mse_mpa2": 1.0,
        "train_mean_full_mse_mpa2": 100.0,
        "zero_mae_mpa": 10.0,
        "center_endpoint_mae_mpa": 2.0,
        "geometry_endpoint_mae_mpa": 2.0,
        "valid_area_mpa": 3.0,
    }
    candidates = [
        {"model": name, "parameter_count": 1, "metrics": metrics}
        for name in ("MEAN_SC", "SPATIAL_SC", "SPATIAL_C")
    ]
    (output / "predictor_gate.json").write_text(
        json.dumps(
            {
                "status": "PREDICTOR_READY",
                "selected_p_all": "MEAN_SC",
                "candidate_manifests": candidates,
            }
        )
    )
    (output / "oof_readiness.json").write_text(
        json.dumps(
            {
                "status": "REWARD_MODELS_READY",
                "fold_manifests": [{"model": "MEAN_SC", "fold": i} for i in range(3)],
            }
        )
    )
    monkeypatch.setattr(actor_training, "load_feature_bank", lambda **_: _bank())

    def forbidden(*args, **kwargs):
        raise AssertionError("must block before loading dependent models")

    monkeypatch.setattr(actor_training, "load_predictor_checkpoint", forbidden)
    with pytest.raises(ValueError, match="missing checkpoints"):
        actor_training._load_oof_predictors(tmp_path, device="cpu")


def test_nonexact_cost_cannot_be_recorded_as_exact_before_training(
    tmp_path, monkeypatch
):
    bank = _bank()
    exact = training._cell_costs(bank)
    library = training.build_validation_library(bank, exact)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "must reject incorrect input before constructing optimizer"
        )

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    with pytest.raises(ValueError, match="float64 native-pixel"):
        training._train_candidate(
            "MEAN_SC",
            bank,
            library,
            exact.astype(np.float32),
            training._constant_metrics(bank),
            device="cpu",
            archive_dir=tmp_path / "bad",
        )
    assert not (tmp_path / "bad").exists()


def test_interrupted_checkpoint_run_keeps_update_reservation_without_double_count(
    tmp_path,
):
    from cmc_bbdm.cai_agent_v3.actor_training import _optimizer_update_upper_bound

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"actual_optimizer_updates":21514}\n'
        '{"actual_optimizer_updates":null,"actual_optimizer_updates_upper_bound":750}\n'
        '{"run_id":"new-w2","status":"STARTED","actual_optimizer_updates":0,'
        '"optimizer_update_reservation":2000}\n'
    )
    assert _optimizer_update_upper_bound(ledger) == 24264
    with ledger.open("a") as handle:
        handle.write(
            '{"run_id":"new-w2","status":"COMPLETED","actual_optimizer_updates":1500}\n'
        )
    assert _optimizer_update_upper_bound(ledger) == 23764
