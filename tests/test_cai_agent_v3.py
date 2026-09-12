from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cmc_bbdm.cai_agent_v3.actor_training import (
    _legal_action_mask,
    _training_rollout_loss,
    run_policy_expansion,
)
from cmc_bbdm.cai_agent_v3.cohort import (
    assign_capture_group_splits,
    build_capture_groups,
)
from cmc_bbdm.cai_agent_v3.diagnostics import (
    _checkpoint_hashes,
    _diagnostic_feature_path,
)
from cmc_bbdm.cai_agent_v3.feature_bank import V3FeatureBank
from cmc_bbdm.cai_agent_v3.final_evaluation import (
    _bootstrap_effects,
    run_internal_test_evaluation,
)
from cmc_bbdm.cai_agent_v3.gates import execute_if_status, predictor_readiness_gate
from cmc_bbdm.cai_agent_v3.gdfs_training import (
    _gdfs_training_loss,
    concrete_group_selection,
)
from cmc_bbdm.cai_agent_v3.metrics import (
    left_error_area_mpa,
    policy_cost_to_go,
    prediction_at_budget,
    torch_policy_cost_to_go,
    trajectory_objective_mpa,
)
from cmc_bbdm.cai_agent_v3.models import (
    MeanFeedbackActor,
    MeanSCPredictor,
    SpatialCAIActor,
    SpatialPredictor,
    TrueStaticActor,
)
from cmc_bbdm.cai_agent_v3.policy import vlm_first_action_mask
from cmc_bbdm.cai_agent_v3.predictor_training import (
    _cell_costs,
    _checkpoint_selection_is_verified,
    _invalidate_downstream_after_checkpoint_selection,
    build_validation_library,
)
from cmc_bbdm.cai_agent_v3.vlm_perception import (
    _prior_failed_specimens,
    run_vlm_perception,
)


@pytest.mark.parametrize(
    (
        "costs",
        "predictions",
        "target",
        "budget",
        "expected_area",
        "expected_j",
        "expected_returns",
    ),
    [
        ([0.0, 0.125, 0.25], [190.0, 194.0, 198.0], 200.0, 0.25, 8.0, 8.5, [8.5, 3.5]),
        ([0.0, 0.1, 0.2], [190.0, 196.0, 198.0], 200.0, 0.25, 6.0, 6.5, [6.5, 2.5]),
        ([0.0], [190.0], 200.0, 0.01, 10.0, 12.5, []),
    ],
)
def test_v3_metric_matches_independent_golden_cases(
    costs, predictions, target, budget, expected_area, expected_j, expected_returns
):
    assert left_error_area_mpa(costs, predictions, target, end=budget) == pytest.approx(
        expected_area
    )
    assert trajectory_objective_mpa(
        costs, predictions, target, budget=budget
    ) == pytest.approx(expected_j)
    assert policy_cost_to_go(
        costs, predictions, target, budget=budget
    ) == pytest.approx(expected_returns)


def test_prediction_at_budget_uses_last_observed_state_without_interpolation():
    assert prediction_at_budget([0.0, 0.1, 0.2], [190.0, 196.0, 198.0], 0.15) == 196.0


def test_torch_return_adapter_matches_pure_math_and_preserves_gradient_path():
    predictions = torch.tensor([190.0, 194.0, 198.0], requires_grad=True)
    returns = torch_policy_cost_to_go(
        torch.tensor([0.0, 0.125, 0.25]), predictions, torch.tensor(200.0), budget=0.25
    )
    assert returns.detach().cpu().numpy() == pytest.approx([8.5, 3.5])
    returns.sum().backward()
    assert torch.isfinite(predictions.grad).all()


def test_capture_groups_union_all_source_identity_relations():
    rows = [
        {
            "dataset_id": "d",
            "specimen_id": "a",
            "cscan_source_sha256": "c1",
            "surface_sha256": "s1",
            "registered_cscan_crop_sha256": "r1",
        },
        {
            "dataset_id": "d",
            "specimen_id": "b",
            "cscan_source_sha256": "c1",
            "surface_sha256": "s2",
            "registered_cscan_crop_sha256": "r2",
        },
        {
            "dataset_id": "d",
            "specimen_id": "c",
            "cscan_source_sha256": "c3",
            "surface_sha256": "s2",
            "registered_cscan_crop_sha256": "r3",
        },
        {
            "dataset_id": "d",
            "specimen_id": "z",
            "cscan_source_sha256": "cz",
            "surface_sha256": "sz",
            "registered_cscan_crop_sha256": "rz",
        },
    ]
    grouped = build_capture_groups(rows)
    assert grouped["d:a"] == grouped["d:b"] == grouped["d:c"]
    assert grouped["d:z"] != grouped["d:a"]


def test_capture_group_split_is_deterministic_and_never_splits_groups():
    rows = []
    for group in range(10):
        for specimen in range(1 + group % 2):
            rows.append(
                {
                    "dataset_id": "d",
                    "specimen_key": f"d:{group}-{specimen}",
                    "capture_group_id": f"g{group}",
                }
            )
    first = assign_capture_group_splits(rows)
    second = assign_capture_group_splits(list(reversed(rows)))
    assert first == second
    assert {split for split in first.values()} == {"TRAIN", "VALID", "TEST"}
    for group in range(10):
        values = {
            first[row["specimen_key"]]
            for row in rows
            if row["capture_group_id"] == f"g{group}"
        }
        assert len(values) == 1


def test_metric_rejects_invalid_cost_trajectory():
    with pytest.raises(ValueError, match="strictly increasing"):
        left_error_area_mpa([0.0, 0.1, 0.1], [1.0, 2.0, 3.0], 2.0, end=0.25)
    with pytest.raises(ValueError, match="endpoint"):
        trajectory_objective_mpa([0.0, 0.3], [1.0, 2.0], 2.0, budget=0.25)


def test_exact_cost_checkpoint_invalidation_propagates_to_downstream(tmp_path):
    (tmp_path / "policy_pilot_gate.json").write_text(
        json.dumps({"status": "POLICY_PILOT_SUPPORTED", "passed": True}),
        encoding="utf-8",
    )
    (tmp_path / "gdfs_pilot.json").write_text(
        json.dumps({"status": "GDFS_ADAPTER_COMPLETE"}), encoding="utf-8"
    )
    (tmp_path / "policy_expansion.json").write_text(
        json.dumps({"status": "RESOURCE_LIMITED"}), encoding="utf-8"
    )

    _invalidate_downstream_after_checkpoint_selection(tmp_path)

    policy = json.loads((tmp_path / "policy_pilot_gate.json").read_text())
    gdfs = json.loads((tmp_path / "gdfs_pilot.json").read_text())
    expansion = json.loads((tmp_path / "policy_expansion.json").read_text())
    assert policy["status"] == "INVALIDATED_UPSTREAM_PREDICTOR_CHECKPOINT_SELECTION"
    assert policy["passed"] is False
    assert gdfs["scientific_use"] == "DIAGNOSTIC_ONLY_INVALIDATED"
    assert expansion["status"] == (
        "NOT_EXECUTED_UPSTREAM_NOT_READY_AND_RESOURCE_LIMITED"
    )


@pytest.mark.parametrize(
    ("current", "historical", "history", "expected"),
    [
        (3, 3, "UPDATE_SELECTED_WITH_FLOAT64_NATIVE_COSTS", False),
        (0, 3, "UPDATE_SELECTED_WITH_FLOAT32_COSTS", False),
        (0, 3, "UPDATE_SELECTED_WITH_FLOAT64_NATIVE_COSTS", True),
        (0, 0, "UPDATE_SELECTED_WITH_FLOAT32_COSTS", True),
    ],
)
def test_exact_cost_checkpoint_selection_requires_unchanged_or_exact_library(
    current, historical, history, expected
):
    assert (
        _checkpoint_selection_is_verified(
            current_changed_rows=current,
            historical_changed_rows=historical,
            checkpoint_selection_history=history,
        )
        is expected
    )


@pytest.mark.parametrize(
    "model",
    [
        MeanSCPredictor(),
        SpatialPredictor(use_surface=True),
        SpatialPredictor(use_surface=False),
    ],
)
def test_predictors_cannot_read_hidden_cscan_tokens(model):
    model.eval()
    surface = torch.randn(2, 64, 512)
    cscan = torch.randn(2, 64, 512)
    changed = cscan.clone()
    changed[:, 10:] += 100.0
    measured = torch.zeros(2, 64, dtype=torch.bool)
    measured[:, :10] = True
    cost = torch.tensor([0.1, 0.1])
    with torch.inference_mode():
        expected = model(surface, cscan, measured, cost=cost)
        actual = model(surface, changed, measured, cost=cost)
    assert actual == pytest.approx(expected)


def test_mean_predictor_soft_group_mask_backpropagates_only_to_mask_when_frozen():
    model = MeanSCPredictor().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    surface = torch.randn(2, 64, 512)
    cscan = torch.randn(2, 64, 512)
    measured_weight = torch.full((2, 64), 0.25, requires_grad=True)
    prediction = model.forward_soft(
        surface,
        cscan,
        measured_weight,
        cost=torch.full((2,), 0.25),
    )
    prediction.sum().backward()
    assert torch.count_nonzero(measured_weight.grad).item() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_concrete_selection_keeps_one_group_and_masks_illegal_groups():
    logits = torch.zeros(2, 64)
    logits[:, 8:] = -torch.inf
    torch.manual_seed(1)
    soft = concrete_group_selection(logits, temperature=0.5, deterministic=False)
    assert soft.shape == (2, 64)
    assert soft.sum(dim=1) == pytest.approx(torch.ones(2))
    assert torch.count_nonzero(soft[:, 8:]).item() == 0
    hard_limit = concrete_group_selection(logits, temperature=0.1, deterministic=True)
    assert hard_limit.sum(dim=1) == pytest.approx(torch.ones(2))
    assert torch.count_nonzero(hard_limit[:, 8:]).item() == 0


def test_gdfs_soft_predictor_path_reaches_spatial_selector():
    class SoftCountPredictor(torch.nn.Module):
        def forward(self, surface, cscan, measured, *, cost):
            del surface, cscan, cost
            weights = torch.arange(64, device=measured.device, dtype=torch.float32)
            return 200.0 + torch.sum(measured * weights, dim=1) / 64.0

        def forward_soft(self, surface, cscan, measured_weight, *, cost):
            del surface, cscan, cost
            weights = torch.arange(
                64, device=measured_weight.device, dtype=torch.float32
            )
            return 200.0 + torch.sum(measured_weight * weights, dim=1) / 64.0

    bank = V3FeatureBank(
        specimen_keys=("d:a", "d:b"),
        dataset_ids=("d", "d"),
        specimen_ids=("a", "b"),
        capture_group_ids=("ga", "gb"),
        splits=("TRAIN", "TRAIN"),
        targets_mpa=np.asarray([200.0, 201.0], dtype=np.float32),
        surface_tokens=np.zeros((2, 64, 512), dtype=np.float32),
        cscan_tokens=np.zeros((2, 64, 512), dtype=np.float32),
        full_surface_tokens=np.zeros((2, 512), dtype=np.float32),
        full_cscan_tokens=np.zeros((2, 512), dtype=np.float32),
        native_shapes=np.asarray([[64, 64], [64, 64]], dtype=np.int64),
    )
    features = SimpleNamespace(
        indicator=np.zeros((2, 64), dtype=np.float32),
        confidence=np.zeros((2, 64), dtype=np.float32),
        available=np.zeros(2, dtype=bool),
        no_reliable=np.ones(2, dtype=bool),
    )
    selector = SpatialCAIActor(use_vlm=True, use_feedback=True)
    loss, _ = _gdfs_training_loss(
        selector,
        bank,
        features,
        {0: SoftCountPredictor()},
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([[0.03] * 64, [0.03] * 64], dtype=np.float64),
        np.asarray([0, 1], dtype=np.int64),
        temperature=0.5,
        target_scale=1.0,
        device="cpu",
    )
    loss.backward()
    assert torch.count_nonzero(selector.action_scorer[-1].weight.grad).item() > 0


def test_spatial_predictor_executes_registered_transformer():
    model = SpatialPredictor(use_surface=True)
    called = []
    hook = model.contextualizer.register_forward_hook(lambda *_: called.append(True))
    model(
        torch.zeros(1, 64, 512),
        torch.zeros(1, 64, 512),
        torch.zeros(1, 64, dtype=torch.bool),
        cost=torch.zeros(1),
    )
    hook.remove()
    assert called == [True]
    assert model.token_width == 128
    assert model.encoder_layers == 2
    assert model.attention_heads == 4


def _actor_inputs():
    return {
        "surface": torch.randn(1, 64, 512),
        "cscan": torch.randn(1, 64, 512),
        "measured": torch.tensor([[True] + [False] * 63]),
        "action_history": torch.tensor([[1.0 / 64.0] + [0.0] * 63]),
        "vlm_indicator": torch.zeros(1, 64),
        "vlm_confidence": torch.zeros(1, 64),
        "vlm_available": torch.ones(1, dtype=torch.bool),
        "vlm_no_reliable": torch.zeros(1, dtype=torch.bool),
        "current_prediction_mpa": torch.tensor([200.0]),
        "cost": torch.tensor([0.02]),
        "remaining_cost": torch.tensor([0.23]),
    }


def test_spatial_feedback_reaches_unmeasured_candidate_score():
    model = SpatialCAIActor(use_vlm=True, use_feedback=True)
    inputs = _actor_inputs()
    inputs["cscan"].requires_grad_(True)
    scores, _ = model(**inputs)
    scores[0, 1].backward()
    assert torch.count_nonzero(inputs["cscan"].grad[0, 0]).item() > 0


def test_spatial_actor_executes_registered_transformer():
    model = SpatialCAIActor(use_vlm=True, use_feedback=True)
    called = []
    hook = model.contextualizer.register_forward_hook(lambda *_: called.append(True))
    model(**_actor_inputs())
    hook.remove()
    assert called == [True]
    assert model.token_width == 128
    assert model.encoder_layers == 2
    assert model.attention_heads == 4


def test_spatial_feedback_actor_cannot_read_hidden_cscan_tokens():
    model = SpatialCAIActor(use_vlm=True, use_feedback=True).eval()
    inputs = _actor_inputs()
    changed = dict(inputs)
    changed["cscan"] = inputs["cscan"].clone()
    changed["cscan"][:, 1:] += 100.0
    with torch.inference_mode():
        expected = model(**inputs)[0]
        actual = model(**changed)[0]
    assert actual == pytest.approx(expected)


def test_mean_feedback_diagnostic_uses_same_visible_input_signature():
    scores, value = MeanFeedbackActor()(**_actor_inputs())
    assert scores.shape == (1, 64)
    assert value.shape == (1,)


def test_open_loop_actor_cannot_read_cscan_or_current_prediction():
    model = SpatialCAIActor(use_vlm=True, use_feedback=False).eval()
    inputs = _actor_inputs()
    with torch.inference_mode():
        expected = model(**inputs)[0]
        inputs["cscan"] = inputs["cscan"] + 100.0
        inputs["current_prediction_mpa"] = torch.tensor([-1000.0])
        actual = model(**inputs)[0]
    assert actual == pytest.approx(expected)


def test_no_vlm_actor_is_invariant_to_vlm_values():
    model = SpatialCAIActor(use_vlm=False, use_feedback=True).eval()
    inputs = _actor_inputs()
    with torch.inference_mode():
        expected = model(**inputs)[0]
        inputs["vlm_indicator"] = torch.ones(1, 64)
        inputs["vlm_confidence"] = torch.ones(1, 64)
        inputs["vlm_available"] = torch.zeros(1, dtype=torch.bool)
        inputs["vlm_no_reliable"] = torch.ones(1, dtype=torch.bool)
        actual = model(**inputs)[0]
    assert actual == pytest.approx(expected)


def test_true_static_actor_has_only_shared_position_logits():
    model = TrueStaticActor()
    first = model(batch_size=2)
    second = model(batch_size=2)
    assert first.shape == (2, 64)
    assert torch.equal(second, first)
    assert list(dict(model.named_parameters())) == ["position_logits"]


def test_vlm_c0_uses_only_highest_reliable_level_then_unlocks():
    legal = torch.ones(1, 64, dtype=torch.bool)
    indicator = torch.zeros(1, 64)
    confidence = torch.zeros(1, 64)
    indicator[0, [1, 2, 3]] = 1.0
    confidence[0, [1, 2, 3]] = torch.tensor([2.0 / 3.0, 1.0, 1.0])
    first, reasons = vlm_first_action_mask(
        legal=legal,
        use_vlm=True,
        action_count=torch.tensor([0]),
        indicator=indicator,
        confidence=confidence,
        available=torch.tensor([True]),
        no_reliable=torch.tensor([False]),
    )
    assert torch.nonzero(first[0], as_tuple=False).flatten().tolist() == [2, 3]
    assert reasons == ("HIGHEST_RELIABLE_CONFIDENCE_C0",)
    later, reasons = vlm_first_action_mask(
        legal=legal,
        use_vlm=True,
        action_count=torch.tensor([1]),
        indicator=indicator,
        confidence=confidence,
        available=torch.tensor([True]),
        no_reliable=torch.tensor([False]),
    )
    assert torch.equal(later, legal)
    assert reasons == ("C0_RELEASED_AFTER_FIRST_ACTION",)


def test_failed_predictor_gate_does_not_execute_actor_stage():
    metrics = {
        "full_mae_mpa": 11.0,
        "train_median_full_mae_mpa": 10.0,
        "full_mse_mpa2": 121.0,
        "train_mean_full_mse_mpa2": 100.0,
        "zero_mae_mpa": 10.0,
        "center_endpoint_mae_mpa": 10.0,
        "geometry_endpoint_mae_mpa": 10.0,
        "valid_area_mpa": 10.0,
    }
    result = predictor_readiness_gate(metrics)
    calls = []
    output = execute_if_status(
        required_status="PREDICTOR_READY",
        observed_status=result.status,
        action=lambda: calls.append("actor"),
    )
    assert result.status == "PREDICTOR_NOT_READY"
    assert output is None
    assert calls == []


def test_native_cost_adapter_accepts_loaded_numpy_integer_shapes():
    bank = V3FeatureBank(
        specimen_keys=("d:s",),
        dataset_ids=("d",),
        specimen_ids=("s",),
        capture_group_ids=("g",),
        splits=("VALID",),
        targets_mpa=np.asarray([200.0], dtype=np.float32),
        surface_tokens=np.zeros((1, 64, 512), dtype=np.float32),
        cscan_tokens=np.zeros((1, 64, 512), dtype=np.float32),
        full_surface_tokens=np.zeros((1, 512), dtype=np.float32),
        full_cscan_tokens=np.zeros((1, 512), dtype=np.float32),
        native_shapes=np.asarray([[65, 67]], dtype=np.int64),
    )
    costs = _cell_costs(bank)
    assert costs.shape == (1, 64)
    assert float(costs.sum()) == pytest.approx(1.0)


def test_validation_routes_retain_float64_native_costs_at_exact_quarter_budget():
    bank = V3FeatureBank(
        specimen_keys=("d:s",),
        dataset_ids=("d",),
        specimen_ids=("s",),
        capture_group_ids=("g",),
        splits=("VALID",),
        targets_mpa=np.asarray([200.0], dtype=np.float32),
        surface_tokens=np.zeros((1, 64, 512), dtype=np.float32),
        cscan_tokens=np.zeros((1, 64, 512), dtype=np.float32),
        full_surface_tokens=np.zeros((1, 512), dtype=np.float32),
        full_cscan_tokens=np.zeros((1, 512), dtype=np.float32),
        native_shapes=np.asarray([[338, 352]], dtype=np.int64),
    )
    library = build_validation_library(bank, _cell_costs(bank))
    geometry = np.asarray(library.route_names) == "GEOMETRY_SPREAD"
    assert library.costs.dtype == np.float64
    assert library.costs[geometry][-1] == pytest.approx(0.25, abs=1e-15)


def test_policy_legality_uses_exact_native_pixel_cost_not_float32_round_trip():
    total_pixels = 338 * 352
    current_pixels = 27_896
    candidate_pixels = 1_848
    exact_cost = current_pixels / total_pixels
    candidate_cost = candidate_pixels / total_pixels
    assert exact_cost + candidate_cost == pytest.approx(0.25, abs=1e-15)
    assert float(np.float32(exact_cost)) + candidate_cost > 0.25 + 1e-12
    legal = _legal_action_mask(
        np.zeros((1, 1), dtype=bool),
        np.asarray([exact_cost], dtype=np.float64),
        np.asarray([[candidate_cost]], dtype=np.float64),
    )
    assert bool(legal[0, 0])


def test_policy_batch_terminates_each_episode_at_its_own_affordable_length():
    class CountPredictor(torch.nn.Module):
        def forward(self, surface, cscan, measured, *, cost):
            del surface, cscan, cost
            return 200.0 + measured.to(torch.float32).sum(dim=1)

    bank = V3FeatureBank(
        specimen_keys=("d:a", "d:b"),
        dataset_ids=("d", "d"),
        specimen_ids=("a", "b"),
        capture_group_ids=("ga", "gb"),
        splits=("TRAIN", "TRAIN"),
        targets_mpa=np.asarray([200.0, 200.0], dtype=np.float32),
        surface_tokens=np.zeros((2, 64, 512), dtype=np.float32),
        cscan_tokens=np.zeros((2, 64, 512), dtype=np.float32),
        full_surface_tokens=np.zeros((2, 512), dtype=np.float32),
        full_cscan_tokens=np.zeros((2, 512), dtype=np.float32),
        native_shapes=np.asarray([[64, 64], [64, 64]], dtype=np.int64),
    )
    features = SimpleNamespace(
        indicator=np.zeros((2, 64), dtype=np.float32),
        confidence=np.zeros((2, 64), dtype=np.float32),
        available=np.zeros(2, dtype=bool),
        no_reliable=np.ones(2, dtype=bool),
    )
    actor = TrueStaticActor()
    loss, terms = _training_rollout_loss(
        actor,
        "LEARNED_STATIC_TRUE",
        bank,
        features,
        {0: CountPredictor()},
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([[0.02] * 64, [0.03] * 64], dtype=np.float64),
        np.asarray([0, 1], dtype=np.int64),
        device="cpu",
        entropy_weight=0.0,
        target_scale=1.0,
    )
    loss.backward()
    assert terms["min_action_count"] == 8
    assert terms["max_action_count"] == 12
    assert torch.isfinite(actor.position_logits.grad).all()


def test_vlm_failure_history_prevents_resampling_after_one_repair(tmp_path):
    ledger = tmp_path / "compute_ledger.jsonl"
    ledger.write_text(
        '{"job":"vlm_perception_fit_d:s","stage":"W3_INPUT",'
        '"actual_vlm_calls":2,"status":"FAILED"}\n',
        encoding="utf-8",
    )
    assert _prior_failed_specimens(ledger, scope="fit") == {"d:s": 2}
    assert _prior_failed_specimens(ledger, scope="test") == {}


def test_failed_policy_expansion_does_not_open_test_inputs(tmp_path):
    output = tmp_path / "results/cai_agent_v3/new_protocol"
    output.mkdir(parents=True)
    (output / "policy_expansion.json").write_text(
        '{"status":"NOT_EXECUTED_POLICY_PILOT_NOT_SUPPORTED"}\n',
        encoding="utf-8",
    )
    result = run_internal_test_evaluation(project_root=tmp_path, device="cpu")
    assert result["status"] == "NOT_EXECUTED_POLICY_EXPANSION_NOT_LOCKED"
    assert result["test_labels_accessed"] is False


def test_resource_limit_blocks_policy_expansion_before_loading_training_data(tmp_path):
    output = tmp_path / "results/cai_agent_v3/new_protocol"
    artifacts = tmp_path / "artifacts/cai_agent_v3"
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (output / "policy_pilot_gate.json").write_text(
        '{"status":"POLICY_PILOT_SUPPORTED"}\n', encoding="utf-8"
    )
    (artifacts / "requirements_review_w3.json").write_text(
        '{"status":"CORE_REVIEW_PASS"}\n', encoding="utf-8"
    )
    (tmp_path / "results/cai_agent_v3/compute_ledger.jsonl").write_text(
        '{"status":"COMPLETED","actual_optimizer_updates":20000}\n',
        encoding="utf-8",
    )
    result = run_policy_expansion(project_root=tmp_path, device="cpu")
    assert result["status"] == "RESOURCE_LIMITED"
    assert result["actual_optimizer_updates"] == 0
    assert result["test_accessed"] is False


def test_resource_limited_expansion_does_not_open_test_vlm(tmp_path):
    output = tmp_path / "results/cai_agent_v3/new_protocol"
    output.mkdir(parents=True)
    (output / "predictor_gate.json").write_text(
        '{"status":"PREDICTOR_READY"}\n', encoding="utf-8"
    )
    (output / "policy_pilot_gate.json").write_text(
        '{"status":"POLICY_PILOT_SUPPORTED"}\n', encoding="utf-8"
    )
    (output / "policy_expansion.json").write_text(
        '{"status":"RESOURCE_LIMITED"}\n', encoding="utf-8"
    )
    result = run_vlm_perception(
        project_root=tmp_path, source_root=tmp_path, scope="test"
    )
    assert result["status"] == "NOT_EXECUTED_POLICY_EXPANSION_NOT_LOCKED"
    assert result["actual_calls"] == 0


def test_diagnostic_checkpoint_hashes_are_bound_to_method_and_seed(tmp_path):
    output = tmp_path / "new_protocol"
    output.mkdir()
    pilot_gate = {
        "actor_manifests": [
            {
                "method": "VLM_SPATIAL_FEEDBACK",
                "seed_panel": 1,
                "checkpoint_sha256": "seed1",
            }
        ]
    }
    (output / "policy_expansion.json").write_text(
        json.dumps(
            {
                "status": "POLICY_SEEDS_1_TO_3_LOCKED",
                "new_actor_manifests": [
                    {
                        "method": "VLM_SPATIAL_FEEDBACK",
                        "seed_panel": 2,
                        "checkpoint_sha256": "seed2",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _checkpoint_hashes(output, pilot_gate) == {
        ("VLM_SPATIAL_FEEDBACK", "1"): "seed1",
        ("VLM_SPATIAL_FEEDBACK", "2"): "seed2",
    }


def test_diagnostic_feature_scope_uses_fit_file_for_validation(tmp_path):
    assert _diagnostic_feature_path(tmp_path, "VALID") == (
        tmp_path / "vlm_actor_features_fit.csv"
    )
    assert _diagnostic_feature_path(tmp_path, "TEST") == (
        tmp_path / "vlm_actor_features_test.csv"
    )


def test_group_bootstrap_point_estimate_weights_physical_specimens():
    effects = {
        "d1:a": ("d1", "g1", 2.0),
        "d1:b": ("d1", "g1", 4.0),
        "d1:c": ("d1", "g2", 10.0),
        "d2:a": ("d2", "g3", 1.0),
        "d2:b": ("d2", "g3", 3.0),
    }
    rows = []
    for key, (domain, group, effect) in effects.items():
        common = {
            "specimen_key": key,
            "dataset_id": domain,
            "capture_group_id": group,
        }
        rows.extend(
            (
                {
                    **common,
                    "method": "VLM_SPATIAL_FEEDBACK",
                    "left_error_area_mpa": 10.0,
                    "early_left_error_area_mpa": 5.0,
                },
                {
                    **common,
                    "method": "LEARNED_STATIC_TRUE",
                    "left_error_area_mpa": 10.0 + effect,
                    "early_left_error_area_mpa": 5.0 + effect,
                },
                {
                    **common,
                    "method": "VLM_SPATIAL_OPEN_LOOP",
                    "left_error_area_mpa": 10.0 + effect,
                    "early_left_error_area_mpa": 5.0 + effect,
                },
                {
                    **common,
                    "method": "NO_VLM_SPATIAL_FEEDBACK",
                    "left_error_area_mpa": 10.0 + effect,
                    "early_left_error_area_mpa": 5.0 + effect,
                },
            )
        )
    output = _bootstrap_effects(rows, best_nonadaptive="LEARNED_STATIC_TRUE")
    expected = ((2.0 + 4.0 + 10.0) / 3.0 + (1.0 + 3.0) / 2.0) / 2.0
    assert [row["effect_mpa"] for row in output] == pytest.approx([expected] * 3)
