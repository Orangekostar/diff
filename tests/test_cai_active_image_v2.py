from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from cmc_bbdm.cai_active_image.contracts import Method
from cmc_bbdm.cai_active_image.data import load_mpa_targets
from cmc_bbdm.cai_active_image.environment import NativeCellGrid, reveal_cells
from cmc_bbdm.cai_active_image.episodes import run_episode
from cmc_bbdm.cai_active_image.features import FeatureBank
from cmc_bbdm.cai_active_image.models import CAIActor, CommonCAIPredictor
from cmc_bbdm.cai_active_image.perception import (
    build_vlm_actor_features,
    load_frozen_vlm_cache,
    proposal_decision_for_method,
    proposal_mask_for_method,
)
from cmc_bbdm.cai_active_image.protocol import load_protocol
from cmc_bbdm.cai_active_image.statistics import (
    domain_clustered_paired_bootstrap,
    normalized_error_area_mpa,
)
from cmc_bbdm.cai_active_image.training import (
    fixed_action_order,
    state_dict_sha256,
    train_actor,
    train_predictor,
)
from cmc_bbdm.cai_active_image.validation import (
    validate_common_predictor_identity,
    validate_trajectory_rows,
)
from cmc_bbdm.learned_cscan.perception import SurfacePercept, SurfaceRegion
from scripts.run_cai_active_image_v2 import build_parser


def _percept(*regions: SurfaceRegion, no_reliable: bool = False) -> SurfacePercept:
    return SurfacePercept(regions=regions, no_reliable_cue=no_reliable)


def _trajectory_row(
    index: int,
    before: float,
    after: float,
    cell: int,
    history: tuple[int, ...] = (),
) -> dict[str, object]:
    legal = ["1"] * 64
    proposal = ["1"] * 64
    return {
        "vlm_cache_key": "a" * 64,
        "vlm_available": True,
        "initial_candidates": "3;4",
        "initial_action": 3,
        "proposal_restricted": False,
        "initial_proposal_reason": "METHOD_DOES_NOT_USE_VLM_C0",
        "environment_legal_mask": "".join(legal),
        "initial_proposal_mask": "".join(proposal),
        "actor_call_index": index,
        "actor_latency_seconds": 0.001,
        "prediction_before_latency_seconds": 0.001,
        "predictor_latency_seconds": 0.001,
        "observed_state_id": "initial" if index == 0 else f"after:{index}",
        "action_history_cells": ";".join(str(item) for item in history),
        "before_cost": before,
        "remaining_cost": 0.25 - before,
        "after_cost": after,
        "next_cell": cell,
        "prediction_mpa": 200.0,
    }


def test_registered_vlm_cells_are_not_rotated_twice() -> None:
    percept = _percept(
        SurfaceRegion((0, 7), "mark", "scratch", "high"),
        SurfaceRegion((56,), "edge", "glare", "medium"),
    )

    features = build_vlm_actor_features(
        percept,
        cache_available=True,
        clockwise_quarter_turns=1,
    )

    assert np.flatnonzero(features.region_indicator).tolist() == [0, 7, 56]
    assert features.confidence[0] == pytest.approx(1.0)
    assert features.confidence[56] == pytest.approx(2.0 / 3.0)
    assert features.available is True
    assert features.no_reliable_cue is False


@pytest.mark.parametrize(
    ("percept", "available"),
    [
        (_percept(SurfaceRegion((9,), "faint", "glare", "low")), True),
        (_percept(no_reliable=True), True),
        (_percept(no_reliable=True), False),
    ],
)
def test_c0_falls_back_to_all_legal_without_reliable_candidates(
    percept: SurfacePercept, available: bool
) -> None:
    features = build_vlm_actor_features(
        percept,
        cache_available=available,
        clockwise_quarter_turns=1,
    )
    legal = np.ones(64, dtype=bool)
    legal[3] = False

    proposal = proposal_mask_for_method(
        Method.VLM_CAI_FEEDBACK_AGENT,
        features=features,
        legal_mask=legal,
        action_count=0,
    )

    assert np.array_equal(proposal, legal)


def test_c0_uses_highest_reliable_confidence_union_then_unlocks() -> None:
    percept = _percept(
        SurfaceRegion((1, 2), "mark", "scratch", "medium"),
        SurfaceRegion((7, 8), "dent", "glare", "high"),
    )
    features = build_vlm_actor_features(
        percept,
        cache_available=True,
        clockwise_quarter_turns=1,
    )
    legal = np.ones(64, dtype=bool)
    legal[2] = False

    first = proposal_mask_for_method(
        Method.VLM_CAI_FEEDBACK_AGENT,
        features=features,
        legal_mask=legal,
        action_count=0,
    )
    later = proposal_mask_for_method(
        Method.VLM_CAI_FEEDBACK_AGENT,
        features=features,
        legal_mask=legal,
        action_count=1,
    )
    no_vlm = proposal_mask_for_method(
        Method.NO_VLM_FEEDBACK,
        features=features,
        legal_mask=legal,
        action_count=0,
    )

    assert np.flatnonzero(first).tolist() == [7, 8]
    assert np.array_equal(later, legal)
    assert np.array_equal(no_vlm, legal)


def test_c0_fallback_reason_is_explicit() -> None:
    legal = np.ones(64, dtype=bool)
    weak = build_vlm_actor_features(
        _percept(SurfaceRegion((5,), "faint", "glare", "low")),
        cache_available=True,
        clockwise_quarter_turns=1,
    )
    unavailable = build_vlm_actor_features(
        _percept(no_reliable=True),
        cache_available=False,
        clockwise_quarter_turns=1,
    )

    weak_decision = proposal_decision_for_method(
        Method.VLM_CAI_FEEDBACK_AGENT,
        features=weak,
        legal_mask=legal,
        action_count=0,
    )
    unavailable_decision = proposal_decision_for_method(
        Method.VLM_CAI_FEEDBACK_AGENT,
        features=unavailable,
        legal_mask=legal,
        action_count=0,
    )

    assert weak_decision.reason == "NO_MEDIUM_OR_HIGH_CONFIDENCE"
    assert unavailable_decision.reason == "VLM_UNAVAILABLE"
    assert np.array_equal(weak_decision.mask, legal)
    assert np.array_equal(unavailable_decision.mask, legal)


def test_native_grid_partitions_every_pixel_and_bills_exact_cost() -> None:
    grid = NativeCellGrid.from_shape((17, 19))
    coverage = np.zeros((17, 19), dtype=np.int64)
    for cell in grid.cells:
        coverage[cell.row_start : cell.row_stop, cell.col_start : cell.col_stop] += 1

    assert np.all(coverage == 1)
    assert sum(cell.pixel_count for cell in grid.cells) == 17 * 19
    assert sum(cell.cost for cell in grid.cells) == pytest.approx(1.0)

    full_scan = np.arange(17 * 19 * 3, dtype=np.uint16).reshape(17, 19, 3)
    visible, pixel_mask = reveal_cells(full_scan, grid, measured_cells={0, 9})
    assert np.array_equal(visible[pixel_mask], full_scan[pixel_mask])
    assert np.all(visible[~pixel_mask] == 0)
    assert int(pixel_mask.sum()) == grid.cells[0].pixel_count + grid.cells[9].pixel_count


def test_unmeasured_cscan_cannot_change_predictor_or_feedback_actor() -> None:
    torch.manual_seed(7)
    predictor = CommonCAIPredictor(token_dimension=8, width=16)
    actor = CAIActor(token_dimension=8, width=16, use_vlm=True, use_feedback=True)
    surface = torch.randn(2, 64, 8)
    cscan_a = torch.randn(2, 64, 8)
    cscan_b = cscan_a.clone()
    cscan_b[:, 5:] = torch.randn_like(cscan_b[:, 5:]) * 100.0
    measured = torch.zeros(2, 64, dtype=torch.bool)
    measured[:, :5] = True
    indicator = torch.zeros(2, 64)
    confidence = torch.zeros(2, 64)
    history = torch.zeros(2, 64)
    indicator[:, 7] = 1.0
    confidence[:, 7] = 1.0

    pred_a = predictor(surface, cscan_a, measured)
    pred_b = predictor(surface, cscan_b, measured)
    score_a, _ = actor(
        surface,
        cscan_a,
        measured,
        history,
        indicator,
        confidence,
        torch.ones(2),
        torch.zeros(2),
        pred_a,
        torch.full((2,), 0.1),
        torch.full((2,), 0.15),
    )
    score_b, _ = actor(
        surface,
        cscan_b,
        measured,
        history,
        indicator,
        confidence,
        torch.ones(2),
        torch.zeros(2),
        pred_b,
        torch.full((2,), 0.1),
        torch.full((2,), 0.15),
    )

    assert torch.equal(pred_a, pred_b)
    assert torch.equal(score_a, score_b)


def test_open_loop_and_no_vlm_ablation_have_no_bypass() -> None:
    torch.manual_seed(9)
    surface = torch.randn(1, 64, 8)
    cscan_a = torch.randn(1, 64, 8)
    cscan_b = torch.randn(1, 64, 8)
    measured = torch.zeros(1, 64, dtype=torch.bool)
    measured[:, :8] = True
    zeros = torch.zeros(1, 64)
    ones = torch.ones(1, 64)
    history = torch.zeros(1, 64)

    open_loop = CAIActor(
        token_dimension=8, width=16, use_vlm=True, use_feedback=False
    )
    open_a, _ = open_loop(
        surface, cscan_a, measured, history, ones, ones, ones[:, 0], zeros[:, 0],
        torch.tensor([100.0]), torch.tensor([0.125]), torch.tensor([0.125]),
    )
    open_b, _ = open_loop(
        surface, cscan_b, measured, history, ones, ones, ones[:, 0], zeros[:, 0],
        torch.tensor([400.0]), torch.tensor([0.125]), torch.tensor([0.125]),
    )
    assert torch.equal(open_a, open_b)

    no_vlm = CAIActor(
        token_dimension=8, width=16, use_vlm=False, use_feedback=True
    )
    no_a, _ = no_vlm(
        surface, cscan_a, measured, history, zeros, zeros, zeros[:, 0], ones[:, 0],
        torch.tensor([100.0]), torch.tensor([0.125]), torch.tensor([0.125]),
    )
    no_b, _ = no_vlm(
        surface, cscan_a, measured, history, ones, ones, ones[:, 0], zeros[:, 0],
        torch.tensor([100.0]), torch.tensor([0.125]), torch.tensor([0.125]),
    )
    assert torch.equal(no_a, no_b)


def test_vlm_features_are_connected_to_main_actor_scores() -> None:
    torch.manual_seed(19)
    actor = CAIActor(token_dimension=8, width=16, use_vlm=True, use_feedback=True)
    surface = torch.randn(1, 64, 8)
    cscan = torch.randn(1, 64, 8)
    measured = torch.zeros(1, 64, dtype=torch.bool)
    zeros = torch.zeros(1, 64)
    history = torch.zeros(1, 64)
    indicator = zeros.clone()
    confidence = zeros.clone()
    indicator[:, 12] = 1.0
    confidence[:, 12] = 1.0

    without, _ = actor(
        surface,
        cscan,
        measured,
        history,
        zeros,
        zeros,
        torch.zeros(1),
        torch.zeros(1),
        torch.tensor([250.0]),
        torch.zeros(1),
        torch.full((1,), 0.25),
    )
    with_prior, _ = actor(
        surface,
        cscan,
        measured,
        history,
        indicator,
        confidence,
        torch.ones(1),
        torch.zeros(1),
        torch.tensor([250.0]),
        torch.zeros(1),
        torch.full((1,), 0.25),
    )

    assert not torch.equal(without, with_prior)


def test_actor_consumes_remaining_cost_and_ordered_action_history() -> None:
    torch.manual_seed(23)
    actor = CAIActor(token_dimension=8, width=16, use_vlm=False, use_feedback=True)
    surface = torch.randn(1, 64, 8)
    cscan = torch.randn(1, 64, 8)
    measured = torch.zeros(1, 64, dtype=torch.bool)
    measured[:, [3, 11]] = True
    zeros = torch.zeros(1, 64)
    first_history = zeros.clone()
    first_history[:, 3] = 1.0 / 64.0
    first_history[:, 11] = 2.0 / 64.0
    second_history = zeros.clone()
    second_history[:, 11] = 1.0 / 64.0
    second_history[:, 3] = 2.0 / 64.0

    baseline, _ = actor(
        surface, cscan, measured, first_history, zeros, zeros,
        torch.zeros(1), torch.zeros(1), torch.tensor([250.0]),
        torch.tensor([0.03125]), torch.tensor([0.21875]),
    )
    changed_history, _ = actor(
        surface, cscan, measured, second_history, zeros, zeros,
        torch.zeros(1), torch.zeros(1), torch.tensor([250.0]),
        torch.tensor([0.03125]), torch.tensor([0.21875]),
    )
    changed_remaining, _ = actor(
        surface, cscan, measured, first_history, zeros, zeros,
        torch.zeros(1), torch.zeros(1), torch.tensor([250.0]),
        torch.tensor([0.03125]), torch.tensor([0.125]),
    )

    assert not torch.equal(baseline, changed_history)
    assert not torch.equal(baseline, changed_remaining)


def test_normalized_error_area_uses_absolute_mpa_and_absolute_zero() -> None:
    value = normalized_error_area_mpa(
        costs=np.asarray([0.0, 0.125, 0.25]),
        predictions_mpa=np.asarray([190.0, 194.0, 198.0]),
        target_mpa=200.0,
        endpoint_budget=0.25,
    )
    assert value == pytest.approx(6.0)


def test_domain_clustered_bootstrap_is_equal_domain_and_reproducible() -> None:
    rows = [
        ("a", "a1", 4.0),
        ("a", "a2", 2.0),
        ("b", "b1", -1.0),
        ("b", "b2", -1.0),
    ]
    first = domain_clustered_paired_bootstrap(rows, replicates=200, seed=42)
    second = domain_clustered_paired_bootstrap(rows, replicates=200, seed=42)

    assert first.estimate == pytest.approx(1.0)
    assert first.confidence_level == pytest.approx(0.9833333333333333)
    assert first == second


def test_target_loader_rejects_non_authoritative_or_ambiguous_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "targets.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["domain_id", "specimen_id", "cai_strength_mpa"],
        )
        writer.writeheader()
        writer.writerow(
            {"domain_id": "d", "specimen_id": "s", "cai_strength_mpa": "123.5"}
        )
    assert load_mpa_targets(path) == {"d:s": 123.5}

    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write("d,s,125.0\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_mpa_targets(path)


def test_predictor_and_step_identity_contracts() -> None:
    validate_common_predictor_identity(
        {
            Method.VLM_CAI_FEEDBACK_AGENT: "a" * 64,
            Method.NO_VLM_FEEDBACK: "a" * 64,
            Method.VLM_OPEN_LOOP: "a" * 64,
            Method.SERPENTINE: "a" * 64,
        }
    )
    with pytest.raises(ValueError, match="common predictor"):
        validate_common_predictor_identity(
            {
                Method.VLM_CAI_FEEDBACK_AGENT: "a" * 64,
                Method.NO_VLM_FEEDBACK: "b" * 64,
            }
        )


def test_trajectory_validator_requires_observability_and_runtime_fields() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        validate_trajectory_rows(
            [
                {
                    "actor_call_index": 0,
                    "observed_state_id": "initial",
                    "before_cost": 0.0,
                    "after_cost": 0.015625,
                    "next_cell": 3,
                }
            ]
        )


def test_registered_protocol_freezes_sources_and_optimizer_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(
        root / "paper_v3/configs/cai_active_image_v2.yaml",
        project_root=root,
    )

    assert protocol.repository_base_sha == "29b3249610c821e8f9c4f58d740588181c443dd5"
    assert protocol.endpoint_budget == 0.25
    assert protocol.early_budget == 0.0625
    assert (
        protocol.initial_proposal_rule
        == "HIGHEST_RELIABLE_CONFIDENCE_C0_THEN_UNLOCK_V1"
    )
    assert protocol.actor_state_protocol == "VISIBLE_ORDERED_HISTORY_BUDGET_V1"
    assert protocol.predictor_updates == (2000, 2000, 2000, 2000)
    assert protocol.predictor_mask_counts == (0, 1, 2, 4, 8, 12, 16, 32, 64)
    assert protocol.predictor_validation_interval == 250
    assert protocol.actor_updates == 1250
    assert protocol.static_updates == 750
    assert protocol.total_optimizer_updates == 21_500
    assert protocol.bootstrap_replicates == 5000


def test_cli_requires_explicit_external_source_root() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["prepare"])


def test_public_v2_outputs_do_not_expose_machine_specific_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    machine_root = Path.home().as_posix()
    paths = (
        root / "scripts/run_cai_active_image_v2.py",
        root / "results/cai_active_image/v2/p0_protocol_audit.json",
        root / "results/cai_active_image/v2/feature_bank_manifest.json",
        root / "artifacts/cai_active_image/v2/CAI_ACTIVE_IMAGE_SOURCE_BINDINGS_V2.md",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert machine_root not in text


def test_artifact_manifest_records_current_result_checksum() -> None:
    root = Path(__file__).resolve().parents[1]
    checksum_path = root / "results/cai_active_image/v2/CHECKSUMS.sha256"
    manifest = json.loads(
        (root / "artifacts/cai_active_image/v2/artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    actual = hashlib.sha256(checksum_path.read_bytes()).hexdigest()
    assert (
        manifest["result_files"]["results/cai_active_image/v2/CHECKSUMS.sha256"]
        == actual
    )


def test_frozen_vlm_cache_binds_all_sixty_specimens_by_cache_key() -> None:
    root = Path(__file__).resolve().parents[1]
    cache = load_frozen_vlm_cache(
        root / "results/learned_cscan_same_perception/surface_percepts.jsonl",
        root / "results/learned_cscan_same_perception/perception_manifest.json",
    )

    assert len(cache) == 60
    assert all(item.cache_hit for item in cache.values())
    assert all(item.actual_call_count == 0 for item in cache.values())
    assert sum(item.original_call_count for item in cache.values()) == 62
    assert all(item.features.available for item in cache.values())


def test_frozen_vlm_cache_rejects_more_than_one_format_repair(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "results/learned_cscan_same_perception/surface_percepts.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    rows[0]["call_count"] = 3
    rows[0]["repaired"] = True
    modified = tmp_path / "surface_percepts.jsonl"
    modified.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="call or repair count"):
        load_frozen_vlm_cache(
            modified,
            root / "results/learned_cscan_same_perception/perception_manifest.json",
        )


def _synthetic_bank() -> FeatureBank:
    rng = np.random.default_rng(12)
    count = 9
    return FeatureBank(
        specimen_keys=tuple(f"d{i % 3}:s{i}" for i in range(count)),
        specimen_ids=tuple(f"s{i}" for i in range(count)),
        dataset_ids=tuple(f"d{i % 3}" for i in range(count)),
        splits=tuple("TRAIN" if i < 6 else "VALID" for i in range(count)),
        targets_mpa=np.linspace(150.0, 350.0, count, dtype=np.float32),
        surface_tokens=rng.normal(size=(count, 64, 8)).astype(np.float32),
        cscan_tokens=rng.normal(size=(count, 64, 8)).astype(np.float32),
        native_shapes=np.tile(np.asarray([[16, 16]], dtype=np.int64), (count, 1)),
        vlm_indicators=np.zeros((count, 64), dtype=np.float32),
        vlm_confidences=np.zeros((count, 64), dtype=np.float32),
        vlm_available=np.ones(count, dtype=bool),
        vlm_no_reliable=np.zeros(count, dtype=bool),
        vlm_cache_keys=tuple("a" * 64 for _ in range(count)),
        surface_sha256=tuple("b" * 64 for _ in range(count)),
        cscan_sha256=tuple("c" * 64 for _ in range(count)),
    )


def test_feature_bank_round_trip_is_pickle_free(tmp_path: Path) -> None:
    bank = _synthetic_bank()
    targets = bank.targets_mpa.copy()
    targets[-1] = np.nan
    splits = (*bank.splits[:-1], "TEST")
    bank = replace(bank, targets_mpa=targets, splits=splits)
    path = tmp_path / "bank.npz"
    bank.save(path)
    loaded = FeatureBank.load(path)

    assert loaded.specimen_keys == bank.specimen_keys
    assert np.array_equal(loaded.targets_mpa, bank.targets_mpa, equal_nan=True)
    assert np.array_equal(loaded.surface_tokens, bank.surface_tokens)
    assert np.array_equal(loaded.cscan_tokens, bank.cscan_tokens)
    assert np.isnan(loaded.targets_mpa[-1])


def test_fixed_action_orders_are_complete_and_seeded() -> None:
    serpentine = fixed_action_order(Method.SERPENTINE, seed=1)
    center = fixed_action_order(Method.CENTER_FIRST, seed=1)
    spread = fixed_action_order(Method.GEOMETRY_SPREAD, seed=1)
    random_a = fixed_action_order(Method.RANDOM, seed=17)
    random_b = fixed_action_order(Method.RANDOM, seed=17)

    assert serpentine[:16] == tuple(range(8)) + tuple(range(15, 7, -1))
    assert center[0] in {27, 28, 35, 36}
    assert spread[0] == 0
    assert all(set(order) == set(range(64)) for order in (serpentine, center, spread, random_a))
    assert random_a == random_b


def test_predictor_training_obeys_exact_update_limit() -> None:
    bank = _synthetic_bank()
    result = train_predictor(
        bank,
        fit_indices=np.arange(6),
        updates=3,
        batch_size=4,
        width=16,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
        seed=3,
        device="cpu",
        validation_indices=np.arange(6, 9),
        validation_interval=1,
    )

    assert result.updates_completed == 3
    assert len(result.losses) == 3
    assert result.selected_update in {1, 2, 3}
    assert len(state_dict_sha256(result.model.state_dict())) == 64


def test_episode_uses_c0_once_and_logs_each_observed_state() -> None:
    bank = _synthetic_bank()
    indicator = bank.vlm_indicators.copy()
    confidence = bank.vlm_confidences.copy()
    indicator[0, [10, 11]] = 1.0
    confidence[0, [10, 11]] = 2.0 / 3.0
    bank = replace(bank, vlm_indicators=indicator, vlm_confidences=confidence)
    torch.manual_seed(4)
    predictor = CommonCAIPredictor(token_dimension=8, width=16)
    actor = CAIActor(token_dimension=8, width=16, use_vlm=True, use_feedback=True)

    episode = run_episode(
        bank,
        specimen_index=0,
        method=Method.VLM_CAI_FEEDBACK_AGENT,
        predictor=predictor,
        actor=actor,
        seed=1,
        endpoint_budget=0.0625,
        device="cpu",
        sample_actions=False,
    )

    assert episode.cells[0] in {10, 11}
    assert episode.rows[0]["proposal_restricted"] is True
    assert all(
        row["actor_call_index"] == index for index, row in enumerate(episode.rows)
    )
    assert len({row["observed_state_id"] for row in episode.rows}) == len(episode.rows)
    assert all(row["vlm_cache_key"] == "a" * 64 for row in episode.rows)
    assert all(len(str(row["environment_legal_mask"])) == 64 for row in episode.rows)
    assert all(len(str(row["initial_proposal_mask"])) == 64 for row in episode.rows)
    assert all(row["initial_proposal_reason"] == "HIGHEST_RELIABLE_CONFIDENCE_C0" for row in episode.rows)
    assert episode.rows[0]["action_history_cells"] == ""
    assert float(episode.rows[0]["remaining_cost"]) == pytest.approx(0.0625)
    assert all(float(row["actor_latency_seconds"]) >= 0.0 for row in episode.rows)
    assert all(float(row["predictor_latency_seconds"]) >= 0.0 for row in episode.rows)
    validate_trajectory_rows(episode.rows)


def test_registered_outputs_report_vlm_initialization_and_runtime_costs() -> None:
    root = Path(__file__).resolve().parents[1]
    prior = json.loads(
        (root / "results/cai_active_image/v2/vlm_prior_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    features = json.loads(
        (root / "results/cai_active_image/v2/feature_bank_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    with (root / "results/cai_active_image/v2/trajectories.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        first_trajectory = next(csv.DictReader(handle))

    assert set(prior["confidence_cell_counts"]) == {
        "unknown",
        "low",
        "medium",
        "high",
    }
    assert 0.0 <= float(prior["no_reliable_cue_fraction"]) <= 1.0
    assert float(features["encoder_elapsed_seconds"]) >= 0.0
    assert len(first_trajectory["environment_legal_mask"]) == 64
    assert len(first_trajectory["initial_proposal_mask"]) == 64
    assert float(first_trajectory["actor_latency_seconds"]) >= 0.0
    assert float(first_trajectory["predictor_latency_seconds"]) >= 0.0
    assert first_trajectory["initial_proposal_reason"]
    assert float(first_trajectory["remaining_cost"]) >= 0.0

    with (root / "results/cai_active_image/v2/domain_cai_performance.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        domain_rows = list(csv.DictReader(handle))
    assert all("gap_to_full_input_mae_mpa" in row for row in domain_rows)

    with (root / "results/cai_active_image/v2/validation_method_selection.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        selection_rows = list(csv.DictReader(handle))
    assert {row["method"] for row in selection_rows} == {method.value for method in Method}
    assert sum(row["selected_best_nonadaptive"] == "True" for row in selection_rows) == 1


def test_actor_training_obeys_exact_update_limit() -> None:
    bank = _synthetic_bank()
    predictor = train_predictor(
        bank,
        fit_indices=np.arange(6),
        updates=1,
        batch_size=2,
        width=16,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
        seed=3,
        device="cpu",
    ).model

    result = train_actor(
        bank,
        method=Method.NO_VLM_FEEDBACK,
        reward_predictors=(predictor,),
        reward_query_folds=(np.arange(6),),
        validation_indices=np.arange(6, 9),
        updates=2,
        batch_size=2,
        width=16,
        endpoint_budget=0.0625,
        validation_interval=1,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
        entropy_weight=0.01,
        value_weight=0.5,
        seed=5,
        device="cpu",
    )

    assert result.updates_completed == 2
    assert len(result.training_losses) == 2
    assert result.selected_update in {1, 2}

    validate_trajectory_rows(
        [
            _trajectory_row(0, 0.0, 0.015625, 3),
            _trajectory_row(1, 0.015625, 0.03125, 4, (3,)),
        ]
    )
    with pytest.raises(ValueError, match="actor call"):
        validate_trajectory_rows([_trajectory_row(1, 0.0, 0.015625, 3)])
