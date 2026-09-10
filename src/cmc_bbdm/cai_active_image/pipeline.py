"""End-to-end staged execution for the frozen CAI active-image v2 study."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import ACTOR_STATE_PROTOCOL, INITIAL_PROPOSAL_RULE, Method
from .data import load_mpa_targets
from .episodes import EpisodeResult, run_episode
from .features import FeatureBank, build_feature_bank, save_feature_bank
from .models import CAIActor, CommonCAIPredictor
from .perception import load_frozen_vlm_cache
from .protocol import (
    EXTERNAL_SOURCE_ROOT_BINDING,
    CAIActiveImageProtocol,
    load_protocol,
)
from .statistics import (
    domain_clustered_paired_bootstrap,
    normalized_error_area_mpa,
)
from .training import (
    ActorTrainingResult,
    PredictorTrainingResult,
    state_dict_sha256,
    train_actor,
    train_predictor,
)
from .validation import validate_common_predictor_identity, validate_trajectory_rows

DEFAULT_CONFIG = Path("paper_v3/configs/cai_active_image_v2.yaml")
_LEARNED_METHODS = (
    Method.VLM_CAI_FEEDBACK_AGENT,
    Method.NO_VLM_FEEDBACK,
    Method.VLM_OPEN_LOOP,
    Method.LEARNED_STATIC,
)
_FIXED_METHODS = (
    Method.SERPENTINE,
    Method.CENTER_FIRST,
    Method.GEOMETRY_SPREAD,
    Method.RANDOM,
)
_NONADAPTIVE_METHODS = (*_FIXED_METHODS, Method.LEARNED_STATIC)
_ALL_METHODS = (*_LEARNED_METHODS, *_FIXED_METHODS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_registered_protocol(project_root: str | Path) -> CAIActiveImageProtocol:
    root = Path(project_root).resolve(strict=True)
    return load_protocol(root / DEFAULT_CONFIG, project_root=root)


def prepare_study(
    *, project_root: str | Path, source_root: str | Path
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    Path(source_root).resolve(strict=True)
    protocol = load_registered_protocol(root)
    cache = load_frozen_vlm_cache(
        protocol.source("vlm_cache"), protocol.source("vlm_manifest")
    )
    protocol.results_dir.mkdir(parents=True, exist_ok=True)
    protocol.artifacts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(protocol.config_path, protocol.results_dir / "config.yaml")

    perception_manifest = json.loads(
        protocol.source("vlm_manifest").read_text(encoding="utf-8")
    )
    split_rows = [
        {
            "specimen_key": row["specimen_key"],
            "split": row["split"],
            "cache_key": row["cache_key"],
        }
        for row in perception_manifest["records"]
    ]
    _write_csv(protocol.results_dir / "split_manifest.csv", split_rows)
    vlm_rows = []
    for specimen_key in sorted(cache):
        item = cache[specimen_key]
        vlm_rows.append(
            {
                "specimen_key": specimen_key,
                "cache_key": item.cache_key,
                "cache_hit": item.cache_hit,
                "actual_call_count": item.actual_call_count,
                "original_call_count": item.original_call_count,
                "repaired": item.repaired,
                "latency_seconds": item.latency_seconds,
                "no_reliable_cue": item.features.no_reliable_cue,
                "reliable_candidates": [
                    int(value) for value in np.flatnonzero(item.features.reliable_candidates)
                ],
                "regions": [
                    {
                        "cells": list(region.cells),
                        "cue": region.cue,
                        "alternative": region.alternative,
                        "confidence": region.confidence,
                    }
                    for region in item.percept.regions
                ],
                "raw_text": item.raw_text,
            }
        )
    vlm_manifest = {
        "schema_version": 2,
        "model_repository": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": protocol.vlm_revision,
        "prompt_sha256": protocol.vlm_prompt_sha256,
        "render_version": protocol.vlm_render_version,
        "initial_proposal_rule": protocol.initial_proposal_rule,
        "cache_source": str(protocol.source("vlm_cache").relative_to(root)),
        "cache_source_sha256": _sha256_file(protocol.source("vlm_cache")),
        "specimen_count": len(cache),
        "cache_hits": len(cache),
        "new_calls": 0,
        "original_calls": sum(item.original_call_count for item in cache.values()),
        "format_repairs": sum(item.repaired for item in cache.values()),
        "confidence_cell_counts": {
            confidence: sum(
                len(region.cells)
                for item in cache.values()
                for region in item.percept.regions
                if region.confidence == confidence
            )
            for confidence in ("unknown", "low", "medium", "high")
        },
        "no_reliable_cue_fraction": sum(
            item.features.no_reliable_cue for item in cache.values()
        )
        / len(cache),
        "reliable_initialization_specimens": sum(
            bool(item.features.reliable_candidates.any()) for item in cache.values()
        ),
        "records": vlm_rows,
    }
    _write_json(protocol.results_dir / "vlm_prior_manifest.json", vlm_manifest)
    audit = {
        "schema_version": 2,
        "status": "P0_PROTOCOL_FROZEN",
        "repository_base_sha": protocol.repository_base_sha,
        "protocol_sha256": protocol.config_sha256,
        "source_root": EXTERNAL_SOURCE_ROOT_BINDING,
        "source_bindings": {item.name: item.sha256 for item in protocol.sources},
        "cohort_count": 60,
        "split_counts": {"TRAIN": 24, "VALID": 12, "TEST": 24},
        "vlm_cache_coverage": "60/60",
        "vlm_new_calls": 0,
        "true_label": "author CAI strength",
        "true_label_unit": "MPa",
        "action_protocol": "NATIVE_8X8_FULL_CELL_V1",
        "initial_proposal_rule": protocol.initial_proposal_rule,
        "actor_state_protocol": protocol.actor_state_protocol,
        "endpoint_budget": protocol.endpoint_budget,
        "total_optimizer_updates": protocol.total_optimizer_updates,
        "no_new_training_during_prepare": True,
    }
    _write_json(protocol.results_dir / "p0_protocol_audit.json", audit)
    return audit


def encode_features(
    *,
    project_root: str | Path,
    source_root: str | Path,
    device: str | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    protocol = load_registered_protocol(root)
    bank, manifest = build_feature_bank(
        protocol,
        project_root=root,
        source_root=source_root,
        device=device,
    )
    bank_path, manifest_path = save_feature_bank(
        bank, manifest, output_dir=protocol.results_dir
    )
    return {
        "status": "P1_FEATURE_BANK_COMPLETE",
        "feature_bank": str(bank_path),
        "feature_bank_sha256": _sha256_file(bank_path),
        "manifest": str(manifest_path),
        "specimen_count": len(bank.specimen_keys),
    }


def _fold_queries(bank: FeatureBank) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignments: list[list[int]] = [[], [], []]
    train = bank.indices("TRAIN")
    domains = {bank.dataset_ids[int(index)] for index in train}
    for domain in sorted(domains):
        members = [
            int(index) for index in train if bank.dataset_ids[int(index)] == domain
        ]
        ranked = sorted(
            members,
            key=lambda index: (
                hashlib.sha256(
                    f"cai-v2-return-fold|{bank.specimen_keys[index]}".encode()
                ).hexdigest(),
                bank.specimen_keys[index],
            ),
        )
        for rank, index in enumerate(ranked):
            assignments[rank % 3].append(index)
    folds = tuple(np.asarray(sorted(values), dtype=np.int64) for values in assignments)
    if set(np.concatenate(folds).tolist()) != set(train.tolist()):
        raise ValueError("TRAIN return folds do not partition the cohort")
    if any({bank.dataset_ids[int(index)] for index in fold} != domains for fold in folds):
        raise ValueError("each TRAIN return fold must cover all domains")
    return folds  # type: ignore[return-value]


def _predictor_path(protocol: CAIActiveImageProtocol, name: str) -> Path:
    return protocol.results_dir / "models" / f"predictor_{name}.pt"


def _actor_path(protocol: CAIActiveImageProtocol, method: Method, seed: int) -> Path:
    return protocol.results_dir / "models" / f"actor_{method.value.lower()}_seed{seed}.pt"


def _save_predictor(
    path: Path,
    result: PredictorTrainingResult,
    *,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    bank: FeatureBank,
    name: str,
) -> dict[str, object]:
    state = {key: value.detach().cpu() for key, value in result.model.state_dict().items()}
    payload = {
        "schema_version": 2,
        "kind": "COMMON_CAI_PREDICTOR",
        "name": name,
        "token_dimension": bank.token_dimension,
        "width": result.model.surface.out_features,
        "target_mean": float(result.model.target_mean.cpu()),
        "target_scale": float(result.model.target_scale.cpu()),
        "updates_completed": result.updates_completed,
        "selected_update": result.selected_update,
        "validation_scores": [list(item) for item in result.validation_scores],
        "fit_specimen_keys": [bank.specimen_keys[int(index)] for index in fit_indices],
        "query_specimen_keys": [bank.specimen_keys[int(index)] for index in query_indices],
        "losses": list(result.losses),
        "state_dict_sha256": state_dict_sha256(state),
        "state_dict": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {key: value for key, value in payload.items() if key != "state_dict"}


def _load_predictor(path: Path, *, device: str) -> tuple[CommonCAIPredictor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if type(payload) is not dict or payload.get("kind") != "COMMON_CAI_PREDICTOR":
        raise ValueError("predictor checkpoint identity is invalid")
    model = CommonCAIPredictor(
        token_dimension=int(payload["token_dimension"]),
        width=int(payload["width"]),
        target_mean=float(payload["target_mean"]),
        target_scale=float(payload["target_scale"]),
    )
    model.load_state_dict(payload["state_dict"])
    if state_dict_sha256(model.state_dict()) != payload["state_dict_sha256"]:
        raise ValueError("predictor checkpoint state hash changed")
    model.to(device).eval()
    return model, payload


def train_predictors(
    *, project_root: str | Path, device: str | None = None
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    protocol = load_registered_protocol(root)
    active_device = device or protocol.device
    bank = FeatureBank.load(protocol.results_dir / "feature_bank.npz")
    train = bank.indices("TRAIN")
    queries = _fold_queries(bank)
    records: list[dict[str, object]] = []
    for fold_index in range(3):
        query = queries[fold_index]
        fit = np.asarray(sorted(set(train.tolist()) - set(query.tolist())), dtype=np.int64)
        name = f"fold{fold_index}"
        path = _predictor_path(protocol, name)
        if path.exists():
            _, payload = _load_predictor(path, device=active_device)
            if payload.get("updates_completed") != protocol.predictor_updates[fold_index]:
                raise ValueError("existing predictor checkpoint update count changed")
            record = {key: value for key, value in payload.items() if key not in {"state_dict", "losses"}}
            record["checkpoint_reused"] = True
        else:
            result = train_predictor(
                bank,
                fit_indices=fit,
                updates=protocol.predictor_updates[fold_index],
                batch_size=protocol.predictor_batch_size,
                width=protocol.predictor_width,
                learning_rate=protocol.learning_rate,
                weight_decay=protocol.weight_decay,
                gradient_clip=protocol.gradient_clip,
                seed=100 + fold_index,
                device=active_device,
                validation_indices=bank.indices("VALID"),
                validation_interval=protocol.predictor_validation_interval,
                mask_counts=protocol.predictor_mask_counts,
            )
            record = _save_predictor(
                path,
                result,
                fit_indices=fit,
                query_indices=query,
                bank=bank,
                name=name,
            )
            record.pop("losses", None)
            record["checkpoint_reused"] = False
        record["checkpoint"] = str(path.relative_to(root))
        record["checkpoint_sha256"] = _sha256_file(path)
        records.append(record)

    path = _predictor_path(protocol, "all")
    if path.exists():
        _, payload = _load_predictor(path, device=active_device)
        if payload.get("updates_completed") != protocol.predictor_updates[3]:
            raise ValueError("existing all-TRAIN predictor update count changed")
        record = {key: value for key, value in payload.items() if key not in {"state_dict", "losses"}}
        record["checkpoint_reused"] = True
    else:
        result = train_predictor(
            bank,
            fit_indices=train,
            updates=protocol.predictor_updates[3],
            batch_size=protocol.predictor_batch_size,
            width=protocol.predictor_width,
            learning_rate=protocol.learning_rate,
            weight_decay=protocol.weight_decay,
            gradient_clip=protocol.gradient_clip,
            seed=103,
            device=active_device,
            validation_indices=bank.indices("VALID"),
            validation_interval=protocol.predictor_validation_interval,
            mask_counts=protocol.predictor_mask_counts,
        )
        record = _save_predictor(
            path,
            result,
            fit_indices=train,
            query_indices=np.asarray([], dtype=np.int64),
            bank=bank,
            name="all",
        )
        record.pop("losses", None)
        record["checkpoint_reused"] = False
    record["checkpoint"] = str(path.relative_to(root))
    record["checkpoint_sha256"] = _sha256_file(path)
    records.append(record)
    manifest = {
        "schema_version": 2,
        "status": "P2_COMMON_PREDICTORS_COMPLETE",
        "models": records,
        "optimizer_updates": sum(protocol.predictor_updates),
        "return_protocol": "TRAIN_THREE_FOLD_OUT_OF_FOLD",
        "deployment_predictor": "all",
    }
    _write_json(protocol.results_dir / "predictor_training_manifest.json", manifest)
    return manifest


def _save_actor(
    path: Path,
    result: ActorTrainingResult,
    *,
    bank: FeatureBank,
    method: Method,
    seed: int,
) -> dict[str, object]:
    state = {key: value.detach().cpu() for key, value in result.model.state_dict().items()}
    payload = {
        "schema_version": 2,
        "kind": "CAI_ACQUISITION_ACTOR",
        "method": method.value,
        "seed": seed,
        "token_dimension": bank.token_dimension,
        "width": result.model.surface.out_features,
        "use_vlm": result.model.use_vlm,
        "use_feedback": result.model.use_feedback,
        "initial_proposal_rule": INITIAL_PROPOSAL_RULE,
        "actor_state_protocol": ACTOR_STATE_PROTOCOL,
        "updates_completed": result.updates_completed,
        "selected_update": result.selected_update,
        "training_losses": list(result.training_losses),
        "validation_scores": [list(item) for item in result.validation_scores],
        "state_dict_sha256": state_dict_sha256(state),
        "state_dict": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {key: value for key, value in payload.items() if key != "state_dict"}


def _load_actor(path: Path, *, device: str) -> tuple[CAIActor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        type(payload) is not dict
        or payload.get("kind") != "CAI_ACQUISITION_ACTOR"
        or payload.get("initial_proposal_rule") != INITIAL_PROPOSAL_RULE
        or payload.get("actor_state_protocol") != ACTOR_STATE_PROTOCOL
    ):
        raise ValueError("actor checkpoint identity is invalid")
    model = CAIActor(
        token_dimension=int(payload["token_dimension"]),
        width=int(payload["width"]),
        use_vlm=bool(payload["use_vlm"]),
        use_feedback=bool(payload["use_feedback"]),
    )
    model.load_state_dict(payload["state_dict"])
    if state_dict_sha256(model.state_dict()) != payload["state_dict_sha256"]:
        raise ValueError("actor checkpoint state hash changed")
    model.to(device).eval()
    return model, payload


def train_actors(
    *, project_root: str | Path, device: str | None = None
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    protocol = load_registered_protocol(root)
    active_device = device or protocol.device
    bank = FeatureBank.load(protocol.results_dir / "feature_bank.npz")
    query_folds = _fold_queries(bank)
    reward_predictors = tuple(
        _load_predictor(_predictor_path(protocol, f"fold{index}"), device=active_device)[0]
        for index in range(3)
    )
    p_all, p_all_payload = _load_predictor(
        _predictor_path(protocol, "all"), device=active_device
    )
    valid = bank.indices("VALID")
    records: list[dict[str, object]] = []
    started = time.monotonic()
    schedule = tuple(
        (method, seed) for seed in protocol.seeds for method in _LEARNED_METHODS
    )
    for method, seed in schedule:
        updates = (
            protocol.static_updates
            if method is Method.LEARNED_STATIC
            else protocol.actor_updates
        )
        elapsed_hours = (time.monotonic() - started) / 3600.0
        if elapsed_hours >= protocol.wall_clock_limit_hours:
            progress = {
                "schema_version": 2,
                "status": "WALL_CLOCK_LIMIT_REACHED",
                "elapsed_hours": elapsed_hours,
                "completed_models": records,
            }
            _write_json(protocol.results_dir / "actor_training_progress.json", progress)
            return progress
        path = _actor_path(protocol, method, seed)
        if path.exists():
            _, payload = _load_actor(path, device=active_device)
            if payload.get("updates_completed") != updates:
                raise ValueError("existing actor checkpoint update count changed")
            record = {
                key: value
                for key, value in payload.items()
                if key not in {"state_dict", "training_losses"}
            }
            record["checkpoint_reused"] = True
        else:
            result = train_actor(
                bank,
                method=method,
                reward_predictors=reward_predictors,
                reward_query_folds=query_folds,
                validation_indices=valid,
                updates=updates,
                batch_size=protocol.actor_batch_size,
                width=protocol.actor_width,
                endpoint_budget=protocol.endpoint_budget,
                validation_interval=protocol.validation_interval,
                learning_rate=protocol.learning_rate,
                weight_decay=protocol.weight_decay,
                gradient_clip=protocol.gradient_clip,
                entropy_weight=protocol.entropy_weight,
                value_weight=protocol.value_weight,
                seed=seed,
                device=active_device,
                validation_predictor=p_all,
            )
            record = _save_actor(path, result, bank=bank, method=method, seed=seed)
            record.pop("training_losses", None)
            record["checkpoint_reused"] = False
        record["checkpoint"] = str(path.relative_to(root))
        record["checkpoint_sha256"] = _sha256_file(path)
        record["common_predictor_state_sha256"] = p_all_payload["state_dict_sha256"]
        records.append(record)
        _write_json(
            protocol.results_dir / "actor_training_progress.json",
            {
                "schema_version": 2,
                "status": "IN_PROGRESS",
                "completed_models": records,
                "elapsed_hours": (time.monotonic() - started) / 3600.0,
            },
        )
    manifest = {
        "schema_version": 2,
        "status": "P3_P4_ACTORS_COMPLETE",
        "models": records,
        "optimizer_updates": len(protocol.seeds)
        * (3 * protocol.actor_updates + protocol.static_updates),
        "common_predictor_state_sha256": p_all_payload["state_dict_sha256"],
        "elapsed_hours": (time.monotonic() - started) / 3600.0,
    }
    _write_json(protocol.results_dir / "actor_training_manifest.json", manifest)
    _write_json(protocol.results_dir / "actor_training_progress.json", manifest)
    return manifest


def _method_seeds(method: Method, protocol: CAIActiveImageProtocol) -> tuple[int, ...]:
    if method.learned or method is Method.RANDOM:
        return protocol.seeds
    return (0,)


def _early_area(episode: EpisodeResult, endpoint: float, target: float) -> float:
    costs = np.asarray(episode.costs)
    predictions = np.asarray(episode.predictions_mpa)
    keep = costs <= endpoint + 1e-12
    selected_costs = costs[keep]
    selected_predictions = predictions[keep]
    if len(selected_costs) == 1 or selected_costs[-1] < endpoint:
        selected_costs = np.append(selected_costs, endpoint)
        selected_predictions = np.append(selected_predictions, selected_predictions[-1])
    return normalized_error_area_mpa(
        costs=selected_costs,
        predictions_mpa=selected_predictions,
        target_mpa=target,
        endpoint_budget=endpoint,
    )


def _prediction_at_budget(episode: EpisodeResult, budget: float) -> float:
    costs = np.asarray(episode.costs)
    valid = np.flatnonzero(costs <= budget + 1e-12)
    if len(valid) == 0:
        raise ValueError("trajectory does not contain absolute zero")
    return float(episode.predictions_mpa[int(valid[-1])])


def _evaluate_split(
    bank: FeatureBank,
    *,
    indices: np.ndarray,
    methods: tuple[Method, ...],
    protocol: CAIActiveImageProtocol,
    predictor: CommonCAIPredictor,
    actors: dict[tuple[Method, int], CAIActor],
    device: str,
) -> tuple[list[EpisodeResult], list[dict[str, object]]]:
    episodes: list[EpisodeResult] = []
    rows: list[dict[str, object]] = []
    for method in methods:
        for seed in _method_seeds(method, protocol):
            actor = actors.get((method, seed))
            for index in indices:
                episode = run_episode(
                    bank,
                    specimen_index=int(index),
                    method=method,
                    predictor=predictor,
                    actor=actor,
                    seed=seed,
                    endpoint_budget=protocol.endpoint_budget,
                    device=device,
                    sample_actions=False,
                )
                validate_trajectory_rows(episode.rows)
                target = float(bank.targets_mpa[int(index)])
                episodes.append(episode)
                rows.append(
                    {
                        "specimen_key": episode.specimen_key,
                        "dataset_id": bank.dataset_ids[int(index)],
                        "split": bank.splits[int(index)],
                        "method": method.value,
                        "seed": seed,
                        "target_mpa": target,
                        "early_normalized_error_area_mpa": _early_area(
                            episode, protocol.early_budget, target
                        ),
                        "normalized_error_area_mpa": episode.normalized_error_area_mpa,
                        "final_prediction_mpa": episode.predictions_mpa[-1],
                        "final_absolute_error_mpa": episode.final_absolute_error_mpa,
                        "final_cost": episode.costs[-1],
                        "acquired_cells": ";".join(str(cell) for cell in episode.cells),
                    }
                )
    return episodes, rows


def _mean_episode_score(rows: list[dict[str, object]], method: Method) -> float:
    by_specimen: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method.value:
            by_specimen[str(row["specimen_key"])].append(
                float(row["normalized_error_area_mpa"])
            )
    return float(np.mean([np.mean(values) for values in by_specimen.values()]))


def _aggregate_prediction_rows(
    bank: FeatureBank,
    episodes: list[EpisodeResult],
    *,
    indices: np.ndarray,
    protocol: CAIActiveImageProtocol,
    predictor: CommonCAIPredictor,
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lookup = {bank.specimen_keys[int(index)]: int(index) for index in indices}
    grouped: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for episode in episodes:
        for budget in protocol.reporting_budgets:
            grouped[(episode.method.value, episode.specimen_key, budget)].append(
                _prediction_at_budget(episode, budget)
            )
    specimen_rows: list[dict[str, object]] = []
    for (method, specimen_key, budget), predictions in sorted(grouped.items()):
        index = lookup[specimen_key]
        target = float(bank.targets_mpa[index])
        prediction = float(np.mean(predictions))
        specimen_rows.append(
            {
                "method": method,
                "budget": budget,
                "specimen_key": specimen_key,
                "dataset_id": bank.dataset_ids[index],
                "target_mpa": target,
                "prediction_mpa": prediction,
                "absolute_error_mpa": abs(prediction - target),
                "seed_count": len(predictions),
            }
        )

    full_predictions: dict[str, float] = {}
    with torch.inference_mode():
        for specimen_key, index in lookup.items():
            surface = torch.from_numpy(bank.surface_tokens[index : index + 1]).to(device)
            cscan = torch.from_numpy(bank.cscan_tokens[index : index + 1]).to(device)
            measured = torch.ones(1, 64, dtype=torch.bool, device=device)
            full_predictions[specimen_key] = float(
                predictor(
                    surface,
                    cscan,
                    measured,
                    cost=torch.ones(1, dtype=torch.float32, device=device),
                )[0].cpu()
            )
    full_errors = np.asarray(
        [
            abs(full_predictions[key] - float(bank.targets_mpa[index]))
            for key, index in lookup.items()
        ]
    )
    full_mae = float(np.mean(full_errors))

    performance_rows: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in specimen_rows}):
        for budget in protocol.reporting_budgets:
            subset = [
                row
                for row in specimen_rows
                if row["method"] == method and float(row["budget"]) == budget
            ]
            y = np.asarray([float(row["target_mpa"]) for row in subset])
            pred = np.asarray([float(row["prediction_mpa"]) for row in subset])
            domain_maes = [
                float(
                    np.mean(
                        [
                            float(row["absolute_error_mpa"])
                            for row in subset
                            if row["dataset_id"] == domain
                        ]
                    )
                )
                for domain in protocol.domain_order
            ]
            denominator = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - float(np.sum((y - pred) ** 2)) / denominator
            performance_rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "state": "ZERO_CSCAN" if budget == 0.0 else ("ENDPOINT" if budget == protocol.endpoint_budget else "INTERMEDIATE"),
                    "mae_mpa": float(np.mean(np.abs(y - pred))),
                    "equal_domain_mae_mpa": float(np.mean(domain_maes)),
                    "rmse_mpa": float(np.sqrt(np.mean((y - pred) ** 2))),
                    "r2": r2,
                    "gap_to_full_input_mae_mpa": float(np.mean(np.abs(y - pred))) - full_mae,
                    "specimen_count": len(subset),
                }
            )
    y_full = np.asarray([float(bank.targets_mpa[index]) for index in lookup.values()])
    pred_full = np.asarray([full_predictions[key] for key in lookup])
    full_domain_maes = [
        float(
            np.mean(
                [
                    abs(full_predictions[key] - float(bank.targets_mpa[index]))
                    for key, index in lookup.items()
                    if bank.dataset_ids[index] == domain
                ]
            )
        )
        for domain in protocol.domain_order
    ]
    denominator = float(np.sum((y_full - np.mean(y_full)) ** 2))
    performance_rows.append(
        {
            "method": "FULL_INPUT_COMMON_PREDICTOR",
            "budget": 1.0,
            "state": "FULL_64_CELL_INPUT",
            "mae_mpa": full_mae,
            "equal_domain_mae_mpa": float(np.mean(full_domain_maes)),
            "rmse_mpa": float(np.sqrt(np.mean((y_full - pred_full) ** 2))),
            "r2": 1.0 - float(np.sum((y_full - pred_full) ** 2)) / denominator,
            "gap_to_full_input_mae_mpa": 0.0,
            "specimen_count": len(lookup),
        }
    )
    for specimen_key, prediction in full_predictions.items():
        index = lookup[specimen_key]
        target = float(bank.targets_mpa[index])
        specimen_rows.append(
            {
                "method": "FULL_INPUT_COMMON_PREDICTOR",
                "budget": 1.0,
                "specimen_key": specimen_key,
                "dataset_id": bank.dataset_ids[index],
                "target_mpa": target,
                "prediction_mpa": prediction,
                "absolute_error_mpa": abs(prediction - target),
                "seed_count": 1,
            }
        )
    return specimen_rows, performance_rows


def _effect_rows(
    episode_rows: list[dict[str, object]],
    *,
    best_nonadaptive: Method,
    protocol: CAIActiveImageProtocol,
) -> list[dict[str, object]]:
    metrics = {
        "VLM_EARLY_GUIDANCE": (
            Method.NO_VLM_FEEDBACK,
            "early_normalized_error_area_mpa",
        ),
        "FEEDBACK_ACQUISITION": (
            Method.VLM_OPEN_LOOP,
            "normalized_error_area_mpa",
        ),
        "BEST_NONADAPTIVE_ACQUISITION": (
            best_nonadaptive,
            "normalized_error_area_mpa",
        ),
    }
    averaged: dict[tuple[str, str], dict[str, object]] = {}
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in episode_rows:
        groups[(str(row["method"]), str(row["specimen_key"]))].append(row)
    for key, values in groups.items():
        averaged[key] = {
            "dataset_id": values[0]["dataset_id"],
            "early_normalized_error_area_mpa": float(
                np.mean([float(row["early_normalized_error_area_mpa"]) for row in values])
            ),
            "normalized_error_area_mpa": float(
                np.mean([float(row["normalized_error_area_mpa"]) for row in values])
            ),
        }
    output: list[dict[str, object]] = []
    for offset, (claim, (comparator, metric)) in enumerate(metrics.items()):
        pairs: list[tuple[str, str, float]] = []
        for (method, specimen_key), main in averaged.items():
            if method != Method.VLM_CAI_FEEDBACK_AGENT.value:
                continue
            other = averaged[(comparator.value, specimen_key)]
            pairs.append(
                (
                    str(main["dataset_id"]),
                    specimen_key,
                    float(other[metric]) - float(main[metric]),
                )
            )
        interval = domain_clustered_paired_bootstrap(
            pairs,
            replicates=protocol.bootstrap_replicates,
            seed=protocol.bootstrap_seed + offset,
        )
        output.append(
            {
                "claim": claim,
                "metric": metric,
                "comparison": f"{comparator.value}_minus_{Method.VLM_CAI_FEEDBACK_AGENT.value}",
                "estimate_mpa": interval.estimate,
                "ci_lower_mpa": interval.lower,
                "ci_upper_mpa": interval.upper,
                "confidence_level": interval.confidence_level,
                "bootstrap_replicates": interval.replicates,
                "bootstrap_seed": interval.seed,
                "direction": "POSITIVE_FAVORS_MAIN",
                "evidence_status": "SUPPORTED" if interval.lower > 0.0 else "NOT_SUPPORTED",
            }
        )
    return output


def evaluate_study(
    *, project_root: str | Path, device: str | None = None
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    protocol = load_registered_protocol(root)
    active_device = device or protocol.device
    bank = FeatureBank.load(protocol.results_dir / "feature_bank.npz")
    targets = load_mpa_targets(protocol.source("cai_mpa_authority"))
    scoring_targets = bank.targets_mpa.copy()
    for index, specimen_key in enumerate(bank.specimen_keys):
        authoritative = targets[specimen_key]
        if bank.splits[index] == "TEST":
            if not np.isnan(scoring_targets[index]):
                raise ValueError("TEST target was present before scoring join")
            scoring_targets[index] = authoritative
        elif not np.isclose(
            scoring_targets[index], authoritative, rtol=0.0, atol=1e-4
        ):
            raise ValueError("fit/validation target differs from CAI authority")
    bank = replace(bank, targets_mpa=scoring_targets)
    predictor, predictor_payload = _load_predictor(
        _predictor_path(protocol, "all"), device=active_device
    )
    actors: dict[tuple[Method, int], CAIActor] = {}
    for method in _LEARNED_METHODS:
        for seed in protocol.seeds:
            actors[(method, seed)] = _load_actor(
                _actor_path(protocol, method, seed), device=active_device
            )[0]
    predictor_hashes = {
        method: str(predictor_payload["state_dict_sha256"])
        for method in (*_LEARNED_METHODS, *_FIXED_METHODS)
    }
    validate_common_predictor_identity(predictor_hashes)

    valid_episodes, valid_rows = _evaluate_split(
        bank,
        indices=bank.indices("VALID"),
        methods=_ALL_METHODS,
        protocol=protocol,
        predictor=predictor,
        actors=actors,
        device=active_device,
    )
    del valid_episodes
    valid_scores = {
        method: _mean_episode_score(valid_rows, method) for method in _ALL_METHODS
    }
    nonadaptive_scores = [
        {
            "method": method.value,
            "valid_normalized_error_area_mpa": valid_scores[method],
        }
        for method in _NONADAPTIVE_METHODS
    ]
    best_row = min(
        nonadaptive_scores,
        key=lambda row: (float(row["valid_normalized_error_area_mpa"]), str(row["method"])),
    )
    best_nonadaptive = Method(str(best_row["method"]))
    main_valid_score = valid_scores[Method.VLM_CAI_FEEDBACK_AGENT]
    selection_rows = [
        {
            "method": method.value,
            "method_role": (
                "MAIN"
                if method is Method.VLM_CAI_FEEDBACK_AGENT
                else (
                    "NONADAPTIVE_CANDIDATE"
                    if method in _NONADAPTIVE_METHODS
                    else "VLM_ABLATION"
                )
            ),
            "valid_normalized_error_area_mpa": valid_scores[method],
            "difference_from_main_mpa": valid_scores[method] - main_valid_score,
            "selected_best_nonadaptive": method is best_nonadaptive,
        }
        for method in _ALL_METHODS
    ]
    _write_csv(protocol.results_dir / "validation_method_selection.csv", selection_rows)

    test_episodes, test_rows = _evaluate_split(
        bank,
        indices=bank.indices("TEST"),
        methods=(*_LEARNED_METHODS, *_FIXED_METHODS),
        protocol=protocol,
        predictor=predictor,
        actors=actors,
        device=active_device,
    )
    trajectory_rows = [row for episode in test_episodes for row in episode.rows]
    _write_csv(protocol.results_dir / "test_episodes.csv", test_rows)
    _write_csv(protocol.results_dir / "trajectories.csv", trajectory_rows)
    initialization_rows = [
        {
            "specimen_key": episode.specimen_key,
            "method": episode.method.value,
            "seed": episode.seed,
            "vlm_cache_key": episode.rows[0]["vlm_cache_key"],
            "vlm_available": episode.rows[0]["vlm_available"],
            "vlm_no_reliable_cue": bank.vlm_no_reliable[
                bank.specimen_keys.index(episode.specimen_key)
            ],
            "vlm_unknown_cell_count": int(
                np.sum(
                    (bank.vlm_indicators[bank.specimen_keys.index(episode.specimen_key)] > 0)
                    & (
                        bank.vlm_confidences[
                            bank.specimen_keys.index(episode.specimen_key)
                        ]
                        == 0.0
                    )
                )
            ),
            "vlm_low_cell_count": int(
                np.sum(
                    bank.vlm_confidences[
                        bank.specimen_keys.index(episode.specimen_key)
                    ]
                    == 1.0 / 3.0
                )
            ),
            "vlm_medium_cell_count": int(
                np.sum(
                    bank.vlm_confidences[
                        bank.specimen_keys.index(episode.specimen_key)
                    ]
                    == 2.0 / 3.0
                )
            ),
            "vlm_high_cell_count": int(
                np.sum(
                    bank.vlm_confidences[
                        bank.specimen_keys.index(episode.specimen_key)
                    ]
                    == 1.0
                )
            ),
            "initial_candidates": episode.rows[0]["initial_candidates"],
            "initial_action": episode.rows[0]["initial_action"],
            "proposal_restricted": episode.rows[0]["proposal_restricted"],
            "initial_proposal_reason": episode.rows[0]["initial_proposal_reason"],
            "initial_proposal_mask": episode.rows[0]["initial_proposal_mask"],
            "first_action_in_initial_proposal": episode.rows[0][
                "initial_proposal_mask"
            ][int(episode.rows[0]["initial_action"])]
            == "1",
        }
        for episode in test_episodes
    ]
    _write_csv(protocol.results_dir / "initialization_summary.csv", initialization_rows)
    specimen_rows, performance_rows = _aggregate_prediction_rows(
        bank,
        test_episodes,
        indices=bank.indices("TEST"),
        protocol=protocol,
        predictor=predictor,
        device=active_device,
    )
    _write_csv(protocol.results_dir / "per_specimen_predictions.csv", specimen_rows)
    _write_csv(protocol.results_dir / "absolute_cai_performance.csv", performance_rows)
    domain_rows: list[dict[str, object]] = []
    full_domain_mae = {
        domain: float(
            np.mean(
                [
                    float(row["absolute_error_mpa"])
                    for row in specimen_rows
                    if row["method"] == "FULL_INPUT_COMMON_PREDICTOR"
                    and row["dataset_id"] == domain
                ]
            )
        )
        for domain in protocol.domain_order
    }
    for method in sorted({str(row["method"]) for row in specimen_rows}):
        for budget in sorted(
            {float(row["budget"]) for row in specimen_rows if row["method"] == method}
        ):
            for domain in protocol.domain_order:
                subset = [
                    row
                    for row in specimen_rows
                    if row["method"] == method
                    and float(row["budget"]) == budget
                    and row["dataset_id"] == domain
                ]
                if subset:
                    mae = float(
                        np.mean([float(row["absolute_error_mpa"]) for row in subset])
                    )
                    domain_rows.append(
                        {
                            "method": method,
                            "budget": budget,
                            "dataset_id": domain,
                            "mae_mpa": mae,
                            "gap_to_full_input_mae_mpa": mae
                            - full_domain_mae[domain],
                            "specimen_count": len(subset),
                        }
                    )
    _write_csv(protocol.results_dir / "domain_cai_performance.csv", domain_rows)
    effects = _effect_rows(
        test_rows, best_nonadaptive=best_nonadaptive, protocol=protocol
    )
    _write_csv(protocol.results_dir / "vlm_ablation_effects.csv", effects)
    status = {
        "VLM_EARLY_GUIDANCE": {
            "wired": True,
            "evidence": effects[0]["evidence_status"],
        },
        "FEEDBACK_ACQUISITION": {
            "wired": True,
            "evidence": effects[1]["evidence_status"],
        },
        "CAI_ABSOLUTE_PERFORMANCE": {
            "wired": True,
            "evidence": "REPORTED",
            "engineering": "ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED",
        },
    }
    summary = {
        "schema_version": 2,
        "status": "P5_EVALUATION_COMPLETE",
        "best_nonadaptive": best_nonadaptive.value,
        "validation": {
            "main_normalized_error_area_mpa": main_valid_score,
            "no_vlm_normalized_error_area_mpa": valid_scores[
                Method.NO_VLM_FEEDBACK
            ],
            "best_nonadaptive_normalized_error_area_mpa": valid_scores[
                best_nonadaptive
            ],
            "main_beats_no_vlm": main_valid_score
            < valid_scores[Method.NO_VLM_FEEDBACK],
            "main_beats_best_nonadaptive": main_valid_score
            < valid_scores[best_nonadaptive],
        },
        "common_predictor_state_sha256": predictor_payload["state_dict_sha256"],
        "test_specimens": len(bank.indices("TEST")),
        "test_episode_count": len(test_episodes),
        "bootstrap_replicates": protocol.bootstrap_replicates,
        "statuses": status,
        "preselected_cases": list(protocol.preselected_cases),
        "initial_proposal_rule": protocol.initial_proposal_rule,
        "actor_state_protocol": protocol.actor_state_protocol,
        "test_target_join": "P5_SCORING_SIDE_ONLY",
        "artifacts": {
            "absolute_cai_performance": "absolute_cai_performance.csv",
            "domain_cai_performance": "domain_cai_performance.csv",
            "vlm_ablation_effects": "vlm_ablation_effects.csv",
            "initialization_summary": "initialization_summary.csv",
            "trajectories": "trajectories.csv",
        },
    }
    _write_json(protocol.results_dir / "summary.json", summary)
    return summary


__all__ = [
    "encode_features",
    "evaluate_study",
    "load_registered_protocol",
    "prepare_study",
    "train_actors",
    "train_predictors",
]
