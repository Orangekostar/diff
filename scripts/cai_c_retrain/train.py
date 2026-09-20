"""Three-actor C training with atomic candidates and one bounded resume."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.cai_c_retrain.context import (
    METHOD_SEEDS,
    TaskContext,
    atomic_json,
    canonical_json,
    sha256_file,
)

CANDIDATE_UPDATES = (250, 500, 750, 1000, 1250)
IMPROVEMENT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class Candidate:
    update: int
    score: float
    checkpoint: str = ""
    episodes: str = ""


def select_candidate(candidates: list[Candidate]) -> Candidate:
    if not candidates:
        raise ValueError("at least one candidate is required")
    if [candidate.update for candidate in candidates] != sorted(
        candidate.update for candidate in candidates
    ):
        raise ValueError("candidate updates must be strictly ordered")
    best = candidates[0]
    if not math.isfinite(best.score):
        raise ValueError("candidate score must be finite")
    for candidate in candidates[1:]:
        if not math.isfinite(candidate.score):
            raise ValueError("candidate score must be finite")
        if candidate.score < best.score - IMPROVEMENT_TOLERANCE:
            best = candidate
    return best


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_snapshot(
    path: str | Path,
    actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    *,
    logical_update: int,
    best_score: float,
    best_update: int,
    stale: int,
    progress: list[dict[str, Any]],
    environment_identity: str,
    input_signature: str,
    run_id: str,
) -> None:
    payload = {
        "schema_version": 1,
        "model": actor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_global_rng": np.random.get_state(),
        "numpy_generator_rng": rng.bit_generator.state,
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all()
        if next(actor.parameters()).is_cuda
        else [],
        "logical_update": logical_update,
        "best_score": best_score,
        "best_update": best_update,
        "stale": stale,
        "progress": progress,
        "environment_identity": environment_identity,
        "input_signature": input_signature,
        "run_id": run_id,
    }
    _atomic_torch(Path(path), payload)


def restore_snapshot(
    path: str | Path,
    actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    *,
    expected_environment: str,
    expected_signature: str,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path), map_location=next(actor.parameters()).device, weights_only=False
    )
    if payload.get("schema_version") != 1:
        raise ValueError("snapshot schema changed")
    if payload.get("environment_identity") != expected_environment:
        raise ValueError("snapshot environment changed")
    if payload.get("input_signature") != expected_signature:
        raise ValueError("snapshot input signature changed")
    actor.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_global_rng"])
    rng.bit_generator.state = payload["numpy_generator_rng"]
    torch.set_rng_state(payload["torch_cpu_rng"])
    if next(actor.parameters()).is_cuda:
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng"])
    return payload


def save_weight_state(
    path: str | Path,
    actor: torch.nn.Module,
    *,
    method: str,
    update: int,
    environment_identity: str,
    input_signature: str,
) -> None:
    _atomic_torch(
        Path(path),
        {
            "schema_version": 1,
            "method": method,
            "update": update,
            "environment_identity": environment_identity,
            "input_signature": input_signature,
            "state_dict": actor.state_dict(),
        },
    )


def load_weight_state(
    path: str | Path,
    actor: torch.nn.Module,
    *,
    expected_method: str,
    expected_environment: str,
    expected_signature: str,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path), map_location=next(actor.parameters()).device, weights_only=False
    )
    if payload.get("schema_version") != 1 or payload.get("method") != expected_method:
        raise ValueError("candidate weight identity changed")
    if payload.get("environment_identity") != expected_environment:
        raise ValueError("candidate environment changed")
    if payload.get("input_signature") != expected_signature:
        raise ValueError("candidate input signature changed")
    actor.load_state_dict(payload["state_dict"])
    return {
        key: payload[key]
        for key in ("method", "update", "environment_identity", "input_signature")
    }


class SegmentLedger:
    def __init__(
        self,
        path: str | Path,
        global_path: str | Path,
        *,
        task_id: str,
        method_cap: int,
    ) -> None:
        self.path = Path(path)
        self.global_path = Path(global_path)
        self.task_id = task_id
        self.method_cap = method_cap

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
        ]

    @staticmethod
    def _append(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(row).decode("utf-8") + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _open(self, method: str) -> list[dict[str, Any]]:
        rows = self._rows()
        terminals = {
            row["reservation_id"]
            for row in rows
            if row.get("status") in {"COMPLETED", "LOST_UPPER_BOUND"}
        }
        return [
            row
            for row in rows
            if row.get("method") == method
            and row.get("status") == "RESERVED"
            and row["reservation_id"] not in terminals
        ]

    def reserve(self, method: str, start: int, end: int, run_id: str) -> str:
        if self._open(method):
            raise ValueError("method already has an open segment")
        if start < 0 or end - start != 250 or end > 1250:
            raise ValueError("segment must be one authorized 250-update interval")
        if self.summary(method)["charged_upper_bound"] + 250 > self.method_cap:
            raise ValueError("method optimizer update cap exceeded")
        reservation_id = uuid.uuid4().hex
        self._append(
            self.path,
            {
                "task_id": self.task_id,
                "method": method,
                "run_id": run_id,
                "reservation_id": reservation_id,
                "start_update": start,
                "end_update": end,
                "status": "RESERVED",
                "recorded_unix": time.time(),
            },
        )
        return reservation_id

    def _reservation(self, reservation_id: str) -> dict[str, Any]:
        matches = [
            row
            for row in self._rows()
            if row.get("reservation_id") == reservation_id
            and row.get("status") == "RESERVED"
        ]
        if len(matches) != 1:
            raise ValueError("segment reservation is missing or ambiguous")
        return matches[0]

    def complete(self, reservation_id: str, *, actual_updates: int) -> None:
        reservation = self._reservation(reservation_id)
        if reservation not in self._open(reservation["method"]):
            raise ValueError("segment reservation is already closed")
        expected = reservation["end_update"] - reservation["start_update"]
        if actual_updates != expected:
            raise ValueError("completed segment update count changed")
        event_id = f"{self.task_id}:{reservation_id}:completed"
        row = {
            **reservation,
            "status": "COMPLETED",
            "event_id": event_id,
            "actual_optimizer_updates": actual_updates,
            "actual_optimizer_updates_upper_bound": 0,
            "recorded_unix": time.time(),
        }
        self._append(self.path, row)
        self._append(self.global_path, row)

    def resume_open(self, method: str, run_id: str) -> tuple[int, int]:
        open_rows = self._open(method)
        if len(open_rows) != 1:
            raise ValueError("resume requires exactly one open segment")
        if self.summary(method)["resume_count"] >= 1:
            raise ValueError("method resume allowance is exhausted")
        reservation = open_rows[0]
        upper = reservation["end_update"] - reservation["start_update"]
        event_id = f"{self.task_id}:{reservation['reservation_id']}:lost"
        row = {
            **reservation,
            "run_id": run_id,
            "status": "LOST_UPPER_BOUND",
            "event_id": event_id,
            "actual_optimizer_updates": 0,
            "actual_optimizer_updates_upper_bound": upper,
            "resume_ordinal": 1,
            "recorded_unix": time.time(),
        }
        self._append(self.path, row)
        self._append(self.global_path, row)
        return int(reservation["start_update"]), int(reservation["end_update"])

    def summary(self, method: str) -> dict[str, int]:
        rows = [row for row in self._rows() if row.get("method") == method]
        completed = [row for row in rows if row.get("status") == "COMPLETED"]
        lost = [row for row in rows if row.get("status") == "LOST_UPPER_BOUND"]
        actual = sum(int(row["actual_optimizer_updates"]) for row in completed)
        lost_upper = sum(
            int(row["actual_optimizer_updates_upper_bound"]) for row in lost
        )
        return {
            "logical_update": max(
                (int(row["end_update"]) for row in completed), default=0
            ),
            "actual_optimizer_updates": actual,
            "lost_updates_upper_bound": lost_upper,
            "charged_upper_bound": actual + lost_upper,
            "resume_count": len(lost),
        }


def _bind_source_tree(root: Path) -> tuple[Any, Any, Any, Any]:
    local_package = str(root / "src/cmc_bbdm")
    import cmc_bbdm

    package_path = [
        str(path) for path in cmc_bbdm.__path__ if str(path) != local_package
    ]
    cmc_bbdm.__path__ = [local_package, *package_path]
    from cmc_bbdm.cai_agent_v3 import actor_selection, actor_training
    from cmc_bbdm.cai_agent_v3.feature_bank import load_feature_bank
    from cmc_bbdm.cai_agent_v3.predictor_training import load_predictor_checkpoint

    modules = (actor_training, actor_selection)
    if any(
        not Path(module.__file__).resolve().is_relative_to(root) for module in modules
    ):
        raise RuntimeError("CAI source modules did not bind to the current worktree")
    return actor_training, actor_selection, load_feature_bank, load_predictor_checkpoint


def _input_identity(context: TaskContext) -> tuple[str, dict[str, Any]]:
    vlm_manifest = context.path("vlm") / "vlm_manifest_fit.json"
    vlm_features = context.path("vlm") / "vlm_actor_features_fit.csv"
    w2 = context.path("predictors")
    files = {
        "vlm_manifest": vlm_manifest,
        "vlm_features": vlm_features,
        "predictor_gate": w2 / "predictor_gate.json",
        "oof_readiness": w2 / "oof_readiness.json",
        "oof_fold_manifest": w2 / "oof_fold_manifest.csv",
        "feature_bank_manifest": context.path("data") / "feature_bank_manifest.json",
        "feature_bank_index": context.path("data") / "feature_bank_index.csv",
        "feature_bank_shards": context.path("data") / "feature_bank_shards.csv",
        "split_manifest": context.path("data") / "split_manifest.csv",
        "candidate_queue": context.path("data") / "candidate_queue.csv",
        "cost_precision_audit": context.path("data") / "cost_precision_audit.json",
        "actor_training_source": context.root
        / "src/cmc_bbdm/cai_agent_v3/actor_training.py",
        "models_source": context.root / "src/cmc_bbdm/cai_agent_v3/models.py",
        "policy_source": context.root / "src/cmc_bbdm/cai_agent_v3/policy.py",
        "training_shell": Path(__file__),
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"training input is missing: {missing}")
    vlm = json.loads(vlm_manifest.read_text(encoding="utf-8"))
    if (
        vlm.get("status") != "C_PRIOR_COMPLETE"
        or vlm.get("rows") != 211
        or vlm.get("terminal_rows") != 211
        or vlm.get("test_rows") != 0
        or vlm.get("feature_csv_sha256") != sha256_file(vlm_features)
    ):
        raise ValueError("complete 211-row C prior is required for training")
    shard_rows = []
    with files["feature_bank_shards"].open(newline="", encoding="utf-8") as handle:
        shard_rows = list(csv.DictReader(handle))
    if len(shard_rows) != 6:
        raise ValueError("feature bank must contain the frozen six shards")
    for row in shard_rows:
        shard_path = context.root / row["shard_path"]
        if not shard_path.is_file() or sha256_file(shard_path) != row["sha256"]:
            raise ValueError(f"frozen feature shard changed: {shard_path}")
        files[f"feature_shard_{row['dataset_id']}"] = shard_path
    hashes = {name: sha256_file(path) for name, path in files.items()}
    payload = {
        "task_id": context.task_id,
        "prior_version": context.scope["prior_version"],
        "files": hashes,
        "methods": METHOD_SEEDS,
        "candidate_updates": CANDIDATE_UPDATES,
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest(), payload


def _validate_predictor_bindings(context: TaskContext) -> dict[str, Any]:
    w2 = context.path("predictors")
    gate = json.loads((w2 / "predictor_gate.json").read_text(encoding="utf-8"))
    readiness = json.loads((w2 / "oof_readiness.json").read_text(encoding="utf-8"))
    p_all = next(
        manifest
        for manifest in gate["candidate_manifests"]
        if manifest["model"] == gate["selected_p_all"]
    )
    if p_all["checkpoint_path"] != context.scope["predictors"]["common_checkpoint"]:
        raise ValueError("W2 common predictor selection changed")
    consumed = [p_all, *readiness["fold_manifests"]]
    for manifest in consumed:
        path = context.root / manifest["checkpoint_path"]
        if sha256_file(path) != manifest["checkpoint_sha256"]:
            raise ValueError(f"frozen W2 checkpoint changed: {path}")
    return {
        "p_all": p_all,
        "folds": readiness["fold_manifests"],
        "status": readiness["status"],
    }


def _write_episodes(path: Path, rows: list[dict[str, Any]]) -> None:
    if len(rows) != 50:
        raise ValueError("each C candidate must contain exactly 50 VALID episodes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_episodes(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _candidate_metadata(method_dir: Path, update: int) -> dict[str, Any]:
    path = method_dir / "commits" / f"update_{update:06d}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, digest_name in (
        ("checkpoint", "checkpoint_sha256"),
        ("episodes", "episodes_sha256"),
        ("snapshot", "snapshot_sha256"),
    ):
        target = method_dir / payload[name]
        if not target.is_file() or sha256_file(target) != payload[digest_name]:
            raise ValueError(f"committed candidate artifact changed: {target}")
    return payload


def _run_metadata_path(method_dir: Path) -> Path:
    return method_dir / "run.json"


def _environment_identity(
    context: TaskContext, method: str, input_signature: str
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "task_id": context.task_id,
                "method": method,
                "seed": METHOD_SEEDS[method],
                "input_signature": input_signature,
                "training": context.scope["training"],
                "evaluation": context.scope["evaluation"],
                "torch": torch.__version__,
                "numpy": np.__version__,
            }
        )
    ).hexdigest()


def _load_training_inputs(context: TaskContext, device: str) -> dict[str, Any]:
    training, _selection, load_feature_bank, load_predictor = _bind_source_tree(
        context.root
    )
    predictor_bindings = _validate_predictor_bindings(context)
    bank = load_feature_bank(project_root=context.root)
    if {split: len(bank.indices(split)) for split in ("TRAIN", "VALID", "TEST")} != {
        "TRAIN": 161,
        "VALID": 50,
        "TEST": 65,
    }:
        raise ValueError("feature-bank cohort changed")
    if not np.isnan(bank.targets_mpa[bank.indices("TEST")]).all():
        raise ValueError("TEST targets are not redacted")
    features = training._VLMFeatures(
        bank, context.path("vlm") / "vlm_actor_features_fit.csv"
    )
    p_all, _ = load_predictor(
        context.root / predictor_bindings["p_all"]["checkpoint_path"], device=device
    )
    p_all.eval().requires_grad_(False)
    oof, folds = training._load_oof_predictors(
        context.root,
        device=device,
        predictor_root=context.path("predictors"),
        bank=bank,
        verify=False,
    )
    return {
        "training": training,
        "bank": bank,
        "features": features,
        "p_all": p_all,
        "oof": oof,
        "folds": folds,
        "cell_costs": training._cell_costs(bank),
        "predictor_bindings": predictor_bindings,
    }


def _initial_training_state(
    context: TaskContext,
    method: str,
    method_dir: Path,
    inputs: dict[str, Any],
    input_signature: str,
    environment: str,
    device: str,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, np.random.Generator, dict[str, Any]]:
    training = inputs["training"]
    bank = inputs["bank"]
    train_targets = bank.targets_mpa[bank.indices("TRAIN")]
    target_mean = float(np.mean(train_targets))
    target_scale = max(float(np.std(train_targets)), 1.0)
    actor, rng = training.seeded_actor(
        method,
        target_mean=target_mean,
        target_scale=target_scale,
        seed=METHOD_SEEDS[method],
        device=device,
    )
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(context.scope["training"]["learning_rate"]),
        weight_decay=float(context.scope["training"]["weight_decay"]),
    )
    run = {
        "schema_version": 1,
        "status": "INCOMPLETE",
        "method": method,
        "training_seed": METHOD_SEEDS[method],
        "run_id": uuid.uuid4().hex,
        "input_signature": input_signature,
        "environment_identity": environment,
        "logical_update": 0,
        "resume_count": 0,
        "initial_state_dict_sha256": training._state_dict_sha256(actor.state_dict()),
        "target_mean_mpa": target_mean,
        "target_scale_mpa": target_scale,
    }
    method_dir.mkdir(parents=True, exist_ok=False)
    snapshot = method_dir / "snapshots/update_000000.pt"
    save_snapshot(
        snapshot,
        actor,
        optimizer,
        rng,
        logical_update=0,
        best_score=math.inf,
        best_update=0,
        stale=0,
        progress=[],
        environment_identity=environment,
        input_signature=input_signature,
        run_id=run["run_id"],
    )
    initial = method_dir / "initial_state.pt"
    initial.parent.mkdir(parents=True, exist_ok=True)
    copy_file(snapshot, initial)
    copy_file(snapshot, method_dir / "latest_training_state.pt")
    atomic_json(_run_metadata_path(method_dir), run)
    return (
        actor,
        optimizer,
        rng,
        {
            "logical_update": 0,
            "best_score": math.inf,
            "best_update": 0,
            "stale": 0,
            "progress": [],
            "run": run,
        },
    )


def copy_file(source: Path, destination: Path) -> None:
    import shutil

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _resume_training_state(
    context: TaskContext,
    method: str,
    method_dir: Path,
    inputs: dict[str, Any],
    input_signature: str,
    environment: str,
    device: str,
    ledger: SegmentLedger,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, np.random.Generator, dict[str, Any]]:
    run_path = _run_metadata_path(method_dir)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("status") != "INCOMPLETE" or run.get("method") != method:
        raise ValueError("only an incomplete matching method can resume")
    if run.get("input_signature") != input_signature:
        raise ValueError("training input signature changed")
    if run.get("environment_identity") != environment:
        raise ValueError("training environment changed")
    if int(run.get("resume_count", 0)) >= 1:
        raise ValueError("method resume allowance is exhausted")
    open_segments = ledger._open(method)
    if open_segments:
        ledger.resume_open(method, run["run_id"])
    committed = ledger.summary(method)["logical_update"]
    snapshot = method_dir / f"snapshots/update_{committed:06d}.pt"
    training = inputs["training"]
    actor, rng = training.seeded_actor(
        method,
        target_mean=run["target_mean_mpa"],
        target_scale=run["target_scale_mpa"],
        seed=METHOD_SEEDS[method],
        device=device,
    )
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(context.scope["training"]["learning_rate"]),
        weight_decay=float(context.scope["training"]["weight_decay"]),
    )
    state = restore_snapshot(
        snapshot,
        actor,
        optimizer,
        rng,
        expected_environment=environment,
        expected_signature=input_signature,
    )
    if state["logical_update"] != committed:
        raise ValueError("snapshot logical update differs from segment ledger")
    run["resume_count"] = int(run.get("resume_count", 0)) + 1
    run["logical_update"] = committed
    atomic_json(run_path, run)
    return actor, optimizer, rng, {**state, "run": run}


def _record_candidate(
    method_dir: Path,
    method: str,
    update: int,
    actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    episodes: list[dict[str, Any]],
    score: float,
    state: dict[str, Any],
    environment: str,
    input_signature: str,
) -> dict[str, Any]:
    checkpoint = method_dir / f"candidates/update_{update:06d}.pt"
    episode_path = method_dir / f"candidate_episodes/update_{update:06d}.csv.gz"
    snapshot = method_dir / f"snapshots/update_{update:06d}.pt"
    save_weight_state(
        checkpoint,
        actor,
        method=method,
        update=update,
        environment_identity=environment,
        input_signature=input_signature,
    )
    _write_episodes(episode_path, episodes)
    save_snapshot(
        snapshot,
        actor,
        optimizer,
        rng,
        logical_update=update,
        best_score=state["best_score"],
        best_update=state["best_update"],
        stale=state["stale"],
        progress=state["progress"],
        environment_identity=environment,
        input_signature=input_signature,
        run_id=state["run"]["run_id"],
    )
    record = {
        "method": method,
        "update": update,
        "score": score,
        "checkpoint": checkpoint.relative_to(method_dir).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "episodes": episode_path.relative_to(method_dir).as_posix(),
        "episodes_sha256": sha256_file(episode_path),
        "episode_count": 50,
        "snapshot": snapshot.relative_to(method_dir).as_posix(),
        "snapshot_sha256": sha256_file(snapshot),
        "environment_identity": environment,
        "input_signature": input_signature,
        "commit_status": "COMPLETE",
    }
    atomic_json(method_dir / "commits" / f"update_{update:06d}.json", record)
    return record


def _release_smoke(
    context: TaskContext,
    method: str,
    selected: dict[str, Any],
    method_dir: Path,
    inputs: dict[str, Any],
    environment: str,
    input_signature: str,
    device: str,
) -> dict[str, Any]:
    training = inputs["training"]
    bank = inputs["bank"]
    actor, _rng = training.seeded_actor(
        method,
        target_mean=float(np.mean(bank.targets_mpa[bank.indices("TRAIN")])),
        target_scale=max(float(np.std(bank.targets_mpa[bank.indices("TRAIN")])), 1.0),
        seed=METHOD_SEEDS[method],
        device=device,
    )
    selected_checkpoint = method_dir / selected["checkpoint"]
    load_weight_state(
        selected_checkpoint,
        actor,
        expected_method=method,
        expected_environment=environment,
        expected_signature=input_signature,
    )
    actor.eval()
    key = context.scope["evaluation"]["release_smoke_case"]
    index = inputs["bank"].specimen_keys.index(key)
    actual = training._evaluate_one(
        actor,
        method,
        inputs["p_all"],
        bank,
        inputs["features"],
        inputs["cell_costs"],
        index,
        device=device,
    )
    original_rows = _read_episodes(method_dir / selected["episodes"])
    original = next(row for row in original_rows if row["specimen_key"] == key)
    expected_cells = tuple(
        int(value) for value in original["cells"].split(";") if value
    )
    expected_costs = tuple(float(value) for value in original["costs"].split(";"))
    expected_predictions = tuple(
        float(value) for value in original["predictions_mpa"].split(";")
    )
    if actual[1] != expected_cells:
        raise ValueError("disk-reloaded winner changed release-smoke actions")
    if not np.allclose(actual[2], expected_costs, rtol=0, atol=1e-12):
        raise ValueError("disk-reloaded winner changed release-smoke costs")
    if not np.allclose(actual[3], expected_predictions, rtol=0, atol=1e-4):
        raise ValueError("disk-reloaded winner changed release-smoke predictions")
    smoke = {
        "status": "PASS",
        "method": method,
        "specimen_key": key,
        "selected_update": selected["update"],
        "actions_match": True,
        "maximum_cost_difference": float(
            np.max(np.abs(np.asarray(actual[2]) - np.asarray(expected_costs)))
        ),
        "maximum_prediction_difference_mpa": float(
            np.max(np.abs(np.asarray(actual[3]) - np.asarray(expected_predictions)))
        ),
    }
    atomic_json(method_dir / "release_smoke.json", smoke)
    return smoke


def _complete_method_manifest(
    context: TaskContext,
    method: str,
    method_dir: Path,
    inputs: dict[str, Any],
    state: dict[str, Any],
    environment: str,
    input_signature: str,
    ledger: SegmentLedger,
    device: str,
) -> dict[str, Any]:
    candidates = [
        _candidate_metadata(method_dir, update) for update in CANDIDATE_UPDATES
    ]
    selected = select_candidate(
        [
            Candidate(
                update=int(row["update"]),
                score=float(row["score"]),
                checkpoint=row["checkpoint"],
                episodes=row["episodes"],
            )
            for row in candidates
        ]
    )
    selected_row = next(row for row in candidates if row["update"] == selected.update)
    selected_checkpoint = (
        context.path("w3") / "models/selected" / f"actor_{method.lower()}_seed1.pt"
    )
    selected_episodes = (
        context.path("w3") / "selected_episodes" / f"{method.lower()}_seed1.csv.gz"
    )
    copy_file(method_dir / selected_row["checkpoint"], selected_checkpoint)
    copy_file(method_dir / selected_row["episodes"], selected_episodes)
    smoke = _release_smoke(
        context,
        method,
        selected_row,
        method_dir,
        inputs,
        environment,
        input_signature,
        device,
    )
    summary = ledger.summary(method)
    if summary["logical_update"] != 1250 or summary["charged_upper_bound"] > 1500:
        raise ValueError("method training budget or logical endpoint is invalid")
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "method": method,
        "prior_version": context.scope["prior_version"],
        "training_seed": METHOD_SEEDS[method],
        "run_id": state["run"]["run_id"],
        "input_signature": input_signature,
        "environment_identity": environment,
        "logical_updates": 1250,
        **summary,
        "candidate_count": 5,
        "candidate_episode_count": 250,
        "candidates": candidates,
        "selected_update": selected.update,
        "selected_score": selected.score,
        "selected_checkpoint": selected_checkpoint.relative_to(context.root).as_posix(),
        "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
        "selected_episodes": selected_episodes.relative_to(context.root).as_posix(),
        "selected_episodes_sha256": sha256_file(selected_episodes),
        "release_smoke": smoke,
        "initial_state_dict_sha256": state["run"]["initial_state_dict_sha256"],
        "first_batch_specimen_keys": state["run"]["first_batch_specimen_keys"],
        "parameter_count": sum(
            tensor.numel()
            for tensor in torch.load(
                selected_checkpoint, map_location="cpu", weights_only=False
            )["state_dict"].values()
        ),
        "progress": state["progress"],
    }
    atomic_json(method_dir / "manifest.json", manifest)
    run = dict(state["run"])
    run.update(status="COMPLETE", logical_update=1250, selected_update=selected.update)
    atomic_json(_run_metadata_path(method_dir), run)
    return manifest


def train_method(
    context: TaskContext,
    method: str,
    *,
    resume: bool,
    inputs: dict[str, Any],
    input_signature: str,
    device: str,
) -> dict[str, Any]:
    if method not in METHOD_SEEDS:
        raise ValueError(f"unauthorized C actor method: {method}")
    method_dir = context.path("w3") / "models" / method.lower()
    manifest_path = method_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "COMPLETE"
            or manifest.get("input_signature") != input_signature
            or manifest.get("method") != method
        ):
            raise ValueError("completed method manifest changed")
        for update in CANDIDATE_UPDATES:
            _candidate_metadata(method_dir, update)
        if (
            sha256_file(context.root / manifest["selected_checkpoint"])
            != manifest["selected_checkpoint_sha256"]
        ):
            raise ValueError("selected actor checkpoint changed")
        return manifest

    environment = _environment_identity(context, method, input_signature)
    ledger = SegmentLedger(
        context.path("w3") / "segment_ledger.jsonl",
        context.path("global_ledger"),
        task_id=context.task_id,
        method_cap=1500,
    )
    if method_dir.exists():
        if not resume:
            raise ValueError(f"incomplete {method} requires --resume")
        actor, optimizer, rng, state = _resume_training_state(
            context,
            method,
            method_dir,
            inputs,
            input_signature,
            environment,
            device,
            ledger,
        )
    else:
        if resume:
            raise ValueError(f"no incomplete {method} state exists to resume")
        actor, optimizer, rng, state = _initial_training_state(
            context,
            method,
            method_dir,
            inputs,
            input_signature,
            environment,
            device,
        )

    training = inputs["training"]
    bank = inputs["bank"]
    completed = ledger.summary(method)["logical_update"]
    if int(state["logical_update"]) != completed:
        raise ValueError("restored logical update differs from committed ledger")
    for target_update in CANDIDATE_UPDATES:
        if target_update <= completed:
            continue
        reservation = ledger.reserve(
            method, completed, target_update, state["run"]["run_id"]
        )
        last_terms: dict[str, float] = {}
        loss_value = math.nan
        for logical_update in range(completed + 1, target_update + 1):
            indices = training._sample_specimens(rng, bank, batch_size=16)
            if logical_update == 1 and "first_batch_specimen_keys" not in state["run"]:
                state["run"]["first_batch_specimen_keys"] = [
                    bank.specimen_keys[int(index)] for index in indices
                ]
                atomic_json(_run_metadata_path(method_dir), state["run"])
            entropy_weight = 0.01 * (1.0 - (logical_update - 1) / 1249.0)
            loss, last_terms = training._training_rollout_loss(
                actor,
                method,
                bank,
                inputs["features"],
                inputs["oof"],
                inputs["folds"],
                inputs["cell_costs"],
                indices,
                device=device,
                entropy_weight=entropy_weight,
                target_scale=state["run"]["target_scale_mpa"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            loss_value = float(loss.detach())

        actor.eval()
        actor_sha = training._state_dict_sha256(actor.state_dict())
        score, episodes = training.evaluate_policy(
            actor,
            method,
            inputs["p_all"],
            bank,
            inputs["features"],
            inputs["cell_costs"],
            device=device,
            execution_identity={
                "prior_version": context.scope["prior_version"],
                "seed_panel": 1,
                "training_seed": METHOD_SEEDS[method],
                "checkpoint_update": target_update,
                "actor_state_dict_sha256": actor_sha,
                "evaluation_environment_identity": environment,
                "execution_code_sha": sha256_file(Path(__file__)),
            },
        )
        actor.train()
        if score < float(state["best_score"]) - IMPROVEMENT_TOLERANCE:
            state["best_score"] = score
            state["best_update"] = target_update
            state["stale"] = 0
        else:
            state["stale"] = int(state["stale"]) + 1
        state["logical_update"] = target_update
        state["progress"].append(
            {
                "method": method,
                "training_seed": METHOD_SEEDS[method],
                "update": target_update,
                "validation_area_mpa": score,
                "total_loss": loss_value,
                "entropy_weight": 0.01 * (1.0 - (target_update - 1) / 1249.0),
                **last_terms,
            }
        )
        record = _record_candidate(
            method_dir,
            method,
            target_update,
            actor,
            optimizer,
            rng,
            episodes,
            score,
            state,
            environment,
            input_signature,
        )
        if abs(float(record["score"]) - score) > IMPROVEMENT_TOLERANCE:
            raise ValueError("stored candidate score differs from evaluation")
        ledger.complete(reservation, actual_updates=250)
        copy_file(
            method_dir / record["snapshot"],
            method_dir / "latest_training_state.pt",
        )
        completed = target_update
        state["run"]["logical_update"] = completed
        atomic_json(_run_metadata_path(method_dir), state["run"])
        context.transition(
            "train",
            "RUNNING",
            method=method,
            logical_update=completed,
            actual_update_upper_bound=ledger.summary(method)["charged_upper_bound"],
        )

    return _complete_method_manifest(
        context,
        method,
        method_dir,
        inputs,
        state,
        environment,
        input_signature,
        ledger,
        device,
    )


def _aggregate_manifests(context: TaskContext) -> dict[str, Any]:
    manifests = []
    selected_rows: list[dict[str, str]] = []
    for method in METHOD_SEEDS:
        path = context.path("w3") / "models" / method.lower() / "manifest.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        selected_rows.extend(
            _read_episodes(context.root / manifest["selected_episodes"])
        )
    logical = sum(int(manifest["logical_updates"]) for manifest in manifests)
    charged = sum(int(manifest["charged_upper_bound"]) for manifest in manifests)
    payload = {
        "schema_version": 1,
        "status": "THREE_C_ACTORS_COMPLETE" if len(manifests) == 3 else "PARTIAL",
        "prior_version": context.scope["prior_version"],
        "method_count": len(manifests),
        "candidate_count": sum(manifest["candidate_count"] for manifest in manifests),
        "candidate_episode_count": sum(
            manifest["candidate_episode_count"] for manifest in manifests
        ),
        "selected_episode_count": len(selected_rows),
        "logical_updates": logical,
        "actual_update_upper_bound": charged,
        "actor_manifests": manifests,
    }
    if logical > 3750 or charged > 4500:
        raise RuntimeError("C actor task update authorization exceeded")
    atomic_json(context.path("w3") / "actor_manifests.json", payload)
    if selected_rows:
        destination = context.path("w3") / "new_selected_policy_episodes.csv.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        fields = list(dict.fromkeys(key for row in selected_rows for key in row))
        with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected_rows)
        temporary.replace(destination)
        payload["selected_episodes"] = destination.relative_to(context.root).as_posix()
        payload["selected_episodes_sha256"] = sha256_file(destination)
        atomic_json(context.path("w3") / "actor_manifests.json", payload)
    return payload


def train_stage(
    context: TaskContext, *, resume: bool = False, method: str | None = None
) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("exactly one CUDA_VISIBLE_DEVICES entry is required")
    if not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")
    torch.set_num_threads(4)
    input_signature, input_payload = _input_identity(context)
    atomic_json(context.path("w3") / "input_bindings.json", input_payload)
    resource_path = context.path("output") / "resource_usage.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if float(resource.get("actor_gpu_seconds", 0.0)) >= 10800:
        raise RuntimeError("Actor GPU session budget is exhausted")
    methods = [method] if method else list(METHOD_SEEDS)
    started = time.monotonic()
    context.append_resource_event(
        {
            "stage": "train",
            "status": "GPU_SESSION_STARTED",
            "physical_gpu": visible,
            "methods": methods,
            "resume": resume,
            "recorded_unix": time.time(),
        }
    )
    outcome = "FAILED"
    try:
        inputs = _load_training_inputs(context, "cuda:0")
        for selected_method in methods:
            method_dir = context.path("w3") / "models" / selected_method.lower()
            train_method(
                context,
                selected_method,
                resume=_resume_mode(
                    resume,
                    explicit_method=method is not None,
                    method_dir=method_dir,
                ),
                inputs=inputs,
                input_signature=input_signature,
                device="cuda:0",
            )
        payload = _aggregate_manifests(context)
        outcome = payload["status"]
    finally:
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        resource["actor_gpu_seconds"] = (
            float(resource.get("actor_gpu_seconds", 0.0)) + elapsed
        )
        aggregated = _aggregate_manifests(context)
        resource["new_actor_updates"] = aggregated["logical_updates"]
        resource["new_actor_update_upper_bound"] = aggregated[
            "actual_update_upper_bound"
        ]
        resource["status"] = outcome
        atomic_json(resource_path, resource)
        context.append_resource_event(
            {
                "stage": "train",
                "status": "GPU_SESSION_ENDED",
                "physical_gpu": visible,
                "elapsed_gpu_seconds": elapsed,
                "outcome": outcome,
                "recorded_unix": time.time(),
            }
        )
    phase_status = (
        "COMPLETE" if payload["status"] == "THREE_C_ACTORS_COMPLETE" else "PARTIAL"
    )
    context.transition(
        "train",
        phase_status,
        method_count=payload["method_count"],
        candidate_count=payload["candidate_count"],
        selected_episode_count=payload["selected_episode_count"],
        logical_updates=payload["logical_updates"],
        actual_update_upper_bound=payload["actual_update_upper_bound"],
    )
    return payload


def _resume_mode(requested: bool, *, explicit_method: bool, method_dir: Path) -> bool:
    if requested and explicit_method and not method_dir.exists():
        raise ValueError("no incomplete method state exists to resume")
    return requested and method_dir.exists()
