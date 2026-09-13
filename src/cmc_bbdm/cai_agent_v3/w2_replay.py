"""W2-only replay scope; shares the registered predictor training loop."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from . import predictor_training as training
from .checkpoint_selection import inspect_archive, validation_identity
from .feature_bank import load_feature_bank
from .files import read_csv, sha256_file, write_json

REPLAY_ID = "W2_EXACT_COST_REPLAY_R1_292b1c74"
RUN = "results/cai_agent_v3/w2_replay/r1_292b1c74"
ART = "artifacts/cai_agent_v3/w2_replay/r1_292b1c74"
AUTH = "docs/cai/w2_formal_replay/W2_REPLAY_AUTHORIZATION.json"


def update_usage(rows):
    total = sum(
        r.get("actual_optimizer_updates", 0)
        or r.get("actual_optimizer_updates_upper_bound", 0)
        for r in rows
    )
    done = {r.get("run_id") for r in rows if r.get("status") == "COMPLETED"}
    reservations = {
        r["run_id"]: r["optimizer_update_reservation"]
        for r in rows
        if r.get("run_id") and "optimizer_update_reservation" in r
    }
    return total + sum(v for k, v in reservations.items() if k not in done)


def check_allocation(*, global_used, replay_used, stage_used, required):
    if (
        global_used + required > 34264
        or replay_used + required > 12000
        or stage_used > 6000
    ):
        raise ValueError("RESOURCE_LIMITED: W2-only update allocation exceeded")


def resolve_output(root, relative):
    path = (root / relative).resolve()
    allowed = (root / "results/cai_agent_v3/w2_replay").resolve()
    if not path.is_relative_to(allowed) or path == allowed:
        raise ValueError("replay output must be isolated from historical results")
    return path


def _rows(root):
    return [
        json.loads(line)
        for line in (root / "results/cai_agent_v3/compute_ledger.jsonl")
        .read_text()
        .splitlines()
    ]


def _atomic_torch(path, payload):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def prepare_run(root):
    output = resolve_output(root, RUN)
    art = root / ART
    authorization = json.loads((root / AUTH).read_text())
    assert authorization["replay_id"] == REPLAY_ID
    assert authorization["budget"]["new_cumulative_cap"] == 34264
    assert authorization["budget"]["new_replay_updates_cap"] == 12000
    if (output / "input_reuse_manifest.json").exists():
        existing = json.loads((output / "replay_authorization.json").read_text())
        if existing["replay_id"] != REPLAY_ID:
            raise ValueError("existing replay identity mismatch")
        return json.loads((output / "input_reuse_manifest.json").read_text())
    check_allocation(
        global_used=update_usage(_rows(root)),
        replay_used=0,
        stage_used=0,
        required=12000,
    )
    source = root / "results/cai_agent_v3/new_protocol"
    bank = load_feature_bank(project_root=root)
    counts = {split: len(bank.indices(split)) for split in ("TRAIN", "VALID", "TEST")}
    groups = {
        split: len({bank.capture_group_ids[int(i)] for i in bank.indices(split)})
        for split in counts
    }
    if counts != {"TRAIN": 161, "VALID": 50, "TEST": 65} or groups != {
        "TRAIN": 152,
        "VALID": 48,
        "TEST": 59,
    }:
        raise ValueError("frozen cohort/split membership counts changed")
    library = training.build_validation_library(bank, training._cell_costs(bank))
    with np.load(source / "validation_prefix_library.npz") as stored:
        for name in (
            "specimen_indices",
            "state_indices",
            "masks",
            "costs",
            "route_names",
        ):
            if not np.array_equal(stored[name], getattr(library, name)):
                raise ValueError(f"exact VALID library differs: {name}")
    fold_rows = read_csv(source / "oof_fold_manifest.csv")
    expected = {
        bank.specimen_keys[i]: f for i, f in training._oof_assignments(bank).items()
    }
    if {r["specimen_key"]: int(r["fold"]) for r in fold_rows} != expected:
        raise ValueError("frozen OOF assignment changed")
    output.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    # Small metadata identity only; no repeated feature-shard or image hashing.
    paths = [
        "feature_bank_index.csv",
        "split_manifest.csv",
        "feature_bank_manifest.json",
        "oof_fold_manifest.csv",
        "validation_prefix_library.npz",
        "ridge_diagnostics.csv",
    ]
    reuse = {
        "status": "FIXED_INPUTS_BOUND",
        "source_root": source.relative_to(root).as_posix(),
        "physical_n": counts,
        "capture_group_n": groups,
        "validation_identity": validation_identity(bank, library),
        "source_metadata_sha256": {n: sha256_file(source / n) for n in paths},
        "feature_shards": json.loads(
            (source / "feature_bank_manifest.json").read_text()
        )["shards"],
        "test_targets_redacted": True,
        "test_predictions": 0,
        "constants": training._constant_metrics(bank),
        "ridge_policy": "Reuse full-input diagnostics; partial legacy result not used for new readiness or claimed exact-cost comparison.",
    }
    shutil.copyfile(
        source / "validation_prefix_library.npz",
        output / "validation_prefix_library.npz",
    )
    authorization["execution_binding"] = {
        "task_source": "User assigned CAI_V3_W2_FORMAL_REPLAY_CODEX_PACKAGE.zip",
        "code_sha_at_prepare": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "ledger_start_lines": len(_rows(root)),
        "ledger_start_bytes": (root / "results/cai_agent_v3/compute_ledger.jsonl")
        .stat()
        .st_size,
        "actual_starting_upper_bound": update_usage(_rows(root)),
        "arithmetic_correction": "28100+6164=34264; old handoff 34464 retained as historical error",
    }
    write_json(output / "replay_authorization.json", authorization)
    write_json(output / "input_reuse_manifest.json", reuse)
    return reuse


class ReplayContext:
    def __init__(self, root, stage):
        self.root = root
        self.output = resolve_output(root, RUN)
        self.art = root / ART
        self.stage = stage
        self.authorization = json.loads(
            (self.output / "replay_authorization.json").read_text()
        )
        if (
            self.authorization["replay_id"] != REPLAY_ID
            or self.authorization["budget"]["new_cumulative_cap"] != 34264
        ):
            raise ValueError("replay authorization does not match bound W2 task")
        self.binding = json.loads(
            (self.output / "input_reuse_manifest.json").read_text()
        )["validation_identity"]
        self.session_id = uuid4().hex
        self.session_start = None
        self.job_ids = {}
        self.job_started = {}
        self.last_updates = {}

    def append(self, **row):
        training._append_ledger(
            self.root / "results/cai_agent_v3/compute_ledger.jsonl",
            dict(replay_id=REPLAY_ID, replay_stage=self.stage, **row),
        )

    def local_rows(self):
        return [r for r in _rows(self.root) if r.get("replay_id") == REPLAY_ID]

    def begin_stage(self, stage):
        if stage != self.stage:
            raise ValueError("wrong replay stage")
        review = json.loads((self.art / "preflight_review.json").read_text())
        if review["status"] != "PREFLIGHT_PASS":
            raise ValueError("preflight review not passed")
        jobs = (
            ["MEAN_SC", "SPATIAL_SC", "SPATIAL_C"]
            if stage == "A"
            else ["fold0", "fold1", "fold2"]
        )
        completed = sum(self.job_file(job).exists() for job in jobs)
        required = 2000 * (3 - completed)
        if stage == "A":
            required += 6000
        local = self.local_rows()
        stage_used = update_usage([r for r in local if r.get("replay_stage") == stage])
        if stage_used + 2000 * (3 - completed) > 6000:
            raise ValueError("RESOURCE_LIMITED: stage update cap")
        check_allocation(
            global_used=update_usage(_rows(self.root)),
            replay_used=update_usage(local),
            stage_used=stage_used,
            required=required,
        )
        self.check_time()

    def start_session(self, device):
        if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise ValueError(
                "this bound replay uses physical GPU1 only (logical cuda:0)"
            )
        self.check_time()
        self.session_start = time.time()
        self.append(
            job="W2_GPU_SESSION",
            session_id=self.session_id,
            status="GPU_SESSION_STARTED",
            started_epoch=self.session_start,
            device=device,
            physical_gpu=1,
            execution_code_sha=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
            ).strip(),
            actual_optimizer_updates=0,
        )

    def finish_session(self, status):
        if self.session_start is None:
            return
        torch.cuda.synchronize()
        self.append(
            job="W2_GPU_SESSION",
            session_id=self.session_id,
            status="GPU_SESSION_ENDED",
            outcome=status,
            elapsed_gpu_seconds=time.time() - self.session_start,
            actual_optimizer_updates=0,
        )

    def check_time(self):
        rows = self.local_rows()
        ends = {
            r["session_id"]: r for r in rows if r.get("status") == "GPU_SESSION_ENDED"
        }
        used = sum(r["elapsed_gpu_seconds"] for r in ends.values())
        for row in rows:
            if (
                row.get("status") == "GPU_SESSION_STARTED"
                and row["session_id"] not in ends
            ):
                used += time.time() - row["started_epoch"]
        if used >= 21600:
            raise ValueError("RESOURCE_LIMITED: new six-GPU-hour window exhausted")

    def job_file(self, key):
        return self.output / "jobs" / f"{self.stage}_{key}.json"

    def cached_job(self, key, *, device):
        path = self.job_file(key)
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        rows = self.local_rows()
        if not any(
            r.get("run_id") == saved["run_id"] and r.get("status") == "COMPLETED"
            for r in rows
        ):
            raise ValueError("incomplete job settlement; do not restart automatically")
        manifest = saved["manifest"]
        archive = self.root / manifest["selection_archive"]
        inspect_archive(archive, validation=self.binding)
        model, _ = training.load_predictor_checkpoint(
            archive / f"update_{manifest['selected_update']:06d}.pt", device=device
        )
        return model, manifest, saved["progress"]

    def start_job(self, key):
        if any(
            r.get("replay_job_key") == key and r.get("replay_stage") == self.stage
            for r in self.local_rows()
        ):
            raise ValueError(
                "existing incomplete job retained; automatic restart not permitted"
            )
        self.check_time()
        run_id = uuid4().hex
        self.job_ids[key] = run_id
        self.job_started[key] = time.time()
        self.append(
            job=f"W2_{self.stage}_{key}",
            replay_job_key=key,
            run_id=run_id,
            status="STARTED",
            actual_optimizer_updates=0,
            optimizer_update_reservation=2000,
            started_epoch=time.time(),
        )

    def before_update(self, key, update):
        self.last_updates[key] = update - 1
        if update > 2000:
            raise ValueError("RESOURCE_LIMITED: per-model update cap")
        self.check_time()

    def save_details(self, key, update, details):
        directory = self.output / "candidate_state_predictions" / f"{self.stage}_{key}"
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"update_{update:06d}.npz", **details)

    def checkpoint(
        self, key, archive, model, optimizer, rng, update, best_update, best_area, stale
    ):
        self.last_updates[key] = update
        if archive.manifest["validation"] != self.binding:
            raise ValueError("fixed VALID identity changed")
        _atomic_torch(
            archive.directory / "latest_training_state.pt",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "numpy_generator_state": rng.bit_generator.state,
                "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all()
                if next(model.parameters()).is_cuda
                else [],
                "update": update,
                "best_update": best_update,
                "best_area": best_area,
                "stale": stale,
                "run_id": self.job_ids[key],
                "validation_identity": self.binding,
                "best_checkpoint": f"update_{best_update:06d}.pt",
            },
        )
        if update == 250:
            row = archive.manifest["candidates"][0]
            if (
                archive.manifest["status"] != "INCOMPLETE"
                or row["update"] != 250
                or not (archive.directory / row["checkpoint"]).is_file()
            ):
                raise ValueError("first real checkpoint evidence missing")
            saved = torch.load(
                archive.directory / row["checkpoint"],
                map_location="cpu",
                weights_only=False,
            )
            if (
                saved["manifest"]["update"] != 250
                or saved["manifest"]["validation"] != self.binding
                or training._state_dict_sha256(saved["state_dict"])
                != training._state_dict_sha256(model.state_dict())
                or not any(
                    r.get("run_id") == self.job_ids[key]
                    and r.get("status") == "STARTED"
                    for r in self.local_rows()
                )
            ):
                raise ValueError("first checkpoint payload/ledger binding failed")
            write_json(
                self.art / f"first_checkpoint_{self.stage}_{key}.json",
                {
                    "status": "FIRST_250_ARCHIVE_CONFIRMED",
                    "archive_status": "INCOMPLETE",
                    "run_id": self.job_ids[key],
                    "checkpoint": (archive.directory / row["checkpoint"])
                    .relative_to(self.root)
                    .as_posix(),
                    "sha256": row["sha256"],
                    "validation_identity": self.binding["identity"],
                    "metrics": row["metrics"],
                },
            )
        self.append(
            job=f"W2_{self.stage}_{key}",
            run_id=self.job_ids[key],
            status="PROGRESS",
            observed_update=update,
            actual_optimizer_updates=0,
        )
        print(
            json.dumps(
                {
                    "stage": self.stage,
                    "model": key,
                    "update": update,
                    "A": archive.manifest["candidates"][-1]["metrics"][
                        "valid_area_mpa"
                    ],
                }
            ),
            flush=True,
        )

    def complete_job(self, key, manifest, progress):
        saved_manifest = {
            **manifest,
            "selection_archive": Path(manifest["selection_archive"])
            .relative_to(self.root)
            .as_posix(),
        }
        self.job_file(key).parent.mkdir(exist_ok=True)
        write_json(
            self.job_file(key),
            {
                "run_id": self.job_ids[key],
                "manifest": saved_manifest,
                "progress": progress,
            },
        )
        self.append(
            job=f"W2_{self.stage}_{key}",
            replay_job_key=key,
            run_id=self.job_ids[key],
            status="COMPLETED",
            actual_optimizer_updates=manifest["updates_completed"],
            elapsed_seconds=time.time() - self.job_started[key],
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "command", choices=["prepare-run", "train-candidates", "train-oof", "summarize"]
    )
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    torch.set_num_threads(4)
    if args.command == "prepare-run":
        result = prepare_run(root)
    elif args.command == "summarize":
        from .w2_replay_results import summarize

        result = summarize(root)
    else:
        stage = "A" if args.command == "train-candidates" else "B"
        context = ReplayContext(root, stage)
        gate = context.output / (
            "predictor_gate.json" if stage == "A" else "oof_readiness.json"
        )
        if gate.exists():
            # Completed stage is read-only reusable, never silently repeated.
            bank = load_feature_bank(project_root=root)
            payload = json.loads(gate.read_text())
            manifests = payload.get(
                "candidate_manifests" if stage == "A" else "fold_manifests", []
            )
            if payload.get("status", "").startswith("NOT_EXECUTED"):
                result = payload
            else:
                library = training.build_validation_library(
                    bank, training._cell_costs(bank)
                )
                if len(manifests) != 3 or not all(
                    training._manifest_selection_verified(
                        root, m, bank, library, validation=context.binding
                    )
                    for m in manifests
                ):
                    raise ValueError(
                        "completed stage lacks complete selection evidence"
                    )
                result = payload
        else:
            context.start_session("cuda:0")
            status = "FAILED"
            try:
                function = (
                    training.run_predictor_candidates
                    if stage == "A"
                    else training.run_oof_reward_predictors
                )
                result = function(
                    project_root=root, device="cuda:0", run_context=context
                )
                status = "COMPLETED"
            finally:
                context.finish_session(status)
    print(json.dumps(result, indent=2, default=str))
