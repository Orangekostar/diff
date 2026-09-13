"""Explicit W3-only orchestration; all optimization uses the original Actor loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from . import actor_training as t
from .actor_selection import (
    ActorArchive,
    inspect_actor_archive,
    read_episodes,
    save_torch,
    write_atomic,
    write_episodes,
)
from .feature_bank import load_feature_bank
from .files import sha256_file, write_csv
from .predictor_training import (
    load_predictor_checkpoint,
    require_predictor_selection_evidence,
)
from .w2_replay import update_usage

TASK_ID = "W3_VALID_PILOT_R1_0e11452a"
DATA = "results/cai_agent_v3/new_protocol"
W2 = "results/cai_agent_v3/w2_replay/r1_292b1c74"
RUN = "results/cai_agent_v3/w3_pilot/r1_0e11452a"
ART = "artifacts/cai_agent_v3/w3_pilot/r1_0e11452a"
AUTH = "docs/cai/w3_valid_pilot/W3_PILOT_AUTHORIZATION.json"
LEDGER = "results/cai_agent_v3/compute_ledger.jsonl"


def check_budget(*, global_used, task_used, method_used, required, method_cap):
    if (
        global_used + required > 40014
        or task_used + required > 5750
        or method_used + required > method_cap
    ):
        raise ValueError(
            "RESOURCE_LIMITED: W3 task/method/global cap; W2 remainder not transferable"
        )


def rows(root):
    return [json.loads(line) for line in (root / LEDGER).read_text().splitlines()]


def sha_payload(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def source_root(root, relative):
    path = (root / relative).resolve()
    if path != (root / W2).resolve():
        raise ValueError("W3 must consume the explicitly bound valid W2 replay")
    return path


def prepare(root):
    output = root / RUN
    if (output / "protocol_snapshot.json").exists():
        return json.loads((output / "protocol_snapshot.json").read_text())
    auth = json.loads((root / AUTH).read_text())
    assert auth["task_id"] == TASK_ID and auth["budget"]["new_global_cap"] == 40014
    check_budget(
        global_used=update_usage(rows(root)),
        task_used=0,
        method_used=0,
        required=5750,
        method_cap=5750,
    )
    bank = load_feature_bank(project_root=root)
    counts = {s: len(bank.indices(s)) for s in ("TRAIN", "VALID", "TEST")}
    assert counts == {"TRAIN": 161, "VALID": 50, "TEST": 65}
    assert np.isnan(bank.targets_mpa[bank.indices("TEST")]).all()
    w2 = source_root(root, W2)
    require_predictor_selection_evidence(root, bank, include_oof=True, output_dir=w2)
    reuse = json.loads((w2 / "input_reuse_manifest.json").read_text())
    for name in (
        "feature_bank_index.csv",
        "split_manifest.csv",
        "feature_bank_manifest.json",
    ):
        assert sha256_file(root / DATA / name) == reuse["source_metadata_sha256"][name]
    gate = json.loads((w2 / "predictor_gate.json").read_text())
    oof = json.loads((w2 / "oof_readiness.json").read_text())
    p_all = next(
        m for m in gate["candidate_manifests"] if m["model"] == gate["selected_p_all"]
    )
    consumed = [p_all, *oof["fold_manifests"]]
    for m in consumed:
        assert sha256_file(root / m["checkpoint_path"]) == m["checkpoint_sha256"]
    vlm = json.loads((root / DATA / "vlm_manifest_fit.json").read_text())
    assert vlm["status"] == "REAL_FROZEN_VLM_PERCEPTION_COMPLETE"
    assert (
        vlm["specimen_count"],
        vlm["vlm_available_count"],
        vlm["vlm_unavailable_count"],
    ) == (211, 205, 6)
    assert (
        sha256_file(root / vlm["actor_features_path"]) == vlm["actor_features_sha256"]
    )
    assert sha256_file(root / vlm["cache_path"]) == vlm["cache_sha256"]
    features = t._VLMFeatures(bank, root / DATA / "vlm_actor_features_fit.csv")
    assert sum(features.available) == 205
    output.mkdir(parents=True, exist_ok=True)
    (root / ART).mkdir(parents=True, exist_ok=True)
    inputs = {
        "data_root": DATA,
        "predictor_root": W2,
        "physical_n": counts,
        "capture_groups": {
            s: len({bank.capture_group_ids[i] for i in bank.indices(s)}) for s in counts
        },
        "predictors": [
            {
                k: m[k]
                for k in (
                    "model",
                    "selected_update",
                    "checkpoint_path",
                    "checkpoint_sha256",
                )
            }
            for m in consumed
        ],
        "w2_input_identity": sha256_file(w2 / "input_reuse_manifest.json"),
        "w2_fold_identity": sha256_file(w2 / "oof_fold_manifest.csv"),
        "validation_identity": reuse["validation_identity"]["identity"],
        "feature_shards": reuse["feature_shards"],
        "data_metadata": reuse["source_metadata_sha256"],
        "vlm_manifest_sha256": sha256_file(root / DATA / "vlm_manifest_fit.json"),
        "vlm_features_sha256": vlm["actor_features_sha256"],
        "vlm_cache_sha256": vlm["cache_sha256"],
        "vlm_unavailable_specimen_keys": vlm["vlm_unavailable_specimen_keys"],
        "upstream_selection_evidence": "VERIFIED_ONCE_NO_MODEL_FORWARD",
    }
    write_atomic(root / ART / "INPUT_BINDINGS.json", inputs)
    code_paths = [
        "actor_training.py",
        "actor_selection.py",
        "models.py",
        "policy.py",
        "metrics.py",
        "w3_pilot.py",
    ]
    protocol = {
        "task_id": TASK_ID,
        "inputs_identity": sha_payload(inputs),
        "geometry": auth["geometry"],
        "training": auth["training"],
        "evaluation": auth["evaluation"],
        "models": auth["models"],
        "fixed_methods": auth["fixed_methods"],
        "random_repeat_seeds": auth["random_repeat_seeds"],
        "source_code_sha256": {
            p: sha256_file(root / "src/cmc_bbdm/cai_agent_v3" / p) for p in code_paths
        },
        "preparation_base_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "versions": {
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
        "torch_threads": 4,
        "single_visible_gpu": True,
        "downstream": "NOT_AUTHORIZED_BEYOND_W3_SEED1_VALID",
    }
    protocol["identity"] = sha_payload(protocol)
    write_atomic(output / "protocol_snapshot.json", protocol)
    auth["execution_binding"] = {
        "task_source": "/home/ww/diff/docs/W3 seed1受控实验/CAI_V3_W3_VALID_PILOT_CODEX_PACKAGE.zip",
        "ledger_start_lines": len(rows(root)),
        "ledger_start_bytes": (root / LEDGER).stat().st_size,
        "actual_starting_upper_bound": update_usage(rows(root)),
        "protocol_identity": protocol["identity"],
    }
    write_atomic(output / "pilot_authorization.json", auth)
    # Prefer the existing outcome-independent VALID case manifest.
    previous_cases = root / DATA / "figure_manifest_valid.json"
    cases = []
    if previous_cases.exists():
        previous = json.loads(previous_cases.read_text())
        assert (
            previous["scope"] == "VALID"
            and previous["selection_uses_outcomes"] is False
        )
        valid_keys = {bank.specimen_keys[i] for i in bank.indices("VALID")}
        for row in previous["records"][:3]:
            assert row["specimen_key"] in valid_keys
            cases.append(
                {
                    "specimen_key": row["specimen_key"],
                    "dataset_id": row["dataset_id"],
                    "scope": "VALID",
                    "selection": "REUSE_FROZEN_VALID_CASE_MANIFEST",
                }
            )
    else:
        for domain in sorted({bank.dataset_ids[i] for i in bank.indices("VALID")})[:3]:
            keys = [
                bank.specimen_keys[i]
                for i in bank.indices("VALID")
                if bank.dataset_ids[i] == domain
            ]
            key = min(
                keys,
                key=lambda k: hashlib.sha256(
                    ("cai-v3-w3-case|" + k).encode()
                ).hexdigest(),
            )
            cases.append(
                {
                    "specimen_key": key,
                    "dataset_id": domain,
                    "scope": "VALID",
                    "selection": "PRE_SCORE_FIXED_HASH",
                }
            )
    write_csv(output / "case_manifest.csv", cases)
    return protocol


class PilotContext:
    def __init__(self, root):
        self.root = root
        self.output = root / RUN
        self.art = root / ART
        self.protocol = json.loads((self.output / "protocol_snapshot.json").read_text())
        self.auth = json.loads((self.output / "pilot_authorization.json").read_text())
        if (
            self.auth["task_id"] != TASK_ID
            or self.auth["budget"]["new_W3_updates_cap_including_failures"] != 5750
        ):
            raise ValueError("invalid W3 authorization")
        self.specs = {m["method"]: m for m in self.auth["models"]}
        self.jobs = {}
        self.archives = {}
        self.environments = {}
        self.session_id = uuid4().hex
        self.code_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        self.session_started = False

    def local(self):
        return [r for r in rows(self.root) if r.get("task_id") == TASK_ID]

    def append(self, **row):
        t._append_ledger(self.root / LEDGER, dict(task_id=TASK_ID, **row))

    def check_time(self):
        local = self.local()
        ends = {
            r["session_id"]: r for r in local if r.get("status") == "GPU_SESSION_ENDED"
        }
        used = sum(r["elapsed_gpu_seconds"] for r in ends.values())
        used += sum(
            time.time() - r["started_epoch"]
            for r in local
            if r.get("status") == "GPU_SESSION_STARTED" and r["session_id"] not in ends
        )
        if used >= 21600:
            raise ValueError("RESOURCE_LIMITED: W3 six-GPU-hour cap")

    def preflight(self):
        if (
            json.loads((self.art / "REQUIREMENTS_REVIEW.json").read_text())[
                "preflight_status"
            ]
            != "PASS"
        ):
            raise ValueError("W3 preflight not passed")
        for name, digest in self.protocol["source_code_sha256"].items():
            if sha256_file(self.root / "src/cmc_bbdm/cai_agent_v3" / name) != digest:
                raise ValueError("bound execution code changed since prepare")
        binding = json.loads((self.art / "INPUT_BINDINGS.json").read_text())
        for model in binding["predictors"]:
            if (
                sha256_file(self.root / model["checkpoint_path"])
                != model["checkpoint_sha256"]
            ):
                raise ValueError("bound upstream weight changed")
        if (
            sha256_file(self.root / DATA / "vlm_actor_features_fit.csv")
            != binding["vlm_features_sha256"]
        ):
            raise ValueError("bound VLM features changed")
        self.check_time()

    def start_session(self, device):
        self.preflight()
        if (
            device != "cuda:0"
            or len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 1
            or not os.environ.get("CUDA_VISIBLE_DEVICES")
        ):
            raise ValueError("one explicitly visible GPU required")
        self.session_started = True
        self.started = time.time()
        self.append(
            job="W3_GPU_SESSION",
            session_id=self.session_id,
            status="GPU_SESSION_STARTED",
            started_epoch=self.started,
            physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
            execution_code_sha=self.code_sha,
            actual_optimizer_updates=0,
        )

    def end_session(self, outcome):
        if not self.session_started:
            return
        torch.cuda.synchronize()
        self.append(
            job="W3_GPU_SESSION",
            session_id=self.session_id,
            status="GPU_SESSION_ENDED",
            elapsed_gpu_seconds=time.time() - self.started,
            outcome=outcome,
            actual_optimizer_updates=0,
        )

    def job_path(self, method):
        return self.output / "jobs" / f"{method}.json"

    def cached_job(self, method, bank, *, device):
        p = self.job_path(method)
        if not p.exists():
            return None
        saved = json.loads(p.read_text())
        m = saved["manifest"]
        if saved["protocol_identity"] != self.protocol["identity"] or not any(
            r.get("run_id") == saved["run_id"] and r.get("status") == "COMPLETED"
            for r in self.local()
        ):
            raise ValueError("incomplete or changed job; no automatic restart")
        inspect_actor_archive(
            self.root / m["selection_archive"], environment=saved["environment"]
        )
        # Loading constructs a model: preserve all Torch training RNG state.
        with torch.random.fork_rng(
            devices=list(range(torch.cuda.device_count()))
            if device.startswith("cuda")
            else []
        ):
            actor, _ = t.load_actor_checkpoint(
                self.root / m["checkpoint_path"], bank, device=device
            )
        return (
            actor,
            m,
            saved["progress"],
            read_episodes(self.root / m["selected_episodes"]),
        )

    def start_job(self, method, limit, seed):
        self.preflight()
        spec = self.specs[method]
        if (limit, seed) != (spec["max_updates"], spec["training_seed"]):
            raise ValueError("method scientific settings changed")
        local = self.local()
        if any(
            r.get("method") == method and r.get("status") == "STARTED" for r in local
        ):
            raise ValueError(
                "existing incomplete job retained; automatic restart prohibited"
            )
        check_budget(
            global_used=update_usage(rows(self.root)),
            task_used=update_usage(local),
            method_used=update_usage([r for r in local if r.get("method") == method]),
            required=limit,
            method_cap=limit,
        )
        self.jobs[method] = {
            "run_id": uuid4().hex,
            "started": time.time(),
            "max_updates": limit,
            "seed": seed,
        }
        self.environments[method] = {
            "protocol_identity": self.protocol["identity"],
            "method": method,
            "training_seed": seed,
            "execution_code_sha": self.code_sha,
        }
        self.append(
            job=f"W3_{method}_seed1",
            method=method,
            run_id=self.jobs[method]["run_id"],
            status="STARTED",
            optimizer_update_reservation=limit,
            actual_optimizer_updates=0,
        )

    def open_archive(self, method, actor, initial_sha):
        flags = {
            "use_vlm": t._uses_vlm(method),
            "use_feedback": method
            not in ("VLM_SPATIAL_OPEN_LOOP", "LEARNED_STATIC_TRUE"),
            "use_surface": method != "LEARNED_STATIC_TRUE",
        }
        self.archives[method] = ActorArchive(
            self.output / "models/selection_history" / f"{method.lower()}_seed1",
            method=method,
            seed=self.jobs[method]["seed"],
            max_updates=self.jobs[method]["max_updates"],
            environment=self.environments[method],
            run_id=self.jobs[method]["run_id"],
            episode_directory=self.output
            / "candidate_episodes"
            / f"{method.lower()}_seed1",
            metadata={
                "initial_state_dict_sha256": initial_sha,
                "architecture": repr(actor),
                "input_flags": flags,
                "parameter_count": sum(p.numel() for p in actor.parameters()),
            },
        )

    def before_update(self, method, update):
        if update > self.jobs[method]["max_updates"]:
            raise ValueError("RESOURCE_LIMITED: model update cap")
        self.check_time()

    def evaluation_identity(self, method, update, actor):
        return {
            "seed_panel": 1,
            "training_seed": self.jobs[method]["seed"],
            "checkpoint_update": update,
            "actor_state_dict_sha256": t._state_dict_sha256(actor.state_dict()),
            "evaluation_environment_identity": sha_payload(self.environments[method]),
            "execution_code_sha": self.code_sha,
        }

    def record(self, method, update, actor, episodes, score):
        candidate = self.archives[method].record(update, actor.state_dict(), episodes)
        if abs(candidate["metrics"]["left_error_area_mpa"] - score) > 1e-12:
            raise ValueError("same-evaluation score differs from stored trajectories")

    def checkpoint(
        self, method, update, actor, optimizer, rng, best_update, best_score, stale
    ):
        archive = self.archives[method]
        save_torch(
            archive.directory / "latest_training_state.pt",
            {
                "model": actor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "numpy_generator_state": rng.bit_generator.state,
                "numpy_global_state": np.random.get_state(),
                "python_rng": random.getstate(),
                "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all()
                if next(actor.parameters()).is_cuda
                else [],
                "update": update,
                "best_update": best_update,
                "best_score": best_score,
                "stale": stale,
                "run_id": self.jobs[method]["run_id"],
                "environment": self.environments[method],
            },
        )
        if update == 250:
            before = torch.get_rng_state().clone()
            cuda_before = (
                torch.cuda.get_rng_state_all()
                if next(actor.parameters()).is_cuda
                else []
            )
            row = archive.manifest["candidates"][0]
            payload = torch.load(
                archive.directory / row["checkpoint"],
                map_location="cpu",
                weights_only=False,
            )
            episodes = read_episodes(archive.directory / row["episodes"])
            assert archive.manifest["status"] == "INCOMPLETE" and len(episodes) == 50
            saved_sha = t._state_dict_sha256(payload["state_dict"])
            assert saved_sha == t._state_dict_sha256(actor.state_dict())
            assert all(e["actor_state_dict_sha256"] == saved_sha for e in episodes)
            assert torch.equal(before, torch.get_rng_state())
            cuda_after = (
                torch.cuda.get_rng_state_all()
                if next(actor.parameters()).is_cuda
                else []
            )
            assert all(
                torch.equal(x, y) for x, y in zip(cuda_before, cuda_after, strict=True)
            )
            assert any(
                r.get("run_id") == self.jobs[method]["run_id"]
                and r.get("status") == "STARTED"
                for r in self.local()
            )
            for episode in episodes:
                trace = json.loads(episode["execution_trace"])
                cells = [int(v) for v in episode["cells"].split(";")]
                assert [entry["cell"] for entry in trace] == cells
                assert [entry["actor_call_index"] for entry in trace] == list(
                    range(1, len(cells) + 1)
                )
                assert len(episode["costs"].split(";")) == len(cells) + 1
                assert all(
                    entry["environment_legal"][entry["cell"]] == "1"
                    and entry["proposal_legal"][entry["cell"]] == "1"
                    for entry in trace
                )
            write_atomic(
                self.art / f"first_250_{method}.json",
                {
                    "status": "FIRST_250_CONFIRMED",
                    "run_id": self.jobs[method]["run_id"],
                    "checkpoint": str(
                        (archive.directory / row["checkpoint"]).relative_to(self.root)
                    ),
                    "episodes_sha256": row["episodes_sha256"],
                    "environment": self.environments[method],
                    "archive_status": "INCOMPLETE",
                    "rng_unchanged": True,
                },
            )
        self.append(
            method=method,
            run_id=self.jobs[method]["run_id"],
            status="PROGRESS",
            observed_update=update,
            actual_optimizer_updates=0,
        )
        print(
            json.dumps(
                {
                    "method": method,
                    "update": update,
                    "validation_area": archive.manifest["candidates"][-1]["metrics"][
                        "left_error_area_mpa"
                    ],
                }
            ),
            flush=True,
        )

    def finish_archive(self, method, update, best_update):
        archive = self.archives[method]
        best = archive.finish(update)
        if best["update"] != best_update:
            raise ValueError("winner mismatch")
        payload = torch.load(
            archive.directory / best["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        return payload["state_dict"], read_episodes(
            archive.directory / best["episodes"]
        )

    def complete_job(self, method, actor, manifest, progress, episodes):
        archive = self.archives[method]
        best = next(
            r
            for r in archive.manifest["candidates"]
            if r["update"] == manifest["selected_update"]
        )
        path = self.output / "models" / f"actor_{method.lower()}_seed1.pt"
        manifest.update(
            selection_archive=str(archive.directory.relative_to(self.root)),
            selected_episodes=str(
                (archive.directory / best["episodes"]).resolve().relative_to(self.root)
            ),
            checkpoint_path=str(path.relative_to(self.root)),
        )
        save_torch(
            path,
            {
                "schema_version": 3,
                "method": method,
                "state_dict": actor.state_dict(),
                "manifest": manifest,
            },
        )
        manifest["checkpoint_sha256"] = sha256_file(path)
        self.job_path(method).parent.mkdir(exist_ok=True)
        write_atomic(
            self.job_path(method),
            {
                "run_id": self.jobs[method]["run_id"],
                "manifest": manifest,
                "progress": progress,
                "protocol_identity": self.protocol["identity"],
                "environment": self.environments[method],
            },
        )
        self.append(
            method=method,
            run_id=self.jobs[method]["run_id"],
            status="COMPLETED",
            actual_optimizer_updates=manifest["updates_completed"],
            elapsed_seconds=time.time() - self.jobs[method]["started"],
        )


def load_inputs(context, device):
    root = context.root
    w2 = source_root(root, W2)
    binding = json.loads((root / ART / "INPUT_BINDINGS.json").read_text())
    bank = load_feature_bank(project_root=root)
    features = t._VLMFeatures(bank, root / DATA / "vlm_actor_features_fit.csv")
    for m in binding["predictors"]:
        if sha256_file(root / m["checkpoint_path"]) != m["checkpoint_sha256"]:
            raise ValueError("frozen predictor changed")
    p_all, _ = load_predictor_checkpoint(
        root / binding["predictors"][0]["checkpoint_path"], device=device
    )
    p_all.eval().requires_grad_(False)
    oof, folds = t._load_oof_predictors(
        root, device=device, predictor_root=w2, bank=bank, verify=False
    )
    return bank, features, p_all, oof, folds, t._cell_costs(bank)


def run_fixed(context, device):
    path = context.output / "fixed_episodes.csv.gz"
    if path.exists():
        return {"status": "FIXED_REUSED", "episode_count": len(read_episodes(path))}
    bank, features, p_all, _, _, costs = load_inputs(context, device)
    episodes = []
    for method in context.auth["fixed_methods"]:
        context.check_time()
        _, current = t.evaluate_policy(
            None,
            method,
            p_all,
            bank,
            features,
            costs,
            device=device,
            execution_identity={
                "seed_panel": 0,
                "training_seed": 0,
                "checkpoint_update": 0,
                "actor_state_dict_sha256": "FIXED_ORDER",
                "evaluation_environment_identity": context.protocol["identity"],
                "execution_code_sha": context.code_sha,
            },
        )
        episodes.extend(current)
    write_episodes(path, episodes)
    from .w3_results import method_metrics

    write_csv(
        context.output / "fixed_metrics.csv",
        [
            method_metrics([r for r in episodes if r["method"] == m])
            for m in context.auth["fixed_methods"]
        ],
    )
    return {"status": "FIXED_COMPLETE", "episode_count": len(episodes)}


def train_pilots(context, device):
    if not (context.output / "fixed_episodes.csv.gz").exists():
        raise ValueError("fixed baseline must be completed first")
    bank, features, p_all, oof, folds, costs = load_inputs(context, device)
    manifests = []
    progress = []
    episodes = []
    for spec in context.auth["models"]:
        _actor, m, p, e = t._train_actor(
            spec["method"],
            spec["max_updates"],
            bank,
            features,
            p_all,
            oof,
            folds,
            costs,
            seed_panel=1,
            training_seed=spec["training_seed"],
            device=device,
            run_context=context,
        )
        manifests.append(m)
        progress.extend(p)
        episodes.extend(e)
        write_csv(context.output / "policy_training_progress.csv", progress)
    write_atomic(
        context.output / "actor_manifests.json",
        {"status": "FIVE_PILOTS_COMPLETE", "actor_manifests": manifests},
    )
    write_episodes(
        context.output / "policy_validation_episodes.csv.gz",
        read_episodes(context.output / "fixed_episodes.csv.gz") + episodes,
    )
    return {
        "status": "FIVE_PILOTS_COMPLETE",
        "actual_updates": sum(m["updates_completed"] for m in manifests),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare-run",
            "run-fixed",
            "train-pilots",
            "summarize",
            "export-figures",
        ],
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    torch.set_num_threads(4)
    if args.command == "prepare-run":
        result = prepare(root)
    elif args.command in ("summarize", "export-figures"):
        from .w3_results import export_figures, summarize

        result = (summarize if args.command == "summarize" else export_figures)(root)
    else:
        context = PilotContext(root)
        context.start_session("cuda:0")
        outcome = "FAILED"
        try:
            result = (run_fixed if args.command == "run-fixed" else train_pilots)(
                context, "cuda:0"
            )
            outcome = "COMPLETED"
        except Exception as error:
            outcome = (
                "RESOURCE_LIMITED" if "RESOURCE_LIMITED" in str(error) else "FAILED"
            )
            write_atomic(
                context.output / "execution_stop.json",
                {"status": outcome, "reason": str(error), "automatic_restart": False},
            )
            raise
        finally:
            context.end_session(outcome)
    print(json.dumps(result, indent=2, default=str))
