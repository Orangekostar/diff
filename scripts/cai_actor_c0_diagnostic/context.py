"""Scope validation, atomic state, hashing, and total-task resource accounting."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_ID = "CAI_ACTOR_C0_MECHANISM_VIS_R2_9e765b04"
BRANCH = "research/cai-vlm-agent-v3-controlled-reuse"
SOURCE_COMMIT = "9e765b043a62b1d18341aefecc1f704aebbd62f6"
PHASES = ("prepare", "replay", "diagnose", "render", "verify", "publish")

_ZERO_LIMITS = (
    "optimizer_updates",
    "new_qwen_generations",
    "new_qwen_forwards",
    "cnn_forwards",
    "oof_predictor_forwards",
    "test_access",
    "paper_writes",
    "bootstrap_draws",
)
_COUNTER_LIMITS = {
    "actor_forward_examples": "actor_forward_examples_including_failures_max",
    "predictor_forward_examples": "predictor_forward_examples_including_failures_max",
    "autograd_gradient_queries": "autograd_gradient_queries_max",
    "native_full_episode_runs": "native_full_episode_runs",
    "c_no_c0_full_episode_runs": "c_no_c0_full_episode_runs",
    "full_episode_runs": "all_full_episode_runs_max",
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _validate_scope(scope: Mapping[str, Any], branch: str) -> None:
    if scope.get("schema_version") != 2 or scope.get("task_id") != TASK_ID:
        raise ValueError("unexpected task identity")
    if scope.get("branch") != BRANCH or branch != BRANCH:
        raise ValueError(f"scope must run on branch {BRANCH}")
    if scope.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("unexpected source commit")
    if scope.get("scope") != "FROZEN_POLICY_DIAGNOSTICS_NOT_TRAINING_OR_PAPER_REVISION":
        raise ValueError("unexpected diagnostic scope")
    if scope.get("cases", {}).get("split") != "VALID":
        raise ValueError("only VALID cases are authorized")
    if scope.get("cases", {}).get("max_specimens") != 6:
        raise ValueError("exactly six specimens are authorized")
    limits = scope.get("limits", {})
    for name in _ZERO_LIMITS:
        if limits.get(name) != 0:
            raise ValueError(f"forbidden resource must remain zero: {name}")
    if limits.get("cpu_threads_max") != 4 or limits.get("gpu_devices_max") != 1:
        raise ValueError("unexpected compute concurrency")
    roots = scope.get("roots", {})
    for name, raw in roots.items():
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"root {name} must be repository-relative")
    writable = {roots.get(name) for name in ("code", "output", "artifacts", "config")}
    immutable = {
        roots.get(name)
        for name in ("data", "c_release", "historical_w3", "predictor")
    }
    if writable & immutable:
        raise ValueError("writable and immutable roots overlap")
    models = scope.get("models", {})
    if models.get("C", {}).get("method") != "VLM_SPATIAL_FEEDBACK":
        raise ValueError("unexpected C model")
    if models.get("N", {}).get("method") != "NO_VLM_SPATIAL_FEEDBACK":
        raise ValueError("unexpected N model")
    delivery = scope.get("delivery", {})
    if delivery.get("actual_commit_push") is not True or delivery.get("same_branch") is not True:
        raise ValueError("delivery must remain on the bound branch")


@dataclass(frozen=True)
class TaskContext:
    root: Path
    scope: dict[str, Any]
    config_path: Path
    branch: str
    head: str

    @classmethod
    def load(cls, config_path: str | Path) -> TaskContext:
        candidate = Path(config_path)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        candidate = candidate.resolve(strict=True)
        root = Path(_git(candidate.parent, "rev-parse", "--show-toplevel")).resolve()
        branch = _git(root, "branch", "--show-current")
        head = _git(root, "rev-parse", "HEAD")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
            cwd=root,
            check=True,
        )
        return cls.from_mapping(
            root,
            json.loads(candidate.read_text(encoding="utf-8")),
            candidate,
            branch=branch,
            head=head,
        )

    @classmethod
    def from_mapping(
        cls,
        root: str | Path,
        scope: Mapping[str, Any],
        config_path: str | Path,
        *,
        branch: str,
        head: str,
    ) -> TaskContext:
        copied = json.loads(json.dumps(scope))
        _validate_scope(copied, branch)
        return cls(
            root=Path(root).resolve(),
            scope=copied,
            config_path=Path(config_path).resolve(),
            branch=branch,
            head=head,
        )

    def path(self, name: str) -> Path:
        try:
            candidate = (self.root / self.scope["roots"][name]).resolve()
        except KeyError as error:
            raise KeyError(f"unknown configured root: {name}") from error
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"configured root escapes repository: {name}")
        return candidate

    def source_path(
        self,
        repository_relative: str | Path,
        *,
        additional_roots: Sequence[str | Path] = (),
    ) -> Path:
        """Resolve immutable data that may live only in the primary checkout."""

        relative = Path(repository_relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source path must be repository-relative")
        local = self.root / relative
        if local.exists():
            return local
        for raw_root in additional_roots:
            candidate = Path(raw_root).resolve() / relative
            if candidate.exists():
                return candidate.resolve(strict=True)
        common_dir = Path(_git(self.root, "rev-parse", "--git-common-dir")).resolve()
        primary = common_dir.parent / relative
        return primary.resolve(strict=True)

    @property
    def scope_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.scope))

    @property
    def state_path(self) -> Path:
        return self.path("output") / "task_state.json"

    def stage_complete(self, stage: str, signature: str) -> bool:
        if stage not in PHASES:
            raise ValueError(f"unknown stage: {stage}")
        if not self.state_path.exists():
            return False
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        row = state.get("phases", {}).get(stage, {})
        return row.get("status") == "COMPLETE" and row.get("signature") == signature

    def complete_stage(self, stage: str, signature: str, result: Mapping[str, Any]) -> None:
        if stage not in PHASES:
            raise ValueError(f"unknown stage: {stage}")
        state = (
            json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state_path.exists()
            else {"task_id": TASK_ID, "phases": {}}
        )
        state.setdefault("phases", {})[stage] = {
            "status": "COMPLETE",
            "signature": signature,
            "result": dict(result),
        }
        atomic_json(self.state_path, state)

    def phase_signature(self, stage: str, inputs: Sequence[str | Path]) -> str:
        records = [
            {
                "path": str(Path(path).resolve().relative_to(self.root)),
                "sha256": sha256_file(path),
            }
            for path in inputs
        ]
        return sha256_bytes(canonical_json({"stage": stage, "inputs": records}))


class ResourceLedger:
    """Durable counters whose caps apply to the complete task, not each command."""

    def __init__(self, path: str | Path, limits: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.limits = dict(limits)
        if self.path.exists():
            self.value = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.value = {
                "task_id": TASK_ID,
                "counts": {name: 0 for name in (*_ZERO_LIMITS, *_COUNTER_LIMITS)},
                "sessions": [],
            }
            atomic_json(self.path, self.value)

    def charge(self, name: str, examples: int) -> int:
        if name not in self.value["counts"] or examples < 0:
            raise ValueError(f"unknown or invalid resource charge: {name}")
        proposed = int(self.value["counts"][name]) + int(examples)
        cap = 0 if name in _ZERO_LIMITS else int(self.limits[_COUNTER_LIMITS[name]])
        if proposed > cap:
            raise ValueError(f"resource cap exceeded: {name} ({proposed}>{cap})")
        self.value["counts"][name] = proposed
        atomic_json(self.path, self.value)
        return proposed

    def reconcile_minimum(self, name: str, observed: int, reason: str) -> int:
        """Raise a stale counter to an independently auditable observed minimum."""

        if name not in self.value["counts"] or observed < 0:
            raise ValueError(f"unknown or invalid resource reconciliation: {name}")
        cap = 0 if name in _ZERO_LIMITS else int(self.limits[_COUNTER_LIMITS[name]])
        if observed > cap:
            raise ValueError(f"resource cap exceeded: {name} ({observed}>{cap})")
        prior = int(self.value["counts"][name])
        self.value["counts"][name] = max(prior, int(observed))
        self.value.setdefault("reconciliations", []).append(
            {"counter": name, "prior": prior, "observed_minimum": int(observed), "reason": reason}
        )
        atomic_json(self.path, self.value)
        return int(self.value["counts"][name])


__all__ = [
    "BRANCH",
    "PHASES",
    "SOURCE_COMMIT",
    "TASK_ID",
    "ResourceLedger",
    "TaskContext",
    "atomic_json",
    "canonical_json",
    "sha256_bytes",
    "sha256_file",
]
