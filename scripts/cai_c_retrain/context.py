"""Scope validation, paths, signatures and durable task state."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

TASK_ID = "CAI_V3_C_RENDER_RETRAIN_RELEASE_R1_331f5295"
BRANCH = "research/cai-vlm-agent-v3-controlled-reuse"
SOURCE_COMMIT = "331f52952b93bd9442ad2e44243cc71c7d966b4d"
PRIOR_VERSION = "C_P0_R1_GLOBAL_V1"
METHOD_SEEDS = {
    "VLM_SPATIAL_FEEDBACK": 2026091301,
    "VLM_SPATIAL_OPEN_LOOP": 2026091303,
    "VLM_MEAN_FEEDBACK": 2026091305,
}
PHASES = (
    "prepare",
    "vlm",
    "train",
    "assemble",
    "analyze",
    "paper",
    "verify",
    "publish",
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _validate_scope(scope: Mapping[str, object], *, branch: str) -> None:
    if scope.get("schema_version") != 1 or scope.get("task_id") != TASK_ID:
        raise ValueError("unexpected C retrain scope identity")
    if scope.get("branch") != BRANCH or branch != BRANCH:
        raise ValueError(f"scope must run on branch {BRANCH}")
    if scope.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("unexpected source commit")
    if scope.get("prior_version") != PRIOR_VERSION:
        raise ValueError("unexpected C prior version")
    if tuple(scope.get("stages", ())) != PHASES:
        raise ValueError("unexpected stage order")

    cohort = scope["cohort"]
    authorized = tuple(cohort["vlm_authorized_splits"])
    if authorized != ("TRAIN", "VALID") or "TEST" in authorized:
        raise ValueError("TEST access is forbidden")
    expected_cohort = {
        "train_physical": 161,
        "valid_physical": 50,
        "reserved_test_physical": 65,
        "c_prior_rows": 211,
    }
    if any(cohort.get(key) != value for key, value in expected_cohort.items()):
        raise ValueError("unexpected cohort counts")
    if cohort.get("test_scoring_allowed") is not False:
        raise ValueError("TEST scoring must remain disabled")

    models = scope["models"]
    actual_methods = {row["method"]: row for row in models}
    if set(actual_methods) != set(METHOD_SEEDS):
        raise ValueError("only the three registered C actors are authorized")
    for method, seed in METHOD_SEEDS.items():
        row = actual_methods[method]
        if row.get("training_seed") != seed:
            raise ValueError(f"unexpected training seed for {method}")
        if row.get("max_logical_updates") != 1250:
            raise ValueError(f"logical updates must be 1250 for {method}")
        if row.get("max_actual_updates_including_replay") != 1500:
            raise ValueError(f"actual update cap must be 1500 for {method}")
        if row.get("max_resumes") != 1:
            raise ValueError(f"resume cap must be one for {method}")

    training = scope["training"]
    if training.get("candidate_updates") != [250, 500, 750, 1000, 1250]:
        raise ValueError("unexpected candidate schedule")
    if training.get("selection_metric") != "DOMAIN_EQUAL_LEFT_ERROR_AREA_MPA":
        raise ValueError("unexpected checkpoint selection metric")
    if training.get("improvement_tolerance") != 1e-12:
        raise ValueError("unexpected selection tolerance")

    budget = scope["budget"]
    expected_budget = {
        "new_planned_optimizer_updates": 3750,
        "new_interruption_replay_reserve": 750,
        "new_actual_optimizer_updates_upper_bound": 4500,
        "new_global_cap": 44514,
        "max_visible_gpus": 1,
        "max_cpu_threads": 4,
    }
    if any(budget.get(key) != value for key, value in expected_budget.items()):
        raise ValueError("unexpected resource authorization")

    permissions = scope["permissions"]
    forbidden = (
        "TEST_evaluation_or_label_join",
        "extra_seed_panels",
        "W2_training",
        "CNN_encoding",
        "Qwen_finetuning",
        "new_attention_diagnostics",
        "GDFS",
        "STOP",
        "new_baselines_or_hyperparameter_search",
        "full_repository_test_sweep",
        "overwrite_frozen_production_sources_results_or_old_paper",
        "paper_journal_submission",
        "create_PR_merge_force_push_or_release_tag",
    )
    if any(permissions.get(name) is not False for name in forbidden):
        raise ValueError("scope enables a forbidden operation")

    roots = scope["roots"]
    for name, raw in roots.items():
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"root {name} must be repository-relative")
    immutable = {
        roots[name]
        for name in (
            "data",
            "predictors",
            "old_w3",
            "old_w3_artifacts",
            "old_evidence",
            "grounding_pilot",
            "old_paper",
        )
    }
    writable = {
        roots[name]
        for name in ("code", "output", "artifacts", "paper", "authorization")
    }
    if immutable & writable:
        raise ValueError("new outputs overlap immutable inputs")


@dataclass(frozen=True)
class TaskContext:
    root: Path
    scope: dict[str, object]
    config_path: Path
    branch: str
    head: str

    @classmethod
    def load(cls, config_path: str | Path) -> TaskContext:
        candidate = Path(config_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve(strict=True)
        root = Path(_git(candidate.parent, "rev-parse", "--show-toplevel")).resolve()
        branch = _git(root, "branch", "--show-current")
        head = _git(root, "rev-parse", "HEAD")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
            cwd=root,
            check=True,
        )
        scope = json.loads(candidate.read_text(encoding="utf-8"))
        return cls.from_mapping(root, scope, candidate, branch=branch, head=head)

    @classmethod
    def from_mapping(
        cls,
        root: str | Path,
        scope: Mapping[str, object],
        config_path: str | Path,
        *,
        branch: str,
        head: str,
    ) -> TaskContext:
        copied = json.loads(json.dumps(scope))
        _validate_scope(copied, branch=branch)
        return cls(
            root=Path(root).resolve(),
            scope=copied,
            config_path=Path(config_path).resolve(),
            branch=branch,
            head=head,
        )

    @property
    def task_id(self) -> str:
        return str(self.scope["task_id"])

    @property
    def scope_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.scope))

    def path(self, name: str) -> Path:
        try:
            relative = Path(self.scope["roots"][name])
        except KeyError as error:
            raise KeyError(f"unknown configured root: {name}") from error
        path = (self.root / relative).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"configured root escapes repository: {name}")
        return path

    @property
    def task_state_path(self) -> Path:
        return self.path("output") / "task_state.json"

    def phase_signature(self, phase: str, inputs: Sequence[Path]) -> str:
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase}")
        records = []
        for raw in inputs:
            path = Path(raw).resolve(strict=True)
            records.append({"path": str(path), "sha256": sha256_file(path)})
        return sha256_bytes(
            canonical_json(
                {
                    "task_id": self.task_id,
                    "scope_sha256": self.scope_sha256,
                    "phase": phase,
                    "inputs": records,
                }
            )
        )

    def require_phase_signature(
        self, phase: str, expected: str, inputs: Sequence[Path]
    ) -> None:
        actual = self.phase_signature(phase, inputs)
        if actual != expected:
            raise ValueError(
                f"{phase} signature changed: expected {expected}, observed {actual}"
            )

    def transition(self, phase: str, status: str, **details: object) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase}")
        state = {
            "task_id": self.task_id,
            "scope_sha256": self.scope_sha256,
            "phases": {},
        }
        if self.task_state_path.is_file():
            state = json.loads(self.task_state_path.read_text(encoding="utf-8"))
            if (
                state.get("task_id") != self.task_id
                or state.get("scope_sha256") != self.scope_sha256
            ):
                raise ValueError("existing task state has a different scope")
        state.setdefault("phases", {})[phase] = {"status": status, **details}
        atomic_json(self.task_state_path, state)

    def append_resource_event(self, event: Mapping[str, object]) -> None:
        path = self.path("output") / "resource_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"task_id": self.task_id, **dict(event)}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(payload).decode("utf-8") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
