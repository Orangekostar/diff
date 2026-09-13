"""Actor candidates bind a fixed environment and their own dynamic trajectories."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import torch

from .files import sha256_file
from .metrics import left_error_area_mpa, trajectory_objective_mpa


def write_atomic(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def save_torch(path, payload):
    path = Path(path)
    temp = path.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    temp.replace(path)


def write_episodes(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with gzip.open(temp, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_episodes(path):
    with gzip.open(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def episode_metrics(rows):
    from .actor_training import _domain_equal_episode_score

    evaluated = []
    for row in rows:
        costs = [float(v) for v in row["costs"].split(";")]
        preds = [float(v) for v in row["predictions_mpa"].split(";")]
        cells = [v for v in row["cells"].split(";") if v]
        if len(costs) != len(preds) or len(cells) + 1 != len(costs):
            raise ValueError("episode state/action lengths differ")
        target = float(row["target_mpa"])
        evaluated.append(
            {
                **row,
                "left_error_area_mpa": left_error_area_mpa(
                    costs, preds, target, end=0.25
                ),
                "early_left_error_area_mpa": left_error_area_mpa(
                    costs, preds, target, end=0.0625
                ),
                "trajectory_objective_mpa": trajectory_objective_mpa(
                    costs, preds, target, budget=0.25
                ),
                "final_error_mpa": abs(preds[-1] - target),
            }
        )
    return {
        k: _domain_equal_episode_score(evaluated, k)
        for k in (
            "left_error_area_mpa",
            "early_left_error_area_mpa",
            "trajectory_objective_mpa",
            "final_error_mpa",
        )
    }


class ActorArchive:
    def __init__(
        self,
        directory,
        *,
        method,
        seed,
        max_updates,
        environment,
        run_id,
        episode_directory=None,
        metadata=None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.episode_directory = (
            Path(episode_directory) if episode_directory else self.directory
        )
        self.episode_directory.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "status": "INCOMPLETE",
            "method": method,
            "seed": seed,
            "max_updates": max_updates,
            "validation_interval": 250,
            "patience": 4,
            "environment": environment,
            "run_id": run_id,
            "metadata": metadata or {},
            "candidates": [],
        }
        write_atomic(self.directory / "selection.json", self.manifest)

    def record(self, update, state, episodes):
        import os

        if self.manifest["status"] != "INCOMPLETE" or (
            self.manifest["candidates"]
            and update <= self.manifest["candidates"][-1]["update"]
        ):
            raise ValueError("candidate order invalid")
        metrics = episode_metrics(episodes)
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("nonfinite candidate metric")
        weights = self.directory / f"update_{update:06d}.pt"
        trajectory = self.episode_directory / f"update_{update:06d}.csv.gz"
        save_torch(
            weights,
            {
                "schema_version": 3,
                "method": self.manifest["method"],
                "state_dict": state,
                "manifest": dict(
                    update=update,
                    environment=self.manifest["environment"],
                    run_id=self.manifest["run_id"],
                    **self.manifest["metadata"],
                ),
            },
        )
        write_episodes(trajectory, episodes)
        candidate = {
            "update": update,
            "checkpoint": weights.name,
            "checkpoint_sha256": sha256_file(weights),
            "episodes": os.path.relpath(trajectory, self.directory),
            "episodes_sha256": sha256_file(trajectory),
            "episode_count": len(episodes),
            "metrics": metrics,
        }
        self.manifest["candidates"].append(candidate)
        write_atomic(self.directory / "selection.json", self.manifest)
        return candidate

    def finish(self, update):
        best = validate_schedule(self.manifest, update)
        self.manifest.update(
            status="COMPLETE", updates_completed=update, selected_update=best["update"]
        )
        write_atomic(self.directory / "selection.json", self.manifest)
        return best


def validate_schedule(manifest, completed):
    candidates = manifest["candidates"]
    expected = list(range(250, completed + 1, 250))
    if completed == manifest["max_updates"] and completed % 250:
        expected.append(completed)
    if (
        completed <= 0
        or completed > manifest["max_updates"]
        or [r["update"] for r in candidates] != expected
    ):
        raise ValueError("incomplete Actor candidate schedule")
    best = None
    stale = 0
    for i, row in enumerate(candidates):
        score = row["metrics"]["left_error_area_mpa"]
        if not math.isfinite(score):
            raise ValueError("nonfinite selection score")
        if best is None or score < best["metrics"]["left_error_area_mpa"] - 1e-12:
            best, stale = row, 0
        else:
            stale += 1
        if stale >= 4 and i < len(candidates) - 1:
            raise ValueError("continued after scientific early stop")
    if completed < manifest["max_updates"] and stale < 4:
        raise ValueError("interruption is not scientific early stopping")
    return best


def inspect_actor_archive(directory, *, environment):
    directory = Path(directory)
    m = json.loads((directory / "selection.json").read_text())
    if m["status"] != "COMPLETE" or m["environment"] != environment:
        raise ValueError("incomplete archive or changed evaluation environment")
    best = validate_schedule(m, m["updates_completed"])
    if best["update"] != m["selected_update"]:
        raise ValueError("selection mismatch")
    for row in m["candidates"]:
        for name, digest in [
            ("checkpoint", "checkpoint_sha256"),
            ("episodes", "episodes_sha256"),
        ]:
            path = directory / row[name]
            if not path.is_file() or sha256_file(path) != row[digest]:
                raise ValueError("missing or changed candidate evidence")
    return m
