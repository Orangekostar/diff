"""Durable W2 selection evidence; never reconstruct missing historical weights."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .files import sha256_file

COST_DEFINITION = {
    "grid": "NATIVE_8X8_FULL_CELL_V1",
    "edges": "rint(linspace(0, native_dimension, 9))",
    "cost": "sum(native_cell_pixels)/(native_height*native_width)",
    "hard_cost_dtype": "float64",
    "model_input_dtype": "float32",
    "budget": 0.25,
    "legality_tolerance": 1e-12,
    "selection_metric": "fixed_four_route_left_area_domain_mean_mpa",
    "checkpoint_improvement_tolerance": 1e-12,
}


def validation_identity(bank, library) -> dict:
    """Bind only VALID inference inputs, labels, grouping and fixed prefix states."""
    valid = bank.indices("VALID")
    metadata = {
        "specimen_keys": [bank.specimen_keys[int(i)] for i in valid],
        "dataset_ids": [bank.dataset_ids[int(i)] for i in valid],
        "capture_group_ids": [bank.capture_group_ids[int(i)] for i in valid],
        "route_names": list(library.route_names),
        "cost_definition": COST_DEFINITION,
    }
    arrays = {
        "valid_indices": valid,
        "native_shapes": bank.native_shapes[valid],
        "targets_mpa": bank.targets_mpa[valid],
        "surface_tokens": bank.surface_tokens[valid],
        "cscan_tokens": bank.cscan_tokens[valid],
        "specimen_indices": library.specimen_indices,
        "state_indices": library.state_indices,
        "masks": library.masks,
        "costs": library.costs,
    }
    hashes = {}
    for name, value in arrays.items():
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
        hashes[name] = digest.hexdigest()
    metadata["array_sha256"] = hashes
    metadata["identity"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    return metadata


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _selected(rows: list[dict]) -> dict:
    best = None
    for row in rows:
        area = float(row["metrics"]["valid_area_mpa"])
        if not math.isfinite(area):
            raise ValueError("nonfinite checkpoint selection score")
        if best is None or area < float(best["metrics"]["valid_area_mpa"]) - 1e-12:
            best = row
    if best is None:
        raise ValueError("incomplete checkpoint selection history")
    return best


def _validate_schedule(manifest: dict) -> None:
    updates = [row["update"] for row in manifest["candidates"]]
    completed = manifest["updates_completed"]
    limit = manifest["max_updates"]
    interval = manifest["validation_interval"]
    expected = list(range(interval, completed + 1, interval))
    if completed == limit and completed % interval:
        expected.append(completed)
    if completed <= 0 or completed > limit or updates != expected:
        raise ValueError("incomplete checkpoint schedule")
    stale = 0
    best = math.inf
    for i, row in enumerate(manifest["candidates"]):
        area = float(row["metrics"]["valid_area_mpa"])
        if not math.isfinite(area):
            raise ValueError("nonfinite checkpoint selection score")
        if area < best - 1e-12:
            best, stale = area, 0
        else:
            stale += 1
        if stale >= manifest["validation_patience"] and i != len(updates) - 1:
            raise ValueError("checkpoint schedule continued after early stop")
    if completed < limit and stale < manifest["validation_patience"]:
        raise ValueError("incomplete checkpoint schedule before early stop")


class CheckpointArchive:
    """One new directory per run; existing runs are never overwritten."""

    def __init__(
        self,
        directory,
        *,
        model_name,
        validation,
        max_updates,
        validation_interval,
        validation_patience,
        run_metadata,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {
            "schema_version": 1,
            "status": "INCOMPLETE",
            "model_name": model_name,
            "validation": validation,
            "cost_definition": COST_DEFINITION,
            "max_updates": max_updates,
            "validation_interval": validation_interval,
            "validation_patience": validation_patience,
            "run_metadata": run_metadata,
            "candidates": [],
        }
        _write(self.directory / "selection.json", self.manifest)

    def record(self, update, state_dict, metrics):
        if self.manifest["status"] != "INCOMPLETE":
            raise ValueError("selection archive is already closed")
        rows = self.manifest["candidates"]
        if update <= (rows[-1]["update"] if rows else 0):
            raise ValueError("checkpoint updates must increase")
        if not all(math.isfinite(float(v)) for v in metrics.values()):
            raise ValueError("nonfinite checkpoint metrics")
        path = self.directory / f"update_{update:06d}.pt"
        if path.exists():
            raise FileExistsError(path)
        payload = {
            "schema_version": 3,
            "model_name": self.manifest["model_name"],
            "state_dict": {k: v.detach().cpu().clone() for k, v in state_dict.items()},
            "manifest": {
                "update": update,
                "metrics": metrics,
                "validation": self.manifest["validation"],
                "cost_definition": COST_DEFINITION,
                "run_metadata": self.manifest["run_metadata"],
            },
        }
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        rows.append(
            {
                "update": update,
                "metrics": dict(metrics),
                "checkpoint": path.name,
                "sha256": sha256_file(path),
            }
        )
        _write(self.directory / "selection.json", self.manifest)

    def finish(self, *, updates_completed):
        self.manifest["updates_completed"] = updates_completed
        _validate_schedule(self.manifest)
        self.manifest["selected_update"] = _selected(self.manifest["candidates"])[
            "update"
        ]
        self.manifest["status"] = "COMPLETE"
        _write(self.directory / "selection.json", self.manifest)
        return dict(self.manifest)


def inspect_archive(directory, *, validation) -> dict:
    """Reject missing candidates or changed input; a history label is not proof."""
    directory = Path(directory)
    manifest = json.loads((directory / "selection.json").read_text())
    if manifest["status"] != "COMPLETE" or manifest["validation"] != validation:
        raise ValueError("incomplete history or changed VALID identity")
    if manifest["cost_definition"] != COST_DEFINITION:
        raise ValueError("changed native cost definition")
    _validate_schedule(manifest)
    for row in manifest["candidates"]:
        path = directory / row["checkpoint"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError("checkpoint identity mismatch")
    if _selected(manifest["candidates"])["update"] != manifest["selected_update"]:
        raise ValueError("selected checkpoint does not match complete ranking")
    return manifest


def rescore_archive(directory, *, validation, score) -> dict:
    """Score every actual selection point, without training or changing old results.

    The caller supplies inference on the fixed bank/library. A different VALID
    identity requires new protocol review, not silent archive relabeling.
    """
    manifest = inspect_archive(directory, validation=validation)
    rows = []
    for row in manifest["candidates"]:
        payload = torch.load(
            Path(directory) / row["checkpoint"], map_location="cpu", weights_only=False
        )
        saved = payload["manifest"]
        if (
            payload["model_name"] != manifest["model_name"]
            or saved["update"] != row["update"]
            or saved["validation"] != validation
            or saved["cost_definition"] != COST_DEFINITION
        ):
            raise ValueError("checkpoint provenance mismatch")
        rows.append({**row, "metrics": dict(score(payload))})
    return {
        "selected_update": _selected(rows)["update"],
        "candidates": rows,
        "validation": validation,
        "optimizer_updates": 0,
    }
