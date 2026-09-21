"""Cache-only provenance and report-table helpers for Actor C0 diagnostics."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .context import TaskContext, atomic_json, canonical_json, sha256_bytes, sha256_file


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean feature value: {value!r}")


def _parse_list(value: str) -> list[Any]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise TypeError("state provenance field must contain a list")
    return parsed


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty report table")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def hash_named_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash named numeric arrays with stable names, shapes, dtypes, and bytes."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if value.dtype.hasobject:
            raise ValueError("object arrays cannot be hashed canonically")
        dtype = value.dtype.newbyteorder("<")
        normalized = np.ascontiguousarray(value.astype(dtype, copy=False))
        descriptor = canonical_json(
            {
                "name": str(name),
                "dtype": f"{dtype.kind}{dtype.itemsize}",
                "shape": list(normalized.shape),
            }
        )
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        payload = normalized.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def prior_channels(row: Mapping[str, str]) -> dict[str, np.ndarray]:
    """Parse the four Actor prior channels using their true input shapes."""

    indicator = np.asarray(
        [float(value) for value in row["region_indicator"].split(";")],
        dtype=np.float32,
    )
    confidence = np.asarray(
        [float(value) for value in row["confidence"].split(";")],
        dtype=np.float32,
    )
    if indicator.shape != (64,) or confidence.shape != (64,):
        raise ValueError("VLM spatial prior channels must each contain 64 cells")
    return {
        "region_indicator": indicator,
        "confidence": confidence,
        "vlm_available": np.asarray([_parse_bool(row["vlm_available"])], dtype=np.float32),
        "no_reliable_cue": np.asarray(
            [_parse_bool(row["no_reliable_cue"])], dtype=np.float32
        ),
    }


def policy_input_hash(
    physical_state_sha256: str,
    *,
    model: str,
    channels: Mapping[str, np.ndarray],
) -> str:
    if len(physical_state_sha256) != 64:
        raise ValueError("physical-state hash must be SHA256")
    if model not in {"C", "N"}:
        raise ValueError("policy-input hash model must be C or N")
    return sha256_bytes(
        canonical_json(
            {
                "physical_state_sha256": physical_state_sha256,
                "model": model,
                "method": (
                    "VLM_SPATIAL_FEEDBACK"
                    if model == "C"
                    else "NO_VLM_SPATIAL_FEEDBACK"
                ),
                "use_vlm": model == "C",
                "supplied_prior_channels_sha256": hash_named_arrays(channels),
            }
        )
    )


def derive_state_provenance(
    row: Mapping[str, str],
    trajectories: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    labels = [str(value) for value in _parse_list(row["sources"])]
    models = [str(value) for value in _parse_list(row["source_models"])]
    prefixes = [int(value) for value in _parse_list(row["prefix_lengths"])]
    if not labels or not (len(labels) == len(models) == len(prefixes)):
        raise ValueError("state provenance fields have inconsistent lengths")
    t = int(row["action_count"])
    origins: list[dict[str, object]] = []
    ordered_prefixes: list[list[int]] = []
    for label, model, prefix in zip(labels, models, prefixes):
        if model not in trajectories:
            raise ValueError(f"missing {model} native trajectory")
        actions = [int(value) for value in trajectories[model]["actions"]]  # type: ignore[index]
        if prefix < 0 or prefix > len(actions):
            raise ValueError(f"prefix length is outside {model} native trajectory")
        if prefix != t:
            raise ValueError("state action_count and source prefix differ")
        prefix_cells = actions[:prefix]
        ordered_prefixes.append(prefix_cells)
        origins.append(
            {
                "label": label,
                "model": model,
                "t": prefix,
                "prefix_cells_in_order": prefix_cells,
            }
        )
    if any(prefix != ordered_prefixes[0] for prefix in ordered_prefixes[1:]):
        raise ValueError("deduplicated physical state has inconsistent ordered histories")
    return {
        "t": t,
        "prefix_cells_in_order": ordered_prefixes[0],
        "state_origins": origins,
    }


def trajectory_difference_rows(
    trajectories: Mapping[str, Mapping[str, object]],
) -> list[dict[str, Any]]:
    required = ("C_NATIVE", "N_NATIVE", "C_NO_C0")
    if any(name not in trajectories for name in required):
        raise ValueError("three trajectory conditions are required")
    maximum = max(len(trajectories[name]["actions"]) for name in required)  # type: ignore[arg-type,index]
    rows: list[dict[str, Any]] = []
    for index in range(maximum):
        row: dict[str, Any] = {"step": index + 1}
        c_action: int | None = None
        c_cost: float | None = None
        for condition in required:
            actions = trajectories[condition]["actions"]  # type: ignore[index]
            costs = trajectories[condition]["costs"]  # type: ignore[index]
            action = int(actions[index]) if index < len(actions) else None  # type: ignore[arg-type]
            cost = float(costs[index + 1]) if action is not None else None  # type: ignore[arg-type,index]
            row[f"{condition.lower()}_action"] = "" if action is None else action
            row[f"{condition.lower()}_cost"] = "" if cost is None else cost
            if condition == "C_NATIVE":
                c_action, c_cost = action, cost
            else:
                row[f"{condition.lower()}_action_differs_from_c"] = (
                    "" if action is None or c_action is None else action != c_action
                )
                row[f"{condition.lower()}_cost_minus_c"] = (
                    "" if cost is None or c_cost is None else cost - c_cost
                )
        rows.append(row)
    return rows


def _carry_render_timing(
    current: Mapping[str, Any], existing: Mapping[str, Any] | None
) -> dict[str, Any]:
    result = dict(current)
    if not existing:
        return result
    seconds = existing.get("cache_report_render_seconds_upper_bound")
    measurement = existing.get("timing_measurement")
    if (
        isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and np.isfinite(seconds)
        and seconds >= 0
        and measurement == "MONOTONIC_WALL_SECONDS_ROUNDED_UP"
    ):
        result["cache_report_render_seconds_upper_bound"] = seconds
        result["timing_measurement"] = measurement
    return result


def enrich_cached_report(context: TaskContext) -> dict[str, Any]:
    """Attach provenance derived only from already-persisted diagnostic caches."""

    output = context.path("output")
    feature_path = context.path("c_release") / "vlm/vlm_actor_features_fit.csv"
    feature_by_key: dict[str, dict[str, str]] = {}
    for feature in _read_csv(feature_path):
        key = feature["specimen_key"]
        if key in feature_by_key:
            raise ValueError(f"duplicate frozen VLM feature row: {key}")
        feature_by_key[key] = feature
    sensitivity = {
        (row["specimen_key"], row["state_id"]): row
        for row in _read_csv(output / "fixed_state_prior_sensitivity.csv")
    }
    states = _read_csv(output / "state_manifest.csv")
    enriched: list[dict[str, Any]] = []
    for state in states:
        key = state["specimen_key"]
        slug = key.replace(":", "_")
        trajectories = {
            model: json.loads(
                (output / "trajectories" / slug / f"{model}_NATIVE.json").read_text(
                    encoding="utf-8"
                )
            )
            for model in ("C", "N")
        }
        provenance = derive_state_provenance(state, trajectories)
        channels = prior_channels(feature_by_key[key])
        c_prior_sha256 = hash_named_arrays(channels)
        policy_hashes = {
            model: policy_input_hash(
                state["physical_state_sha256"], model=model, channels=channels
            )
            for model in ("C", "N")
        }
        current: dict[str, Any] = dict(state)
        current.update(
            {
                "t": provenance["t"],
                "prefix_cells_in_order": json.dumps(
                    provenance["prefix_cells_in_order"], separators=(",", ":")
                ),
                "state_origins": json.dumps(
                    provenance["state_origins"], sort_keys=True, separators=(",", ":")
                ),
                "policy_input_sha256": json.dumps(
                    policy_hashes, sort_keys=True, separators=(",", ":")
                ),
                "c_prior_sha256": c_prior_sha256,
            }
        )
        state_root = output / "states" / slug / state["state_id"]
        metadata_path = state_root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "t": provenance["t"],
                "prefix_cells_in_order": provenance["prefix_cells_in_order"],
                "state_origins": provenance["state_origins"],
                "policy_input_sha256": policy_hashes,
                "c_prior_sha256": c_prior_sha256,
            }
        )
        atomic_json(metadata_path, metadata)
        sensitivity_row = sensitivity[(key, state["state_id"])]
        atomic_json(
            state_root / "C/zero_prior_sensitivity.json",
            {
                "scope": "FIXED_STATE_C_PRIOR_CHANNEL_QUERY_NO_ROLLOUT",
                "source_table": "fixed_state_prior_sensitivity.csv",
                **sensitivity_row,
            },
        )
        enriched.append(current)
    _write_csv(output / "state_manifest.csv", enriched)
    provenance_path = output / "report_provenance.json"
    existing_provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else None
    )
    result = _carry_render_timing({
        "status": "PASS_CACHE_ONLY_PROVENANCE",
        "state_count": len(enriched),
        "feature_source": str(feature_path.relative_to(context.root)),
        "feature_source_sha256": sha256_file(feature_path),
        "model_forwards": 0,
    }, existing_provenance)
    atomic_json(provenance_path, result)
    return result


__all__ = [
    "derive_state_provenance",
    "enrich_cached_report",
    "hash_named_arrays",
    "policy_input_hash",
    "prior_channels",
    "trajectory_difference_rows",
]
