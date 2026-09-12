"""Trace and fixed-hash figure exports for completed v3 policy pilots."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from cmc_bbdm.cai_active_image.environment import NativeCellGrid
from cmc_bbdm.vlm_cscan.runtime import render_surface_inputs

from .actor_training import _uses_vlm, _VLMFeatures
from .feature_bank import load_feature_bank
from .files import read_csv, sha256_file, write_csv, write_json
from .policy import vlm_first_action_mask

_FIXED_METHODS = {"CENTER_FIRST", "GEOMETRY_SPREAD", "SERPENTINE", "RANDOM"}


def _values(text: str, *, cast=float) -> list:
    return [cast(value) for value in text.split(";") if value != ""]


def _mask_text(cells: set[int]) -> str:
    return "".join("1" if cell in cells else "0" for cell in range(64))


def _diagnostic_feature_path(output: Path, scope: str) -> Path:
    name = "fit" if scope == "VALID" else "test"
    return output / f"vlm_actor_features_{name}.csv"


def _draw_grid(axis, shape: tuple[int, int], *, color: str = "white") -> None:
    height, width = shape
    for fraction in np.linspace(0.0, 1.0, 9):
        axis.axhline(fraction * height - 0.5, color=color, linewidth=0.5, alpha=0.7)
        axis.axvline(fraction * width - 0.5, color=color, linewidth=0.5, alpha=0.7)


def _draw_cells(
    axis,
    grid: NativeCellGrid,
    cells: list[int],
    *,
    color: str,
    fill: bool,
    linewidth: float = 1.8,
) -> None:
    for index in cells:
        cell = grid.cells[index]
        axis.add_patch(
            Rectangle(
                (cell.col_start - 0.5, cell.row_start - 0.5),
                cell.col_stop - cell.col_start,
                cell.row_stop - cell.row_start,
                facecolor=color if fill else "none",
                edgecolor=color,
                alpha=0.22 if fill else 1.0,
                linewidth=linewidth,
            )
        )


def _measured_image(
    image: np.ndarray, grid: NativeCellGrid, cells: list[int]
) -> np.ndarray:
    result = np.full_like(image, 238)
    for index in cells:
        cell = grid.cells[index]
        result[cell.row_start : cell.row_stop, cell.col_start : cell.col_stop] = image[
            cell.row_start : cell.row_stop, cell.col_start : cell.col_stop
        ]
    return result


def _checkpoint_hashes(
    output: Path, policy_gate: dict[str, object]
) -> dict[tuple[str, str], str]:
    manifests = list(policy_gate["actor_manifests"])
    expansion_path = output / "policy_expansion.json"
    if expansion_path.is_file():
        expansion = json.loads(expansion_path.read_text(encoding="utf-8"))
        if expansion.get("status") == "POLICY_SEEDS_1_TO_3_LOCKED":
            manifests.extend(expansion["new_actor_manifests"])
    return {
        (str(row["method"]), str(row["seed_panel"])): str(
            row["checkpoint_sha256"]
        )
        for row in manifests
    }


def build_action_trace(
    *, project_root: str | Path, scope: str = "VALID"
) -> list[dict[str, object]]:
    root = Path(project_root).resolve(strict=True)
    output = root / "results/cai_agent_v3/new_protocol"
    scope_upper = scope.upper()
    if scope_upper not in {"VALID", "TEST"}:
        raise ValueError("diagnostic scope must be VALID or TEST")
    episode_path = (
        output / "policy_validation_episodes.csv"
        if scope_upper == "VALID"
        else output / "policy_test_episodes.csv"
    )
    feature_path = _diagnostic_feature_path(output, scope_upper)
    bank = load_feature_bank(project_root=root)
    features = _VLMFeatures(bank, feature_path, required_splits=(scope_upper,))
    feature_rows = {row["specimen_key"]: row for row in read_csv(feature_path)}
    key_index = {key: index for index, key in enumerate(bank.specimen_keys)}
    policy_gate = json.loads(
        (output / "policy_pilot_gate.json").read_text(encoding="utf-8")
    )
    actor_hashes = _checkpoint_hashes(output, policy_gate)
    predictor_gate = json.loads(
        (output / "predictor_gate.json").read_text(encoding="utf-8")
    )
    predictor_manifest = next(
        row
        for row in predictor_gate["candidate_manifests"]
        if row["model"] == predictor_gate["selected_p_all"]
    )
    predictor_hash = str(predictor_manifest["checkpoint_sha256"])
    trace_rows = []
    for episode in read_csv(episode_path):
        key = episode["specimen_key"]
        index = key_index[key]
        if bank.splits[index] != scope_upper:
            raise ValueError("episode scope and feature-bank split differ")
        cells = _values(episode["cells"], cast=int)
        costs = _values(episode["costs"])
        predictions = _values(episode["predictions_mpa"])
        if len(costs) != len(cells) + 1 or len(predictions) != len(costs):
            raise ValueError("policy episode trajectory is misaligned")
        grid = NativeCellGrid.from_shape(
            tuple(int(value) for value in bank.native_shapes[index])
        )
        measured: set[int] = set()
        method = episode["method"]
        cache_key = feature_rows[key]["cache_key"]
        for action_index, cell_index in enumerate(cells):
            legal_np = grid.legal_mask(frozenset(measured), endpoint_budget=0.25)
            if method in _FIXED_METHODS:
                c0_reason = "FIXED_ORDER_NO_ACTOR"
                initial_candidates = np.flatnonzero(legal_np)
                actor_call_index: int | str = ""
                decision_kind = "FIXED_ORDER"
            else:
                proposal, reasons = vlm_first_action_mask(
                    legal=torch.from_numpy(legal_np[None, :]),
                    use_vlm=_uses_vlm(method),
                    action_count=torch.tensor([action_index]),
                    indicator=torch.from_numpy(features.indicator[index : index + 1]),
                    confidence=torch.from_numpy(features.confidence[index : index + 1]),
                    available=torch.from_numpy(features.available[index : index + 1]),
                    no_reliable=torch.from_numpy(
                        features.no_reliable[index : index + 1]
                    ),
                )
                c0_reason = reasons[0]
                initial_candidates = (
                    torch.nonzero(proposal[0], as_tuple=False).flatten().numpy()
                )
                actor_call_index = action_index + 1
                decision_kind = "ACTOR_ARGMAX"
            if cell_index not in initial_candidates:
                raise ValueError(
                    "stored action is outside its reconstructed proposal set"
                )
            cell = grid.cells[cell_index]
            before = set(measured)
            measured.add(cell_index)
            trace_rows.append(
                {
                    "scope": scope_upper,
                    "specimen_key": key,
                    "dataset_id": episode["dataset_id"],
                    "capture_group_id": episode["capture_group_id"],
                    "method": method,
                    "seed_panel": episode["seed_panel"],
                    "training_seed": episode["training_seed"],
                    "run": episode["run"],
                    "action_index": action_index,
                    "actor_call_index": actor_call_index,
                    "decision_kind": decision_kind,
                    "cell": cell_index,
                    "cell_row": cell.row,
                    "cell_column": cell.column,
                    "row_start": cell.row_start,
                    "row_stop": cell.row_stop,
                    "col_start": cell.col_start,
                    "col_stop": cell.col_stop,
                    "pixel_count": cell.pixel_count,
                    "before_cost": costs[action_index],
                    "after_cost": costs[action_index + 1],
                    "visible_mask_before": _mask_text(before),
                    "visible_mask_after": _mask_text(measured),
                    "proposal_cells": ";".join(
                        str(value) for value in initial_candidates.tolist()
                    ),
                    "c0_reason": c0_reason,
                    "prediction_before_mpa": predictions[action_index],
                    "prediction_after_mpa": predictions[action_index + 1],
                    "target_mpa": float(episode["target_mpa"]),
                    "vlm_cache_key": cache_key,
                    "vlm_available": bool(features.available[index]),
                    "actor_checkpoint_sha256": (
                        "NOT_APPLICABLE_FIXED_ORDER"
                        if method in _FIXED_METHODS
                        else actor_hashes[(method, str(episode["seed_panel"]))]
                    ),
                    "predictor_checkpoint_sha256": predictor_hash,
                }
            )
    return trace_rows


def _fixed_hash_cases(episodes: list[dict[str, str]], *, scope: str) -> list[str]:
    keys_by_domain: dict[str, set[str]] = defaultdict(set)
    for row in episodes:
        if row["method"] == "VLM_SPATIAL_FEEDBACK" and row["seed_panel"] == "1":
            keys_by_domain[row["dataset_id"]].add(row["specimen_key"])
    selected = []
    for domain in sorted(keys_by_domain)[:3]:
        selected.append(
            min(
                keys_by_domain[domain],
                key=lambda key: hashlib.sha256(
                    f"cai-agent-v3-figure|{scope}|{domain}|{key}".encode("ascii")
                ).hexdigest(),
            )
        )
    return selected


def export_policy_diagnostics(
    *, project_root: str | Path, source_root: str | Path, scope: str = "VALID"
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    external = Path(source_root).resolve(strict=True)
    output = root / "results/cai_agent_v3/new_protocol"
    scope_upper = scope.upper()
    episodes = read_csv(
        output
        / (
            "policy_validation_episodes.csv"
            if scope_upper == "VALID"
            else "policy_test_episodes.csv"
        )
    )
    trace_rows = build_action_trace(project_root=root, scope=scope_upper)
    trace_path = output / f"policy_{scope_upper.lower()}_action_trace.csv"
    write_csv(trace_path, trace_rows)
    queue = {
        row["specimen_key"]: row for row in read_csv(output / "candidate_queue.csv")
    }
    feature_rows = {
        row["specimen_key"]: row
        for row in read_csv(_diagnostic_feature_path(output, scope_upper))
    }
    selected = _fixed_hash_cases(episodes, scope=scope_upper)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for key in selected:
        episode = next(
            row
            for row in episodes
            if row["specimen_key"] == key
            and row["method"] == "VLM_SPATIAL_FEEDBACK"
            and row["seed_panel"] == "1"
        )
        source = queue[key]
        surface_path = external / source["impacted_surface_path"]
        cscan_path = external / source["registered_cscan_crop_path"]
        if sha256_file(surface_path) != source["surface_sha256"]:
            raise ValueError("figure surface source hash changed")
        if sha256_file(cscan_path) != source["registered_cscan_crop_sha256"]:
            raise ValueError("figure C-scan source hash changed")
        with Image.open(surface_path) as image:
            surface = np.asarray(
                render_surface_inputs(image, max_edge=1024).clean, dtype=np.uint8
            )
        with Image.open(cscan_path) as image:
            cscan = np.asarray(image.convert("RGB"), dtype=np.uint8)
        grid = NativeCellGrid.from_shape(cscan.shape[:2])
        surface_grid = NativeCellGrid.from_shape(surface.shape[:2])
        cells = _values(episode["cells"], cast=int)
        costs = _values(episode["costs"])
        predictions = _values(episode["predictions_mpa"])
        vlm_cells = [
            index
            for index, value in enumerate(
                _values(feature_rows[key]["region_indicator"])
            )
            if value > 0.0
        ]
        figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
        axes[0, 0].imshow(surface)
        _draw_grid(axes[0, 0], surface.shape[:2])
        _draw_cells(axes[0, 0], surface_grid, vlm_cells, color="#00a6a6", fill=True)
        axes[0, 0].set_title("Registered surface + frozen VLM cues")
        axes[0, 1].imshow(surface)
        _draw_grid(axes[0, 1], surface.shape[:2])
        _draw_cells(axes[0, 1], surface_grid, [cells[0]], color="#d62728", fill=True)
        axes[0, 1].set_title(f"Actor start: cell {cells[0]}")
        counts = [min(value, len(cells)) for value in (1, 4, 8, len(cells))]
        for axis, count in zip(
            (axes[0, 2], axes[0, 3], axes[1, 0], axes[1, 1]),
            counts,
            strict=True,
        ):
            acquired = cells[:count]
            axis.imshow(_measured_image(cscan, grid, acquired))
            _draw_grid(axis, cscan.shape[:2], color="#555555")
            _draw_cells(axis, grid, acquired, color="#2ca02c", fill=False)
            if count < len(cells):
                _draw_cells(
                    axis,
                    grid,
                    [cells[count]],
                    color="#d62728",
                    fill=False,
                    linewidth=2.6,
                )
                suffix = f", next={cells[count]}"
            else:
                suffix = ", endpoint"
            axis.set_title(f"k={count}, cost={costs[count]:.4f}{suffix}")
        target = float(
            next(
                row["target_mpa"]
                for row in trace_rows
                if row["specimen_key"] == key
                and row["method"] == "VLM_SPATIAL_FEEDBACK"
                and str(row["seed_panel"]) == "1"
            )
        )
        axes[1, 2].step(costs, predictions, where="post", color="#d62728")
        axes[1, 2].axhline(target, color="black", linestyle="--", label="CAI target")
        axes[1, 2].set_xlim(0.0, 0.25)
        axes[1, 2].set_xlabel("Exact native-raster cost")
        axes[1, 2].set_ylabel("Predicted CAI (MPa)")
        axes[1, 2].set_title("Prediction trajectory")
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(alpha=0.2)
        axes[1, 3].imshow(cscan)
        _draw_grid(axes[1, 3], cscan.shape[:2])
        axes[1, 3].set_title("Full C-scan (post-hoc diagnostic only)")
        for axis in (*axes[0], axes[1, 0], axes[1, 1], axes[1, 3]):
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            f"CAI Agent v3 {scope_upper} fixed-hash case: {key}", fontsize=15
        )
        path = figure_dir / f"case_{scope_upper.lower()}_{key.replace(':', '_')}.png"
        figure.savefig(path, dpi=180, facecolor="white")
        plt.close(figure)
        with Image.open(path) as exported:
            pixel_size = list(exported.size)
        records.append(
            {
                "specimen_key": key,
                "dataset_id": source["dataset_id"],
                "split": scope_upper,
                "selection_rule": "minimum SHA256(cai-agent-v3-figure|scope|domain|specimen) in each of first three sorted domains",
                "file": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "pixel_size": pixel_size,
                "main_method": "VLM_SPATIAL_FEEDBACK",
                "seed_panel": 1,
                "full_cscan_panel_is_post_hoc": True,
            }
        )
    payload = {
        "schema_version": 3,
        "scope": scope_upper,
        "selection_uses_outcomes": False,
        "maximum_allowed_cases": 3,
        "case_count": len(records),
        "action_trace_path": trace_path.relative_to(root).as_posix(),
        "action_trace_sha256": sha256_file(trace_path),
        "records": records,
    }
    write_json(output / f"figure_manifest_{scope_upper.lower()}.json", payload)
    return payload


__all__ = ["build_action_trace", "export_policy_diagnostics"]
