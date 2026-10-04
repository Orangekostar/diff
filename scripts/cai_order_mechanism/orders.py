"""Deterministic fixed-set orders, native clocks, and cache identities."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scripts.cai_order_mechanism.analysis import Episode, parse_episode
from scripts.cai_order_mechanism.inputs import (
    MAIN_METHOD,
    TaskContext,
    atomic_json,
    atomic_text,
    canonical_json,
    read_csv_rows,
    read_gzip_csv_rows,
    sha256_bytes,
    sha256_file,
    write_csv_atomic,
)

if TYPE_CHECKING:
    from cmc_bbdm.cai_active_image.environment import NativeCellGrid

NAMESPACE = "CAI_ORDER_MECHANISM_R1_37B3C404|PERMUTED_V1"


@dataclass(frozen=True, slots=True)
class OrderRecord:
    specimen_key: str
    dataset_id: str
    capture_group_id: str
    source_method: str
    source_run: int
    original_order: tuple[int, ...]
    order: tuple[int, ...]
    variant: str
    repeat: int
    final_mask_bits: int
    action_count: int


@dataclass(frozen=True, slots=True)
class PrefixState:
    step: int
    acquired_cell: int | None
    mask_bits: int
    pixels: int
    cost64: float
    cost32: np.float32

    @property
    def measured_mask(self) -> np.ndarray:
        return np.asarray(
            [bool(self.mask_bits & (1 << cell)) for cell in range(64)], dtype=bool
        )


@dataclass(frozen=True, slots=True)
class CacheBinding:
    protocol_sha256: str
    predictor_sha256: str
    feature_shard_sha256: str
    engine_signature: str


def fixed_permutation(
    specimen_key: str,
    repeat: int,
    cells: Sequence[int],
    *,
    namespace: str = NAMESPACE,
) -> tuple[int, ...]:
    unique_cells = sorted(int(cell) for cell in cells)
    if len(unique_cells) != len(set(unique_cells)):
        raise ValueError("permutation cells must be unique")
    if any(cell < 0 or cell >= 64 for cell in unique_cells):
        raise ValueError("permutation cells must be in [0, 63]")
    if not 0 <= repeat < 5:
        raise ValueError("permutation repeat must be in [0, 4]")
    return tuple(
        sorted(
            unique_cells,
            key=lambda cell: (
                hashlib.sha256(
                    f"{namespace}|{specimen_key}|{repeat}|{cell}".encode()
                ).hexdigest(),
                cell,
            ),
        )
    )


def build_order_plan(
    episodes: Sequence[Episode], *, namespace: str = NAMESPACE
) -> list[OrderRecord]:
    records: list[OrderRecord] = []
    for episode in sorted(episodes, key=lambda value: value.specimen_key):
        native = episode.cells
        final_mask = sum(1 << cell for cell in native)
        variants = [
            ("NATIVE_REPLAY", 0, native),
            ("REVERSE", 0, tuple(reversed(native))),
            *(
                (
                    "PERMUTED",
                    repeat,
                    fixed_permutation(
                        episode.specimen_key, repeat, native, namespace=namespace
                    ),
                )
                for repeat in range(5)
            ),
        ]
        for variant, repeat, order in variants:
            if set(order) != set(native) or len(order) != len(native):
                raise ValueError("order changed the fixed terminal set")
            records.append(
                OrderRecord(
                    specimen_key=episode.specimen_key,
                    dataset_id=episode.dataset_id,
                    capture_group_id=episode.capture_group_id,
                    source_method=episode.method,
                    source_run=episode.run,
                    original_order=native,
                    order=tuple(order),
                    variant=variant,
                    repeat=repeat,
                    final_mask_bits=final_mask,
                    action_count=len(native),
                )
            )
    return records


def prefix_states(order: Sequence[int], grid: NativeCellGrid) -> list[PrefixState]:
    if len(order) != len(set(order)) or any(cell < 0 or cell >= 64 for cell in order):
        raise ValueError("prefix order must contain unique cell indices")
    total_pixels = int(np.prod(grid.native_shape, dtype=np.int64))
    states = [
        PrefixState(
            step=0,
            acquired_cell=None,
            mask_bits=0,
            pixels=0,
            cost64=0.0,
            cost32=np.float32(0.0),
        )
    ]
    mask_bits = 0
    pixels = 0
    for step, cell in enumerate(order, start=1):
        mask_bits |= 1 << int(cell)
        pixels += int(grid.cells[int(cell)].pixel_count)
        cost64 = float(pixels / total_pixels)
        states.append(
            PrefixState(
                step=step,
                acquired_cell=int(cell),
                mask_bits=mask_bits,
                pixels=pixels,
                cost64=cost64,
                cost32=np.float32(cost64),
            )
        )
    return states


def prefix_cache_key(
    binding: CacheBinding,
    specimen_key: str,
    mask_bits: int,
    cost32: np.float32,
) -> str:
    if not 0 <= int(mask_bits) < 1 << 64:
        raise ValueError("mask must be an unsigned 64-bit value")
    digest = hashlib.sha256()
    for value in (
        binding.protocol_sha256,
        binding.predictor_sha256,
        binding.feature_shard_sha256,
        binding.engine_signature,
        specimen_key,
    ):
        encoded = value.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    digest.update(struct.pack("<Q", int(mask_bits)))
    digest.update(np.asarray(cost32, dtype="<f4").tobytes())
    return digest.hexdigest()


def plan_orders(context: TaskContext) -> list[Path]:
    results = context.results_root
    archive_rows = [
        row
        for row in read_gzip_csv_rows(results / "archived_cohort_episodes.csv.gz")
        if row["method"] == MAIN_METHOD
    ]
    if len(archive_rows) != 50:
        raise ValueError("order plan requires exactly 50 Proposed trajectories")
    episodes = [parse_episode(row) for row in archive_rows]
    source_by_key = {row["specimen_key"]: row for row in archive_rows}
    cohort = {
        row["specimen_key"]: row
        for row in read_csv_rows(results / "cohort_manifest.csv")
    }
    plan = build_order_plan(episodes, namespace=context.scope["intervention"]["hash_namespace"])
    if len(plan) != 350:
        raise ValueError("order plan must contain exactly 350 scenarios")
    records = []
    nominal_prefix_records = 0
    clock_rows = []
    for row in plan:
        metadata = cohort[row.specimen_key]
        from cmc_bbdm.cai_active_image.environment import NativeCellGrid

        grid = NativeCellGrid.from_shape(
            (int(metadata["native_height"]), int(metadata["native_width"]))
        )
        states = prefix_states(row.order, grid)
        native_states = prefix_states(row.original_order, grid)
        total_pixels = int(metadata["total_pixels"])
        final = states[-1]
        if final.mask_bits != row.final_mask_bits:
            raise ValueError("planned order final mask changed")
        if 4 * final.pixels > total_pixels:
            raise ValueError("fixed final set exceeds the 0.25 pixel budget")
        nominal_prefix_records += len(states)
        source = source_by_key[row.specimen_key]
        records.append(
            {
                "order_id": f"{row.specimen_key}|{row.variant}|{row.repeat}",
                "identity_family": "FIXED_SET_ORDERS",
                "cohort": "VALID50_CAPTURE48_DOMAIN6",
                "clock": "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL",
                "specimen_key": row.specimen_key,
                "dataset_id": row.dataset_id,
                "capture_group_id": row.capture_group_id,
                "source_method": row.source_method,
                "source_run": row.source_run,
                "source_checkpoint_update": int(source["checkpoint_update"]),
                "source_actor_state_dict_sha256": source["actor_state_dict_sha256"],
                "variant": row.variant,
                "variant_display_label": {
                    "NATIVE_REPLAY": "Native order",
                    "REVERSE": "Reverse order",
                    "PERMUTED": f"Permuted {row.repeat}",
                }[row.variant],
                "repeat": row.repeat,
                "original_order": list(row.original_order),
                "order": list(row.order),
                "sorted_final_set": sorted(row.order),
                "final_set_sha256": sha256_bytes(
                    canonical_json(sorted(row.order))
                ),
                "final_mask_bits": row.final_mask_bits,
                "final_mask_hex": f"{row.final_mask_bits:016x}",
                "action_count": row.action_count,
                "final_pixels": final.pixels,
                "total_pixels": total_pixels,
                "final_cost64": final.cost64,
                "final_cost32_hex": np.asarray(final.cost32, dtype="<f4")
                .tobytes()
                .hex(),
                "label_inputs_used": False,
            }
        )
        if row.variant == "NATIVE_REPLAY":
            episode = next(
                episode
                for episode in episodes
                if episode.specimen_key == row.specimen_key
            )
            new_costs = np.asarray([state.cost64 for state in native_states])
            if not np.allclose(new_costs, episode.costs, rtol=0, atol=1e-12):
                raise ValueError("integer native clock differs from archived cost beyond tolerance")
            for cap in (0.0, 0.0625, 0.125, 0.1875, 0.25):
                archive_step = int(np.searchsorted(episode.costs, cap, side="right") - 1)
                integer_step = int(np.searchsorted(new_costs, cap, side="right") - 1)
                clock_rows.append(
                    {
                        "identity_family": "CLOCK_BOUNDARY_AUDIT",
                        "cohort": "VALID50_CAPTURE48_DOMAIN6",
                        "specimen_key": row.specimen_key,
                        "cap": cap,
                        "archived_clock": "ARCHIVED_FLOAT64",
                        "integer_clock": (
                            "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL"
                        ),
                        "archived_step": archive_step,
                        "integer_step": integer_step,
                        "archived_selected_cost": float(episode.costs[archive_step]),
                        "integer_selected_cost": float(new_costs[integer_step]),
                        "step_membership_changed": archive_step != integer_step,
                        "max_state_cost_abs_difference": float(
                            np.max(np.abs(episode.costs - new_costs))
                        ),
                        "units": "cost=fraction",
                    }
                )
    if nominal_prefix_records > context.scope["intervention"]["max_nominal_prefix_records"]:
        raise ValueError("nominal prefix request count exceeds the fixed limit")
    payload = {
        "schema_version": 1,
        "task_id": context.scope["task_id"],
        "namespace": context.scope["intervention"]["hash_namespace"],
        "plan_rule": context.scope["intervention"]["plan_rule"],
        "label_inputs_used": False,
        "scenario_count": len(records),
        "reordered_scenario_count": sum(
            row["variant"] != "NATIVE_REPLAY" for row in records
        ),
        "nominal_prefix_request_count": nominal_prefix_records,
        "records": records,
    }
    plan_path = atomic_json(results / "order_plan.json", payload)
    hash_path = atomic_text(
        results / "order_plan.sha256",
        f"{sha256_file(plan_path)}  order_plan.json\n",
    )
    return [
        plan_path,
        hash_path,
        write_csv_atomic(results / "clock_boundary_audit.csv", clock_rows),
    ]


__all__ = [
    "NAMESPACE",
    "CacheBinding",
    "OrderRecord",
    "PrefixState",
    "build_order_plan",
    "fixed_permutation",
    "plan_orders",
    "prefix_cache_key",
    "prefix_states",
]
