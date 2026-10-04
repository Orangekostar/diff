"""Bounded predictor replay with durable resource accounting and cache."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cmc_bbdm.cai_agent_v3.metrics import (
    left_error_area_mpa,
    trajectory_objective_mpa,
)
from cmc_bbdm.cai_agent_v3.predictor_training import load_predictor_checkpoint
from scripts.cai_order_mechanism.analysis import parse_episode
from scripts.cai_order_mechanism.inputs import (
    TaskContext,
    atomic_json,
    canonical_json,
    load_valid_features,
    read_gzip_csv_rows,
    sha256_bytes,
    sha256_file,
    write_csv_atomic,
    write_gzip_csv_atomic,
)
from scripts.cai_order_mechanism.orders import (
    CacheBinding,
    prefix_cache_key,
    prefix_states,
)


def read_resource_events(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]


class ResourceLedger:
    def __init__(self, path: str | Path, *, evaluated_row_cap: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.evaluated_row_cap = int(evaluated_row_cap)
        events = read_resource_events(self.path)
        self._charged = sum(
            int(row["evaluated_rows"])
            for row in events
            if row.get("state") == "INTENDED"
        )
        self._next_batch_id = 1 + max(
            (int(row.get("batch_id", 0)) for row in events), default=0
        )

    @property
    def charged_rows(self) -> int:
        return self._charged

    def _append(self, row: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def begin(self, evaluated_rows: int, *, kind: str) -> int:
        rows = int(evaluated_rows)
        if rows <= 0:
            raise ValueError("evaluated rows must be positive")
        if self._charged + rows > self.evaluated_row_cap:
            raise ValueError(
                f"predictor evaluated-row cap exceeded: "
                f"{self._charged + rows}>{self.evaluated_row_cap}"
            )
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        self._charged += rows
        self._append(
            {
                "batch_id": batch_id,
                "state": "INTENDED",
                "kind": kind,
                "synthetic": kind.startswith("SYNTHETIC"),
                "evaluated_rows": rows,
                "charged_total_rows": self._charged,
            }
        )
        return batch_id

    def complete(self, batch_id: int, evaluated_rows: int, *, kind: str) -> None:
        self._append(
            {
                "batch_id": int(batch_id),
                "state": "COMPLETE",
                "kind": kind,
                "synthetic": kind.startswith("SYNTHETIC"),
                "evaluated_rows": int(evaluated_rows),
                "charged_total_rows": self._charged,
            }
        )


@dataclass(frozen=True, slots=True)
class PrefixRequest:
    cache_key: str
    specimen_key: str
    feature_shard_sha256: str
    surface_tokens: np.ndarray
    cscan_tokens: np.ndarray
    measured_mask: np.ndarray
    mask_bits: int
    cost32: np.float32


@dataclass(frozen=True, slots=True)
class CacheEntry:
    prediction_mpa: float
    specimen_key: str
    feature_shard_sha256: str
    mask_bits: int
    cost32: np.float32


class PredictorCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, CacheEntry] = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as archive:
                for index, key in enumerate(archive["keys"]):
                    self.entries[str(key)] = CacheEntry(
                        prediction_mpa=float(archive["predictions_mpa"][index]),
                        specimen_key=str(archive["specimen_keys"][index]),
                        feature_shard_sha256=str(archive["feature_shard_sha256"][index]),
                        mask_bits=int(archive["mask_bits"][index]),
                        cost32=np.float32(archive["cost32"][index]),
                    )

    def get(self, key: str) -> float | None:
        entry = self.entries.get(key)
        return None if entry is None else entry.prediction_mpa

    def put(self, request: PrefixRequest, prediction_mpa: float) -> None:
        new = CacheEntry(
            prediction_mpa=float(prediction_mpa),
            specimen_key=request.specimen_key,
            feature_shard_sha256=request.feature_shard_sha256,
            mask_bits=int(request.mask_bits),
            cost32=np.float32(request.cost32),
        )
        old = self.entries.get(request.cache_key)
        if old is not None and old != new:
            raise ValueError("cache key collision with different scientific state")
        self.entries[request.cache_key] = new

    def remove(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.entries.pop(key, None)
        if self.entries:
            self.save_atomic()
        elif self.path.exists():
            self.path.unlink()

    def save_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        keys = sorted(self.entries)
        values = [self.entries[key] for key in keys]
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                keys=np.asarray(keys),
                predictions_mpa=np.asarray(
                    [value.prediction_mpa for value in values], dtype=np.float32
                ),
                specimen_keys=np.asarray([value.specimen_key for value in values]),
                feature_shard_sha256=np.asarray(
                    [value.feature_shard_sha256 for value in values]
                ),
                mask_bits=np.asarray([value.mask_bits for value in values], dtype=np.uint64),
                cost32=np.asarray([value.cost32 for value in values], dtype=np.float32),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)


class ReplayEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        cache: PredictorCache,
        ledger: ResourceLedger,
        *,
        device: torch.device,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.model = model.to(device).eval()
        self.cache = cache
        self.ledger = ledger
        self.device = device
        self.batch_size = int(batch_size)
        self.forward_calls = 0
        self.request_count = 0
        self.cache_hits = 0

    def evaluate(
        self, requests: Sequence[PrefixRequest], *, kind: str = "RESEARCH_PREFIX"
    ) -> np.ndarray:
        self.request_count += len(requests)
        missing: dict[str, PrefixRequest] = {}
        for request in requests:
            if self.cache.get(request.cache_key) is None:
                missing.setdefault(request.cache_key, request)
            else:
                self.cache_hits += 1
        pending = list(missing.values())
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            batch_id = self.ledger.begin(len(batch), kind=kind)
            surface = torch.from_numpy(
                np.stack([request.surface_tokens for request in batch])
            ).to(self.device, dtype=torch.float32)
            cscan = torch.from_numpy(
                np.stack([request.cscan_tokens for request in batch])
            ).to(self.device, dtype=torch.float32)
            measured = torch.from_numpy(
                np.stack([request.measured_mask for request in batch])
            ).to(self.device, dtype=torch.bool)
            cost = torch.from_numpy(
                np.asarray([request.cost32 for request in batch], dtype=np.float32)
            ).to(self.device)
            with torch.inference_mode():
                output = self.model(surface, cscan, measured, cost=cost)
            prediction = np.asarray(
                output.detach().to("cpu", dtype=torch.float32).numpy()
            ).reshape(-1)
            if len(prediction) != len(batch) or not np.isfinite(prediction).all():
                raise ValueError("predictor returned invalid batch output")
            self.forward_calls += 1
            for request, value in zip(batch, prediction, strict=True):
                self.cache.put(request, float(value))
            self.cache.save_atomic()
            self.ledger.complete(batch_id, len(batch), kind=kind)
        return np.asarray(
            [self.cache.entries[request.cache_key].prediction_mpa for request in requests],
            dtype=np.float64,
        )

    def evaluate_uncached(
        self, requests: Sequence[PrefixRequest], *, kind: str
    ) -> np.ndarray:
        predictions = []
        for start in range(0, len(requests), self.batch_size):
            batch = list(requests[start : start + self.batch_size])
            batch_id = self.ledger.begin(len(batch), kind=kind)
            surface = torch.from_numpy(
                np.stack([request.surface_tokens for request in batch])
            ).to(self.device, dtype=torch.float32)
            cscan = torch.from_numpy(
                np.stack([request.cscan_tokens for request in batch])
            ).to(self.device, dtype=torch.float32)
            measured = torch.from_numpy(
                np.stack([request.measured_mask for request in batch])
            ).to(self.device, dtype=torch.bool)
            cost = torch.from_numpy(
                np.asarray([request.cost32 for request in batch], dtype=np.float32)
            ).to(self.device)
            with torch.inference_mode():
                output = self.model(surface, cscan, measured, cost=cost)
            values = np.asarray(
                output.detach().to("cpu", dtype=torch.float32).numpy()
            ).reshape(-1)
            if len(values) != len(batch) or not np.isfinite(values).all():
                raise ValueError("predictor returned invalid uncached output")
            predictions.extend(float(value) for value in values)
            self.forward_calls += 1
            self.ledger.complete(batch_id, len(batch), kind=kind)
        return np.asarray(predictions, dtype=np.float64)


def _engine_identity(device: torch.device, batch_size: int) -> tuple[str, dict[str, object]]:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = "CPU"
    payload = {
        "device": str(device),
        "physical_gpu_index": os.environ.get("CAI_ORDER_PHYSICAL_GPU"),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": "float32",
        "batch_size": int(batch_size),
        "amp": False,
        "tf32": False,
        "inference_mode": True,
    }
    return sha256_bytes(canonical_json(payload)), payload


def _requests_for_record(
    record: dict[str, object], feature, binding: CacheBinding
) -> tuple[list[object], list[PrefixRequest]]:
    states = prefix_states(tuple(int(value) for value in record["order"]), feature.grid)
    requests = [
        PrefixRequest(
            cache_key=prefix_cache_key(
                binding, feature.specimen_key, state.mask_bits, state.cost32
            ),
            specimen_key=feature.specimen_key,
            feature_shard_sha256=feature.shard_sha256,
            surface_tokens=feature.surface_tokens,
            cscan_tokens=feature.cscan_tokens,
            measured_mask=state.measured_mask,
            mask_bits=state.mask_bits,
            cost32=state.cost32,
        )
        for state in states
    ]
    return states, requests


def _trial_native_reproduction(
    engine: ReplayEngine,
    records: list[dict[str, object]],
    features: dict[str, object],
    binding: CacheBinding,
    archived: dict[str, object],
    case_keys: Sequence[str],
) -> tuple[bool, float, float, list[str]]:
    maximum_prediction_difference = 0.0
    maximum_cost_difference = 0.0
    cache_keys = []
    for key in case_keys:
        record = next(
            row
            for row in records
            if row["specimen_key"] == key and row["variant"] == "NATIVE_REPLAY"
        )
        feature = features[key]
        per_binding = CacheBinding(
            binding.protocol_sha256,
            binding.predictor_sha256,
            feature.shard_sha256,
            binding.engine_signature,
        )
        states, requests = _requests_for_record(record, feature, per_binding)
        values = engine.evaluate(requests, kind="NATIVE_REPRODUCTION_TRIAL")
        source = archived[key]
        new_costs = np.asarray([state.cost64 for state in states])
        maximum_cost_difference = max(
            maximum_cost_difference,
            float(np.max(np.abs(new_costs - source.costs))),
        )
        maximum_prediction_difference = max(
            maximum_prediction_difference,
            float(np.max(np.abs(values - source.predictions_mpa))),
        )
        cache_keys.extend(request.cache_key for request in requests)
    passed = maximum_cost_difference <= 1e-12 and maximum_prediction_difference <= 1e-4
    return passed, maximum_prediction_difference, maximum_cost_difference, cache_keys


def infer_orders(context: TaskContext) -> list[Path]:
    started = time.monotonic()
    results = context.results_root
    plan_path = results / "order_plan.json"
    expected_plan_hash = (results / "order_plan.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if sha256_file(plan_path) != expected_plan_hash:
        raise ValueError("order plan changed after it was locked")
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    records = plan_payload["records"]
    if plan_payload["scenario_count"] != 350 or len(records) != 350:
        raise ValueError("locked order plan scenario count changed")
    keys = sorted({str(row["specimen_key"]) for row in records})
    bindings = json.loads((results / "input_bindings.json").read_text(encoding="utf-8"))
    shard_hashes = {
        row["shard_path"]: row["sha256"] for row in bindings["feature_shards"]
    }
    features = load_valid_features(context, keys, shard_hashes=shard_hashes)
    archive_rows = [
        row
        for row in read_gzip_csv_rows(results / "archived_cohort_episodes.csv.gz")
        if row["method"] == "NO_VLM_SPATIAL_FEEDBACK"
    ]
    archived = {row["specimen_key"]: parse_episode(row) for row in archive_rows}

    device_name = os.environ.get("CAI_ORDER_DEVICE", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    predictor_path = context.source("frozen_models.predictor")
    model, manifest = load_predictor_checkpoint(predictor_path, device=str(device))
    predictor_scope = context.scope["frozen_models"]["predictor"]
    if (
        manifest.get("model") != "MEAN_SC"
        or int(manifest.get("selected_update", -1)) != 1750
        or sha256_file(predictor_path) != predictor_scope["sha256"]
    ):
        raise ValueError("loaded predictor differs from the frozen selection")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    cache = PredictorCache(results / "predictor_cache.npz")
    ledger = ResourceLedger(
        results / "resource_events.jsonl",
        evaluated_row_cap=context.scope["resource_limits"][
            "predictor_evaluated_state_rows_including_padding_retries_qa"
        ],
    )
    batch_size = int(context.scope["numerics"]["default_batch_size"])
    engine_signature, engine_details = _engine_identity(device, batch_size)
    binding = CacheBinding(
        context.scope_sha256,
        predictor_scope["sha256"],
        "PER_REQUEST_SHARD_HASH",
        engine_signature,
    )
    engine = ReplayEngine(
        model, cache, ledger, device=device, batch_size=batch_size
    )
    trial_passed, trial_prediction_max, trial_cost_max, trial_cache_keys = (
        _trial_native_reproduction(
            engine,
            records,
            features,
            binding,
            archived,
            context.scope["case_keys"],
        )
    )
    fallback_used = False
    if not trial_passed:
        cache.remove(trial_cache_keys)
        fallback_used = True
        batch_size = 1
        engine_signature, engine_details = _engine_identity(device, batch_size)
        binding = CacheBinding(
            context.scope_sha256,
            predictor_scope["sha256"],
            "PER_REQUEST_SHARD_HASH",
            engine_signature,
        )
        engine = ReplayEngine(
            model, cache, ledger, device=device, batch_size=batch_size
        )
        trial_passed, trial_prediction_max, trial_cost_max, _ = (
            _trial_native_reproduction(
                engine,
                records,
                features,
                binding,
                archived,
                context.scope["case_keys"],
            )
        )
    if not trial_passed:
        raise RuntimeError("BLOCKED_PREDICTOR_REPRODUCTION")

    all_requests: list[PrefixRequest] = []
    record_slices: dict[str, tuple[int, int, list[object]]] = {}
    prefix_rows = []
    seen_cache_keys: set[str] = set()
    for record in records:
        feature = features[record["specimen_key"]]
        per_binding = CacheBinding(
            binding.protocol_sha256,
            binding.predictor_sha256,
            feature.shard_sha256,
            binding.engine_signature,
        )
        states, requests = _requests_for_record(record, feature, per_binding)
        start = len(all_requests)
        all_requests.extend(requests)
        stop = len(all_requests)
        record_slices[record["order_id"]] = (start, stop, states)
        for state, request in zip(states, requests, strict=True):
            first = request.cache_key not in seen_cache_keys
            seen_cache_keys.add(request.cache_key)
            prefix_rows.append(
                {
                    "identity_family": "FIXED_SET_ORDERS",
                    "cohort": "VALID50_CAPTURE48_DOMAIN6",
                    "clock": (
                        "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL"
                    ),
                    "order_id": record["order_id"],
                    "specimen_key": record["specimen_key"],
                    "dataset_id": record["dataset_id"],
                    "variant": record["variant"],
                    "variant_display_label": record["variant_display_label"],
                    "repeat": record["repeat"],
                    "step": state.step,
                    "acquired_cell": "" if state.acquired_cell is None else state.acquired_cell,
                    "mask_bits": state.mask_bits,
                    "mask_hex": f"{state.mask_bits:016x}",
                    "pixels": state.pixels,
                    "cost64": state.cost64,
                    "cost32_hex": np.asarray(state.cost32, dtype="<f4")
                    .tobytes()
                    .hex(),
                    "cache_key": request.cache_key,
                    "first_request_for_cache_key": first,
                    "units": "pixels=count; cost=fraction",
                }
            )
    if len(all_requests) != plan_payload["nominal_prefix_request_count"]:
        raise ValueError("prefix request map differs from the locked plan")
    predictions = engine.evaluate(all_requests, kind="ORDER_PREFIX_INFERENCE")

    trajectory_rows = []
    native_rows = []
    endpoint_by_key: dict[str, dict[str, object]] = defaultdict(dict)
    maximum_native_prediction_difference = 0.0
    maximum_native_cost_difference = 0.0
    maximum_native_area_difference = 0.0
    for record in records:
        start, stop, states = record_slices[record["order_id"]]
        values = predictions[start:stop]
        costs = np.asarray([state.cost64 for state in states], dtype=np.float64)
        feature = features[record["specimen_key"]]
        area = left_error_area_mpa(costs, values, feature.target_mpa, end=0.25)
        early = left_error_area_mpa(costs, values, feature.target_mpa, end=0.0625)
        final_error = abs(float(values[-1]) - feature.target_mpa)
        objective = trajectory_objective_mpa(
            costs, values, feature.target_mpa, budget=0.25, terminal_weight=0.25
        )
        trajectory_rows.append(
            {
                "identity_family": "FIXED_SET_ORDERS",
                "cohort": "VALID50_CAPTURE48_DOMAIN6",
                "clock": "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL",
                "order_id": record["order_id"],
                "specimen_key": record["specimen_key"],
                "dataset_id": record["dataset_id"],
                "capture_group_id": record["capture_group_id"],
                "variant": record["variant"],
                "variant_display_label": record["variant_display_label"],
                "repeat": record["repeat"],
                "target_mpa": feature.target_mpa,
                "left_error_area_mpa": area,
                "early_left_error_area_mpa": early,
                "trajectory_objective_mpa": objective,
                "final_error_mpa": final_error,
                "action_count": len(record["order"]),
                "final_pixels": states[-1].pixels,
                "final_cost": costs[-1],
                "cells": ";".join(str(value) for value in record["order"]),
                "costs": ";".join(repr(float(value)) for value in costs),
                "predictions_mpa": ";".join(repr(float(value)) for value in values),
                "difference_definition": (
                    "ORDER_EFFECT_ROWS_SCORED_AFTER_LABEL_FREE_PLAN_LOCK"
                ),
                "units": "area,error,objective=MPa; cost=fraction; pixels=count",
            }
        )
        endpoint_by_key[record["specimen_key"]][record["order_id"]] = {
            "prediction": float(values[-1]),
            "mask": states[-1].mask_bits,
            "pixels": states[-1].pixels,
            "cost32": states[-1].cost32,
            "variant": record["variant"],
            "repeat": record["repeat"],
        }
        if record["variant"] == "NATIVE_REPLAY":
            source = archived[record["specimen_key"]]
            cost_difference = np.abs(costs - source.costs)
            prediction_difference = np.abs(values - source.predictions_mpa)
            maximum_native_cost_difference = max(
                maximum_native_cost_difference, float(cost_difference.max())
            )
            maximum_native_prediction_difference = max(
                maximum_native_prediction_difference,
                float(prediction_difference.max()),
            )
            archived_area = left_error_area_mpa(
                source.costs,
                source.predictions_mpa,
                source.target_mpa,
                end=0.25,
            )
            area_difference = abs(area - archived_area)
            maximum_native_area_difference = max(
                maximum_native_area_difference, area_difference
            )
            for step, (state, prediction) in enumerate(
                zip(states, values, strict=True)
            ):
                native_rows.append(
                    {
                        "identity_family": "NATIVE_REPRODUCTION",
                        "cohort": "VALID50_CAPTURE48_DOMAIN6",
                        "clock": (
                            "ARCHIVED_FLOAT64_VS_INTEGER_CUMULATIVE_PIXELS"
                        ),
                        "record_level": "STATE",
                        "specimen_key": record["specimen_key"],
                        "dataset_id": record["dataset_id"],
                        "step": step,
                        "archived_cost": float(source.costs[step]),
                        "recomputed_cost": state.cost64,
                        "cost_abs_difference": float(cost_difference[step]),
                        "archived_prediction_mpa": float(
                            source.predictions_mpa[step]
                        ),
                        "recomputed_prediction_mpa": float(prediction),
                        "prediction_abs_difference_mpa": float(
                            prediction_difference[step]
                        ),
                        "passed": bool(
                            cost_difference[step] <= 1e-12
                            and prediction_difference[step] <= 1e-4
                        ),
                        "units": "cost=fraction; prediction=MPa",
                    }
                )
            native_rows.append(
                {
                    "identity_family": "NATIVE_REPRODUCTION",
                    "cohort": "VALID50_CAPTURE48_DOMAIN6",
                    "clock": "ARCHIVED_FLOAT64_VS_INTEGER_CUMULATIVE_PIXELS",
                    "record_level": "TRAJECTORY",
                    "specimen_key": record["specimen_key"],
                    "dataset_id": record["dataset_id"],
                    "step": "ALL",
                    "archived_cost": float(source.costs[-1]),
                    "recomputed_cost": float(costs[-1]),
                    "cost_abs_difference": float(cost_difference.max()),
                    "archived_prediction_mpa": float(source.predictions_mpa[-1]),
                    "recomputed_prediction_mpa": float(values[-1]),
                    "prediction_abs_difference_mpa": float(
                        prediction_difference.max()
                    ),
                    "archived_area_mpa": archived_area,
                    "recomputed_area_mpa": area,
                    "area_abs_difference_mpa": area_difference,
                    "passed": bool(
                        cost_difference.max() <= 1e-12
                        and prediction_difference.max() <= 1e-4
                        and area_difference <= 1e-4
                    ),
                    "units": "cost=fraction; prediction,area=MPa",
                }
            )
    if (
        maximum_native_cost_difference > 1e-12
        or maximum_native_prediction_difference > 1e-4
        or maximum_native_area_difference > 1e-4
    ):
        raise RuntimeError("BLOCKED_PREDICTOR_REPRODUCTION")

    endpoint_rows = []
    for key in keys:
        values = endpoint_by_key[key]
        native = next(row for row in values.values() if row["variant"] == "NATIVE_REPLAY")
        for order_id, row in values.items():
            mask_same = row["mask"] == native["mask"]
            pixels_same = row["pixels"] == native["pixels"]
            cost_same = np.asarray(row["cost32"], dtype="<f4").tobytes() == np.asarray(
                native["cost32"], dtype="<f4"
            ).tobytes()
            input_same = mask_same and pixels_same and cost_same
            prediction_difference = float(row["prediction"] - native["prediction"])
            endpoint_rows.append(
                {
                    "identity_family": "FIXED_SET_ORDERS",
                    "cohort": "VALID50_CAPTURE48_DOMAIN6",
                    "clock": "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL",
                    "order_id": order_id,
                    "specimen_key": key,
                    "dataset_id": features[key].dataset_id,
                    "variant": row["variant"],
                    "repeat": row["repeat"],
                    "final_mask_equal_native": mask_same,
                    "final_pixels_equal_native": pixels_same,
                    "final_cost32_equal_native": cost_same,
                    "endpoint_prediction_mpa": row["prediction"],
                    "native_endpoint_prediction_mpa": native["prediction"],
                    "prediction_difference_variant_minus_native_mpa": (
                        prediction_difference
                    ),
                    "passed": input_same and abs(prediction_difference) <= 1e-4,
                    "difference_definition": "VARIANT_MINUS_NATIVE_ENDPOINT",
                    "units": "prediction=MPa",
                }
            )
    if not all(row["passed"] for row in endpoint_rows):
        raise ValueError("cached endpoint invariance failed")

    qa_keys = []
    for domain in sorted({features[key].dataset_id for key in keys}):
        local = [key for key in keys if features[key].dataset_id == domain]
        qa_keys.append(
            min(
                local,
                key=lambda key: sha256_bytes(f"ENDPOINT_QA|{key}".encode()),
            )
        )
    qa_requests = []
    qa_metadata = []
    for key in qa_keys:
        for variant in ("NATIVE_REPLAY", "REVERSE"):
            record = next(
                row
                for row in records
                if row["specimen_key"] == key and row["variant"] == variant
            )
            feature = features[key]
            per_binding = CacheBinding(
                binding.protocol_sha256,
                binding.predictor_sha256,
                feature.shard_sha256,
                binding.engine_signature,
            )
            states, requests = _requests_for_record(record, feature, per_binding)
            qa_requests.append(requests[-1])
            qa_metadata.append((key, variant, states[-1]))
    qa_predictions = engine.evaluate_uncached(
        qa_requests, kind="ENDPOINT_UNCACHED_QA"
    )
    qa_rows = []
    for index in range(0, len(qa_metadata), 2):
        native_meta = qa_metadata[index]
        reverse_meta = qa_metadata[index + 1]
        input_same = (
            native_meta[2].mask_bits == reverse_meta[2].mask_bits
            and native_meta[2].pixels == reverse_meta[2].pixels
            and np.asarray(native_meta[2].cost32, dtype="<f4").tobytes()
            == np.asarray(reverse_meta[2].cost32, dtype="<f4").tobytes()
        )
        difference = float(qa_predictions[index + 1] - qa_predictions[index])
        if not input_same or abs(difference) > 1e-4:
            raise ValueError("uncached endpoint QA failed")
        for offset in (0, 1):
            key, variant, state = qa_metadata[index + offset]
            qa_rows.append(
                {
                    "identity_family": "ENDPOINT_UNCACHED_QA",
                    "cohort": "VALID50_CAPTURE48_DOMAIN6",
                    "clock": "INTEGER_CUMULATIVE_NATIVE_PIXELS_DIVIDED_BY_TOTAL",
                    "specimen_key": key,
                    "dataset_id": features[key].dataset_id,
                    "selection_hash": sha256_bytes(f"ENDPOINT_QA|{key}".encode()),
                    "variant": variant,
                    "cache_bypassed": True,
                    "mask_bits": state.mask_bits,
                    "pixels": state.pixels,
                    "cost32_hex": np.asarray(state.cost32, dtype="<f4")
                    .tobytes()
                    .hex(),
                    "prediction_mpa": float(qa_predictions[index + offset]),
                    "reverse_minus_native_mpa": difference,
                    "input_same_with_pair": input_same,
                    "passed": True,
                    "units": "prediction=MPa; pixels=count; cost=float32 bytes",
                }
            )

    events = read_resource_events(ledger.path)
    intended = [row for row in events if row["state"] == "INTENDED"]
    complete = [row for row in events if row["state"] == "COMPLETE"]
    intended_rows = sum(int(row["evaluated_rows"]) for row in intended)
    completed_rows = sum(int(row["evaluated_rows"]) for row in complete)
    if intended_rows != completed_rows or intended_rows > 7000:
        raise ValueError("resource ledger is incomplete or exceeds the row limit")
    elapsed = time.monotonic() - started
    time_cap = (
        context.scope["resource_limits"]["gpu_session_seconds"]
        if device.type == "cuda"
        else context.scope["resource_limits"]["cpu_process_seconds_including_report"]
    )
    if elapsed > time_cap:
        raise ValueError("inference time cap exceeded")
    usage = {
        "task_id": context.scope["task_id"],
        "device": str(device),
        "physical_gpu_index": os.environ.get("CAI_ORDER_PHYSICAL_GPU"),
        "engine_signature": engine_signature,
        "engine_details": engine_details,
        "fallback_used": fallback_used,
        "trial_prediction_max_abs_difference_mpa": trial_prediction_max,
        "trial_cost_max_abs_difference": trial_cost_max,
        "scenario_count": 350,
        "new_reordered_scenarios": 300,
        "nominal_prefix_requests": len(all_requests),
        "unique_cached_prefixes": len(cache.entries),
        "cache_hits_for_nominal_requests": len(all_requests) - len(seen_cache_keys),
        "predictor_top_level_forward_calls": len(complete),
        "predictor_call_batch_sizes": [int(row["evaluated_rows"]) for row in complete],
        "predictor_evaluated_state_rows_including_retries_qa": intended_rows,
        "predictor_evaluated_state_row_cap": 7000,
        "endpoint_uncached_qa_rows": 12,
        "elapsed_seconds": elapsed,
        "optimizer_updates": 0,
        "actor_forwards": 0,
        "qwen_forwards": 0,
        "cnn_forwards": 0,
        "oof_predictor_forwards": 0,
        "autograd": 0,
        "test_forwards": 0,
        "test_access": 0,
        "report_render_count": 0,
        "research_checkpoint_files_created": 0,
    }
    runtime_path = results / "runtime_lock.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime.update(
        {
            "engine_status": "LOCKED_AFTER_NATIVE_REPRODUCTION",
            "engine_signature": engine_signature,
            "engine_details": engine_details,
            "fallback_used": fallback_used,
            "predictor_manifest": manifest,
        }
    )
    atomic_json(runtime_path, runtime)
    return [
        write_gzip_csv_atomic(results / "prefix_request_map.csv.gz", prefix_rows),
        cache.path,
        write_gzip_csv_atomic(results / "reorder_trajectories.csv.gz", trajectory_rows),
        write_csv_atomic(results / "native_reproduction.csv", native_rows),
        write_csv_atomic(results / "endpoint_invariance.csv", endpoint_rows),
        write_csv_atomic(results / "endpoint_uncached_checks.csv", qa_rows),
        ledger.path,
        atomic_json(results / "resource_usage.json", usage),
        runtime_path,
    ]


__all__ = [
    "CacheEntry",
    "PredictorCache",
    "PrefixRequest",
    "ReplayEngine",
    "ResourceLedger",
    "infer_orders",
    "read_resource_events",
]
