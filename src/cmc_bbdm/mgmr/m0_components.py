"""Frozen B0--B4 component roster for the MGMR M0 gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .authority import MGMRM0Authority
from .evaluation import NestedLODORun, nested_lodo_predictions
from .feature_bank import MGMRFeatureBank
from .protocol import MGMRProtocol


class MGMRComponentError(ValueError):
    """Raised when the frozen M0 component roster cannot be evaluated."""


_METHODS = ("B0", "B1", "B2", "B3", "B4")
_B0_DIMENSIONS = (8, 32, 8, 8, 8, 8)


def _readonly(value: object, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MGMRComponentError(f"{label} must be numeric") from error
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise MGMRComponentError(f"{label} must be a finite vector")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    output.setflags(write=False)
    return output


def _state(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(value.tobytes(order="C"))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
    return digest.hexdigest()


def _domain_mae(
    targets: np.ndarray,
    predictions: np.ndarray,
    dataset_ids: Sequence[str],
    domain_order: Sequence[str],
) -> tuple[tuple[float, ...], float]:
    domains = np.asarray(tuple(dataset_ids), dtype=object)
    values: list[float] = []
    for domain in domain_order:
        errors = np.abs(targets[domains == domain] - predictions[domains == domain])
        if not errors.size:
            raise MGMRComponentError("baseline domain roster is incomplete")
        values.append(math.fsum(float(item) for item in errors) / errors.size)
    return tuple(values), math.fsum(values) / len(values)


@dataclass(frozen=True, slots=True)
class RegisteredB0:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    targets: np.ndarray
    predictions: np.ndarray
    pca_dimensions: tuple[int, ...]
    domain_mae: tuple[float, ...]
    equal_domain_mae: float
    maximum_target_error: float
    source_sha256: str
    state_sha256: str

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)


@dataclass(frozen=True, slots=True)
class ComponentEvaluation:
    methods: tuple[str, ...]
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    targets: np.ndarray
    baseline_dimensions: tuple[int, ...]
    runs: Mapping[str, NestedLODORun]
    predictions: Mapping[str, np.ndarray]
    state_sha256: str


def load_registered_b0(
    protocol: MGMRProtocol,
    authority: MGMRM0Authority,
    *,
    project_root: str | Path,
) -> RegisteredB0:
    """Load P1 I_frozen after exact cohort, target, and checksum validation."""

    if type(protocol) is not MGMRProtocol or type(authority) is not MGMRM0Authority:
        raise MGMRComponentError("issued protocol and authority are required")
    root = Path(project_root).resolve(strict=True)
    source = protocol.sources["p1_predictions"]
    path = root / source.path
    if hashlib.sha256(path.read_bytes()).hexdigest() != source.sha256:
        raise MGMRComponentError("registered P1 prediction bytes changed")
    records: dict[str, tuple[str, float, float]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames != [
                "method",
                "specimen_id",
                "dataset_id",
                "target",
                "prediction",
                "seed",
            ]:
                raise MGMRComponentError("registered P1 prediction schema changed")
            for row in reader:
                if row["method"] != "I_frozen":
                    continue
                specimen_id = row["specimen_id"]
                if specimen_id in records or row["seed"] != "0":
                    raise MGMRComponentError("registered B0 prediction identity changed")
                records[specimen_id] = (
                    row["dataset_id"],
                    float(row["target"]),
                    float(row["prediction"]),
                )
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as error:
        if isinstance(error, MGMRComponentError):
            raise
        raise MGMRComponentError("registered B0 predictions are unavailable") from error
    if set(records) != set(authority.specimen_ids):
        raise MGMRComponentError("registered B0 specimen roster changed")
    ordered = tuple(records[item] for item in authority.specimen_ids)
    dataset_ids = tuple(row[0] for row in ordered)
    if dataset_ids != authority.dataset_ids:
        raise MGMRComponentError("registered B0 domain order changed")
    saved_targets = _readonly([row[1] for row in ordered], "registered B0 targets")
    predictions = _readonly([row[2] for row in ordered], "registered B0 predictions")
    maximum_target_error = float(np.max(np.abs(saved_targets - authority.targets)))
    if maximum_target_error > 1.0e-12:
        raise MGMRComponentError("registered B0 targets do not match M0 authority")
    domain_mae, equal_mae = _domain_mae(
        authority.targets, predictions, dataset_ids, protocol.domain_order
    )
    if abs(equal_mae - protocol.baseline_mae) > 1.0e-12:
        raise MGMRComponentError("registered B0 MAE changed")
    state = _state(
        "registered-b0",
        authority.specimen_ids,
        dataset_ids,
        saved_targets,
        predictions,
        _B0_DIMENSIONS,
        domain_mae,
        equal_mae,
        source.sha256,
    )
    return RegisteredB0(
        specimen_ids=authority.specimen_ids,
        dataset_ids=dataset_ids,
        targets=saved_targets,
        predictions=predictions,
        pca_dimensions=_B0_DIMENSIONS,
        domain_mae=domain_mae,
        equal_domain_mae=equal_mae,
        maximum_target_error=maximum_target_error,
        source_sha256=source.sha256,
        state_sha256=state,
    )


def evaluate_component_arrays(
    *,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    domain_order: Sequence[str],
    targets: object,
    metadata: object,
    full: object,
    coarse: object,
    boundary: object,
    baseline_predictions: object,
    baseline_dimensions: Sequence[int],
    pca_dimensions: Sequence[int],
    ridge_alpha: float,
    tie_tolerance: float,
) -> ComponentEvaluation:
    """Evaluate the exact direct component roster on aligned arrays."""

    samples = tuple(specimen_ids)
    datasets = tuple(dataset_ids)
    domains = tuple(domain_order)
    y = _readonly(targets, "targets")
    baseline = _readonly(baseline_predictions, "baseline predictions")
    if (
        len(samples) != y.size
        or len(datasets) != y.size
        or baseline.shape != y.shape
        or len(set(samples)) != y.size
        or set(datasets) != set(domains)
    ):
        raise MGMRComponentError("component arrays are not cohort aligned")
    b0_dimensions = tuple(baseline_dimensions)
    if len(b0_dimensions) != len(domains):
        raise MGMRComponentError("B0 dimensions do not cover every domain")
    roster = {
        "B1": {"coarse": coarse},
        "B2": {"boundary": boundary},
        "B3": {"coarse": coarse, "boundary": boundary},
        "B4": {"full": full, "boundary": boundary},
    }
    runs: dict[str, NestedLODORun] = {}
    for method, blocks in roster.items():
        runs[method] = nested_lodo_predictions(
            method=method,
            metadata=metadata,
            blocks=blocks,
            targets=y,
            specimen_ids=samples,
            dataset_ids=datasets,
            domain_order=domains,
            pca_dimensions=pca_dimensions,
            ridge_alpha=ridge_alpha,
            tie_tolerance=tie_tolerance,
        )
    predictions = {"B0": baseline}
    predictions.update({method: run.predictions for method, run in runs.items()})
    immutable_runs = MappingProxyType(runs)
    immutable_predictions = MappingProxyType(predictions)
    state_parts: list[object] = [
        "m0-components",
        samples,
        datasets,
        y,
        b0_dimensions,
    ]
    for method in _METHODS:
        state_parts.extend((method, predictions[method]))
    state_parts.append(tuple((method, runs[method].state_sha256) for method in runs))
    state = _state(*state_parts)
    return ComponentEvaluation(
        methods=_METHODS,
        specimen_ids=samples,
        dataset_ids=datasets,
        targets=y,
        baseline_dimensions=b0_dimensions,
        runs=immutable_runs,
        predictions=immutable_predictions,
        state_sha256=state,
    )


def evaluate_components(
    protocol: MGMRProtocol,
    authority: MGMRM0Authority,
    features: MGMRFeatureBank,
    baseline: RegisteredB0,
) -> ComponentEvaluation:
    """Evaluate B0--B4 after strict authority and feature-bank alignment."""

    if (
        type(protocol) is not MGMRProtocol
        or type(authority) is not MGMRM0Authority
        or type(features) is not MGMRFeatureBank
        or type(baseline) is not RegisteredB0
    ):
        raise MGMRComponentError("issued M0 inputs are required")
    if (
        features.specimen_ids != authority.specimen_ids
        or features.dataset_ids != authority.dataset_ids
        or features.config_sha256 != protocol.config_sha256
        or baseline.specimen_ids != authority.specimen_ids
        or baseline.dataset_ids != authority.dataset_ids
        or not np.array_equal(features.full_global, authority.full_global)
    ):
        raise MGMRComponentError("M0 component authorities are not aligned")
    return evaluate_component_arrays(
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        domain_order=protocol.domain_order,
        targets=authority.targets,
        metadata=authority.metadata13,
        full=features.full_global,
        coarse=features.coarse_gap,
        boundary=features.full_directional,
        baseline_predictions=baseline.predictions,
        baseline_dimensions=baseline.pca_dimensions,
        pca_dimensions=protocol.pca_dimensions,
        ridge_alpha=protocol.ridge_alpha,
        tie_tolerance=protocol.pca_tie_tolerance,
    )


__all__ = [
    "ComponentEvaluation",
    "MGMRComponentError",
    "RegisteredB0",
    "evaluate_component_arrays",
    "evaluate_components",
    "load_registered_b0",
]
