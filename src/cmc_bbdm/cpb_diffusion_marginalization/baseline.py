"""Exact P1 I_frozen reproduction for the D8 baseline gate."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import (
    V3Data,
    build_nested_lodo,
    validate_issued_data_authority,
)
from cmc_bbdm.cpb_v3.pipeline import (
    _fit_p1_tabular_candidate,
    _load_p1_feature_bundle,
)

from .config import DOMAIN_ORDER, D8Config


class D8BaselineError(ValueError):
    """Raised when I_frozen cannot be reproduced exactly."""


@dataclass(frozen=True, slots=True)
class D8BaselineResult:
    specimen_count: int
    pca_dimensions: tuple[int, ...]
    domain_mae: tuple[float, ...]
    equal_domain_mae: float
    maximum_prediction_error: float
    maximum_target_error: float
    predictions: np.ndarray
    targets: np.ndarray
    state_sha256: str

    def __post_init__(self) -> None:
        predictions = _readonly_vector(self.predictions, "predictions")
        targets = _readonly_vector(self.targets, "targets")
        if predictions.shape != targets.shape or len(predictions) != self.specimen_count:
            raise D8BaselineError("baseline result rows do not align")
        if len(self.pca_dimensions) != len(DOMAIN_ORDER):
            raise D8BaselineError("baseline PCA dimensions do not cover six domains")
        if len(self.domain_mae) != len(DOMAIN_ORDER):
            raise D8BaselineError("baseline domain MAEs do not cover six domains")
        state = _baseline_state(
            predictions,
            targets,
            self.pca_dimensions,
            self.domain_mae,
            self.equal_domain_mae,
            self.maximum_prediction_error,
            self.maximum_target_error,
        )
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "state_sha256", state)


def _readonly_vector(value: object, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise D8BaselineError(f"{label} must be numeric") from error
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise D8BaselineError(f"{label} must be a finite vector")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _baseline_state(
    predictions: np.ndarray,
    targets: np.ndarray,
    pca_dimensions: tuple[int, ...],
    domain_mae: tuple[float, ...],
    equal_domain_mae: float,
    maximum_prediction_error: float,
    maximum_target_error: float,
) -> str:
    digest = hashlib.sha256()
    for label, array in (("predictions", predictions), ("targets", targets)):
        digest.update(label.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    for value in (
        *pca_dimensions,
        *domain_mae,
        equal_domain_mae,
        maximum_prediction_error,
        maximum_target_error,
    ):
        digest.update(repr(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _registered_predictions(path: Path) -> tuple[dict[str, tuple[float, float]], ...]:
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
                raise D8BaselineError("P1 prediction schema changed")
            records: dict[str, tuple[float, float]] = {}
            domains: dict[str, str] = {}
            for row in reader:
                if row["method"] != "I_frozen":
                    continue
                specimen = row["specimen_id"]
                if specimen in records or row["seed"] != "0":
                    raise D8BaselineError("P1 I_frozen prediction identity changed")
                try:
                    target = float(row["target"])
                    prediction = float(row["prediction"])
                except (TypeError, ValueError) as error:
                    raise D8BaselineError("P1 prediction is not numeric") from error
                if not math.isfinite(target) or not math.isfinite(prediction):
                    raise D8BaselineError("P1 prediction is not finite")
                records[specimen] = (target, prediction)
                domains[specimen] = row["dataset_id"]
    except (OSError, UnicodeError, csv.Error) as error:
        raise D8BaselineError("P1 predictions are unavailable") from error
    if len(records) != 276 or set(domains.values()) != set(DOMAIN_ORDER):
        raise D8BaselineError("P1 I_frozen prediction roster changed")
    return records, domains


def _equal_domain_mae(
    targets: np.ndarray, predictions: np.ndarray, domains: np.ndarray
) -> tuple[tuple[float, ...], float]:
    values: list[float] = []
    for domain in DOMAIN_ORDER:
        mask = domains == domain
        errors = np.abs(targets[mask] - predictions[mask])
        values.append(math.fsum(float(value) for value in errors) / len(errors))
    domain_mae = tuple(values)
    return domain_mae, math.fsum(domain_mae) / len(domain_mae)


def reproduce_internal_only_baseline(
    data: object,
    *,
    config: D8Config,
    project_root: str | Path,
    device: str = "cuda",
) -> D8BaselineResult:
    """Rebuild I_frozen and compare every target and prediction with P1."""

    if type(data) is not V3Data:
        raise D8BaselineError("baseline reproduction requires exact V3Data")
    try:
        validate_issued_data_authority(data)
    except (TypeError, ValueError) as error:
        raise D8BaselineError("baseline data lacks loader authority") from error
    if type(config) is not D8Config:
        raise D8BaselineError("baseline reproduction requires exact D8Config")
    if tuple(config.outer_domains) != DOMAIN_ORDER:
        raise D8BaselineError("D8 domain order changed")
    root = Path(project_root).resolve(strict=True)
    p1_path = root / config.sources["p1_config"].path
    p1_config = load_v3_config(p1_path, project_root=root)
    _resolved_root, bundle, response = _load_p1_feature_bundle(
        p1_config, data, device=device
    )
    pca_by_domain = {
        "74t7kcdgkr": 8,
        "cgtnjyggtm": 32,
        "w68dtmpfyf": 8,
        "xcmzfsbd9t": 8,
        "yfxyg8jm46": 8,
        "ykhs7s2dck": 8,
    }
    predictions = np.full(data.n_samples, np.nan, dtype=np.float64)
    pca_dimensions: list[int] = []
    for outer in build_nested_lodo(data):
        train_authority = data.subset(outer.train_indices)
        dimension = pca_by_domain[outer.fold_id]
        fold_predictions, _pca, _ridge = _fit_p1_tabular_candidate(
            bundle,
            response,
            "I_frozen",
            outer.train_indices,
            outer.test_indices,
            pca_dimension=dimension,
            outer_train_authority=train_authority,
            heldout_domain=outer.fold_id,
        )
        predictions[outer.test_indices] = fold_predictions
        pca_dimensions.append(dimension)
    if not np.all(np.isfinite(predictions)):
        raise D8BaselineError("baseline reproduction left missing predictions")
    registered, registered_domains = _registered_predictions(
        root / config.sources["p1_predictions"].path
    )
    sample_ids = tuple(str(item) for item in data.sample_ids.tolist())
    saved_targets = np.asarray(
        [registered[specimen][0] for specimen in sample_ids], dtype=np.float64
    )
    saved_predictions = np.asarray(
        [registered[specimen][1] for specimen in sample_ids], dtype=np.float64
    )
    domains = np.asarray(data.dataset_ids)
    if tuple(registered_domains[specimen] for specimen in sample_ids) != tuple(
        str(item) for item in domains.tolist()
    ):
        raise D8BaselineError("P1 prediction domain order changed")
    targets = np.asarray(data.cai_ratio, dtype=np.float64)
    maximum_prediction_error = float(np.max(np.abs(predictions - saved_predictions)))
    maximum_target_error = float(np.max(np.abs(targets - saved_targets)))
    domain_mae, equal_domain_mae = _equal_domain_mae(targets, predictions, domains)
    return D8BaselineResult(
        specimen_count=data.n_samples,
        pca_dimensions=tuple(pca_dimensions),
        domain_mae=domain_mae,
        equal_domain_mae=equal_domain_mae,
        maximum_prediction_error=maximum_prediction_error,
        maximum_target_error=maximum_target_error,
        predictions=predictions,
        targets=targets,
        state_sha256="",
    )


__all__ = [
    "D8BaselineError",
    "D8BaselineResult",
    "reproduce_internal_only_baseline",
]
