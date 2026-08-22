"""Prediction-level verification and byte-identical replay of S1 packages."""

from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
from pathlib import Path

from .artifacts import S1ArtifactError, S1PackageValidation, validate_s1_package


class MSSSReplayError(ValueError):
    """Raised when stored predictions cannot reproduce an S1 result."""


def _rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            values = [dict(row) for row in csv.DictReader(handle, strict=True)]
    except (OSError, UnicodeError, csv.Error) as error:
        raise MSSSReplayError(f"replay table is unreadable: {path.name}") from error
    if not values:
        raise MSSSReplayError(f"replay table is empty: {path.name}")
    return values


def _number(value: str, label: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise MSSSReplayError(f"replay value is invalid: {label}") from error
    if not math.isfinite(output):
        raise MSSSReplayError(f"replay value is non-finite: {label}")
    return output


def _mean(values: list[float]) -> float:
    if not values:
        raise MSSSReplayError("replay aggregation is empty")
    return math.fsum(values) / len(values)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-14)


def _verify_candidate_curves(root: Path) -> None:
    predictions = _rows(root / "candidate_predictions.csv")
    by_key: dict[tuple[str, str, str], list[float]] = {}
    for row in predictions:
        key = (row["axis"], row["condition_id"], row["dataset_id"])
        by_key.setdefault(key, []).append(_number(row["absolute_error"], "absolute_error"))
    domain_rows = _rows(root / "domain_scale_metrics.csv")
    domain_mae: dict[tuple[str, str, str], float] = {}
    for row in domain_rows:
        key = (row["axis"], row["condition_id"], row["dataset_id"])
        if key in domain_mae or key not in by_key:
            raise MSSSReplayError("domain-scale roster does not match predictions")
        computed = _mean(by_key[key])
        reported = _number(row["mae"], "domain MAE")
        if not _close(computed, reported):
            raise MSSSReplayError("domain-scale MAE does not reproduce")
        domain_mae[key] = computed
    if set(domain_mae) != set(by_key):
        raise MSSSReplayError("candidate prediction roster is incomplete")
    for axis in ("sampling", "gaussian", "wavelet"):
        for row in _rows(root / f"{axis}_curve.csv"):
            condition = row["condition_id"]
            values = [
                value
                for (item_axis, item_condition, _dataset), value in domain_mae.items()
                if item_axis == axis and item_condition == condition
            ]
            computed = _mean(values)
            reported = _number(row["equal_domain_mae"], "equal-domain MAE")
            if not _close(computed, reported):
                raise MSSSReplayError("scale curve does not reproduce from predictions")


def _verify_spatial_specificity(root: Path) -> None:
    predictions = _rows(root / "spatial_predictions.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in predictions:
        grouped.setdefault((row["axis"], row["dataset_id"]), []).append(row)
    table = _rows(root / "spatial_specificity.csv")
    domain_effects: dict[str, list[float]] = {}
    seen: set[tuple[str, str]] = set()
    for row in table:
        axis, dataset = row["axis"], row["dataset_id"]
        if dataset == "EQUAL_DOMAIN":
            continue
        key = (axis, dataset)
        values = grouped.get(key)
        if key in seen or not values:
            raise MSSSReplayError("spatial-specificity roster does not match predictions")
        seen.add(key)
        regular_by_specimen: dict[str, float] = {}
        shuffled: list[float] = []
        for prediction in values:
            specimen = prediction["specimen_id"]
            regular = _number(prediction["regular_absolute_error"], "regular error")
            previous = regular_by_specimen.setdefault(specimen, regular)
            if not _close(previous, regular):
                raise MSSSReplayError("regular error changed across shuffle seeds")
            shuffled.append(_number(prediction["shuffled_absolute_error"], "shuffled error"))
        regular_mae = _mean(list(regular_by_specimen.values()))
        shuffled_mae = _mean(shuffled)
        effect = shuffled_mae - regular_mae
        if not all(
            (
                _close(regular_mae, _number(row["regular_mae"], "regular MAE")),
                _close(shuffled_mae, _number(row["shuffled_mae"], "shuffled MAE")),
                _close(effect, _number(row["ssg"], "SSG")),
            )
        ):
            raise MSSSReplayError("spatial-specificity metrics do not reproduce")
        domain_effects.setdefault(axis, []).append(effect)
    if set(grouped) != seen:
        raise MSSSReplayError("spatial prediction roster is incomplete")
    equal_rows = {
        row["axis"]: row for row in table if row["dataset_id"] == "EQUAL_DOMAIN"
    }
    if set(equal_rows) != set(domain_effects):
        raise MSSSReplayError("equal-domain specificity rows are incomplete")
    for axis, effects in domain_effects.items():
        if not _close(_mean(effects), _number(equal_rows[axis]["ssg"], "equal-domain SSG")):
            raise MSSSReplayError("equal-domain specificity does not reproduce")


def replay_s1_package(
    source: str | Path,
    destination: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> S1PackageValidation:
    """Recompute primary table values, then atomically reproduce the package."""

    source_root = Path(source).resolve(strict=True)
    output = Path(destination).resolve()
    if output.exists():
        raise MSSSReplayError(f"replay output already exists: {output}")
    try:
        source_validation = validate_s1_package(
            source_root, project_root=project_root, config_path=config_path
        )
        _verify_candidate_curves(source_root)
        _verify_spatial_specificity(source_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
        try:
            for path in sorted(source_root.iterdir()):
                if not path.is_file():
                    raise MSSSReplayError("nested replay content is not allowed")
                shutil.copy2(path, staging / path.name)
            replay_validation = validate_s1_package(
                staging, project_root=project_root, config_path=config_path
            )
            if replay_validation != source_validation:
                raise MSSSReplayError("replay package digest changed")
            os.replace(staging, output)
            return replay_validation
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    except MSSSReplayError:
        raise
    except (S1ArtifactError, OSError, ValueError) as error:
        raise MSSSReplayError("S1 replay verification failed") from error


from .transfer_artifacts import replay_s2_package

__all__ = ["MSSSReplayError", "replay_s1_package", "replay_s2_package"]
