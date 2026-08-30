from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS
from cmc_bbdm.damage_response.feature_views import PRIMARY_TARGET_FIELDS
from cmc_bbdm.damage_response.p2_evaluation import P2OOFPrediction

P2_BOOTSTRAP_SEED = 20260830
P2_BOOTSTRAP_REPLICATES = 100_000
PRIMARY_CONTRAST_COUNT = 6
FAMILYWISE_LOWER_QUANTILE = 0.025 / PRIMARY_CONTRAST_COUNT
FAMILYWISE_UPPER_QUANTILE = 1.0 - 0.025 / PRIMARY_CONTRAST_COUNT
DOMAIN_ORDER = tuple(PRIMARY_COUNTS)

_SELECTED_VIEWS = ("F2", "F3", "F4")
_BOOTSTRAP_CHUNK_SIZE = 2048


class P2StatisticsError(ValueError):
    """Raised when P2 paired contrasts do not match the frozen protocol."""


@dataclass(frozen=True, slots=True)
class _ContrastSpec:
    name: str
    endpoint: str
    reference_view: str
    candidate_view: str
    primary_family: bool


@dataclass(frozen=True, slots=True)
class P2ContrastResult:
    name: str
    endpoint: str
    reference_view: str
    candidate_view: str
    primary_family: bool
    observed_reference_equal_domain_mae: float
    observed_candidate_equal_domain_mae: float
    observed_improvement: float
    relative_improvement: float
    improved_domain_count: int
    domain_improvements: tuple[tuple[str, float], ...]
    bootstrap_mean: float
    ordinary_interval: tuple[float, float]
    familywise_interval: tuple[float, float] | None
    probability_positive: float
    bootstrap_column: int
    replicate_sha256: str


@dataclass(frozen=True, slots=True)
class P2ContrastAnalysis:
    seed: int
    replicates: int
    contrasts: tuple[P2ContrastResult, ...]
    synchronized_replicate_sha256: str
    bootstrap_samples: np.ndarray


def _readonly(value: np.ndarray, *, dtype: str = "<f8") -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def synchronized_within_domain_bootstrap(
    contrasts_by_domain: Mapping[str, object],
    *,
    seed: int,
    replicates: int,
) -> np.ndarray:
    """Resample specimens within each domain with one shared contrast index set."""

    if (
        not isinstance(contrasts_by_domain, Mapping)
        or set(contrasts_by_domain) != set(DOMAIN_ORDER)
    ):
        raise P2StatisticsError("bootstrap requires the six canonical domains")
    if type(seed) is not int or type(replicates) is not int or replicates < 1:
        raise P2StatisticsError("bootstrap seed/count is invalid")
    matrices: dict[str, np.ndarray] = {}
    column_count: int | None = None
    for domain in DOMAIN_ORDER:
        try:
            matrix = np.asarray(contrasts_by_domain[domain], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise P2StatisticsError("bootstrap contrast matrix is not numeric") from error
        if (
            matrix.ndim != 2
            or matrix.shape[0] < 1
            or matrix.shape[1] < 1
            or not np.all(np.isfinite(matrix))
        ):
            raise P2StatisticsError("bootstrap contrast matrix is invalid")
        if column_count is None:
            column_count = matrix.shape[1]
        elif matrix.shape[1] != column_count:
            raise P2StatisticsError("bootstrap contrast column count changed")
        matrices[domain] = matrix
    if column_count is None:
        raise P2StatisticsError("bootstrap contrast registry is empty")

    generator = np.random.Generator(np.random.PCG64(seed))
    samples = np.zeros((replicates, column_count), dtype=np.float64)
    domain_weight = 1.0 / len(DOMAIN_ORDER)
    for domain in DOMAIN_ORDER:
        matrix = matrices[domain]
        specimen_count = matrix.shape[0]
        for start in range(0, replicates, _BOOTSTRAP_CHUNK_SIZE):
            stop = min(start + _BOOTSTRAP_CHUNK_SIZE, replicates)
            draws = generator.integers(
                0,
                specimen_count,
                size=(stop - start, specimen_count),
                dtype=np.int64,
            )
            samples[start:stop] += (
                np.mean(matrix[draws], axis=1, dtype=np.float64) * domain_weight
            )
    if not np.all(np.isfinite(samples)):
        raise P2StatisticsError("bootstrap returned nonfinite samples")
    return _readonly(samples)


def _contrast_specs() -> tuple[_ContrastSpec, ...]:
    primary = tuple(
        _ContrastSpec(
            name=f"{endpoint}__{candidate}_vs_F2",
            endpoint=endpoint,
            reference_view="F2",
            candidate_view=candidate,
            primary_family=True,
        )
        for endpoint in PRIMARY_TARGET_FIELDS
        for candidate in ("F3", "F4")
    )
    secondary = tuple(
        _ContrastSpec(
            name=f"{endpoint}__F4_vs_F3",
            endpoint=endpoint,
            reference_view="F3",
            candidate_view="F4",
            primary_family=False,
        )
        for endpoint in PRIMARY_TARGET_FIELDS
    )
    return (*primary, *secondary)


def _prediction_registry(
    predictions: Sequence[P2OOFPrediction],
) -> dict[tuple[str, str, str, str], P2OOFPrediction]:
    selected = tuple(
        row
        for row in predictions
        if isinstance(row, P2OOFPrediction)
        and row.endpoint in PRIMARY_TARGET_FIELDS
        and row.view_name in _SELECTED_VIEWS
    )
    expected_per_view: set[tuple[str, str]] | None = None
    by_key: dict[tuple[str, str, str, str], P2OOFPrediction] = {}
    for row in selected:
        if row.domain_id not in DOMAIN_ORDER or row.held_out_domain != row.domain_id:
            raise P2StatisticsError("contrast prediction split identity changed")
        numeric = (
            row.truth,
            row.prediction,
            row.absolute_error,
            row.standardized_absolute_error,
            row.source_target_std,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise P2StatisticsError("contrast prediction contains nonfinite values")
        if not math.isclose(
            row.absolute_error,
            abs(row.truth - row.prediction),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise P2StatisticsError("contrast absolute error does not reconcile")
        key = (row.endpoint, row.view_name, row.domain_id, row.specimen_id)
        if key in by_key:
            raise P2StatisticsError("contrast prediction identity is duplicate")
        by_key[key] = row

    for endpoint in PRIMARY_TARGET_FIELDS:
        for view_name in _SELECTED_VIEWS:
            membership = {
                (row.domain_id, row.specimen_id)
                for row in selected
                if row.endpoint == endpoint and row.view_name == view_name
            }
            if not membership:
                raise P2StatisticsError("contrast prediction membership is empty")
            if expected_per_view is None:
                expected_per_view = membership
            elif membership != expected_per_view:
                raise P2StatisticsError("contrast prediction membership differs")
    if expected_per_view is None:
        raise P2StatisticsError("contrast prediction registry is empty")
    if {domain for domain, _specimen in expected_per_view} != set(DOMAIN_ORDER):
        raise P2StatisticsError("contrast prediction domains are incomplete")

    for endpoint in PRIMARY_TARGET_FIELDS:
        for domain, specimen_id in expected_per_view:
            truths = {
                by_key[(endpoint, view, domain, specimen_id)].truth
                for view in _SELECTED_VIEWS
            }
            if len(truths) != 1:
                raise P2StatisticsError("contrast truth differs across views")
    return by_key


def analyze_p2_contrasts(
    predictions: Sequence[P2OOFPrediction],
    *,
    seed: int = P2_BOOTSTRAP_SEED,
    replicates: int = P2_BOOTSTRAP_REPLICATES,
) -> P2ContrastAnalysis:
    """Build all primary/secondary paired P2 MAE contrasts and intervals."""

    by_key = _prediction_registry(tuple(predictions))
    specs = _contrast_specs()
    if len([spec for spec in specs if spec.primary_family]) != PRIMARY_CONTRAST_COUNT:
        raise P2StatisticsError("primary contrast family changed")
    contrasts_by_domain: dict[str, np.ndarray] = {}
    domain_ids: dict[str, tuple[str, ...]] = {}
    for domain in DOMAIN_ORDER:
        specimen_ids = tuple(
            sorted(
                {
                    specimen_id
                    for endpoint, view, row_domain, specimen_id in by_key
                    if endpoint == PRIMARY_TARGET_FIELDS[0]
                    and view == "F2"
                    and row_domain == domain
                }
            )
        )
        if not specimen_ids:
            raise P2StatisticsError(f"contrast domain is empty: {domain}")
        matrix = np.empty((len(specimen_ids), len(specs)), dtype=np.float64)
        for row_index, specimen_id in enumerate(specimen_ids):
            for column, spec in enumerate(specs):
                reference = by_key[
                    (spec.endpoint, spec.reference_view, domain, specimen_id)
                ]
                candidate = by_key[
                    (spec.endpoint, spec.candidate_view, domain, specimen_id)
                ]
                matrix[row_index, column] = (
                    reference.absolute_error - candidate.absolute_error
                )
        contrasts_by_domain[domain] = matrix
        domain_ids[domain] = specimen_ids

    samples = synchronized_within_domain_bootstrap(
        contrasts_by_domain, seed=seed, replicates=replicates
    )
    results: list[P2ContrastResult] = []
    for column, spec in enumerate(specs):
        domain_improvements = tuple(
            (
                domain,
                float(
                    np.mean(contrasts_by_domain[domain][:, column], dtype=np.float64)
                ),
            )
            for domain in DOMAIN_ORDER
        )
        reference_domain_mae: list[float] = []
        candidate_domain_mae: list[float] = []
        for domain in DOMAIN_ORDER:
            reference_domain_mae.append(
                float(
                    np.mean(
                        [
                            by_key[
                                (
                                    spec.endpoint,
                                    spec.reference_view,
                                    domain,
                                    specimen_id,
                                )
                            ].absolute_error
                            for specimen_id in domain_ids[domain]
                        ],
                        dtype=np.float64,
                    )
                )
            )
            candidate_domain_mae.append(
                float(
                    np.mean(
                        [
                            by_key[
                                (
                                    spec.endpoint,
                                    spec.candidate_view,
                                    domain,
                                    specimen_id,
                                )
                            ].absolute_error
                            for specimen_id in domain_ids[domain]
                        ],
                        dtype=np.float64,
                    )
                )
            )
        reference_equal = float(np.mean(reference_domain_mae, dtype=np.float64))
        candidate_equal = float(np.mean(candidate_domain_mae, dtype=np.float64))
        if reference_equal <= 0.0:
            raise P2StatisticsError("contrast reference MAE must be positive")
        observed = reference_equal - candidate_equal
        column_samples = np.asarray(samples[:, column], dtype=np.float64)
        ordinary_raw = np.quantile(
            column_samples, (0.025, 0.975), method="linear"
        )
        familywise: tuple[float, float] | None = None
        if spec.primary_family:
            familywise_raw = np.quantile(
                column_samples,
                (FAMILYWISE_LOWER_QUANTILE, FAMILYWISE_UPPER_QUANTILE),
                method="linear",
            )
            familywise = (float(familywise_raw[0]), float(familywise_raw[1]))
        results.append(
            P2ContrastResult(
                name=spec.name,
                endpoint=spec.endpoint,
                reference_view=spec.reference_view,
                candidate_view=spec.candidate_view,
                primary_family=spec.primary_family,
                observed_reference_equal_domain_mae=reference_equal,
                observed_candidate_equal_domain_mae=candidate_equal,
                observed_improvement=observed,
                relative_improvement=observed / reference_equal,
                improved_domain_count=sum(
                    value > 0.0 for _domain, value in domain_improvements
                ),
                domain_improvements=domain_improvements,
                bootstrap_mean=float(np.mean(column_samples, dtype=np.float64)),
                ordinary_interval=(float(ordinary_raw[0]), float(ordinary_raw[1])),
                familywise_interval=familywise,
                probability_positive=float(
                    np.mean(column_samples > 0.0, dtype=np.float64)
                ),
                bootstrap_column=column,
                replicate_sha256=hashlib.sha256(
                    np.ascontiguousarray(column_samples, dtype="<f8").tobytes()
                ).hexdigest(),
            )
        )
    matrix_bytes = np.ascontiguousarray(samples, dtype="<f8").tobytes(order="C")
    return P2ContrastAnalysis(
        seed=seed,
        replicates=replicates,
        contrasts=tuple(results),
        synchronized_replicate_sha256=hashlib.sha256(matrix_bytes).hexdigest(),
        bootstrap_samples=samples,
    )
