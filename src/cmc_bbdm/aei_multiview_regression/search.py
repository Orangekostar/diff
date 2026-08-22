"""Registered source-domain selection for cooperative regression."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CooperativeCandidate:
    loss: str
    lambda_consistency: float

    def __post_init__(self) -> None:
        if self.loss not in {"mse", "huber"}:
            raise ValueError("candidate loss is not registered")
        if not np.isfinite(self.lambda_consistency) or self.lambda_consistency < 0.0:
            raise ValueError("candidate consistency strength is invalid")


@dataclass(frozen=True, slots=True)
class CooperativeCandidateScore:
    candidate: CooperativeCandidate
    domain_mae: tuple[tuple[str, float], ...]
    equal_domain_mae: float
    worst_domain_mae: float
    domain_mae_sd: float
    mean_absolute_disagreement: float
    prediction_variances: tuple[float, ...]
    residual_correlations: tuple[float, ...]
    collapsed: bool


@dataclass(frozen=True, slots=True)
class CooperativeSearchResult:
    selected: CooperativeCandidate
    scores: tuple[CooperativeCandidateScore, ...]


def select_cooperative_oof(
    candidates: Mapping[CooperativeCandidate, object],
    *,
    targets: object,
    domains: Sequence[str],
) -> CooperativeSearchResult:
    """Rank complete source-OOF candidate predictions without query access."""

    y = np.asarray(targets, dtype=np.float64)
    domain_ids = tuple(domains)
    if (
        y.ndim != 1
        or len(y) < 3
        or not np.all(np.isfinite(y))
        or len(domain_ids) != len(y)
        or any(type(item) is not str or not item for item in domain_ids)
        or not isinstance(candidates, Mapping)
        or not candidates
    ):
        raise ValueError("cooperative search inputs are invalid")
    order = tuple(dict.fromkeys(domain_ids))
    domain_array = np.asarray(domain_ids)
    scores: list[CooperativeCandidateScore] = []
    ranked: list[tuple[tuple[object, ...], CooperativeCandidate]] = []
    for insertion_order, (candidate, raw) in enumerate(candidates.items()):
        if not isinstance(candidate, CooperativeCandidate):
            raise TypeError("cooperative candidate key is invalid")
        predictions = np.asarray(raw, dtype=np.float64)
        if (
            predictions.ndim != 2
            or predictions.shape[0] != len(y)
            or predictions.shape[1] < 1
            or not np.all(np.isfinite(predictions))
        ):
            raise ValueError("candidate OOF predictions are invalid")
        ensemble = np.mean(predictions, axis=1)
        domain_mae = tuple(
            (
                domain,
                float(
                    np.mean(
                        np.abs(
                            y[domain_array == domain] - ensemble[domain_array == domain]
                        )
                    )
                ),
            )
            for domain in order
        )
        domain_values = np.asarray([item[1] for item in domain_mae])
        disagreements = [
            np.mean(np.abs(predictions[:, left] - predictions[:, right]))
            for left in range(predictions.shape[1])
            for right in range(left + 1, predictions.shape[1])
        ]
        mean_disagreement = float(np.mean(disagreements)) if disagreements else 0.0
        residuals = y[:, None] - predictions
        residual_correlations = []
        for left in range(predictions.shape[1]):
            for right in range(left + 1, predictions.shape[1]):
                left_residual = residuals[:, left]
                right_residual = residuals[:, right]
                if np.ptp(left_residual) == 0.0 or np.ptp(right_residual) == 0.0:
                    correlation = 0.0
                else:
                    correlation = float(
                        np.corrcoef(left_residual, right_residual)[0, 1]
                    )
                residual_correlations.append(
                    correlation if np.isfinite(correlation) else 0.0
                )
        collapsed = bool(
            candidate.lambda_consistency > 0.0
            and predictions.shape[1] > 1
            and mean_disagreement <= 1e-12
        )
        score = CooperativeCandidateScore(
            candidate=candidate,
            domain_mae=domain_mae,
            equal_domain_mae=float(np.mean(domain_values)),
            worst_domain_mae=float(np.max(domain_values)),
            domain_mae_sd=float(np.std(domain_values, ddof=0)),
            mean_absolute_disagreement=mean_disagreement,
            prediction_variances=tuple(
                float(np.var(predictions[:, index], ddof=0))
                for index in range(predictions.shape[1])
            ),
            residual_correlations=tuple(residual_correlations),
            collapsed=collapsed,
        )
        scores.append(score)
        ranked.append(
            (
                (
                    score.equal_domain_mae,
                    score.worst_domain_mae,
                    score.domain_mae_sd,
                    0 if candidate.loss == "mse" else 1,
                    candidate.lambda_consistency,
                    insertion_order,
                ),
                candidate,
            )
        )
    selected = min(ranked, key=lambda item: item[0])[1]
    return CooperativeSearchResult(selected=selected, scores=tuple(scores))
