"""Strict source-only Ridge and small shared scorers for MVD M1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional

from .observability_dataset import ObservedValueExamples

_MODES = ("candidate_only", "global_candidate", "a5_initial")


def _matrix(examples: ObservedValueExamples, mode: str) -> np.ndarray:
    if type(examples) is not ObservedValueExamples or mode not in _MODES:
        raise ValueError("observability feature request changed")
    count = examples.specimen_count
    candidate = examples.candidate_features.reshape(count * 64, 8)
    if mode == "candidate_only":
        return np.ascontiguousarray(candidate, dtype=np.float64)
    global_features = np.column_stack(
        (examples.initial_embeddings, examples.current_predictions)
    )
    if mode == "a5_initial":
        # Initial A5 levels are all zero. Budget scalars use exact measured/native
        # counts, matching the historical policy-state contract.
        global_features = np.column_stack(
            (
                examples.initial_embeddings,
                np.zeros((count, 64), dtype=np.float64),
                examples.current_predictions,
                examples.initial_used_budgets,
                0.25 - examples.initial_used_budgets,
            )
        )
    repeated = np.repeat(global_features, 64, axis=0)
    return np.ascontiguousarray(np.column_stack((repeated, candidate)), dtype=np.float64)


def _weights(examples: ObservedValueExamples) -> np.ndarray:
    domains = tuple(dict.fromkeys(examples.dataset_ids))
    counts = {domain: examples.dataset_ids.count(domain) for domain in domains}
    specimen = np.asarray(
        [1.0 / len(domains) / counts[domain] for domain in examples.dataset_ids],
        dtype=np.float64,
    )
    return np.repeat(specimen / 64.0, 64)


@dataclass(frozen=True, slots=True)
class RidgeScorer:
    mode: str
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    fit_domains: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    state_sha256: str

    def predict(self, examples: ObservedValueExamples) -> np.ndarray:
        matrix = _matrix(examples, self.mode)
        output = ((matrix - self.mean) / self.scale) @ self.coefficients + self.intercept
        values = np.asarray(output, dtype=np.float64).reshape(examples.specimen_count, 64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Ridge observability prediction is nonfinite")
        return values


def fit_ridge_scorer(
    examples: ObservedValueExamples,
    *,
    outer_domain: str,
    mode: str,
    alpha: float,
) -> RidgeScorer:
    """Fit one weighted Ridge without permitting the outer target domain."""

    if (
        type(examples) is not ObservedValueExamples
        or examples.role != "source_train"
        or examples.outer_domain != outer_domain
        or outer_domain in examples.dataset_ids
        or len(set(examples.dataset_ids)) < 2
        or mode not in _MODES
        or float(alpha) not in {0.1, 1.0, 10.0, 100.0}
    ):
        raise ValueError("outer domain must be excluded from observability training")
    matrix = _matrix(examples, mode)
    weights = _weights(examples)
    mean = np.average(matrix, axis=0, weights=weights)
    variance = np.average((matrix - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
    model.fit(
        (matrix - mean) / scale,
        examples.mechanical_values.reshape(-1),
        sample_weight=weights,
    )
    coefficients = np.ascontiguousarray(model.coef_, dtype=np.float64)
    digest = hashlib.sha256(
        json.dumps(
            {
                "alpha": float(alpha),
                "fit_domains": tuple(dict.fromkeys(examples.dataset_ids)),
                "fit_specimen_ids": examples.specimen_ids,
                "mode": mode,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for value in (mean, scale, coefficients, np.asarray([model.intercept_])):
        digest.update(np.asarray(value, dtype=np.float64).tobytes(order="C"))
    return RidgeScorer(
        mode=mode,
        alpha=float(alpha),
        mean=np.ascontiguousarray(mean, dtype=np.float64),
        scale=np.ascontiguousarray(scale, dtype=np.float64),
        coefficients=coefficients,
        intercept=float(model.intercept_),
        fit_domains=tuple(dict.fromkeys(examples.dataset_ids)),
        fit_specimen_ids=examples.specimen_ids,
        state_sha256=digest.hexdigest(),
    )


def _global_matrix(examples: ObservedValueExamples, mode: str) -> np.ndarray | None:
    if mode == "candidate_only":
        return None
    count = examples.specimen_count
    if mode == "global_candidate":
        return np.ascontiguousarray(
            np.column_stack(
                (examples.initial_embeddings, examples.current_predictions)
            ),
            dtype=np.float64,
        )
    if mode == "a5_initial":
        return np.ascontiguousarray(
            np.column_stack(
                (
                    examples.initial_embeddings,
                    np.zeros((count, 64), dtype=np.float64),
                    examples.current_predictions,
                    examples.initial_used_budgets,
                    0.25 - examples.initial_used_budgets,
                )
            ),
            dtype=np.float64,
        )
    raise ValueError("MLP observability mode changed")


class SharedValueMLP(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in _MODES:
            raise ValueError("MLP observability mode changed")
        self.mode = mode
        global_dimension = 0 if mode == "candidate_only" else 513 if mode == "global_candidate" else 579
        self.global_network = (
            None
            if global_dimension == 0
            else nn.Sequential(
                nn.Linear(global_dimension, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
        )
        self.candidate_network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        scorer_input = 16 if global_dimension == 0 else 48
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, global_features: torch.Tensor | None, candidate_features: torch.Tensor
    ) -> torch.Tensor:
        candidate_hidden = self.candidate_network(candidate_features)
        if self.global_network is None:
            combined = candidate_hidden
        else:
            if global_features is None:
                raise ValueError("MLP global features are missing")
            global_hidden = self.global_network(global_features)
            combined = torch.cat(
                (
                    global_hidden[:, None, :].expand(-1, candidate_hidden.shape[1], -1),
                    candidate_hidden,
                ),
                dim=2,
            )
        return self.scorer(combined).squeeze(2)


@dataclass(frozen=True, slots=True)
class MLPScorer:
    mode: str
    loss: str
    ranking_lambda: float
    model: SharedValueMLP
    global_mean: np.ndarray | None
    global_scale: np.ndarray | None
    candidate_mean: np.ndarray
    candidate_scale: np.ndarray
    fit_domains: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    loss_trace: tuple[float, ...]
    parameter_count: int
    state_sha256: str

    def predict(self, examples: ObservedValueExamples) -> np.ndarray:
        candidates = (
            examples.candidate_features - self.candidate_mean
        ) / self.candidate_scale
        global_values = _global_matrix(examples, self.mode)
        global_tensor = None
        if global_values is not None:
            assert self.global_mean is not None and self.global_scale is not None
            global_tensor = torch.from_numpy(
                (global_values - self.global_mean) / self.global_scale
            ).to(dtype=torch.float64)
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                global_tensor,
                torch.from_numpy(candidates).to(dtype=torch.float64),
            ).cpu().numpy()
        values = np.asarray(output, dtype=np.float64)
        if values.shape != (examples.specimen_count, 64) or not np.all(np.isfinite(values)):
            raise ValueError("MLP observability prediction changed")
        return values


def _normalizer(
    examples: ObservedValueExamples, mode: str
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    global_values = _global_matrix(examples, mode)
    global_mean = None if global_values is None else np.mean(global_values, axis=0)
    global_scale = None if global_values is None else np.std(global_values, axis=0)
    if global_scale is not None:
        global_scale[global_scale <= np.finfo(np.float64).eps] = 1.0
    candidate_mean = np.mean(examples.candidate_features.reshape(-1, 8), axis=0)
    candidate_scale = np.std(examples.candidate_features.reshape(-1, 8), axis=0)
    candidate_scale[candidate_scale <= np.finfo(np.float64).eps] = 1.0
    return global_mean, global_scale, candidate_mean, candidate_scale


def _mlp_state(
    *,
    model: SharedValueMLP,
    mode: str,
    loss: str,
    ranking_lambda: float,
    global_mean: np.ndarray | None,
    global_scale: np.ndarray | None,
    candidate_mean: np.ndarray,
    candidate_scale: np.ndarray,
    examples: ObservedValueExamples,
    loss_trace: tuple[float, ...],
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "fit_domains": tuple(dict.fromkeys(examples.dataset_ids)),
                "fit_specimen_ids": examples.specimen_ids,
                "loss": loss,
                "loss_trace": loss_trace,
                "mode": mode,
                "ranking_lambda": ranking_lambda,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    arrays = {
        "candidate_mean": candidate_mean,
        "candidate_scale": candidate_scale,
        **({} if global_mean is None else {"global_mean": global_mean}),
        **({} if global_scale is None else {"global_scale": global_scale}),
        **{
            f"model:{name}": value.detach().cpu().numpy()
            for name, value in model.state_dict().items()
        },
    }
    for name in sorted(arrays):
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(arrays[name]).tobytes(order="C"))
    return digest.hexdigest()


def fit_mlp_scorer(
    examples: ObservedValueExamples,
    *,
    outer_domain: str,
    mode: str,
    loss: str,
    ranking_lambda: float = 0.0,
    seed: int = 20260824,
    epochs: int = 50,
) -> MLPScorer:
    """Fit the frozen deterministic small scorer on source specimens only."""

    if (
        type(examples) is not ObservedValueExamples
        or examples.role != "source_train"
        or examples.outer_domain != outer_domain
        or outer_domain in examples.dataset_ids
        or mode not in _MODES
        or loss not in {"top1", "huber", "huber_rank"}
        or (loss == "huber_rank") != (float(ranking_lambda) in {0.1, 0.5, 1.0})
        or (loss != "huber_rank" and float(ranking_lambda) != 0.0)
        or type(seed) is not int
        or epochs != 50
    ):
        raise ValueError("outer domain must be excluded from MLP observability training")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = SharedValueMLP(mode).to(dtype=torch.float64)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count >= 100_000:
        raise ValueError("M1 model exceeds the parameter cap")
    global_mean, global_scale, candidate_mean, candidate_scale = _normalizer(examples, mode)
    global_values = _global_matrix(examples, mode)
    global_array = (
        None
        if global_values is None
        else (global_values - global_mean) / global_scale
    )
    candidates = (examples.candidate_features - candidate_mean) / candidate_scale
    truth = examples.mechanical_values
    domains = tuple(dict.fromkeys(examples.dataset_ids))
    domain_counts = {domain: examples.dataset_ids.count(domain) for domain in domains}
    specimen_weights = np.asarray(
        [1.0 / len(domains) / domain_counts[domain] for domain in examples.dataset_ids],
        dtype=np.float64,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-3, weight_decay=1.0e-4
    )
    trace: list[float] = []
    for _epoch in range(epochs):
        epoch_value = 0.0
        for start in range(0, examples.specimen_count, 128):
            stop = min(start + 128, examples.specimen_count)
            candidate_tensor = torch.from_numpy(candidates[start:stop]).to(dtype=torch.float64)
            global_tensor = (
                None
                if global_array is None
                else torch.from_numpy(global_array[start:stop]).to(dtype=torch.float64)
            )
            target_tensor = torch.from_numpy(truth[start:stop].copy()).to(
                dtype=torch.float64
            )
            weights = torch.from_numpy(specimen_weights[start:stop]).to(dtype=torch.float64)
            weights = weights / torch.sum(weights)
            scores = model(global_tensor, candidate_tensor)
            if loss == "top1":
                selected = torch.argmax(target_tensor, dim=1)
                selected_scores = scores.gather(1, selected[:, None])
                pairwise = functional.softplus(scores - selected_scores)
                pairwise.scatter_(1, selected[:, None], 0.0)
                per_specimen = torch.sum(pairwise, dim=1) / 63.0
                objective = torch.sum(per_specimen * weights)
            else:
                value_loss = functional.smooth_l1_loss(
                    scores, target_tensor, reduction="none"
                ).mean(dim=1)
                objective = torch.sum(value_loss * weights)
                if loss == "huber_rank":
                    target_difference = target_tensor[:, :, None] - target_tensor[:, None, :]
                    score_difference = scores[:, :, None] - scores[:, None, :]
                    mask = target_difference > 0.0
                    weighted = functional.softplus(-score_difference) * target_difference.abs()
                    rank_loss = torch.sum(weighted * mask, dim=(1, 2)) / torch.clamp(
                        torch.sum(mask, dim=(1, 2)), min=1
                    )
                    objective = objective + float(ranking_lambda) * torch.sum(
                        rank_loss * weights
                    )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_value += float(objective.detach()) * (stop - start)
        trace.append(epoch_value / examples.specimen_count)
        if (_epoch + 1) % 10 == 0 or _epoch + 1 == epochs:
            print(
                f"[M1 {mode} {loss} lambda={ranking_lambda:g}] "
                f"epoch {_epoch + 1}/{epochs}",
                flush=True,
            )
    loss_trace = tuple(trace)
    state = _mlp_state(
        model=model,
        mode=mode,
        loss=loss,
        ranking_lambda=float(ranking_lambda),
        global_mean=global_mean,
        global_scale=global_scale,
        candidate_mean=candidate_mean,
        candidate_scale=candidate_scale,
        examples=examples,
        loss_trace=loss_trace,
    )
    return MLPScorer(
        mode=mode,
        loss=loss,
        ranking_lambda=float(ranking_lambda),
        model=model,
        global_mean=global_mean,
        global_scale=global_scale,
        candidate_mean=candidate_mean,
        candidate_scale=candidate_scale,
        fit_domains=tuple(dict.fromkeys(examples.dataset_ids)),
        fit_specimen_ids=examples.specimen_ids,
        loss_trace=loss_trace,
        parameter_count=parameter_count,
        state_sha256=state,
    )


__all__ = [
    "MLPScorer",
    "RidgeScorer",
    "SharedValueMLP",
    "fit_mlp_scorer",
    "fit_ridge_scorer",
]
