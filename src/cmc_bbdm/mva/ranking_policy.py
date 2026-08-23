"""Fixed supervised ranking policy and canonical A5 model packages."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional


class RankingPolicyError(ValueError):
    """Raised when A5 policy data, training, or serialization drifts."""


@dataclass(frozen=True, slots=True)
class RankingExample:
    dataset_id: str
    specimen_id: str
    global_features: np.ndarray
    candidate_features: np.ndarray
    selected_index: int


@dataclass(frozen=True, slots=True)
class FeatureNormalizer:
    global_mean: np.ndarray
    global_scale: np.ndarray
    candidate_mean: np.ndarray
    candidate_scale: np.ndarray

    def transform_global(self, values: object) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1:] != (579,) or not np.all(np.isfinite(array)):
            raise RankingPolicyError("global policy features are invalid")
        return np.asarray((array - self.global_mean) / self.global_scale)

    def transform_candidates(self, values: object) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim < 2 or array.shape[-1] != 8 or not np.all(np.isfinite(array)):
            raise RankingPolicyError("candidate policy features are invalid")
        return np.asarray((array - self.candidate_mean) / self.candidate_scale)


class RankingPolicy(nn.Module):
    """Registered 41,617-parameter shared candidate scorer."""

    def __init__(self) -> None:
        super().__init__()
        self.global_network = nn.Sequential(
            nn.Linear(579, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.candidate_network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        global_features: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            global_features.ndim != 2
            or global_features.shape[1] != 579
            or candidate_features.ndim != 3
            or candidate_features.shape[0] != global_features.shape[0]
            or candidate_features.shape[2] != 8
            or candidate_mask.shape != candidate_features.shape[:2]
            or candidate_mask.dtype != torch.bool
        ):
            raise RankingPolicyError("ranking policy tensor shapes changed")
        global_hidden = self.global_network(global_features)
        candidate_hidden = self.candidate_network(candidate_features)
        expanded = global_hidden[:, None, :].expand(-1, candidate_hidden.shape[1], -1)
        scores = self.scorer(torch.cat((expanded, candidate_hidden), dim=2)).squeeze(2)
        return scores.masked_fill(~candidate_mask, -torch.inf)


@dataclass(frozen=True, slots=True)
class TrainedRankingPolicy:
    model: RankingPolicy
    normalizer: FeatureNormalizer
    loss_trace: tuple[float, ...]
    source_domains: tuple[str, ...]
    source_specimen_ids: tuple[str, ...]
    seed: int
    epochs: int
    batch_states: int
    state_sha256: str

    def score_features(
        self, global_features: np.ndarray, candidate_features: np.ndarray
    ) -> np.ndarray:
        global_array = self.normalizer.transform_global(global_features)
        candidate_array = self.normalizer.transform_candidates(candidate_features)
        if global_array.shape != (579,) or candidate_array.shape[0] < 1:
            raise RankingPolicyError("policy scoring features are invalid")
        global_tensor = torch.from_numpy(global_array[None, :]).to(
            dtype=torch.float64
        )
        candidates = torch.from_numpy(candidate_array[None, :, :]).to(
            dtype=torch.float64
        )
        mask = torch.ones((1, candidates.shape[1]), dtype=torch.bool)
        self.model.eval()
        with torch.no_grad():
            output = self.model(global_tensor, candidates, mask)[0].cpu().numpy()
        return np.asarray(output, dtype=np.float64)

    def score(self, example: RankingExample) -> np.ndarray:
        checked = _validate_example(example)
        return self.score_features(
            checked.global_features, checked.candidate_features
        )


def _readonly(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise RankingPolicyError("policy package array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(shape)
    output.setflags(write=False)
    return output


def _validate_example(value: RankingExample) -> RankingExample:
    if (
        type(value) is not RankingExample
        or type(value.dataset_id) is not str
        or not value.dataset_id
        or type(value.specimen_id) is not str
        or not value.specimen_id
        or type(value.selected_index) is not int
    ):
        raise RankingPolicyError("ranking example identity is invalid")
    global_features = _readonly(value.global_features, (579,))
    candidates = np.asarray(value.candidate_features, dtype=np.float64)
    if (
        candidates.ndim != 2
        or candidates.shape[1] != 8
        or candidates.shape[0] < 2
        or not 0 <= value.selected_index < candidates.shape[0]
    ):
        raise RankingPolicyError("ranking example candidates are invalid")
    frozen_candidates = _readonly(candidates, candidates.shape)
    return RankingExample(
        dataset_id=value.dataset_id,
        specimen_id=value.specimen_id,
        global_features=global_features,
        candidate_features=frozen_candidates,
        selected_index=value.selected_index,
    )


def equal_hierarchy_weights(
    dataset_ids: tuple[str, ...], specimen_ids: tuple[str, ...]
) -> np.ndarray:
    """Assign equal domain, specimen-within-domain, and state mass."""

    if (
        type(dataset_ids) is not tuple
        or type(specimen_ids) is not tuple
        or not dataset_ids
        or len(dataset_ids) != len(specimen_ids)
        or any(type(value) is not str or not value for value in (*dataset_ids, *specimen_ids))
    ):
        raise RankingPolicyError("policy hierarchy is invalid")
    specimen_domain: dict[str, str] = {}
    for dataset_id, specimen_id in zip(dataset_ids, specimen_ids, strict=True):
        previous = specimen_domain.setdefault(specimen_id, dataset_id)
        if previous != dataset_id:
            raise RankingPolicyError("one specimen appears in multiple domains")
    domains = tuple(dict.fromkeys(dataset_ids))
    specimens_per_domain = Counter(specimen_domain.values())
    states_per_specimen = Counter(specimen_ids)
    weights = np.asarray(
        [
            1.0
            / len(domains)
            / specimens_per_domain[dataset_id]
            / states_per_specimen[specimen_id]
            for dataset_id, specimen_id in zip(dataset_ids, specimen_ids, strict=True)
        ],
        dtype=np.float64,
    )
    if not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1.0e-12):
        raise RankingPolicyError("policy hierarchy weights do not sum to one")
    return weights


def pairwise_ranking_loss(
    scores: torch.Tensor,
    *,
    selected_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    state_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted teacher-vs-all pairwise logistic ranking loss."""

    if (
        scores.ndim != 2
        or selected_indices.shape != (scores.shape[0],)
        or candidate_mask.shape != scores.shape
        or candidate_mask.dtype != torch.bool
        or state_weights.shape != (scores.shape[0],)
        or torch.any(state_weights < 0.0)
    ):
        raise RankingPolicyError("pairwise loss tensors are invalid")
    losses: list[torch.Tensor] = []
    for row in range(scores.shape[0]):
        selected = int(selected_indices[row])
        if (
            not 0 <= selected < scores.shape[1]
            or not bool(candidate_mask[row, selected])
            or int(torch.count_nonzero(candidate_mask[row])) < 2
        ):
            raise RankingPolicyError("pairwise teacher selection is invalid")
        other_mask = candidate_mask[row].clone()
        other_mask[selected] = False
        differences = scores[row, other_mask] - scores[row, selected]
        losses.append(torch.mean(functional.softplus(differences)))
    return torch.sum(torch.stack(losses) * state_weights)


def _fit_normalizer(examples: tuple[RankingExample, ...]) -> FeatureNormalizer:
    global_values = np.vstack([example.global_features for example in examples])
    candidate_values = np.vstack(
        [example.candidate_features for example in examples]
    )
    global_mean = np.mean(global_values, axis=0, dtype=np.float64)
    global_scale = np.std(global_values, axis=0, dtype=np.float64)
    candidate_mean = np.mean(candidate_values, axis=0, dtype=np.float64)
    candidate_scale = np.std(candidate_values, axis=0, dtype=np.float64)
    global_scale[global_scale <= np.finfo(np.float64).eps] = 1.0
    candidate_scale[candidate_scale <= np.finfo(np.float64).eps] = 1.0
    global_mean[512:576] = 0.0
    global_scale[512:576] = 1.0
    candidate_mean[2] = 0.0
    candidate_scale[2] = 1.0
    return FeatureNormalizer(
        global_mean=_readonly(global_mean, (579,)),
        global_scale=_readonly(global_scale, (579,)),
        candidate_mean=_readonly(candidate_mean, (8,)),
        candidate_scale=_readonly(candidate_scale, (8,)),
    )


def _batch(
    examples: tuple[RankingExample, ...],
    normalizer: FeatureNormalizer,
    indices: range,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_examples = tuple(examples[index] for index in indices)
    maximum = max(example.candidate_features.shape[0] for example in selected_examples)
    global_values = np.vstack(
        [normalizer.transform_global(example.global_features) for example in selected_examples]
    )
    candidates = np.zeros((len(selected_examples), maximum, 8), dtype=np.float64)
    mask = np.zeros((len(selected_examples), maximum), dtype=np.bool_)
    selected = np.empty(len(selected_examples), dtype=np.int64)
    for row, example in enumerate(selected_examples):
        count = example.candidate_features.shape[0]
        candidates[row, :count] = normalizer.transform_candidates(
            example.candidate_features
        )
        mask[row, :count] = True
        selected[row] = example.selected_index
    return (
        torch.from_numpy(global_values),
        torch.from_numpy(candidates),
        torch.from_numpy(mask),
        torch.from_numpy(selected),
    )


def _model_arrays(model: RankingPolicy) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float64)
        for name, value in model.state_dict().items()
    }


def _policy_state(
    *,
    model: RankingPolicy,
    normalizer: FeatureNormalizer,
    loss_trace: tuple[float, ...],
    source_domains: tuple[str, ...],
    source_specimen_ids: tuple[str, ...],
    seed: int,
    epochs: int,
    batch_states: int,
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "batch_states": batch_states,
                "epochs": epochs,
                "loss_trace": loss_trace,
                "seed": seed,
                "source_domains": source_domains,
                "source_specimen_ids": source_specimen_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    arrays = {
        "candidate_mean": normalizer.candidate_mean,
        "candidate_scale": normalizer.candidate_scale,
        "global_mean": normalizer.global_mean,
        "global_scale": normalizer.global_scale,
        **{f"model:{name}": value for name, value in _model_arrays(model).items()},
    }
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def train_ranking_policy(
    examples: tuple[RankingExample, ...],
    *,
    seed: int,
    epochs: int = 50,
    batch_states: int = 128,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    gradient_clip: float = 5.0,
) -> TrainedRankingPolicy:
    """Train one deterministic full-epoch-gradient A5 policy on CPU float64."""

    if (
        type(examples) is not tuple
        or not examples
        or type(seed) is not int
        or type(epochs) is not int
        or epochs < 1
        or type(batch_states) is not int
        or batch_states < 1
        or learning_rate != 1.0e-3
        or weight_decay != 1.0e-4
        or gradient_clip != 5.0
    ):
        raise RankingPolicyError("ranking training authority changed")
    checked = tuple(_validate_example(example) for example in examples)
    dataset_ids = tuple(example.dataset_id for example in checked)
    specimen_ids = tuple(example.specimen_id for example in checked)
    weights = equal_hierarchy_weights(dataset_ids, specimen_ids)
    normalizer = _fit_normalizer(checked)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = RankingPolicy().to(device="cpu", dtype=torch.float64)
    if sum(parameter.numel() for parameter in model.parameters()) != 41_617:
        raise RankingPolicyError("registered policy parameter count changed")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    trace: list[float] = []
    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        objective = 0.0
        for start in range(0, len(checked), batch_states):
            stop = min(start + batch_states, len(checked))
            global_values, candidates, mask, selected = _batch(
                checked, normalizer, range(start, stop)
            )
            scores = model(global_values, candidates, mask)
            loss = pairwise_ranking_loss(
                scores,
                selected_indices=selected,
                candidate_mask=mask,
                state_weights=torch.from_numpy(weights[start:stop]),
            )
            loss.backward()
            objective += float(loss.detach())
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()
        trace.append(objective)
    model.eval()
    domains = tuple(dict.fromkeys(dataset_ids))
    specimens = tuple(dict.fromkeys(specimen_ids))
    loss_trace = tuple(trace)
    state_sha256 = _policy_state(
        model=model,
        normalizer=normalizer,
        loss_trace=loss_trace,
        source_domains=domains,
        source_specimen_ids=specimens,
        seed=seed,
        epochs=epochs,
        batch_states=batch_states,
    )
    return TrainedRankingPolicy(
        model=model,
        normalizer=normalizer,
        loss_trace=loss_trace,
        source_domains=domains,
        source_specimen_ids=specimens,
        seed=seed,
        epochs=epochs,
        batch_states=batch_states,
        state_sha256=state_sha256,
    )


def save_policy_package(
    path: str | Path, policy: TrainedRankingPolicy
) -> Path:
    """Atomically save a content-bound policy package."""

    if type(policy) is not TrainedRankingPolicy:
        raise RankingPolicyError("issued trained policy is required")
    expected = _policy_state(
        model=policy.model,
        normalizer=policy.normalizer,
        loss_trace=policy.loss_trace,
        source_domains=policy.source_domains,
        source_specimen_ids=policy.source_specimen_ids,
        seed=policy.seed,
        epochs=policy.epochs,
        batch_states=policy.batch_states,
    )
    if expected != policy.state_sha256:
        raise RankingPolicyError("trained policy content changed")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "batch_states": policy.batch_states,
            "epochs": policy.epochs,
            "loss_trace": policy.loss_trace,
            "schema_version": 1,
            "seed": policy.seed,
            "source_domains": policy.source_domains,
            "source_specimen_ids": policy.source_specimen_ids,
            "state_sha256": policy.state_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    arrays = {
        "metadata": np.frombuffer(metadata, dtype=np.uint8),
        "normalizer_candidate_mean": policy.normalizer.candidate_mean,
        "normalizer_candidate_scale": policy.normalizer.candidate_scale,
        "normalizer_global_mean": policy.normalizer.global_mean,
        "normalizer_global_scale": policy.normalizer.global_scale,
        **{f"model_{name}": value for name, value in _model_arrays(policy.model).items()},
    }
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".npz"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    loaded = load_policy_package(destination)
    if loaded.state_sha256 != policy.state_sha256:
        raise RankingPolicyError("saved policy validation failed")
    return destination


def load_policy_package(path: str | Path) -> TrainedRankingPolicy:
    """Load and independently validate one canonical policy package."""

    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            names = set(archive.files)
            model = RankingPolicy().to(device="cpu", dtype=torch.float64)
            model_names = tuple(model.state_dict())
            expected_names = {
                "metadata",
                "normalizer_candidate_mean",
                "normalizer_candidate_scale",
                "normalizer_global_mean",
                "normalizer_global_scale",
                *(f"model_{name}" for name in model_names),
            }
            if names != expected_names:
                raise RankingPolicyError("policy package file roster changed")
            metadata = json.loads(np.asarray(archive["metadata"], dtype=np.uint8).tobytes())
            if set(metadata) != {
                "batch_states",
                "epochs",
                "loss_trace",
                "schema_version",
                "seed",
                "source_domains",
                "source_specimen_ids",
                "state_sha256",
            } or metadata["schema_version"] != 1:
                raise RankingPolicyError("policy package metadata changed")
            normalizer = FeatureNormalizer(
                global_mean=_readonly(archive["normalizer_global_mean"], (579,)),
                global_scale=_readonly(archive["normalizer_global_scale"], (579,)),
                candidate_mean=_readonly(archive["normalizer_candidate_mean"], (8,)),
                candidate_scale=_readonly(archive["normalizer_candidate_scale"], (8,)),
            )
            state_dict = {
                name: torch.from_numpy(
                    np.asarray(archive[f"model_{name}"], dtype=np.float64).copy()
                )
                for name in model_names
            }
            model.load_state_dict(state_dict, strict=True)
    except RankingPolicyError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise RankingPolicyError("policy package cannot be loaded") from error
    model.eval()
    policy = TrainedRankingPolicy(
        model=model,
        normalizer=normalizer,
        loss_trace=tuple(float(value) for value in metadata["loss_trace"]),
        source_domains=tuple(str(value) for value in metadata["source_domains"]),
        source_specimen_ids=tuple(
            str(value) for value in metadata["source_specimen_ids"]
        ),
        seed=int(metadata["seed"]),
        epochs=int(metadata["epochs"]),
        batch_states=int(metadata["batch_states"]),
        state_sha256=str(metadata["state_sha256"]),
    )
    expected = _policy_state(
        model=policy.model,
        normalizer=policy.normalizer,
        loss_trace=policy.loss_trace,
        source_domains=policy.source_domains,
        source_specimen_ids=policy.source_specimen_ids,
        seed=policy.seed,
        epochs=policy.epochs,
        batch_states=policy.batch_states,
    )
    if expected != policy.state_sha256:
        raise RankingPolicyError("policy package content digest changed")
    return policy


__all__ = [
    "FeatureNormalizer",
    "RankingExample",
    "RankingPolicy",
    "RankingPolicyError",
    "TrainedRankingPolicy",
    "equal_hierarchy_weights",
    "load_policy_package",
    "pairwise_ranking_loss",
    "save_policy_package",
    "train_ranking_policy",
]
