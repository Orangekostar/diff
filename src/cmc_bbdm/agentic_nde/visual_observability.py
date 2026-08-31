"""Source-isolated P1 visual observability models, scores, and action metrics."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from .p1 import P1Config
from .surface_cells import SurfaceCellAuthority

_ROLES = frozenset(
    {"source_train", "source_validation", "outer_inference", "outer_evaluation"}
)
_REPRESENTATIONS = frozenset({"OLD", "GLOBAL", "LOCAL", "LOCAL_GLOBAL"})
_MODEL_LOCK = threading.RLock()
_HEX = frozenset("0123456789abcdef")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("P1 visual array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _examples_state(value: VisualExamples) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "dataset_ids": value.dataset_ids,
                "feature_control": value.feature_control,
                "outer_domain": value.outer_domain,
                "role": value.role,
                "specimen_ids": value.specimen_ids,
            }
        )
    )
    for array in (
        value.initial_embeddings,
        value.current_predictions,
        value.candidate_features,
        value.global_embeddings,
        value.local_embeddings,
    ):
        digest.update(array.tobytes(order="C"))
    if value.mechanical_values is not None:
        digest.update(value.mechanical_values.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class VisualExamples:
    outer_domain: str
    role: str
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    initial_embeddings: np.ndarray
    current_predictions: np.ndarray
    candidate_features: np.ndarray
    global_embeddings: np.ndarray
    local_embeddings: np.ndarray
    mechanical_values: np.ndarray | None
    feature_control: str
    state_sha256: str

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)

    @classmethod
    def create(
        cls,
        *,
        outer_domain: str,
        role: str,
        specimen_ids: tuple[str, ...],
        dataset_ids: tuple[str, ...],
        initial_embeddings: object,
        current_predictions: object,
        candidate_features: object,
        global_embeddings: object,
        local_embeddings: object,
        mechanical_values: object | None,
        feature_control: str,
    ) -> VisualExamples:
        count = len(specimen_ids)
        if (
            not outer_domain
            or role not in _ROLES
            or count < 1
            or len(dataset_ids) != count
            or len(set(specimen_ids)) != count
            or any(not value for value in (*specimen_ids, *dataset_ids))
            or not feature_control
        ):
            raise ValueError("P1 visual example identity is invalid")
        source_role = role in {"source_train", "source_validation"}
        if (
            source_role and outer_domain in dataset_ids
        ) or (
            not source_role and set(dataset_ids) != {outer_domain}
        ):
            raise ValueError("P1 outer-domain isolation changed")
        if (role == "outer_inference") != (mechanical_values is None):
            raise ValueError("P1 target labels violate the inference barrier")
        result = cls(
            outer_domain=outer_domain,
            role=role,
            specimen_ids=specimen_ids,
            dataset_ids=dataset_ids,
            initial_embeddings=_readonly(
                initial_embeddings, dtype="<f8", shape=(count, 512)
            ),
            current_predictions=_readonly(
                current_predictions, dtype="<f8", shape=(count,)
            ),
            candidate_features=_readonly(
                candidate_features, dtype="<f8", shape=(count, 64, 8)
            ),
            global_embeddings=_readonly(
                global_embeddings, dtype="<f4", shape=(count, 512)
            ),
            local_embeddings=_readonly(
                local_embeddings, dtype="<f4", shape=(count, 64, 512)
            ),
            mechanical_values=(
                None
                if mechanical_values is None
                else _readonly(
                    mechanical_values, dtype="<f8", shape=(count, 64)
                )
            ),
            feature_control=feature_control,
            state_sha256="",
        )
        object.__setattr__(result, "state_sha256", _examples_state(result))
        return result


@dataclass(frozen=True, slots=True)
class P1DeployableAuthority:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    current_predictions: np.ndarray
    predictor_state_sha256: tuple[str, ...]
    surface_authority_state_sha256: str
    target_source_sha256: str
    state_sha256: str

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)


@dataclass(frozen=True, slots=True)
class P1MechanicalLabels:
    outer_domain: str
    role: str
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    mechanical_values: np.ndarray
    target_source_sha256: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class P1OuterExamples:
    source: VisualExamples
    inference: VisualExamples
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class SourceCandidateSelection:
    candidate_id: str
    ndcg_10: float
    next_action_regret: float
    parameter_count: int
    aggregates: pl.DataFrame
    state_sha256: str


@dataclass(frozen=True, slots=True)
class FusionSelection:
    value: float
    audit: pl.DataFrame
    state_sha256: str


@dataclass(frozen=True, slots=True)
class OuterVisualModelFit:
    outer_domain: str
    correct_representation: str
    correct_config_id: str
    global_config_id: str
    old_config_id: str
    correct_lambda: float
    global_lambda: float
    models: Mapping[str, Any]
    model_feature_controls: Mapping[str, str]
    selection_audit: pl.DataFrame
    selection_state_sha256: str


@dataclass(frozen=True, slots=True)
class _HeadSpecification:
    family: str
    config_id: str
    alpha: float | None
    parameter_count: int


@dataclass(frozen=True, slots=True)
class FrozenOuterScores:
    outer_domain: str
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    methods: tuple[str, ...]
    scores: Mapping[str, np.ndarray]
    model_state_sha256: Mapping[str, str]
    selection_state_sha256: str
    inference_state_sha256: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenC0Scores:
    method: str
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    scores: np.ndarray
    source_sha256: str
    state_sha256: str


def _valid_hash(value: object) -> bool:
    return type(value) is str and len(value) == 64 and not (set(value) - _HEX)


def _a2_scan(config: P1Config) -> pl.LazyFrame:
    if type(config) is not P1Config:
        raise TypeError("issued P1Config is required")
    path = config.project_root / config.sources["a2_oracle_values"].path
    return pl.scan_parquet(path).filter(
        (pl.col("method") == "mechanical_oracle")
        & (pl.col("step") == 0)
        & (pl.col("from_level") == 0)
        & (pl.col("to_level") == 1)
        & (pl.col("nominal_checkpoint") == 0.0625)
    )


def _groups(table: pl.DataFrame) -> dict[tuple[str, str], pl.DataFrame]:
    output: dict[tuple[str, str], pl.DataFrame] = {}
    for key, group in table.partition_by(
        ["dataset_id", "specimen_id"], as_dict=True, include_key=False
    ).items():
        normalized = key if isinstance(key, tuple) else (str(key),)
        if len(normalized) != 2:
            raise ValueError("P1 specimen grouping changed")
        identity = str(normalized[0]), str(normalized[1])
        if identity in output:
            raise ValueError("P1 specimen group is duplicated")
        output[identity] = group.sort("cell_index")
    return output


def load_p1_deployable_authority(
    config: P1Config, surface: SurfaceCellAuthority
) -> P1DeployableAuthority:
    """Load only A2 old-state inputs; do not select the mechanical label column."""

    if (
        type(surface) is not SurfaceCellAuthority
        or surface.specimen_count != config.authorized_specimen_count
    ):
        raise ValueError("P1 deployable roster is not the P0R authority")
    try:
        table = (
            _a2_scan(config)
            .select(
                "dataset_id",
                "specimen_id",
                "cell_index",
                "current_prediction",
                "p_a_predictor_state_sha256",
            )
            .collect()
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ValueError("P1 deployable A2 inputs cannot be read") from error
    if table.height != config.target_rows:
        raise ValueError("P1 deployable A2 row count changed")
    grouped = _groups(table)
    expected_keys = set(zip(surface.dataset_ids, surface.specimen_ids, strict=True))
    if set(grouped) != expected_keys:
        raise ValueError("P1 deployable A2 specimen roster changed")
    current = np.empty(surface.specimen_count, dtype="<f8")
    predictors: list[str] = []
    for index, identity in enumerate(
        zip(surface.dataset_ids, surface.specimen_ids, strict=True)
    ):
        rows = grouped[identity]
        values = set(rows["current_prediction"])
        hashes = set(rows["p_a_predictor_state_sha256"])
        if (
            rows.height != 64
            or tuple(rows["cell_index"]) != tuple(range(64))
            or len(values) != 1
            or len(hashes) != 1
        ):
            raise ValueError("P1 deployable A2 state changed")
        current[index] = float(values.pop())
        predictor = str(hashes.pop())
        if not _valid_hash(predictor):
            raise ValueError("P1 deployable predictor hash changed")
        predictors.append(predictor)
    frozen_current = _readonly(
        current, dtype="<f8", shape=(surface.specimen_count,)
    )
    metadata = {
        "dataset_ids": surface.dataset_ids,
        "predictor_state_sha256": predictors,
        "specimen_ids": surface.specimen_ids,
        "surface_authority_state_sha256": surface.state_sha256,
        "target_source_sha256": config.sources["a2_oracle_values"].sha256,
    }
    digest = hashlib.sha256(_canonical_json(metadata))
    digest.update(frozen_current.tobytes(order="C"))
    return P1DeployableAuthority(
        specimen_ids=surface.specimen_ids,
        dataset_ids=surface.dataset_ids,
        current_predictions=frozen_current,
        predictor_state_sha256=tuple(predictors),
        surface_authority_state_sha256=surface.state_sha256,
        target_source_sha256=config.sources["a2_oracle_values"].sha256,
        state_sha256=digest.hexdigest(),
    )


def _load_label_subset(
    config: P1Config,
    deployable: P1DeployableAuthority,
    *,
    outer_domain: str,
    target: bool,
) -> P1MechanicalLabels:
    if (
        type(deployable) is not P1DeployableAuthority
        or outer_domain not in config.domain_order
        or len(deployable.specimen_ids) != config.authorized_specimen_count
        or deployable.target_source_sha256
        != config.sources["a2_oracle_values"].sha256
    ):
        raise ValueError("P1 label authority identity changed")
    predicate = (
        pl.col("dataset_id") == outer_domain
        if target
        else pl.col("dataset_id") != outer_domain
    )
    try:
        table = (
            _a2_scan(config)
            .filter(predicate)
            .select("dataset_id", "specimen_id", "cell_index", "primary_value")
            .collect()
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ValueError("P1 mechanical labels cannot be read") from error
    expected_indices = tuple(
        index
        for index, domain in enumerate(deployable.dataset_ids)
        if (domain == outer_domain) is target
    )
    expected_ids = tuple(deployable.specimen_ids[index] for index in expected_indices)
    expected_domains = tuple(
        deployable.dataset_ids[index] for index in expected_indices
    )
    grouped = _groups(table)
    expected_keys = set(zip(expected_domains, expected_ids, strict=True))
    if set(grouped) != expected_keys or table.height != len(expected_ids) * 64:
        raise ValueError("P1 mechanical label roster changed")
    values = np.empty((len(expected_ids), 64), dtype="<f8")
    for index, identity in enumerate(zip(expected_domains, expected_ids, strict=True)):
        rows = grouped[identity]
        if rows.height != 64 or tuple(rows["cell_index"]) != tuple(range(64)):
            raise ValueError("P1 mechanical cell roster changed")
        values[index] = rows["primary_value"].to_numpy()
    frozen = _readonly(values, dtype="<f8", shape=(len(expected_ids), 64))
    role = "outer_evaluation" if target else "source_train"
    metadata = {
        "dataset_ids": expected_domains,
        "outer_domain": outer_domain,
        "role": role,
        "specimen_ids": expected_ids,
        "target_source_sha256": deployable.target_source_sha256,
    }
    digest = hashlib.sha256(_canonical_json(metadata))
    digest.update(frozen.tobytes(order="C"))
    return P1MechanicalLabels(
        outer_domain=outer_domain,
        role=role,
        specimen_ids=expected_ids,
        dataset_ids=expected_domains,
        mechanical_values=frozen,
        target_source_sha256=deployable.target_source_sha256,
        state_sha256=digest.hexdigest(),
    )


def load_p1_source_labels(
    config: P1Config,
    deployable: P1DeployableAuthority,
    *,
    outer_domain: str,
) -> P1MechanicalLabels:
    """Load A2 labels for exactly the five source domains."""

    return _load_label_subset(
        config, deployable, outer_domain=outer_domain, target=False
    )


def assemble_p1_outer_examples(
    *,
    outer_domain: str,
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    initial_embeddings: object,
    current_predictions: object,
    candidate_features: object,
    global_embeddings: object,
    local_embeddings: object,
    source_labels: P1MechanicalLabels,
    feature_control: str,
) -> P1OuterExamples:
    """Split deployable inputs while keeping outer labels outside inference."""

    count = len(specimen_ids)
    if (
        not outer_domain
        or count < 2
        or len(dataset_ids) != count
        or outer_domain not in dataset_ids
        or type(source_labels) is not P1MechanicalLabels
        or source_labels.outer_domain != outer_domain
        or source_labels.role != "source_train"
        or outer_domain in source_labels.dataset_ids
        or not feature_control
    ):
        raise ValueError("P1 outer example authority changed")
    source_indices = tuple(
        index for index, domain in enumerate(dataset_ids) if domain != outer_domain
    )
    target_indices = tuple(
        index for index, domain in enumerate(dataset_ids) if domain == outer_domain
    )
    expected_source_ids = tuple(specimen_ids[index] for index in source_indices)
    expected_source_domains = tuple(dataset_ids[index] for index in source_indices)
    if (
        source_labels.specimen_ids != expected_source_ids
        or source_labels.dataset_ids != expected_source_domains
        or not source_indices
        or not target_indices
    ):
        raise ValueError("P1 source label roster changed")
    initial = np.asarray(initial_embeddings)
    current = np.asarray(current_predictions)
    candidates = np.asarray(candidate_features)
    global_values = np.asarray(global_embeddings)
    local_values = np.asarray(local_embeddings)
    if (
        initial.shape != (count, 512)
        or current.shape != (count,)
        or candidates.shape != (count, 64, 8)
        or global_values.shape != (count, 512)
        or local_values.shape != (count, 64, 512)
    ):
        raise ValueError("P1 deployable input shape changed")

    def take(indices: tuple[int, ...], *, role: str) -> VisualExamples:
        selection = np.asarray(indices, dtype=np.int64)
        return VisualExamples.create(
            outer_domain=outer_domain,
            role=role,
            specimen_ids=tuple(specimen_ids[index] for index in indices),
            dataset_ids=tuple(dataset_ids[index] for index in indices),
            initial_embeddings=initial[selection],
            current_predictions=current[selection],
            candidate_features=candidates[selection],
            global_embeddings=global_values[selection],
            local_embeddings=local_values[selection],
            mechanical_values=(
                source_labels.mechanical_values if role == "source_train" else None
            ),
            feature_control=feature_control,
        )

    source = take(source_indices, role="source_train")
    inference = take(target_indices, role="outer_inference")
    payload = {
        "feature_control": feature_control,
        "inference_state_sha256": inference.state_sha256,
        "outer_domain": outer_domain,
        "source_label_state_sha256": source_labels.state_sha256,
        "source_state_sha256": source.state_sha256,
    }
    return P1OuterExamples(
        source=source,
        inference=inference,
        source_indices=source_indices,
        target_indices=target_indices,
        state_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
    )


def freeze_outer_scores(
    inference: VisualExamples,
    *,
    scores: Mapping[str, object],
    model_state_sha256: Mapping[str, str],
    selection_state_sha256: str,
) -> FrozenOuterScores:
    """Hash target scores before any outer mechanical/CAI label is requested."""

    if (
        type(inference) is not VisualExamples
        or inference.role != "outer_inference"
        or inference.mechanical_values is not None
        or not scores
        or set(scores) != set(model_state_sha256)
        or not _valid_hash(selection_state_sha256)
        or any(not _valid_hash(value) for value in model_state_sha256.values())
    ):
        raise ValueError("P1 target scores cannot be frozen")
    methods = tuple(sorted(scores))
    frozen_scores: dict[str, np.ndarray] = {}
    digest = hashlib.sha256(
        _canonical_json(
            {
                "dataset_ids": inference.dataset_ids,
                "inference_state_sha256": inference.state_sha256,
                "methods": methods,
                "model_state_sha256": {
                    method: model_state_sha256[method] for method in methods
                },
                "outer_domain": inference.outer_domain,
                "selection_state_sha256": selection_state_sha256,
                "specimen_ids": inference.specimen_ids,
            }
        )
    )
    for method in methods:
        if not method or not method.replace("_", "").isalnum():
            raise ValueError("P1 target score method is invalid")
        value = _readonly(
            scores[method],
            dtype="<f8",
            shape=(inference.specimen_count, 64),
        )
        frozen_scores[method] = value
        digest.update(method.encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return FrozenOuterScores(
        outer_domain=inference.outer_domain,
        specimen_ids=inference.specimen_ids,
        dataset_ids=inference.dataset_ids,
        methods=methods,
        scores=MappingProxyType(frozen_scores),
        model_state_sha256=MappingProxyType(
            {method: model_state_sha256[method] for method in methods}
        ),
        selection_state_sha256=selection_state_sha256,
        inference_state_sha256=inference.state_sha256,
        state_sha256=digest.hexdigest(),
    )


def load_p1_target_labels(
    config: P1Config,
    deployable: P1DeployableAuthority,
    *,
    outer_domain: str,
    frozen_scores: FrozenOuterScores,
) -> P1MechanicalLabels:
    """Load outer labels only after a matching immutable score table exists."""

    if type(frozen_scores) is not FrozenOuterScores:
        raise TypeError("frozen outer scores are required before target labels")
    expected_ids = tuple(
        specimen
        for specimen, domain in zip(
            deployable.specimen_ids, deployable.dataset_ids, strict=True
        )
        if domain == outer_domain
    )
    if (
        frozen_scores.outer_domain != outer_domain
        or frozen_scores.specimen_ids != expected_ids
        or set(frozen_scores.dataset_ids) != {outer_domain}
    ):
        raise ValueError("frozen target scores do not match the target roster")
    return _load_label_subset(
        config, deployable, outer_domain=outer_domain, target=True
    )


def attach_target_labels(
    inference: VisualExamples,
    labels: P1MechanicalLabels,
    *,
    frozen_scores: FrozenOuterScores,
) -> VisualExamples:
    """Issue an evaluation view only for the exact previously frozen inference."""

    if (
        type(inference) is not VisualExamples
        or inference.role != "outer_inference"
        or type(labels) is not P1MechanicalLabels
        or labels.role != "outer_evaluation"
        or type(frozen_scores) is not FrozenOuterScores
        or frozen_scores.inference_state_sha256 != inference.state_sha256
        or labels.outer_domain != inference.outer_domain
        or labels.specimen_ids != inference.specimen_ids
        or labels.dataset_ids != inference.dataset_ids
    ):
        raise ValueError("P1 target evaluation barrier changed")
    return VisualExamples.create(
        outer_domain=inference.outer_domain,
        role="outer_evaluation",
        specimen_ids=inference.specimen_ids,
        dataset_ids=inference.dataset_ids,
        initial_embeddings=inference.initial_embeddings,
        current_predictions=inference.current_predictions,
        candidate_features=inference.candidate_features,
        global_embeddings=inference.global_embeddings,
        local_embeddings=inference.local_embeddings,
        mechanical_values=labels.mechanical_values,
        feature_control=inference.feature_control,
    )


def load_frozen_c0_scores(
    config: P1Config, deployable: P1DeployableAuthority
) -> FrozenC0Scores:
    """Bind exact MVD O2 predictions without reading its historical teacher."""

    if (
        type(config) is not P1Config
        or type(deployable) is not P1DeployableAuthority
        or deployable.specimen_count != config.authorized_specimen_count
    ):
        raise ValueError("P1 C0 authority identity changed")
    path = config.project_root / config.sources["mvd_o2_predictions"].path
    try:
        table = (
            pl.scan_parquet(path)
            .filter(pl.col("method") == "o2_global_candidate")
            .select(
                "outer_domain",
                "dataset_id",
                "specimen_id",
                "cell_index",
                "predicted_value",
            )
            .collect()
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ValueError("P1 frozen C0 scores cannot be read") from error
    if (
        table.height != config.target_rows
        or table.filter(pl.col("outer_domain") != pl.col("dataset_id")).height
    ):
        raise ValueError("P1 frozen C0 score roster changed")
    grouped = _groups(table)
    expected_keys = set(
        zip(deployable.dataset_ids, deployable.specimen_ids, strict=True)
    )
    if set(grouped) != expected_keys:
        raise ValueError("P1 frozen C0 specimen roster changed")
    scores = np.empty((deployable.specimen_count, 64), dtype="<f8")
    for index, identity in enumerate(
        zip(deployable.dataset_ids, deployable.specimen_ids, strict=True)
    ):
        rows = grouped[identity]
        if rows.height != 64 or tuple(rows["cell_index"]) != tuple(range(64)):
            raise ValueError("P1 frozen C0 cell roster changed")
        scores[index] = rows["predicted_value"].to_numpy()
    frozen = _readonly(
        scores, dtype="<f8", shape=(deployable.specimen_count, 64)
    )
    source_hash = config.sources["mvd_o2_predictions"].sha256
    digest = hashlib.sha256(
        _canonical_json(
            {
                "dataset_ids": deployable.dataset_ids,
                "method": "o2_global_candidate",
                "source_sha256": source_hash,
                "specimen_ids": deployable.specimen_ids,
            }
        )
    )
    digest.update(frozen.tobytes(order="C"))
    return FrozenC0Scores(
        method="o2_global_candidate",
        specimen_ids=deployable.specimen_ids,
        dataset_ids=deployable.dataset_ids,
        scores=frozen,
        source_sha256=source_hash,
        state_sha256=digest.hexdigest(),
    )


def subset_visual_examples(
    examples: VisualExamples,
    *,
    included_domains: tuple[str, ...],
    role: str,
) -> VisualExamples:
    """Take a labeled source subset without permitting the outer domain."""

    if (
        type(examples) is not VisualExamples
        or examples.role not in {"source_train", "source_validation"}
        or examples.mechanical_values is None
        or role not in {"source_train", "source_validation"}
        or not included_domains
        or len(set(included_domains)) != len(included_domains)
        or examples.outer_domain in included_domains
        or not set(included_domains) <= set(examples.dataset_ids)
    ):
        raise ValueError("P1 source subset request is invalid")
    indices = np.asarray(
        [
            index
            for index, domain in enumerate(examples.dataset_ids)
            if domain in included_domains
        ],
        dtype=np.int64,
    )
    return VisualExamples.create(
        outer_domain=examples.outer_domain,
        role=role,
        specimen_ids=tuple(examples.specimen_ids[index] for index in indices),
        dataset_ids=tuple(examples.dataset_ids[index] for index in indices),
        initial_embeddings=examples.initial_embeddings[indices],
        current_predictions=examples.current_predictions[indices],
        candidate_features=examples.candidate_features[indices],
        global_embeddings=examples.global_embeddings[indices],
        local_embeddings=examples.local_embeddings[indices],
        mechanical_values=examples.mechanical_values[indices],
        feature_control=examples.feature_control,
    )


def replace_surface_features(
    examples: VisualExamples,
    *,
    global_embeddings: object,
    local_embeddings: object,
    feature_control: str,
) -> VisualExamples:
    """Replace only label-free surface inputs for a preregistered control."""

    if type(examples) is not VisualExamples:
        raise TypeError("issued VisualExamples are required")
    return VisualExamples.create(
        outer_domain=examples.outer_domain,
        role=examples.role,
        specimen_ids=examples.specimen_ids,
        dataset_ids=examples.dataset_ids,
        initial_embeddings=examples.initial_embeddings,
        current_predictions=examples.current_predictions,
        candidate_features=examples.candidate_features,
        global_embeddings=global_embeddings,
        local_embeddings=local_embeddings,
        mechanical_values=examples.mechanical_values,
        feature_control=feature_control,
    )


def representation_matrix(
    examples: VisualExamples, representation: str, *, dtype: object = "<f8"
) -> np.ndarray:
    """Construct one shared candidate matrix from deployable inputs only."""

    if type(examples) is not VisualExamples or representation not in _REPRESENTATIONS:
        raise ValueError("P1 representation is invalid")
    count = examples.specimen_count
    initial = np.broadcast_to(examples.initial_embeddings[:, None, :], (count, 64, 512))
    current = np.broadcast_to(examples.current_predictions[:, None, None], (count, 64, 1))
    values = [initial, current, examples.candidate_features]
    if representation in {"GLOBAL", "LOCAL_GLOBAL"}:
        values.append(
            np.broadcast_to(examples.global_embeddings[:, None, :], (count, 64, 512))
        )
    if representation in {"LOCAL", "LOCAL_GLOBAL"}:
        values.append(examples.local_embeddings)
    output = np.ascontiguousarray(np.concatenate(values, axis=2), dtype=dtype)
    expected = {
        "OLD": 521,
        "GLOBAL": 1033,
        "LOCAL": 1033,
        "LOCAL_GLOBAL": 1545,
    }[representation]
    if output.shape != (count, 64, expected) or not np.all(np.isfinite(output)):
        raise ValueError("P1 representation matrix is invalid")
    return output


def _specimen_weights(examples: VisualExamples) -> np.ndarray:
    domains = tuple(dict.fromkeys(examples.dataset_ids))
    counts = Counter(examples.dataset_ids)
    values = np.asarray(
        [1.0 / len(domains) / counts[domain] for domain in examples.dataset_ids],
        dtype=np.float64,
    )
    if not np.isclose(float(np.sum(values)), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("P1 specimen weights changed")
    return values


def _normalization(
    matrix: np.ndarray, examples: VisualExamples
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, cells, dimension = matrix.shape
    specimen = _specimen_weights(examples)
    row_weights = np.repeat(specimen / cells, cells)
    flat = matrix.reshape(count * cells, dimension)
    mean = np.sum(flat * row_weights[:, None], axis=0, dtype=np.float64)
    centered = flat - mean
    variance = np.sum(
        centered * centered * row_weights[:, None], axis=0, dtype=np.float64
    )
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    return mean, scale, row_weights


def _model_state(
    metadata: dict[str, object], arrays: dict[str, np.ndarray]
) -> str:
    digest = hashlib.sha256(_canonical_json(metadata))
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RidgeVisualScorer:
    representation: str
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    fit_domains: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    parameter_count: int
    state_sha256: str

    @property
    def config_id(self) -> str:
        return f"ridge_alpha_{self.alpha:g}"

    def predict(self, examples: VisualExamples) -> np.ndarray:
        matrix = representation_matrix(examples, self.representation, dtype="<f8")
        flat = matrix.reshape(-1, matrix.shape[2])
        values = self.intercept + ((flat - self.mean) / self.scale) @ self.coefficients
        output = np.asarray(values, dtype="<f8").reshape(examples.specimen_count, 64)
        if not np.all(np.isfinite(output)):
            raise ValueError("P1 Ridge prediction is invalid")
        return output


def fit_ridge_family(
    examples: VisualExamples,
    *,
    representation: str,
    alphas: tuple[float, ...],
) -> tuple[RidgeVisualScorer, ...]:
    """Fit all frozen Ridge alphas with one weighted eigendecomposition."""

    if (
        type(examples) is not VisualExamples
        or examples.role != "source_train"
        or examples.mechanical_values is None
        or examples.outer_domain in examples.dataset_ids
        or representation not in _REPRESENTATIONS
        or not alphas
        or len(set(alphas)) != len(alphas)
        or any(not np.isfinite(value) or value <= 0.0 for value in alphas)
    ):
        raise ValueError("P1 Ridge fit must use labeled source domains only")
    matrix = representation_matrix(examples, representation, dtype="<f8")
    mean, scale, normalized_weights = _normalization(matrix, examples)
    rows = examples.specimen_count * 64
    dimension = matrix.shape[2]
    x = ((matrix.reshape(rows, dimension) - mean) / scale).astype(
        np.float64, copy=False
    )
    y = examples.mechanical_values.reshape(rows)
    y_mean = float(np.sum(y * normalized_weights, dtype=np.float64))
    centered_y = y - y_mean
    fit_weights = normalized_weights * rows
    sqrt_weight = np.sqrt(fit_weights)
    xw = x * sqrt_weight[:, None]
    yw = centered_y * sqrt_weight
    if rows < dimension:
        gram = xw @ xw.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)

        def coefficients(alpha: float) -> np.ndarray:
            projected = eigenvectors.T @ yw
            return xw.T @ (eigenvectors @ (projected / (eigenvalues + alpha)))

    else:
        gram = xw.T @ xw
        right = xw.T @ yw
        eigenvalues, eigenvectors = np.linalg.eigh(gram)

        def coefficients(alpha: float) -> np.ndarray:
            return eigenvectors @ ((eigenvectors.T @ right) / (eigenvalues + alpha))

    fit_domains = tuple(dict.fromkeys(examples.dataset_ids))
    models: list[RidgeVisualScorer] = []
    for alpha in alphas:
        coefficient = np.ascontiguousarray(coefficients(float(alpha)), dtype="<f8")
        metadata = {
            "alpha": float(alpha),
            "fit_domains": fit_domains,
            "fit_specimen_ids": examples.specimen_ids,
            "representation": representation,
        }
        state = _model_state(
            metadata,
            {"coefficients": coefficient, "mean": mean, "scale": scale},
        )
        models.append(
            RidgeVisualScorer(
                representation=representation,
                alpha=float(alpha),
                mean=_readonly(mean, dtype="<f8", shape=(dimension,)),
                scale=_readonly(scale, dtype="<f8", shape=(dimension,)),
                coefficients=_readonly(
                    coefficient, dtype="<f8", shape=(dimension,)
                ),
                intercept=y_mean,
                fit_domains=fit_domains,
                fit_specimen_ids=examples.specimen_ids,
                parameter_count=dimension + 1,
                state_sha256=state,
            )
        )
    return tuple(models)


@dataclass(frozen=True, slots=True)
class MLPVisualScorer:
    representation: str
    model: Any
    torch: Any
    device: str
    mean: np.ndarray
    scale: np.ndarray
    fit_domains: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    loss_trace: tuple[float, ...]
    parameter_count: int
    state_sha256: str

    @property
    def config_id(self) -> str:
        return "mlp_smooth_l1_32_16"

    def predict(self, examples: VisualExamples) -> np.ndarray:
        matrix = representation_matrix(examples, self.representation, dtype="<f4")
        normalized = np.ascontiguousarray(
            (matrix - self.mean.astype(np.float32)) / self.scale.astype(np.float32),
            dtype="<f4",
        )
        outputs: list[np.ndarray] = []
        self.model.eval()
        with _MODEL_LOCK, self.torch.inference_mode():
            for start in range(0, examples.specimen_count, 128):
                stop = min(start + 128, examples.specimen_count)
                tensor = self.torch.from_numpy(normalized[start:stop]).to(
                    self.device, non_blocking=False
                )
                values = self.model(tensor).squeeze(-1).detach().to("cpu").numpy()
                outputs.append(np.asarray(values, dtype="<f4"))
        combined = np.concatenate(outputs, axis=0).astype("<f8")
        if combined.shape != (examples.specimen_count, 64) or not np.all(
            np.isfinite(combined)
        ):
            raise ValueError("P1 MLP prediction is invalid")
        return combined


def _torch_model_state(model: Any) -> dict[str, np.ndarray]:
    return {
        f"model:{name}": np.ascontiguousarray(value.detach().cpu().numpy())
        for name, value in model.state_dict().items()
    }


def fit_mlp_scorer(
    examples: VisualExamples,
    *,
    representation: str,
    seed: int,
    epochs: int,
    device: str,
) -> MLPVisualScorer:
    """Fit the preregistered deterministic shared 32-16 Smooth-L1 head."""

    if (
        type(examples) is not VisualExamples
        or examples.role != "source_train"
        or examples.mechanical_values is None
        or examples.outer_domain in examples.dataset_ids
        or representation not in _REPRESENTATIONS
        or type(seed) is not int
        or epochs != 50
        or device != "cuda:0"
    ):
        raise ValueError("P1 MLP fit must use the frozen source-only contract")
    try:
        import torch
        from torch import nn
        from torch.nn import functional
    except Exception as error:  # pragma: no cover - dependency failure
        raise RuntimeError("torch is required for the P1 MLP") from error
    if not torch.cuda.is_available():
        raise ValueError("formal P1 MLP CUDA device is unavailable")
    matrix = representation_matrix(examples, representation, dtype="<f4")
    mean, scale, _row_weights = _normalization(
        matrix.astype("<f8", copy=False), examples
    )
    normalized = np.ascontiguousarray(
        (matrix - mean.astype(np.float32)) / scale.astype(np.float32), dtype="<f4"
    )
    truth = examples.mechanical_values.astype("<f4", copy=False)
    specimen_weights = _specimen_weights(examples).astype("<f4")
    dimension = matrix.shape[2]
    with _MODEL_LOCK:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_benchmark = torch.backends.cudnn.benchmark
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            model = nn.Sequential(
                nn.Linear(dimension, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            ).to(device=device, dtype=torch.float32)
            parameter_count = sum(value.numel() for value in model.parameters())
            if parameter_count >= 100_000:
                raise ValueError("P1 MLP exceeds the preregistered parameter cap")
            optimizer = torch.optim.Adam(
                model.parameters(), lr=1.0e-3, weight_decay=1.0e-4
            )
            trace: list[float] = []
            model.train()
            for _epoch in range(epochs):
                total = 0.0
                for start in range(0, examples.specimen_count, 128):
                    stop = min(start + 128, examples.specimen_count)
                    inputs = torch.from_numpy(normalized[start:stop]).to(
                        device, non_blocking=False
                    )
                    target = torch.from_numpy(truth[start:stop].copy()).to(
                        device, non_blocking=False
                    )
                    weights = torch.from_numpy(specimen_weights[start:stop].copy()).to(
                        device, non_blocking=False
                    )
                    weights = weights / torch.sum(weights)
                    scores = model(inputs).squeeze(-1)
                    per_specimen = functional.smooth_l1_loss(
                        scores, target, reduction="none"
                    ).mean(dim=1)
                    objective = torch.sum(per_specimen * weights)
                    optimizer.zero_grad(set_to_none=True)
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total += float(objective.detach().to("cpu")) * (stop - start)
                trace.append(total / examples.specimen_count)
            model.eval()
        finally:
            torch.use_deterministic_algorithms(old_deterministic)
            torch.backends.cudnn.benchmark = old_benchmark
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
    metadata = {
        "epochs": epochs,
        "fit_domains": tuple(dict.fromkeys(examples.dataset_ids)),
        "fit_specimen_ids": examples.specimen_ids,
        "loss_trace": trace,
        "representation": representation,
        "seed": seed,
    }
    arrays = {
        "mean": np.ascontiguousarray(mean, dtype="<f8"),
        "scale": np.ascontiguousarray(scale, dtype="<f8"),
        **_torch_model_state(model),
    }
    state = _model_state(metadata, arrays)
    return MLPVisualScorer(
        representation=representation,
        model=model,
        torch=torch,
        device=device,
        mean=_readonly(mean, dtype="<f8", shape=(dimension,)),
        scale=_readonly(scale, dtype="<f8", shape=(dimension,)),
        fit_domains=tuple(dict.fromkeys(examples.dataset_ids)),
        fit_specimen_ids=examples.specimen_ids,
        loss_trace=tuple(trace),
        parameter_count=parameter_count,
        state_sha256=state,
    )


def _order(scores: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(range(64), key=lambda cell: (-float(scores[cell]), cell)))


def stable_rank_percentiles(scores: object) -> np.ndarray:
    """Return frozen ordinal within-state percentiles, with lower-cell tie break."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim not in {1, 2} or values.shape[-1] != 64 or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("P1 score vector shape changed")
    rows = values.reshape(-1, 64)
    output = np.empty_like(rows)
    for row_index, row in enumerate(rows):
        for position, cell in enumerate(_order(row)):
            output[row_index, cell] = (63 - position) / 63.0
    return output.reshape(values.shape)


def fuse_rank_scores(old_scores: object, visual_scores: object, value: float) -> np.ndarray:
    """Apply the preregistered simple rank-space fusion."""

    if value not in {0.0, 0.25, 0.5, 0.75, 1.0}:
        raise ValueError("P1 fusion lambda is not preregistered")
    old = stable_rank_percentiles(old_scores)
    visual = stable_rank_percentiles(visual_scores)
    if old.shape != visual.shape:
        raise ValueError("P1 fusion score shapes differ")
    return np.asarray((1.0 - value) * old + value * visual, dtype="<f8")


def center_prior_scores() -> np.ndarray:
    """Return the deterministic no-image center prior for cells 0..63."""

    values = np.asarray(
        [
            -((row - 3.5) ** 2 + (column - 3.5) ** 2)
            for row in range(8)
            for column in range(8)
        ],
        dtype="<f8",
    )
    values.setflags(write=False)
    return values


@dataclass(frozen=True, slots=True)
class ActionMetrics:
    next_action_regret: float
    one_step_cai_utility: float
    spearman: float
    ndcg_10: float
    recall_5: float
    top_10_percent_overlap: float
    top_1_oracle_match: float


def evaluate_action_scores(mechanical_values: object, predicted_scores: object) -> ActionMetrics:
    """Evaluate one 64-cell score vector at the physical-specimen level."""

    truth = np.asarray(mechanical_values, dtype=np.float64)
    scores = np.asarray(predicted_scores, dtype=np.float64)
    if (
        truth.shape != (64,)
        or scores.shape != (64,)
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError("P1 action metric input is invalid")
    predicted = _order(scores)
    oracle = _order(truth)
    selected = predicted[0]
    association = (
        0.0
        if np.ptp(truth) <= np.finfo(np.float64).eps
        or np.ptp(scores) <= np.finfo(np.float64).eps
        else float(spearmanr(truth, scores).statistic)
    )
    if not np.isfinite(association):
        association = 0.0
    relevance = truth - float(np.min(truth))
    weights = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    denominator = float(np.sum(relevance[list(oracle[:10])] * weights))
    ndcg = (
        1.0
        if denominator <= np.finfo(np.float64).eps
        else float(np.sum(relevance[list(predicted[:10])] * weights) / denominator)
    )
    top5 = set(oracle[:5])
    top10_percent = set(oracle[:7])
    return ActionMetrics(
        next_action_regret=float(truth[oracle[0]] - truth[selected]),
        one_step_cai_utility=float(truth[selected]),
        spearman=association,
        ndcg_10=ndcg,
        recall_5=len(top5 & set(predicted[:5])) / 5.0,
        top_10_percent_overlap=(
            len(top10_percent & set(predicted[:7])) / len(top10_percent)
        ),
        top_1_oracle_match=float(selected == oracle[0]),
    )


def select_source_candidate(
    metrics: pl.DataFrame, *, domain_order: tuple[str, ...]
) -> SourceCandidateSelection:
    """Apply the frozen source-only NDCG/regret/size/ID selection order."""

    required = {
        "candidate_id",
        "validation_domain",
        "ndcg_10",
        "next_action_regret",
        "parameter_count",
    }
    if (
        type(metrics) is not pl.DataFrame
        or not required <= set(metrics.columns)
        or len(domain_order) != 5
        or len(set(domain_order)) != 5
        or set(metrics["validation_domain"]) != set(domain_order)
        or metrics.unique(subset=["candidate_id", "validation_domain"]).height
        != metrics.height
        or not bool(
            metrics.select(
                pl.col("ndcg_10").is_finite().all()
                & pl.col("next_action_regret").is_finite().all()
            ).item()
        )
    ):
        raise ValueError("P1 source selection metrics changed")
    candidates = tuple(sorted(str(value) for value in metrics["candidate_id"].unique()))
    if not candidates or any(not value for value in candidates):
        raise ValueError("P1 source selection candidate identity changed")
    aggregates: list[dict[str, object]] = []
    for candidate_id in candidates:
        selected = metrics.filter(pl.col("candidate_id") == candidate_id)
        parameters = {int(value) for value in selected["parameter_count"]}
        if selected.height != len(domain_order) or len(parameters) != 1:
            raise ValueError("P1 source selection candidate coverage changed")
        parameter_count = parameters.pop()
        if parameter_count < 0:
            raise ValueError("P1 source selection parameter count changed")
        aggregates.append(
            {
                "candidate_id": candidate_id,
                "ndcg_10": float(
                    np.mean(selected["ndcg_10"].to_numpy(), dtype=np.float64)
                ),
                "next_action_regret": float(
                    np.mean(
                        selected["next_action_regret"].to_numpy(), dtype=np.float64
                    )
                ),
                "parameter_count": parameter_count,
            }
        )
    chosen = min(
        aggregates,
        key=lambda row: (
            -float(row["ndcg_10"]),
            float(row["next_action_regret"]),
            int(row["parameter_count"]),
            str(row["candidate_id"]),
        ),
    )
    table = pl.DataFrame(
        [
            {
                **row,
                "selected": row["candidate_id"] == chosen["candidate_id"],
            }
            for row in aggregates
        ]
    ).sort("candidate_id")
    payload = {
        "aggregates": table.to_dicts(),
        "domain_order": domain_order,
        "selected_candidate_id": chosen["candidate_id"],
    }
    return SourceCandidateSelection(
        candidate_id=str(chosen["candidate_id"]),
        ndcg_10=float(chosen["ndcg_10"]),
        next_action_regret=float(chosen["next_action_regret"]),
        parameter_count=int(chosen["parameter_count"]),
        aggregates=table,
        state_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
    )


def select_fusion_lambda(
    *,
    mechanical_values: object,
    dataset_ids: tuple[str, ...],
    old_scores: object,
    visual_scores: object,
    values: tuple[float, ...],
) -> FusionSelection:
    """Select the rank-fusion coefficient from five source domains only."""

    truth = np.asarray(mechanical_values, dtype=np.float64)
    old = np.asarray(old_scores, dtype=np.float64)
    visual = np.asarray(visual_scores, dtype=np.float64)
    count = len(dataset_ids)
    domain_order = tuple(dict.fromkeys(dataset_ids))
    if (
        truth.shape != (count, 64)
        or old.shape != truth.shape
        or visual.shape != truth.shape
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(old))
        or not np.all(np.isfinite(visual))
        or len(domain_order) != 5
        or any(not value for value in dataset_ids)
        or values != (0.0, 0.25, 0.5, 0.75, 1.0)
    ):
        raise ValueError("P1 fusion selection authority changed")
    rows: list[dict[str, object]] = []
    for value in values:
        fused = fuse_rank_scores(old, visual, value)
        metrics = [
            evaluate_action_scores(truth[index], fused[index])
            for index in range(count)
        ]
        for domain in domain_order:
            indices = [index for index, item in enumerate(dataset_ids) if item == domain]
            rows.append(
                {
                    "candidate_id": f"lambda_{value:g}",
                    "validation_domain": domain,
                    "ndcg_10": float(
                        np.mean([metrics[index].ndcg_10 for index in indices])
                    ),
                    "next_action_regret": float(
                        np.mean(
                            [metrics[index].next_action_regret for index in indices]
                        )
                    ),
                    "parameter_count": 0,
                    "lambda": value,
                }
            )
    audit = pl.DataFrame(rows).sort(["candidate_id", "validation_domain"])
    selected = select_source_candidate(audit, domain_order=domain_order)
    value_lookup = {
        f"lambda_{value:g}": value for value in values
    }
    chosen = float(value_lookup[selected.candidate_id])
    audit = audit.with_columns(
        (pl.col("candidate_id") == selected.candidate_id).alias("selected")
    )
    payload = {
        "audit": audit.to_dicts(),
        "selection_state_sha256": selected.state_sha256,
        "value": chosen,
    }
    return FusionSelection(
        value=chosen,
        audit=audit,
        state_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
    )


def _head_specification(model: object) -> _HeadSpecification:
    config_id = getattr(model, "config_id", None)
    parameter_count = getattr(model, "parameter_count", None)
    state = getattr(model, "state_sha256", None)
    if (
        type(config_id) is not str
        or not config_id
        or type(parameter_count) is not int
        or parameter_count < 1
        or not _valid_hash(state)
    ):
        raise ValueError("P1 fitted head identity changed")
    if config_id.startswith("ridge_alpha_"):
        try:
            alpha = float(config_id.removeprefix("ridge_alpha_"))
        except ValueError as error:
            raise ValueError("P1 Ridge configuration identity changed") from error
        family = "ridge"
    elif config_id == "mlp_smooth_l1_32_16":
        alpha = None
        family = "mlp"
    else:
        raise ValueError("P1 head configuration is not preregistered")
    return _HeadSpecification(
        family=family,
        config_id=config_id,
        alpha=alpha,
        parameter_count=parameter_count,
    )


def _fit_head_candidates(
    examples: VisualExamples,
    *,
    representation: str,
    ridge_alphas: tuple[float, ...],
    model_seed: int,
    epochs: int,
    device: str,
) -> tuple[tuple[object, _HeadSpecification], ...]:
    ridge = fit_ridge_family(
        examples, representation=representation, alphas=ridge_alphas
    )
    mlp = fit_mlp_scorer(
        examples,
        representation=representation,
        seed=model_seed,
        epochs=epochs,
        device=device,
    )
    values = tuple((model, _head_specification(model)) for model in (*ridge, mlp))
    ids = tuple(specification.config_id for _, specification in values)
    if len(values) != 5 or len(set(ids)) != len(ids):
        raise ValueError("P1 head candidate roster changed")
    return values


def _fit_selected_head(
    examples: VisualExamples,
    *,
    representation: str,
    specification: _HeadSpecification,
    ridge_alphas: tuple[float, ...],
    model_seed: int,
    epochs: int,
    device: str,
) -> object:
    if specification.family == "ridge":
        models = fit_ridge_family(
            examples, representation=representation, alphas=ridge_alphas
        )
        matches = [model for model in models if model.config_id == specification.config_id]
        if len(matches) != 1:
            raise ValueError("P1 selected Ridge head is unavailable")
        return matches[0]
    if specification.family != "mlp":
        raise ValueError("P1 selected head family changed")
    model = fit_mlp_scorer(
        examples,
        representation=representation,
        seed=model_seed,
        epochs=epochs,
        device=device,
    )
    if model.config_id != specification.config_id:
        raise ValueError("P1 selected MLP head is unavailable")
    return model


def _mean_action_metrics(
    truth: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    values = [
        evaluate_action_scores(truth[index], scores[index])
        for index in range(truth.shape[0])
    ]
    return (
        float(np.mean([value.ndcg_10 for value in values], dtype=np.float64)),
        float(
            np.mean(
                [value.next_action_regret for value in values], dtype=np.float64
            )
        ),
    )


def _validate_outer_example_controls(
    correct: P1OuterExamples,
    controls: tuple[P1OuterExamples, ...],
    c0_source_scores: object,
) -> np.ndarray:
    if type(correct) is not P1OuterExamples or any(
        type(value) is not P1OuterExamples for value in controls
    ):
        raise ValueError("P1 outer control examples changed")
    for value in controls:
        if (
            value.source.outer_domain != correct.source.outer_domain
            or value.source.specimen_ids != correct.source.specimen_ids
            or value.source.dataset_ids != correct.source.dataset_ids
            or value.inference.specimen_ids != correct.inference.specimen_ids
            or value.inference.dataset_ids != correct.inference.dataset_ids
            or value.source.mechanical_values is None
            or value.inference.mechanical_values is not None
        ):
            raise ValueError("P1 outer control roster or label barrier changed")
    scores = np.asarray(c0_source_scores, dtype=np.float64)
    if (
        correct.source.mechanical_values is None
        or scores.shape != (correct.source.specimen_count, 64)
        or not np.all(np.isfinite(scores))
        or len(set(correct.source.dataset_ids)) != 5
    ):
        raise ValueError("P1 source-only C0 score authority changed")
    return scores


def fit_outer_visual_models(
    *,
    correct: P1OuterExamples,
    shuffled: P1OuterExamples,
    wrong_orientation: P1OuterExamples,
    spatial_derangement: P1OuterExamples,
    c0_source_scores: object,
    ridge_alphas: tuple[float, ...],
    fusion_values: tuple[float, ...],
    model_seed: int,
    epochs: int,
    device: str,
) -> OuterVisualModelFit:
    """Select and refit P1 heads using labels from five source domains only."""

    c0 = _validate_outer_example_controls(
        correct,
        (shuffled, wrong_orientation, spatial_derangement),
        c0_source_scores,
    )
    if (
        ridge_alphas != (0.1, 1.0, 10.0, 100.0)
        or fusion_values != (0.0, 0.25, 0.5, 0.75, 1.0)
        or type(model_seed) is not int
        or epochs != 50
        or device != "cuda:0"
    ):
        raise ValueError("P1 model-selection roster changed")
    source = correct.source
    truth = source.mechanical_values
    assert truth is not None
    source_domains = tuple(dict.fromkeys(source.dataset_ids))
    source_positions = {
        specimen: index for index, specimen in enumerate(source.specimen_ids)
    }
    audit_rows: list[dict[str, object]] = []
    oof: dict[str, dict[str, np.ndarray]] = {}
    specifications: dict[str, dict[str, _HeadSpecification]] = {}
    selections: dict[str, SourceCandidateSelection] = {}
    for representation in ("OLD", "GLOBAL", "LOCAL", "LOCAL_GLOBAL"):
        representation_oof: dict[str, np.ndarray] = {}
        representation_specs: dict[str, _HeadSpecification] = {}
        for validation_domain in source_domains:
            train_domains = tuple(
                domain for domain in source_domains if domain != validation_domain
            )
            train = subset_visual_examples(
                source, included_domains=train_domains, role="source_train"
            )
            validation = subset_visual_examples(
                source,
                included_domains=(validation_domain,),
                role="source_validation",
            )
            fitted = _fit_head_candidates(
                train,
                representation=representation,
                ridge_alphas=ridge_alphas,
                model_seed=model_seed,
                epochs=epochs,
                device=device,
            )
            validation_indices = np.asarray(
                [source_positions[value] for value in validation.specimen_ids],
                dtype=np.int64,
            )
            validation_truth = validation.mechanical_values
            assert validation_truth is not None
            for model, specification in fitted:
                previous = representation_specs.setdefault(
                    specification.config_id, specification
                )
                if previous != specification:
                    raise ValueError("P1 head specification changed across folds")
                scores = np.asarray(model.predict(validation), dtype=np.float64)
                if scores.shape != validation_truth.shape:
                    raise ValueError("P1 inner prediction shape changed")
                matrix = representation_oof.setdefault(
                    specification.config_id,
                    np.full((source.specimen_count, 64), np.nan, dtype=np.float64),
                )
                matrix[validation_indices] = scores
                ndcg, regret = _mean_action_metrics(validation_truth, scores)
                audit_rows.append(
                    {
                        "alpha": specification.alpha,
                        "candidate_id": specification.config_id,
                        "family": specification.family,
                        "feature_control": source.feature_control,
                        "fit_domains": "|".join(train_domains),
                        "lambda": None,
                        "method": None,
                        "model_state_sha256": str(model.state_sha256),
                        "ndcg_10": ndcg,
                        "next_action_regret": regret,
                        "outer_domain": source.outer_domain,
                        "parameter_count": specification.parameter_count,
                        "representation": representation,
                        "role": "source_validation",
                        "selected": False,
                        "stage": "HEAD_INNER",
                        "validation_domain": validation_domain,
                        "validation_specimen_count": validation.specimen_count,
                    }
                )
        if any(not np.all(np.isfinite(value)) for value in representation_oof.values()):
            raise ValueError("P1 inner OOF prediction coverage changed")
        candidate_metrics = pl.DataFrame(
            [
                {
                    "candidate_id": row["candidate_id"],
                    "validation_domain": row["validation_domain"],
                    "ndcg_10": row["ndcg_10"],
                    "next_action_regret": row["next_action_regret"],
                    "parameter_count": row["parameter_count"],
                }
                for row in audit_rows
                if row["stage"] == "HEAD_INNER"
                and row["representation"] == representation
            ]
        )
        selection = select_source_candidate(
            candidate_metrics, domain_order=source_domains
        )
        for row in audit_rows:
            if (
                row["stage"] == "HEAD_INNER"
                and row["representation"] == representation
            ):
                row["selected"] = row["candidate_id"] == selection.candidate_id
        for row in selection.aggregates.iter_rows(named=True):
            specification = representation_specs[str(row["candidate_id"])]
            audit_rows.append(
                {
                    "alpha": specification.alpha,
                    "candidate_id": row["candidate_id"],
                    "family": specification.family,
                    "feature_control": source.feature_control,
                    "fit_domains": "|".join(source_domains),
                    "lambda": None,
                    "method": None,
                    "model_state_sha256": None,
                    "ndcg_10": row["ndcg_10"],
                    "next_action_regret": row["next_action_regret"],
                    "outer_domain": source.outer_domain,
                    "parameter_count": row["parameter_count"],
                    "representation": representation,
                    "role": "source_validation",
                    "selected": row["selected"],
                    "stage": "HEAD_AGGREGATE",
                    "validation_domain": "EQUAL_SOURCE_MEAN",
                    "validation_specimen_count": source.specimen_count,
                }
            )
        oof[representation] = representation_oof
        specifications[representation] = representation_specs
        selections[representation] = selection

    route_rows: list[dict[str, object]] = []
    for representation in ("LOCAL", "LOCAL_GLOBAL"):
        chosen_id = selections[representation].candidate_id
        for row in audit_rows:
            if (
                row["stage"] == "HEAD_INNER"
                and row["representation"] == representation
                and row["candidate_id"] == chosen_id
            ):
                route_rows.append(
                    {
                        "candidate_id": representation,
                        "validation_domain": row["validation_domain"],
                        "ndcg_10": row["ndcg_10"],
                        "next_action_regret": row["next_action_regret"],
                        "parameter_count": row["parameter_count"],
                    }
                )
    route = select_source_candidate(
        pl.DataFrame(route_rows), domain_order=source_domains
    )
    correct_representation = route.candidate_id
    for row in route_rows:
        audit_rows.append(
            {
                "alpha": specifications[correct_representation][
                    selections[correct_representation].candidate_id
                ].alpha,
                "candidate_id": row["candidate_id"],
                "family": specifications[row["candidate_id"]][
                    selections[row["candidate_id"]].candidate_id
                ].family,
                "feature_control": source.feature_control,
                "fit_domains": "|".join(source_domains),
                "lambda": None,
                "method": None,
                "model_state_sha256": None,
                "ndcg_10": row["ndcg_10"],
                "next_action_regret": row["next_action_regret"],
                "outer_domain": source.outer_domain,
                "parameter_count": row["parameter_count"],
                "representation": row["candidate_id"],
                "role": "source_validation",
                "selected": row["candidate_id"] == correct_representation,
                "stage": "CORRECT_ROUTE",
                "validation_domain": row["validation_domain"],
                "validation_specimen_count": sum(
                    domain == row["validation_domain"] for domain in source.dataset_ids
                ),
            }
        )
    correct_config_id = selections[correct_representation].candidate_id
    global_config_id = selections["GLOBAL"].candidate_id
    old_config_id = selections["OLD"].candidate_id
    correct_fusion = select_fusion_lambda(
        mechanical_values=truth,
        dataset_ids=source.dataset_ids,
        old_scores=c0,
        visual_scores=oof[correct_representation][correct_config_id],
        values=fusion_values,
    )
    global_fusion = select_fusion_lambda(
        mechanical_values=truth,
        dataset_ids=source.dataset_ids,
        old_scores=c0,
        visual_scores=oof["GLOBAL"][global_config_id],
        values=fusion_values,
    )
    for stage, fusion in (
        ("FUSION_CORRECT", correct_fusion),
        ("FUSION_GLOBAL", global_fusion),
    ):
        representation = correct_representation if stage == "FUSION_CORRECT" else "GLOBAL"
        for row in fusion.audit.iter_rows(named=True):
            audit_rows.append(
                {
                    "alpha": None,
                    "candidate_id": row["candidate_id"],
                    "family": "rank_fusion",
                    "feature_control": source.feature_control,
                    "fit_domains": "|".join(source_domains),
                    "lambda": row["lambda"],
                    "method": None,
                    "model_state_sha256": None,
                    "ndcg_10": row["ndcg_10"],
                    "next_action_regret": row["next_action_regret"],
                    "outer_domain": source.outer_domain,
                    "parameter_count": 0,
                    "representation": representation,
                    "role": "source_validation",
                    "selected": row["selected"],
                    "stage": stage,
                    "validation_domain": row["validation_domain"],
                    "validation_specimen_count": sum(
                        domain == row["validation_domain"]
                        for domain in source.dataset_ids
                    ),
                }
            )

    correct_spec = specifications[correct_representation][correct_config_id]
    global_spec = specifications["GLOBAL"][global_config_id]
    old_spec = specifications["OLD"][old_config_id]

    def fit(examples: P1OuterExamples, representation: str, spec: _HeadSpecification):
        return _fit_selected_head(
            examples.source,
            representation=representation,
            specification=spec,
            ridge_alphas=ridge_alphas,
            model_seed=model_seed,
            epochs=epochs,
            device=device,
        )

    models = {
        "old_refit_diagnostic": fit(correct, "OLD", old_spec),
        "proposed": fit(correct, correct_representation, correct_spec),
        "c2_global_context": fit(correct, "GLOBAL", global_spec),
        "c3_shuffled_surface": fit(
            shuffled, correct_representation, correct_spec
        ),
        "c4_wrong_orientation": fit(
            wrong_orientation, correct_representation, correct_spec
        ),
        "c5_spatial_derangement": fit(
            spatial_derangement, correct_representation, correct_spec
        ),
        "c3_shuffled_global": fit(shuffled, "GLOBAL", global_spec),
    }
    controls = {
        "old_refit_diagnostic": correct.source.feature_control,
        "proposed": correct.source.feature_control,
        "c2_global_context": correct.source.feature_control,
        "c3_shuffled_surface": shuffled.source.feature_control,
        "c4_wrong_orientation": wrong_orientation.source.feature_control,
        "c5_spatial_derangement": spatial_derangement.source.feature_control,
        "c3_shuffled_global": shuffled.source.feature_control,
    }
    for method, model in models.items():
        specification = _head_specification(model)
        audit_rows.append(
            {
                "alpha": specification.alpha,
                "candidate_id": specification.config_id,
                "family": specification.family,
                "feature_control": controls[method],
                "fit_domains": "|".join(source_domains),
                "lambda": (
                    global_fusion.value
                    if method in {"c2_global_context", "c3_shuffled_global"}
                    else correct_fusion.value
                    if method not in {"old_refit_diagnostic"}
                    else None
                ),
                "method": method,
                "model_state_sha256": str(model.state_sha256),
                "ndcg_10": None,
                "next_action_regret": None,
                "outer_domain": source.outer_domain,
                "parameter_count": specification.parameter_count,
                "representation": str(model.representation),
                "role": "source_train",
                "selected": True,
                "stage": "FINAL_FIT",
                "validation_domain": None,
                "validation_specimen_count": None,
            }
        )
    audit = pl.DataFrame(audit_rows, infer_schema_length=None).sort(
        [
            "outer_domain",
            "stage",
            "representation",
            "candidate_id",
            "validation_domain",
            "method",
        ],
        nulls_last=True,
    )
    selection_payload = {
        "control_example_states": {
            "correct": correct.state_sha256,
            "shuffled": shuffled.state_sha256,
            "spatial_derangement": spatial_derangement.state_sha256,
            "wrong_orientation": wrong_orientation.state_sha256,
        },
        "correct_config_id": correct_config_id,
        "correct_fusion_state_sha256": correct_fusion.state_sha256,
        "correct_lambda": correct_fusion.value,
        "correct_representation": correct_representation,
        "global_config_id": global_config_id,
        "global_fusion_state_sha256": global_fusion.state_sha256,
        "global_lambda": global_fusion.value,
        "head_selection_states": {
            key: value.state_sha256 for key, value in sorted(selections.items())
        },
        "old_config_id": old_config_id,
        "outer_domain": source.outer_domain,
        "route_selection_state_sha256": route.state_sha256,
    }
    return OuterVisualModelFit(
        outer_domain=source.outer_domain,
        correct_representation=correct_representation,
        correct_config_id=correct_config_id,
        global_config_id=global_config_id,
        old_config_id=old_config_id,
        correct_lambda=correct_fusion.value,
        global_lambda=global_fusion.value,
        models=MappingProxyType(models),
        model_feature_controls=MappingProxyType(controls),
        selection_audit=audit,
        selection_state_sha256=hashlib.sha256(
            _canonical_json(selection_payload)
        ).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class SpecimenBootstrapEffect:
    effect_id: str
    control: str
    proposed: str
    value_column: str
    point_estimate: float
    lower: float
    upper: float
    improved_domains: int
    specimen_count: int
    domain_effects: tuple[tuple[str, float], ...]
    seed: int
    resamples: int
    state_sha256: str


def paired_specimen_bootstrap(
    table: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    control: str,
    proposed: str,
    value_column: str,
    seed: int,
    resamples: int,
    effect_id: str,
) -> SpecimenBootstrapEffect:
    """Pair specimens, resample within domain, then give every domain equal mass."""

    required = {"outer_domain", "specimen_id", "method", value_column}
    if (
        type(table) is not pl.DataFrame
        or not required <= set(table.columns)
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
        or not control
        or not proposed
        or control == proposed
        or type(seed) is not int
        or type(resamples) is not int
        or resamples < 1
        or not effect_id
    ):
        raise ValueError("P1 paired bootstrap contract changed")
    selected = table.filter(pl.col("method").is_in([control, proposed])).select(
        "outer_domain", "specimen_id", "method", value_column
    )
    if (
        set(selected["outer_domain"]) != set(domain_order)
        or set(selected["method"]) != {control, proposed}
        or selected.unique(
            subset=["outer_domain", "specimen_id", "method"]
        ).height
        != selected.height
        or not bool(selected.select(pl.col(value_column).is_finite().all()).item())
    ):
        raise ValueError("P1 paired bootstrap rows changed")
    differences: dict[str, np.ndarray] = {}
    domain_effects: list[tuple[str, float]] = []
    specimen_count = 0
    for domain in domain_order:
        domain_rows = selected.filter(pl.col("outer_domain") == domain)
        left = domain_rows.filter(pl.col("method") == control).select(
            "specimen_id", pl.col(value_column).alias("control")
        )
        right = domain_rows.filter(pl.col("method") == proposed).select(
            "specimen_id", pl.col(value_column).alias("proposed")
        )
        paired = left.join(right, on="specimen_id", how="inner").sort("specimen_id")
        if (
            paired.height < 1
            or paired.height != left.height
            or paired.height != right.height
        ):
            raise ValueError("P1 paired specimen roster changed")
        values = np.asarray(
            paired["control"].to_numpy() - paired["proposed"].to_numpy(),
            dtype=np.float64,
        )
        differences[domain] = values
        effect = float(np.mean(values, dtype=np.float64))
        domain_effects.append((domain, effect))
        specimen_count += values.size
    point = float(np.mean([value for _, value in domain_effects], dtype=np.float64))
    generator = np.random.Generator(np.random.PCG64(seed))
    samples = np.zeros(resamples, dtype=np.float64)
    for domain in domain_order:
        values = differences[domain]
        indices = generator.integers(
            0, values.size, size=(resamples, values.size), endpoint=False
        )
        samples += np.mean(values[indices], axis=1, dtype=np.float64) / len(
            domain_order
        )
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    metadata = {
        "control": control,
        "domain_effects": domain_effects,
        "effect_id": effect_id,
        "point_estimate": point,
        "proposed": proposed,
        "resamples": resamples,
        "seed": seed,
        "specimen_count": specimen_count,
        "value_column": value_column,
    }
    digest = hashlib.sha256(_canonical_json(metadata))
    digest.update(samples.tobytes(order="C"))
    return SpecimenBootstrapEffect(
        effect_id=effect_id,
        control=control,
        proposed=proposed,
        value_column=value_column,
        point_estimate=point,
        lower=float(lower),
        upper=float(upper),
        improved_domains=sum(value > 0.0 for _, value in domain_effects),
        specimen_count=specimen_count,
        domain_effects=tuple(domain_effects),
        seed=seed,
        resamples=resamples,
        state_sha256=digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class P1Decision:
    status: str
    go: bool
    authorized_route: str | None
    oracle_gap_closure: float | None
    spatial_conditions: Mapping[str, bool]
    global_conditions: Mapping[str, bool]
    ranking_improvement: bool
    effects: Mapping[str, SpecimenBootstrapEffect]
    state_sha256: str


def _equal_domain_metric(
    table: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    method: str,
    column: str,
) -> float:
    grouped = (
        table.filter(pl.col("method") == method)
        .group_by("outer_domain")
        .agg(pl.col(column).mean().alias("value"))
    )
    lookup = dict(zip(grouped["outer_domain"], grouped["value"], strict=True))
    if set(lookup) != set(domain_order):
        raise ValueError("P1 equal-domain metric coverage changed")
    return float(np.mean([lookup[domain] for domain in domain_order], dtype=np.float64))


def decide_p1(
    specimen_auebc: pl.DataFrame,
    ranking_metrics: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> P1Decision:
    """Apply the frozen spatial/global/descriptive/no-go decision tree."""

    required_methods = {
        "c0_mvd_m1_o2",
        "proposed",
        "c2_global_context",
        "c3_shuffled_surface",
        "c4_wrong_orientation",
        "c5_spatial_derangement",
        "c3_shuffled_global",
        "mechanical_oracle_diagnostic",
    }
    if (
        type(specimen_auebc) is not pl.DataFrame
        or not {"outer_domain", "specimen_id", "method", "cai_auebc"}
        <= set(specimen_auebc.columns)
        or type(ranking_metrics) is not pl.DataFrame
        or not {
            "outer_domain",
            "specimen_id",
            "method",
            "ndcg_10",
            "next_action_regret",
        }
        <= set(ranking_metrics.columns)
        or not required_methods <= set(specimen_auebc["method"])
        or not {"c0_mvd_m1_o2", "proposed"} <= set(ranking_metrics["method"])
        or len(domain_order) != 6
    ):
        raise ValueError("P1 decision inputs changed")

    def effect(control: str, proposed: str, effect_id: str) -> SpecimenBootstrapEffect:
        return paired_specimen_bootstrap(
            specimen_auebc,
            domain_order=domain_order,
            control=control,
            proposed=proposed,
            value_column="cai_auebc",
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
            effect_id=effect_id,
        )

    effects = {
        "c0_minus_proposed": effect(
            "c0_mvd_m1_o2", "proposed", "c0_minus_proposed_cai_auebc"
        ),
        "shuffled_minus_proposed": effect(
            "c3_shuffled_surface",
            "proposed",
            "shuffled_minus_proposed_cai_auebc",
        ),
        "wrong_minus_proposed": effect(
            "c4_wrong_orientation",
            "proposed",
            "wrong_orientation_minus_proposed_cai_auebc",
        ),
        "deranged_minus_proposed": effect(
            "c5_spatial_derangement",
            "proposed",
            "spatial_derangement_minus_proposed_cai_auebc",
        ),
        "global_minus_proposed": effect(
            "c2_global_context", "proposed", "global_minus_proposed_cai_auebc"
        ),
        "c0_minus_global": effect(
            "c0_mvd_m1_o2",
            "c2_global_context",
            "c0_minus_global_cai_auebc",
        ),
        "shuffled_global_minus_global": effect(
            "c3_shuffled_global",
            "c2_global_context",
            "shuffled_global_minus_global_cai_auebc",
        ),
    }
    old = _equal_domain_metric(
        specimen_auebc,
        domain_order=domain_order,
        method="c0_mvd_m1_o2",
        column="cai_auebc",
    )
    proposed = _equal_domain_metric(
        specimen_auebc,
        domain_order=domain_order,
        method="proposed",
        column="cai_auebc",
    )
    oracle = _equal_domain_metric(
        specimen_auebc,
        domain_order=domain_order,
        method="mechanical_oracle_diagnostic",
        column="cai_auebc",
    )
    denominator = old - oracle
    gap_closure = None if denominator <= 0.0 else (old - proposed) / denominator
    old_effect = effects["c0_minus_proposed"]
    spatial_conditions = {
        "proposed_auebc_lower_than_c0": proposed < old,
        "c0_minus_proposed_lower_positive": old_effect.lower > 0.0,
        "at_least_four_domains_improve": old_effect.improved_domains >= 4,
        "oracle_gap_closure_at_least_20_percent": (
            gap_closure is not None and gap_closure >= 0.20
        ),
        "beats_shuffled_surface": effects["shuffled_minus_proposed"].lower > 0.0,
        "beats_wrong_and_deranged": (
            effects["wrong_minus_proposed"].lower > 0.0
            and effects["deranged_minus_proposed"].lower > 0.0
        ),
        "local_adds_beyond_global": effects["global_minus_proposed"].lower > 0.0,
    }
    global_effect = effects["c0_minus_global"]
    global_value = _equal_domain_metric(
        specimen_auebc,
        domain_order=domain_order,
        method="c2_global_context",
        column="cai_auebc",
    )
    global_conditions = {
        "global_auebc_lower_than_c0": global_value < old,
        "c0_minus_global_lower_positive": global_effect.lower > 0.0,
        "at_least_four_domains_improve": global_effect.improved_domains >= 4,
        "beats_shuffled_global": effects["shuffled_global_minus_global"].lower
        > 0.0,
    }
    proposed_ndcg = _equal_domain_metric(
        ranking_metrics,
        domain_order=domain_order,
        method="proposed",
        column="ndcg_10",
    )
    old_ndcg = _equal_domain_metric(
        ranking_metrics,
        domain_order=domain_order,
        method="c0_mvd_m1_o2",
        column="ndcg_10",
    )
    proposed_regret = _equal_domain_metric(
        ranking_metrics,
        domain_order=domain_order,
        method="proposed",
        column="next_action_regret",
    )
    old_regret = _equal_domain_metric(
        ranking_metrics,
        domain_order=domain_order,
        method="c0_mvd_m1_o2",
        column="next_action_regret",
    )
    ranking_improvement = proposed_ndcg > old_ndcg and proposed_regret < old_regret
    if all(spatial_conditions.values()):
        status = "P1_SPATIAL_VISUAL_OBSERVABILITY_GO"
        route = "SPATIAL"
        go = True
    elif all(global_conditions.values()):
        status = "P1_GLOBAL_VISUAL_CONTEXT_GO"
        route = "GLOBAL_CONTEXT"
        go = True
    elif ranking_improvement:
        status = "P1_DESCRIPTIVE_SPATIAL_SIGNAL_ONLY"
        route = None
        go = False
    else:
        status = "P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO"
        route = None
        go = False
    payload = {
        "authorized_route": route,
        "effects": {
            name: value.state_sha256 for name, value in sorted(effects.items())
        },
        "global_conditions": global_conditions,
        "go": go,
        "oracle_gap_closure": gap_closure,
        "ranking_improvement": ranking_improvement,
        "spatial_conditions": spatial_conditions,
        "status": status,
    }
    return P1Decision(
        status=status,
        go=go,
        authorized_route=route,
        oracle_gap_closure=gap_closure,
        spatial_conditions=MappingProxyType(spatial_conditions),
        global_conditions=MappingProxyType(global_conditions),
        ranking_improvement=ranking_improvement,
        effects=MappingProxyType(effects),
        state_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
    )


__all__ = [
    "ActionMetrics",
    "FrozenC0Scores",
    "FrozenOuterScores",
    "FusionSelection",
    "MLPVisualScorer",
    "OuterVisualModelFit",
    "P1Decision",
    "P1DeployableAuthority",
    "P1MechanicalLabels",
    "P1OuterExamples",
    "RidgeVisualScorer",
    "SourceCandidateSelection",
    "SpecimenBootstrapEffect",
    "VisualExamples",
    "assemble_p1_outer_examples",
    "attach_target_labels",
    "center_prior_scores",
    "decide_p1",
    "evaluate_action_scores",
    "fit_mlp_scorer",
    "fit_outer_visual_models",
    "fit_ridge_family",
    "freeze_outer_scores",
    "fuse_rank_scores",
    "load_frozen_c0_scores",
    "load_p1_deployable_authority",
    "load_p1_source_labels",
    "load_p1_target_labels",
    "paired_specimen_bootstrap",
    "replace_surface_features",
    "representation_matrix",
    "select_fusion_lambda",
    "select_source_candidate",
    "stable_rank_percentiles",
    "subset_visual_examples",
]
