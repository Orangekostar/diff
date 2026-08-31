"""P1 outer-fold inputs, controls, and pre-label target score freezing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import polars as pl

from .p1 import P1Config
from .surface_cells import (
    SurfaceCellAuthority,
    shuffled_surface_donors,
    spatial_derangement,
)
from .surface_encoder import SurfaceFeatureBank
from .visual_observability import (
    FrozenC0Scores,
    FrozenOuterScores,
    OuterVisualModelFit,
    P1DeployableAuthority,
    P1OuterExamples,
    VisualExamples,
    assemble_p1_outer_examples,
    attach_target_labels,
    center_prior_scores,
    evaluate_action_scores,
    freeze_outer_scores,
    fuse_rank_scores,
    load_p1_source_labels,
    load_p1_target_labels,
)


class P1PipelineError(ValueError):
    """Raised when P1 execution would violate a frozen authority or barrier."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _hash_payload(metadata: object, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(_canonical_json(metadata))
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise P1PipelineError(f"{label} mapping changed")
    return value


@dataclass(frozen=True, slots=True)
class P1OuterData:
    outer_domain: str
    initial_budget: float
    correct: P1OuterExamples
    shuffled: P1OuterExamples
    wrong_orientation: P1OuterExamples
    spatial_derangement: P1OuterExamples
    c0_source_scores: np.ndarray
    c0_target_scores: np.ndarray
    target_candidate_costs: np.ndarray
    target_native_shapes: tuple[tuple[int, int], ...]
    target_grid_state_sha256: tuple[str, ...]
    candidate_bank_state_sha256: str
    observed_feature_state_sha256: str
    surface_feature_state_sha256: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class P1OuterScoreEvaluation:
    outer_domain: str
    per_state_scores: pl.DataFrame
    per_specimen_metrics: pl.DataFrame
    score_matrices: Mapping[str, np.ndarray]
    state_sha256: str


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise P1PipelineError("P1 outer execution array changed")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _identity_reindex(
    *,
    source_specimen_ids: tuple[str, ...],
    source_dataset_ids: tuple[str, ...],
    target_specimen_ids: tuple[str, ...],
    target_dataset_ids: tuple[str, ...],
) -> np.ndarray:
    """Map a unique target identity order onto an authority source order."""

    if (
        not source_specimen_ids
        or len(source_specimen_ids) != len(source_dataset_ids)
        or not target_specimen_ids
        or len(target_specimen_ids) != len(target_dataset_ids)
    ):
        raise P1PipelineError("P1 identity roster changed")
    source_pairs = tuple(
        zip(source_dataset_ids, source_specimen_ids, strict=True)
    )
    target_pairs = tuple(
        zip(target_dataset_ids, target_specimen_ids, strict=True)
    )
    lookup = {identity: index for index, identity in enumerate(source_pairs)}
    if (
        len(lookup) != len(source_pairs)
        or len(set(target_pairs)) != len(target_pairs)
        or not set(target_pairs) <= set(lookup)
    ):
        raise P1PipelineError("P1 identity roster changed")
    output = np.asarray([lookup[identity] for identity in target_pairs], dtype=np.int64)
    output.setflags(write=False)
    return output


def load_p1_outer_data(
    config: P1Config,
    surface: SurfaceCellAuthority,
    features: SurfaceFeatureBank,
    deployable: P1DeployableAuthority,
    c0: FrozenC0Scores,
    *,
    outer_domain: str,
) -> P1OuterData:
    """Load one outer fold without reading its mechanical target labels."""

    from cmc_bbdm.mvd.authority import load_compact_mvd_authority
    from cmc_bbdm.mvd.config import load_mvd_config
    from cmc_bbdm.mvd.observability_dataset import (
        load_observed_candidate_feature_bank,
    )

    if (
        type(config) is not P1Config
        or type(surface) is not SurfaceCellAuthority
        or type(features) is not SurfaceFeatureBank
        or type(deployable) is not P1DeployableAuthority
        or type(c0) is not FrozenC0Scores
        or outer_domain not in config.domain_order
        or surface.specimen_ids != deployable.specimen_ids
        or surface.dataset_ids != deployable.dataset_ids
        or features.specimen_ids != surface.specimen_ids
        or features.dataset_ids != surface.dataset_ids
        or c0.specimen_ids != surface.specimen_ids
        or c0.dataset_ids != surface.dataset_ids
        or features.authority_state_sha256 != surface.state_sha256
        or features.transform_sha256 != config.surface_transform_sha256
    ):
        raise P1PipelineError("P1 outer execution authority changed")
    mvd = load_mvd_config(
        config.project_root / config.sources["mvd_config"].path,
        project_root=config.project_root,
    )
    compact = load_compact_mvd_authority(mvd, project_root=config.project_root)
    if compact.specimen_count != config.authorized_specimen_count:
        raise P1PipelineError("P1 MVD and P0R rosters differ")
    authority_order = _identity_reindex(
        source_specimen_ids=compact.specimen_ids,
        source_dataset_ids=compact.dataset_ids,
        target_specimen_ids=surface.specimen_ids,
        target_dataset_ids=surface.dataset_ids,
    )
    if authority_order.size != surface.specimen_count:
        raise P1PipelineError("P1 MVD and P0R rosters differ")
    old_state = _mapping(config.raw.get("old_state"), "P1 old state")
    budgets = _mapping(old_state.get("initial_budgets"), "P1 initial budgets")
    try:
        initial_budget = float(budgets[outer_domain])
    except (KeyError, TypeError, ValueError) as error:
        raise P1PipelineError("P1 outer initial budget changed") from error
    token = str(initial_budget).replace(".", "p")
    observed = load_observed_candidate_feature_bank(
        config.project_root / config.sources[f"observed_features_{token}"].path,
        compact=compact,
        initial_budget=initial_budget,
    )
    bank = compact.candidate_banks[initial_budget]
    source_labels = load_p1_source_labels(
        config, deployable, outer_domain=outer_domain
    )
    donor_ids = shuffled_surface_donors(
        surface.specimen_ids,
        surface.dataset_ids,
        seed=str(
            _mapping(
                _mapping(config.raw.get("controls"), "P1 controls").get("C3"),
                "P1 shuffled control",
            )["seed"]
        ),
    )
    specimen_indices = {
        specimen_id: index for index, specimen_id in enumerate(surface.specimen_ids)
    }
    donor_indices = np.asarray(
        [specimen_indices[specimen_id] for specimen_id in donor_ids], dtype=np.int64
    )
    controls = _mapping(config.raw.get("controls"), "P1 controls")
    derangement_seed = str(
        _mapping(controls.get("C5"), "P1 spatial derangement")["seed"]
    )
    local_deranged = np.empty_like(features.local_correct_embeddings)
    derangements: list[tuple[int, ...]] = []
    for index, (specimen_id, dataset_id) in enumerate(
        zip(surface.specimen_ids, surface.dataset_ids, strict=True)
    ):
        permutation = spatial_derangement(
            specimen_id, dataset_id=dataset_id, seed=derangement_seed
        )
        derangements.append(permutation)
        local_deranged[index] = features.local_correct_embeddings[
            index, np.asarray(permutation, dtype=np.int64)
        ]

    shared = {
        "outer_domain": outer_domain,
        "specimen_ids": surface.specimen_ids,
        "dataset_ids": surface.dataset_ids,
        "initial_embeddings": bank.initial_embeddings[authority_order],
        "current_predictions": deployable.current_predictions,
        "candidate_features": observed.candidate_features[authority_order],
        "source_labels": source_labels,
    }
    correct = assemble_p1_outer_examples(
        **shared,
        global_embeddings=features.global_embeddings,
        local_embeddings=features.local_correct_embeddings,
        feature_control="correct_registration",
    )
    shuffled = assemble_p1_outer_examples(
        **shared,
        global_embeddings=features.global_embeddings[donor_indices],
        local_embeddings=features.local_correct_embeddings[donor_indices],
        feature_control="shuffled_surface_specimen",
    )
    wrong = assemble_p1_outer_examples(
        **shared,
        global_embeddings=features.global_embeddings,
        local_embeddings=features.local_wrong_orientation_embeddings,
        feature_control="wrong_orientation",
    )
    deranged = assemble_p1_outer_examples(
        **shared,
        global_embeddings=features.global_embeddings,
        local_embeddings=local_deranged,
        feature_control="spatial_derangement",
    )
    source_indices = np.asarray(correct.source_indices, dtype=np.int64)
    target_indices = np.asarray(correct.target_indices, dtype=np.int64)
    source_count = source_indices.size
    target_count = target_indices.size
    c0_source = _readonly(
        c0.scores[source_indices], dtype="<f8", shape=(source_count, 64)
    )
    c0_target = _readonly(
        c0.scores[target_indices], dtype="<f8", shape=(target_count, 64)
    )
    target_costs = _readonly(
        observed.candidate_costs[authority_order[target_indices]],
        dtype="<i8",
        shape=(target_count, 64),
    )
    native_shapes = tuple(
        tuple(int(value) for value in bank.native_shapes[authority_order[index]])
        for index in target_indices
    )
    grid_states = tuple(
        bank.grid_state_sha256[authority_order[index]] for index in target_indices
    )
    metadata = {
        "candidate_bank_state_sha256": bank.state_sha256,
        "control_example_states": {
            "correct": correct.state_sha256,
            "shuffled": shuffled.state_sha256,
            "spatial_derangement": deranged.state_sha256,
            "wrong_orientation": wrong.state_sha256,
        },
        "derangements": derangements,
        "donor_ids": donor_ids,
        "grid_state_sha256": grid_states,
        "initial_budget": initial_budget,
        "observed_feature_state_sha256": observed.state_sha256,
        "outer_domain": outer_domain,
        "surface_feature_state_sha256": features.state_sha256,
        "target_native_shapes": native_shapes,
    }
    return P1OuterData(
        outer_domain=outer_domain,
        initial_budget=initial_budget,
        correct=correct,
        shuffled=shuffled,
        wrong_orientation=wrong,
        spatial_derangement=deranged,
        c0_source_scores=c0_source,
        c0_target_scores=c0_target,
        target_candidate_costs=target_costs,
        target_native_shapes=native_shapes,
        target_grid_state_sha256=grid_states,
        candidate_bank_state_sha256=bank.state_sha256,
        observed_feature_state_sha256=observed.state_sha256,
        surface_feature_state_sha256=features.state_sha256,
        state_sha256=_hash_payload(metadata, c0_source, c0_target, target_costs),
    )


def _composite_model_state(
    *,
    method: str,
    model_state_sha256: str,
    feature_control: str,
    inference_state_sha256: str,
    fusion_lambda: float | None,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "feature_control": feature_control,
                "fusion_lambda": fusion_lambda,
                "inference_state_sha256": inference_state_sha256,
                "method": method,
                "model_state_sha256": model_state_sha256,
            }
        )
    ).hexdigest()


def freeze_p1_outer_predictions(
    data: P1OuterData, fitted: OuterVisualModelFit
) -> FrozenOuterScores:
    """Issue immutable target scores without loading target outcomes."""

    expected_models = {
        "old_refit_diagnostic",
        "proposed",
        "c2_global_context",
        "c3_shuffled_surface",
        "c4_wrong_orientation",
        "c5_spatial_derangement",
        "c3_shuffled_global",
    }
    if (
        type(data) is not P1OuterData
        or type(fitted) is not OuterVisualModelFit
        or fitted.outer_domain != data.outer_domain
        or set(fitted.models) != expected_models
        or set(fitted.model_feature_controls) != expected_models
        or fitted.correct_lambda not in {0.0, 0.25, 0.5, 0.75, 1.0}
        or fitted.global_lambda not in {0.0, 0.25, 0.5, 0.75, 1.0}
    ):
        raise P1PipelineError("P1 frozen target model identity changed")
    examples = {
        "old_refit_diagnostic": data.correct.inference,
        "proposed": data.correct.inference,
        "c2_global_context": data.correct.inference,
        "c3_shuffled_surface": data.shuffled.inference,
        "c4_wrong_orientation": data.wrong_orientation.inference,
        "c5_spatial_derangement": data.spatial_derangement.inference,
        "c3_shuffled_global": data.shuffled.inference,
    }
    scores: dict[str, np.ndarray] = {
        "c0_mvd_m1_o2": data.c0_target_scores,
        "c1_center_prior": np.tile(
            center_prior_scores(), (data.correct.inference.specimen_count, 1)
        ),
    }
    states = {
        "c0_mvd_m1_o2": _hash_payload(
            {
                "data_state_sha256": data.state_sha256,
                "method": "c0_mvd_m1_o2",
            },
            data.c0_target_scores,
        ),
        "c1_center_prior": _hash_payload(
            {"method": "c1_center_prior"}, center_prior_scores()
        ),
    }
    for method, model in fitted.models.items():
        inference = examples[method]
        expected_control = fitted.model_feature_controls[method]
        if (
            inference.mechanical_values is not None
            or inference.feature_control != expected_control
        ):
            raise P1PipelineError("P1 target inference control changed")
        visual = np.asarray(model.predict(inference), dtype=np.float64)
        if method in {"proposed", "c3_shuffled_surface", "c4_wrong_orientation", "c5_spatial_derangement"}:
            fusion_lambda = fitted.correct_lambda
            issued = fuse_rank_scores(data.c0_target_scores, visual, fusion_lambda)
        elif method in {"c2_global_context", "c3_shuffled_global"}:
            fusion_lambda = fitted.global_lambda
            issued = fuse_rank_scores(data.c0_target_scores, visual, fusion_lambda)
        else:
            fusion_lambda = None
            issued = visual
        model_state = str(getattr(model, "state_sha256", ""))
        if len(model_state) != 64:
            raise P1PipelineError("P1 target model state changed")
        scores[method] = issued
        states[method] = _composite_model_state(
            method=method,
            model_state_sha256=model_state,
            feature_control=expected_control,
            inference_state_sha256=inference.state_sha256,
            fusion_lambda=fusion_lambda,
        )
    return freeze_outer_scores(
        data.correct.inference,
        scores=scores,
        model_state_sha256=states,
        selection_state_sha256=fitted.selection_state_sha256,
    )


def evaluate_p1_outer_score_metrics(
    evaluation: VisualExamples,
    frozen_scores: FrozenOuterScores,
    *,
    candidate_costs: object,
) -> P1OuterScoreEvaluation:
    """Evaluate already-frozen scores and add the oracle only as a diagnostic."""

    if (
        type(evaluation) is not VisualExamples
        or evaluation.role != "outer_evaluation"
        or evaluation.mechanical_values is None
        or type(frozen_scores) is not FrozenOuterScores
        or frozen_scores.outer_domain != evaluation.outer_domain
        or frozen_scores.specimen_ids != evaluation.specimen_ids
        or frozen_scores.dataset_ids != evaluation.dataset_ids
        or frozen_scores.inference_state_sha256 == evaluation.state_sha256
        or not frozen_scores.methods
    ):
        raise P1PipelineError("P1 frozen score evaluation identity changed")
    count = evaluation.specimen_count
    costs = _readonly(candidate_costs, dtype="<i8", shape=(count, 64))
    matrices = dict(frozen_scores.scores)
    matrices["mechanical_oracle_diagnostic"] = evaluation.mechanical_values
    model_states = dict(frozen_scores.model_state_sha256)
    model_states["mechanical_oracle_diagnostic"] = hashlib.sha256(
        _canonical_json(
            {
                "evaluation_state_sha256": evaluation.state_sha256,
                "method": "mechanical_oracle_diagnostic",
                "role": "evaluation_only_diagnostic",
            }
        )
    ).hexdigest()
    state_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for specimen_index, (specimen_id, dataset_id) in enumerate(
        zip(evaluation.specimen_ids, evaluation.dataset_ids, strict=True)
    ):
        truth = evaluation.mechanical_values[specimen_index]
        for method in sorted(matrices):
            scores = np.asarray(matrices[method][specimen_index], dtype=np.float64)
            metrics = evaluate_action_scores(truth, scores)
            metric_rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "model_state_sha256": model_states[method],
                    "ndcg_10": metrics.ndcg_10,
                    "next_action_regret": metrics.next_action_regret,
                    "one_step_cai_utility": metrics.one_step_cai_utility,
                    "outer_domain": evaluation.outer_domain,
                    "recall_5": metrics.recall_5,
                    "spearman": metrics.spearman,
                    "specimen_id": specimen_id,
                    "top_10_percent_overlap": metrics.top_10_percent_overlap,
                    "top_1_oracle_match": metrics.top_1_oracle_match,
                }
            )
            role = (
                "evaluation_only_diagnostic"
                if method == "mechanical_oracle_diagnostic"
                else "frozen_before_target_labels"
            )
            for cell_index in range(64):
                state_rows.append(
                    {
                        "candidate_cost": int(costs[specimen_index, cell_index]),
                        "cell_index": cell_index,
                        "dataset_id": dataset_id,
                        "mechanical_value": float(truth[cell_index]),
                        "method": method,
                        "model_state_sha256": model_states[method],
                        "outer_domain": evaluation.outer_domain,
                        "predicted_score": float(scores[cell_index]),
                        "score_freeze_state_sha256": frozen_scores.state_sha256,
                        "score_role": role,
                        "selection_state_sha256": frozen_scores.selection_state_sha256,
                        "specimen_id": specimen_id,
                    }
                )
    per_state = pl.DataFrame(state_rows, infer_schema_length=None).sort(
        ["outer_domain", "specimen_id", "method", "cell_index"]
    )
    per_specimen = pl.DataFrame(metric_rows, infer_schema_length=None).sort(
        ["outer_domain", "specimen_id", "method"]
    )
    digest = hashlib.sha256(
        _canonical_json(
            {
                "frozen_score_state_sha256": frozen_scores.state_sha256,
                "methods": tuple(sorted(matrices)),
                "outer_domain": evaluation.outer_domain,
                "target_evaluation_state_sha256": evaluation.state_sha256,
            }
        )
    )
    for table in (per_state, per_specimen):
        for row in table.iter_rows(named=True):
            digest.update(_canonical_json(row))
    return P1OuterScoreEvaluation(
        outer_domain=evaluation.outer_domain,
        per_state_scores=per_state,
        per_specimen_metrics=per_specimen,
        score_matrices=MappingProxyType(matrices),
        state_sha256=digest.hexdigest(),
    )


def evaluate_p1_outer_scores(
    config: P1Config,
    deployable: P1DeployableAuthority,
    data: P1OuterData,
    frozen_scores: FrozenOuterScores,
) -> P1OuterScoreEvaluation:
    """Cross the target-label barrier only after matching scores are frozen."""

    if (
        type(data) is not P1OuterData
        or data.outer_domain != frozen_scores.outer_domain
    ):
        raise P1PipelineError("P1 target score/data identity changed")
    labels = load_p1_target_labels(
        config,
        deployable,
        outer_domain=data.outer_domain,
        frozen_scores=frozen_scores,
    )
    evaluation = attach_target_labels(
        data.correct.inference, labels, frozen_scores=frozen_scores
    )
    return evaluate_p1_outer_score_metrics(
        evaluation,
        frozen_scores,
        candidate_costs=data.target_candidate_costs,
    )


__all__ = [
    "P1OuterData",
    "P1OuterScoreEvaluation",
    "P1PipelineError",
    "evaluate_p1_outer_score_metrics",
    "evaluate_p1_outer_scores",
    "freeze_p1_outer_predictions",
    "load_p1_outer_data",
]
