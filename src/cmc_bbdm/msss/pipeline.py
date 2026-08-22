"""End-to-end orchestration for the frozen S1 MSSS experiment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .authority import MSSSAuthority
from .protocol import MSSSProtocol
from .s1 import S1Run, summarize_s1
from .scale_evaluator import AxisEvaluation, evaluate_axis
from .scale_features import (
    ScaleFeatureBank,
    build_scale_feature_bank,
    load_frozen_encoder,
    rematerialize_conditions,
)
from .spatial_specificity import (
    build_shuffled_feature_bank,
    evaluate_spatial_specificity,
)


class MSSSPipelineError(ValueError):
    """Raised when an S1 execution request violates the frozen protocol."""


StatusHook = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class S1Execution:
    mode: str
    test_only: bool
    indices: np.ndarray
    bank: ScaleFeatureBank
    run: S1Run


def execution_indices(
    dataset_ids: Sequence[str], *, group_order: Sequence[str], mode: str
) -> np.ndarray:
    """Return the complete roster or the deterministic nine-per-domain smoke roster."""

    datasets = tuple(dataset_ids)
    groups = tuple(group_order)
    if (
        not datasets
        or not groups
        or len(set(groups)) != len(groups)
        or set(datasets) != set(groups)
        or mode not in {"smoke", "full"}
    ):
        raise MSSSPipelineError("execution roster or mode is invalid")
    if mode == "full":
        if tuple(dict.fromkeys(datasets)) != groups:
            raise MSSSPipelineError("full roster domain order changed")
        output = np.arange(len(datasets), dtype=np.int64)
    else:
        values: list[int] = []
        for group in groups:
            positions = [index for index, value in enumerate(datasets) if value == group]
            if len(positions) < 9:
                raise MSSSPipelineError("smoke execution requires nine specimens per domain")
            values.extend(positions[:9])
        output = np.asarray(values, dtype=np.int64)
    output.setflags(write=False)
    return output


def _status(hook: StatusHook | None, message: str) -> None:
    if hook is not None:
        hook(message)


def run_s1_experiment(
    protocol: MSSSProtocol,
    authority: MSSSAuthority,
    *,
    project_root: str | Path,
    mode: str,
    status_hook: StatusHook | None = None,
) -> S1Execution:
    """Execute all fixed candidates, source selection, specificity, and S1 gate."""

    if type(protocol) is not MSSSProtocol or type(authority) is not MSSSAuthority:
        raise MSSSPipelineError("issued MSSS protocol and authority are required")
    root = Path(project_root).resolve(strict=True)
    indices = execution_indices(
        authority.dataset_ids, group_order=protocol.domain_order, mode=mode
    )
    targets = np.asarray(authority.targets[indices], dtype=np.float64)
    metadata = np.asarray(authority.metadata13[indices], dtype=np.float64)
    images = tuple(authority.registered_inputs.images[int(index)] for index in indices)

    _status(status_hook, "loading frozen ResNet18 encoder")
    encoder = load_frozen_encoder(protocol, project_root=root)
    _status(status_hook, "materializing and encoding 37 registered scale conditions")
    feature_build = build_scale_feature_bank(
        protocol,
        authority,
        project_root=root,
        encoder=encoder,
        indices=indices,
        retain_materializations=False,
    )
    bank = feature_build.bank
    evaluations: list[AxisEvaluation] = []
    for axis in ("sampling", "gaussian", "wavelet"):
        _status(status_hook, f"evaluating source-only {axis} scale curve")
        evaluations.append(
            evaluate_axis(
                bank,
                targets=targets,
                metadata13=metadata,
                axis=axis,
                pca_dimensions=protocol.pca_dimensions,
                primary_margin=protocol.primary_margin,
                margins=protocol.noninferiority_margins,
            )
        )
    selected_conditions = tuple(
        dict.fromkeys(
            selection.selected_condition_id
            for evaluation in evaluations
            for selection in evaluation.scale_selections
        )
    )
    _status(status_hook, "building registered post-scale P3 shuffle controls")
    selected_build = rematerialize_conditions(
        bank, images=images, condition_ids=selected_conditions
    )
    shuffled = build_shuffled_feature_bank(
        selected_build,
        selected_condition_ids=selected_conditions,
        specimen_ids=bank.specimen_ids,
        dataset_ids=bank.dataset_ids,
        seeds=protocol.specificity_seeds,
        encoder=encoder,
    )
    specificity = []
    for evaluation in evaluations:
        _status(status_hook, f"evaluating {evaluation.axis} spatial specificity")
        specificity.append(
            evaluate_spatial_specificity(
                evaluation,
                regular_bank=bank,
                shuffled_bank=shuffled,
                targets=targets,
                metadata13=metadata,
                pca_dimensions=protocol.pca_dimensions,
            )
        )
    resamples = protocol.bootstrap_resamples if mode == "full" else 2_000
    _status(status_hook, f"aggregating S1 gate with {resamples} bootstrap resamples")
    run = summarize_s1(
        protocol,
        bank=bank,
        evaluations=tuple(evaluations),
        specificity=tuple(specificity),
        bootstrap_resamples=resamples,
    )
    snapshot = np.frombuffer(indices.astype("<i8", copy=False).tobytes(), dtype="<i8")
    snapshot.setflags(write=False)
    return S1Execution(
        mode=mode,
        test_only=mode == "smoke",
        indices=snapshot,
        bank=bank,
        run=run,
    )


__all__ = [
    "MSSSPipelineError",
    "S1Execution",
    "execution_indices",
    "run_s1_experiment",
]
