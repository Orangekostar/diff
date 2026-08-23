"""Target-side execution for source-frozen MVA A4 acquisition masks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np
import polars as pl

from .a4_candidate_bank import (
    build_candidate_bank,
    candidate_bank_path,
    load_candidate_bank,
    save_candidate_bank,
)
from .a4_config import load_a4_config
from .a4_source_labels import (
    METHODS,
    SourceLabelResult,
    generate_source_labels,
    validate_source_label_result,
)
from .acquisition_grid import INITIAL_BUDGETS, build_acquisition_grid
from .authority import MVAAuthority, load_mva_authority
from .cai_evaluator import CAIPredictor
from .config import load_mva_config
from .crossfit import FitAudit, fit_outer_source_predictor
from .encoder_session import MVAEncoderSession
from .global_mask import GlobalMaskRanking
from .interpolation import RefinementPatchCache
from .measurement_state import initial_state
from .oracle_execution import _global_ssim, _materialize_control
from .oracle_trajectory import run_static_mask_trajectory
from .pipeline import _encoder
from .reconstruction_value import normalized_rgb_mse


class A4ExecutionError(ValueError):
    """Raised when an A4 outer evaluation violates its frozen contract."""


class _Predictor(Protocol):
    fit_domains: tuple[str, ...]
    state_sha256: str

    def predict(self, metadata: object, embeddings: object) -> np.ndarray: ...


class _Encoder(Protocol):
    def encode(self, images: list[np.ndarray]) -> np.ndarray: ...

    def validate(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticOuterEvaluation:
    outer_domain: str
    domain_order: tuple[str, ...]
    source_domains: tuple[str, ...]
    target_specimen_ids: tuple[str, ...]
    source_specimen_ids: tuple[str, ...]
    initial_budget: float
    checkpoints: tuple[float, ...]
    source_label_state_sha256: str
    p_a_predictor_state_sha256: str
    p_b_predictor_states: tuple[tuple[float, str], ...]
    cell_orders: tuple[tuple[str, tuple[int, ...]], ...]
    states: tuple[dict[str, object], ...]
    trajectories: tuple[dict[str, object], ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluatorFitAudit:
    evaluator: str
    checkpoint: float | None
    stage: str
    held_out_target_domain: str
    query_domains: tuple[str, ...]
    fit_domains: tuple[str, ...]
    query_specimen_ids: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    pca_dimension: int
    predictor_state_sha256: str


@dataclass(frozen=True, slots=True)
class OuterEvaluationModels:
    outer_domain: str
    domain_order: tuple[str, ...]
    source_domains: tuple[str, ...]
    checkpoints: tuple[float, ...]
    p_a_model: CAIPredictor
    p_b_models: Mapping[float, CAIPredictor]
    fit_audits: tuple[EvaluatorFitAudit, ...]
    state_sha256: str


_STATE_KEYS = {
    "cumulative_actions",
    "dataset_id",
    "effective_budget",
    "initial_budget",
    "measured_count",
    "method",
    "native_count",
    "nominal_checkpoint",
    "normalized_rgb_mse",
    "outer_domain",
    "p_a_absolute_error",
    "p_a_prediction",
    "p_a_predictor_state_sha256",
    "p_b_absolute_error",
    "p_b_prediction",
    "p_b_predictor_state_sha256",
    "ranking_source_domains",
    "source_label_state_sha256",
    "specimen_id",
    "ssim",
    "target",
}

_TRAJECTORY_KEYS = {
    "cell_index",
    "dataset_id",
    "from_level",
    "method",
    "nominal_checkpoint",
    "outer_domain",
    "ranking_position",
    "ranking_score",
    "source_label_state_sha256",
    "specimen_id",
    "to_level",
}


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _result_state(result: StaticOuterEvaluation) -> str:
    digest = hashlib.sha256()
    header = {
        "cell_orders": result.cell_orders,
        "checkpoints": result.checkpoints,
        "domain_order": result.domain_order,
        "initial_budget": result.initial_budget,
        "outer_domain": result.outer_domain,
        "p_a_predictor_state_sha256": result.p_a_predictor_state_sha256,
        "p_b_predictor_states": result.p_b_predictor_states,
        "source_domains": result.source_domains,
        "source_label_state_sha256": result.source_label_state_sha256,
        "source_specimen_ids": result.source_specimen_ids,
        "target_specimen_ids": result.target_specimen_ids,
    }
    digest.update(
        json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    )
    for row in (*result.states, *result.trajectories):
        digest.update(
            json.dumps(
                row, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("ascii")
        )
    return digest.hexdigest()


def _model_state(bundle: OuterEvaluationModels) -> str:
    payload = {
        "audits": [asdict(value) for value in bundle.fit_audits],
        "checkpoints": bundle.checkpoints,
        "domain_order": bundle.domain_order,
        "outer_domain": bundle.outer_domain,
        "p_a": bundle.p_a_model.state_sha256,
        "p_b": tuple(
            (checkpoint, bundle.p_b_models[checkpoint].state_sha256)
            for checkpoint in bundle.checkpoints
        ),
        "source_domains": bundle.source_domains,
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()


def _evaluator_audit(
    value: FitAudit, *, evaluator: str, checkpoint: float | None
) -> EvaluatorFitAudit:
    return EvaluatorFitAudit(
        evaluator=evaluator,
        checkpoint=checkpoint,
        stage=value.stage,
        held_out_target_domain=value.outer_domain,
        query_domains=value.query_domains,
        fit_domains=value.fit_domains,
        query_specimen_ids=value.query_specimen_ids,
        fit_specimen_ids=value.fit_specimen_ids,
        pca_dimension=value.pca_dimension,
        predictor_state_sha256=value.predictor_state_sha256,
    )


def validate_outer_evaluation_models(bundle: OuterEvaluationModels) -> None:
    """Validate all A4 evaluator fits and target-domain barriers."""

    if type(bundle) is not OuterEvaluationModels:
        raise A4ExecutionError("issued outer evaluator bundle is required")
    expected_sources = tuple(
        domain for domain in bundle.domain_order if domain != bundle.outer_domain
    )
    if (
        len(bundle.domain_order) != 6
        or bundle.outer_domain not in bundle.domain_order
        or bundle.source_domains != expected_sources
        or set(bundle.p_a_model.fit_domains) != set(expected_sources)
        or set(bundle.p_b_models) != set(bundle.checkpoints)
        or any(
            set(model.fit_domains) != set(expected_sources)
            for model in bundle.p_b_models.values()
        )
    ):
        raise A4ExecutionError("outer evaluator model roster changed")
    expected_evaluators = {
        "P-A": (None, bundle.p_a_model.state_sha256),
        **{
            f"P-B:{checkpoint}": (
                checkpoint,
                bundle.p_b_models[checkpoint].state_sha256,
            )
            for checkpoint in bundle.checkpoints
        },
    }
    observed_final: dict[str, str] = {}
    for audit in bundle.fit_audits:
        expected = expected_evaluators.get(audit.evaluator)
        if (
            expected is None
            or audit.checkpoint != expected[0]
            or audit.held_out_target_domain != bundle.outer_domain
            or bundle.outer_domain in audit.fit_domains
            or set(audit.query_domains) & set(audit.fit_domains)
            or audit.stage not in {"inner", "outer"}
        ):
            raise A4ExecutionError("outer evaluator fit barrier changed")
        if audit.stage == "outer":
            if (
                audit.query_domains != (bundle.outer_domain,)
                or set(audit.fit_domains) != set(expected_sources)
                or audit.predictor_state_sha256 != expected[1]
            ):
                raise A4ExecutionError("outer evaluator final fit changed")
            observed_final[audit.evaluator] = audit.predictor_state_sha256
    if set(observed_final) != set(expected_evaluators):
        raise A4ExecutionError("outer evaluator final audit is incomplete")
    if not _is_sha256(bundle.state_sha256) or _model_state(bundle) != bundle.state_sha256:
        raise A4ExecutionError("outer evaluator content digest changed")


def fit_outer_evaluation_models(
    *,
    outer_domain: str,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    targets: object,
    metadata: object,
    full_embeddings: object,
    uniform_embeddings: Mapping[float, np.ndarray],
    pca_dimensions: tuple[int, ...],
    ridge_alpha: float,
    tie_tolerance: float = 1.0e-12,
) -> OuterEvaluationModels:
    """Fit one source-FULL P-A and checkpoint-specific source-uniform P-B heads."""

    if (
        type(domain_order) is not tuple
        or len(domain_order) != 6
        or outer_domain not in domain_order
        or type(checkpoints) is not tuple
        or not checkpoints
        or tuple(sorted(set(checkpoints))) != checkpoints
        or set(uniform_embeddings) != set(checkpoints)
    ):
        raise A4ExecutionError("outer evaluator request changed")
    p_a_fit = fit_outer_source_predictor(
        method="MVA_P_A",
        outer_domain=outer_domain,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domain_order,
        targets=targets,
        metadata=metadata,
        embeddings=full_embeddings,
        pca_dimensions=pca_dimensions,
        ridge_alpha=ridge_alpha,
        tie_tolerance=tie_tolerance,
    )
    p_b_fits = {
        checkpoint: fit_outer_source_predictor(
            method=f"MVA_P_B_{checkpoint}",
            outer_domain=outer_domain,
            specimen_ids=specimen_ids,
            dataset_ids=dataset_ids,
            domain_order=domain_order,
            targets=targets,
            metadata=metadata,
            embeddings=uniform_embeddings[checkpoint],
            pca_dimensions=pca_dimensions,
            ridge_alpha=ridge_alpha,
            tie_tolerance=tie_tolerance,
        )
        for checkpoint in checkpoints
    }
    audits = tuple(
        [
            _evaluator_audit(value, evaluator="P-A", checkpoint=None)
            for value in p_a_fit.fit_audits
        ]
        + [
            _evaluator_audit(
                value,
                evaluator=f"P-B:{checkpoint}",
                checkpoint=checkpoint,
            )
            for checkpoint in checkpoints
            for value in p_b_fits[checkpoint].fit_audits
        ]
    )
    bundle = OuterEvaluationModels(
        outer_domain=outer_domain,
        domain_order=domain_order,
        source_domains=tuple(
            domain for domain in domain_order if domain != outer_domain
        ),
        checkpoints=checkpoints,
        p_a_model=p_a_fit.model,
        p_b_models=MappingProxyType(
            {checkpoint: p_b_fits[checkpoint].model for checkpoint in checkpoints}
        ),
        fit_audits=audits,
        state_sha256="",
    )
    output = replace(bundle, state_sha256=_model_state(bundle))
    validate_outer_evaluation_models(output)
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranking_rows(source_labels: SourceLabelResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ranking in source_labels.rankings:
        positions = {
            cell_index: position
            for position, cell_index in enumerate(ranking.cell_order)
        }
        for cell_index in range(64):
            rows.append(
                {
                    "outer_domain": ranking.outer_domain,
                    "method": ranking.method,
                    "cell_index": cell_index,
                    "ranking_position": positions[cell_index],
                    "cell_score": ranking.cell_scores[cell_index],
                    "mean_raw_value": ranking.mean_raw_values[cell_index],
                    "mean_value_per_measurement": ranking.mean_value_per_measurement[
                        cell_index
                    ],
                    "source_domains": "|".join(ranking.source_domains),
                    "source_specimen_count": ranking.source_specimen_count,
                    "source_label_state_sha256": source_labels.state_sha256,
                }
            )
    return rows


def _fit_audit_rows(
    source_labels: SourceLabelResult, evaluator_models: OuterEvaluationModels
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for audit in (*source_labels.fit_audits, *source_labels.selection_audits):
        rows.append(
            {
                "audit_family": "source_value",
                "evaluator": "P-A-label",
                "checkpoint": None,
                "stage": audit.stage,
                "held_out_target_domain": audit.held_out_target_domain,
                "query_source_domain": audit.query_source_domain,
                "query_domains": "|".join(audit.query_domains),
                "fit_domains": "|".join(audit.fit_domains),
                "query_specimen_ids": "|".join(audit.query_specimen_ids),
                "fit_specimen_ids": "|".join(audit.fit_specimen_ids),
                "pca_dimension": audit.pca_dimension,
                "predictor_state_sha256": audit.predictor_state_sha256,
            }
        )
    for audit in evaluator_models.fit_audits:
        rows.append(
            {
                "audit_family": "target_evaluator",
                "evaluator": audit.evaluator,
                "checkpoint": audit.checkpoint,
                "stage": audit.stage,
                "held_out_target_domain": audit.held_out_target_domain,
                "query_source_domain": None,
                "query_domains": "|".join(audit.query_domains),
                "fit_domains": "|".join(audit.fit_domains),
                "query_specimen_ids": "|".join(audit.query_specimen_ids),
                "fit_specimen_ids": "|".join(audit.fit_specimen_ids),
                "pca_dimension": audit.pca_dimension,
                "predictor_state_sha256": audit.predictor_state_sha256,
            }
        )
    return rows


def _validate_published_outer_shard(
    path: Path,
    *,
    source_label_state_sha256: str,
    evaluator_model_state_sha256: str,
    evaluation_state_sha256: str,
) -> None:
    expected_files = {
        "complete.json",
        "fit_audits.csv",
        "ranking_stability.csv",
        "rankings.csv",
        "source_values.parquet",
        "states.parquet",
        "trajectories.parquet",
    }
    if not path.is_dir() or {item.name for item in path.iterdir()} != expected_files:
        raise A4ExecutionError("outer shard file roster changed")
    try:
        complete = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise A4ExecutionError("outer shard completion record is invalid") from error
    expected_keys = {
        "candidate_bank_state_sha256",
        "evaluator_model_state_sha256",
        "evaluation_state_sha256",
        "file_sha256",
        "fit_audit_rows",
        "outer_domain",
        "ranking_rows",
        "source_label_state_sha256",
        "source_value_rows",
        "state_rows",
        "target_specimen_count",
        "trajectory_rows",
    }
    if (
        set(complete) != expected_keys
        or complete["source_label_state_sha256"] != source_label_state_sha256
        or complete["evaluator_model_state_sha256"]
        != evaluator_model_state_sha256
        or complete["evaluation_state_sha256"] != evaluation_state_sha256
    ):
        raise A4ExecutionError("outer shard completion state changed")
    data_files = expected_files - {"complete.json"}
    hashes = complete["file_sha256"]
    if (
        not isinstance(hashes, dict)
        or set(hashes) != data_files
        or any(_sha256_file(path / name) != hashes[name] for name in data_files)
    ):
        raise A4ExecutionError("outer shard file digest changed")
    try:
        observed_counts = {
            "source_value_rows": pl.read_parquet(
                path / "source_values.parquet"
            ).height,
            "state_rows": pl.read_parquet(path / "states.parquet").height,
            "trajectory_rows": pl.read_parquet(path / "trajectories.parquet").height,
            "ranking_rows": pl.read_csv(path / "rankings.csv").height,
            "fit_audit_rows": pl.read_csv(path / "fit_audits.csv").height,
        }
    except (OSError, pl.exceptions.PolarsError) as error:
        raise A4ExecutionError("outer shard table cannot be read") from error
    if any(complete[name] != count for name, count in observed_counts.items()):
        raise A4ExecutionError("outer shard row count changed")


def publish_outer_shard(
    destination: str | Path,
    *,
    source_labels: SourceLabelResult,
    evaluator_models: OuterEvaluationModels,
    evaluation: StaticOuterEvaluation,
) -> Path:
    """Transactionally publish one validated outer-domain work shard."""

    validate_source_label_result(source_labels)
    validate_outer_evaluation_models(evaluator_models)
    validate_outer_evaluation(evaluation)
    if (
        source_labels.outer_domain != evaluator_models.outer_domain
        or source_labels.outer_domain != evaluation.outer_domain
        or source_labels.domain_order != evaluator_models.domain_order
        or source_labels.domain_order != evaluation.domain_order
        or source_labels.source_domains != evaluation.source_domains
        or source_labels.source_specimen_ids != evaluation.source_specimen_ids
        or source_labels.state_sha256 != evaluation.source_label_state_sha256
        or evaluator_models.p_a_model.state_sha256
        != evaluation.p_a_predictor_state_sha256
        or tuple(
            (checkpoint, evaluator_models.p_b_models[checkpoint].state_sha256)
            for checkpoint in evaluator_models.checkpoints
        )
        != evaluation.p_b_predictor_states
    ):
        raise A4ExecutionError("outer shard components are inconsistent")

    output = Path(destination)
    if output.exists():
        _validate_published_outer_shard(
            output,
            source_label_state_sha256=source_labels.state_sha256,
            evaluator_model_state_sha256=evaluator_models.state_sha256,
            evaluation_state_sha256=evaluation.state_sha256,
        )
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        pl.DataFrame(source_labels.rows, infer_schema_length=None).write_parquet(
            temporary / "source_values.parquet", compression="zstd"
        )
        pl.DataFrame(evaluation.states, infer_schema_length=None).write_parquet(
            temporary / "states.parquet", compression="zstd"
        )
        pl.DataFrame(evaluation.trajectories, infer_schema_length=None).write_parquet(
            temporary / "trajectories.parquet", compression="zstd"
        )
        ranking_rows = _ranking_rows(source_labels)
        fit_audit_rows = _fit_audit_rows(source_labels, evaluator_models)
        stability_rows = [asdict(value) for value in source_labels.stability]
        pl.DataFrame(ranking_rows, infer_schema_length=None).write_csv(
            temporary / "rankings.csv"
        )
        pl.DataFrame(fit_audit_rows, infer_schema_length=None).write_csv(
            temporary / "fit_audits.csv"
        )
        pl.DataFrame(stability_rows, infer_schema_length=None).write_csv(
            temporary / "ranking_stability.csv"
        )
        data_files = (
            "fit_audits.csv",
            "ranking_stability.csv",
            "rankings.csv",
            "source_values.parquet",
            "states.parquet",
            "trajectories.parquet",
        )
        complete = {
            "outer_domain": source_labels.outer_domain,
            "candidate_bank_state_sha256": source_labels.candidate_bank_state_sha256,
            "source_label_state_sha256": source_labels.state_sha256,
            "evaluator_model_state_sha256": evaluator_models.state_sha256,
            "evaluation_state_sha256": evaluation.state_sha256,
            "target_specimen_count": len(evaluation.target_specimen_ids),
            "source_value_rows": len(source_labels.rows),
            "state_rows": len(evaluation.states),
            "trajectory_rows": len(evaluation.trajectories),
            "ranking_rows": len(ranking_rows),
            "fit_audit_rows": len(fit_audit_rows),
            "file_sha256": {
                name: _sha256_file(temporary / name) for name in data_files
            },
        }
        (temporary / "complete.json").write_text(
            json.dumps(
                complete,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _validate_published_outer_shard(
            temporary,
            source_label_state_sha256=source_labels.state_sha256,
            evaluator_model_state_sha256=evaluator_models.state_sha256,
            evaluation_state_sha256=evaluation.state_sha256,
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def _selected_initial_budgets(
    root: Path, domain_order: tuple[str, ...], source_path: Path
) -> dict[str, float]:
    try:
        payload = json.loads((root / source_path).read_text(encoding="utf-8"))
        raw = payload["initial_survey_selection"]
        selected = {str(domain): float(value) for domain, value in raw.items()}
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise A4ExecutionError("A1 initial survey selection cannot be read") from error
    if set(selected) != set(domain_order) or any(
        value not in {0.015625, 0.03125} for value in selected.values()
    ):
        raise A4ExecutionError("A1 initial survey selection changed")
    return {domain: selected[domain] for domain in domain_order}


def _token(value: float) -> str:
    return str(value).replace(".", "p")


def _load_uniform_embeddings(
    root: Path,
    authority: MVAAuthority,
    *,
    initial_budget: float,
    checkpoints: tuple[float, ...],
) -> dict[float, np.ndarray]:
    path = root / "results/mva/.work" / f"uniform_bank_{_token(initial_budget)}.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            specimen_ids = tuple(str(value) for value in archive["specimen_ids"])
            authority_state = str(archive["authority_state"][0])
            saved_budget = float(archive["initial_budget"][0])
            output = {
                checkpoint: np.asarray(
                    archive[f"embedding_{_token(checkpoint)}"], dtype=np.float64
                )
                for checkpoint in checkpoints
            }
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        raise A4ExecutionError("uniform embedding bank cannot be loaded") from error
    if (
        specimen_ids != authority.specimen_ids
        or authority_state != authority.state_sha256
        or saved_budget != initial_budget
        or any(
            value.shape != (authority.specimen_count, 512)
            or not np.all(np.isfinite(value))
            for value in output.values()
        )
    ):
        raise A4ExecutionError("uniform embedding bank authority changed")
    return output


def prepare_a4_candidate_bank(
    config_path: str | Path,
    *,
    project_root: str | Path,
    initial_budget: float,
    device: str,
) -> Path:
    """Build or validate one registered A4 candidate cache."""

    root = Path(project_root).resolve(strict=True)
    a4_config = load_a4_config(config_path, project_root=root)
    base_config = load_mva_config(
        root / a4_config.sources["a0_a3_config"].path, project_root=root
    )
    if initial_budget not in {0.015625, 0.03125}:
        raise A4ExecutionError("A4 candidate budget is not selected by any outer fold")
    authority = load_mva_authority(base_config, project_root=root)
    output = candidate_bank_path(root, initial_budget)
    if output.is_file():
        load_candidate_bank(
            output,
            expected_authority_state_sha256=authority.state_sha256,
            expected_specimen_ids=authority.specimen_ids,
            expected_image_sha256=authority.image_sha256,
            expected_initial_budget=initial_budget,
        )
        return output
    encoder = MVAEncoderSession(_encoder(root, device))
    bank = build_candidate_bank(
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        images=authority.images,
        image_sha256=authority.image_sha256,
        authority_state_sha256=authority.state_sha256,
        initial_budget=initial_budget,
        encoder=encoder,
    )
    save_candidate_bank(output, bank)
    load_candidate_bank(
        output,
        expected_authority_state_sha256=authority.state_sha256,
        expected_specimen_ids=authority.specimen_ids,
        expected_image_sha256=authority.image_sha256,
        expected_initial_budget=initial_budget,
    )
    return output


def run_a4_outer_worker(
    config_path: str | Path,
    *,
    project_root: str | Path,
    outer_domain: str,
    device: str,
) -> Path:
    """Execute and transactionally publish one complete formal A4 outer shard."""

    root = Path(project_root).resolve(strict=True)
    a4_config = load_a4_config(config_path, project_root=root)
    if outer_domain not in a4_config.domain_order:
        raise A4ExecutionError("outer domain is not registered")
    base_config = load_mva_config(
        root / a4_config.sources["a0_a3_config"].path, project_root=root
    )
    authority = load_mva_authority(base_config, project_root=root)
    selected = _selected_initial_budgets(
        root,
        a4_config.domain_order,
        a4_config.sources["a1_summary"].path,
    )
    initial_budget = selected[outer_domain]
    bank = load_candidate_bank(
        candidate_bank_path(root, initial_budget),
        expected_authority_state_sha256=authority.state_sha256,
        expected_specimen_ids=authority.specimen_ids,
        expected_image_sha256=authority.image_sha256,
        expected_initial_budget=initial_budget,
    )
    source_labels = generate_source_labels(
        outer_domain=outer_domain,
        domain_order=a4_config.domain_order,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        targets=authority.targets,
        metadata=authority.metadata13,
        bank=bank,
        pca_dimensions=a4_config.pca_dimensions,
        ridge_alpha=a4_config.ridge_alpha,
        tie_tolerance=1.0e-12,
    )
    uniform_embeddings = _load_uniform_embeddings(
        root,
        authority,
        initial_budget=initial_budget,
        checkpoints=a4_config.checkpoints,
    )
    evaluator_models = fit_outer_evaluation_models(
        outer_domain=outer_domain,
        domain_order=a4_config.domain_order,
        checkpoints=a4_config.checkpoints,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        targets=authority.targets,
        metadata=authority.metadata13,
        full_embeddings=authority.full_embeddings,
        uniform_embeddings=uniform_embeddings,
        pca_dimensions=a4_config.pca_dimensions,
        ridge_alpha=a4_config.ridge_alpha,
        tie_tolerance=1.0e-12,
    )
    target_indices = np.flatnonzero(
        np.asarray(authority.dataset_ids, dtype=object) == outer_domain
    )
    evaluation = evaluate_outer_static_masks(
        outer_domain=outer_domain,
        domain_order=a4_config.domain_order,
        specimen_ids=tuple(authority.specimen_ids[index] for index in target_indices),
        dataset_ids=tuple(authority.dataset_ids[index] for index in target_indices),
        images=tuple(authority.images[index] for index in target_indices),
        targets=authority.targets[target_indices],
        metadata=authority.metadata13[target_indices],
        initial_budget=initial_budget,
        checkpoints=a4_config.checkpoints,
        rankings=source_labels.rankings,
        source_specimen_ids=source_labels.source_specimen_ids,
        source_label_state_sha256=source_labels.state_sha256,
        p_a_model=evaluator_models.p_a_model,
        p_b_models=evaluator_models.p_b_models,
        encoder=MVAEncoderSession(_encoder(root, device)),
    )
    return publish_outer_shard(
        root / "results/mva/.work/a4_domains" / outer_domain,
        source_labels=source_labels,
        evaluator_models=evaluator_models,
        evaluation=evaluation,
    )


def _encode_many(encoder: _Encoder, images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise A4ExecutionError("outer evaluation has no images to encode")
    output = np.asarray(encoder.encode(images), dtype=np.float64)
    if output.shape != (len(images), 512) or not np.all(np.isfinite(output)):
        raise A4ExecutionError("outer encoder output is invalid")
    return output


def _validate_request(
    *,
    outer_domain: str,
    domain_order: tuple[str, ...],
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    images: tuple[np.ndarray, ...],
    targets: object,
    metadata: object,
    initial_budget: float,
    checkpoints: tuple[float, ...],
    rankings: tuple[GlobalMaskRanking, ...],
    source_specimen_ids: tuple[str, ...],
    source_label_state_sha256: str,
    p_a_model: _Predictor,
    p_b_models: Mapping[float, _Predictor],
    encoder: _Encoder,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(specimen_ids)
    try:
        response = np.asarray(targets, dtype=np.float64)
        meta = np.asarray(metadata, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise A4ExecutionError("outer evaluation arrays must be numeric") from error
    source_domains = tuple(domain for domain in domain_order if domain != outer_domain)
    if (
        type(domain_order) is not tuple
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
        or outer_domain not in domain_order
        or type(specimen_ids) is not tuple
        or type(dataset_ids) is not tuple
        or type(images) is not tuple
        or count < 1
        or len(set(specimen_ids)) != count
        or len(dataset_ids) != count
        or set(dataset_ids) != {outer_domain}
        or len(images) != count
        or response.shape != (count,)
        or meta.ndim != 2
        or meta.shape[0] != count
        or not np.all(np.isfinite(response))
        or np.any(np.isinf(meta))
        or initial_budget not in INITIAL_BUDGETS
        or type(checkpoints) is not tuple
        or not checkpoints
        or any(
            not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0
            for value in checkpoints
        )
        or tuple(sorted(set(checkpoints))) != checkpoints
        or type(source_specimen_ids) is not tuple
        or not source_specimen_ids
        or len(set(source_specimen_ids)) != len(source_specimen_ids)
        or set(specimen_ids) & set(source_specimen_ids)
        or not _is_sha256(source_label_state_sha256)
    ):
        raise A4ExecutionError("outer evaluation authority changed")
    if any(
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        for image in images
    ):
        raise A4ExecutionError("outer target image is invalid")
    if (
        type(rankings) is not tuple
        or tuple(ranking.method for ranking in rankings) != METHODS
        or any(
            type(ranking) is not GlobalMaskRanking
            or ranking.outer_domain != outer_domain
            or ranking.source_domains != source_domains
            or ranking.source_specimen_count != len(source_specimen_ids)
            or set(ranking.cell_order) != set(range(64))
            for ranking in rankings
        )
    ):
        raise A4ExecutionError("outer source ranking authority changed")
    if (
        set(p_a_model.fit_domains) != set(source_domains)
        or not _is_sha256(p_a_model.state_sha256)
        or set(p_b_models) != set(checkpoints)
        or any(
            set(model.fit_domains) != set(source_domains)
            or not _is_sha256(model.state_sha256)
            for model in p_b_models.values()
        )
        or not callable(getattr(encoder, "encode", None))
        or not callable(getattr(encoder, "validate", None))
    ):
        raise A4ExecutionError("outer evaluator information barrier changed")
    return response, meta


def validate_outer_evaluation(result: StaticOuterEvaluation) -> None:
    """Recompute target roster, budget, error, shared-head, and action invariants."""

    if type(result) is not StaticOuterEvaluation:
        raise A4ExecutionError("issued outer evaluation is required")
    expected_state_count = (
        len(result.target_specimen_ids) * len(METHODS) * len(result.checkpoints)
    )
    if (
        result.source_domains
        != tuple(
            domain for domain in result.domain_order if domain != result.outer_domain
        )
        or set(result.target_specimen_ids) & set(result.source_specimen_ids)
        or len(result.states) != expected_state_count
        or not _is_sha256(result.source_label_state_sha256)
        or not _is_sha256(result.p_a_predictor_state_sha256)
        or tuple(method for method, _order in result.cell_orders) != METHODS
    ):
        raise A4ExecutionError("outer evaluation roster changed")
    p_b_states = dict(result.p_b_predictor_states)
    if set(p_b_states) != set(result.checkpoints) or not all(
        _is_sha256(value) for value in p_b_states.values()
    ):
        raise A4ExecutionError("outer P-B roster changed")
    state_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in result.states:
        if set(row) != _STATE_KEYS:
            raise A4ExecutionError("outer state schema changed")
        specimen_id = row["specimen_id"]
        method = row["method"]
        checkpoint = row["nominal_checkpoint"]
        measured = row["measured_count"]
        native = row["native_count"]
        effective = row["effective_budget"]
        target = row["target"]
        p_a = row["p_a_prediction"]
        p_b = row["p_b_prediction"]
        actions = row["cumulative_actions"]
        if (
            specimen_id not in result.target_specimen_ids
            or row["dataset_id"] != result.outer_domain
            or row["outer_domain"] != result.outer_domain
            or method not in METHODS
            or checkpoint not in result.checkpoints
            or type(measured) is not int
            or type(native) is not int
            or type(actions) is not int
            or not 0 < measured <= native
            or not 0 <= actions <= 64
            or not math.isclose(
                float(effective), measured / native, rel_tol=0.0, abs_tol=0.0
            )
            or (actions > 0 and float(effective) > float(checkpoint))
            or row["initial_budget"] != result.initial_budget
            or row["source_label_state_sha256"]
            != result.source_label_state_sha256
            or row["ranking_source_domains"] != "|".join(result.source_domains)
            or row["p_a_predictor_state_sha256"]
            != result.p_a_predictor_state_sha256
            or row["p_b_predictor_state_sha256"] != p_b_states[checkpoint]
            or not all(
                math.isfinite(float(value))
                for value in (
                    effective,
                    target,
                    p_a,
                    p_b,
                    row["normalized_rgb_mse"],
                    row["ssim"],
                )
            )
            or float(row["normalized_rgb_mse"]) < 0.0
            or not math.isclose(
                float(row["p_a_absolute_error"]),
                abs(float(target) - float(p_a)),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                float(row["p_b_absolute_error"]),
                abs(float(target) - float(p_b)),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise A4ExecutionError("outer state row changed")
        state_groups.setdefault((str(specimen_id), str(method)), []).append(row)
    if len(state_groups) != len(result.target_specimen_ids) * len(METHODS):
        raise A4ExecutionError("outer state groups are incomplete")
    for rows in state_groups.values():
        ordered = sorted(rows, key=lambda row: float(row["nominal_checkpoint"]))
        if (
            tuple(float(row["nominal_checkpoint"]) for row in ordered)
            != result.checkpoints
            or any(
                int(second["measured_count"]) < int(first["measured_count"])
                or int(second["cumulative_actions"])
                < int(first["cumulative_actions"])
                for first, second in pairwise(ordered)
            )
        ):
            raise A4ExecutionError("outer state trajectory changed")

    orders = dict(result.cell_orders)
    trajectory_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in result.trajectories:
        if set(row) != _TRAJECTORY_KEYS:
            raise A4ExecutionError("outer trajectory schema changed")
        specimen_id = row["specimen_id"]
        method = row["method"]
        position = row["ranking_position"]
        if (
            specimen_id not in result.target_specimen_ids
            or row["dataset_id"] != result.outer_domain
            or row["outer_domain"] != result.outer_domain
            or method not in METHODS
            or type(position) is not int
            or not 0 <= position < 64
            or row["cell_index"] != orders[method][position]
            or row["from_level"] != 0
            or row["to_level"] != 1
            or row["nominal_checkpoint"] not in result.checkpoints
            or row["source_label_state_sha256"]
            != result.source_label_state_sha256
            or not math.isfinite(float(row["ranking_score"]))
        ):
            raise A4ExecutionError("outer trajectory row changed")
        trajectory_groups.setdefault((str(specimen_id), str(method)), []).append(row)
    for key, state_rows in state_groups.items():
        actions = sorted(
            trajectory_groups.get(key, []), key=lambda row: int(row["ranking_position"])
        )
        final_actions = max(int(row["cumulative_actions"]) for row in state_rows)
        if (
            len(actions) != final_actions
            or tuple(int(row["ranking_position"]) for row in actions)
            != tuple(range(final_actions))
        ):
            raise A4ExecutionError("outer action prefix changed")
    if _result_state(result) != result.state_sha256:
        raise A4ExecutionError("outer evaluation content digest changed")


def evaluate_outer_static_masks(
    *,
    outer_domain: str,
    domain_order: tuple[str, ...],
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    images: tuple[np.ndarray, ...],
    targets: object,
    metadata: object,
    initial_budget: float,
    checkpoints: tuple[float, ...],
    rankings: tuple[GlobalMaskRanking, ...],
    source_specimen_ids: tuple[str, ...],
    source_label_state_sha256: str,
    p_a_model: _Predictor,
    p_b_models: Mapping[float, _Predictor],
    encoder: _Encoder,
) -> StaticOuterEvaluation:
    """Evaluate fixed rankings after action selection, with one shared P-B head."""

    response, meta = _validate_request(
        outer_domain=outer_domain,
        domain_order=domain_order,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        images=images,
        targets=targets,
        metadata=metadata,
        initial_budget=initial_budget,
        checkpoints=checkpoints,
        rankings=rankings,
        source_specimen_ids=source_specimen_ids,
        source_label_state_sha256=source_label_state_sha256,
        p_a_model=p_a_model,
        p_b_models=p_b_models,
        encoder=encoder,
    )
    states: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    source_domains = tuple(domain for domain in domain_order if domain != outer_domain)
    for specimen_index, (specimen_id, dataset_id, image) in enumerate(
        zip(specimen_ids, dataset_ids, images, strict=True)
    ):
        grid = build_acquisition_grid(
            image.shape[0], image.shape[1], initial_budget=initial_budget
        )
        patch_cache = RefinementPatchCache(image=image, grid=grid)
        entries: list[tuple[GlobalMaskRanking, object]] = []
        for ranking in rankings:
            trajectory = run_static_mask_trajectory(
                grid,
                initial_state(grid),
                cell_order=ranking.cell_order,
                checkpoints=checkpoints,
                method=ranking.method,
            )
            snapshots = _materialize_control(
                image,
                grid,
                trajectory,
                specimen_id=specimen_id,
                dataset_id=dataset_id,
                patch_cache=patch_cache,
            )
            previous = 0
            for snapshot in trajectory.snapshots:
                for position in range(previous, snapshot.cumulative_actions):
                    action = trajectory.actions[position]
                    trajectories.append(
                        {
                            "specimen_id": specimen_id,
                            "dataset_id": dataset_id,
                            "outer_domain": outer_domain,
                            "method": ranking.method,
                            "ranking_position": position,
                            "cell_index": action.cell_index,
                            "from_level": action.from_level,
                            "to_level": action.to_level,
                            "nominal_checkpoint": snapshot.nominal_checkpoint,
                            "ranking_score": ranking.cell_scores[action.cell_index],
                            "source_label_state_sha256": source_label_state_sha256,
                        }
                    )
                previous = snapshot.cumulative_actions
            entries.extend((ranking, snapshot) for snapshot in snapshots)
        vectors = _encode_many(encoder, [entry[1].image for entry in entries])
        for row_index, (ranking, snapshot) in enumerate(entries):
            checkpoint = snapshot.checkpoint
            vector = vectors[row_index : row_index + 1]
            specimen_metadata = meta[specimen_index : specimen_index + 1]
            p_a = float(p_a_model.predict(specimen_metadata, vector)[0])
            p_b = float(p_b_models[checkpoint].predict(specimen_metadata, vector)[0])
            target = float(response[specimen_index])
            states.append(
                {
                    "specimen_id": specimen_id,
                    "dataset_id": dataset_id,
                    "outer_domain": outer_domain,
                    "method": ranking.method,
                    "initial_budget": initial_budget,
                    "nominal_checkpoint": checkpoint,
                    "measured_count": snapshot.measured_count,
                    "native_count": snapshot.native_count,
                    "effective_budget": snapshot.effective_budget,
                    "cumulative_actions": snapshot.state.levels.count(1),
                    "target": target,
                    "p_a_prediction": p_a,
                    "p_a_absolute_error": abs(target - p_a),
                    "p_b_prediction": p_b,
                    "p_b_absolute_error": abs(target - p_b),
                    "normalized_rgb_mse": normalized_rgb_mse(image, snapshot.image),
                    "ssim": _global_ssim(image, snapshot.image),
                    "p_a_predictor_state_sha256": p_a_model.state_sha256,
                    "p_b_predictor_state_sha256": p_b_models[
                        checkpoint
                    ].state_sha256,
                    "ranking_source_domains": "|".join(source_domains),
                    "source_label_state_sha256": source_label_state_sha256,
                }
            )
    encoder.validate()
    result = StaticOuterEvaluation(
        outer_domain=outer_domain,
        domain_order=domain_order,
        source_domains=source_domains,
        target_specimen_ids=specimen_ids,
        source_specimen_ids=source_specimen_ids,
        initial_budget=float(initial_budget),
        checkpoints=checkpoints,
        source_label_state_sha256=source_label_state_sha256,
        p_a_predictor_state_sha256=p_a_model.state_sha256,
        p_b_predictor_states=tuple(
            (checkpoint, p_b_models[checkpoint].state_sha256)
            for checkpoint in checkpoints
        ),
        cell_orders=tuple(
            (ranking.method, ranking.cell_order) for ranking in rankings
        ),
        states=tuple(states),
        trajectories=tuple(trajectories),
        state_sha256="",
    )
    output = replace(result, state_sha256=_result_state(result))
    validate_outer_evaluation(output)
    return output


__all__ = [
    "A4ExecutionError",
    "EvaluatorFitAudit",
    "OuterEvaluationModels",
    "StaticOuterEvaluation",
    "evaluate_outer_static_masks",
    "fit_outer_evaluation_models",
    "prepare_a4_candidate_bank",
    "publish_outer_shard",
    "run_a4_outer_worker",
    "validate_outer_evaluation",
    "validate_outer_evaluation_models",
]
