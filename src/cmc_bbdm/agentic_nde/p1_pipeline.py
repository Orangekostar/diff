"""Restartable P1 score-freeze and formal pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .p1 import P1Config, load_p1_config
from .p1_artifacts import write_p1_package
from .p1_cai import (
    P1AggregateEvaluation,
    P1OuterCAIEvaluation,
    aggregate_p1_evaluation,
    run_p1_cai_outer,
)
from .p1_execution import (
    evaluate_p1_outer_scores,
    freeze_p1_outer_predictions,
    load_p1_outer_data,
)
from .surface_cells import SurfaceCellAuthority, load_surface_cell_authority
from .surface_encoder import (
    SurfaceFeatureBank,
    build_surface_resnet18,
    load_surface_feature_bank,
    materialize_surface_feature_bank,
)
from .visual_observability import (
    FrozenOuterScores,
    VisualExamples,
    fit_outer_visual_models,
    freeze_outer_scores,
    load_frozen_c0_scores,
    load_p1_deployable_authority,
)


class P1RunError(ValueError):
    """Raised when a P1 run shard is incomplete, changed, or unsafe."""


_SCORE_FREEZE_FILES = {"scores.parquet", "selection.csv", "manifest.json"}
_FORBIDDEN_SCORE_COLUMNS = {
    "cai",
    "mechanical_value",
    "oracle",
    "selected_action",
    "target",
    "teacher",
}
_CAI_SHARD_FILES = {"acquisition_curves.parquet", "manifest.json"}


@dataclass(frozen=True, slots=True)
class P1FormalResult:
    package: Path
    decision_artifact: Path
    spatial_analysis_artifact: Path
    summary: Mapping[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as error:
        raise P1RunError("P1 score-freeze metadata is not canonicalizable") from error
    return (text + "\n").encode("ascii")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise P1RunError(f"{label} mapping changed")
    return value


def _score_table(frozen: FrozenOuterScores) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for specimen_index, (specimen_id, dataset_id) in enumerate(
        zip(frozen.specimen_ids, frozen.dataset_ids, strict=True)
    ):
        for method in frozen.methods:
            for cell_index in range(64):
                rows.append(
                    {
                        "cell_index": cell_index,
                        "dataset_id": dataset_id,
                        "method": method,
                        "model_state_sha256": frozen.model_state_sha256[method],
                        "outer_domain": frozen.outer_domain,
                        "predicted_score": float(
                            frozen.scores[method][specimen_index, cell_index]
                        ),
                        "specimen_id": specimen_id,
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["outer_domain", "specimen_id", "method", "cell_index"]
    )


def write_p1_score_freeze(
    destination: str | Path,
    frozen: FrozenOuterScores,
    *,
    selection_audit: pl.DataFrame,
) -> Path:
    """Persist target scores and selection identity before target labels are read."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise P1RunError("P1 score-freeze output already exists")
    if (
        type(frozen) is not FrozenOuterScores
        or type(selection_audit) is not pl.DataFrame
        or selection_audit.is_empty()
        or "outer_domain" not in selection_audit.columns
        or set(selection_audit["outer_domain"]) != {frozen.outer_domain}
    ):
        raise P1RunError("P1 score-freeze inputs changed")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise P1RunError("P1 score-freeze parent is unavailable")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        scores = _score_table(frozen)
        scores.write_parquet(
            temporary / "scores.parquet", compression="zstd", statistics=True
        )
        sort_keys = [
            value
            for value in (
                "outer_domain",
                "stage",
                "representation",
                "candidate_id",
                "validation_domain",
                "method",
            )
            if value in selection_audit.columns
        ]
        selection = selection_audit.sort(sort_keys, nulls_last=True)
        selection.write_csv(temporary / "selection.csv")
        manifest = {
            "dataset_ids": list(frozen.dataset_ids),
            "files": {
                "scores.parquet": _sha256_file(temporary / "scores.parquet"),
                "selection.csv": _sha256_file(temporary / "selection.csv"),
            },
            "frozen_score_state_sha256": frozen.state_sha256,
            "inference_state_sha256": frozen.inference_state_sha256,
            "methods": list(frozen.methods),
            "model_state_sha256": dict(frozen.model_state_sha256),
            "outer_domain": frozen.outer_domain,
            "schema_version": 1,
            "selection_state_sha256": frozen.selection_state_sha256,
            "specimen_ids": list(frozen.specimen_ids),
        }
        (temporary / "manifest.json").write_bytes(_json(manifest))
        load_p1_score_freeze(temporary, inference=None)
        if target.exists() or target.is_symlink():
            raise P1RunError("P1 score-freeze output already exists")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _load_manifest(root: Path) -> dict[str, object]:
    try:
        raw = (root / "manifest.json").read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P1RunError("P1 score-freeze manifest is invalid") from error
    if raw != _json(value) or type(value) is not dict:
        raise P1RunError("P1 score-freeze manifest is not canonical")
    required = {
        "dataset_ids",
        "files",
        "frozen_score_state_sha256",
        "inference_state_sha256",
        "methods",
        "model_state_sha256",
        "outer_domain",
        "schema_version",
        "selection_state_sha256",
        "specimen_ids",
    }
    if (
        set(value) != required
        or value["schema_version"] != 1
        or type(value["files"]) is not dict
        or set(value["files"]) != {"scores.parquet", "selection.csv"}
    ):
        raise P1RunError("P1 score-freeze manifest schema changed")
    return value


def load_p1_score_freeze(
    path: str | Path, *, inference: VisualExamples | None
) -> tuple[FrozenOuterScores | None, pl.DataFrame]:
    """Verify a disk freeze and, with inference, recompute its score state."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise P1RunError("P1 score-freeze directory is unavailable")
    entries = tuple(root.iterdir())
    if (
        any(value.is_symlink() or not value.is_file() for value in entries)
        or {value.name for value in entries} != _SCORE_FREEZE_FILES
    ):
        raise P1RunError("P1 score-freeze membership changed")
    manifest = _load_manifest(root)
    for name, expected in manifest["files"].items():
        if type(expected) is not str or _sha256_file(root / name) != expected:
            raise P1RunError(f"P1 score-freeze file hash changed: {name}")
    try:
        scores = pl.read_parquet(root / "scores.parquet")
        selection = pl.read_csv(root / "selection.csv")
    except (OSError, pl.exceptions.PolarsError) as error:
        raise P1RunError("P1 score-freeze table cannot be read") from error
    required = {
        "cell_index",
        "dataset_id",
        "method",
        "model_state_sha256",
        "outer_domain",
        "predicted_score",
        "specimen_id",
    }
    if (
        set(scores.columns) != required
        or _FORBIDDEN_SCORE_COLUMNS & set(scores.columns)
        or scores.unique(
            subset=["outer_domain", "specimen_id", "method", "cell_index"]
        ).height
        != scores.height
        or not bool(scores.select(pl.col("predicted_score").is_finite().all()).item())
        or set(scores["outer_domain"]) != {manifest["outer_domain"]}
    ):
        raise P1RunError("P1 score-freeze score schema changed")
    if inference is None:
        return None, selection
    if (
        type(inference) is not VisualExamples
        or inference.role != "outer_inference"
        or inference.mechanical_values is not None
        or inference.outer_domain != manifest["outer_domain"]
        or inference.state_sha256 != manifest["inference_state_sha256"]
        or list(inference.specimen_ids) != manifest["specimen_ids"]
        or list(inference.dataset_ids) != manifest["dataset_ids"]
    ):
        raise P1RunError("P1 score-freeze inference identity changed")
    methods = tuple(str(value) for value in manifest["methods"])
    if tuple(sorted(methods)) != methods or set(scores["method"]) != set(methods):
        raise P1RunError("P1 score-freeze method roster changed")
    matrices: dict[str, np.ndarray] = {}
    model_states: dict[str, str] = {}
    for method in methods:
        table = scores.filter(pl.col("method") == method)
        grouped = {
            str(key[0] if isinstance(key, tuple) else key): value.sort("cell_index")
            for key, value in table.partition_by(
                "specimen_id", as_dict=True, include_key=False
            ).items()
        }
        matrix = np.empty((inference.specimen_count, 64), dtype=np.float64)
        states = set(table["model_state_sha256"])
        if len(states) != 1:
            raise P1RunError("P1 score-freeze model state changed")
        for index, specimen_id in enumerate(inference.specimen_ids):
            rows = grouped.get(specimen_id)
            if rows is None or tuple(rows["cell_index"]) != tuple(range(64)):
                raise P1RunError("P1 score-freeze cell roster changed")
            matrix[index] = rows["predicted_score"].to_numpy()
        matrices[method] = matrix
        model_states[method] = str(states.pop())
    frozen = freeze_outer_scores(
        inference,
        scores=matrices,
        model_state_sha256=model_states,
        selection_state_sha256=str(manifest["selection_state_sha256"]),
    )
    if frozen.state_sha256 != manifest["frozen_score_state_sha256"]:
        raise P1RunError("P1 score-freeze state hash changed")
    return frozen, selection


def _write_p1_cai_shard(
    destination: Path,
    evaluation: P1OuterCAIEvaluation,
    *,
    data_state_sha256: str,
    score_evaluation_state_sha256: str,
) -> Path:
    if destination.exists() or destination.is_symlink():
        raise P1RunError("P1 CAI shard already exists")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise P1RunError("P1 CAI shard parent is unavailable")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        curves_path = staging / "acquisition_curves.parquet"
        evaluation.acquisition_curves.sort(
            ["outer_domain", "specimen_id", "method", "nominal_checkpoint"]
        ).write_parquet(curves_path, compression="zstd", statistics=True)
        manifest = {
            "data_state_sha256": data_state_sha256,
            "evaluator_state_sha256": evaluation.evaluator_state_sha256,
            "files": {
                "acquisition_curves.parquet": _sha256_file(curves_path),
            },
            "outer_domain": evaluation.outer_domain,
            "schema_version": 1,
            "score_evaluation_state_sha256": score_evaluation_state_sha256,
            "state_sha256": evaluation.state_sha256,
        }
        (staging / "manifest.json").write_bytes(_json(manifest))
        _load_p1_cai_shard(
            staging,
            outer_domain=evaluation.outer_domain,
            data_state_sha256=data_state_sha256,
            score_evaluation_state_sha256=score_evaluation_state_sha256,
        )
        if destination.exists() or destination.is_symlink():
            raise P1RunError("P1 CAI shard already exists")
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _load_p1_cai_shard(
    path: Path,
    *,
    outer_domain: str,
    data_state_sha256: str,
    score_evaluation_state_sha256: str,
) -> P1OuterCAIEvaluation:
    if path.is_symlink() or not path.is_dir():
        raise P1RunError("P1 CAI shard is unavailable")
    entries = tuple(path.iterdir())
    if (
        any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or {entry.name for entry in entries} != _CAI_SHARD_FILES
    ):
        raise P1RunError("P1 CAI shard membership changed")
    manifest = _load_manifest_file(path / "manifest.json", "P1 CAI shard")
    required = {
        "data_state_sha256",
        "evaluator_state_sha256",
        "files",
        "outer_domain",
        "schema_version",
        "score_evaluation_state_sha256",
        "state_sha256",
    }
    files = manifest.get("files")
    if (
        set(manifest) != required
        or manifest["schema_version"] != 1
        or manifest["outer_domain"] != outer_domain
        or manifest["data_state_sha256"] != data_state_sha256
        or manifest["score_evaluation_state_sha256"]
        != score_evaluation_state_sha256
        or type(files) is not dict
        or set(files) != {"acquisition_curves.parquet"}
        or type(files["acquisition_curves.parquet"]) is not str
        or _sha256_file(path / "acquisition_curves.parquet")
        != files["acquisition_curves.parquet"]
    ):
        raise P1RunError("P1 CAI shard identity changed")
    try:
        curves = pl.read_parquet(path / "acquisition_curves.parquet").sort(
            ["outer_domain", "specimen_id", "method", "nominal_checkpoint"]
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise P1RunError("P1 CAI shard table cannot be read") from error
    if (
        curves.is_empty()
        or set(curves["outer_domain"]) != {outer_domain}
        or type(manifest["evaluator_state_sha256"]) is not str
        or type(manifest["state_sha256"]) is not str
    ):
        raise P1RunError("P1 CAI shard values changed")
    return P1OuterCAIEvaluation(
        outer_domain=outer_domain,
        acquisition_curves=curves,
        evaluator_state_sha256=str(manifest["evaluator_state_sha256"]),
        state_sha256=str(manifest["state_sha256"]),
    )


def _load_manifest_file(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P1RunError(f"{label} manifest is invalid") from error
    if raw != _json(value) or type(value) is not dict:
        raise P1RunError(f"{label} manifest is not canonical")
    return value


def _load_authorities(
    config: P1Config, *, feature_root: Path
) -> tuple[SurfaceCellAuthority, SurfaceFeatureBank, object, object]:
    root = config.project_root
    surface = load_surface_cell_authority(
        root / config.sources["p0r_surface_manifest"].path,
        root / config.sources["p0r_registration"].path,
        root / config.sources["p0r_grid_mapping"].path,
    )
    features = load_surface_feature_bank(
        feature_root,
        authority=surface,
        expected_transform_sha256=config.surface_transform_sha256,
    )
    deployable = load_p1_deployable_authority(config, surface)
    c0 = load_frozen_c0_scores(config, deployable)
    return surface, features, deployable, c0


def materialize_p1_features(
    config_path: str | Path,
    *,
    project_root: str | Path,
    surface_root: str | Path,
    output: str | Path,
    notify: Callable[[str], None] | None = None,
) -> SurfaceFeatureBank:
    """Reproduce the preregistered label-free P1 feature cache."""

    config = load_p1_config(config_path, project_root=project_root)
    root = config.project_root
    surface = load_surface_cell_authority(
        root / config.sources["p0r_surface_manifest"].path,
        root / config.sources["p0r_registration"].path,
        root / config.sources["p0r_grid_mapping"].path,
    )
    controls = _mapping(config.raw["controls"], "P1 controls")
    wrong = _mapping(controls["C4"], "P1 wrong-orientation control")
    encoder = build_surface_resnet18(config)
    return materialize_surface_feature_bank(
        surface,
        external_root=surface_root,
        output=output,
        encoder=encoder,
        wrong_orientation_seed=str(wrong["seed"]),
        notify=notify,
    )


def _selection_summary(selection: pl.DataFrame) -> dict[str, dict[str, object]]:
    rows = selection.filter(
        (pl.col("stage") == "FINAL_FIT") & pl.col("selected")
    )
    output: dict[str, dict[str, object]] = {}
    for outer_domain in sorted(str(value) for value in rows["outer_domain"].unique()):
        domain_rows = rows.filter(pl.col("outer_domain") == outer_domain)
        output[outer_domain] = {
            str(row["method"]): {
                "candidate_id": row["candidate_id"],
                "feature_control": row["feature_control"],
                "fusion_lambda": row["lambda"],
                "representation": row["representation"],
            }
            for row in domain_rows.iter_rows(named=True)
        }
    return output


def _authorized_roster(surface: SurfaceCellAuthority) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "dataset_id": record.dataset_id,
                "specimen_id": record.specimen_id,
                "surface_path": record.surface_path.as_posix(),
                "surface_sha256": record.surface_sha256,
                "transform_sha256": record.transform_sha256,
            }
            for record in surface.records
        ],
        infer_schema_length=None,
    ).sort(["dataset_id", "specimen_id"])


def _visual_feature_manifest(features: SurfaceFeatureBank) -> pl.DataFrame:
    arrays = {
        "global": features.global_embeddings,
        "local_correct": features.local_correct_embeddings,
        "local_wrong_orientation": features.local_wrong_orientation_embeddings,
    }
    provenance = json.dumps(
        dict(features.encoder_provenance),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return pl.DataFrame(
        [
            {
                "array_name": name,
                "authority_state_sha256": features.authority_state_sha256,
                "dtype": str(array.dtype),
                "encoder_provenance": provenance,
                "feature_bank_state_sha256": features.state_sha256,
                "feature_manifest_sha256": features.manifest_sha256,
                "sha256": features.array_sha256[name],
                "shape": "x".join(str(value) for value in array.shape),
                "transform_sha256": features.transform_sha256,
            }
            for name, array in sorted(arrays.items())
        ],
        infer_schema_length=None,
    )


def _key_metrics(control_results: pl.DataFrame) -> dict[str, dict[str, float]]:
    columns = (
        "cai_auebc",
        "next_action_regret",
        "ndcg_10",
        "one_step_cai_utility",
    )
    return {
        str(row["method"]): {
            column: float(row[column]) for column in columns if column in row
        }
        for row in control_results.iter_rows(named=True)
    }


def _summary(
    config: P1Config,
    surface: SurfaceCellAuthority,
    features: SurfaceFeatureBank,
    aggregate: P1AggregateEvaluation,
    selection: pl.DataFrame,
) -> dict[str, object]:
    decision = aggregate.decision
    return {
        "aggregate_state_sha256": aggregate.state_sha256,
        "authorized_route": decision.authorized_route,
        "bootstrap_resamples": config.bootstrap_resamples,
        "config_sha256": config.config_sha256,
        "decision_state_sha256": decision.state_sha256,
        "domain_count": len(config.domain_order),
        "domain_order": list(config.domain_order),
        "effects": {
            name: {
                "improved_domains": effect.improved_domains,
                "lower": effect.lower,
                "point_estimate": effect.point_estimate,
                "state_sha256": effect.state_sha256,
                "upper": effect.upper,
            }
            for name, effect in sorted(decision.effects.items())
        },
        "feature_bank_state_sha256": features.state_sha256,
        "global_conditions": dict(decision.global_conditions),
        "go": decision.go,
        "key_metrics": _key_metrics(aggregate.control_results),
        "oracle_gap_closure": decision.oracle_gap_closure,
        "outer_selection": _selection_summary(selection),
        "p0r_surface_authority_state_sha256": surface.state_sha256,
        "ranking_improvement": decision.ranking_improvement,
        "schema_version": 1,
        "spatial_conditions": dict(decision.spatial_conditions),
        "specimen_count": surface.specimen_count,
        "stage": "P1_VISUAL_OBSERVABILITY",
        "status": decision.status,
    }


def _report(summary: Mapping[str, object]) -> str:
    metrics = _mapping(summary["key_metrics"], "P1 key metrics")
    lines = [
        "# P1 Visual Observability",
        "",
        f"Status: {summary['status']}",
        f"Authorized route: {summary['authorized_route'] or 'NONE'}",
        f"Specimens: {summary['specimen_count']}",
        f"Outer domains: {summary['domain_count']}",
        f"Bootstrap resamples: {summary['bootstrap_resamples']}",
        "",
        "## Equal-domain metrics",
        "",
        "| method | CAI AUEBC | next-action regret | NDCG@10 |",
        "|---|---:|---:|---:|",
    ]
    for method, raw in sorted(metrics.items()):
        values = _mapping(raw, f"P1 {method} metrics")
        lines.append(
            f"| {method} | {float(values['cai_auebc']):.9g} | "
            f"{float(values['next_action_regret']):.9g} | "
            f"{float(values['ndcg_10']):.9g} |"
        )
    lines.extend(
        [
            "",
            "Target-domain scores were frozen before target mechanical labels were read.",
            "All selection used source domains only. CAI uses exact native-raster cost.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_texts(summary: Mapping[str, object]) -> tuple[str, str]:
    metrics = _mapping(summary["key_metrics"], "P1 key metrics")
    effects = _mapping(summary["effects"], "P1 effects")
    decision = [
        "# P1 Visual Observability Decision",
        "",
        f"- Status: `{summary['status']}`",
        f"- Authorized route: `{summary['authorized_route'] or 'NONE'}`",
        f"- Specimens: `{summary['specimen_count']}`",
        f"- Bootstrap resamples: `{summary['bootstrap_resamples']}`",
        f"- Oracle-gap closure: `{summary['oracle_gap_closure']}`",
        f"- Decision state SHA-256: `{summary['decision_state_sha256']}`",
        "",
        "## Gate Effects",
        "",
        "| effect | point | 95% CI | improved domains |",
        "|---|---:|---:|---:|",
    ]
    for name, raw in sorted(effects.items()):
        value = _mapping(raw, f"P1 {name} effect")
        decision.append(
            f"| {name} | {float(value['point_estimate']):.9g} | "
            f"[{float(value['lower']):.9g}, {float(value['upper']):.9g}] | "
            f"{value['improved_domains']}/6 |"
        )
    analysis = [
        "# P1 Spatial vs Global Context Analysis",
        "",
        "Equal-domain aggregate results from the preregistered six-fold evaluation.",
        "",
        "| method | CAI AUEBC | next-action regret | NDCG@10 |",
        "|---|---:|---:|---:|",
    ]
    for method, raw in sorted(metrics.items()):
        value = _mapping(raw, f"P1 {method} metrics")
        analysis.append(
            f"| {method} | {float(value['cai_auebc']):.9g} | "
            f"{float(value['next_action_regret']):.9g} | "
            f"{float(value['ndcg_10']):.9g} |"
        )
    analysis.extend(
        [
            "",
            "Correctly registered local evidence is interpreted only through the frozen gate.",
            "Global, shuffled, wrong-orientation, and deranged controls remain explicit.",
            "",
        ]
    )
    return "\n".join(decision) + "\n", "\n".join(analysis) + "\n"


def _write_artifact(path: Path, payload: str) -> Path:
    if path.exists() or path.is_symlink():
        raise P1RunError("P1 decision artifact already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise P1RunError("P1 decision artifact parent is unavailable")
    temporary = path.parent / f".{path.name}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise P1RunError("P1 decision artifact staging path exists")
    try:
        temporary.write_text(payload, encoding="ascii")
        os.rename(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def run_p1_visual_observability(
    config_path: str | Path,
    *,
    project_root: str | Path,
    research_root: str | Path,
    feature_root: str | Path | None = None,
    device: str = "cuda:0",
    notify: Callable[[str], None] | None = None,
) -> P1FormalResult:
    """Run or resume the exact preregistered six-domain P1 experiment."""

    config = load_p1_config(config_path, project_root=project_root)
    root = config.project_root
    work = root / config.output_work
    if work.is_symlink():
        raise P1RunError("P1 work path must not be a symlink")
    work.mkdir(parents=True, exist_ok=True)
    features_path = (
        Path(feature_root).resolve(strict=True)
        if feature_root is not None
        else work / "features"
    )
    surface, features, deployable, c0 = _load_authorities(
        config, feature_root=features_path
    )
    models = _mapping(config.raw["models"], "P1 models")
    ridge = _mapping(models["ridge"], "P1 ridge")
    mlp = _mapping(models["mlp"], "P1 MLP")
    fusion = _mapping(config.raw["fusion"], "P1 fusion")
    seeds = _mapping(config.raw["random_seeds"], "P1 random seeds")
    acquisition = _mapping(config.raw["acquisition"], "P1 acquisition")
    selection_tables: list[pl.DataFrame] = []
    state_tables: list[pl.DataFrame] = []
    ranking_tables: list[pl.DataFrame] = []
    curve_tables: list[pl.DataFrame] = []
    for outer_domain in config.domain_order:
        if notify is not None:
            notify(f"[{outer_domain}] P1 outer fold")
        data = load_p1_outer_data(
            config,
            surface,
            features,
            deployable,
            c0,
            outer_domain=outer_domain,
        )
        fold = work / "folds" / outer_domain
        score_freeze = fold / "score_freeze"
        if score_freeze.exists():
            frozen, selection = load_p1_score_freeze(
                score_freeze, inference=data.correct.inference
            )
            assert frozen is not None
        else:
            fitted = fit_outer_visual_models(
                correct=data.correct,
                shuffled=data.shuffled,
                wrong_orientation=data.wrong_orientation,
                spatial_derangement=data.spatial_derangement,
                c0_source_scores=data.c0_source_scores,
                ridge_alphas=tuple(float(value) for value in ridge["alphas"]),
                fusion_values=tuple(float(value) for value in fusion["lambdas"]),
                model_seed=int(seeds["model"]),
                epochs=int(mlp["epochs"]),
                device=device,
            )
            frozen = freeze_p1_outer_predictions(data, fitted)
            write_p1_score_freeze(
                score_freeze,
                frozen,
                selection_audit=fitted.selection_audit,
            )
            frozen, selection = load_p1_score_freeze(
                score_freeze, inference=data.correct.inference
            )
            assert frozen is not None
        score_evaluation = evaluate_p1_outer_scores(
            config, deployable, data, frozen
        )
        cai_path = fold / "cai"
        if cai_path.exists():
            cai = _load_p1_cai_shard(
                cai_path,
                outer_domain=outer_domain,
                data_state_sha256=data.state_sha256,
                score_evaluation_state_sha256=score_evaluation.state_sha256,
            )
        else:
            cai = run_p1_cai_outer(
                config,
                data,
                score_evaluation,
                research_root=research_root,
                device=device,
                notify=notify,
            )
            _write_p1_cai_shard(
                cai_path,
                cai,
                data_state_sha256=data.state_sha256,
                score_evaluation_state_sha256=score_evaluation.state_sha256,
            )
        selection_tables.append(selection)
        state_tables.append(score_evaluation.per_state_scores)
        ranking_tables.append(score_evaluation.per_specimen_metrics)
        curve_tables.append(cai.acquisition_curves)
    selection = pl.concat(selection_tables, how="vertical_relaxed")
    per_state = pl.concat(state_tables, how="vertical_relaxed")
    ranking = pl.concat(ranking_tables, how="vertical_relaxed")
    curves = pl.concat(curve_tables, how="vertical_relaxed")
    aggregate = aggregate_p1_evaluation(
        curves,
        ranking,
        domain_order=config.domain_order,
        checkpoints=tuple(float(value) for value in acquisition["checkpoints"]),
        bootstrap_seed=int(seeds["bootstrap"]),
        bootstrap_resamples=config.bootstrap_resamples,
    )
    summary = _summary(config, surface, features, aggregate, selection)
    package = write_p1_package(
        root / config.output_result,
        config_bytes=config.config_path.read_bytes(),
        tables={
            "authorized_roster": _authorized_roster(surface),
            "visual_feature_manifest": _visual_feature_manifest(features),
            "outer_model_selection": selection,
            "per_state_scores": per_state,
            "per_specimen_metrics": aggregate.per_specimen_metrics,
            "domain_metrics": aggregate.domain_metrics,
            "bootstrap": aggregate.bootstrap,
            "acquisition_curves": curves,
            "control_results": aggregate.control_results,
        },
        summary=summary,
        report=_report(summary),
    )
    outputs = _mapping(config.raw["outputs"], "P1 outputs")
    decision_text, analysis_text = _artifact_texts(summary)
    decision_artifact = _write_artifact(
        root / str(outputs["decision"]), decision_text
    )
    spatial_artifact = _write_artifact(
        root / str(outputs["spatial_analysis"]), analysis_text
    )
    return P1FormalResult(
        package=package,
        decision_artifact=decision_artifact,
        spatial_analysis_artifact=spatial_artifact,
        summary=summary,
    )


__all__ = [
    "P1FormalResult",
    "P1RunError",
    "load_p1_score_freeze",
    "materialize_p1_features",
    "run_p1_visual_observability",
    "write_p1_score_freeze",
]
