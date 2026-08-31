"""Deterministic, checksum-bound P1 visual-observability packages."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from .p1 import load_p1_config
from .p1_cai import aggregate_p1_evaluation
from .surface_cells import load_surface_cell_authority
from .surface_encoder import load_surface_feature_bank
from .visual_observability import evaluate_action_scores, select_source_candidate


class P1ArtifactError(ValueError):
    """Raised when a P1 package is incomplete, unsafe, or changed."""


REQUIRED_P1_FILES = frozenset(
    {
        "config.yaml",
        "authorized_roster.csv",
        "visual_feature_manifest.csv",
        "outer_model_selection.csv",
        "per_state_scores.parquet",
        "per_specimen_metrics.csv",
        "domain_metrics.csv",
        "bootstrap.csv",
        "acquisition_curves.csv",
        "control_results.csv",
        "summary.json",
        "REPORT.md",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
)
_PAYLOAD_FILES = REQUIRED_P1_FILES - {
    "artifact_manifest.json",
    "CHECKSUMS.sha256",
}
_TABLE_FILES = {
    "authorized_roster": "authorized_roster.csv",
    "visual_feature_manifest": "visual_feature_manifest.csv",
    "outer_model_selection": "outer_model_selection.csv",
    "per_state_scores": "per_state_scores.parquet",
    "per_specimen_metrics": "per_specimen_metrics.csv",
    "domain_metrics": "domain_metrics.csv",
    "bootstrap": "bootstrap.csv",
    "acquisition_curves": "acquisition_curves.csv",
    "control_results": "control_results.csv",
}
_SORT_KEYS = {
    "authorized_roster": ("dataset_id", "specimen_id"),
    "visual_feature_manifest": ("array_name",),
    "outer_model_selection": (
        "outer_domain",
        "stage",
        "representation",
        "candidate_id",
        "validation_domain",
        "method",
    ),
    "per_state_scores": ("outer_domain", "specimen_id", "method", "cell_index"),
    "per_specimen_metrics": ("outer_domain", "specimen_id", "method"),
    "domain_metrics": ("outer_domain", "method"),
    "bootstrap": ("effect_key",),
    "acquisition_curves": (
        "outer_domain",
        "specimen_id",
        "method",
        "nominal_checkpoint",
    ),
    "control_results": ("method",),
}
_STATUSES = {
    "P1_SPATIAL_VISUAL_OBSERVABILITY_GO",
    "P1_GLOBAL_VISUAL_CONTEXT_GO",
    "P1_DESCRIPTIVE_SPATIAL_SIGNAL_ONLY",
    "P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO",
}
_HEX = frozenset("0123456789abcdef")
_ABSOLUTE_TEXT = re.compile(
    r"(?:^|[\s:=\"'(])(?:/|\\\\|[A-Za-z]:[\\/])", flags=re.MULTILINE
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and not set(value) - _HEX


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
        raise P1ArtifactError("P1 JSON payload is not canonicalizable") from error
    return (text + "\n").encode("ascii")


def _contains_absolute_path(value: object) -> bool:
    if type(value) is str:
        return value.startswith(("/", "\\\\")) or (
            len(value) > 2 and value[1:3] in {":/", ":\\"}
        )
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _canonical_table(name: str, table: pl.DataFrame) -> pl.DataFrame:
    if type(table) is not pl.DataFrame or table.is_empty():
        raise P1ArtifactError(f"P1 {name} table is empty or invalid")
    keys = tuple(key for key in _SORT_KEYS[name] if key in table.columns)
    if not keys:
        raise P1ArtifactError(f"P1 {name} sort identity is unavailable")
    try:
        return table.sort(list(keys), nulls_last=True)
    except pl.exceptions.PolarsError as error:
        raise P1ArtifactError(f"P1 {name} cannot be sorted") from error


def _table_bytes(name: str, table: pl.DataFrame) -> bytes:
    canonical = _canonical_table(name, table)
    if name == "per_state_scores":
        stream = io.BytesIO()
        canonical.write_parquet(stream, compression="zstd", statistics=True)
        return stream.getvalue()
    try:
        return canonical.write_csv().encode("utf-8")
    except (UnicodeEncodeError, pl.exceptions.PolarsError) as error:
        raise P1ArtifactError(f"P1 {name} cannot be encoded") from error


def _validate_summary(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or value.get("schema_version") != 1
        or value.get("stage") != "P1_VISUAL_OBSERVABILITY"
        or value.get("status") not in _STATUSES
        or _contains_absolute_path(value)
    ):
        raise P1ArtifactError("P1 summary identity changed")
    return dict(value)


def write_p1_package(
    destination: str | Path,
    *,
    config_bytes: bytes,
    tables: Mapping[str, pl.DataFrame],
    summary: Mapping[str, object],
    report: str,
) -> Path:
    """Write the exact P1 package by verified atomic rename without overwrite."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise P1ArtifactError("P1 output already exists")
    if (
        type(config_bytes) is not bytes
        or not config_bytes
        or type(report) is not str
        or not report
        or _ABSOLUTE_TEXT.search(report)
        or set(tables) != set(_TABLE_FILES)
        or any(type(value) is not pl.DataFrame for value in tables.values())
    ):
        raise P1ArtifactError("P1 package input changed")
    try:
        config_bytes.decode("ascii")
        report_bytes = report.encode("ascii")
    except UnicodeError as error:
        raise P1ArtifactError("P1 text payload must be ASCII") from error
    summary_payload = _validate_summary(dict(summary))
    payload: dict[str, bytes] = {
        "config.yaml": config_bytes,
        "summary.json": _json(summary_payload),
        "REPORT.md": report_bytes,
    }
    payload.update(
        {
            filename: _table_bytes(name, tables[name])
            for name, filename in _TABLE_FILES.items()
        }
    )
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise P1ArtifactError("P1 output parent is unavailable")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        manifest = _json(
            {
                "files": {
                    name: {"sha256": _sha256(value), "size": len(value)}
                    for name, value in sorted(payload.items())
                },
                "schema_version": 1,
                "stage": "P1_VISUAL_OBSERVABILITY",
                "status": summary_payload["status"],
            }
        )
        complete = dict(payload)
        complete["artifact_manifest.json"] = manifest
        complete["CHECKSUMS.sha256"] = "".join(
            f"{_sha256(value)}  {name}\n"
            for name, value in sorted(complete.items())
        ).encode("ascii")
        if set(complete) != REQUIRED_P1_FILES:
            raise P1ArtifactError("P1 package membership is internally inconsistent")
        for name, value in complete.items():
            (temporary / name).write_bytes(value)
        replay_p1_package(temporary)
        if target.exists() or target.is_symlink():
            raise P1ArtifactError("P1 output already exists")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _load_json(root: Path, name: str) -> object:
    try:
        raw = (root / name).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P1ArtifactError(f"P1 {name} is invalid") from error
    if raw != _json(value):
        raise P1ArtifactError(f"P1 {name} is not canonical")
    return value


def replay_p1_package(path: str | Path) -> dict[str, object]:
    """Verify exact package membership, canonical metadata, and all hashes."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise P1ArtifactError("P1 package is unavailable")
    entries = tuple(root.iterdir())
    if (
        any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or {entry.name for entry in entries} != REQUIRED_P1_FILES
    ):
        raise P1ArtifactError("P1 package membership changed")
    manifest = _load_json(root, "artifact_manifest.json")
    if (
        type(manifest) is not dict
        or set(manifest) != {"files", "schema_version", "stage", "status"}
        or manifest["schema_version"] != 1
        or manifest["stage"] != "P1_VISUAL_OBSERVABILITY"
        or manifest["status"] not in _STATUSES
        or type(manifest["files"]) is not dict
        or set(manifest["files"]) != _PAYLOAD_FILES
    ):
        raise P1ArtifactError("P1 manifest schema changed")
    for name, record in manifest["files"].items():
        if (
            type(record) is not dict
            or set(record) != {"sha256", "size"}
            or type(record["size"]) is not int
            or record["size"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise P1ArtifactError("P1 manifest file record changed")
        value = (root / name).read_bytes()
        if len(value) != record["size"] or _sha256(value) != record["sha256"]:
            raise P1ArtifactError(f"P1 file hash or size changed: {name}")
    expected_checksums = "".join(
        f"{_sha256((root / name).read_bytes())}  {name}\n"
        for name in sorted(_PAYLOAD_FILES | {"artifact_manifest.json"})
    ).encode("ascii")
    if (root / "CHECKSUMS.sha256").read_bytes() != expected_checksums:
        raise P1ArtifactError("P1 checksum ledger changed")
    summary = _load_json(root, "summary.json")
    validated = _validate_summary(summary)
    if validated["status"] != manifest["status"]:
        raise P1ArtifactError("P1 summary and manifest statuses differ")
    try:
        for name, filename in _TABLE_FILES.items():
            if filename.endswith(".parquet"):
                table = pl.read_parquet(root / filename)
            else:
                table = pl.read_csv(root / filename)
            _canonical_table(name, table)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise P1ArtifactError("P1 table payload cannot be read") from error
    return validated


def _read_p1_tables(root: Path) -> dict[str, pl.DataFrame]:
    output: dict[str, pl.DataFrame] = {}
    try:
        for name, filename in _TABLE_FILES.items():
            output[name] = (
                pl.read_parquet(root / filename)
                if filename.endswith(".parquet")
                else pl.read_csv(root / filename)
            )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise P1ArtifactError("P1 table payload cannot be read") from error
    return output


def _assert_table(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    *,
    name: str,
    check_dtypes: bool = True,
) -> None:
    try:
        assert_frame_equal(
            actual,
            expected,
            check_dtypes=check_dtypes,
            check_row_order=True,
            check_column_order=True,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    except AssertionError as error:
        raise P1ArtifactError(f"P1 replay mismatch: {name}") from error


def _replay_model_selection(
    table: pl.DataFrame, *, domain_order: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    required = {
        "candidate_id",
        "feature_control",
        "lambda",
        "method",
        "ndcg_10",
        "next_action_regret",
        "outer_domain",
        "parameter_count",
        "representation",
        "selected",
        "stage",
        "validation_domain",
    }
    if (
        not required <= set(table.columns)
        or set(table["outer_domain"]) != set(domain_order)
    ):
        raise P1ArtifactError("P1 model-selection audit schema changed")
    summary: dict[str, dict[str, object]] = {}
    expected_methods = {
        "old_refit_diagnostic",
        "proposed",
        "c2_global_context",
        "c3_shuffled_surface",
        "c4_wrong_orientation",
        "c5_spatial_derangement",
        "c3_shuffled_global",
    }
    for outer_domain in domain_order:
        source_domains = tuple(value for value in domain_order if value != outer_domain)
        outer = table.filter(pl.col("outer_domain") == outer_domain)
        head_results: dict[str, object] = {}
        for representation in ("OLD", "GLOBAL", "LOCAL", "LOCAL_GLOBAL"):
            inner = outer.filter(
                (pl.col("stage") == "HEAD_INNER")
                & (pl.col("representation") == representation)
            )
            metrics = inner.select(
                "candidate_id",
                "validation_domain",
                "ndcg_10",
                "next_action_regret",
                "parameter_count",
            )
            try:
                selected = select_source_candidate(
                    metrics, domain_order=source_domains
                )
            except ValueError as error:
                raise P1ArtifactError("P1 head selection cannot replay") from error
            selected_rows = inner.filter(pl.col("selected"))
            if (
                {str(value) for value in selected_rows["candidate_id"]}
                != {selected.candidate_id}
                or {str(value) for value in selected_rows["validation_domain"]}
                != set(source_domains)
            ):
                raise P1ArtifactError("P1 head selected identity changed")
            recorded = (
                outer.filter(
                    (pl.col("stage") == "HEAD_AGGREGATE")
                    & (pl.col("representation") == representation)
                )
                .select(
                    "candidate_id",
                    "ndcg_10",
                    "next_action_regret",
                    "parameter_count",
                    "selected",
                )
                .sort("candidate_id")
            )
            _assert_table(
                recorded,
                selected.aggregates,
                name=f"{outer_domain} {representation} head selection",
                check_dtypes=False,
            )
            head_results[representation] = selected

        route_rows: list[dict[str, object]] = []
        route_recorded = outer.filter(pl.col("stage") == "CORRECT_ROUTE")
        for representation in ("LOCAL", "LOCAL_GLOBAL"):
            selected = head_results[representation]
            candidate_id = str(selected.candidate_id)
            inner = outer.filter(
                (pl.col("stage") == "HEAD_INNER")
                & (pl.col("representation") == representation)
                & (pl.col("candidate_id") == candidate_id)
            )
            for row in inner.iter_rows(named=True):
                route_rows.append(
                    {
                        "candidate_id": representation,
                        "validation_domain": row["validation_domain"],
                        "ndcg_10": row["ndcg_10"],
                        "next_action_regret": row["next_action_regret"],
                        "parameter_count": row["parameter_count"],
                    }
                )
        try:
            route = select_source_candidate(
                pl.DataFrame(route_rows), domain_order=source_domains
            )
        except ValueError as error:
            raise P1ArtifactError("P1 correct route cannot replay") from error
        expected_route = pl.DataFrame(route_rows).with_columns(
            (pl.col("candidate_id") == route.candidate_id).alias("selected")
        ).select(
            "candidate_id",
            "validation_domain",
            "ndcg_10",
            "next_action_regret",
            "parameter_count",
            "selected",
        ).sort(["candidate_id", "validation_domain"])
        recorded_route = route_recorded.select(
            "candidate_id",
            "validation_domain",
            "ndcg_10",
            "next_action_regret",
            "parameter_count",
            "selected",
        ).sort(["candidate_id", "validation_domain"])
        _assert_table(
            recorded_route,
            expected_route,
            name=f"{outer_domain} correct route",
            check_dtypes=False,
        )
        fusion_values: dict[str, float] = {}
        for stage in ("FUSION_CORRECT", "FUSION_GLOBAL"):
            fusion = outer.filter(pl.col("stage") == stage)
            try:
                selected = select_source_candidate(
                    fusion.select(
                        "candidate_id",
                        "validation_domain",
                        "ndcg_10",
                        "next_action_regret",
                        "parameter_count",
                    ),
                    domain_order=source_domains,
                )
            except ValueError as error:
                raise P1ArtifactError("P1 fusion selection cannot replay") from error
            selected_rows = fusion.filter(pl.col("selected"))
            if (
                {str(value) for value in selected_rows["candidate_id"]}
                != {selected.candidate_id}
                or selected_rows["lambda"].n_unique() != 1
            ):
                raise P1ArtifactError("P1 fusion selected identity changed")
            fusion_values[stage] = float(selected_rows["lambda"][0])
        final = outer.filter(
            (pl.col("stage") == "FINAL_FIT") & pl.col("selected")
        )
        if {str(value) for value in final["method"]} != expected_methods:
            raise P1ArtifactError("P1 final model roster changed")
        by_method = {
            str(row["method"]): row for row in final.iter_rows(named=True)
        }
        correct_representation = route.candidate_id
        correct_head = head_results[correct_representation]
        global_head = head_results["GLOBAL"]
        old_head = head_results["OLD"]
        for method in (
            "proposed",
            "c3_shuffled_surface",
            "c4_wrong_orientation",
            "c5_spatial_derangement",
        ):
            row = by_method[method]
            if (
                row["representation"] != correct_representation
                or row["candidate_id"] != correct_head.candidate_id
                or float(row["lambda"]) != fusion_values["FUSION_CORRECT"]
            ):
                raise P1ArtifactError("P1 registered final model identity changed")
        for method in ("c2_global_context", "c3_shuffled_global"):
            row = by_method[method]
            if (
                row["representation"] != "GLOBAL"
                or row["candidate_id"] != global_head.candidate_id
                or float(row["lambda"]) != fusion_values["FUSION_GLOBAL"]
            ):
                raise P1ArtifactError("P1 global final model identity changed")
        old_row = by_method["old_refit_diagnostic"]
        if (
            old_row["representation"] != "OLD"
            or old_row["candidate_id"] != old_head.candidate_id
            or old_row["lambda"] is not None
        ):
            raise P1ArtifactError("P1 old-state final model identity changed")
        summary[outer_domain] = {
            str(method): {
                "candidate_id": row["candidate_id"],
                "feature_control": row["feature_control"],
                "fusion_lambda": row["lambda"],
                "representation": row["representation"],
            }
            for method, row in by_method.items()
        }
    return summary


def _recompute_ranking_metrics(per_state: pl.DataFrame) -> pl.DataFrame:
    required = {
        "dataset_id",
        "mechanical_value",
        "method",
        "model_state_sha256",
        "outer_domain",
        "predicted_score",
        "specimen_id",
        "cell_index",
    }
    if not required <= set(per_state.columns):
        raise P1ArtifactError("P1 per-state score schema changed")
    rows: list[dict[str, object]] = []
    for key, group in per_state.group_by(
        "outer_domain", "dataset_id", "specimen_id", "method",
        maintain_order=True,
    ):
        outer_domain, dataset_id, specimen_id, method = (
            str(value) for value in key
        )
        ordered = group.sort("cell_index")
        if ordered.height != 64 or tuple(ordered["cell_index"]) != tuple(range(64)):
            raise P1ArtifactError("P1 per-state cell roster changed")
        states = {str(value) for value in ordered["model_state_sha256"]}
        if len(states) != 1:
            raise P1ArtifactError("P1 per-state model state changed")
        metrics = evaluate_action_scores(
            ordered["mechanical_value"].to_numpy(),
            ordered["predicted_score"].to_numpy(),
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "model_state_sha256": states.pop(),
                "ndcg_10": metrics.ndcg_10,
                "next_action_regret": metrics.next_action_regret,
                "one_step_cai_utility": metrics.one_step_cai_utility,
                "outer_domain": outer_domain,
                "recall_5": metrics.recall_5,
                "spearman": metrics.spearman,
                "specimen_id": specimen_id,
                "top_10_percent_overlap": metrics.top_10_percent_overlap,
                "top_1_oracle_match": metrics.top_1_oracle_match,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["outer_domain", "specimen_id", "method"]
    )


def _replay_authorities(
    root: Path,
    config: object,
    tables: Mapping[str, pl.DataFrame],
    summary: Mapping[str, object],
    *,
    feature_root: Path,
) -> None:
    surface = load_surface_cell_authority(
        root / config.sources["p0r_surface_manifest"].path,
        root / config.sources["p0r_registration"].path,
        root / config.sources["p0r_grid_mapping"].path,
    )
    expected_roster = pl.DataFrame(
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
    _assert_table(
        _canonical_table("authorized_roster", tables["authorized_roster"]),
        expected_roster,
        name="authorized roster",
        check_dtypes=False,
    )
    if (
        summary.get("p0r_surface_authority_state_sha256") != surface.state_sha256
        or summary.get("specimen_count") != surface.specimen_count
    ):
        raise P1ArtifactError("P1 P0R authority state changed")
    features = load_surface_feature_bank(
        feature_root,
        authority=surface,
        expected_transform_sha256=config.surface_transform_sha256,
    )
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
    expected_features = pl.DataFrame(
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
    _assert_table(
        _canonical_table(
            "visual_feature_manifest", tables["visual_feature_manifest"]
        ),
        expected_features,
        name="visual feature manifest",
        check_dtypes=False,
    )
    if summary.get("feature_bank_state_sha256") != features.state_sha256:
        raise P1ArtifactError("P1 feature-bank state changed")


def replay_p1_science(
    path: str | Path,
    *,
    project_root: str | Path,
    feature_root: str | Path | None = None,
    replay_output: str | Path | None = None,
) -> dict[str, object]:
    """Recompute P1 sources, selection, metrics, bootstrap, gate, and package."""

    package = Path(path).resolve(strict=True)
    summary = replay_p1_package(package)
    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        raise P1ArtifactError("P1 replay project root is unavailable") from error
    config = load_p1_config(package / "config.yaml", project_root=root)
    tables = _read_p1_tables(package)
    features_path = (
        Path(feature_root).resolve(strict=True)
        if feature_root is not None
        else root / config.output_work / "features"
    )
    _replay_authorities(
        root, config, tables, summary, feature_root=features_path
    )
    selection = _replay_model_selection(
        tables["outer_model_selection"], domain_order=config.domain_order
    )
    if summary.get("outer_selection") != selection:
        raise P1ArtifactError("P1 summary model selection changed")
    ranking = _recompute_ranking_metrics(tables["per_state_scores"])
    ranking_columns = tuple(ranking.columns)
    recorded_ranking = tables["per_specimen_metrics"].select(ranking_columns).sort(
        ["outer_domain", "specimen_id", "method"]
    )
    _assert_table(
        recorded_ranking,
        ranking,
        name="ranking metrics",
        check_dtypes=False,
    )
    acquisition = config.raw["acquisition"]
    seeds = config.raw["random_seeds"]
    aggregate = aggregate_p1_evaluation(
        tables["acquisition_curves"],
        ranking,
        domain_order=config.domain_order,
        checkpoints=tuple(float(value) for value in acquisition["checkpoints"]),
        bootstrap_seed=int(seeds["bootstrap"]),
        bootstrap_resamples=config.bootstrap_resamples,
    )
    for name, recomputed in (
        ("per_specimen_metrics", aggregate.per_specimen_metrics),
        ("domain_metrics", aggregate.domain_metrics),
        ("bootstrap", aggregate.bootstrap),
        ("control_results", aggregate.control_results),
    ):
        _assert_table(
            _canonical_table(name, tables[name]),
            _canonical_table(name, recomputed),
            name=name,
            check_dtypes=False,
        )
    decision = aggregate.decision
    recorded_gap = summary.get("oracle_gap_closure")
    gap_matches = (recorded_gap is None and decision.oracle_gap_closure is None) or (
        recorded_gap is not None
        and decision.oracle_gap_closure is not None
        and np.isclose(
            float(recorded_gap),
            float(decision.oracle_gap_closure),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    )
    if (
        summary.get("status") != decision.status
        or summary.get("authorized_route") != decision.authorized_route
        or summary.get("go") != decision.go
        or summary.get("decision_state_sha256") != decision.state_sha256
        or summary.get("aggregate_state_sha256") != aggregate.state_sha256
        or summary.get("spatial_conditions") != dict(decision.spatial_conditions)
        or summary.get("global_conditions") != dict(decision.global_conditions)
        or summary.get("ranking_improvement") != decision.ranking_improvement
        or not gap_matches
    ):
        raise P1ArtifactError("P1 decision gate changed")
    output = (
        Path(replay_output)
        if replay_output is not None
        else root / config.output_replay
    )
    rebuilt = write_p1_package(
        output,
        config_bytes=(package / "config.yaml").read_bytes(),
        tables=tables,
        summary=summary,
        report=(package / "REPORT.md").read_text(encoding="ascii"),
    )
    for name in REQUIRED_P1_FILES:
        if (package / name).read_bytes() != (rebuilt / name).read_bytes():
            raise P1ArtifactError(f"P1 byte replay mismatch: {name}")
    return summary


__all__ = [
    "REQUIRED_P1_FILES",
    "P1ArtifactError",
    "replay_p1_package",
    "replay_p1_science",
    "write_p1_package",
]
