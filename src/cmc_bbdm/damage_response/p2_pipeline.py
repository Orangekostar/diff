from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np
import yaml

from cmc_bbdm.damage_response.artifacts import (
    ArtifactError,
    replay_p0,
    replay_p1,
    replay_p2,
    write_p2_package,
)
from cmc_bbdm.damage_response.authority import AuthorityError, snapshot_file
from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS, StageStatus
from cmc_bbdm.damage_response.feature_views import PRIMARY_TARGET_FIELDS
from cmc_bbdm.damage_response.p2_evaluation import (
    DOMAIN_ORDER,
    P2_ENDPOINTS,
    P2_VIEWS,
    REGISTERED_P2_PROTOCOL,
    evaluate_p2_nested_lodo,
)
from cmc_bbdm.damage_response.p2_features import (
    PROFILE_STAT_NAMES,
    SCALAR_DAMAGE_NAMES,
    load_p2_feature_authority,
    serialize_feature_authority_csv,
)
from cmc_bbdm.damage_response.p2_gate import (
    MINIMUM_IMPROVED_DOMAIN_COUNT,
    MINIMUM_RELATIVE_EQUAL_DOMAIN_MAE_IMPROVEMENT,
    evaluate_p2_gate,
)
from cmc_bbdm.damage_response.p2_statistics import (
    P2_BOOTSTRAP_REPLICATES,
    P2_BOOTSTRAP_SEED,
    analyze_p2_contrasts,
)
from cmc_bbdm.damage_response.pairing import load_feature_identities
from cmc_bbdm.damage_response.sources import (
    read_primary_design_metadata,
    read_specimen_sizes,
)

_EXPECTED_BASE_SHA = "3951f71f28b6efdf8c74eea0fe274b2a78a9cd57"
_EXPECTED_CONFIG_SHA256 = (
    "c04a206be7fc6847dbcb43b1eb9252733dce173901276ee4e62dcfc5f3494d92"
)
_P0_PACKAGE_RELATIVE = "results/damage_to_failure_response/p0_data_audit"
_P0_SUMMARY_NAME = "summary.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class P2PipelineError(RuntimeError):
    """Raised when P2 orchestration or deterministic publication fails."""


@dataclass(frozen=True, slots=True)
class P2SourceSpec:
    root: str
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class P2Config:
    base_sha: str
    config_sha256: str
    config_payload: bytes
    p1_package_relative_path: str
    p1_summary_sha256: str
    p1_checksums_sha256: str
    p1_descriptor_table_sha256: str
    endpoints: tuple[str, ...]
    sources: Mapping[str, P2SourceSpec]
    expected_n: int
    domain_counts: Mapping[str, int]
    ridge_alphas: tuple[float, ...]
    pca_dimensions: tuple[int, ...]
    tie_tolerance: float
    bootstrap_seed: int
    bootstrap_replicates: int
    minimum_relative_improvement: float
    minimum_improved_domains: int


@dataclass(frozen=True, slots=True)
class P2UpstreamAuthority:
    p0_package: Path
    p1_package: Path
    descriptor_table: Path
    p0_summary_sha256: str


@dataclass(frozen=True, slots=True)
class P2ExecutionResult:
    status: StageStatus
    output: Path
    decision_output: Path
    passing_contrasts: tuple[tuple[str, str, str], ...]


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise P2PipelineError(f"{label} must be a SHA-256")
    result = value.strip().casefold()
    if _SHA256_RE.fullmatch(result) is None:
        raise P2PipelineError(f"{label} must be a SHA-256")
    return result


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise P2PipelineError(f"{label} must be a relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise P2PipelineError(f"{label} must be a safe relative path")
    return path.as_posix()


def _bound_bytes(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    relative_path: str,
    max_bytes: int = 256 * 1024 * 1024,
) -> bytes:
    expected = _digest(expected_sha256, label=f"{label} expected SHA-256")
    try:
        snapshot = snapshot_file(
            path,
            max_bytes=max_bytes,
            logical_source=label,
            relative_path=relative_path,
        )
    except AuthorityError as error:
        raise P2PipelineError(str(error)) from error
    if snapshot.sha256 != expected:
        raise P2PipelineError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {snapshot.sha256}"
        )
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise P2PipelineError(f"{label} cannot be read") from error
    if (
        len(payload) != snapshot.size
        or hashlib.sha256(payload).hexdigest() != snapshot.sha256
    ):
        raise P2PipelineError(f"{label} changed during read")
    return payload


def _regular_directory(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    try:
        info = candidate.lstat()
    except OSError as error:
        raise P2PipelineError(f"explicit external root is unavailable: {label}") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise P2PipelineError(f"explicit external root must be a directory: {label}")
    return candidate


def load_p2_config(path: Path) -> P2Config:
    """Load the byte-frozen P2 protocol and expose its registered controls."""

    payload = _bound_bytes(
        Path(path),
        expected_sha256=_EXPECTED_CONFIG_SHA256,
        label="P2 config",
        relative_path="paper_v3/configs/damage_to_failure_response_p2.yaml",
        max_bytes=1024 * 1024,
    )
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise P2PipelineError("P2 config cannot be decoded") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise P2PipelineError("P2 config schema changed")
    if value.get("base_sha") != _EXPECTED_BASE_SHA:
        raise P2PipelineError("P2 config base identity changed")
    try:
        p1 = value["p1"]
        cohort = value["cohort"]
        features = value["features"]
        evaluation = value["evaluation"]
        bootstrap = value["bootstrap"]
        gate = value["gate"]
        source_values = value["sources"]
    except KeyError as error:
        raise P2PipelineError("P2 config section is missing") from error
    if (
        p1.get("required_status") != StageStatus.P1_GO.value
        or tuple(p1.get("endpoints", ())) != P2_ENDPOINTS
        or cohort.get("expected_n") != sum(PRIMARY_COUNTS.values())
        or cohort.get("domain_counts") != dict(PRIMARY_COUNTS)
        or tuple(float(item) for item in evaluation.get("ridge_alphas", ()))
        != REGISTERED_P2_PROTOCOL.ridge_alphas
        or tuple(features.get("pca_dimensions", ()))
        != REGISTERED_P2_PROTOCOL.pca_dimensions
        or float(evaluation.get("tie_tolerance", -1.0))
        != REGISTERED_P2_PROTOCOL.tie_tolerance
        or evaluation.get("outer_split") != "leave_one_domain_out"
        or evaluation.get("inner_split") != "leave_one_source_domain_out"
        or evaluation.get("aggregation") != "equal_domain_raw_mae"
        or evaluation.get("estimator") != "Ridge"
        or bootstrap.get("generator") != "PCG64"
        or bootstrap.get("seed") != P2_BOOTSTRAP_SEED
        or bootstrap.get("replicates") != P2_BOOTSTRAP_REPLICATES
        or bootstrap.get("primary_contrast_count") != 6
        or gate.get("minimum_relative_equal_domain_mae_improvement")
        != MINIMUM_RELATIVE_EQUAL_DOMAIN_MAE_IMPROVEMENT
        or gate.get("minimum_improved_domain_count")
        != MINIMUM_IMPROVED_DOMAIN_COUNT
        or gate.get("familywise_lower_bound_strictly_positive") is not True
        or gate.get("primary_reference") != "F2"
        or gate.get("primary_candidates") != ["F3", "F4"]
        or gate.get("all_fail_status") != StageStatus.MACK_EXTENSION_NO_GO.value
    ):
        raise P2PipelineError("P2 registered protocol changed")
    expected_sources = {
        "feature_bank",
        "embedding_config",
        "design_manifest",
        "feature_cache",
        "physical_descriptors",
        "provenance_specimens",
        "lvi_workbook",
        "size_workbook",
    }
    if not isinstance(source_values, dict) or set(source_values) != expected_sources:
        raise P2PipelineError("P2 source registry changed")
    sources: dict[str, P2SourceSpec] = {}
    for name in sorted(source_values):
        source = source_values[name]
        if not isinstance(source, dict) or set(source) != {
            "root",
            "relative_path",
            "sha256",
        }:
            raise P2PipelineError(f"P2 source schema changed: {name}")
        root = source["root"]
        if root not in {"repository", "legacy", "hasebe_v3"}:
            raise P2PipelineError(f"P2 source root changed: {name}")
        sources[name] = P2SourceSpec(
            root=root,
            relative_path=_relative_path(
                source["relative_path"], label=f"P2 source {name}"
            ),
            sha256=_digest(source["sha256"], label=f"P2 source {name}"),
        )
    return P2Config(
        base_sha=_EXPECTED_BASE_SHA,
        config_sha256=_EXPECTED_CONFIG_SHA256,
        config_payload=payload,
        p1_package_relative_path=_relative_path(
            p1["package_relative_path"], label="P1 package"
        ),
        p1_summary_sha256=_digest(
            p1["summary_sha256"], label="P1 summary"
        ),
        p1_checksums_sha256=_digest(
            p1["checksums_sha256"], label="P1 checksums"
        ),
        p1_descriptor_table_sha256=_digest(
            p1["descriptor_table_sha256"], label="P1 descriptor table"
        ),
        endpoints=tuple(p1["endpoints"]),
        sources=MappingProxyType(sources),
        expected_n=int(cohort["expected_n"]),
        domain_counts=MappingProxyType(dict(cohort["domain_counts"])),
        ridge_alphas=REGISTERED_P2_PROTOCOL.ridge_alphas,
        pca_dimensions=REGISTERED_P2_PROTOCOL.pca_dimensions,
        tie_tolerance=REGISTERED_P2_PROTOCOL.tie_tolerance,
        bootstrap_seed=P2_BOOTSTRAP_SEED,
        bootstrap_replicates=P2_BOOTSTRAP_REPLICATES,
        minimum_relative_improvement=(
            MINIMUM_RELATIVE_EQUAL_DOMAIN_MAE_IMPROVEMENT
        ),
        minimum_improved_domains=MINIMUM_IMPROVED_DOMAIN_COUNT,
    )


def validate_p2_upstream_authority(
    config: P2Config, repo_root: Path
) -> P2UpstreamAuthority:
    """Require exact P0/P1 packages before any external P2 source is read."""

    repository = Path(repo_root)
    p1_package = repository / config.p1_package_relative_path
    summary_payload = _bound_bytes(
        p1_package / "summary.json",
        expected_sha256=config.p1_summary_sha256,
        label="P1 summary authority",
        relative_path=f"{config.p1_package_relative_path}/summary.json",
        max_bytes=16 * 1024 * 1024,
    )
    _bound_bytes(
        p1_package / "CHECKSUMS.sha256",
        expected_sha256=config.p1_checksums_sha256,
        label="P1 checksums authority",
        relative_path=f"{config.p1_package_relative_path}/CHECKSUMS.sha256",
        max_bytes=1024 * 1024,
    )
    descriptor_table = p1_package / "descriptor_table.csv"
    _bound_bytes(
        descriptor_table,
        expected_sha256=config.p1_descriptor_table_sha256,
        label="P1 descriptor-table authority",
        relative_path=f"{config.p1_package_relative_path}/descriptor_table.csv",
        max_bytes=16 * 1024 * 1024,
    )
    try:
        replay_p1(p1_package)
    except ArtifactError as error:
        raise P2PipelineError(f"P1 package replay failed: {error}") from error
    try:
        summary = json.loads(summary_payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise P2PipelineError("P1 summary cannot be decoded") from error
    if (
        not isinstance(summary, dict)
        or summary.get("base_sha") != config.base_sha
        or not isinstance(summary.get("gate"), dict)
        or summary["gate"].get("status") != StageStatus.P1_GO.value
        or tuple(summary["gate"].get("passing_endpoints", ())) != config.endpoints
        or not isinstance(summary.get("downstream_status"), dict)
        or summary["downstream_status"].get("P2") != "AUTHORIZED_NOT_RUN"
        or summary.get("primary_cohort", {}).get("valid") != config.expected_n
        or summary.get("primary_cohort", {}).get("per_domain_valid")
        != dict(config.domain_counts)
        or summary.get("strain_status") != "STRAIN_UNIT_UNRESOLVED"
    ):
        raise P2PipelineError("P1 summary does not authorize P2")
    p0_reference = summary.get("p0_authority")
    if not isinstance(p0_reference, dict):
        raise P2PipelineError("P1 summary lacks P0 authority")
    p0_sha256 = _digest(
        p0_reference.get("summary_sha256"), label="P1-bound P0 summary"
    )
    if p0_reference.get("required_status") != StageStatus.P0_GO.value:
        raise P2PipelineError("P1-bound P0 status changed")
    p0_package = repository / _P0_PACKAGE_RELATIVE
    _bound_bytes(
        p0_package / _P0_SUMMARY_NAME,
        expected_sha256=p0_sha256,
        label="P0 summary authority",
        relative_path=f"{_P0_PACKAGE_RELATIVE}/{_P0_SUMMARY_NAME}",
        max_bytes=16 * 1024 * 1024,
    )
    try:
        replay_p0(p0_package)
    except ArtifactError as error:
        raise P2PipelineError(f"P0 package replay failed: {error}") from error
    return P2UpstreamAuthority(
        p0_package=p0_package,
        p1_package=p1_package,
        descriptor_table=descriptor_table,
        p0_summary_sha256=p0_sha256,
    )


def _csv_value(value: object) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise P2PipelineError("P2 CSV contains a nonfinite float")
        return format(value, ".17g")
    return value


def _csv_payload(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
    return stream.getvalue().encode("ascii")


def _json_payload(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("ascii")


def _source_path(
    spec: P2SourceSpec,
    *,
    repository: Path,
    legacy: Path,
    hasebe_v3: Path,
) -> Path:
    roots = {
        "repository": repository,
        "legacy": legacy,
        "hasebe_v3": hasebe_v3,
    }
    return roots[spec.root] / spec.relative_path


def _targets_from_descriptor_table(
    payload: bytes,
    *,
    specimen_ids: tuple[str, ...],
    domain_ids: tuple[str, ...],
) -> Mapping[str, np.ndarray]:
    try:
        reader = csv.DictReader(payload.decode("ascii").splitlines())
    except UnicodeDecodeError as error:
        raise P2PipelineError("P1 descriptor table is not ASCII") from error
    expected_fields = (
        "specimen_id",
        "domain_id",
        "published_cai_strength_mpa",
        "extension_peak_mm",
        "slope_u20_u60_mpa_per_mm",
        "normalized_prepeak_auc",
        "q_midpoint",
        "peak_row",
        "baseline_extension_mm",
        "baseline_stress_mpa",
        "zeroed_peak_stress_mpa",
        "unique_extension_positions",
        "extraction_sha256",
    )
    if tuple(reader.fieldnames or ()) != expected_fields:
        raise P2PipelineError("P1 descriptor-table schema changed")
    rows: dict[str, dict[str, str]] = {}
    for row in reader:
        specimen_id = str(row["specimen_id"] or "").strip().casefold()
        if not specimen_id or specimen_id in rows or set(row) != set(expected_fields):
            raise P2PipelineError("P1 descriptor-table identities changed")
        rows[specimen_id] = row
    if set(rows) != set(specimen_ids) or len(rows) != len(specimen_ids):
        raise P2PipelineError("P1 descriptor table differs from P2 cohort")
    targets: dict[str, np.ndarray] = {}
    for endpoint in PRIMARY_TARGET_FIELDS:
        values: list[float] = []
        for specimen_id, domain_id in zip(specimen_ids, domain_ids, strict=True):
            row = rows[specimen_id]
            if row["domain_id"].strip().casefold() != domain_id:
                raise P2PipelineError(f"P1 descriptor domain differs: {specimen_id}")
            try:
                value = float(row[endpoint])
            except (TypeError, ValueError) as error:
                raise P2PipelineError(
                    f"P1 descriptor target is invalid: {specimen_id}/{endpoint}"
                ) from error
            if not math.isfinite(value):
                raise P2PipelineError(
                    f"P1 descriptor target is nonfinite: {specimen_id}/{endpoint}"
                )
            values.append(value)
        array = np.asarray(values, dtype=np.float64)
        array.setflags(write=False)
        targets[endpoint] = array
    return MappingProxyType(targets)


def _implementation_source_hashes(repository: Path) -> dict[str, str]:
    paths = (
        "paper_v3/configs/damage_to_failure_response_p2.yaml",
        "src/cmc_bbdm/damage_response/p2_features.py",
        "src/cmc_bbdm/damage_response/p2_views.py",
        "src/cmc_bbdm/damage_response/p2_evaluation.py",
        "src/cmc_bbdm/damage_response/p2_statistics.py",
        "src/cmc_bbdm/damage_response/p2_gate.py",
        "src/cmc_bbdm/damage_response/p2_pipeline.py",
        "src/cmc_bbdm/damage_response/artifacts.py",
        "src/cmc_bbdm/damage_response/pipeline.py",
        "scripts/run_damage_response.py",
    )
    result: dict[str, str] = {}
    for relative in paths:
        try:
            payload = (repository / relative).read_bytes()
        except OSError as error:
            raise P2PipelineError(f"P2 implementation source is missing: {relative}") from error
        result[relative] = hashlib.sha256(payload).hexdigest()
    return result


def _build_payloads(
    *,
    config: P2Config,
    upstream: P2UpstreamAuthority,
    repository: Path,
    authority,
    feature_authority_payload: bytes,
    evaluation,
    analysis,
    gate,
) -> tuple[dict[str, bytes], bytes]:
    source_registry = {
        name: {
            "root": spec.root,
            "relative_path": spec.relative_path,
            "sha256": spec.sha256,
        }
        for name, spec in sorted(config.sources.items())
    }
    feature_provenance = {
        "authority_replay_byte_identical": True,
        "cohort": {
            "domain_counts": dict(config.domain_counts),
            "specimen_count": len(authority.specimen_ids),
        },
        "embedding": {
            "dimension": 512,
            "encoder_sha256": authority.encoder_sha256,
            "row_sha256_registry": hashlib.sha256(
                "".join(authority.full_embedding_row_sha256).encode("ascii")
            ).hexdigest(),
            "state_sha256": authority.embedding_state_sha256,
            "view": authority.full_embedding_view,
        },
        "feature_contract": {
            "damage_scalar_names": list(SCALAR_DAMAGE_NAMES),
            "full_embedding_values_serialized": False,
            "post_cai_content_used": False,
            "response_or_strength_values_serialized": False,
            "surface_profile_names": list(PROFILE_STAT_NAMES),
        },
        "implementation_sha256": _implementation_source_hashes(repository),
        "schema_version": 1,
        "sources": source_registry,
    }

    inner_fields = (
        "held_out_domain",
        "endpoint",
        "view_name",
        "ridge_alpha",
        "pca_dimension",
        *(f"inner_mae_{domain}" for domain in DOMAIN_ORDER),
        "inner_equal_domain_mae",
        "selected",
    )
    inner_rows: list[dict[str, object]] = []
    for row in evaluation.inner_scores:
        domain_mae = dict(row.inner_domain_mae)
        inner_rows.append(
            {
                "held_out_domain": row.held_out_domain,
                "endpoint": row.endpoint,
                "view_name": row.view_name,
                "ridge_alpha": row.ridge_alpha,
                "pca_dimension": row.pca_dimension,
                **{
                    f"inner_mae_{domain}": domain_mae.get(domain)
                    for domain in DOMAIN_ORDER
                },
                "inner_equal_domain_mae": row.inner_equal_domain_mae,
                "selected": row.selected,
            }
        )
    oof_rows = [
        {
            "specimen_id": row.specimen_id,
            "domain_id": row.domain_id,
            "held_out_domain": row.held_out_domain,
            "endpoint": row.endpoint,
            "view_name": row.view_name,
            "truth": row.truth,
            "prediction": row.prediction,
            "absolute_error": row.absolute_error,
            "standardized_absolute_error": row.standardized_absolute_error,
            "source_target_std": row.source_target_std,
            "selected_ridge_alpha": row.selected_ridge_alpha,
            "selected_pca_dimension": row.selected_pca_dimension,
            "preprocessor_state_sha256": row.preprocessor_state_sha256,
            "fold_state_sha256": row.fold_state_sha256,
        }
        for row in evaluation.predictions
    ]
    aggregate_rows = [
        {
            "endpoint": row.endpoint,
            "view_name": row.view_name,
            "specimen_count": row.specimen_count,
            "equal_domain_mae": row.equal_domain_mae,
            "pooled_rmse": row.pooled_rmse,
            "pooled_r2": row.pooled_r2,
            "equal_domain_standardized_mae": (
                row.equal_domain_standardized_mae
            ),
        }
        for row in evaluation.metrics
    ]
    domain_rows = [
        {
            "endpoint": row.endpoint,
            "view_name": row.view_name,
            "domain_id": row.domain_id,
            "specimen_count": row.specimen_count,
            "mae": row.mae,
            "rmse": row.rmse,
            "standardized_mae": row.standardized_mae,
        }
        for row in evaluation.domain_metrics
    ]
    bootstrap_fields = (
        "name",
        "endpoint",
        "reference_view",
        "candidate_view",
        "primary_family",
        "reference_equal_domain_mae",
        "candidate_equal_domain_mae",
        "observed_improvement",
        "relative_improvement",
        "improved_domain_count",
        *(f"improvement_{domain}" for domain in DOMAIN_ORDER),
        "bootstrap_mean",
        "ordinary_lower",
        "ordinary_upper",
        "familywise_lower",
        "familywise_upper",
        "probability_positive",
        "bootstrap_seed",
        "bootstrap_replicates",
        "bootstrap_column",
        "replicate_sha256",
        "synchronized_replicate_sha256",
    )
    bootstrap_rows: list[dict[str, object]] = []
    for row in analysis.contrasts:
        domain_improvement = dict(row.domain_improvements)
        bootstrap_rows.append(
            {
                "name": row.name,
                "endpoint": row.endpoint,
                "reference_view": row.reference_view,
                "candidate_view": row.candidate_view,
                "primary_family": row.primary_family,
                "reference_equal_domain_mae": (
                    row.observed_reference_equal_domain_mae
                ),
                "candidate_equal_domain_mae": (
                    row.observed_candidate_equal_domain_mae
                ),
                "observed_improvement": row.observed_improvement,
                "relative_improvement": row.relative_improvement,
                "improved_domain_count": row.improved_domain_count,
                **{
                    f"improvement_{domain}": domain_improvement[domain]
                    for domain in DOMAIN_ORDER
                },
                "bootstrap_mean": row.bootstrap_mean,
                "ordinary_lower": row.ordinary_interval[0],
                "ordinary_upper": row.ordinary_interval[1],
                "familywise_lower": (
                    None if row.familywise_interval is None else row.familywise_interval[0]
                ),
                "familywise_upper": (
                    None if row.familywise_interval is None else row.familywise_interval[1]
                ),
                "probability_positive": row.probability_positive,
                "bootstrap_seed": analysis.seed,
                "bootstrap_replicates": analysis.replicates,
                "bootstrap_column": row.bootstrap_column,
                "replicate_sha256": row.replicate_sha256,
                "synchronized_replicate_sha256": (
                    analysis.synchronized_replicate_sha256
                ),
            }
        )

    metric_summary: dict[str, dict[str, object]] = {}
    for endpoint in P2_ENDPOINTS:
        metric_summary[endpoint] = {}
        for view_name in P2_VIEWS:
            row = next(
                item
                for item in evaluation.metrics
                if item.endpoint == endpoint and item.view_name == view_name
            )
            metric_summary[endpoint][view_name] = {
                "equal_domain_mae": row.equal_domain_mae,
                "equal_domain_standardized_mae": (
                    row.equal_domain_standardized_mae
                ),
                "pooled_r2": row.pooled_r2,
                "pooled_rmse": row.pooled_rmse,
            }
    contrast_summary = [
        {
            "candidate_view": row.candidate_view,
            "endpoint": row.endpoint,
            "familywise_interval": (
                None
                if row.familywise_interval is None
                else list(row.familywise_interval)
            ),
            "improved_domain_count": row.improved_domain_count,
            "name": row.name,
            "ordinary_interval": list(row.ordinary_interval),
            "primary_family": row.primary_family,
            "reference_view": row.reference_view,
            "relative_improvement": row.relative_improvement,
        }
        for row in analysis.contrasts
    ]
    gate_summary = [
        {
            "candidate_view": row.candidate_view,
            "domain_count_passed": row.domain_count_passed,
            "endpoint": row.endpoint,
            "familywise_interval_passed": row.familywise_interval_passed,
            "familywise_lower_bound": row.familywise_lower_bound,
            "improved_domain_count": row.improved_domain_count,
            "passed": row.passed,
            "point_threshold_passed": row.point_threshold_passed,
            "reasons": list(row.reasons),
            "reference_view": row.reference_view,
            "relative_improvement": row.relative_improvement,
        }
        for row in gate.contrast_decisions
    ]
    downstream_status = {
        "P3": StageStatus.NOT_RUN_NOT_AUTHORIZED.value,
        "P4": StageStatus.NOT_RUN_NOT_AUTHORIZED.value,
        "P5": StageStatus.NOT_RUN_NOT_AUTHORIZED.value,
    }
    summary = {
        "artifact_replay_byte_identical": True,
        "base_sha": config.base_sha,
        "bootstrap": {
            "familywise_primary_contrast_count": 6,
            "replicates": analysis.replicates,
            "seed": analysis.seed,
            "synchronized_replicate_sha256": (
                analysis.synchronized_replicate_sha256
            ),
            "unit": "specimen_within_domain_then_equal_domain",
        },
        "cohort": {
            "domain_counts": dict(config.domain_counts),
            "specimen_count": len(authority.specimen_ids),
        },
        "config_sha256": config.config_sha256,
        "contrasts": contrast_summary,
        "downstream_status": downstream_status,
        "evaluation": {
            "inner_candidate_count": len(evaluation.inner_scores),
            "inner_ridge_fit_count": len(evaluation.inner_scores) * 5,
            "oof_prediction_count": len(evaluation.predictions),
            "outer_fold_state_count": len(evaluation.fold_states),
            "outer_ridge_fit_count": len(evaluation.fold_states),
            "pca_dimensions": list(config.pca_dimensions),
            "ridge_alphas": list(config.ridge_alphas),
            "split": "strict_nested_leave_one_domain_out",
        },
        "gate": {
            "contrast_decisions": gate_summary,
            "minimum_improved_domains": config.minimum_improved_domains,
            "minimum_relative_improvement": config.minimum_relative_improvement,
            "passing_contrasts": [list(item) for item in gate.passing_contrasts],
            "status": gate.status.value,
        },
        "input_boundary": {
            "F5_privileged_only": True,
            "post_cai_image_used": False,
            "published_or_true_cai_strength_used_as_input": False,
            "raw_cai_trace_used_as_input": False,
            "target_domain_fit_state_used": False,
            "true_impact_context_used_in_primary_views": False,
        },
        "metrics": metric_summary,
        "model_scope": {
            "neural_model_trained": False,
            "registered_estimator": "Ridge",
        },
        "paper_route_authorized": gate.status is StageStatus.P2_GO,
        "p1_authority": {
            "descriptor_table_sha256": config.p1_descriptor_table_sha256,
            "required_status": StageStatus.P1_GO.value,
            "summary_sha256": config.p1_summary_sha256,
        },
        "p0_authority": {
            "required_status": StageStatus.P0_GO.value,
            "summary_sha256": upstream.p0_summary_sha256,
        },
        "schema_version": 1,
        "strain_status": "STRAIN_UNIT_UNRESOLVED",
    }

    metric_lines = []
    for endpoint in P2_ENDPOINTS:
        for view_name in P2_VIEWS:
            metric = metric_summary[endpoint][view_name]
            metric_lines.append(
                f"- `{endpoint}` / `{view_name}`: equal-domain MAE "
                f"`{metric['equal_domain_mae']:.17g}`, RMSE "
                f"`{metric['pooled_rmse']:.17g}`, pooled R2 "
                f"`{metric['pooled_r2']:.17g}`."
            )
    contrast_lines = []
    for row in analysis.contrasts:
        familywise = (
            "secondary"
            if row.familywise_interval is None
            else (
                f"familywise CI [{row.familywise_interval[0]:.17g}, "
                f"{row.familywise_interval[1]:.17g}]"
            )
        )
        contrast_lines.append(
            f"- `{row.name}`: relative improvement "
            f"`{row.relative_improvement:.17g}`, domains "
            f"`{row.improved_domain_count}/6`, {familywise}."
        )
    gate_lines = []
    for row in gate.contrast_decisions:
        gate_lines.append(
            f"- `{row.endpoint}` / `{row.candidate_view}` vs `F2`: "
            f"`{'PASS' if row.passed else 'FAIL'}`; "
            f"{'; '.join(row.reasons) if row.reasons else 'all criteria passed'}."
        )
    report = f"""# P2 Cross-Domain Damage-to-Response Baselines

Status: `{gate.status.value}`

## Authority and scope

- P0: `{upstream.p0_summary_sha256}` / `P0_GO`
- P1: `{config.p1_summary_sha256}` / `P1_GO`
- Cohort: {len(authority.specimen_ids)}/276 across all six registered domains.
- Estimator: source-selected Ridge only; no neural model.
- F5 is privileged sensitivity only. F0-F4 contain no true impact context.
- No CAI strength, raw CAI trace, or post-CAI image is an inference input.

## Metrics

{chr(10).join(metric_lines)}

## Paired contrasts

{chr(10).join(contrast_lines)}

## Gate

{chr(10).join(gate_lines)}

- Passing primary contrasts: {list(gate.passing_contrasts)}
- P3-P5: `NOT_RUN_NOT_AUTHORIZED`
- Existing Paper 1 manuscript/evidence: unchanged.
"""
    decision_payload = f"""# P2 Damage-to-Response Decision

Status: `{gate.status.value}`

- Base SHA: `{config.base_sha}`
- P0 authority: `{upstream.p0_summary_sha256}` / `P0_GO`
- P1 authority: `{config.p1_summary_sha256}` / `P1_GO`
- Cohort: {len(authority.specimen_ids)}/276, 6/6 domains
- Primary reference: `F2`
- Primary candidates: `F3`, `F4`
- Required relative improvement: at least 10%
- Required improved domains: at least 4/6
- Required familywise lower bound: strictly above zero

{chr(10).join(gate_lines)}

- Passing contrasts: {list(gate.passing_contrasts)}
- P3-P5: `NOT_RUN_NOT_AUTHORIZED`
- New paper route: `{'AUTHORIZED' if gate.status is StageStatus.P2_GO else 'NOT_AUTHORIZED'}`
- Evidence: `results/damage_to_failure_response/p2_response_baselines/`
""".encode("ascii")

    payloads = {
        "config.yaml": config.config_payload,
        "feature_authority.csv": feature_authority_payload,
        "feature_provenance.json": _json_payload(feature_provenance),
        "inner_selection.csv": _csv_payload(inner_fields, inner_rows),
        "oof_predictions.csv": _csv_payload(
            (
                "specimen_id",
                "domain_id",
                "held_out_domain",
                "endpoint",
                "view_name",
                "truth",
                "prediction",
                "absolute_error",
                "standardized_absolute_error",
                "source_target_std",
                "selected_ridge_alpha",
                "selected_pca_dimension",
                "preprocessor_state_sha256",
                "fold_state_sha256",
            ),
            oof_rows,
        ),
        "aggregate_metrics.csv": _csv_payload(
            (
                "endpoint",
                "view_name",
                "specimen_count",
                "equal_domain_mae",
                "pooled_rmse",
                "pooled_r2",
                "equal_domain_standardized_mae",
            ),
            aggregate_rows,
        ),
        "domain_metrics.csv": _csv_payload(
            (
                "endpoint",
                "view_name",
                "domain_id",
                "specimen_count",
                "mae",
                "rmse",
                "standardized_mae",
            ),
            domain_rows,
        ),
        "bootstrap_contrasts.csv": _csv_payload(
            bootstrap_fields, bootstrap_rows
        ),
        "summary.json": _json_payload(summary),
        "REPORT.md": report.encode("ascii"),
    }
    return payloads, decision_payload


def _verify_package_determinism(parent: Path, payloads: Mapping[str, bytes]) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".p2-replay-", dir=parent) as directory:
        root = Path(directory)
        first = root / "first"
        second = root / "second"
        write_p2_package(first, payloads)
        write_p2_package(second, payloads)
        replay_p2(first)
        replay_p2(second)
        names = sorted(path.name for path in first.iterdir())
        if names != sorted(path.name for path in second.iterdir()) or any(
            (first / name).read_bytes() != (second / name).read_bytes()
            for name in names
        ):
            raise P2PipelineError("P2 package replay is not byte-identical")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise P2PipelineError(f"P2 decision cannot be created: {path}") from error
    completed = False
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
        completed = True
    except OSError as error:
        raise P2PipelineError(f"P2 decision cannot be written: {path}") from error
    finally:
        os.close(descriptor)
        if not completed:
            try:
                path.unlink()
            except OSError:
                pass


def run_p2_audit(
    *,
    config_path: Path,
    repo_root: Path,
    legacy_root: Path,
    hasebe_v3_root: Path,
    output: Path,
    decision_output: Path,
) -> P2ExecutionResult:
    """Execute and atomically publish the preregistered P2 audit."""

    destination = Path(output)
    decision_path = Path(decision_output)
    for label, path in (("P2 output", destination), ("P2 decision", decision_path)):
        if path.exists() or path.is_symlink():
            raise P2PipelineError(f"{label} already exists: {path}")
    config = load_p2_config(config_path)
    repository = _regular_directory(repo_root, label="repository")
    upstream = validate_p2_upstream_authority(config, repository)
    legacy = _regular_directory(legacy_root, label="legacy_root")
    hasebe_v3 = _regular_directory(hasebe_v3_root, label="hasebe_v3_root")
    roots = {
        "repository": repository,
        "legacy": legacy,
        "hasebe_v3": hasebe_v3,
    }
    for name, spec in config.sources.items():
        _bound_bytes(
            roots[spec.root] / spec.relative_path,
            expected_sha256=spec.sha256,
            label=f"P2 source {name}",
            relative_path=spec.relative_path,
        )

    bank = config.sources["feature_bank"]
    feature_identities = load_feature_identities(
        _source_path(
            bank, repository=repository, legacy=legacy, hasebe_v3=hasebe_v3
        ),
        expected_sha256=bank.sha256,
    )
    size_spec = config.sources["size_workbook"]
    sizes = read_specimen_sizes(
        _source_path(
            size_spec,
            repository=repository,
            legacy=legacy,
            hasebe_v3=hasebe_v3,
        ),
        expected_sha256=size_spec.sha256,
    )
    design_spec = config.sources["design_manifest"]
    roster = read_primary_design_metadata(
        _source_path(
            design_spec,
            repository=repository,
            legacy=legacy,
            hasebe_v3=hasebe_v3,
        ),
        design_spec.sha256,
        feature_identities,
        sizes,
    )

    def load_authority():
        cache = config.sources["feature_cache"]
        descriptors = config.sources["physical_descriptors"]
        provenance = config.sources["provenance_specimens"]
        lvi = config.sources["lvi_workbook"]
        return load_p2_feature_authority(
            roster=roster,
            feature_bank_path=_source_path(
                bank,
                repository=repository,
                legacy=legacy,
                hasebe_v3=hasebe_v3,
            ),
            feature_bank_sha256=bank.sha256,
            feature_cache_path=_source_path(
                cache,
                repository=repository,
                legacy=legacy,
                hasebe_v3=hasebe_v3,
            ),
            feature_cache_sha256=cache.sha256,
            physical_descriptors_path=_source_path(
                descriptors,
                repository=repository,
                legacy=legacy,
                hasebe_v3=hasebe_v3,
            ),
            physical_descriptors_sha256=descriptors.sha256,
            provenance_path=_source_path(
                provenance,
                repository=repository,
                legacy=legacy,
                hasebe_v3=hasebe_v3,
            ),
            provenance_sha256=provenance.sha256,
            lvi_workbook_path=_source_path(
                lvi,
                repository=repository,
                legacy=legacy,
                hasebe_v3=hasebe_v3,
            ),
            lvi_workbook_sha256=lvi.sha256,
        )

    authority = load_authority()
    repeat_authority = load_authority()
    feature_authority_payload = serialize_feature_authority_csv(authority)
    if feature_authority_payload != serialize_feature_authority_csv(repeat_authority):
        raise P2PipelineError("P2 feature authority replay is not byte-identical")
    counts = Counter(authority.domain_ids)
    if (
        len(authority.specimen_ids) != config.expected_n
        or dict(counts) != dict(config.domain_counts)
    ):
        raise P2PipelineError("P2 feature authority cohort changed")
    descriptor_payload = _bound_bytes(
        upstream.descriptor_table,
        expected_sha256=config.p1_descriptor_table_sha256,
        label="P1 descriptor-table authority",
        relative_path=(
            f"{config.p1_package_relative_path}/descriptor_table.csv"
        ),
        max_bytes=16 * 1024 * 1024,
    )
    targets = _targets_from_descriptor_table(
        descriptor_payload,
        specimen_ids=authority.specimen_ids,
        domain_ids=authority.domain_ids,
    )
    evaluation = evaluate_p2_nested_lodo(
        authority, targets, protocol=REGISTERED_P2_PROTOCOL
    )
    analysis = analyze_p2_contrasts(
        evaluation.predictions,
        seed=config.bootstrap_seed,
        replicates=config.bootstrap_replicates,
    )
    gate = evaluate_p2_gate(analysis)
    payloads, decision_payload = _build_payloads(
        config=config,
        upstream=upstream,
        repository=repository,
        authority=authority,
        feature_authority_payload=feature_authority_payload,
        evaluation=evaluation,
        analysis=analysis,
        gate=gate,
    )
    try:
        _verify_package_determinism(destination.parent, payloads)
        write_p2_package(destination, payloads)
        replay_p2(destination)
    except ArtifactError as error:
        raise P2PipelineError(str(error)) from error
    _write_exclusive(decision_path, decision_payload)
    return P2ExecutionResult(
        status=gate.status,
        output=destination,
        decision_output=decision_path,
        passing_contrasts=gate.passing_contrasts,
    )
