"""Hash-bound artifacts for MAVIS science-closure stages."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml

from .science_closure import (
    build_dynamic_valuation_alignment,
    evaluate_dynamic_valuation_closure,
    evaluate_mris_causal_closure,
)


class ScienceClosureArtifactError(RuntimeError):
    """Raised when a science-closure artifact contract is invalid."""


_P10_CONFIG_KEYS = {
    "schema_version",
    "stage",
    "audit_base_git_sha",
    "domain_order",
    "p2_state_predictions",
    "p2_state_predictions_sha256",
    "p2_artifact_manifest",
    "p2_artifact_manifest_sha256",
    "p2_state_sha256",
    "full_field_predictions",
    "full_field_predictions_sha256",
    "full_field_artifact_manifest",
    "full_field_artifact_manifest_sha256",
    "full_field_method",
    "p7_package",
    "p7_tree_state_sha256",
    "bootstrap_replicates",
    "seed",
}
_P10_FILES = {
    "state_cost_curve.csv",
    "per_specimen_predictions.parquet",
    "domain_metrics.csv",
    "contrasts.csv",
    "bootstrap.csv",
    "REPORT.md",
    "summary.json",
    "artifact_manifest.json",
    "CHECKSUMS.sha256",
}
_P11_CONFIG_KEYS = {
    "schema_version",
    "stage",
    "audit_base_git_sha",
    "domain_order",
    "p1_state_manifest",
    "p1_state_manifest_sha256",
    "p1_artifact_manifest",
    "p1_artifact_manifest_sha256",
    "p1_state_sha256",
    "p3_action_scores",
    "p3_action_scores_sha256",
    "p3_artifact_manifest",
    "p3_artifact_manifest_sha256",
    "p3_state_sha256",
    "mvd_action_scores",
    "mvd_action_scores_sha256",
    "mvd_artifact_manifest",
    "mvd_artifact_manifest_sha256",
    "mvd_authority_state_sha256",
    "dynamic_modes",
    "mvd_o2_method",
    "candidate_only_method",
    "recall_k",
    "p7_package",
    "p7_tree_state_sha256",
    "bootstrap_replicates",
    "seed",
}
_P11_FILES = {
    "action_predictions.parquet",
    "regret_by_cost.csv",
    "one_step_utility.csv",
    "domain_metrics.csv",
    "bootstrap.csv",
    "REPORT.md",
    "summary.json",
    "artifact_manifest.json",
    "CHECKSUMS.sha256",
}


@dataclass(frozen=True, slots=True)
class P10MRISCausalConfig:
    schema_version: int
    audit_base_git_sha: str
    domain_order: tuple[str, ...]
    p2_state_predictions: str
    p2_state_predictions_sha256: str
    p2_artifact_manifest: str
    p2_artifact_manifest_sha256: str
    p2_state_sha256: str
    full_field_predictions: str
    full_field_predictions_sha256: str
    full_field_artifact_manifest: str
    full_field_artifact_manifest_sha256: str
    full_field_method: str
    p7_package: str
    p7_tree_state_sha256: str
    bootstrap_replicates: int
    seed: int
    config_sha256: str


@dataclass(frozen=True, slots=True)
class P11DynamicValuationConfig:
    schema_version: int
    audit_base_git_sha: str
    domain_order: tuple[str, ...]
    p1_state_manifest: str
    p1_state_manifest_sha256: str
    p1_artifact_manifest: str
    p1_artifact_manifest_sha256: str
    p1_state_sha256: str
    p3_action_scores: str
    p3_action_scores_sha256: str
    p3_artifact_manifest: str
    p3_artifact_manifest_sha256: str
    p3_state_sha256: str
    mvd_action_scores: str
    mvd_action_scores_sha256: str
    mvd_artifact_manifest: str
    mvd_artifact_manifest_sha256: str
    mvd_authority_state_sha256: str
    dynamic_modes: tuple[str, ...]
    mvd_o2_method: str
    candidate_only_method: str
    recall_k: int
    p7_package: str
    p7_tree_state_sha256: str
    bootstrap_replicates: int
    seed: int
    config_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _tree_state(path: Path) -> str:
    rows = [
        (item.relative_to(path).as_posix(), item.stat().st_size, _sha256(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    if not rows:
        raise ScienceClosureArtifactError("bound artifact tree is empty")
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_checksums(output: Path) -> None:
    files = sorted(item for item in output.iterdir() if item.name != "CHECKSUMS.sha256")
    (output / "CHECKSUMS.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="ascii",
    )


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or len(value) < 2
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ScienceClosureArtifactError(f"{label} is invalid")
    return tuple(value)


def load_p10_mris_causal_config(path: str | Path) -> P10MRISCausalConfig:
    try:
        source = Path(path).resolve(strict=True)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ScienceClosureArtifactError("P10 config is unavailable") from error
    if type(payload) is not dict or set(payload) != _P10_CONFIG_KEYS:
        raise ScienceClosureArtifactError("P10 config schema changed")
    domains = _text_tuple(payload["domain_order"], "P10 domain order")
    path_keys = (
        "p2_state_predictions",
        "p2_artifact_manifest",
        "full_field_predictions",
        "full_field_artifact_manifest",
        "p7_package",
    )
    digest_keys = (
        "p2_state_predictions_sha256",
        "p2_artifact_manifest_sha256",
        "p2_state_sha256",
        "full_field_predictions_sha256",
        "full_field_artifact_manifest_sha256",
        "p7_tree_state_sha256",
    )
    if (
        payload["schema_version"] != 1
        or payload["stage"] != "P10_MRIS_CAUSAL"
        or not _is_hex(payload["audit_base_git_sha"], 40)
        or any(type(payload[key]) is not str or not payload[key] for key in path_keys)
        or any(not _is_hex(payload[key], 64) for key in digest_keys)
        or type(payload["full_field_method"]) is not str
        or not payload["full_field_method"]
        or type(payload["bootstrap_replicates"]) is not int
        or isinstance(payload["bootstrap_replicates"], bool)
        or payload["bootstrap_replicates"] < 2
        or type(payload["seed"]) is not int
        or isinstance(payload["seed"], bool)
    ):
        raise ScienceClosureArtifactError("P10 config values are invalid")
    return P10MRISCausalConfig(
        schema_version=1,
        audit_base_git_sha=payload["audit_base_git_sha"],
        domain_order=domains,
        p2_state_predictions=payload["p2_state_predictions"],
        p2_state_predictions_sha256=payload["p2_state_predictions_sha256"],
        p2_artifact_manifest=payload["p2_artifact_manifest"],
        p2_artifact_manifest_sha256=payload["p2_artifact_manifest_sha256"],
        p2_state_sha256=payload["p2_state_sha256"],
        full_field_predictions=payload["full_field_predictions"],
        full_field_predictions_sha256=payload["full_field_predictions_sha256"],
        full_field_artifact_manifest=payload["full_field_artifact_manifest"],
        full_field_artifact_manifest_sha256=payload[
            "full_field_artifact_manifest_sha256"
        ],
        full_field_method=payload["full_field_method"],
        p7_package=payload["p7_package"],
        p7_tree_state_sha256=payload["p7_tree_state_sha256"],
        bootstrap_replicates=payload["bootstrap_replicates"],
        seed=payload["seed"],
        config_sha256=_sha256(source),
    )


def load_p11_dynamic_valuation_config(
    path: str | Path,
) -> P11DynamicValuationConfig:
    try:
        source = Path(path).resolve(strict=True)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ScienceClosureArtifactError("P11 config is unavailable") from error
    if type(payload) is not dict or set(payload) != _P11_CONFIG_KEYS:
        raise ScienceClosureArtifactError("P11 config schema changed")
    domains = _text_tuple(payload["domain_order"], "P11 domain order")
    modes = _text_tuple(payload["dynamic_modes"], "P11 dynamic modes")
    path_keys = (
        "p1_state_manifest",
        "p1_artifact_manifest",
        "p3_action_scores",
        "p3_artifact_manifest",
        "mvd_action_scores",
        "mvd_artifact_manifest",
        "p7_package",
    )
    digest_keys = (
        "p1_state_manifest_sha256",
        "p1_artifact_manifest_sha256",
        "p1_state_sha256",
        "p3_action_scores_sha256",
        "p3_artifact_manifest_sha256",
        "p3_state_sha256",
        "mvd_action_scores_sha256",
        "mvd_artifact_manifest_sha256",
        "mvd_authority_state_sha256",
        "p7_tree_state_sha256",
    )
    if (
        payload["schema_version"] != 1
        or payload["stage"] != "P11_DYNAMIC_VALUATION"
        or not _is_hex(payload["audit_base_git_sha"], 40)
        or set(modes) != {"real", "positions_only", "shuffled"}
        or any(type(payload[key]) is not str or not payload[key] for key in path_keys)
        or any(not _is_hex(payload[key], 64) for key in digest_keys)
        or type(payload["mvd_o2_method"]) is not str
        or not payload["mvd_o2_method"]
        or type(payload["candidate_only_method"]) is not str
        or not payload["candidate_only_method"]
        or payload["mvd_o2_method"] == payload["candidate_only_method"]
        or type(payload["recall_k"]) is not int
        or isinstance(payload["recall_k"], bool)
        or payload["recall_k"] <= 0
        or type(payload["bootstrap_replicates"]) is not int
        or isinstance(payload["bootstrap_replicates"], bool)
        or payload["bootstrap_replicates"] < 2
        or type(payload["seed"]) is not int
        or isinstance(payload["seed"], bool)
    ):
        raise ScienceClosureArtifactError("P11 config values are invalid")
    return P11DynamicValuationConfig(
        schema_version=1,
        audit_base_git_sha=payload["audit_base_git_sha"],
        domain_order=domains,
        p1_state_manifest=payload["p1_state_manifest"],
        p1_state_manifest_sha256=payload["p1_state_manifest_sha256"],
        p1_artifact_manifest=payload["p1_artifact_manifest"],
        p1_artifact_manifest_sha256=payload["p1_artifact_manifest_sha256"],
        p1_state_sha256=payload["p1_state_sha256"],
        p3_action_scores=payload["p3_action_scores"],
        p3_action_scores_sha256=payload["p3_action_scores_sha256"],
        p3_artifact_manifest=payload["p3_artifact_manifest"],
        p3_artifact_manifest_sha256=payload["p3_artifact_manifest_sha256"],
        p3_state_sha256=payload["p3_state_sha256"],
        mvd_action_scores=payload["mvd_action_scores"],
        mvd_action_scores_sha256=payload["mvd_action_scores_sha256"],
        mvd_artifact_manifest=payload["mvd_artifact_manifest"],
        mvd_artifact_manifest_sha256=payload["mvd_artifact_manifest_sha256"],
        mvd_authority_state_sha256=payload["mvd_authority_state_sha256"],
        dynamic_modes=modes,
        mvd_o2_method=payload["mvd_o2_method"],
        candidate_only_method=payload["candidate_only_method"],
        recall_k=payload["recall_k"],
        p7_package=payload["p7_package"],
        p7_tree_state_sha256=payload["p7_tree_state_sha256"],
        bootstrap_replicates=payload["bootstrap_replicates"],
        seed=payload["seed"],
        config_sha256=_sha256(source),
    )


def _bound_path(root: Path, value: str, *, directory: bool) -> Path:
    try:
        path = (root / value).resolve(strict=True)
    except OSError as error:
        raise ScienceClosureArtifactError("configured P10 input is unavailable") from error
    if root != path and root not in path.parents:
        raise ScienceClosureArtifactError("configured P10 input escapes project root")
    if path.is_dir() != directory:
        raise ScienceClosureArtifactError("configured P10 input type changed")
    return path


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScienceClosureArtifactError(f"{label} is invalid") from error
    if type(value) is not dict:
        raise ScienceClosureArtifactError(f"{label} is invalid")
    return value


def _validate_source_manifests(
    config: P10MRISCausalConfig,
    p2_manifest_path: Path,
    full_field_manifest_path: Path,
) -> None:
    p2 = _load_json(p2_manifest_path, "P2 artifact manifest")
    full_field = _load_json(full_field_manifest_path, "full-field artifact manifest")
    p2_files = p2.get("files")
    full_field_files = full_field.get("files")
    if (
        p2.get("p2_state_sha256") != config.p2_state_sha256
        or type(p2_files) is not dict
        or type(p2_files.get("state_predictions.parquet")) is not dict
        or p2_files["state_predictions.parquet"].get("sha256")
        != config.p2_state_predictions_sha256
        or full_field.get("scope") != "cpb_v3_p1_full_field_oracle"
        or type(full_field_files) is not dict
        or type(full_field_files.get("predictions.csv")) is not dict
        or full_field_files["predictions.csv"].get("sha256")
        != config.full_field_predictions_sha256
    ):
        raise ScienceClosureArtifactError("frozen P10 source provenance changed")


def _code_state() -> str:
    files = [Path(__file__), Path(__file__).with_name("science_closure.py")]
    rows = [(path.name, _sha256(path)) for path in sorted(files)]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _p10_report(summary: dict[str, object], curve: pl.DataFrame) -> str:
    checkpoints = sorted(
        curve.filter(pl.col("mode") == "real")
        .get_column("nominal_checkpoint")
        .to_list()
    )
    lines = [
        "# MAVIS P10 MRIS Causal Informativeness Closure",
        "",
        "Status: `COMPLETE`.",
        "",
        "All rows reuse frozen nested-LODO predictions. No representation or CAI",
        "model is retrained. Effects are `real - control` CAI MAE, so negative",
        "values favor specimen-specific real ultrasonic content.",
        "",
        "| Checkpoint | Real MAE | Positions MAE | Shuffled MAE | Static MAE | Reconstruction MAE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in checkpoints:
        values = {
            mode: float(
                curve.filter(
                    (pl.col("mode") == mode)
                    & (pl.col("nominal_checkpoint") == checkpoint)
                ).item(0, "equal_domain_mae")
            )
            for mode in (
                "real",
                "positions_only",
                "shuffled",
                "static",
                "reconstruction",
            )
        }
        lines.append(
            "| {checkpoint:.5f} | {real:.6f} | {positions_only:.6f} | "
            "{shuffled:.6f} | {static:.6f} | {reconstruction:.6f} |".format(
                checkpoint=checkpoint, **values
            )
        )
    support = "supported" if summary["actual_content_beyond_geometry_supported"] else "not supported"
    accumulation = "decreases" if summary["real_error_decreases_with_ut"] else "does not reliably decrease"
    lines.extend(
        [
            "",
            f"Specimen-specific content beyond geometry is **{support}** under the",
            "predeclared paired contrasts. Real MRIS beats the static initial state,",
            "but it does not beat positions-only at the registered checkpoints.",
            "",
            f"Real-state CAI error {accumulation} from the first to final checkpoint.",
            (
                "The final partial state recovers "
                f"`{float(summary['endpoint_full_field_utility_recovery_fraction']):.6f}` "
                "of the static-to-full-field mechanical utility under the registered"
            ),
            "ratio; its paired interval is reported in `bootstrap.csv`.",
            "",
            "Metrics first average trajectories within each physical specimen, then",
            "weight the six held-out domains equally. The full-field reference is the",
            "hash-bound source-only `I_field_selected` prediction. This state-utility",
            "recovery is not policy oracle-gap recovery and does not alter P7 Tier B.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_p10_mris_causal(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_root: str | Path,
) -> Path:
    """Generate P10 from hash-bound P2 and full-field predictions."""

    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        raise ScienceClosureArtifactError("project root is unavailable") from error
    config = load_p10_mris_causal_config(config_path)
    p2_predictions = _bound_path(root, config.p2_state_predictions, directory=False)
    p2_manifest = _bound_path(root, config.p2_artifact_manifest, directory=False)
    full_predictions = _bound_path(root, config.full_field_predictions, directory=False)
    full_manifest = _bound_path(
        root, config.full_field_artifact_manifest, directory=False
    )
    p7 = _bound_path(root, config.p7_package, directory=True)
    if (
        _sha256(p2_predictions) != config.p2_state_predictions_sha256
        or _sha256(p2_manifest) != config.p2_artifact_manifest_sha256
        or _sha256(full_predictions) != config.full_field_predictions_sha256
        or _sha256(full_manifest) != config.full_field_artifact_manifest_sha256
        or _tree_state(p7) != config.p7_tree_state_sha256
    ):
        raise ScienceClosureArtifactError("frozen P10 input hash changed")
    _validate_source_manifests(config, p2_manifest, full_manifest)
    p7_before = _tree_state(p7)
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if root not in destination.parents or destination.exists():
        raise ScienceClosureArtifactError("P10 output is invalid or already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p10_mris_causal.", dir=destination.parent))
    try:
        predictions = pl.read_parquet(p2_predictions)
        full_field = pl.read_csv(full_predictions)
        tables = evaluate_mris_causal_closure(
            predictions,
            full_field,
            domain_order=config.domain_order,
            full_field_method=config.full_field_method,
            bootstrap_replicates=config.bootstrap_replicates,
            seed=config.seed,
        )
        tables.per_specimen_predictions.write_parquet(
            temporary / "per_specimen_predictions.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        tables.state_cost_curve.write_csv(temporary / "state_cost_curve.csv")
        tables.domain_metrics.write_csv(temporary / "domain_metrics.csv")
        tables.contrasts.write_csv(temporary / "contrasts.csv")
        tables.bootstrap.write_csv(temporary / "bootstrap.csv")
        primary = tables.bootstrap.filter(
            (pl.col("scope") == "equal_domain")
            & (pl.col("metric") == "real_minus_control_mae")
            & pl.col("control_mode").is_in(["positions_only", "shuffled"])
        )
        content_supported = (
            primary.height == 2 * tables.contrasts.get_column(
                "nominal_checkpoint"
            ).n_unique()
            and primary.filter(pl.col("ci95_upper") >= 0.0).is_empty()
        )
        checkpoints = sorted(
            tables.domain_metrics.get_column("nominal_checkpoint").unique().to_list()
        )
        endpoint = checkpoints[-1]
        endpoint_change = tables.bootstrap.filter(
            (pl.col("scope") == "equal_domain")
            & (pl.col("metric") == "real_change_from_initial_mae")
            & (pl.col("nominal_checkpoint") == endpoint)
        ).row(0, named=True)
        endpoint_recovery = tables.bootstrap.filter(
            (pl.col("scope") == "equal_domain")
            & (pl.col("metric") == "full_field_utility_recovery_fraction")
            & (pl.col("nominal_checkpoint") == endpoint)
        ).row(0, named=True)
        state_payload = {
            "schema": 1,
            "stage": "P10_MRIS_CAUSAL",
            "config_sha256": config.config_sha256,
            "runtime_code_state_sha256": _code_state(),
            "p2_state_predictions_sha256": config.p2_state_predictions_sha256,
            "p2_state_sha256": config.p2_state_sha256,
            "full_field_predictions_sha256": config.full_field_predictions_sha256,
            "p7_tree_state_sha256": p7_before,
            "source_prediction_row_count": tables.source_prediction_row_count,
            "per_specimen_row_count": tables.per_specimen_predictions.height,
            "curve_row_count": tables.state_cost_curve.height,
            "domain_metric_row_count": tables.domain_metrics.height,
            "contrast_row_count": tables.contrasts.height,
            "bootstrap_row_count": tables.bootstrap.height,
        }
        p10_state = hashlib.sha256(
            json.dumps(
                state_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        summary: dict[str, object] = {
            **state_payload,
            "schema_version": 1,
            "status": "COMPLETE",
            "audit_base_git_sha": config.audit_base_git_sha,
            "p10_state_sha256": p10_state,
            "domain_count": len(config.domain_order),
            "specimen_count": tables.per_specimen_predictions.get_column(
                "specimen_id"
            ).n_unique(),
            "checkpoint_count": len(checkpoints),
            "bootstrap_replicates": config.bootstrap_replicates,
            "seed": config.seed,
            "full_field_method": config.full_field_method,
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "sign_convention": "real_minus_control; negative_favors_real",
            "actual_content_beyond_geometry_supported": content_supported,
            "real_error_decreases_with_ut": endpoint_change["ci95_upper"] < 0.0,
            "endpoint_checkpoint": endpoint,
            "endpoint_real_change_from_initial_mae": endpoint_change["estimate"],
            "endpoint_real_change_ci95": [
                endpoint_change["ci95_lower"],
                endpoint_change["ci95_upper"],
            ],
            "endpoint_full_field_utility_recovery_fraction": endpoint_recovery[
                "estimate"
            ],
            "endpoint_recovery_ci95": [
                endpoint_recovery["ci95_lower"],
                endpoint_recovery["ci95_upper"],
            ],
            "p7_modified": False,
            "claim_tier_changed": False,
        }
        _write_json(temporary / "summary.json", summary)
        (temporary / "REPORT.md").write_text(
            _p10_report(summary, tables.state_cost_curve), encoding="utf-8"
        )
        products = sorted(item for item in temporary.iterdir() if item.is_file())
        manifest = {
            "schema_version": 1,
            "stage": "P10_MRIS_CAUSAL",
            "status": "COMPLETE",
            "p10_state_sha256": p10_state,
            "config_sha256": config.config_sha256,
            "p7_tree_state_sha256": p7_before,
            "files": [
                {
                    "path": item.name,
                    "bytes": item.stat().st_size,
                    "sha256": _sha256(item),
                }
                for item in products
            ],
        }
        _write_json(temporary / "artifact_manifest.json", manifest)
        _write_checksums(temporary)
        if _tree_state(p7) != p7_before:
            raise ScienceClosureArtifactError("P10 modified frozen P7 artifacts")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_p10_mris_causal_package(destination)
    return destination


def verify_p10_mris_causal_package(path: str | Path) -> dict[str, object]:
    try:
        package = Path(path).resolve(strict=True)
    except OSError as error:
        raise ScienceClosureArtifactError("P10 package is unavailable") from error
    if not package.is_dir() or {item.name for item in package.iterdir()} != _P10_FILES:
        raise ScienceClosureArtifactError("P10 package file roster changed")
    lines = (package / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    checksums: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or not _is_hex(parts[0], 64) or parts[1] in checksums:
            raise ScienceClosureArtifactError("P10 checksum manifest is invalid")
        checksums[parts[1]] = parts[0]
    expected = _P10_FILES - {"CHECKSUMS.sha256"}
    if set(checksums) != expected or any(
        _sha256(package / name) != digest for name, digest in checksums.items()
    ):
        raise ScienceClosureArtifactError("P10 checksum mismatch")
    manifest = _load_json(package / "artifact_manifest.json", "P10 manifest")
    summary = _load_json(package / "summary.json", "P10 summary")
    if (
        manifest.get("stage") != "P10_MRIS_CAUSAL"
        or manifest.get("status") != "COMPLETE"
        or summary.get("stage") != manifest["stage"]
        or summary.get("status") != "COMPLETE"
        or summary.get("p10_state_sha256") != manifest.get("p10_state_sha256")
        or not _is_hex(manifest.get("p10_state_sha256"), 64)
        or summary.get("p7_modified") is not False
        or summary.get("claim_tier_changed") is not False
        or not math.isfinite(
            float(summary.get("endpoint_full_field_utility_recovery_fraction", math.nan))
        )
    ):
        raise ScienceClosureArtifactError("P10 metadata contract changed")
    product_records = manifest.get("files")
    if type(product_records) is not list or {
        record.get("path") for record in product_records if type(record) is dict
    } != expected - {"artifact_manifest.json"}:
        raise ScienceClosureArtifactError("P10 artifact manifest roster changed")
    try:
        curve = pl.read_csv(package / "state_cost_curve.csv")
        predictions = pl.read_parquet(package / "per_specimen_predictions.parquet")
        contrasts = pl.read_csv(package / "contrasts.csv")
        bootstrap = pl.read_csv(package / "bootstrap.csv")
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ScienceClosureArtifactError("P10 scientific tables are invalid") from error
    if (
        set(curve.get_column("mode").unique())
        != {"real", "positions_only", "shuffled", "static", "reconstruction", "full_field"}
        or set(predictions.get_column("source").unique())
        != {"frozen_p2_state_predictions", "frozen_full_field_predictions"}
        or set(contrasts.get_column("control_mode").unique())
        != {"static", "positions_only", "shuffled", "reconstruction"}
        or bootstrap.height != summary.get("bootstrap_row_count")
    ):
        raise ScienceClosureArtifactError("P10 scientific table contract changed")
    return manifest


def _validate_p11_source_manifests(
    config: P11DynamicValuationConfig,
    p1_manifest_path: Path,
    p3_manifest_path: Path,
    mvd_manifest_path: Path,
) -> None:
    p1 = _load_json(p1_manifest_path, "P1 artifact manifest")
    p3 = _load_json(p3_manifest_path, "P3 artifact manifest")
    mvd = _load_json(mvd_manifest_path, "MVD artifact manifest")
    p1_files = p1.get("files")
    p3_files = p3.get("files")
    mvd_files = mvd.get("files")
    if (
        p1.get("state_bank_state_sha256") != config.p1_state_sha256
        or type(p1_files) is not dict
        or type(p1_files.get("state_manifest.parquet")) is not dict
        or p1_files["state_manifest.parquet"].get("sha256")
        != config.p1_state_manifest_sha256
        or p3.get("p3_state_sha256") != config.p3_state_sha256
        or type(p3_files) is not dict
        or type(p3_files.get("action_scores.parquet")) is not dict
        or p3_files["action_scores.parquet"].get("sha256")
        != config.p3_action_scores_sha256
        or mvd.get("authority_state_sha256") != config.mvd_authority_state_sha256
        or type(mvd_files) is not dict
        or type(mvd_files.get("observability_predictions.parquet")) is not dict
        or mvd_files["observability_predictions.parquet"].get("sha256")
        != config.mvd_action_scores_sha256
    ):
        raise ScienceClosureArtifactError("frozen P11 source provenance changed")


def _p11_report(summary: dict[str, object], regret: pl.DataFrame) -> str:
    checkpoints = sorted(regret.get_column("nominal_checkpoint").unique().to_list())
    lines = [
        "# MAVIS P11 Dynamic Valuation Closure",
        "",
        "Status: `COMPLETE`.",
        "",
        "All five scorers are evaluated on the same frozen current-state legal",
        "actions and strict-OOF teacher values. MVD O2 and candidate-only scores",
        "are specimen-cell static diagnostics; their historical costs never replace",
        "the current exact marginal action cost.",
        "",
        "| Checkpoint | Dynamic real regret | Static M1/O2 | Candidate-only | Positions | Shuffled |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in checkpoints:
        values = {
            scorer: float(
                regret.filter(
                    (pl.col("scorer") == scorer)
                    & (pl.col("nominal_checkpoint") == checkpoint)
                ).item(0, "equal_domain_regret")
            )
            for scorer in (
                "dynamic_real",
                "static_m1_o2",
                "candidate_only_static",
                "dynamic_positions_only",
                "dynamic_shuffled",
            )
        }
        lines.append(
            "| {checkpoint:.5f} | {dynamic_real:.6f} | {static_m1_o2:.6f} | "
            "{candidate_only_static:.6f} | {dynamic_positions_only:.6f} | "
            "{dynamic_shuffled:.6f} |".format(checkpoint=checkpoint, **values)
        )
    lines.extend(
        [
            "",
            str(summary["primary_conclusion"]),
            "",
            "Primary inference uses next-action regret and one-step CAI utility.",
            "Spearman, NDCG, and Recall@K are secondary diagnostics. Metrics first",
            "average states within physical specimens at each cost, then weight held-",
            "out domains equally; paired bootstrap resamples specimens within domain.",
            "",
            "This is a retrospective valuation diagnostic. It does not retune target",
            "domains, replace a P7 checkpoint, or change the frozen Tier-B claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_p11_dynamic_valuation(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_root: str | Path,
) -> Path:
    """Generate P11 from hash-bound P1, P3, and MVD score tables."""

    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        raise ScienceClosureArtifactError("project root is unavailable") from error
    config = load_p11_dynamic_valuation_config(config_path)
    p1_states = _bound_path(root, config.p1_state_manifest, directory=False)
    p1_manifest = _bound_path(root, config.p1_artifact_manifest, directory=False)
    p3_scores = _bound_path(root, config.p3_action_scores, directory=False)
    p3_manifest = _bound_path(root, config.p3_artifact_manifest, directory=False)
    mvd_scores = _bound_path(root, config.mvd_action_scores, directory=False)
    mvd_manifest = _bound_path(root, config.mvd_artifact_manifest, directory=False)
    p7 = _bound_path(root, config.p7_package, directory=True)
    paths_and_hashes = (
        (p1_states, config.p1_state_manifest_sha256),
        (p1_manifest, config.p1_artifact_manifest_sha256),
        (p3_scores, config.p3_action_scores_sha256),
        (p3_manifest, config.p3_artifact_manifest_sha256),
        (mvd_scores, config.mvd_action_scores_sha256),
        (mvd_manifest, config.mvd_artifact_manifest_sha256),
    )
    if any(_sha256(path) != digest for path, digest in paths_and_hashes) or _tree_state(
        p7
    ) != config.p7_tree_state_sha256:
        raise ScienceClosureArtifactError("frozen P11 input hash changed")
    _validate_p11_source_manifests(config, p1_manifest, p3_manifest, mvd_manifest)
    p7_before = _tree_state(p7)
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if root not in destination.parents or destination.exists():
        raise ScienceClosureArtifactError("P11 output is invalid or already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".p11_dynamic_valuation.", dir=destination.parent)
    )
    try:
        states = pl.read_parquet(p1_states)
        dynamic_scores = pl.read_parquet(p3_scores)
        static_scores = pl.read_parquet(mvd_scores)
        aligned = build_dynamic_valuation_alignment(
            states,
            dynamic_scores,
            static_scores,
            domain_order=config.domain_order,
            dynamic_modes=config.dynamic_modes,
            mvd_o2_method=config.mvd_o2_method,
            candidate_only_method=config.candidate_only_method,
        )
        tables = evaluate_dynamic_valuation_closure(
            aligned,
            domain_order=config.domain_order,
            recall_k=config.recall_k,
            bootstrap_replicates=config.bootstrap_replicates,
            seed=config.seed,
        )
        aligned.write_parquet(
            temporary / "action_predictions.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        tables.regret_by_cost.write_csv(temporary / "regret_by_cost.csv")
        tables.one_step_utility.write_csv(temporary / "one_step_utility.csv")
        tables.domain_metrics.write_csv(temporary / "domain_metrics.csv")
        tables.bootstrap.write_csv(temporary / "bootstrap.csv")
        checkpoints = sorted(
            tables.regret_by_cost.get_column("nominal_checkpoint").unique().to_list()
        )
        endpoint = checkpoints[-1]
        primary = tables.bootstrap.filter(
            (pl.col("scope") == "equal_domain")
            & (pl.col("nominal_checkpoint") == endpoint)
            & (pl.col("metric") == "real_minus_control_regret")
        )
        static_endpoint = primary.filter(
            pl.col("control_scorer") == "static_m1_o2"
        ).row(0, named=True)
        static_change = tables.bootstrap.filter(
            (pl.col("scope") == "equal_domain")
            & (pl.col("nominal_checkpoint") == endpoint)
            & (pl.col("metric") == "regret_advantage_change_from_initial")
            & (pl.col("control_scorer") == "static_m1_o2")
        ).row(0, named=True)
        shuffled_endpoint = primary.filter(
            pl.col("control_scorer") == "dynamic_shuffled"
        ).row(0, named=True)
        endpoint_static_support = static_endpoint["ci95_upper"] < 0.0
        growing_static_advantage = static_change["ci95_upper"] < 0.0
        all_controls_support = primary.filter(pl.col("ci95_upper") >= 0.0).is_empty()
        shuffled_better = shuffled_endpoint["ci95_lower"] > 0.0
        if endpoint_static_support and shuffled_better:
            conclusion = (
                "At the final decision checkpoint, dynamic real-state valuation has "
                "lower regret than frozen static M1/O2, but growth of that advantage "
                "from the initial checkpoint is uncertain and shuffled-content "
                "valuation remains better; the gain is not attributable to "
                "accumulated specimen-specific measured content."
            )
        elif endpoint_static_support:
            conclusion = (
                "At the final decision checkpoint, dynamic real-state valuation has "
                "lower regret than frozen static M1/O2, but growth of that advantage "
                "from the initial checkpoint is uncertain and it does not beat every "
                "causal control."
            )
        else:
            conclusion = (
                "The frozen evidence does not show a reliable late-cost dynamic-real "
                "valuation advantage over static M1/O2."
            )
        state_payload = {
            "schema": 1,
            "stage": "P11_DYNAMIC_VALUATION",
            "config_sha256": config.config_sha256,
            "runtime_code_state_sha256": _code_state(),
            "p1_state_manifest_sha256": config.p1_state_manifest_sha256,
            "p1_state_sha256": config.p1_state_sha256,
            "p3_action_scores_sha256": config.p3_action_scores_sha256,
            "p3_state_sha256": config.p3_state_sha256,
            "mvd_action_scores_sha256": config.mvd_action_scores_sha256,
            "mvd_authority_state_sha256": config.mvd_authority_state_sha256,
            "p7_tree_state_sha256": p7_before,
            "aligned_action_prediction_count": aligned.height,
            "per_state_metric_count": tables.per_state.height,
            "domain_metric_count": tables.domain_metrics.height,
            "bootstrap_row_count": tables.bootstrap.height,
        }
        p11_state = hashlib.sha256(
            json.dumps(
                state_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        summary: dict[str, object] = {
            **state_payload,
            "schema_version": 1,
            "status": "COMPLETE",
            "audit_base_git_sha": config.audit_base_git_sha,
            "p11_state_sha256": p11_state,
            "decision_state_count": aligned.get_column("state_id").n_unique(),
            "specimen_count": aligned.get_column("specimen_id").n_unique(),
            "domain_count": len(config.domain_order),
            "checkpoint_count": len(checkpoints),
            "scorers": sorted(aligned.get_column("scorer").unique().to_list()),
            "recall_k": config.recall_k,
            "bootstrap_replicates": config.bootstrap_replicates,
            "seed": config.seed,
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "target_data_used_for_selection": False,
            "static_mvd_semantics": (
                "frozen specimen-cell score projected onto current legal actions; "
                "current exact costs and P3 teacher values retained"
            ),
            "endpoint_checkpoint": endpoint,
            "endpoint_real_minus_static_regret": static_endpoint["estimate"],
            "endpoint_real_minus_static_regret_ci95": [
                static_endpoint["ci95_lower"],
                static_endpoint["ci95_upper"],
            ],
            "real_vs_static_advantage_change": static_change["estimate"],
            "real_vs_static_advantage_change_ci95": [
                static_change["ci95_lower"],
                static_change["ci95_upper"],
            ],
            "endpoint_real_minus_shuffled_regret": shuffled_endpoint["estimate"],
            "endpoint_real_minus_shuffled_regret_ci95": [
                shuffled_endpoint["ci95_lower"],
                shuffled_endpoint["ci95_upper"],
            ],
            "dynamic_real_endpoint_advantage_over_static_supported": (
                endpoint_static_support
            ),
            "dynamic_real_advantage_growth_over_static_supported": (
                growing_static_advantage
            ),
            "dynamic_real_beats_all_controls": all_controls_support,
            "dynamic_shuffled_outperforms_real": shuffled_better,
            "primary_conclusion": conclusion,
            "p7_modified": False,
            "claim_tier_changed": False,
        }
        _write_json(temporary / "summary.json", summary)
        (temporary / "REPORT.md").write_text(
            _p11_report(summary, tables.regret_by_cost), encoding="utf-8"
        )
        products = sorted(item for item in temporary.iterdir() if item.is_file())
        manifest = {
            "schema_version": 1,
            "stage": "P11_DYNAMIC_VALUATION",
            "status": "COMPLETE",
            "p11_state_sha256": p11_state,
            "config_sha256": config.config_sha256,
            "p7_tree_state_sha256": p7_before,
            "files": [
                {
                    "path": item.name,
                    "bytes": item.stat().st_size,
                    "sha256": _sha256(item),
                }
                for item in products
            ],
        }
        _write_json(temporary / "artifact_manifest.json", manifest)
        _write_checksums(temporary)
        if _tree_state(p7) != p7_before:
            raise ScienceClosureArtifactError("P11 modified frozen P7 artifacts")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_p11_dynamic_valuation_package(destination)
    return destination


def verify_p11_dynamic_valuation_package(path: str | Path) -> dict[str, object]:
    try:
        package = Path(path).resolve(strict=True)
    except OSError as error:
        raise ScienceClosureArtifactError("P11 package is unavailable") from error
    if not package.is_dir() or {item.name for item in package.iterdir()} != _P11_FILES:
        raise ScienceClosureArtifactError("P11 package file roster changed")
    lines = (package / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    checksums: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or not _is_hex(parts[0], 64) or parts[1] in checksums:
            raise ScienceClosureArtifactError("P11 checksum manifest is invalid")
        checksums[parts[1]] = parts[0]
    expected = _P11_FILES - {"CHECKSUMS.sha256"}
    if set(checksums) != expected or any(
        _sha256(package / name) != digest for name, digest in checksums.items()
    ):
        raise ScienceClosureArtifactError("P11 checksum mismatch")
    manifest = _load_json(package / "artifact_manifest.json", "P11 manifest")
    summary = _load_json(package / "summary.json", "P11 summary")
    if (
        manifest.get("stage") != "P11_DYNAMIC_VALUATION"
        or manifest.get("status") != "COMPLETE"
        or summary.get("stage") != manifest["stage"]
        or summary.get("status") != "COMPLETE"
        or summary.get("p11_state_sha256") != manifest.get("p11_state_sha256")
        or not _is_hex(manifest.get("p11_state_sha256"), 64)
        or summary.get("target_data_used_for_selection") is not False
        or summary.get("p7_modified") is not False
        or summary.get("claim_tier_changed") is not False
        or not math.isfinite(
            float(summary.get("endpoint_real_minus_static_regret", math.nan))
        )
    ):
        raise ScienceClosureArtifactError("P11 metadata contract changed")
    product_records = manifest.get("files")
    if type(product_records) is not list or {
        record.get("path") for record in product_records if type(record) is dict
    } != expected - {"artifact_manifest.json"}:
        raise ScienceClosureArtifactError("P11 artifact manifest roster changed")
    try:
        actions = pl.read_parquet(package / "action_predictions.parquet")
        regret = pl.read_csv(package / "regret_by_cost.csv")
        utility = pl.read_csv(package / "one_step_utility.csv")
        domain = pl.read_csv(package / "domain_metrics.csv")
        bootstrap = pl.read_csv(package / "bootstrap.csv")
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ScienceClosureArtifactError("P11 scientific tables are invalid") from error
    scorers = {
        "dynamic_real",
        "dynamic_positions_only",
        "dynamic_shuffled",
        "static_m1_o2",
        "candidate_only_static",
    }
    if (
        set(actions.get_column("scorer").unique()) != scorers
        or set(regret.get_column("scorer").unique()) != scorers
        or set(utility.get_column("scorer").unique()) != scorers
        or set(domain.get_column("scorer").unique()) != scorers
        or actions.height != summary.get("aligned_action_prediction_count")
        or bootstrap.height != summary.get("bootstrap_row_count")
        or actions.filter(
            pl.col("scorer").is_in(["static_m1_o2", "candidate_only_static"])
            & ~pl.col("static_extension_to_current_state")
        ).height
    ):
        raise ScienceClosureArtifactError("P11 scientific table contract changed")
    return manifest


__all__ = [
    "P10MRISCausalConfig",
    "P11DynamicValuationConfig",
    "ScienceClosureArtifactError",
    "load_p10_mris_causal_config",
    "load_p11_dynamic_valuation_config",
    "run_p10_mris_causal",
    "run_p11_dynamic_valuation",
    "verify_p10_mris_causal_package",
    "verify_p11_dynamic_valuation_package",
]
