"""Finalize and verify the MAVIS P3 dynamic-VoI evidence package."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from .config import load_mavis_config
from .dynamic_metrics import (
    aggregate_dynamic_metrics,
    bootstrap_dynamic_contrasts,
)
from .mris_data import load_mris_feature_bank
from .mris_package import verify_mris_package
from .state_bank_package import verify_state_bank_package


class MAVISDynamicPackageError(RuntimeError):
    """Raised when P3 workers or their formal package are inconsistent."""


_MODES = ("static", "positions_only", "real", "shuffled")
_CODE_PATHS = (
    "src/cmc_bbdm/mavis/dynamic_data.py",
    "src/cmc_bbdm/mavis/dynamic_execution.py",
    "src/cmc_bbdm/mavis/dynamic_metrics.py",
    "src/cmc_bbdm/mavis/dynamic_package.py",
    "src/cmc_bbdm/mavis/dynamic_training.py",
    "src/cmc_bbdm/mavis/dynamic_voi.py",
    "src/cmc_bbdm/mavis/losses.py",
)
_GIT_BLOB_LIMIT = 100 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_checksums(output: Path) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    (output / "CHECKSUMS.sha256").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n"
            for path in files
        ),
        encoding="ascii",
    )


def _repository_state(root: Path) -> tuple[str, str, bool]:
    digest = hashlib.sha256()
    for relative in _CODE_PATHS:
        path = root / relative
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise MAVISDynamicPackageError(
                f"P3 runtime source is unavailable: {relative}"
            ) from error
        digest.update(relative.encode("utf-8"))
        digest.update(hashlib.sha256(payload).digest())
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise MAVISDynamicPackageError("P3 Git provenance is unavailable") from error
    if len(git_sha) != 40:
        raise MAVISDynamicPackageError("P3 Git SHA is invalid")
    return git_sha, digest.hexdigest(), dirty


def _decision_state_roster(
    states: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    feature_state_ids: tuple[str, ...],
) -> tuple[frozenset[str], dict[str, int]]:
    required = {"state_id", "domain_id", "candidate_cell_indices"}
    if (
        not isinstance(states, pl.DataFrame)
        or not required <= set(states.columns)
        or type(domain_order) is not tuple
        or not domain_order
        or len(set(domain_order)) != len(domain_order)
        or type(feature_state_ids) is not tuple
        or not feature_state_ids
        or len(set(feature_state_ids)) != len(feature_state_ids)
        or states.height != len(feature_state_ids)
        or states.get_column("state_id").n_unique() != states.height
        or states.get_column("candidate_cell_indices").null_count()
        or set(states.get_column("state_id")) != set(feature_state_ids)
        or set(states.get_column("domain_id")) != set(domain_order)
    ):
        raise MAVISDynamicPackageError("P3 P1 state roster is invalid")
    decision_states = states.filter(
        pl.col("candidate_cell_indices").list.len() > 0
    )
    state_ids = frozenset(decision_states.get_column("state_id"))
    counts = {
        domain: decision_states.filter(pl.col("domain_id") == domain).height
        for domain in domain_order
    }
    if not state_ids or any(count <= 0 for count in counts.values()):
        raise MAVISDynamicPackageError("P3 decision-state roster is empty")
    return state_ids, counts


def _worker(
    path: Path,
    *,
    outer_domain: str,
    input_state_sha256: str,
    target_state_sha256: str,
    target_group_count: int,
) -> dict[str, object]:
    try:
        payload = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISDynamicPackageError("P3 worker summary is invalid") from error
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("outer_domain") != outer_domain
        or payload.get("feature_bank_input_state_sha256") != input_state_sha256
        or payload.get("feature_bank_target_state_sha256") != target_state_sha256
        or tuple(payload.get("modes", ())) != _MODES
        or payload.get("target_group_count") != target_group_count
        or payload.get("inner_fold_count") != 20
        or payload.get("target_data_used_for_selection") is not False
        or set(payload.get("p2_model_state_sha256", {})) != set(_MODES)
        or set(payload.get("dynamic_model_state_sha256", {})) != set(_MODES)
        or type(payload.get("inner_p2_model_state_sha256")) is not dict
        or len(payload["inner_p2_model_state_sha256"]) != 20
        or type(payload.get("inner_dynamic_model_state_sha256")) is not dict
        or len(payload["inner_dynamic_model_state_sha256"]) != 20
        or type(payload.get("files")) is not dict
    ):
        raise MAVISDynamicPackageError("P3 worker contract changed")
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "complete.json"
    }
    if actual != set(payload["files"]):
        raise MAVISDynamicPackageError("P3 worker file roster changed")
    for relative, digest in payload["files"].items():
        if type(digest) is not str or _sha256(path / relative) != digest:
            raise MAVISDynamicPackageError(f"P3 worker checksum mismatch: {relative}")
    return payload


def _validate_tables(
    action_scores: pl.DataFrame,
    state_metrics: pl.DataFrame,
    audits: pl.DataFrame,
    *,
    decision_state_ids: frozenset[str],
    domain_count: int,
) -> None:
    if (
        state_metrics.height != len(decision_state_ids) * len(_MODES)
        or state_metrics.unique(subset=["state_id", "mode"]).height
        != state_metrics.height
        or set(state_metrics.get_column("state_id")) != decision_state_ids
        or set(state_metrics.get_column("mode").unique()) != set(_MODES)
        or state_metrics.filter(pl.col("outer_domain") != pl.col("domain_id")).height
        or action_scores.height == 0
        or set(action_scores.get_column("state_id")) != decision_state_ids
        or action_scores.unique(subset=["state_id", "mode", "candidate_index"]).height
        != action_scores.height
        or action_scores.filter(pl.col("outer_domain") != pl.col("domain_id")).height
        or audits.height != domain_count * len(_MODES) * 6
    ):
        raise MAVISDynamicPackageError("P3 scientific table roster is incomplete")
    inner = audits.filter(pl.col("record_type") == "inner_fold")
    final = audits.filter(pl.col("record_type") == "final_refit")
    inner_ok = inner.select(
        (
            (pl.col("fit_domains").list.len() == domain_count - 2)
            & (~pl.col("fit_domains").list.contains(pl.col("outer_domain")))
            & (~pl.col("fit_domains").list.contains(pl.col("validation_domain")))
            & (pl.col("target_data_used_for_selection") == False)
        ).all()
    ).item()
    final_ok = final.select(
        (
            (pl.col("fit_domains").list.len() == domain_count - 1)
            & (~pl.col("fit_domains").list.contains(pl.col("outer_domain")))
            & pl.col("validation_domain").is_null()
            & (pl.col("target_data_used_for_selection") == False)
        ).all()
    ).item()
    if not inner_ok or not final_ok:
        raise MAVISDynamicPackageError("P3 source-only model selection audit failed")
    for table in (action_scores, state_metrics):
        if table.select(pl.any_horizontal(pl.selectors.numeric().is_nan())).to_series().any():
            raise MAVISDynamicPackageError("P3 scientific table contains NaN")


def _bootstrap_summary(bootstrap: pl.DataFrame) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for control in sorted(bootstrap.get_column("control_mode").unique()):
        table = bootstrap.filter(pl.col("control_mode") == control)
        regret = table.get_column("control_minus_reference_regret").to_numpy()
        utility = table.get_column("reference_minus_control_utility").to_numpy()
        output[control] = {
            "mean_control_minus_real_regret": float(np.mean(regret)),
            "regret_ci95_lower": float(np.quantile(regret, 0.025)),
            "regret_ci95_upper": float(np.quantile(regret, 0.975)),
            "mean_real_minus_control_utility": float(np.mean(utility)),
            "utility_ci95_lower": float(np.quantile(utility, 0.025)),
            "utility_ci95_upper": float(np.quantile(utility, 0.975)),
        }
    return output


def finalize_dynamic_package(
    config_path: str | Path,
    *,
    project_root: str | Path,
    feature_bank_path: str | Path,
    worker_root: str | Path,
    p1_package: str | Path,
    p2_package: str | Path,
    bootstrap_replicates: int,
) -> Path:
    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_mavis_config(config_file, project_root=root)
    bank = load_mris_feature_bank(feature_bank_path)
    p1_root = Path(p1_package).resolve(strict=True)
    p1_manifest = verify_state_bank_package(p1_root)
    p2_root = Path(p2_package).resolve(strict=True)
    p2_manifest = verify_mris_package(p2_root)
    workers_root = Path(worker_root).resolve(strict=True)
    if bank.domain_order != config.domain_order or bank.row_count != 8280:
        raise MAVISDynamicPackageError("P3 feature bank disagrees with config")
    try:
        p1_states = pl.read_parquet(
            p1_root / "state_manifest.parquet",
            columns=["state_id", "domain_id", "candidate_cell_indices"],
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise MAVISDynamicPackageError("P3 P1 state roster is unavailable") from error
    decision_state_ids, decision_counts = _decision_state_roster(
        p1_states,
        domain_order=config.domain_order,
        feature_state_ids=bank.state_ids,
    )
    action_parts: list[pl.DataFrame] = []
    metric_parts: list[pl.DataFrame] = []
    audit_parts: list[pl.DataFrame] = []
    workers: list[dict[str, object]] = []
    for outer_domain in config.domain_order:
        target_count = decision_counts[outer_domain]
        worker_path = workers_root / outer_domain
        workers.append(
            _worker(
                worker_path,
                outer_domain=outer_domain,
                input_state_sha256=bank.input_state_sha256,
                target_state_sha256=bank.target_state_sha256,
                target_group_count=target_count,
            )
        )
        try:
            action_parts.append(pl.read_parquet(worker_path / "action_scores.parquet"))
            metric_parts.append(pl.read_parquet(worker_path / "state_metrics.parquet"))
            audit_parts.append(
                pl.read_parquet(worker_path / "model_selection_audit.parquet")
            )
        except (OSError, pl.exceptions.PolarsError) as error:
            raise MAVISDynamicPackageError("P3 worker table is invalid") from error
    action_scores = pl.concat(action_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "state_id", "mode", "candidate_index"]
    )
    state_metrics = pl.concat(metric_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "state_id", "mode"]
    )
    audits = pl.concat(audit_parts, how="vertical_relaxed").sort(
        ["outer_domain", "mode", "record_type", "validation_domain"],
        nulls_last=True,
    )
    _validate_tables(
        action_scores,
        state_metrics,
        audits,
        decision_state_ids=decision_state_ids,
        domain_count=len(config.domain_order),
    )
    metrics = aggregate_dynamic_metrics(state_metrics)
    bootstrap = bootstrap_dynamic_contrasts(
        metrics.per_specimen,
        reference_mode="real",
        control_modes=("static", "positions_only", "shuffled"),
        domain_order=config.domain_order,
        replicates=bootstrap_replicates,
        seed=config.seed + 300,
    )
    git_sha, code_state_sha256, dirty = _repository_state(root)
    p3_state = hashlib.sha256(
        json.dumps(
            {
                "schema": 1,
                "config_sha256": config.config_sha256,
                "p1_state_sha256": p1_manifest.get("state_bank_state_sha256"),
                "p2_state_sha256": p2_manifest.get("p2_state_sha256"),
                "feature_bank_input_state_sha256": bank.input_state_sha256,
                "workers": [worker["dynamic_model_state_sha256"] for worker in workers],
                "runtime_code_state_sha256": code_state_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    output = root / config.output_root / "p3_dynamic_voi"
    if output.exists():
        raise MAVISDynamicPackageError("P3 formal package already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p3_dynamic_voi.", dir=output.parent))
    try:
        action_scores.write_parquet(
            temporary / "action_scores.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        state_metrics.write_parquet(
            temporary / "state_metrics.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        audits.write_parquet(
            temporary / "model_selection_audit.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        metrics.per_specimen.write_csv(temporary / "per_specimen_metrics.csv")
        metrics.per_domain.write_csv(temporary / "domain_metrics.csv")
        metrics.aggregate.write_csv(temporary / "aggregate_metrics.csv")
        bootstrap.write_csv(temporary / "bootstrap.csv")
        shutil.copyfile(config_file, temporary / "config.yaml")
        checkpoint_root = temporary / "checkpoints"
        checkpoint_root.mkdir()
        inner_checkpoint_root = checkpoint_root / "inner"
        inner_checkpoint_root.mkdir()
        for outer_domain in config.domain_order:
            for mode in _MODES:
                shutil.copyfile(
                    workers_root / outer_domain / "checkpoints" / f"{mode}.npz",
                    checkpoint_root / f"{outer_domain}__{mode}.npz",
                )
                for validation_domain in config.domain_order:
                    if validation_domain == outer_domain:
                        continue
                    shutil.copyfile(
                        workers_root
                        / outer_domain
                        / "checkpoints"
                        / "inner"
                        / f"{validation_domain}__{mode}.npz",
                        inner_checkpoint_root
                        / f"{outer_domain}__{validation_domain}__{mode}.npz",
                    )
        summary = {
            "schema_version": 1,
            "artifact": "mavis_p3_dynamic_voi",
            "status": "COMPLETE",
            "p3_state_sha256": p3_state,
            "git_sha": git_sha,
            "git_worktree_dirty": dirty,
            "runtime_code_state_sha256": code_state_sha256,
            "config_sha256": config.config_sha256,
            "p1_state_sha256": p1_manifest.get("state_bank_state_sha256"),
            "p2_state_sha256": p2_manifest.get("p2_state_sha256"),
            "input_state_count": bank.row_count,
            "state_count": len(decision_state_ids),
            "decision_state_count": len(decision_state_ids),
            "terminal_state_count": bank.row_count - len(decision_state_ids),
            "action_score_count": action_scores.height,
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "target_data_used_for_selection": False,
            "bootstrap": _bootstrap_summary(bootstrap),
            "claim_tier_assigned": False,
        }
        _write_json(temporary / "summary.json", summary)
        real = metrics.aggregate.filter(pl.col("mode") == "real").row(0, named=True)
        report = (
            "# MAVIS P3 Dynamic Mechanical Value of Information\n\n"
            "Status: `COMPLETE`. This development stage does not assign the final "
            "MAVIS claim tier.\n\n"
            f"The equal-domain real-state next-action regret is "
            f"`{float(real['next_action_regret']):.10f}` and selected one-step CAI "
            f"utility is `{float(real['one_step_cai_utility']):.10f}`.\n\n"
            "Each outer target is evaluation only. Dynamic model early stopping and "
            "final fitting use source domains; target CAI and unacquired target "
            "measurements are unavailable to the scorer. Target teacher values are "
            "computed only after scoring from strict-OOF P1 predictions. Metrics "
            "aggregate states to physical specimens and then weight held-out domains "
            "equally.\n"
        )
        (temporary / "REPORT.md").write_text(report, encoding="utf-8")
        scientific_files = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        )
        manifest = {
            "schema_version": 1,
            "artifact": "mavis_p3_dynamic_voi",
            "p3_state_sha256": p3_state,
            "config_sha256": config.config_sha256,
            "runtime_code_state_sha256": code_state_sha256,
            "files": {
                name: {
                    "bytes": (temporary / name).stat().st_size,
                    "sha256": _sha256(temporary / name),
                }
                for name in scientific_files
            },
        }
        _write_json(temporary / "artifact_manifest.json", manifest)
        _write_checksums(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    verify_dynamic_package(output)
    return output


def verify_dynamic_package(path: str | Path) -> dict[str, object]:
    root = Path(path)
    try:
        manifest = json.loads(
            (root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        lines = (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISDynamicPackageError("P3 package metadata is invalid") from error
    expected = set(manifest.get("files", {})) | {"artifact_manifest.json"}
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "CHECKSUMS.sha256"
    }
    if actual != expected or len(lines) != len(expected):
        raise MAVISDynamicPackageError("P3 package file roster changed")
    ledger: dict[str, str] = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise MAVISDynamicPackageError("P3 checksum ledger is invalid") from error
        if name in ledger or len(digest) != 64:
            raise MAVISDynamicPackageError("P3 checksum ledger is invalid")
        ledger[name] = digest
    if set(ledger) != expected:
        raise MAVISDynamicPackageError("P3 checksum roster changed")
    for name, digest in ledger.items():
        if _sha256(root / name) != digest:
            raise MAVISDynamicPackageError(f"P3 checksum mismatch: {name}")
    for name, metadata in manifest["files"].items():
        file_path = root / name
        if (
            metadata.get("bytes") != file_path.stat().st_size
            or metadata.get("sha256") != _sha256(file_path)
        ):
            raise MAVISDynamicPackageError(f"P3 manifest mismatch: {name}")
    if any((root / name).stat().st_size >= _GIT_BLOB_LIMIT for name in expected):
        raise MAVISDynamicPackageError("P3 package contains an oversized Git blob")
    return manifest


__all__ = [
    "MAVISDynamicPackageError",
    "finalize_dynamic_package",
    "verify_dynamic_package",
]
