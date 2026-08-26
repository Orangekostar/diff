"""Finalize and verify the MAVIS P5 source-only aggregation package."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from .config import load_mavis_config
from .dynamic_package import verify_dynamic_package
from .mris_data import load_mris_feature_bank
from .mris_package import verify_mris_package
from .state_bank_package import (
    load_state_manifest_package,
    verify_state_bank_package,
)


class MAVISAggregationPackageError(RuntimeError):
    """Raised when P5 workers or their formal package are inconsistent."""


_WORKER_FILES = {
    "aggregated_states.parquet",
    "checkpoint.npz",
    "round_audit.parquet",
    "source_rollout_trajectories.parquet",
}
_CODE_PATHS = (
    "src/cmc_bbdm/mavis/aggregation.py",
    "src/cmc_bbdm/mavis/aggregation_execution.py",
    "src/cmc_bbdm/mavis/aggregation_package.py",
    "src/cmc_bbdm/mavis/dynamic_data.py",
    "src/cmc_bbdm/mavis/dynamic_training.py",
    "src/cmc_bbdm/mavis/dynamic_voi.py",
    "src/cmc_bbdm/mavis/policy.py",
    "src/cmc_bbdm/mavis/rollout.py",
    "src/cmc_bbdm/mavis/state_candidates.py",
    "src/cmc_bbdm/mavis/teacher.py",
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
            raise MAVISAggregationPackageError(
                f"P5 runtime source is unavailable: {relative}"
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
        raise MAVISAggregationPackageError("P5 Git provenance is unavailable") from error
    if len(git_sha) != 40:
        raise MAVISAggregationPackageError("P5 Git SHA is invalid")
    return git_sha, digest.hexdigest(), dirty


def _worker(
    path: Path,
    *,
    outer_domain: str,
    config_sha256: str,
    feature_input_state: str,
    feature_target_state: str,
    source_domains: tuple[str, ...],
    source_specimen_count: int,
    rounds: int,
) -> dict[str, object]:
    try:
        payload = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISAggregationPackageError("P5 worker summary is invalid") from error
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("outer_domain") != outer_domain
        or payload.get("config_sha256") != config_sha256
        or payload.get("feature_bank_input_state_sha256") != feature_input_state
        or payload.get("feature_bank_target_state_sha256") != feature_target_state
        or payload.get("round_count") != rounds
        or tuple(payload.get("source_domains", ())) != source_domains
        or payload.get("source_specimen_count") != source_specimen_count
        or payload.get("target_state_count") != 0
        or payload.get("target_data_used_for_training") is not False
        or type(payload.get("initial_state_count")) is not int
        or type(payload.get("final_state_count")) is not int
        or payload["final_state_count"] < payload["initial_state_count"]
        or type(payload.get("files")) is not dict
        or set(payload["files"]) != _WORKER_FILES
    ):
        raise MAVISAggregationPackageError("P5 worker contract changed")
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "complete.json"
    }
    if actual != _WORKER_FILES:
        raise MAVISAggregationPackageError("P5 worker file roster changed")
    for relative, expected in payload["files"].items():
        if type(expected) is not str or _sha256(path / relative) != expected:
            raise MAVISAggregationPackageError(
                f"P5 worker checksum mismatch: {relative}"
            )
    return payload


def _validate_tables(
    audits: pl.DataFrame,
    states: pl.DataFrame,
    trajectories: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    rounds: int,
) -> None:
    if (
        audits.height != len(domain_order) * rounds
        or audits.unique(subset=["outer_domain", "round_index"]).height
        != audits.height
        or set(audits.get_column("outer_domain").unique()) != set(domain_order)
        or audits.filter(pl.col("target_state_count") != 0).height
        or states.height == 0
        or states.unique(subset=["outer_domain", "state_id"]).height != states.height
        or states.filter(pl.col("outer_domain") == pl.col("domain_id")).height
        or trajectories.height == 0
        or trajectories.filter(pl.col("outer_domain") == pl.col("domain_id")).height
        or set(trajectories.get_column("round_index").unique())
        != set(range(rounds))
    ):
        raise MAVISAggregationPackageError("P5 scientific table roster is incomplete")


def finalize_aggregation_package(
    config_path: str | Path,
    *,
    project_root: str | Path,
    feature_bank_path: str | Path,
    worker_root: str | Path,
    p1_package: str | Path,
    p2_package: str | Path,
    p3_package: str | Path,
) -> Path:
    root = Path(project_root).resolve(strict=True)
    config = load_mavis_config(Path(config_path).resolve(strict=True), project_root=root)
    bank = load_mris_feature_bank(feature_bank_path)
    p1_root = Path(p1_package).resolve(strict=True)
    p2_root = Path(p2_package).resolve(strict=True)
    p3_root = Path(p3_package).resolve(strict=True)
    p1_manifest = verify_state_bank_package(p1_root)
    p2_manifest = verify_mris_package(p2_root)
    p3_manifest = verify_dynamic_package(p3_root)
    state_manifest = load_state_manifest_package(p1_root)
    workers_root = Path(worker_root).resolve(strict=True)
    audits_parts: list[pl.DataFrame] = []
    states_parts: list[pl.DataFrame] = []
    trajectory_parts: list[pl.DataFrame] = []
    workers: list[dict[str, object]] = []
    for outer_domain in config.domain_order:
        source_domains = tuple(
            domain for domain in config.domain_order if domain != outer_domain
        )
        source_count = state_manifest.filter(
            pl.col("domain_id") != outer_domain
        ).get_column("specimen_id").n_unique()
        worker_path = workers_root / outer_domain
        workers.append(
            _worker(
                worker_path,
                outer_domain=outer_domain,
                config_sha256=config.config_sha256,
                feature_input_state=bank.input_state_sha256,
                feature_target_state=bank.target_state_sha256,
                source_domains=source_domains,
                source_specimen_count=source_count,
                rounds=config.on_policy_rounds,
            )
        )
        try:
            audits_parts.append(pl.read_parquet(worker_path / "round_audit.parquet"))
            states_parts.append(
                pl.read_parquet(worker_path / "aggregated_states.parquet")
            )
            trajectory_parts.append(
                pl.read_parquet(worker_path / "source_rollout_trajectories.parquet")
            )
        except (OSError, pl.exceptions.PolarsError) as error:
            raise MAVISAggregationPackageError("P5 worker table is invalid") from error
    audits = pl.concat(audits_parts, how="vertical_relaxed").sort(
        ["outer_domain", "round_index"]
    )
    states = pl.concat(states_parts, how="vertical_relaxed").sort(
        ["outer_domain", "domain_id", "specimen_id", "state_id"]
    )
    trajectories = pl.concat(trajectory_parts, how="vertical_relaxed").sort(
        ["outer_domain", "round_index", "domain_id", "specimen_id", "step"]
    )
    _validate_tables(
        audits,
        states,
        trajectories,
        domain_order=config.domain_order,
        rounds=config.on_policy_rounds,
    )
    git_sha, code_state_sha256, dirty = _repository_state(root)
    p5_state = hashlib.sha256(
        json.dumps(
            {
                "schema": 1,
                "config_sha256": config.config_sha256,
                "p1_state_sha256": p1_manifest.get("state_bank_state_sha256"),
                "p2_state_sha256": p2_manifest.get("p2_state_sha256"),
                "p3_state_sha256": p3_manifest.get("p3_state_sha256"),
                "workers": workers,
                "runtime_code_state_sha256": code_state_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    output = root / config.output_root / "p5_aggregation"
    if output.exists():
        raise MAVISAggregationPackageError("P5 formal package already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p5_aggregation.", dir=output.parent))
    try:
        audits.write_parquet(
            temporary / "round_audit.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        states.write_parquet(
            temporary / "aggregated_states.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        trajectories.write_parquet(
            temporary / "source_rollout_trajectories.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        checkpoint_root = temporary / "checkpoints"
        checkpoint_root.mkdir()
        for outer_domain in config.domain_order:
            shutil.copy2(
                workers_root / outer_domain / "checkpoint.npz",
                checkpoint_root / f"{outer_domain}__real.npz",
            )
        summary = {
            "schema_version": 1,
            "stage": "P5_SOURCE_ONLY_ON_POLICY_AGGREGATION",
            "status": "COMPLETE",
            "git_sha": git_sha,
            "git_worktree_dirty_at_run": dirty,
            "runtime_code_state_sha256": code_state_sha256,
            "config_sha256": config.config_sha256,
            "p1_state_sha256": p1_manifest.get("state_bank_state_sha256"),
            "p2_state_sha256": p2_manifest.get("p2_state_sha256"),
            "p3_state_sha256": p3_manifest.get("p3_state_sha256"),
            "p5_state_sha256": p5_state,
            "outer_domain_count": len(config.domain_order),
            "round_count": config.on_policy_rounds,
            "final_state_count": states.height,
            "source_trajectory_row_count": trajectories.height,
            "target_state_count": 0,
            "target_data_used_for_training": False,
        }
        _write_json(temporary / "summary.json", summary)
        (temporary / "REPORT.md").write_text(
            "# MAVIS P5 Source-only On-policy Aggregation\n\n"
            "Status: `COMPLETE`. Three aggregation rounds use only source-domain "
            "rollouts and strict-OOF source teachers. At most one visited decision "
            "state per active budget checkpoint and physical specimen is retained "
            "per round. Outer-target states, labels, and outcomes are excluded from "
            "training and model selection.\n",
            encoding="utf-8",
        )
        scientific_files = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        )
        manifest = {
            "schema_version": 1,
            "artifact": "mavis_p5_aggregation",
            "p5_state_sha256": p5_state,
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
    verify_aggregation_package(output)
    return output


def verify_aggregation_package(path: str | Path) -> dict[str, object]:
    root = Path(path)
    try:
        manifest = json.loads(
            (root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        lines = (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISAggregationPackageError("P5 package metadata is invalid") from error
    if type(manifest) is not dict or type(manifest.get("files")) is not dict:
        raise MAVISAggregationPackageError("P5 package manifest is invalid")
    expected = set(manifest["files"]) | {"artifact_manifest.json"}
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "CHECKSUMS.sha256"
    }
    if actual != expected or len(lines) != len(expected):
        raise MAVISAggregationPackageError("P5 package file roster changed")
    ledger: dict[str, str] = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise MAVISAggregationPackageError("P5 checksum ledger is invalid") from error
        if name in ledger or len(digest) != 64:
            raise MAVISAggregationPackageError("P5 checksum ledger is invalid")
        ledger[name] = digest
    if set(ledger) != expected:
        raise MAVISAggregationPackageError("P5 checksum roster changed")
    for name, expected_digest in ledger.items():
        if _sha256(root / name) != expected_digest:
            raise MAVISAggregationPackageError(f"P5 checksum mismatch: {name}")
    for name, metadata in manifest["files"].items():
        file_path = root / name
        if (
            type(metadata) is not dict
            or metadata.get("bytes") != file_path.stat().st_size
            or metadata.get("sha256") != _sha256(file_path)
        ):
            raise MAVISAggregationPackageError(f"P5 manifest mismatch: {name}")
    if any((root / name).stat().st_size >= _GIT_BLOB_LIMIT for name in expected):
        raise MAVISAggregationPackageError("P5 package contains an oversized Git blob")
    return manifest


__all__ = [
    "MAVISAggregationPackageError",
    "finalize_aggregation_package",
    "verify_aggregation_package",
]
