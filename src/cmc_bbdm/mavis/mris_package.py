"""Finalize and verify the MAVIS P2 MRIS evidence package."""

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
from .mris_data import load_mris_feature_bank
from .mris_figure import render_mris_cost_curve
from .mris_metrics import bootstrap_mris_contrasts, evaluate_mris_predictions
from .state_bank_package import verify_state_bank_package


class MAVISMRISPackageError(RuntimeError):
    """Raised when P2 workers or their formal evidence package are invalid."""


_TRAINABLE_MODES = ("static", "positions_only", "real", "shuffled")
_ALL_MODES = (*_TRAINABLE_MODES, "reconstruction")
_CODE_PATHS = (
    "src/cmc_bbdm/mavis/mechanics_head.py",
    "src/cmc_bbdm/mavis/mris_data.py",
    "src/cmc_bbdm/mavis/mris_execution.py",
    "src/cmc_bbdm/mavis/mris_figure.py",
    "src/cmc_bbdm/mavis/mris_metrics.py",
    "src/cmc_bbdm/mavis/mris_package.py",
    "src/cmc_bbdm/mavis/mris_training.py",
    "src/cmc_bbdm/mavis/state_encoder.py",
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
            raise MAVISMRISPackageError(
                f"P2 runtime source is unavailable: {relative}"
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
        raise MAVISMRISPackageError("P2 Git provenance is unavailable") from error
    if len(git_sha) != 40:
        raise MAVISMRISPackageError("P2 Git SHA is invalid")
    return git_sha, digest.hexdigest(), dirty


def _worker_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISMRISPackageError("P2 worker summary is invalid") from error
    if type(payload) is not dict:
        raise MAVISMRISPackageError("P2 worker summary is invalid")
    return payload


def _validate_worker(
    worker: Path,
    *,
    outer_domain: str,
    feature_input_state: str,
    feature_target_state: str,
    expected_prediction_count: int,
) -> dict[str, object]:
    payload = _worker_payload(worker / "complete.json")
    if (
        payload.get("schema_version") != 1
        or payload.get("outer_domain") != outer_domain
        or payload.get("feature_bank_input_state_sha256") != feature_input_state
        or payload.get("feature_bank_target_state_sha256") != feature_target_state
        or tuple(payload.get("trainable_modes", ())) != _TRAINABLE_MODES
        or payload.get("prediction_count") != expected_prediction_count
        or payload.get("inner_fold_count") != 20
        or set(payload.get("model_state_sha256", {})) != set(_ALL_MODES)
        or type(payload.get("inner_model_state_sha256")) is not dict
        or len(payload.get("inner_model_state_sha256", {})) != 20
        or type(payload.get("files")) is not dict
    ):
        raise MAVISMRISPackageError("P2 worker contract changed")
    expected_files = set(payload["files"])
    actual_files = {
        path.relative_to(worker).as_posix()
        for path in worker.rglob("*")
        if path.is_file() and path.name != "complete.json"
    }
    if actual_files != expected_files:
        raise MAVISMRISPackageError("P2 worker file roster changed")
    for relative, digest in payload["files"].items():
        if type(digest) is not str or _sha256(worker / relative) != digest:
            raise MAVISMRISPackageError(f"P2 worker checksum mismatch: {relative}")
    return payload


def _validate_scientific_tables(
    predictions: pl.DataFrame,
    audits: pl.DataFrame,
    donors: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    specimen_count: int,
    state_count: int,
) -> None:
    if (
        predictions.height != state_count * len(_ALL_MODES)
        or predictions.unique(subset=["state_id", "mode"]).height
        != predictions.height
        or set(predictions.get_column("mode").unique()) != set(_ALL_MODES)
        or audits.height != len(domain_order) * len(_TRAINABLE_MODES) * 6
        or donors.height != len(domain_order) * specimen_count
        or donors.unique(subset=["outer_domain", "recipient_id"]).height
        != donors.height
        or donors.filter(pl.col("recipient_id") == pl.col("donor_id")).height != 0
    ):
        raise MAVISMRISPackageError("P2 scientific table roster is incomplete")
    inner = audits.filter(pl.col("record_type") == "inner_fold")
    final = audits.filter(pl.col("record_type") == "final_refit")
    inner_ok = inner.select(
        (
            (pl.col("fit_domains").list.len() == 4)
            & (pl.col("validation_domains").list.len() == 1)
            & (~pl.col("fit_domains").list.contains(pl.col("outer_domain")))
            & (~pl.col("fit_domains").list.contains(pl.col("validation_domain")))
            & (pl.col("target_data_used_for_selection") == False)
        ).all()
    ).item()
    final_ok = final.select(
        (
            (pl.col("fit_domains").list.len() == 5)
            & (pl.col("validation_domains").list.len() == 0)
            & (~pl.col("fit_domains").list.contains(pl.col("outer_domain")))
            & (pl.col("target_data_used_for_selection") == False)
        ).all()
    ).item()
    if not inner_ok or not final_ok:
        raise MAVISMRISPackageError("P2 source-only selection audit failed")


def _bootstrap_summary(bootstrap: pl.DataFrame) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for control in sorted(bootstrap.get_column("control_mode").unique()):
        values = bootstrap.filter(pl.col("control_mode") == control).get_column(
            "control_minus_reference_auebc"
        ).to_numpy()
        output[control] = {
            "mean_control_minus_real_auebc": float(np.mean(values, dtype=np.float64)),
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
            "fraction_control_worse_than_real": float(np.mean(values > 0.0)),
        }
    return output


def finalize_mris_package(
    config_path: str | Path,
    *,
    project_root: str | Path,
    feature_bank_path: str | Path,
    worker_root: str | Path,
    p1_package: str | Path,
    bootstrap_replicates: int,
) -> Path:
    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_mavis_config(config_file, project_root=root)
    bank = load_mris_feature_bank(feature_bank_path)
    p1_root = Path(p1_package).resolve(strict=True)
    p1_manifest = verify_state_bank_package(p1_root)
    workers_root = Path(worker_root).resolve(strict=True)
    if bank.domain_order != config.domain_order or bank.row_count != 8280:
        raise MAVISMRISPackageError("P2 feature bank disagrees with config")
    predictions_parts: list[pl.DataFrame] = []
    audit_parts: list[pl.DataFrame] = []
    donor_parts: list[pl.DataFrame] = []
    workers: list[dict[str, object]] = []
    for outer_domain in config.domain_order:
        target_rows = sum(domain == outer_domain for domain in bank.domain_ids)
        worker = workers_root / outer_domain
        workers.append(
            _validate_worker(
                worker,
                outer_domain=outer_domain,
                feature_input_state=bank.input_state_sha256,
                feature_target_state=bank.target_state_sha256,
                expected_prediction_count=target_rows * len(_ALL_MODES),
            )
        )
        try:
            predictions_parts.append(pl.read_parquet(worker / "predictions.parquet"))
            audit_parts.append(pl.read_parquet(worker / "model_selection_audit.parquet"))
            donor_parts.append(pl.read_parquet(worker / "donor_mapping.parquet"))
        except (OSError, pl.exceptions.PolarsError) as error:
            raise MAVISMRISPackageError("P2 worker tables are invalid") from error
    predictions = pl.concat(predictions_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "mode", "method", "nominal_checkpoint"]
    )
    audits = pl.concat(audit_parts, how="vertical_relaxed").sort(
        ["outer_domain", "mode", "record_type", "validation_domain"],
        nulls_last=True,
    )
    donors = pl.concat(donor_parts, how="vertical_relaxed").sort(
        ["outer_domain", "recipient_id"]
    )
    _validate_scientific_tables(
        predictions,
        audits,
        donors,
        domain_order=config.domain_order,
        specimen_count=config.specimen_count,
        state_count=bank.row_count,
    )
    metrics = evaluate_mris_predictions(predictions, domain_order=config.domain_order)
    bootstrap = bootstrap_mris_contrasts(
        metrics.per_specimen_metrics,
        reference_mode="real",
        control_modes=("static", "positions_only", "shuffled", "reconstruction"),
        domain_order=config.domain_order,
        replicates=bootstrap_replicates,
        seed=config.seed,
    )
    output = root / config.output_root / "p2_mris"
    if output.exists():
        raise MAVISMRISPackageError("P2 formal package already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p2_mris.", dir=output.parent))
    try:
        predictions.write_parquet(
            temporary / "state_predictions.parquet",
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
        donors.write_csv(temporary / "donor_mapping.csv")
        metrics.per_specimen_metrics.write_csv(temporary / "per_specimen_metrics.csv")
        metrics.domain_metrics.write_csv(temporary / "domain_metrics.csv")
        metrics.aggregate_metrics.write_csv(temporary / "aggregate_metrics.csv")
        metrics.domain_auebc.write_csv(temporary / "domain_auebc.csv")
        metrics.aggregate_auebc.write_csv(temporary / "aggregate_auebc.csv")
        bootstrap.write_csv(temporary / "bootstrap.csv")
        shutil.copyfile(config_file, temporary / "config.yaml")
        checkpoint_root = temporary / "checkpoints"
        checkpoint_root.mkdir()
        inner_checkpoint_root = checkpoint_root / "inner"
        inner_checkpoint_root.mkdir()
        for outer_domain in config.domain_order:
            for mode in _TRAINABLE_MODES:
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
        figures = temporary / "figures"
        render_mris_cost_curve(metrics.aggregate_metrics, output_root=figures)
        real_auebc = metrics.aggregate_auebc.filter(pl.col("mode") == "real").item(
            0, "domain_balanced_auebc"
        )
        comparisons = {
            row["mode"]: float(row["domain_balanced_auebc"] - real_auebc)
            for row in metrics.aggregate_auebc.filter(pl.col("mode") != "real").to_dicts()
        }
        bootstrap_summary = _bootstrap_summary(bootstrap)
        git_sha, code_state_sha256, git_dirty = _repository_state(root)
        core_files = sorted(
            path
            for path in temporary.rglob("*")
            if path.is_file() and path.name not in {"summary.json", "REPORT.md"}
        )
        state_digest = hashlib.sha256()
        state_digest.update(bank.input_state_sha256.encode("ascii"))
        state_digest.update(bank.target_state_sha256.encode("ascii"))
        for path in core_files:
            state_digest.update(path.relative_to(temporary).as_posix().encode("utf-8"))
            state_digest.update(bytes.fromhex(_sha256(path)))
        p2_state = state_digest.hexdigest()
        summary = {
            "schema_version": 1,
            "stage": "P2_MRIS_INFORMATIVENESS",
            "status": "COMPLETE",
            "git_sha": git_sha,
            "git_worktree_dirty_at_run": git_dirty,
            "runtime_code_state_sha256": code_state_sha256,
            "config_sha256": config.config_sha256,
            "p1_state_bank_state_sha256": p1_manifest[
                "state_bank_state_sha256"
            ],
            "feature_bank_input_state_sha256": bank.input_state_sha256,
            "feature_bank_target_state_sha256": bank.target_state_sha256,
            "p2_state_sha256": p2_state,
            "specimen_count": config.specimen_count,
            "state_count": bank.row_count,
            "prediction_count": predictions.height,
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "real_domain_balanced_auebc": float(real_auebc),
            "control_minus_real_auebc": comparisons,
            "bootstrap": bootstrap_summary,
            "claim_tier_assigned": False,
        }
        _write_json(temporary / "summary.json", summary)
        comparison_lines = "\n".join(
            f"- `{mode}` minus `real`: `{value:.10f}` AUEBC"
            for mode, value in sorted(comparisons.items())
        )
        report = (
            "# MAVIS P2 MRIS Informativeness\n\n"
            "Status: `COMPLETE`. This stage does not assign the final MAVIS claim tier.\n\n"
            f"The domain-balanced real-state CAI AUEBC is `{real_auebc:.10f}`. "
            "Control differences are positive when the control has higher error:\n\n"
            f"{comparison_lines}\n\n"
            "All predictions are nested leave-one-domain-out. Model selection and "
            "early stopping use source domains only. Metrics first aggregate state "
            "rows to physical specimens and then weight the six held-out domains "
            "equally. Shuffled content retains recipient positions and exact cost "
            "while using a recorded different donor specimen. Reconstruction values "
            "reuse strict-OOF P1 predictions and introduce no new reconstruction "
            "network.\n"
        )
        (temporary / "REPORT.md").write_text(report, encoding="utf-8")
        caption = (
            "CAI mean absolute error versus mean exact acquired native-raster "
            "fraction for static, positions-only, real measured, shuffled-content, "
            "and strict-OOF reconstruction states. Errors are averaged within each "
            "physical specimen and then equally across six held-out domains. Source: "
            "aggregate_metrics.csv; retrospective normalized-raster evaluation.\n"
        )
        (temporary / "FIGURE_CAPTION.md").write_text(caption, encoding="utf-8")
        scientific_files = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        )
        manifest = {
            "schema_version": 1,
            "artifact": "mavis_p2_mris",
            "p2_state_sha256": p2_state,
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
    verify_mris_package(output)
    return output


def verify_mris_package(path: str | Path) -> dict[str, object]:
    root = Path(path)
    try:
        manifest = json.loads(
            (root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        lines = (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISMRISPackageError("P2 package metadata is invalid") from error
    if type(manifest) is not dict or type(manifest.get("files")) is not dict:
        raise MAVISMRISPackageError("P2 package manifest is invalid")
    expected_files = set(manifest["files"]) | {"artifact_manifest.json"}
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if actual_files != expected_files or len(lines) != len(expected_files):
        raise MAVISMRISPackageError("P2 package file roster changed")
    ledger: dict[str, str] = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise MAVISMRISPackageError("P2 checksum ledger is invalid") from error
        if name in ledger or len(digest) != 64:
            raise MAVISMRISPackageError("P2 checksum ledger is invalid")
        ledger[name] = digest
    if set(ledger) != expected_files:
        raise MAVISMRISPackageError("P2 checksum coverage changed")
    for name, digest in ledger.items():
        if _sha256(root / name) != digest:
            raise MAVISMRISPackageError(f"P2 checksum mismatch: {name}")
    for name, metadata in manifest["files"].items():
        file_path = root / name
        if (
            type(metadata) is not dict
            or metadata.get("sha256") != _sha256(file_path)
            or metadata.get("bytes") != file_path.stat().st_size
        ):
            raise MAVISMRISPackageError(f"P2 manifest mismatch: {name}")
    if any((root / name).stat().st_size >= _GIT_BLOB_LIMIT for name in expected_files):
        raise MAVISMRISPackageError("P2 package contains an oversized Git blob")
    return manifest


__all__ = [
    "MAVISMRISPackageError",
    "finalize_mris_package",
    "verify_mris_package",
]
