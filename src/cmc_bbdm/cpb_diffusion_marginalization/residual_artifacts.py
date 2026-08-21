"""Artifact recording and publication for the D8 residual pre-outer search."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np

from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import V3Data, load_data

from .authority import issue_inner_fold, issue_search_view
from .config import DOMAIN_ORDER, load_d8_config
from .residual_config import (
    ResidualDiffusionConfig,
    load_residual_diffusion_config,
)
from .residual_model import load_residual_checkpoint, save_residual_checkpoint
from .residual_search import (
    ResidualCandidateSummary,
    ResidualCellEvaluation,
    ResidualCellRun,
    ResidualOuterSearchRun,
    ResidualSearchCell,
    ResidualSearchError,
    load_b0_incumbent_evidence,
    load_pilot_incumbent_evidence,
    promote_stage_a_outer,
    select_stage_b_pipeline,
    summarize_candidate_cells,
)
from .residual_training import ResidualFinalTrainingResult

_ROOT_ENTRIES = frozenset(
    {
        "config.yaml",
        "candidate_index.csv",
        "training.csv",
        "inner_predictions.csv",
        "inner_metrics.csv",
        "checkpoint_index.csv",
        "selected_generators.json",
        "frozen_pipelines.json",
        "models",
        "REPORT.md",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
)
_SOURCE_ENTRIES = _ROOT_ENTRIES - {"artifact_manifest.json", "CHECKSUMS.sha256"}
_CANDIDATE_FIELDS = (
    "candidate_id",
    "base_channels",
    "prediction_type",
    "beta_schedule",
    "bottleneck_attention",
    "spectral_weight",
    "low_pass_weight",
    "parameter_count",
    "config_sha256",
)
_TRAINING_FIELDS = (
    "role",
    "outer_domain",
    "query_domain",
    "candidate_id",
    "training_seed",
    "epochs",
    "fit_specimen_ids",
    "fit_dataset_ids",
    "target_state_sha256",
    "split_sha256",
    "epoch_losses",
    "sample_count",
    "batch_count",
    "response_read_count",
    "checkpoint_scientific_digest",
    "state_dict_sha256",
    "feature_bundle_sha256",
    "training_state_sha256",
    "evaluation_state_sha256",
    "run_state_sha256",
    "checkpoint_retained",
    "test_scale_override",
)
_PREDICTION_FIELDS = (
    "stage",
    "outer_domain",
    "query_domain",
    "candidate_id",
    "training_seed",
    "specimen_id",
    "dataset_id",
    "target",
    "prediction",
    "accepted_proposals",
    "proposed_variants",
    "checkpoint_scientific_digest",
    "prediction_sha256",
    "evaluation_state_sha256",
)
_METRIC_FIELDS = (
    "stage",
    "outer_domain",
    "candidate_id",
    "training_seeds",
    "domain_mae",
    "mean_mae",
    "worst_mae",
    "domain_sd",
    "objective",
    "overall_acceptance",
    "domain_acceptance",
    "eligible",
    "failed_domains",
    "oof_count",
    "target_sha256",
    "prediction_sha256",
    "cell_state_sha256",
    "state_sha256",
)
_CHECKPOINT_FIELDS = (
    "role",
    "outer_domain",
    "query_domain",
    "candidate_id",
    "training_seed",
    "split_sha256",
    "checkpoint_scientific_digest",
    "state_dict_sha256",
    "weights_path",
    "weights_bytes",
    "weights_sha256",
    "metadata_path",
    "metadata_bytes",
    "metadata_sha256",
)
_PARAMETER_COUNTS = {False: 2_471_747, True: 10_128_515}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_V3_CODE_DEPENDENCIES = (
    "src/cmc_bbdm/cpb_v3/artifacts.py",
    "src/cmc_bbdm/cpb_v3/config.py",
    "src/cmc_bbdm/cpb_v3/data.py",
    "src/cmc_bbdm/cpb_v3/embeddings.py",
    "src/cmc_bbdm/cpb_v3/morphology.py",
    "src/cmc_bbdm/cpb_v3/pipeline.py",
)
_SELECTED_RECORD_FIELDS = frozenset(
    {
        "outer_domain",
        "stage_a_sha256",
        "finalists",
        "selection_sha256",
        "best_residual",
        "best_incumbent",
        "incumbents",
        "residual_improvement",
        "residual_promoted",
        "ensemble",
        "ensemble_promoted",
        "selected_pipeline",
        "selected_components",
        "final_checkpoint_sha256",
    }
)
_FROZEN_RECORD_FIELDS = _SELECTED_RECORD_FIELDS | {
    "outer_run_sha256",
    "stage_a_run_sha256",
    "stage_b_run_sha256",
    "outer_evaluation_started",
}


class ResidualArtifactError(ValueError):
    """Raised when residual pre-outer artifacts are incomplete or altered."""


@dataclass(frozen=True, slots=True)
class D8ValidatedResidualSearchPackage:
    outer_evaluation_count: int
    test_scale_override: bool
    training_count: int
    prediction_count: int
    checkpoint_count: int
    pipeline_count: int
    scientific_digest: str
    output_tree_sha256: str
    artifact_manifest_sha256: str


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_regular_file(path)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(repr(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_cell(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and set(value) <= _SHA256_CHARACTERS
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
        newline="\n",
    )


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise ResidualArtifactError(f"{path.name} row schema changed")
            writer.writerow(dict(row))


def _read_csv(path: Path, fields: Sequence[str]) -> tuple[dict[str, str], ...]:
    try:
        payload = _regular_file(path).decode("utf-8")
        reader = csv.DictReader(io.StringIO(payload, newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ResidualArtifactError(f"{path.name} schema changed")
        rows = tuple(dict(row) for row in reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ResidualArtifactError(f"{path.name} cannot be decoded") from error
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ResidualArtifactError(f"{path.name} row width changed")
    return rows


def _read_json(path: Path, *, label: str) -> object:
    try:
        return json.loads(_regular_file(path).decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResidualArtifactError(f"{label} cannot be decoded") from error


def _regular_file(path: Path, *, maximum_bytes: int = 128 << 20) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ResidualArtifactError(f"artifact file is unavailable: {path.name}") from error
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size <= 0
            or info.st_size > maximum_bytes
        ):
            raise ResidualArtifactError(f"artifact file is not regular: {path.name}")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ResidualArtifactError(
                    f"artifact file is too large: {path.name}"
                )
        payload = b"".join(chunks)
        if len(payload) != info.st_size:
            raise ResidualArtifactError(
                f"artifact file changed while reading: {path.name}"
            )
        return payload
    except OSError as error:
        raise ResidualArtifactError(
            f"artifact file cannot be read: {path.name}"
        ) from error
    finally:
        os.close(descriptor)


def _package_records(root: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in {"artifact_manifest.json", "CHECKSUMS.sha256"}:
            continue
        if path.is_symlink():
            raise ResidualArtifactError("artifact package contains a symlink")
        if path.is_dir():
            if relative != "models":
                raise ResidualArtifactError("artifact package contains an unknown directory")
            continue
        payload = _regular_file(path)
        records[relative] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return records


def _tree_sha256(records: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _checksum_payload(root: Path) -> bytes:
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    return "".join(
        f"{hashlib.sha256(_regular_file(root / name)).hexdigest()}  {name}\n"
        for name in names
    ).encode("ascii")


def _execution_provenance(
    output: Path,
    *,
    config: ResidualDiffusionConfig,
    project_root: Path,
) -> dict[str, object]:
    d8_root = project_root / "src/cmc_bbdm/cpb_diffusion_marginalization"
    code_paths = tuple(
        sorted(
            path.relative_to(project_root).as_posix()
            for path in d8_root.glob("*.py")
        )
    ) + _V3_CODE_DEPENDENCIES
    if len(set(code_paths)) != len(code_paths):
        raise ResidualArtifactError("execution code roster is not unique")
    code: dict[str, dict[str, object]] = {}
    for relative in code_paths:
        payload = _regular_file(project_root / relative, maximum_bytes=16 << 20)
        code[relative] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    training = _read_csv(output / "training.csv", _TRAINING_FIELDS)
    split_records = {
        (
            row["outer_domain"],
            row["query_domain"],
            row["split_sha256"],
        )
        for row in training
    }
    return {
        "runtime": dict(config.runtime),
        "sources": {
            key: {"path": value.path, "sha256": value.sha256}
            for key, value in config.sources.items()
        },
        "code": code,
        "splits": [
            {
                "outer_domain": outer,
                "query_domain": query,
                "split_sha256": split,
            }
            for outer, query, split in sorted(split_records)
        ],
    }


def _scientific_digest(
    evidence_digest: str,
    execution_provenance: Mapping[str, object],
) -> str:
    return _canonical_sha256(
        {
            "evidence_digest": evidence_digest,
            "execution_provenance": execution_provenance,
        }
    )


def _selection_payload(
    *,
    outer_domain: str,
    stage_a: object,
    selection: object,
    final_training_sha256: Sequence[str],
) -> dict[str, object]:
    ensemble = selection.ensemble
    return {
        "outer_domain": outer_domain,
        "stage_a_sha256": stage_a.state_sha256,
        "finalists": list(stage_a.finalists),
        "selection_sha256": selection.state_sha256,
        "best_residual": (
            None
            if selection.best_residual is None
            else {
                "candidate_id": selection.best_residual.candidate_id,
                "objective": selection.best_residual.objective,
                "state_sha256": selection.best_residual.state_sha256,
            }
        ),
        "best_incumbent": {
            "pipeline_id": selection.best_incumbent.pipeline_id,
            "objective": selection.best_incumbent_objective,
            "state_sha256": selection.best_incumbent.state_sha256,
        },
        "incumbents": [
            {
                "pipeline_id": value.pipeline_id,
                "evidence_sha256": value.evidence_sha256,
                "state_sha256": value.state_sha256,
            }
            for value in selection.incumbents
        ],
        "residual_improvement": selection.residual_improvement,
        "residual_promoted": selection.residual_promoted,
        "ensemble": (
            None
            if ensemble is None
            else {
                "weights": ensemble.weights.tolist(),
                "best_member_objective": ensemble.best_member_objective,
                "objective": ensemble.objective,
                "objective_gain": ensemble.objective_gain,
                "state_sha256": ensemble.state_sha256,
            }
        ),
        "ensemble_promoted": selection.ensemble_promoted,
        "selected_pipeline": selection.selected_pipeline,
        "selected_components": list(selection.selected_components),
        "final_checkpoint_sha256": list(final_training_sha256),
    }


class ResidualArtifactRecorder:
    """Stream cell evidence and retained checkpoints into one isolated source."""

    def __init__(
        self,
        root: str | Path,
        *,
        config: ResidualDiffusionConfig,
        config_path: str | Path,
    ) -> None:
        if type(config) is not ResidualDiffusionConfig:
            raise TypeError("exact ResidualDiffusionConfig is required")
        if config.outer_evaluation_count != 0:
            raise ResidualArtifactError("outer evaluation count changed")
        source_config = Path(config_path).resolve(strict=True)
        if hashlib.sha256(source_config.read_bytes()).hexdigest() != config.config_sha256:
            raise ResidualArtifactError("residual config bytes changed")
        output = Path(root)
        if output.exists():
            raise ResidualArtifactError("artifact recorder target already exists")
        output.mkdir(parents=True)
        (output / "models").mkdir()
        self._root = output
        self._config = config
        self._config_path = source_config
        self._training_rows: list[dict[str, Any]] = []
        self._prediction_rows: list[dict[str, Any]] = []
        self._checkpoint_rows: list[dict[str, Any]] = []
        self._cell_states: set[str] = set()
        self._final_states: set[str] = set()
        self._metric_rows: list[dict[str, Any]] = []
        self._selected_generators: list[dict[str, Any]] = []
        self._frozen_pipelines: list[dict[str, Any]] = []
        self._outer_states: set[str] = set()
        self._finalized = False

    @property
    def root(self) -> Path:
        return self._root

    @property
    def training_rows(self) -> tuple[MappingProxyType[str, Any], ...]:
        return tuple(MappingProxyType(dict(row)) for row in self._training_rows)

    @property
    def prediction_rows(self) -> tuple[MappingProxyType[str, Any], ...]:
        return tuple(MappingProxyType(dict(row)) for row in self._prediction_rows)

    @property
    def checkpoint_rows(self) -> tuple[MappingProxyType[str, Any], ...]:
        return tuple(MappingProxyType(dict(row)) for row in self._checkpoint_rows)

    def record_cell(
        self,
        run: ResidualCellRun,
        *,
        retain_checkpoint: bool,
    ) -> None:
        if self._finalized:
            raise ResidualArtifactError("artifact recorder is already finalized")
        if type(run) is not ResidualCellRun or type(retain_checkpoint) is not bool:
            raise TypeError("exact residual cell run and retention flag are required")
        if retain_checkpoint != (run.cell.stage == "B"):
            raise ResidualArtifactError("checkpoint retention differs from stage contract")
        if run.state_sha256 in self._cell_states:
            raise ResidualArtifactError("duplicate residual cell run")
        training = run.training
        evaluation = run.evaluation
        role = "stage_a" if run.cell.stage == "A" else "stage_b"
        checkpoint = training.checkpoint
        checkpoint_row: dict[str, Any] | None = None
        if retain_checkpoint:
            basename = "__".join(
                (
                    role,
                    run.cell.outer_domain,
                    run.cell.query_domain,
                    run.cell.candidate_id,
                    str(run.cell.training_seed),
                )
            )
            prefix = self._root / "models" / basename
            weights, metadata = save_residual_checkpoint(prefix, checkpoint)
            checkpoint_row = {
                "role": role,
                "outer_domain": run.cell.outer_domain,
                "query_domain": run.cell.query_domain,
                "candidate_id": run.cell.candidate_id,
                "training_seed": run.cell.training_seed,
                "split_sha256": training.split_sha256,
                "checkpoint_scientific_digest": checkpoint.scientific_digest,
                "state_dict_sha256": checkpoint.state_dict_sha256,
                "weights_path": weights.relative_to(self._root).as_posix(),
                "weights_bytes": weights.stat().st_size,
                "weights_sha256": _sha256_file(weights),
                "metadata_path": metadata.relative_to(self._root).as_posix(),
                "metadata_bytes": metadata.stat().st_size,
                "metadata_sha256": _sha256_file(metadata),
            }
        training_row = {
            "role": role,
            "outer_domain": run.cell.outer_domain,
            "query_domain": run.cell.query_domain,
            "candidate_id": run.cell.candidate_id,
            "training_seed": run.cell.training_seed,
            "epochs": training.epochs,
            "fit_specimen_ids": _json_cell(training.fit_specimen_ids),
            "fit_dataset_ids": _json_cell(training.fit_dataset_ids),
            "target_state_sha256": training.target_state_sha256,
            "split_sha256": training.split_sha256,
            "epoch_losses": _json_cell(
                [asdict(value) for value in training.epoch_losses]
            ),
            "sample_count": training.sample_count,
            "batch_count": training.batch_count,
            "response_read_count": training.response_read_count,
            "checkpoint_scientific_digest": checkpoint.scientific_digest,
            "state_dict_sha256": checkpoint.state_dict_sha256,
            "feature_bundle_sha256": run.feature_bundle_sha256,
            "training_state_sha256": training.state_sha256,
            "evaluation_state_sha256": evaluation.state_sha256,
            "run_state_sha256": run.state_sha256,
            "checkpoint_retained": retain_checkpoint,
            "test_scale_override": training.test_scale_override,
        }
        prediction_rows = [
            {
                "stage": run.cell.stage,
                "outer_domain": run.cell.outer_domain,
                "query_domain": run.cell.query_domain,
                "candidate_id": run.cell.candidate_id,
                "training_seed": run.cell.training_seed,
                "specimen_id": specimen_id,
                "dataset_id": run.cell.query_domain,
                "target": float(target),
                "prediction": float(prediction),
                "accepted_proposals": evaluation.accepted_proposals,
                "proposed_variants": evaluation.proposed_variants,
                "checkpoint_scientific_digest": checkpoint.scientific_digest,
                "prediction_sha256": evaluation.prediction_sha256,
                "evaluation_state_sha256": evaluation.state_sha256,
            }
            for specimen_id, target, prediction in zip(
                evaluation.specimen_ids,
                evaluation.targets,
                evaluation.predictions,
                strict=True,
            )
        ]
        self._training_rows.append(training_row)
        self._prediction_rows.extend(prediction_rows)
        if checkpoint_row is not None:
            self._checkpoint_rows.append(checkpoint_row)
        self._cell_states.add(run.state_sha256)

    def record_final(self, result: ResidualFinalTrainingResult) -> None:
        if self._finalized:
            raise ResidualArtifactError("artifact recorder is already finalized")
        if type(result) is not ResidualFinalTrainingResult:
            raise TypeError("exact ResidualFinalTrainingResult is required")
        expected_domains = tuple(
            domain for domain in DOMAIN_ORDER if domain != result.outer_domain
        )
        if (
            result.outer_domain not in DOMAIN_ORDER
            or result.candidate_id not in self._config.candidate_ids
            or result.seed not in self._config.training_seeds
            or tuple(dict.fromkeys(result.fit_dataset_ids)) != expected_domains
            or result.response_read_count != 0
        ):
            raise ResidualArtifactError("final checkpoint authority changed")
        if result.state_sha256 in self._final_states:
            raise ResidualArtifactError("duplicate final training result")
        checkpoint = result.checkpoint
        basename = "__".join(
            (
                "final",
                result.outer_domain,
                result.candidate_id,
                str(result.seed),
            )
        )
        weights, metadata = save_residual_checkpoint(
            self._root / "models" / basename,
            checkpoint,
        )
        self._training_rows.append(
            {
                "role": "final",
                "outer_domain": result.outer_domain,
                "query_domain": "",
                "candidate_id": result.candidate_id,
                "training_seed": result.seed,
                "epochs": result.epochs,
                "fit_specimen_ids": _json_cell(result.fit_specimen_ids),
                "fit_dataset_ids": _json_cell(result.fit_dataset_ids),
                "target_state_sha256": result.target_state_sha256,
                "split_sha256": result.split_sha256,
                "epoch_losses": _json_cell(
                    [asdict(value) for value in result.epoch_losses]
                ),
                "sample_count": result.sample_count,
                "batch_count": result.batch_count,
                "response_read_count": result.response_read_count,
                "checkpoint_scientific_digest": checkpoint.scientific_digest,
                "state_dict_sha256": checkpoint.state_dict_sha256,
                "feature_bundle_sha256": "",
                "training_state_sha256": result.state_sha256,
                "evaluation_state_sha256": "",
                "run_state_sha256": result.state_sha256,
                "checkpoint_retained": True,
                "test_scale_override": result.test_scale_override,
            }
        )
        self._checkpoint_rows.append(
            {
                "role": "final",
                "outer_domain": result.outer_domain,
                "query_domain": "",
                "candidate_id": result.candidate_id,
                "training_seed": result.seed,
                "split_sha256": result.split_sha256,
                "checkpoint_scientific_digest": checkpoint.scientific_digest,
                "state_dict_sha256": checkpoint.state_dict_sha256,
                "weights_path": weights.relative_to(self._root).as_posix(),
                "weights_bytes": weights.stat().st_size,
                "weights_sha256": _sha256_file(weights),
                "metadata_path": metadata.relative_to(self._root).as_posix(),
                "metadata_bytes": metadata.stat().st_size,
                "metadata_sha256": _sha256_file(metadata),
            }
        )
        self._final_states.add(result.state_sha256)

    def record_outer(self, result: ResidualOuterSearchRun) -> None:
        if self._finalized:
            raise ResidualArtifactError("artifact recorder is already finalized")
        if type(result) is not ResidualOuterSearchRun:
            raise TypeError("exact ResidualOuterSearchRun is required")
        if result.state_sha256 in self._outer_states:
            raise ResidualArtifactError("duplicate outer search result")
        summaries = (*result.stage_a.summaries, *result.selection.candidate_summaries)
        for summary in summaries:
            self._metric_rows.append(
                {
                    "stage": summary.stage,
                    "outer_domain": summary.outer_domain,
                    "candidate_id": summary.candidate_id,
                    "training_seeds": _json_cell(summary.training_seeds),
                    "domain_mae": _json_cell(dict(summary.domain_mae)),
                    "mean_mae": summary.mean_mae,
                    "worst_mae": summary.worst_mae,
                    "domain_sd": summary.domain_sd,
                    "objective": summary.objective,
                    "overall_acceptance": summary.overall_acceptance,
                    "domain_acceptance": _json_cell(
                        dict(summary.domain_acceptance)
                    ),
                    "eligible": summary.eligible,
                    "failed_domains": _json_cell(summary.failed_domains),
                    "oof_count": len(summary.oof_specimen_ids),
                    "target_sha256": _array_sha256(summary.oof_targets),
                    "prediction_sha256": _array_sha256(summary.oof_predictions),
                    "cell_state_sha256": _json_cell(summary.cell_state_sha256),
                    "state_sha256": summary.state_sha256,
                }
            )
        selected_payload = _selection_payload(
            outer_domain=result.outer_domain,
            stage_a=result.stage_a,
            selection=result.selection,
            final_training_sha256=result.final_training_sha256,
        )
        self._selected_generators.append(selected_payload)
        self._frozen_pipelines.append(
            {
                **selected_payload,
                "outer_run_sha256": result.state_sha256,
                "stage_a_run_sha256": list(result.stage_a_run_sha256),
                "stage_b_run_sha256": list(result.stage_b_run_sha256),
                "outer_evaluation_started": False,
            }
        )
        self._outer_states.add(result.state_sha256)

    def finalize_source(
        self,
        *,
        test_scale_override: bool,
        expected_outer_domains: tuple[str, ...] | None = None,
    ) -> Path:
        if self._finalized:
            raise ResidualArtifactError("artifact recorder is already finalized")
        if type(test_scale_override) is not bool:
            raise TypeError("test scale flag must be boolean")
        if expected_outer_domains is not None:
            worker_assignments = tuple(
                DOMAIN_ORDER[index : index + 2]
                for index in range(0, len(DOMAIN_ORDER), 2)
            )
            if (
                type(expected_outer_domains) is not tuple
                or expected_outer_domains not in worker_assignments
            ):
                raise ResidualArtifactError("worker outer roster is not registered")
            expected_counts = {
                (role, outer): count
                for outer in expected_outer_domains
                for role, count in (("stage_a", 40), ("stage_b", 30))
            }
            observed_counts = {
                key: sum(
                    row["role"] == key[0] and row["outer_domain"] == key[1]
                    for row in self._training_rows
                )
                for key in expected_counts
            }
            if (
                observed_counts != expected_counts
                or any(
                    row["outer_domain"] not in expected_outer_domains
                    for row in self._training_rows
                )
            ):
                raise ResidualArtifactError("worker training roster is incomplete")
            final_counts = {
                outer: sum(
                    row["role"] == "final" and row["outer_domain"] == outer
                    for row in self._training_rows
                )
                for outer in expected_outer_domains
            }
            if any(count not in {0, 3} for count in final_counts.values()):
                raise ResidualArtifactError("worker final roster is incomplete")
        observed_flags = {
            bool(row["test_scale_override"]) for row in self._training_rows
        }
        if observed_flags != {test_scale_override}:
            raise ResidualArtifactError("training rows use a different scale contract")
        if not test_scale_override and expected_outer_domains is None:
            roles = [str(row["role"]) for row in self._training_rows]
            if roles.count("stage_a") != 240 or roles.count("stage_b") != 180:
                raise ResidualArtifactError("formal search training roster is incomplete")
        config_target = self._root / "config.yaml"
        config_target.write_bytes(self._config_path.read_bytes())
        candidate_rows = tuple(
            {
                "candidate_id": candidate_id,
                "base_channels": candidate.base_channels,
                "prediction_type": candidate.prediction_type,
                "beta_schedule": candidate.beta_schedule,
                "bottleneck_attention": candidate.bottleneck_attention,
                "spectral_weight": candidate.spectral_weight,
                "low_pass_weight": candidate.low_pass_weight,
                "parameter_count": _PARAMETER_COUNTS[
                    candidate.bottleneck_attention
                ],
                "config_sha256": self._config.config_sha256,
            }
            for candidate_id in self._config.candidate_ids
            for candidate in (self._config.candidate(candidate_id),)
        )
        _write_csv(
            self._root / "candidate_index.csv",
            _CANDIDATE_FIELDS,
            candidate_rows,
        )
        _write_csv(
            self._root / "training.csv",
            _TRAINING_FIELDS,
            self._training_rows,
        )
        _write_csv(
            self._root / "inner_predictions.csv",
            _PREDICTION_FIELDS,
            self._prediction_rows,
        )
        _write_csv(
            self._root / "inner_metrics.csv",
            _METRIC_FIELDS,
            self._metric_rows,
        )
        _write_csv(
            self._root / "checkpoint_index.csv",
            _CHECKPOINT_FIELDS,
            self._checkpoint_rows,
        )
        ordered_selected = sorted(
            self._selected_generators,
            key=lambda value: DOMAIN_ORDER.index(str(value["outer_domain"])),
        )
        ordered_frozen = sorted(
            self._frozen_pipelines,
            key=lambda value: DOMAIN_ORDER.index(str(value["outer_domain"])),
        )
        if expected_outer_domains is not None and (
            tuple(str(row["outer_domain"]) for row in ordered_selected)
            != expected_outer_domains
            or tuple(str(row["outer_domain"]) for row in ordered_frozen)
            != expected_outer_domains
        ):
            raise ResidualArtifactError("worker pipeline roster is incomplete")
        if (
            expected_outer_domains is None
            and not test_scale_override
            and (len(ordered_selected) != 6 or len(ordered_frozen) != 6)
        ):
            raise ResidualArtifactError("formal outer selection roster is incomplete")
        common = {
            "schema_version": 1,
            "config_sha256": self._config.config_sha256,
            "outer_evaluation_count": 0,
            "test_scale_override": test_scale_override,
        }
        _write_json(
            self._root / "selected_generators.json",
            {
                **common,
                "scope": "cpb_d8_residual_selected_generators",
                "selections": ordered_selected,
            },
        )
        _write_json(
            self._root / "frozen_pipelines.json",
            {
                **common,
                "scope": "cpb_d8_residual_frozen_pipelines",
                "selections": ordered_frozen,
            },
        )
        (self._root / "REPORT.md").write_text(
            "# D8 Residual Diffusion Pre-Outer Search\n\n"
            f"- outer_evaluation_count: 0\n"
            f"- test_scale_override: {str(test_scale_override).lower()}\n",
            encoding="ascii",
            newline="\n",
        )
        self._finalized = True
        return self._root


def _root_entry_set(root: Path, *, expected: frozenset[str]) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ResidualArtifactError("residual artifact root is not a directory")
    entries = tuple(root.iterdir())
    if {path.name for path in entries} != set(expected):
        raise ResidualArtifactError("residual artifact root entries changed")
    for path in entries:
        info = path.stat(follow_symlinks=False)
        if path.is_symlink():
            raise ResidualArtifactError("residual artifact root contains a symlink")
        if path.name == "models":
            if not stat.S_ISDIR(info.st_mode):
                raise ResidualArtifactError("models entry is not a directory")
        elif not stat.S_ISREG(info.st_mode):
            raise ResidualArtifactError("artifact root entry is not a regular file")


def _boolean(value: str, *, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ResidualArtifactError(f"{label} is not canonical boolean")


def _integer(value: str, *, label: str, minimum: int = 0) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ResidualArtifactError(f"{label} is not an integer") from error
    if str(result) != value or result < minimum:
        raise ResidualArtifactError(f"{label} is not canonical")
    return result


def _finite(value: str, *, label: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ResidualArtifactError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise ResidualArtifactError(f"{label} is not finite")
    return result


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _single_cell_value(
    rows: Sequence[Mapping[str, str]],
    field: str,
) -> str:
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise ResidualArtifactError(f"prediction cell {field} changed")
    return next(iter(values))


def _recompute_metric_summaries(
    *,
    predictions: Sequence[Mapping[str, str]],
    metrics: Sequence[Mapping[str, str]],
    training_by_key: Mapping[tuple[str, str, str, str, int], Mapping[str, str]],
    config: ResidualDiffusionConfig,
) -> tuple[
    dict[tuple[str, str, str], ResidualCandidateSummary],
    dict[tuple[str, str, str], tuple[ResidualCellEvaluation, ...]],
]:
    grouped_rows: dict[
        tuple[str, str, str, str, int], list[Mapping[str, str]]
    ] = {}
    for row in predictions:
        key = (
            row["stage"],
            row["outer_domain"],
            row["query_domain"],
            row["candidate_id"],
            _integer(row["training_seed"], label="prediction seed"),
        )
        grouped_rows.setdefault(key, []).append(row)

    evaluations: dict[
        tuple[str, str, str], list[ResidualCellEvaluation]
    ] = {}
    observed_training_keys: set[tuple[str, str, str, str, int]] = set()
    for key, rows in grouped_rows.items():
        stage, outer, query, candidate_id, seed = key
        specimen_ids = tuple(row["specimen_id"] for row in rows)
        if len(set(specimen_ids)) != len(specimen_ids):
            raise ResidualArtifactError("duplicate prediction specimen identity")
        cell = ResidualSearchCell(stage, outer, query, candidate_id, seed)
        checkpoint_sha256 = _single_cell_value(
            rows, "checkpoint_scientific_digest"
        )
        prediction_sha256 = _single_cell_value(rows, "prediction_sha256")
        accepted = _integer(
            _single_cell_value(rows, "accepted_proposals"),
            label="accepted proposals",
        )
        proposed = _integer(
            _single_cell_value(rows, "proposed_variants"),
            label="proposed variants",
            minimum=1,
        )
        evaluation = ResidualCellEvaluation(
            cell=cell,
            specimen_ids=specimen_ids,
            targets=np.asarray(
                [_finite(row["target"], label="prediction target") for row in rows],
                dtype=np.float64,
            ),
            predictions=np.asarray(
                [_finite(row["prediction"], label="prediction") for row in rows],
                dtype=np.float64,
            ),
            accepted_proposals=accepted,
            proposed_variants=proposed,
            checkpoint_sha256=checkpoint_sha256,
            prediction_sha256=prediction_sha256,
        )
        declared_evaluation = _single_cell_value(rows, "evaluation_state_sha256")
        role = "stage_a" if stage == "A" else "stage_b"
        training_key = (role, outer, query, candidate_id, seed)
        training = training_by_key.get(training_key)
        if training is None:
            raise ResidualArtifactError("prediction cell lacks training authority")
        expected_run = _canonical_sha256(
            {
                "cell_sha256": cell.state_sha256,
                "training_sha256": training["training_state_sha256"],
                "feature_bundle_sha256": training["feature_bundle_sha256"],
                "evaluation_sha256": evaluation.state_sha256,
            }
        )
        if (
            declared_evaluation != evaluation.state_sha256
            or training["evaluation_state_sha256"] != evaluation.state_sha256
            or training["checkpoint_scientific_digest"] != checkpoint_sha256
            or training["run_state_sha256"] != expected_run
        ):
            raise ResidualArtifactError("prediction cell state changed")
        observed_training_keys.add(training_key)
        evaluations.setdefault((stage, outer, candidate_id), []).append(evaluation)

    expected_training_keys = {
        key for key in training_by_key if key[0] in {"stage_a", "stage_b"}
    }
    if observed_training_keys != expected_training_keys:
        raise ResidualArtifactError("training and prediction cell rosters differ")

    metric_by_key: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for row in metrics:
        key = (row["stage"], row["outer_domain"], row["candidate_id"])
        if key in metric_by_key:
            raise ResidualArtifactError("duplicate metric summary identity")
        metric_by_key[key] = row
    complete_keys = {
        key
        for key, values in evaluations.items()
        if len(values) == (5 if key[0] == "A" else 15)
    }
    if set(metric_by_key) != complete_keys:
        raise ResidualArtifactError("metric summary roster changed")
    recomputed: dict[tuple[str, str, str], ResidualCandidateSummary] = {}
    for key, row in metric_by_key.items():
        try:
            summary = summarize_candidate_cells(
                tuple(evaluations[key]), config=config, stage=key[0]
            )
        except ResidualSearchError as error:
            raise ResidualArtifactError("metric summary cannot be recomputed") from error
        expected = {
            "stage": summary.stage,
            "outer_domain": summary.outer_domain,
            "candidate_id": summary.candidate_id,
            "training_seeds": _json_cell(summary.training_seeds),
            "domain_mae": _json_cell(dict(summary.domain_mae)),
            "mean_mae": str(summary.mean_mae),
            "worst_mae": str(summary.worst_mae),
            "domain_sd": str(summary.domain_sd),
            "objective": str(summary.objective),
            "overall_acceptance": str(summary.overall_acceptance),
            "domain_acceptance": _json_cell(dict(summary.domain_acceptance)),
            "eligible": str(summary.eligible),
            "failed_domains": _json_cell(summary.failed_domains),
            "oof_count": str(len(summary.oof_specimen_ids)),
            "target_sha256": _array_sha256(summary.oof_targets),
            "prediction_sha256": _array_sha256(summary.oof_predictions),
            "cell_state_sha256": _json_cell(summary.cell_state_sha256),
            "state_sha256": summary.state_sha256,
        }
        if dict(row) != expected:
            raise ResidualArtifactError("metric summary differs from predictions")
        recomputed[key] = summary
    return (
        recomputed,
        {key: tuple(values) for key, values in evaluations.items()},
    )


def _json_finite(value: object, *, label: str) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise ResidualArtifactError(f"selection record {label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ResidualArtifactError(f"selection record {label} is not finite")
    return result


def _validate_selection_records(
    selected: Mapping[str, object],
    frozen: Mapping[str, object],
    *,
    config: ResidualDiffusionConfig,
) -> None:
    selected_rows = selected["selections"]
    frozen_rows = frozen["selections"]
    assert isinstance(selected_rows, list) and isinstance(frozen_rows, list)
    if len(selected_rows) != len(frozen_rows):
        raise ResidualArtifactError("selection record rosters differ")
    observed_outers: list[str] = []
    for selected_row, frozen_row in zip(
        selected_rows,
        frozen_rows,
        strict=True,
    ):
        if (
            type(selected_row) is not dict
            or set(selected_row) != _SELECTED_RECORD_FIELDS
            or type(frozen_row) is not dict
            or set(frozen_row) != _FROZEN_RECORD_FIELDS
            or any(frozen_row[field] != selected_row[field] for field in selected_row)
        ):
            raise ResidualArtifactError("selection record schema changed")
        outer = selected_row["outer_domain"]
        finalists = selected_row["finalists"]
        pipeline = selected_row["selected_pipeline"]
        components = selected_row["selected_components"]
        final_states = selected_row["final_checkpoint_sha256"]
        if (
            type(outer) is not str
            or outer not in DOMAIN_ORDER
            or type(finalists) is not list
            or len(finalists) != config.finalists_per_outer
            or len(set(finalists)) != len(finalists)
            or any(value not in config.candidate_ids for value in finalists)
            or pipeline not in {"INCUMBENT", "RESIDUAL", "ENSEMBLE"}
            or type(components) is not list
            or not components
            or type(final_states) is not list
            or any(not _valid_sha256(value) for value in final_states)
            or any(
                not _valid_sha256(selected_row[field])
                for field in ("stage_a_sha256", "selection_sha256")
            )
        ):
            raise ResidualArtifactError("selection record identity changed")
        expected_final_count = 0 if pipeline == "INCUMBENT" else 3
        if len(final_states) != expected_final_count:
            raise ResidualArtifactError("selection record final roster changed")
        if pipeline == "INCUMBENT":
            valid_components = (
                len(components) == 1 and components[0] in {"PILOT", "B0"}
            )
        elif pipeline == "RESIDUAL":
            valid_components = len(components) == 1 and components[0] in finalists
        else:
            valid_components = (
                len(components) == 2
                and components[0] in finalists
                and components[1] in {"PILOT", "B0"}
            )
        if not valid_components:
            raise ResidualArtifactError("selection record components changed")
        best_residual = selected_row["best_residual"]
        if best_residual is not None and (
            type(best_residual) is not dict
            or set(best_residual) != {"candidate_id", "objective", "state_sha256"}
            or best_residual["candidate_id"] not in finalists
            or not _valid_sha256(best_residual["state_sha256"])
        ):
            raise ResidualArtifactError("selection record residual changed")
        if best_residual is not None:
            _json_finite(best_residual["objective"], label="residual objective")
        best_incumbent = selected_row["best_incumbent"]
        incumbents = selected_row["incumbents"]
        if (
            type(best_incumbent) is not dict
            or set(best_incumbent) != {"pipeline_id", "objective", "state_sha256"}
            or best_incumbent["pipeline_id"] not in {"PILOT", "B0"}
            or not _valid_sha256(best_incumbent["state_sha256"])
            or type(incumbents) is not list
            or len(incumbents) != 2
            or {value.get("pipeline_id") for value in incumbents if type(value) is dict}
            != {"PILOT", "B0"}
        ):
            raise ResidualArtifactError("selection record incumbent changed")
        _json_finite(best_incumbent["objective"], label="incumbent objective")
        for incumbent in incumbents:
            if (
                type(incumbent) is not dict
                or set(incumbent)
                != {"pipeline_id", "evidence_sha256", "state_sha256"}
                or not _valid_sha256(incumbent["evidence_sha256"])
                or not _valid_sha256(incumbent["state_sha256"])
            ):
                raise ResidualArtifactError("selection record incumbent changed")
        if (
            type(selected_row["residual_promoted"]) is not bool
            or type(selected_row["ensemble_promoted"]) is not bool
            or selected_row["ensemble_promoted"]
            != (selected_row["ensemble"] is not None)
        ):
            raise ResidualArtifactError("selection record promotion changed")
        improvement = selected_row["residual_improvement"]
        if improvement is not None:
            _json_finite(improvement, label="residual improvement")
        ensemble = selected_row["ensemble"]
        if ensemble is not None:
            if (
                type(ensemble) is not dict
                or set(ensemble)
                != {
                    "weights",
                    "best_member_objective",
                    "objective",
                    "objective_gain",
                    "state_sha256",
                }
                or type(ensemble["weights"]) is not list
                or len(ensemble["weights"]) != 2
                or not _valid_sha256(ensemble["state_sha256"])
            ):
                raise ResidualArtifactError("selection record ensemble changed")
            for field in (
                "best_member_objective",
                "objective",
                "objective_gain",
            ):
                _json_finite(ensemble[field], label=f"ensemble {field}")
            weights = [
                _json_finite(value, label="ensemble weight")
                for value in ensemble["weights"]
            ]
            if any(value < 0.0 for value in weights) or not math.isclose(
                math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ResidualArtifactError("selection record ensemble weights changed")
        if (
            frozen_row["outer_evaluation_started"] is not False
            or not _valid_sha256(frozen_row["outer_run_sha256"])
            or type(frozen_row["stage_a_run_sha256"]) is not list
            or len(frozen_row["stage_a_run_sha256"]) != 40
            or type(frozen_row["stage_b_run_sha256"]) is not list
            or len(frozen_row["stage_b_run_sha256"]) != 30
            or any(
                not _valid_sha256(value)
                for value in (
                    *frozen_row["stage_a_run_sha256"],
                    *frozen_row["stage_b_run_sha256"],
                )
            )
        ):
            raise ResidualArtifactError("selection record frozen pipeline changed")
        observed_outers.append(outer)
    expected_order = [domain for domain in DOMAIN_ORDER if domain in observed_outers]
    if observed_outers != expected_order or len(set(observed_outers)) != len(
        observed_outers
    ):
        raise ResidualArtifactError("selection record outer order changed")


def _validate_selection_evidence(
    selected: Mapping[str, object],
    frozen: Mapping[str, object],
    *,
    evaluations: Mapping[
        tuple[str, str, str], tuple[ResidualCellEvaluation, ...]
    ],
    training_by_key: Mapping[
        tuple[str, str, str, str, int], Mapping[str, str]
    ],
    config: ResidualDiffusionConfig,
    project_root: Path,
    data: V3Data,
    test_scale_override: bool,
) -> None:
    selected_rows = selected["selections"]
    frozen_rows = frozen["selections"]
    assert isinstance(selected_rows, list) and isinstance(frozen_rows, list)
    if not selected_rows:
        return
    pilot = {
        value.outer_domain: value
        for value in load_pilot_incumbent_evidence(
            config,
            project_root=project_root,
        )
    }
    b0 = {
        value.outer_domain: value
        for value in load_b0_incumbent_evidence(
            data,
            config=config,
            project_root=project_root,
        )
    }
    for record, frozen_record in zip(selected_rows, frozen_rows, strict=True):
        assert isinstance(record, dict) and isinstance(frozen_record, dict)
        outer = str(record["outer_domain"])
        stage_a_evaluations = tuple(
            evaluation
            for candidate_id in config.candidate_ids
            for evaluation in evaluations.get(("A", outer, candidate_id), ())
        )
        if len(stage_a_evaluations) != 40:
            raise ResidualArtifactError("selection evidence lacks Stage-A summaries")
        try:
            promoted = promote_stage_a_outer(
                stage_a_evaluations,
                config=config,
                test_scale_override=test_scale_override,
            )
        except ResidualSearchError as error:
            raise ResidualArtifactError(
                "selection evidence Stage-A state changed"
            ) from error
        stage_b_evaluations = tuple(
            evaluation
            for candidate_id in promoted.finalists
            for evaluation in evaluations.get(("B", outer, candidate_id), ())
        )
        if len(stage_b_evaluations) != 30:
            raise ResidualArtifactError("selection evidence lacks Stage-B summaries")
        try:
            selection = select_stage_b_pipeline(
                stage_b_evaluations,
                incumbents=(pilot[outer], b0[outer]),
                finalists=promoted.finalists,
                config=config,
            )
        except (KeyError, ResidualSearchError, ValueError) as error:
            raise ResidualArtifactError(
                "selection evidence cannot be recomputed"
            ) from error
        stage_a_runs = [
            row["run_state_sha256"]
            for key, row in training_by_key.items()
            if key[0] == "stage_a" and key[1] == outer
        ]
        stage_b_runs = [
            row["run_state_sha256"]
            for key, row in training_by_key.items()
            if key[0] == "stage_b" and key[1] == outer
        ]
        final_rows = sorted(
            (
                (key, row)
                for key, row in training_by_key.items()
                if key[0] == "final" and key[1] == outer
            ),
            key=lambda value: config.training_seeds.index(value[0][4]),
        )
        expected_final = [row["training_state_sha256"] for _key, row in final_rows]
        expected_record = _selection_payload(
            outer_domain=outer,
            stage_a=promoted,
            selection=selection,
            final_training_sha256=expected_final,
        )
        if record != expected_record:
            raise ResidualArtifactError("selection evidence decision changed")
        if selection.best_residual is not None and final_rows and any(
            key[3] != selection.best_residual.candidate_id
            or key[4] != config.training_seeds[index]
            for index, (key, _row) in enumerate(final_rows)
        ):
            raise ResidualArtifactError("selection evidence final checkpoint changed")
        outer_state = _canonical_sha256(
            {
                "outer_domain": outer,
                "stage_a_sha256": promoted.state_sha256,
                "stage_a_run_sha256": tuple(stage_a_runs),
                "stage_b_run_sha256": tuple(stage_b_runs),
                "selection_sha256": selection.state_sha256,
                "final_training_sha256": tuple(expected_final),
                "outer_evaluation_count": 0,
            }
        )
        expected_frozen = {
            **expected_record,
            "outer_run_sha256": outer_state,
            "stage_a_run_sha256": stage_a_runs,
            "stage_b_run_sha256": stage_b_runs,
            "outer_evaluation_started": False,
        }
        if frozen_record != expected_frozen:
            raise ResidualArtifactError("selection evidence outer state changed")


def _validate_training_split_authorities(
    training_by_key: Mapping[
        tuple[str, str, str, str, int], Mapping[str, str]
    ],
    *,
    config: ResidualDiffusionConfig,
    project_root: Path,
    test_scale: bool,
) -> V3Data:
    exploration = load_d8_config(
        project_root / config.sources["exploration_config"].path,
        project_root=project_root,
    )
    v3_config = load_v3_config(
        project_root / exploration.sources["p1_config"].path,
        project_root=project_root,
    )
    data = load_data(v3_config, project_root)
    search_views = {
        outer: issue_search_view(data, outer_domain=outer, config=exploration)
        for outer in DOMAIN_ORDER
    }
    inner_folds = {
        (outer, query): issue_inner_fold(
            search_views[outer],
            query_domain=query,
        )
        for outer in DOMAIN_ORDER
        for query in DOMAIN_ORDER
        if query != outer
    }
    for key, row in training_by_key.items():
        role, outer, query, candidate_id, seed = key
        try:
            fit_ids = json.loads(row["fit_specimen_ids"])
            fit_domains = json.loads(row["fit_dataset_ids"])
            epoch_losses = json.loads(row["epoch_losses"])
        except json.JSONDecodeError as error:
            raise ResidualArtifactError("split authority JSON changed") from error
        if role == "final":
            authority = search_views[outer]
            expected_ids = list(authority.specimen_ids)
            expected_domains = list(authority.dataset_ids)
            expected_split = authority.state_sha256
            expected_epochs = 1 if test_scale else config.rerank_epochs
        else:
            authority = inner_folds[(outer, query)]
            expected_ids = list(authority.fit_specimen_ids)
            expected_domains = list(authority.fit_dataset_ids)
            expected_split = authority.state_sha256
            expected_epochs = (
                1
                if test_scale
                else (
                    config.screening_epochs
                    if role == "stage_a"
                    else config.rerank_epochs
                )
            )
        if (
            fit_ids != expected_ids
            or fit_domains != expected_domains
            or row["split_sha256"] != expected_split
            or _integer(row["epochs"], label="epochs", minimum=1)
            != expected_epochs
            or (role == "stage_a" and seed != config.screening_seed)
            or (role != "stage_a" and seed not in config.training_seeds)
        ):
            raise ResidualArtifactError("training split authority changed")
        if not isinstance(epoch_losses, list) or len(epoch_losses) != expected_epochs:
            raise ResidualArtifactError("training split authority losses changed")
        loss_keys = {
            "epoch",
            "total",
            "diffusion",
            "spectral",
            "low_pass",
            "sample_count",
            "batch_count",
        }
        for epoch, loss in enumerate(epoch_losses, start=1):
            if (
                type(loss) is not dict
                or set(loss) != loss_keys
                or type(loss["epoch"]) is not int
                or loss["epoch"] != epoch
                or type(loss["sample_count"]) is not int
                or loss["sample_count"] < 1
                or type(loss["batch_count"]) is not int
                or loss["batch_count"] < 1
                or any(
                    type(loss[field]) is not float
                    or not math.isfinite(loss[field])
                    or loss[field] < 0.0
                    for field in ("total", "diffusion", "spectral", "low_pass")
                )
            ):
                raise ResidualArtifactError("training split authority losses changed")
        common = {
            "outer_domain": outer,
            "candidate_id": candidate_id,
            "seed": seed,
            "epochs": expected_epochs,
            "fit_specimen_ids": fit_ids,
            "fit_dataset_ids": fit_domains,
            "target_state_sha256": row["target_state_sha256"],
            "split_sha256": expected_split,
            "epoch_losses": epoch_losses,
            "checkpoint_scientific_digest": row[
                "checkpoint_scientific_digest"
            ],
            "sample_count": _integer(
                row["sample_count"], label="sample count", minimum=1
            ),
            "batch_count": _integer(
                row["batch_count"], label="batch count", minimum=1
            ),
            "response_read_count": 0,
            "test_scale_override": test_scale,
        }
        payload = common if role == "final" else {"query_domain": query, **common}
        expected_state = _canonical_sha256(payload)
        if (
            row["training_state_sha256"] != expected_state
            or (role == "final" and row["run_state_sha256"] != expected_state)
            or (role == "final" and row["feature_bundle_sha256"] != "")
            or (role == "final" and row["evaluation_state_sha256"] != "")
        ):
            raise ResidualArtifactError("training split authority state changed")
    return data


def _validate_semantics(
    output: Path,
    *,
    config: ResidualDiffusionConfig,
    project_root: Path,
) -> tuple[bool, int, int, int, int, str]:
    candidates = _read_csv(output / "candidate_index.csv", _CANDIDATE_FIELDS)
    training = _read_csv(output / "training.csv", _TRAINING_FIELDS)
    predictions = _read_csv(output / "inner_predictions.csv", _PREDICTION_FIELDS)
    metrics = _read_csv(output / "inner_metrics.csv", _METRIC_FIELDS)
    checkpoints = _read_csv(output / "checkpoint_index.csv", _CHECKPOINT_FIELDS)
    if len(candidates) != 8 or tuple(row["candidate_id"] for row in candidates) != tuple(
        config.candidate_ids
    ):
        raise ResidualArtifactError("candidate index roster changed")
    for row in candidates:
        candidate = config.candidate(row["candidate_id"])
        expected = {
            "candidate_id": candidate.candidate_id,
            "base_channels": str(candidate.base_channels),
            "prediction_type": candidate.prediction_type,
            "beta_schedule": candidate.beta_schedule,
            "bottleneck_attention": str(candidate.bottleneck_attention),
            "spectral_weight": str(candidate.spectral_weight),
            "low_pass_weight": str(candidate.low_pass_weight),
            "parameter_count": str(
                _PARAMETER_COUNTS[candidate.bottleneck_attention]
            ),
            "config_sha256": config.config_sha256,
        }
        if row != expected:
            raise ResidualArtifactError("candidate index values changed")
    scale_flags: set[bool] = set()
    training_keys: set[tuple[str, str, str, str, int]] = set()
    training_by_key: dict[
        tuple[str, str, str, str, int], Mapping[str, str]
    ] = {}
    for row in training:
        role = row["role"]
        if role not in {"stage_a", "stage_b", "final"}:
            raise ResidualArtifactError("training role changed")
        outer = row["outer_domain"]
        query = row["query_domain"]
        candidate_id = row["candidate_id"]
        seed = _integer(row["training_seed"], label="training seed")
        if (
            outer not in DOMAIN_ORDER
            or candidate_id not in config.candidate_ids
            or seed not in config.training_seeds
            or (role == "final") != (query == "")
            or (query and (query not in DOMAIN_ORDER or query == outer))
        ):
            raise ResidualArtifactError("training identity changed")
        key = (role, outer, query, candidate_id, seed)
        if key in training_keys:
            raise ResidualArtifactError("duplicate training identity")
        training_keys.add(key)
        training_by_key[key] = row
        retained = _boolean(row["checkpoint_retained"], label="retention flag")
        if retained != (role in {"stage_b", "final"}):
            raise ResidualArtifactError("checkpoint retention state changed")
        scale_flags.add(_boolean(row["test_scale_override"], label="scale flag"))
        if (
            _integer(row["epochs"], label="epochs", minimum=1) < 1
            or _integer(row["sample_count"], label="sample count", minimum=1) < 1
            or _integer(row["batch_count"], label="batch count", minimum=1) < 1
            or _integer(row["response_read_count"], label="response reads") != 0
            or any(
                not _valid_sha256(row[field])
                for field in (
                    "target_state_sha256",
                    "split_sha256",
                    "checkpoint_scientific_digest",
                    "state_dict_sha256",
                    "training_state_sha256",
                    "run_state_sha256",
                )
            )
            or (role != "final" and not _valid_sha256(row["feature_bundle_sha256"]))
            or (role != "final" and not _valid_sha256(row["evaluation_state_sha256"]))
        ):
            raise ResidualArtifactError("training evidence changed")
        try:
            fit_ids = json.loads(row["fit_specimen_ids"])
            fit_domains = json.loads(row["fit_dataset_ids"])
            losses = json.loads(row["epoch_losses"])
        except json.JSONDecodeError as error:
            raise ResidualArtifactError("training JSON cell changed") from error
        if (
            not isinstance(fit_ids, list)
            or not isinstance(fit_domains, list)
            or len(fit_ids) != len(fit_domains)
            or len(fit_ids) != _integer(
                row["sample_count"], label="sample count", minimum=1
            )
            or not isinstance(losses, list)
            or len(losses) != _integer(row["epochs"], label="epochs", minimum=1)
        ):
            raise ResidualArtifactError("training roster or losses changed")
    if len(scale_flags) != 1:
        raise ResidualArtifactError("training scale contract is mixed")
    test_scale = next(iter(scale_flags))
    data = _validate_training_split_authorities(
        training_by_key,
        config=config,
        project_root=project_root,
        test_scale=test_scale,
    )
    for row in predictions:
        if (
            row["stage"] not in {"A", "B"}
            or row["outer_domain"] not in DOMAIN_ORDER
            or row["query_domain"] not in DOMAIN_ORDER
            or row["dataset_id"] != row["query_domain"]
            or row["candidate_id"] not in config.candidate_ids
            or _integer(row["training_seed"], label="prediction seed")
            not in config.training_seeds
            or not row["specimen_id"]
            or not all(
                _valid_sha256(row[field])
                for field in (
                    "checkpoint_scientific_digest",
                    "prediction_sha256",
                    "evaluation_state_sha256",
                )
            )
        ):
            raise ResidualArtifactError("inner prediction identity changed")
        _finite(row["target"], label="prediction target")
        _finite(row["prediction"], label="prediction")
        accepted = _integer(row["accepted_proposals"], label="accepted proposals")
        proposed = _integer(
            row["proposed_variants"], label="proposed variants", minimum=1
        )
        if accepted > proposed:
            raise ResidualArtifactError("proposal counts changed")
    _summaries, evaluations = _recompute_metric_summaries(
        predictions=predictions,
        metrics=metrics,
        training_by_key=training_by_key,
        config=config,
    )
    checkpoint_keys: set[tuple[str, str, str, str, int]] = set()
    model_paths: set[str] = set()
    for row in checkpoints:
        role = row["role"]
        outer = row["outer_domain"]
        query = row["query_domain"]
        candidate_id = row["candidate_id"]
        seed = _integer(row["training_seed"], label="checkpoint seed")
        key = (role, outer, query, candidate_id, seed)
        if (
            role not in {"stage_b", "final"}
            or key in checkpoint_keys
            or key not in training_keys
            or not _valid_sha256(row["split_sha256"])
            or not _valid_sha256(row["checkpoint_scientific_digest"])
            or not _valid_sha256(row["state_dict_sha256"])
        ):
            raise ResidualArtifactError("checkpoint identity changed")
        checkpoint_keys.add(key)
        weights_path = row["weights_path"]
        metadata_path = row["metadata_path"]
        if (
            not weights_path.startswith("models/")
            or not metadata_path.startswith("models/")
            or Path(weights_path).is_absolute()
            or Path(metadata_path).is_absolute()
            or ".." in Path(weights_path).parts
            or ".." in Path(metadata_path).parts
        ):
            raise ResidualArtifactError("checkpoint path escaped models")
        weights = _regular_file(output / weights_path)
        metadata = _regular_file(output / metadata_path)
        if (
            len(weights)
            != _integer(row["weights_bytes"], label="checkpoint bytes", minimum=1)
            or hashlib.sha256(weights).hexdigest() != row["weights_sha256"]
            or len(metadata)
            != _integer(row["metadata_bytes"], label="metadata bytes", minimum=1)
            or hashlib.sha256(metadata).hexdigest() != row["metadata_sha256"]
        ):
            raise ResidualArtifactError("checkpoint file hash changed")
        loaded = load_residual_checkpoint(
            output / weights_path,
            output / metadata_path,
            candidate=config.candidate(candidate_id),
            config_sha256=config.config_sha256,
            split_sha256=row["split_sha256"],
        )
        if (
            loaded.training_seed != seed
            or loaded.scientific_digest != row["checkpoint_scientific_digest"]
            or loaded.state_dict_sha256 != row["state_dict_sha256"]
        ):
            raise ResidualArtifactError("checkpoint semantic state changed")
        model_paths.update((weights_path, metadata_path))
    retained_training_keys = {
        key
        for key, row in training_by_key.items()
        if _boolean(row["checkpoint_retained"], label="retention flag")
    }
    if checkpoint_keys != retained_training_keys:
        raise ResidualArtifactError("checkpoint and training rosters differ")
    observed_models = {
        path.relative_to(output).as_posix()
        for path in (output / "models").iterdir()
        if path.is_file()
    }
    if model_paths != observed_models:
        raise ResidualArtifactError("models directory differs from checkpoint index")
    selected = _read_json(
        output / "selected_generators.json", label="selected generators"
    )
    frozen = _read_json(output / "frozen_pipelines.json", label="frozen pipelines")
    common_keys = {
        "schema_version",
        "scope",
        "config_sha256",
        "outer_evaluation_count",
        "test_scale_override",
        "selections",
    }
    for payload, scope in (
        (selected, "cpb_d8_residual_selected_generators"),
        (frozen, "cpb_d8_residual_frozen_pipelines"),
    ):
        if (
            not isinstance(payload, dict)
            or set(payload) != common_keys
            or payload["schema_version"] != 1
            or payload["scope"] != scope
            or payload["config_sha256"] != config.config_sha256
            or payload["outer_evaluation_count"] != 0
            or payload["test_scale_override"] is not test_scale
            or not isinstance(payload["selections"], list)
        ):
            raise ResidualArtifactError("selection document changed")
    assert isinstance(selected, dict) and isinstance(frozen, dict)
    _validate_selection_records(selected, frozen, config=config)
    _validate_selection_evidence(
        selected,
        frozen,
        evaluations=evaluations,
        training_by_key=training_by_key,
        config=config,
        project_root=project_root,
        data=data,
        test_scale_override=test_scale,
    )
    pipeline_count = len(frozen["selections"])
    if not test_scale:
        role_counts = {
            role: sum(row["role"] == role for row in training)
            for role in ("stage_a", "stage_b", "final")
        }
        if (
            role_counts["stage_a"] != 240
            or role_counts["stage_b"] != 180
            or len(checkpoints) != 180 + role_counts["final"]
            or role_counts["final"] > 18
            or len(metrics) != 60
            or pipeline_count != 6
            or len(selected["selections"]) != 6
        ):
            raise ResidualArtifactError("formal residual package roster is incomplete")
    scientific = hashlib.sha256(
        json.dumps(
            {
                "config_sha256": config.config_sha256,
                "candidate_index_sha256": _sha256_file(
                    output / "candidate_index.csv"
                ),
                "training_sha256": _sha256_file(output / "training.csv"),
                "inner_predictions_sha256": _sha256_file(
                    output / "inner_predictions.csv"
                ),
                "inner_metrics_sha256": _sha256_file(output / "inner_metrics.csv"),
                "checkpoint_index_sha256": _sha256_file(
                    output / "checkpoint_index.csv"
                ),
                "selected_generators_sha256": _sha256_file(
                    output / "selected_generators.json"
                ),
                "frozen_pipelines_sha256": _sha256_file(
                    output / "frozen_pipelines.json"
                ),
                "checkpoint_states": [
                    row["checkpoint_scientific_digest"] for row in checkpoints
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return (
        test_scale,
        len(training),
        len(predictions),
        len(checkpoints),
        pipeline_count,
        scientific,
    )


def build_residual_search_package(
    output_dir: str | Path,
    *,
    source_dir: str | Path,
    project_root: str | Path,
    config_path: str | Path,
) -> D8ValidatedResidualSearchPackage:
    """Build one residual pre-outer package and validate it before use."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_residual_diffusion_config(config_file, project_root=root)
    source = Path(source_dir)
    _root_entry_set(source, expected=_SOURCE_ENTRIES)
    output = Path(output_dir)
    if output.exists():
        raise ResidualArtifactError("residual package target already exists")
    try:
        shutil.copytree(source, output, symlinks=False)
        loaded = load_residual_diffusion_config(
            output / "config.yaml", project_root=root
        )
        if loaded.config_sha256 != config.config_sha256:
            raise ResidualArtifactError("packaged config changed")
        semantics = _validate_semantics(
            output,
            config=config,
            project_root=root,
        )
        provenance = _execution_provenance(
            output,
            config=config,
            project_root=root,
        )
        scientific_digest = _scientific_digest(semantics[5], provenance)
        records = _package_records(output)
        manifest = {
            "schema_version": 1,
            "scope": "cpb_d8_residual_diffusion_preouter_package",
            "config_sha256": config.config_sha256,
            "outer_domains": list(DOMAIN_ORDER),
            "outer_evaluation_count": 0,
            "test_scale_override": semantics[0],
            "training_count": semantics[1],
            "prediction_count": semantics[2],
            "checkpoint_count": semantics[3],
            "pipeline_count": semantics[4],
            "scientific_digest": scientific_digest,
            "execution_provenance": provenance,
            "outputs": records,
            "output_tree_sha256": _tree_sha256(records),
        }
        _write_json(output / "artifact_manifest.json", manifest)
        (output / "CHECKSUMS.sha256").write_bytes(_checksum_payload(output))
        return validate_residual_search_package(
            output,
            project_root=root,
            config_path=config_file,
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def validate_residual_search_package(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> D8ValidatedResidualSearchPackage:
    """Reload and independently validate one residual pre-outer package."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_residual_diffusion_config(config_file, project_root=root)
    output = Path(output_dir)
    _root_entry_set(output, expected=_ROOT_ENTRIES)
    expected_checksums = _checksum_payload(output)
    actual_checksums = _regular_file(output / "CHECKSUMS.sha256")
    if actual_checksums != expected_checksums:
        raise ResidualArtifactError("residual package checksums changed")
    loaded = load_residual_diffusion_config(output / "config.yaml", project_root=root)
    if loaded.config_sha256 != config.config_sha256:
        raise ResidualArtifactError("packaged config authority changed")
    semantics = _validate_semantics(
        output,
        config=config,
        project_root=root,
    )
    provenance = _execution_provenance(
        output,
        config=config,
        project_root=root,
    )
    scientific_digest = _scientific_digest(semantics[5], provenance)
    manifest = _read_json(output / "artifact_manifest.json", label="artifact manifest")
    required = {
        "schema_version",
        "scope",
        "config_sha256",
        "outer_domains",
        "outer_evaluation_count",
        "test_scale_override",
        "training_count",
        "prediction_count",
        "checkpoint_count",
        "pipeline_count",
        "scientific_digest",
        "execution_provenance",
        "outputs",
        "output_tree_sha256",
    }
    records = _package_records(output)
    expected = {
        "schema_version": 1,
        "scope": "cpb_d8_residual_diffusion_preouter_package",
        "config_sha256": config.config_sha256,
        "outer_domains": list(DOMAIN_ORDER),
        "outer_evaluation_count": 0,
        "test_scale_override": semantics[0],
        "training_count": semantics[1],
        "prediction_count": semantics[2],
        "checkpoint_count": semantics[3],
        "pipeline_count": semantics[4],
        "scientific_digest": scientific_digest,
        "execution_provenance": provenance,
        "outputs": records,
        "output_tree_sha256": _tree_sha256(records),
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ResidualArtifactError("residual artifact manifest changed")
    if manifest["execution_provenance"] != provenance:
        raise ResidualArtifactError("execution provenance changed")
    if manifest != expected:
        raise ResidualArtifactError("residual artifact manifest changed")
    return D8ValidatedResidualSearchPackage(
        outer_evaluation_count=0,
        test_scale_override=semantics[0],
        training_count=semantics[1],
        prediction_count=semantics[2],
        checkpoint_count=semantics[3],
        pipeline_count=semantics[4],
        scientific_digest=scientific_digest,
        output_tree_sha256=str(manifest["output_tree_sha256"]),
        artifact_manifest_sha256=_sha256_file(output / "artifact_manifest.json"),
    )


def _atomic_replace(source: Path, target: Path) -> None:
    source.replace(target)


@contextmanager
def _publication_lock(output: Path):
    lock = output.parent / f".{output.name}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ResidualArtifactError("residual publication lock is not regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResidualArtifactError(
                "residual publication is already active"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _owner_payload(output: Path, transaction_uuid: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "cpb_d8_residual_publication_transaction",
        "output": str(output.resolve()),
        "transaction_uuid": transaction_uuid,
    }


def _owned_transaction(transaction: Path, output: Path) -> bool:
    owner = _read_json(
        transaction / "transaction_owner.json",
        label="residual transaction owner",
    )
    transaction_uuid = transaction.name.rsplit("-", 1)[-1]
    return owner == _owner_payload(output, transaction_uuid)


def _transactions(output: Path) -> tuple[Path, ...]:
    candidates = tuple(
        sorted(output.parent.glob(f".{output.name}.transaction-*"))
    )
    try:
        foreign = any(
            not item.is_dir()
            or item.is_symlink()
            or not _owned_transaction(item, output)
            for item in candidates
        )
    except ResidualArtifactError as error:
        raise ResidualArtifactError(
            "foreign residual publication transaction exists"
        ) from error
    if foreign:
        raise ResidualArtifactError("foreign residual publication transaction exists")
    if len(candidates) > 1:
        raise ResidualArtifactError("multiple residual publication transactions exist")
    return candidates


def _valid_package_candidate(
    candidate: Path,
    *,
    project_root: Path,
    config_path: Path,
) -> bool:
    try:
        validate_residual_search_package(
            candidate,
            project_root=project_root,
            config_path=config_path,
        )
    except (OSError, ResidualArtifactError, ValueError):
        return False
    return True


def _recover_residual_publication_unlocked(
    output: Path,
    *,
    project_root: Path,
    config_path: Path,
) -> D8ValidatedResidualSearchPackage:
    transactions = _transactions(output)
    if not transactions:
        return validate_residual_search_package(
            output,
            project_root=project_root,
            config_path=config_path,
        )
    transaction = transactions[0]
    if output.exists():
        result = validate_residual_search_package(
            output,
            project_root=project_root,
            config_path=config_path,
        )
        shutil.rmtree(transaction)
        return result
    previous = transaction / "previous"
    staged = transaction / "staged"
    if _valid_package_candidate(
        previous,
        project_root=project_root,
        config_path=config_path,
    ):
        candidate = previous
    elif _valid_package_candidate(
        staged,
        project_root=project_root,
        config_path=config_path,
    ):
        candidate = staged
    else:
        raise ResidualArtifactError(
            "interrupted residual publication has no recoverable package"
        )
    _atomic_replace(candidate, output)
    result = validate_residual_search_package(
        output,
        project_root=project_root,
        config_path=config_path,
    )
    shutil.rmtree(transaction)
    return result


def _commit_staged_residual_package(
    staged: Path,
    output: Path,
    transaction: Path,
    *,
    project_root: Path,
    config_path: Path,
) -> D8ValidatedResidualSearchPackage:
    previous = transaction / "previous"
    invalid = transaction / "invalid-output"
    moved_previous = False
    committed = False
    rollback_succeeded = False
    try:
        if output.exists():
            validate_residual_search_package(
                output,
                project_root=project_root,
                config_path=config_path,
            )
            _atomic_replace(output, previous)
            moved_previous = True
        _atomic_replace(staged, output)
        result = validate_residual_search_package(
            output,
            project_root=project_root,
            config_path=config_path,
        )
        committed = True
        return result
    except Exception:
        if output.exists():
            _atomic_replace(output, invalid)
        if moved_previous and previous.exists():
            _atomic_replace(previous, output)
            validate_residual_search_package(
                output,
                project_root=project_root,
                config_path=config_path,
            )
            rollback_succeeded = True
        raise
    finally:
        if committed:
            if previous.exists():
                shutil.rmtree(previous)
            if invalid.exists():
                shutil.rmtree(invalid)
            if transaction.exists():
                shutil.rmtree(transaction)
        elif rollback_succeeded:
            if invalid.exists():
                shutil.rmtree(invalid)
            if transaction.exists():
                shutil.rmtree(transaction)


def _publish_built_residual_package(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> D8ValidatedResidualSearchPackage:
    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(output):
        transactions = _transactions(output)
        if transactions:
            _recover_residual_publication_unlocked(
                output,
                project_root=root,
                config_path=config_file,
            )
        elif output.exists():
            validate_residual_search_package(
                output,
                project_root=root,
                config_path=config_file,
            )
        transaction_uuid = uuid4().hex
        allocated = Path(
            tempfile.mkdtemp(prefix=".d8-residual-allocation-", dir=output.parent)
        )
        transaction = (
            output.parent / f".{output.name}.transaction-{transaction_uuid}"
        )
        _atomic_replace(allocated, transaction)
        _write_json(
            transaction / "transaction_owner.json",
            _owner_payload(output, transaction_uuid),
        )
        staged = transaction / "staged"
        try:
            build_residual_search_package(
                staged,
                source_dir=source_dir,
                project_root=root,
                config_path=config_file,
            )
        except Exception:
            if transaction.exists() and not staged.exists():
                shutil.rmtree(transaction)
            raise
        return _commit_staged_residual_package(
            staged,
            output,
            transaction,
            project_root=root,
            config_path=config_file,
        )


def _registered_output(
    output_dir: str | Path,
    *,
    project_root: Path,
    config: ResidualDiffusionConfig,
) -> Path:
    output = Path(output_dir).resolve()
    allowed = {
        (project_root / config.output_dir).resolve(),
        (project_root / config.replay_output_dir).resolve(),
    }
    if output not in allowed or output.parent == project_root:
        raise ResidualArtifactError(
            "residual publication target is not a registered safe leaf"
        )
    return output


def publish_residual_search_package(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> D8ValidatedResidualSearchPackage:
    """Publish only to the registered production or replay result leaf."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_residual_diffusion_config(config_file, project_root=root)
    output = _registered_output(
        output_dir,
        project_root=root,
        config=config,
    )
    return _publish_built_residual_package(
        source_dir,
        output,
        project_root=root,
        config_path=config_file,
    )


def recover_interrupted_residual_publication(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> D8ValidatedResidualSearchPackage:
    """Recover one owned transaction for a registered result leaf."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_residual_diffusion_config(config_file, project_root=root)
    output = _registered_output(
        output_dir,
        project_root=root,
        config=config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(output):
        return _recover_residual_publication_unlocked(
            output,
            project_root=root,
            config_path=config_file,
        )


__all__ = [
    "D8ValidatedResidualSearchPackage",
    "ResidualArtifactError",
    "ResidualArtifactRecorder",
    "build_residual_search_package",
    "publish_residual_search_package",
    "recover_interrupted_residual_publication",
    "validate_residual_search_package",
]
