#!/usr/bin/env python3
"""Run the registered D8 residual-diffusion pre-outer commands."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import stat
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import torch

from cmc_bbdm.cpb_diffusion_marginalization.authority import (
    issue_inner_fold,
    issue_search_view,
)
from cmc_bbdm.cpb_diffusion_marginalization.config import (
    DOMAIN_ORDER,
    load_d8_config,
)
from cmc_bbdm.cpb_diffusion_marginalization.features import (
    create_d8_frozen_encoder,
)
from cmc_bbdm.cpb_diffusion_marginalization.pilot import (
    load_registered_pilot_assets,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_artifacts import (
    ResidualArtifactRecorder,
    build_residual_search_package,
    publish_residual_search_package,
    validate_residual_search_package,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_config import (
    ResidualDiffusionConfig,
    load_residual_diffusion_config,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_search import (
    load_b0_incumbent_evidence,
    load_pilot_incumbent_evidence,
    load_pilot_scaffold_candidates,
    run_residual_outer_search,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_targets import (
    load_pilot_diffusion_scaffolds,
    load_search_residual_field_bank,
)
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTERED_CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_residual_diffusion.yaml"
_COMMANDS = ("smoke", "train", "validate", "replay")
_WORKER_SOURCE_ENTRIES = frozenset(
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
    }
)
_MERGED_CSV_FILES = (
    "training.csv",
    "inner_predictions.csv",
    "inner_metrics.csv",
    "checkpoint_index.csv",
)
_V3_CODE_ROOT = PROJECT_ROOT / "src/cmc_bbdm/cpb_v3"
_D8_CODE_ROOT = PROJECT_ROOT / "src/cmc_bbdm/cpb_diffusion_marginalization"


class D8ResidualExecutionError(RuntimeError):
    """Raised when a registered residual command cannot run safely."""


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


def _read_regular(path: Path, *, maximum_bytes: int = 128 << 20) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise D8ResidualExecutionError(f"worker file is unavailable: {path.name}") from error
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size <= 0
            or info.st_size > maximum_bytes
        ):
            raise D8ResidualExecutionError(
                f"worker file is not a bounded regular file: {path.name}"
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum_bytes:
                raise D8ResidualExecutionError(
                    f"worker file is too large: {path.name}"
                )
        payload = b"".join(chunks)
        if len(payload) != info.st_size:
            raise D8ResidualExecutionError(
                f"worker file changed while reading: {path.name}"
            )
        return payload
    except OSError as error:
        raise D8ResidualExecutionError(f"worker file cannot be read: {path.name}") from error
    finally:
        os.close(descriptor)


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


def _source_records(
    root: Path,
    *,
    manifest_present: bool,
) -> dict[str, dict[str, object]]:
    expected_entries = set(_WORKER_SOURCE_ENTRIES)
    if manifest_present:
        expected_entries.add("worker_manifest.json")
    if (
        not root.is_dir()
        or root.is_symlink()
        or {path.name for path in root.iterdir()} != expected_entries
    ):
        raise D8ResidualExecutionError("worker source entry set changed")
    records: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == "worker_manifest.json":
            continue
        if path.is_symlink():
            raise D8ResidualExecutionError("worker source contains a symlink")
        if path.is_dir():
            if relative != "models":
                raise D8ResidualExecutionError(
                    "worker source contains an unknown directory"
                )
            continue
        payload = _read_regular(path)
        records[relative] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if not any(name.startswith("models/") for name in records):
        raise D8ResidualExecutionError("worker source has no retained checkpoints")
    return records


def _code_records() -> dict[str, dict[str, object]]:
    paths = (
        PROJECT_ROOT / "scripts/run_d8_residual_diffusion.py",
        *sorted(_D8_CODE_ROOT.glob("*.py")),
        *sorted(_V3_CODE_ROOT.glob("*.py")),
    )
    records: dict[str, dict[str, object]] = {}
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if relative in records:
            raise D8ResidualExecutionError("worker code roster is not unique")
        payload = _read_regular(path, maximum_bytes=16 << 20)
        records[relative] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return records


def _read_csv(path: Path) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    try:
        text = _read_regular(path).decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        fields = tuple(reader.fieldnames or ())
        rows = tuple(dict(row) for row in reader)
    except (UnicodeError, csv.Error) as error:
        raise D8ResidualExecutionError(
            f"worker CSV cannot be decoded: {path.name}"
        ) from error
    if (
        not fields
        or any(None in row or any(value is None for value in row.values()) for row in rows)
    ):
        raise D8ResidualExecutionError(f"worker CSV shape changed: {path.name}")
    return fields, rows


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(_read_regular(path).decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise D8ResidualExecutionError(
            f"worker JSON cannot be decoded: {path.name}"
        ) from error
    if type(value) is not dict:
        raise D8ResidualExecutionError(f"worker JSON is not a mapping: {path.name}")
    return value


def _worker_semantics(
    root: Path,
    *,
    outer_domains: tuple[str, ...],
    config: ResidualDiffusionConfig,
    test_scale_override: bool,
) -> dict[str, object]:
    fields, rows = _read_csv(root / "training.csv")
    if "role" not in fields or "outer_domain" not in fields:
        raise D8ResidualExecutionError("worker training identity columns changed")
    expected_counts = {
        (role, outer): count
        for outer in outer_domains
        for role, count in (("stage_a", 40), ("stage_b", 30))
    }
    observed_counts = {
        key: sum(
            row["role"] == key[0] and row["outer_domain"] == key[1]
            for row in rows
        )
        for key in expected_counts
    }
    extra = tuple(
        row
        for row in rows
        if row["role"] not in {"stage_a", "stage_b", "final"}
        or row["outer_domain"] not in outer_domains
    )
    if (
        any(observed_counts[key] != count for key, count in expected_counts.items())
        or extra
    ):
        raise D8ResidualExecutionError("worker training outer allocation changed")
    final_counts = {
        outer: sum(
            row["role"] == "final" and row["outer_domain"] == outer
            for row in rows
        )
        for outer in outer_domains
    }
    if any(count not in {0, 3} for count in final_counts.values()):
        raise D8ResidualExecutionError("worker final checkpoint roster changed")
    selections: dict[str, list[object]] = {}
    for name, scope in (
        ("selected_generators.json", "cpb_d8_residual_selected_generators"),
        ("frozen_pipelines.json", "cpb_d8_residual_frozen_pipelines"),
    ):
        payload = _read_json(root / name)
        expected_keys = {
            "schema_version",
            "scope",
            "config_sha256",
            "outer_evaluation_count",
            "test_scale_override",
            "selections",
        }
        values = payload.get("selections")
        if (
            set(payload) != expected_keys
            or payload["schema_version"] != 1
            or payload["scope"] != scope
            or payload["config_sha256"] != config.config_sha256
            or payload["outer_evaluation_count"] != 0
            or payload["test_scale_override"] is not test_scale_override
            or type(values) is not list
            or tuple(
                value.get("outer_domain")
                for value in values
                if type(value) is dict
            )
            != outer_domains
        ):
            raise D8ResidualExecutionError("worker selection allocation changed")
        selections[name] = values
    return {
        "stage_a_count": sum(observed_counts[("stage_a", outer)] for outer in outer_domains),
        "stage_b_count": sum(observed_counts[("stage_b", outer)] for outer in outer_domains),
        "final_count": sum(final_counts.values()),
        "pipeline_count": len(selections["frozen_pipelines.json"]),
    }


def _expected_worker_manifest(
    root: Path,
    *,
    worker_index: int,
    gpu_index: int,
    outer_domains: tuple[str, ...],
    config: ResidualDiffusionConfig,
    test_scale_override: bool,
    manifest_present: bool = False,
) -> dict[str, object]:
    if (
        type(worker_index) is not int
        or worker_index not in range(3)
        or type(gpu_index) is not int
        or gpu_index != worker_index
        or outer_domains != DOMAIN_ORDER[2 * worker_index : 2 * worker_index + 2]
        or type(test_scale_override) is not bool
    ):
        raise D8ResidualExecutionError("worker identity or outer allocation changed")
    if hashlib.sha256(_read_regular(root / "config.yaml")).hexdigest() != (
        config.config_sha256
    ):
        raise D8ResidualExecutionError("worker config bytes changed")
    records = _source_records(root, manifest_present=manifest_present)
    semantics = _worker_semantics(
        root,
        outer_domains=outer_domains,
        config=config,
        test_scale_override=test_scale_override,
    )
    payload = {
        "schema_version": 1,
        "scope": "cpb_d8_residual_diffusion_worker",
        "worker_index": worker_index,
        "gpu_index": gpu_index,
        "outer_domains": list(outer_domains),
        "outer_evaluation_count": 0,
        "test_scale_override": test_scale_override,
        "config_sha256": config.config_sha256,
        "runtime": dict(config.runtime),
        "registered_sources": {
            key: {"path": value.path, "sha256": value.sha256}
            for key, value in config.sources.items()
        },
        "code_records": _code_records(),
        "source_records": records,
        "source_tree_sha256": _canonical_sha256(records),
        "semantics": semantics,
    }
    return {**payload, "state_sha256": _canonical_sha256(payload)}


def write_worker_manifest(
    root: str | Path,
    *,
    worker_index: int,
    gpu_index: int,
    outer_domains: tuple[str, ...],
    config: ResidualDiffusionConfig,
    test_scale_override: bool,
) -> dict[str, object]:
    """Write one independently reproducible worker-source manifest."""

    source = Path(root)
    target = source / "worker_manifest.json"
    if target.exists():
        raise D8ResidualExecutionError("worker manifest already exists")
    manifest = _expected_worker_manifest(
        source,
        worker_index=worker_index,
        gpu_index=gpu_index,
        outer_domains=outer_domains,
        config=config,
        test_scale_override=test_scale_override,
    )
    _write_json(target, manifest)
    return manifest


def validate_worker_manifest(
    root: str | Path,
    *,
    worker_index: int,
    gpu_index: int,
    outer_domains: tuple[str, ...],
    config: ResidualDiffusionConfig,
    test_scale_override: bool,
) -> dict[str, object]:
    """Recompute one worker manifest from source bytes and frozen authorities."""

    source = Path(root)
    manifest_path = source / "worker_manifest.json"
    actual = _read_json(manifest_path)
    expected = _expected_worker_manifest(
        source,
        worker_index=worker_index,
        gpu_index=gpu_index,
        outer_domains=outer_domains,
        config=config,
        test_scale_override=test_scale_override,
        manifest_present=True,
    )
    if actual != expected:
        raise D8ResidualExecutionError("worker manifest or source tree changed")
    return expected


def _merge_csv(worker_roots: Sequence[Path], output: Path, name: str) -> None:
    fields: tuple[str, ...] | None = None
    rows: list[Mapping[str, str]] = []
    for root in worker_roots:
        observed_fields, observed_rows = _read_csv(root / name)
        if fields is None:
            fields = observed_fields
        elif observed_fields != fields:
            raise D8ResidualExecutionError(f"worker {name} schemas differ")
        rows.extend(observed_rows)
    assert fields is not None
    with (output / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def merge_worker_sources(
    worker_roots: Sequence[str | Path],
    output: str | Path,
    *,
    config: ResidualDiffusionConfig,
    test_scale_override: bool,
) -> Path:
    """Validate and deterministically merge the three isolated worker sources."""

    roots = tuple(Path(value) for value in worker_roots)
    assignments = registered_worker_assignments()
    target = Path(output)
    if len(roots) != 3 or target.exists():
        raise D8ResidualExecutionError("exactly three new worker sources are required")
    for worker_index, (root, (gpu_index, outer_domains)) in enumerate(
        zip(roots, assignments, strict=True)
    ):
        validate_worker_manifest(
            root,
            worker_index=worker_index,
            gpu_index=gpu_index,
            outer_domains=outer_domains,
            config=config,
            test_scale_override=test_scale_override,
        )
    target.mkdir(parents=True)
    (target / "models").mkdir()
    try:
        for name in ("config.yaml", "candidate_index.csv"):
            payloads = tuple(_read_regular(root / name) for root in roots)
            if len(set(payloads)) != 1:
                raise D8ResidualExecutionError(f"worker {name} files differ")
            (target / name).write_bytes(payloads[0])
        for name in _MERGED_CSV_FILES:
            _merge_csv(roots, target, name)
        for root in roots:
            for path in sorted((root / "models").iterdir()):
                payload = _read_regular(path)
                destination = target / "models" / path.name
                if destination.exists():
                    raise D8ResidualExecutionError("worker checkpoint names collide")
                destination.write_bytes(payload)
        for name, scope in (
            ("selected_generators.json", "cpb_d8_residual_selected_generators"),
            ("frozen_pipelines.json", "cpb_d8_residual_frozen_pipelines"),
        ):
            rows: list[object] = []
            for root in roots:
                rows.extend(_read_json(root / name)["selections"])
            if tuple(
                row.get("outer_domain") for row in rows if type(row) is dict
            ) != DOMAIN_ORDER:
                raise D8ResidualExecutionError("merged outer selection order changed")
            _write_json(
                target / name,
                {
                    "schema_version": 1,
                    "scope": scope,
                    "config_sha256": config.config_sha256,
                    "outer_evaluation_count": 0,
                    "test_scale_override": test_scale_override,
                    "selections": rows,
                },
            )
        (target / "REPORT.md").write_text(
            "# D8 Residual Diffusion Pre-Outer Search\n\n"
            "- workers: 3\n"
            "- prospective outer pipelines: 6\n"
            "- outer_evaluation_count: 0\n",
            encoding="ascii",
            newline="\n",
        )
        if {path.name for path in target.iterdir()} != _WORKER_SOURCE_ENTRIES:
            raise D8ResidualExecutionError("merged source entry set changed")
        return target
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def registered_worker_assignments() -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Return the frozen two-outer allocation for each visible GPU."""

    return tuple(
        (gpu, DOMAIN_ORDER[2 * gpu : 2 * gpu + 2]) for gpu in range(3)
    )


def require_registered_gpu_inventory(torch_module: object) -> tuple[str, ...]:
    """Require exactly three visible NVIDIA A40 devices."""

    try:
        cuda = torch_module.cuda
        count = cuda.device_count()
        names = tuple(str(cuda.get_device_name(index)) for index in range(count))
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise D8ResidualExecutionError(
            "exactly three visible NVIDIA A40 GPUs are required"
        ) from error
    if count != 3 or any("A40" not in name for name in names):
        raise D8ResidualExecutionError(
            "exactly three visible NVIDIA A40 GPUs are required"
        )
    return names


def registered_output(command: str, *, config: object) -> Path:
    """Resolve only the frozen production and replay leaves."""

    if command == "train":
        relative = config.output_dir
    elif command == "replay":
        relative = config.replay_output_dir
    else:
        raise ValueError("command has no registered publication output")
    return PROJECT_ROOT / str(relative)


def _config() -> ResidualDiffusionConfig:
    return load_residual_diffusion_config(
        REGISTERED_CONFIG,
        project_root=PROJECT_ROOT,
    )


def _load_worker_context(device: str) -> SimpleNamespace:
    config = _config()
    exploration = load_d8_config(
        PROJECT_ROOT / config.sources["exploration_config"].path,
        project_root=PROJECT_ROOT,
    )
    v3_config = load_v3_config(
        PROJECT_ROOT / exploration.sources["p1_config"].path,
        project_root=PROJECT_ROOT,
    )
    data = load_data(v3_config, PROJECT_ROOT)
    pilot = {
        value.outer_domain: value
        for value in load_pilot_incumbent_evidence(
            config,
            project_root=PROJECT_ROOT,
        )
    }
    b0 = {
        value.outer_domain: value
        for value in load_b0_incumbent_evidence(
            data,
            config=config,
            project_root=PROJECT_ROOT,
        )
    }
    if tuple(pilot) != DOMAIN_ORDER or tuple(b0) != DOMAIN_ORDER:
        raise D8ResidualExecutionError("worker incumbent roster changed")
    return SimpleNamespace(
        config=config,
        exploration=exploration,
        data=data,
        assets=load_registered_pilot_assets(
            data,
            config=exploration,
            project_root=PROJECT_ROOT,
        ),
        scaffolds=load_pilot_diffusion_scaffolds(
            config,
            project_root=PROJECT_ROOT,
        ),
        candidates=load_pilot_scaffold_candidates(
            config,
            project_root=PROJECT_ROOT,
        ),
        incumbents={outer: (pilot[outer], b0[outer]) for outer in DOMAIN_ORDER},
        encoder=create_d8_frozen_encoder(
            project_root=PROJECT_ROOT,
            device=device,
        ),
    )


def run_worker_source(
    output: str | Path,
    *,
    worker_index: int,
    gpu_index: int,
    outer_domains: tuple[str, ...],
    test_scale_override: bool,
) -> dict[str, object]:
    """Run exactly two prospective outer studies on one assigned GPU."""

    assignments = registered_worker_assignments()
    if (
        type(worker_index) is not int
        or worker_index not in range(len(assignments))
        or assignments[worker_index] != (gpu_index, outer_domains)
        or type(test_scale_override) is not bool
    ):
        raise D8ResidualExecutionError("worker execution assignment changed")
    device = f"cuda:{gpu_index}"
    context = _load_worker_context(device)
    recorder = ResidualArtifactRecorder(
        Path(output),
        config=context.config,
        config_path=REGISTERED_CONFIG,
    )
    for outer_domain in outer_domains:
        search = issue_search_view(
            context.data,
            outer_domain=outer_domain,
            config=context.exploration,
        )
        folds = {
            domain: issue_inner_fold(search, query_domain=domain)
            for domain in DOMAIN_ORDER
            if domain != outer_domain
        }
        field_bank = load_search_residual_field_bank(
            search,
            project_root=PROJECT_ROOT,
        )
        result = run_residual_outer_search(
            search,
            folds=folds,
            config=context.config,
            scaffold=context.scaffolds[outer_domain],
            pilot_candidate=context.candidates[outer_domain],
            field_bank=field_bank,
            incumbents=context.incumbents[outer_domain],
            assets=context.assets,
            encoder=context.encoder,
            device=device,
            cell_recorder=recorder.record_cell,
            final_recorder=recorder.record_final,
            test_scale_override=test_scale_override,
        )
        recorder.record_outer(result)
    recorder.finalize_source(
        test_scale_override=test_scale_override,
        expected_outer_domains=outer_domains,
    )
    manifest = write_worker_manifest(
        recorder.root,
        worker_index=worker_index,
        gpu_index=gpu_index,
        outer_domains=outer_domains,
        config=context.config,
        test_scale_override=test_scale_override,
    )
    validated = validate_worker_manifest(
        recorder.root,
        worker_index=worker_index,
        gpu_index=gpu_index,
        outer_domains=outer_domains,
        config=context.config,
        test_scale_override=test_scale_override,
    )
    if validated != manifest:
        raise D8ResidualExecutionError("worker manifest replay changed")
    return {
        "worker_index": worker_index,
        "gpu_index": gpu_index,
        "outer_domains": list(outer_domains),
        "outer_evaluation_count": 0,
        "state_sha256": manifest["state_sha256"],
    }


def _worker_process_entry(
    source: Path,
    status: Path,
    *,
    worker_index: int,
    gpu_index: int,
    outer_domains: tuple[str, ...],
    test_scale_override: bool,
) -> None:
    try:
        result = run_worker_source(
            source,
            worker_index=worker_index,
            gpu_index=gpu_index,
            outer_domains=outer_domains,
            test_scale_override=test_scale_override,
        )
        _write_json(status, {"status": "PASS", "result": result})
    except BaseException as error:
        _write_json(
            status,
            {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def run_isolated_workers(
    root: str | Path,
    *,
    test_scale_override: bool,
) -> tuple[Path, ...]:
    """Launch the three registered workers and validate every result."""

    work_root = Path(root)
    if work_root.exists():
        raise D8ResidualExecutionError("worker allocation root already exists")
    work_root.mkdir(parents=True)
    context = multiprocessing.get_context("spawn")
    processes: list[tuple[object, Path, Path]] = []
    for worker_index, (gpu_index, outer_domains) in enumerate(
        registered_worker_assignments()
    ):
        allocation = work_root / f"worker-{worker_index}"
        allocation.mkdir()
        source = allocation / "source"
        status = allocation / "status.json"
        process = context.Process(
            target=_worker_process_entry,
            kwargs={
                "source": source,
                "status": status,
                "worker_index": worker_index,
                "gpu_index": gpu_index,
                "outer_domains": outer_domains,
                "test_scale_override": test_scale_override,
            },
            name=f"d8-residual-worker-{worker_index}",
        )
        process.start()
        processes.append((process, source, status))
    for process, _source, _status in processes:
        process.join()
    config = _config()
    roots: list[Path] = []
    for worker_index, ((gpu_index, outer_domains), item) in enumerate(
        zip(registered_worker_assignments(), processes, strict=True)
    ):
        process, source, status = item
        if process.exitcode != 0 or not status.is_file():
            raise D8ResidualExecutionError(
                f"residual worker {worker_index} failed without a valid status"
            )
        payload = _read_json(status)
        if (
            set(payload) != {"status", "result"}
            or payload["status"] != "PASS"
            or type(payload["result"]) is not dict
            or payload["result"].get("outer_evaluation_count") != 0
        ):
            raise D8ResidualExecutionError(
                f"residual worker {worker_index} did not pass"
            )
        validate_worker_manifest(
            source,
            worker_index=worker_index,
            gpu_index=gpu_index,
            outer_domains=outer_domains,
            config=config,
            test_scale_override=test_scale_override,
        )
        roots.append(source)
    return tuple(roots)


def _result_payload(command: str, result: object) -> dict[str, object]:
    if result.outer_evaluation_count != 0 or result.pipeline_count != 6:
        raise D8ResidualExecutionError("pre-outer result gate changed")
    return {
        "command": command,
        "status": "PASS",
        "outer_evaluation_count": 0,
        "pipeline_count": result.pipeline_count,
        "training_count": result.training_count,
        "checkpoint_count": result.checkpoint_count,
        "scientific_digest": result.scientific_digest,
        "output_tree_sha256": result.output_tree_sha256,
    }


def _execute_preouter(
    command: str,
    *,
    test_scale_override: bool,
) -> dict[str, object]:
    config = _config()
    require_registered_gpu_inventory(torch)
    if command == "smoke":
        scratch_parent = PROJECT_ROOT / "results"
        publication = None
    else:
        publication = registered_output(command, config=config)
        scratch_parent = publication.parent
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".d8-residual-{command}-",
        dir=scratch_parent,
    ) as directory:
        scratch = Path(directory)
        worker_roots = run_isolated_workers(
            scratch / "workers",
            test_scale_override=test_scale_override,
        )
        merged = merge_worker_sources(
            worker_roots,
            scratch / "merged-source",
            config=config,
            test_scale_override=test_scale_override,
        )
        if command == "smoke":
            result = build_residual_search_package(
                scratch / "validated-package",
                source_dir=merged,
                project_root=PROJECT_ROOT,
                config_path=REGISTERED_CONFIG,
            )
        else:
            assert publication is not None
            result = publish_residual_search_package(
                merged,
                publication,
                project_root=PROJECT_ROOT,
                config_path=REGISTERED_CONFIG,
            )
    return _result_payload(command, result)


def execute_smoke() -> dict[str, object]:
    return _execute_preouter("smoke", test_scale_override=True)


def execute_train() -> dict[str, object]:
    return _execute_preouter("train", test_scale_override=False)


def execute_replay() -> dict[str, object]:
    return _execute_preouter("replay", test_scale_override=False)


def checkpoint_scientific_records(root: str | Path) -> tuple[tuple[str, ...], ...]:
    """Return the inference-relevant checkpoint records in package order."""

    fields, rows = _read_csv(Path(root) / "checkpoint_index.csv")
    selected = (
        "role",
        "outer_domain",
        "query_domain",
        "candidate_id",
        "training_seed",
        "split_sha256",
        "checkpoint_scientific_digest",
        "state_dict_sha256",
    )
    if any(field not in fields for field in selected):
        raise D8ResidualExecutionError("checkpoint scientific schema changed")
    return tuple(tuple(row[field] for field in selected) for row in rows)


def execute_validate() -> dict[str, object]:
    config = _config()
    production = validate_residual_search_package(
        registered_output("train", config=config),
        project_root=PROJECT_ROOT,
        config_path=REGISTERED_CONFIG,
    )
    replay = validate_residual_search_package(
        registered_output("replay", config=config),
        project_root=PROJECT_ROOT,
        config_path=REGISTERED_CONFIG,
    )
    if production.scientific_digest != replay.scientific_digest:
        raise D8ResidualExecutionError("production and replay scientific states differ")
    if checkpoint_scientific_records(
        registered_output("train", config=config)
    ) != checkpoint_scientific_records(registered_output("replay", config=config)):
        raise D8ResidualExecutionError(
            "production and replay checkpoint scientific records differ"
        )
    return {
        "command": "validate",
        "status": "PASS",
        "outer_evaluation_count": 0,
        "scientific_digest": production.scientific_digest,
        "production_tree_sha256": production.output_tree_sha256,
        "replay_tree_sha256": replay.output_tree_sha256,
    }


def run_registered_command(command: str) -> dict[str, object]:
    functions = {
        "smoke": execute_smoke,
        "train": execute_train,
        "validate": execute_validate,
        "replay": execute_replay,
    }
    try:
        function = functions[command]
    except KeyError as error:
        raise D8ResidualExecutionError(
            "residual diffusion command is not registered"
        ) from error
    return function()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run registered D8 residual-diffusion pre-outer stages."
    )
    parser.add_argument("command", choices=_COMMANDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    print(
        json.dumps(
            run_registered_command(args.command),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
