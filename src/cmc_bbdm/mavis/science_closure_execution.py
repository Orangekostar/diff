"""Formal execution and packaging for MAVIS science-closure diagnostics."""

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
    aggregate_value_evolution,
    bootstrap_value_evolution,
    build_value_evolution,
    evaluate_value_evolution,
)


class ScienceClosureExecutionError(RuntimeError):
    """Raised when a science-closure execution or package is invalid."""


_CONFIG_KEYS = {
    "schema_version",
    "audit_base_git_sha",
    "domain_order",
    "p1_state_manifest",
    "p1_state_manifest_sha256",
    "p3_action_scores",
    "p3_action_scores_sha256",
    "p7_package",
    "p7_tree_state_sha256",
    "modes",
    "top_k",
    "bootstrap_replicates",
    "seed",
}
_P9_FILES = {
    "value_evolution.parquet",
    "pair_metrics.parquet",
    "per_specimen_metrics.parquet",
    "aggregate_metrics.csv",
    "domain_metrics.csv",
    "bootstrap.csv",
    "REPORT.md",
    "summary.json",
    "artifact_manifest.json",
    "CHECKSUMS.sha256",
}


@dataclass(frozen=True, slots=True)
class ScienceClosureConfig:
    schema_version: int
    audit_base_git_sha: str
    domain_order: tuple[str, ...]
    p1_state_manifest: str
    p1_state_manifest_sha256: str
    p3_action_scores: str
    p3_action_scores_sha256: str
    p7_package: str
    p7_tree_state_sha256: str
    modes: tuple[str, ...]
    top_k: int
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


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or not value
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ScienceClosureExecutionError(f"{label} is invalid")
    return tuple(value)


def load_science_closure_config(path: str | Path) -> ScienceClosureConfig:
    try:
        source = Path(path).resolve(strict=True)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ScienceClosureExecutionError("science-closure config is unavailable") from error
    if type(payload) is not dict or set(payload) != _CONFIG_KEYS:
        raise ScienceClosureExecutionError("science-closure config schema changed")
    domains = _text_tuple(payload["domain_order"], "domain order")
    modes = _text_tuple(payload["modes"], "mode roster")
    required_modes = {"real", "positions_only", "shuffled", "static"}
    if (
        payload["schema_version"] != 1
        or not _is_hex(payload["audit_base_git_sha"], 40)
        or not _is_hex(payload["p1_state_manifest_sha256"], 64)
        or not _is_hex(payload["p3_action_scores_sha256"], 64)
        or not _is_hex(payload["p7_tree_state_sha256"], 64)
        or set(modes) != required_modes
        or type(payload["top_k"]) is not int
        or payload["top_k"] <= 0
        or type(payload["bootstrap_replicates"]) is not int
        or payload["bootstrap_replicates"] < 2
        or type(payload["seed"]) is not int
        or isinstance(payload["seed"], bool)
        or any(
            type(payload[key]) is not str or not payload[key]
            for key in ("p1_state_manifest", "p3_action_scores", "p7_package")
        )
    ):
        raise ScienceClosureExecutionError("science-closure config values are invalid")
    return ScienceClosureConfig(
        schema_version=1,
        audit_base_git_sha=payload["audit_base_git_sha"],
        domain_order=domains,
        p1_state_manifest=payload["p1_state_manifest"],
        p1_state_manifest_sha256=payload["p1_state_manifest_sha256"],
        p3_action_scores=payload["p3_action_scores"],
        p3_action_scores_sha256=payload["p3_action_scores_sha256"],
        p7_package=payload["p7_package"],
        p7_tree_state_sha256=payload["p7_tree_state_sha256"],
        modes=modes,
        top_k=payload["top_k"],
        bootstrap_replicates=payload["bootstrap_replicates"],
        seed=payload["seed"],
        config_sha256=_sha256(source),
    )


def _bound_path(root: Path, value: str, *, directory: bool) -> Path:
    try:
        path = (root / value).resolve(strict=True)
    except OSError as error:
        raise ScienceClosureExecutionError("configured input is unavailable") from error
    if root != path and root not in path.parents:
        raise ScienceClosureExecutionError("configured input escapes the project root")
    if directory != path.is_dir():
        raise ScienceClosureExecutionError("configured input type changed")
    return path


def _tree_state(path: Path) -> str:
    rows = [
        (item.relative_to(path).as_posix(), item.stat().st_size, _sha256(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    if not rows:
        raise ScienceClosureExecutionError("bound package is empty")
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(table: pl.DataFrame, path: Path) -> None:
    table.write_csv(path, float_scientific=False)


def _code_state() -> str:
    files = [Path(__file__), Path(__file__).with_name("science_closure.py")]
    rows = [(path.name, _sha256(path)) for path in sorted(files)]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _checkpoint_records(table: pl.DataFrame, source: str) -> list[dict[str, object]]:
    columns = [
        "current_checkpoint",
        "rank_spearman",
        "top_k_jaccard",
        "best_action_turnover",
        "mean_absolute_value_shift",
        "dynamic_vs_initial_opportunity",
    ]
    return table.filter(pl.col("value_source") == source).select(columns).sort(
        "current_checkpoint"
    ).to_dicts()


def _report(summary: dict[str, object]) -> str:
    teacher = summary["teacher_by_checkpoint"]
    real = summary["real_by_checkpoint"]
    lines = [
        "# MAVIS P9 Conditional Value Evolution",
        "",
        "Status: `COMPLETE`.",
        "",
        "The strict-OOF teacher value of the same still-legal action changes as",
        "causally acquired ultrasonic evidence accumulates. Candidate identity is",
        "the exact `(cell_index, from_level, to_level)` tuple; initial and current",
        "marginal costs are retained separately because overlapping acquired masks",
        "can change incremental cost without executing that candidate.",
        "",
        "| Checkpoint | Teacher Spearman | Teacher Top-{} Jaccard | Teacher turnover | Teacher value shift | Teacher dynamic opportunity | Real dynamic opportunity |".format(
            summary["top_k"]
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for teacher_row, real_row in zip(teacher, real, strict=True):
        lines.append(
            "| {current_checkpoint:.5f} | {rank_spearman:.6f} | "
            "{top_k_jaccard:.6f} | {best_action_turnover:.6f} | "
            "{mean_absolute_value_shift:.6f} | "
            "{dynamic_vs_initial_opportunity:.6f} | {real_opportunity:.6f} |".format(
                **teacher_row,
                real_opportunity=real_row["dynamic_vs_initial_opportunity"],
            )
        )
    lines.extend(
        [
            "",
            str(summary["primary_conclusion"]),
            "",
            "All inference uses physical specimens resampled within held-out",
            "domains and then weights domains equally. Teacher rows are the mean",
            "of the registered strict-OOF fold predictions. No future unacquired",
            "target content is read by this analysis.",
            "",
            "This is a diagnostic value-opportunity result. It does not upgrade the",
            "frozen P7 deployable performance claim and does not establish scanner-",
            "time reduction or external generalization.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_checksums(output: Path) -> None:
    files = sorted(item for item in output.iterdir() if item.name != "CHECKSUMS.sha256")
    (output / "CHECKSUMS.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="ascii",
    )


def run_p9_value_evolution(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_root: str | Path,
) -> Path:
    """Generate the formal P9 package from frozen P1/P3 evidence."""

    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        raise ScienceClosureExecutionError("project root is unavailable") from error
    config = load_science_closure_config(config_path)
    state_path = _bound_path(root, config.p1_state_manifest, directory=False)
    action_path = _bound_path(root, config.p3_action_scores, directory=False)
    p7_path = _bound_path(root, config.p7_package, directory=True)
    if (
        _sha256(state_path) != config.p1_state_manifest_sha256
        or _sha256(action_path) != config.p3_action_scores_sha256
        or _tree_state(p7_path) != config.p7_tree_state_sha256
    ):
        raise ScienceClosureExecutionError("frozen science-closure input hash changed")
    p7_before = _tree_state(p7_path)
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if root not in destination.parents or destination.exists():
        raise ScienceClosureExecutionError("P9 output is invalid or already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p9_value_evolution.", dir=destination.parent))
    try:
        states = pl.read_parquet(state_path)
        action_scores = pl.read_parquet(action_path)
        evolution = build_value_evolution(
            states,
            action_scores,
            domain_order=config.domain_order,
            modes=config.modes,
        )
        pair_metrics = evaluate_value_evolution(evolution, top_k=config.top_k)
        tables = aggregate_value_evolution(
            pair_metrics,
            domain_order=config.domain_order,
        )
        bootstrap = bootstrap_value_evolution(
            tables.per_specimen,
            domain_order=config.domain_order,
            replicates=config.bootstrap_replicates,
            seed=config.seed,
        )
        evolution.write_parquet(
            temporary / "value_evolution.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        pair_metrics.write_parquet(
            temporary / "pair_metrics.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        tables.per_specimen.write_parquet(
            temporary / "per_specimen_metrics.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        _write_csv(tables.aggregate, temporary / "aggregate_metrics.csv")
        _write_csv(tables.per_domain, temporary / "domain_metrics.csv")
        _write_csv(bootstrap, temporary / "bootstrap.csv")
        teacher_bootstrap = bootstrap.filter(
            (pl.col("metric") == "dynamic_vs_initial_opportunity")
            & (pl.col("contrast") == "teacher")
        )
        teacher_supported = (
            teacher_bootstrap.height
            == tables.aggregate.get_column("current_checkpoint").n_unique()
            and teacher_bootstrap.filter(pl.col("ci95_lower") <= 0.0).is_empty()
        )
        real_bootstrap = bootstrap.filter(
            (pl.col("metric") == "dynamic_vs_initial_opportunity")
            & (pl.col("contrast") == "real")
        )
        real_supported = (
            real_bootstrap.height == teacher_bootstrap.height
            and real_bootstrap.filter(pl.col("ci95_lower") <= 0.0).is_empty()
        )
        conclusion = (
            "True conditional action value evolves materially under causal "
            "measurement histories, but the frozen real-state scorer does not "
            "reliably convert that opportunity into higher teacher utility."
            if teacher_supported and not real_supported
            else "Conditional value-evolution support is mixed under the frozen protocol."
        )
        state_payload = {
            "schema": 1,
            "stage": "P9_CONDITIONAL_VALUE_EVOLUTION",
            "config_sha256": config.config_sha256,
            "p1_state_manifest_sha256": config.p1_state_manifest_sha256,
            "p3_action_scores_sha256": config.p3_action_scores_sha256,
            "p7_tree_state_sha256": p7_before,
            "runtime_code_state_sha256": _code_state(),
            "value_evolution_row_count": evolution.height,
            "pair_metric_row_count": pair_metrics.height,
            "specimen_metric_row_count": tables.per_specimen.height,
            "domain_metric_row_count": tables.per_domain.height,
            "bootstrap_row_count": bootstrap.height,
        }
        p9_state_sha256 = hashlib.sha256(
            json.dumps(
                state_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        summary: dict[str, object] = {
            **state_payload,
            "schema_version": 1,
            "status": "COMPLETE",
            "audit_base_git_sha": config.audit_base_git_sha,
            "p9_state_sha256": p9_state_sha256,
            "domain_count": len(config.domain_order),
            "specimen_count": states.get_column("specimen_id").n_unique(),
            "trajectory_count": states.get_column("trajectory_id").n_unique(),
            "decision_state_count": action_scores.get_column("state_id").n_unique(),
            "checkpoint_count": tables.aggregate.get_column(
                "current_checkpoint"
            ).n_unique(),
            "modes": list(config.modes),
            "top_k": config.top_k,
            "bootstrap_replicates": config.bootstrap_replicates,
            "seed": config.seed,
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "teacher_dynamic_opportunity_supported": teacher_supported,
            "real_scorer_dynamic_opportunity_supported": real_supported,
            "teacher_by_checkpoint": _checkpoint_records(tables.aggregate, "teacher"),
            "real_by_checkpoint": _checkpoint_records(tables.aggregate, "real"),
            "primary_conclusion": conclusion,
            "p7_modified": False,
        }
        _write_json(temporary / "summary.json", summary)
        (temporary / "REPORT.md").write_text(_report(summary), encoding="utf-8")
        products = sorted(item for item in temporary.iterdir() if item.is_file())
        manifest = {
            "schema_version": 1,
            "stage": "P9_CONDITIONAL_VALUE_EVOLUTION",
            "status": "COMPLETE",
            "p9_state_sha256": p9_state_sha256,
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
        if _tree_state(p7_path) != p7_before:
            raise ScienceClosureExecutionError("P9 modified frozen P7 artifacts")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_p9_value_evolution_package(destination)
    return destination


def verify_p9_value_evolution_package(path: str | Path) -> dict[str, object]:
    try:
        package = Path(path).resolve(strict=True)
    except OSError as error:
        raise ScienceClosureExecutionError("P9 package is unavailable") from error
    if not package.is_dir() or {item.name for item in package.iterdir()} != _P9_FILES:
        raise ScienceClosureExecutionError("P9 package file roster changed")
    lines = (package / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    checksums: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or not _is_hex(parts[0], 64) or parts[1] in checksums:
            raise ScienceClosureExecutionError("P9 checksum manifest is invalid")
        checksums[parts[1]] = parts[0]
    expected = _P9_FILES - {"CHECKSUMS.sha256"}
    if set(checksums) != expected or any(
        _sha256(package / name) != digest for name, digest in checksums.items()
    ):
        raise ScienceClosureExecutionError("P9 checksum mismatch")
    try:
        manifest = json.loads(
            (package / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        summary = json.loads((package / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScienceClosureExecutionError("P9 metadata is invalid") from error
    if (
        type(manifest) is not dict
        or type(summary) is not dict
        or manifest.get("stage") != "P9_CONDITIONAL_VALUE_EVOLUTION"
        or manifest.get("status") != "COMPLETE"
        or summary.get("stage") != manifest["stage"]
        or summary.get("status") != "COMPLETE"
        or summary.get("p9_state_sha256") != manifest.get("p9_state_sha256")
        or not _is_hex(manifest.get("p9_state_sha256"), 64)
        or summary.get("p7_modified") is not False
        or not math.isfinite(float(summary.get("value_evolution_row_count", math.nan)))
    ):
        raise ScienceClosureExecutionError("P9 metadata contract changed")
    product_records = manifest.get("files")
    if type(product_records) is not list or {
        record.get("path") for record in product_records if type(record) is dict
    } != expected - {"artifact_manifest.json"}:
        raise ScienceClosureExecutionError("P9 artifact manifest roster changed")
    return manifest


__all__ = [
    "ScienceClosureConfig",
    "ScienceClosureExecutionError",
    "load_science_closure_config",
    "run_p9_value_evolution",
    "verify_p9_value_evolution_package",
]
