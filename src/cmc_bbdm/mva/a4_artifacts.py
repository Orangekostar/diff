"""Staging, publication, and validation for the formal MVA A4 package."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import polars as pl

from .a4_config import A4Config, load_a4_config
from .a4_evaluation import A4Aggregation, aggregate_a4_tables
from .artifacts import MVAPackageValidation, validate_mva_package
from .config import MVAConfig, load_mva_config


class A4ArtifactError(ValueError):
    """Raised when an A4 evidence package is incomplete or inconsistent."""


REQUIRED_A4_OUTPUTS = (
    "source_values.parquet",
    "fit_audits.csv",
    "rankings.csv",
    "ranking_stability.csv",
    "fixed_trajectories.parquet",
    "state_metrics.parquet",
    "cai_curves.csv",
    "image_curves.csv",
    "domain_metrics.csv",
    "budget_metrics.csv",
    "specimen_metrics.csv",
    "bootstrap.csv",
    "summary.json",
    "REPORT.md",
    "figures/A4_global_rankings.png",
    "figures/A4_global_rankings.svg",
    "figures/A4_cai_error_budget.png",
    "figures/A4_cai_error_budget.svg",
    "figures/A4_image_task_tradeoff.png",
    "figures/A4_image_task_tradeoff.svg",
    "figures/source_data.csv",
)
_METADATA_FILES = frozenset(("artifact_manifest.json", "CHECKSUMS.sha256"))
_WORK_FILES = (
    "source_values.parquet",
    "fit_audits.csv",
    "rankings.csv",
    "ranking_stability.csv",
    "fixed_trajectories.parquet",
    "state_metrics.parquet",
    "cai_curves.csv",
    "image_curves.csv",
    "domain_metrics.csv",
    "budget_metrics.csv",
    "specimen_metrics.csv",
    "bootstrap.csv",
    "summary.json",
    "config.yaml",
)


@dataclass(frozen=True, slots=True)
class A4PackageValidation:
    global_mask_status: str
    a5_status: str
    output_tree_sha256: str
    manifest_sha256: str
    aggregation_state_sha256: str
    file_sha256: Mapping[str, str]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _records(output: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise A4ArtifactError("A4 package must not contain symlinks")
        if not path.is_file() or path.name in _METADATA_FILES:
            continue
        relative = path.relative_to(output).as_posix()
        payload = path.read_bytes()
        records[relative] = {"bytes": len(payload), "sha256": _sha(payload)}
    return records


def _tree_sha(records: Mapping[str, Mapping[str, object]]) -> str:
    return _sha(
        _json_bytes({name: dict(records[name]) for name in sorted(records)})
    )


def _check_privacy(output: Path, root: Path) -> None:
    forbidden = {root.as_posix().encode(), Path.home().as_posix().encode()}
    text_suffixes = {".csv", ".json", ".md", ".sha256", ".svg", ".yaml"}
    for path in output.rglob("*"):
        if path.is_file() and (
            path.suffix in text_suffixes or path.name == "CHECKSUMS.sha256"
        ):
            payload = path.read_bytes()
            if any(value and value in payload for value in forbidden):
                raise A4ArtifactError("A4 package contains a private absolute path")


def _base_config(root: Path, config: A4Config) -> tuple[Path, MVAConfig]:
    path = root / config.sources["a0_a3_config"].path
    return path, load_mva_config(path, project_root=root)


def _validate_a2(
    root: Path, config: A4Config, base_path: Path, base_config: MVAConfig
) -> MVAPackageValidation:
    validation = validate_mva_package(
        root / base_config.output_dir / "a2_oracle_value",
        project_root=root,
        config_path=base_path,
    )
    if (
        validation.status != "MVA_ORACLE_GO"
        or validation.manifest_sha256 != config.sources["a2_manifest"].sha256
    ):
        raise A4ArtifactError("bound A2 package changed")
    return validation


def _read_complete(path: Path, outer_domain: str) -> dict[str, object]:
    try:
        complete = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise A4ArtifactError(f"A4 outer shard is incomplete: {outer_domain}") from error
    expected_files = {
        "fit_audits.csv",
        "ranking_stability.csv",
        "rankings.csv",
        "source_values.parquet",
        "states.parquet",
        "trajectories.parquet",
    }
    hashes = complete.get("file_sha256")
    if (
        complete.get("outer_domain") != outer_domain
        or not isinstance(hashes, dict)
        or set(hashes) != expected_files
        or any(_sha_file(path / name) != hashes[name] for name in expected_files)
    ):
        raise A4ArtifactError(f"A4 outer shard digest changed: {outer_domain}")
    return complete


def _bootstrap_rows(aggregation: A4Aggregation, config: A4Config) -> list[dict[str, object]]:
    return [
        {
            "effect_id": effect.effect_id,
            "point_estimate": effect.point_estimate,
            "lower": effect.lower,
            "upper": effect.upper,
            "improved_domains": effect.improved_domains,
            "domain_effects": json.dumps(effect.domain_effects, separators=(",", ":")),
            "seed": config.bootstrap_seed,
            "resamples": config.bootstrap_resamples,
            "indices_sha256": effect.indices_sha256,
        }
        for effect in aggregation.bootstrap_effects
    ]


def _write_aggregation_tables(
    output: Path,
    *,
    aggregation: A4Aggregation,
    config: A4Config,
) -> None:
    curves = list(aggregation.curves)
    pl.DataFrame(curves, infer_schema_length=None).write_csv(output / "cai_curves.csv")
    image_rows = [
        row
        for row in curves
        if row["method"] in config.methods and row["protocol"] == "P-B"
    ]
    pl.DataFrame(image_rows, infer_schema_length=None).write_csv(
        output / "image_curves.csv"
    )
    pl.DataFrame(aggregation.domain_metrics, infer_schema_length=None).write_csv(
        output / "domain_metrics.csv"
    )
    pl.DataFrame(aggregation.budget_metrics, infer_schema_length=None).write_csv(
        output / "budget_metrics.csv"
    )
    pl.DataFrame(aggregation.specimen_metrics, infer_schema_length=None).write_csv(
        output / "specimen_metrics.csv"
    )
    pl.DataFrame(_bootstrap_rows(aggregation, config), infer_schema_length=None).write_csv(
        output / "bootstrap.csv"
    )


def aggregate_a4_shards(
    config_path: str | Path, *, project_root: str | Path
) -> Path:
    """Validate six outer shards and transactionally stage all A4 evidence tables."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_a4_config(config_file, project_root=root)
    base_path, base_config = _base_config(root, config)
    a2_validation = _validate_a2(root, config, base_path, base_config)
    shard_root = root / "results/mva/.work/a4_domains"
    states: list[pl.DataFrame] = []
    trajectories: list[pl.DataFrame] = []
    source_values: list[pl.DataFrame] = []
    fit_audits: list[pl.DataFrame] = []
    rankings: list[pl.DataFrame] = []
    stability: list[pl.DataFrame] = []
    outer_states: dict[str, dict[str, object]] = {}
    for domain in config.domain_order:
        leaf = shard_root / domain
        complete = _read_complete(leaf, domain)
        outer_states[domain] = {
            name: complete[name]
            for name in (
                "candidate_bank_state_sha256",
                "source_label_state_sha256",
                "evaluator_model_state_sha256",
                "evaluation_state_sha256",
                "target_specimen_count",
            )
        }
        states.append(pl.read_parquet(leaf / "states.parquet"))
        trajectories.append(pl.read_parquet(leaf / "trajectories.parquet"))
        source_values.append(pl.read_parquet(leaf / "source_values.parquet"))
        fit_audits.append(pl.read_csv(leaf / "fit_audits.csv"))
        rankings.append(pl.read_csv(leaf / "rankings.csv"))
        stability.append(pl.read_csv(leaf / "ranking_stability.csv"))
    a4_states = pl.concat(states, how="vertical_relaxed").sort(
        ["dataset_id", "specimen_id", "method", "nominal_checkpoint"]
    )
    a4_trajectories = pl.concat(trajectories, how="vertical_relaxed").sort(
        ["dataset_id", "specimen_id", "method", "ranking_position"]
    )
    source_table = pl.concat(source_values, how="vertical_relaxed").sort(
        ["outer_domain", "method", "dataset_id", "specimen_id", "cell_index"]
    )
    fit_table = pl.concat(fit_audits, how="vertical_relaxed")
    ranking_table = pl.concat(rankings, how="vertical_relaxed").sort(
        ["outer_domain", "method", "cell_index"]
    )
    stability_table = pl.concat(stability, how="vertical_relaxed").sort(
        ["outer_domain", "method", "removed_domain"]
    )
    reference = pl.read_parquet(
        root / base_config.output_dir / "a2_oracle_value/state_metrics.parquet"
    )
    aggregation = aggregate_a4_tables(
        a4_states,
        reference,
        domain_order=config.domain_order,
        checkpoints=config.checkpoints,
        random_seeds=base_config.random_seeds,
        full_mae=base_config.full_mae,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    destination = root / "results/mva/.work/a4_aggregate"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        source_table.write_parquet(
            temporary / "source_values.parquet", compression="zstd"
        )
        fit_table.write_csv(temporary / "fit_audits.csv")
        ranking_table.write_csv(temporary / "rankings.csv")
        stability_table.write_csv(temporary / "ranking_stability.csv")
        a4_trajectories.write_parquet(
            temporary / "fixed_trajectories.parquet", compression="zstd"
        )
        a4_states.write_parquet(temporary / "state_metrics.parquet", compression="zstd")
        _write_aggregation_tables(temporary, aggregation=aggregation, config=config)
        shutil.copyfile(config_file, temporary / "config.yaml")
        summary = {
            "schema_version": 1,
            "scope": config.scope,
            "global_mask_status": aggregation.gate.global_mask_status,
            "a5_status": aggregation.gate.a5_status,
            "aggregation_state_sha256": aggregation.state_sha256,
            "bootstrap_indices_sha256": aggregation.bootstrap_effects[0].indices_sha256,
            "a2_output_tree_sha256": a2_validation.output_tree_sha256,
            "outer_states": outer_states,
            "gate": asdict(aggregation.gate),
        }
        _write_json(temporary / "summary.json", summary)
        complete = {
            "aggregation_state_sha256": aggregation.state_sha256,
            "file_sha256": {
                name: _sha_file(temporary / name) for name in _WORK_FILES
            },
        }
        _write_json(temporary / "WORK_COMPLETE.json", complete)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _summary(output: Path, config: A4Config) -> dict[str, object]:
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise A4ArtifactError("A4 summary is invalid") from error
    if (
        summary.get("global_mask_status") not in config.a4_statuses
        or summary.get("a5_status") not in config.a5_statuses
        or summary.get("scope") != config.scope
    ):
        raise A4ArtifactError("A4 summary status changed")
    return summary


def _same_scalar(observed: object, expected: object) -> bool:
    if expected is None or observed is None:
        return observed is expected
    if (
        isinstance(observed, Real)
        and not isinstance(observed, bool)
        and isinstance(expected, Real)
        and not isinstance(expected, bool)
    ):
        return bool(
            float(observed) == float(expected)
            or abs(float(observed) - float(expected)) <= 1.0e-15
        )
    return observed == expected


def _check_derived_table(
    path: Path,
    expected: pl.DataFrame,
    *,
    keys: tuple[str, ...],
) -> None:
    try:
        observed = pl.read_csv(path)
        observed_rows = observed.sort(list(keys)).to_dicts()
        expected_rows = expected.sort(list(keys)).to_dicts()
    except (OSError, pl.exceptions.PolarsError) as error:
        raise A4ArtifactError(f"A4 derived evidence changed: {path.name}") from error
    if (
        observed.columns != expected.columns
        or len(observed_rows) != len(expected_rows)
        or any(
            set(actual) != set(wanted)
            or any(
                not _same_scalar(actual[column], wanted[column])
                for column in wanted
            )
            for actual, wanted in zip(observed_rows, expected_rows, strict=True)
        )
    ):
        raise A4ArtifactError(f"A4 derived evidence changed: {path.name}")


def _validate_aggregation_evidence(
    output: Path,
    *,
    root: Path,
    config: A4Config,
    base_config: MVAConfig,
    a2_validation: MVAPackageValidation,
    summary: Mapping[str, object],
) -> A4Aggregation:
    try:
        states = pl.read_parquet(output / "state_metrics.parquet")
        reference = pl.read_parquet(
            root / base_config.output_dir / "a2_oracle_value/state_metrics.parquet"
        )
        recomputed = aggregate_a4_tables(
            states,
            reference,
            domain_order=config.domain_order,
            checkpoints=config.checkpoints,
            random_seeds=base_config.random_seeds,
            full_mae=base_config.full_mae,
            bootstrap_seed=config.bootstrap_seed,
            bootstrap_resamples=config.bootstrap_resamples,
        )
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        raise A4ArtifactError("A4 derived evidence changed: state metrics") from error

    expected_summary_keys = {
        "schema_version",
        "scope",
        "global_mask_status",
        "a5_status",
        "aggregation_state_sha256",
        "bootstrap_indices_sha256",
        "a2_output_tree_sha256",
        "outer_states",
        "gate",
    }
    expected_gate = json.loads(_json_bytes(asdict(recomputed.gate)))
    if (
        set(summary) != expected_summary_keys
        or summary["schema_version"] != 1
        or summary["scope"] != config.scope
        or summary["global_mask_status"] != recomputed.gate.global_mask_status
        or summary["a5_status"] != recomputed.gate.a5_status
        or summary["aggregation_state_sha256"] != recomputed.state_sha256
        or summary["bootstrap_indices_sha256"]
        != recomputed.bootstrap_effects[0].indices_sha256
        or summary["a2_output_tree_sha256"] != a2_validation.output_tree_sha256
        or summary["gate"] != expected_gate
    ):
        raise A4ArtifactError("A4 derived evidence changed: summary")

    outer_states = summary["outer_states"]
    state_keys = {
        "candidate_bank_state_sha256",
        "source_label_state_sha256",
        "evaluator_model_state_sha256",
        "evaluation_state_sha256",
        "target_specimen_count",
    }
    if not isinstance(outer_states, dict) or set(outer_states) != set(
        config.domain_order
    ):
        raise A4ArtifactError("A4 derived evidence changed: outer states")
    for domain in config.domain_order:
        item = outer_states[domain]
        count = states.filter(pl.col("dataset_id") == domain)[
            "specimen_id"
        ].n_unique()
        if (
            not isinstance(item, dict)
            or set(item) != state_keys
            or item["target_specimen_count"] != count
            or any(
                not _is_sha256(item[name])
                for name in state_keys
                if name.endswith("sha256")
            )
        ):
            raise A4ArtifactError("A4 derived evidence changed: outer states")

    curves = pl.DataFrame(recomputed.curves, infer_schema_length=None)
    image = curves.filter(
        pl.col("method").is_in(list(config.methods))
        & (pl.col("protocol") == "P-B")
    )
    expected_tables = (
        ("cai_curves.csv", curves, ("protocol", "method", "nominal_checkpoint")),
        ("image_curves.csv", image, ("method", "nominal_checkpoint")),
        (
            "domain_metrics.csv",
            pl.DataFrame(recomputed.domain_metrics, infer_schema_length=None),
            ("dataset_id", "method"),
        ),
        (
            "budget_metrics.csv",
            pl.DataFrame(recomputed.budget_metrics, infer_schema_length=None),
            ("method",),
        ),
        (
            "specimen_metrics.csv",
            pl.DataFrame(recomputed.specimen_metrics, infer_schema_length=None),
            ("specimen_id", "method"),
        ),
        (
            "bootstrap.csv",
            pl.DataFrame(
                _bootstrap_rows(recomputed, config), infer_schema_length=None
            ),
            ("effect_id",),
        ),
    )
    for name, expected, keys in expected_tables:
        _check_derived_table(output / name, expected, keys=keys)
    return recomputed


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _check_required(output: Path) -> None:
    missing = [
        name
        for name in (*REQUIRED_A4_OUTPUTS, "config.yaml")
        if not (output / name).is_file()
    ]
    if missing:
        raise A4ArtifactError(f"required A4 outputs are missing: {missing}")


def publish_a4_manifest(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> A4PackageValidation:
    """Publish manifest/checksums after all formal A4 evidence exists."""

    root = Path(project_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_a4_config(config_file, project_root=root)
    base_path, base_config = _base_config(root, config)
    a2_validation = _validate_a2(root, config, base_path, base_config)
    _check_required(output)
    _check_privacy(output, root)
    if (output / "config.yaml").read_bytes() != config_file.read_bytes():
        raise A4ArtifactError("published A4 config differs from its authority")
    summary = _summary(output, config)
    _validate_aggregation_evidence(
        output,
        root=root,
        config=config,
        base_config=base_config,
        a2_validation=a2_validation,
        summary=summary,
    )
    records = _records(output)
    manifest = {
        "schema_version": 1,
        "scope": config.scope,
        "global_mask_status": summary["global_mask_status"],
        "a5_status": summary["a5_status"],
        "aggregation_state_sha256": summary["aggregation_state_sha256"],
        "config_sha256": _sha(config_file.read_bytes()),
        "sources": {
            name: {"path": source.path.as_posix(), "sha256": source.sha256}
            for name, source in sorted(config.sources.items())
        },
        "a2_output_tree_sha256": a2_validation.output_tree_sha256,
        "outputs": records,
        "output_tree_sha256": _tree_sha(records),
    }
    manifest_bytes = _json_bytes(manifest)
    (output / "artifact_manifest.json").write_bytes(manifest_bytes)
    checksum_records = {
        **{name: str(record["sha256"]) for name, record in records.items()},
        "artifact_manifest.json": _sha(manifest_bytes),
    }
    (output / "CHECKSUMS.sha256").write_text(
        "".join(
            f"{checksum_records[name]}  {name}\n" for name in sorted(checksum_records)
        ),
        encoding="ascii",
    )
    return validate_a4_package(output, project_root=root, config_path=config_file)


def validate_a4_package(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> A4PackageValidation:
    """Validate A4 package hashes, authority, statuses, and derived evidence."""

    root = Path(project_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_a4_config(config_file, project_root=root)
    base_path, base_config = _base_config(root, config)
    a2_validation = _validate_a2(root, config, base_path, base_config)
    _check_required(output)
    _check_privacy(output, root)
    if (output / "config.yaml").read_bytes() != config_file.read_bytes():
        raise A4ArtifactError("published A4 config differs from its authority")
    summary = _summary(output, config)
    try:
        manifest_payload = (output / "artifact_manifest.json").read_bytes()
        manifest = json.loads(manifest_payload)
        checksum_lines = (output / "CHECKSUMS.sha256").read_text(
            encoding="ascii"
        ).splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise A4ArtifactError("A4 artifact metadata cannot be read") from error
    records = _records(output)
    expected_manifest = {
        "schema_version": 1,
        "scope": config.scope,
        "global_mask_status": summary["global_mask_status"],
        "a5_status": summary["a5_status"],
        "aggregation_state_sha256": summary["aggregation_state_sha256"],
        "config_sha256": _sha(config_file.read_bytes()),
        "sources": {
            name: {"path": source.path.as_posix(), "sha256": source.sha256}
            for name, source in sorted(config.sources.items())
        },
        "a2_output_tree_sha256": a2_validation.output_tree_sha256,
        "outputs": records,
        "output_tree_sha256": _tree_sha(records),
    }
    if manifest != expected_manifest:
        raise A4ArtifactError("A4 artifact manifest changed")
    checksums: dict[str, str] = {}
    for line in checksum_lines:
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or parts[1] in checksums:
            raise A4ArtifactError("A4 CHECKSUMS.sha256 is invalid")
        checksums[parts[1]] = parts[0]
    expected_checksums = {
        **{name: str(record["sha256"]) for name, record in records.items()},
        "artifact_manifest.json": _sha(manifest_payload),
    }
    if checksums != expected_checksums:
        raise A4ArtifactError("A4 CHECKSUMS.sha256 mismatch")
    _validate_aggregation_evidence(
        output,
        root=root,
        config=config,
        base_config=base_config,
        a2_validation=a2_validation,
        summary=summary,
    )
    return A4PackageValidation(
        global_mask_status=str(summary["global_mask_status"]),
        a5_status=str(summary["a5_status"]),
        output_tree_sha256=str(manifest["output_tree_sha256"]),
        manifest_sha256=_sha(manifest_payload),
        aggregation_state_sha256=str(summary["aggregation_state_sha256"]),
        file_sha256=MappingProxyType(
            {name: str(record["sha256"]) for name, record in records.items()}
        ),
    )


def finalize_a4_package(
    work_dir: str | Path,
    destination_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> A4PackageValidation:
    """Validate in staging and atomically publish the immutable formal package."""

    work = Path(work_dir).resolve(strict=True)
    destination = Path(destination_dir).resolve()
    if work == destination:
        raise A4ArtifactError("A4 work and formal directories must differ")
    _check_required(work)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    )
    try:
        for relative in (*REQUIRED_A4_OUTPUTS, "config.yaml"):
            source = work / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        staged = publish_a4_manifest(
            staging,
            project_root=project_root,
            config_path=config_path,
        )
        if destination.exists():
            existing = validate_a4_package(
                destination,
                project_root=project_root,
                config_path=config_path,
            )
            if existing != staged:
                raise A4ArtifactError(
                    "existing formal A4 package differs from validated staging"
                )
            return existing
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    published = validate_a4_package(
        destination,
        project_root=project_root,
        config_path=config_path,
    )
    if published != staged:
        raise A4ArtifactError("formal A4 package changed during publication")
    return published


__all__ = [
    "REQUIRED_A4_OUTPUTS",
    "A4ArtifactError",
    "A4PackageValidation",
    "aggregate_a4_shards",
    "finalize_a4_package",
    "publish_a4_manifest",
    "validate_a4_package",
]
