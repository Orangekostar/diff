"""Aggregation, publication, and validation for the formal MVA A5 package."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import polars as pl

from .a5_config import A5Config, load_a5_config
from .a5_evaluation import A5Aggregation, aggregate_a5_tables
from .config import MVAConfig, load_mva_config
from .ranking_policy import load_policy_package


class A5ArtifactError(ValueError):
    """Raised when A5 evidence is incomplete, inconsistent, or modified."""


REQUIRED_A5_OUTPUTS = (
    "teacher_fit_audits.csv",
    "teacher_index.csv",
    "policy_training.csv",
    "target_trajectories.parquet",
    "state_metrics.parquet",
    "cai_curves.csv",
    "domain_metrics.csv",
    "budget_metrics.csv",
    "specimen_metrics.csv",
    "bootstrap.csv",
    "summary.json",
    "REPORT.md",
    "figures/A5_error_budget.png",
    "figures/A5_error_budget.svg",
    "figures/A5_domain_effects.png",
    "figures/A5_domain_effects.svg",
    "figures/A5_training_gap.png",
    "figures/A5_training_gap.svg",
    "figures/source_data.csv",
)
_MODEL_FILES = tuple(f"models/{domain}.npz" for domain in (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
))
_METADATA_FILES = frozenset(("artifact_manifest.json", "CHECKSUMS.sha256"))
_WORK_FILES = (
    "teacher_fit_audits.csv",
    "teacher_index.csv",
    "policy_training.csv",
    "target_trajectories.parquet",
    "state_metrics.parquet",
    "cai_curves.csv",
    "domain_metrics.csv",
    "budget_metrics.csv",
    "specimen_metrics.csv",
    "bootstrap.csv",
    "summary.json",
    "config.yaml",
    *_MODEL_FILES,
)


@dataclass(frozen=True, slots=True)
class A5PackageValidation:
    a5_status: str
    a6_status: str
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


def _base_config(root: Path, config: A5Config) -> tuple[Path, MVAConfig]:
    path = root / config.sources["a0_a3_config"].path
    return path, load_mva_config(path, project_root=root)


def _outer_complete(
    path: Path, *, outer_domain: str, config_sha256: str
) -> dict[str, object]:
    try:
        complete = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise A5ArtifactError(f"A5 outer shard is incomplete: {outer_domain}") from error
    expected_files = {
        "policy.npz",
        "policy_training.csv",
        "states.parquet",
        "teacher_fit_audits.csv",
        "teacher_index.csv",
        "trajectories.parquet",
    }
    hashes = complete.get("file_sha256")
    if (
        complete.get("outer_domain") != outer_domain
        or complete.get("config_sha256") != config_sha256
        or not isinstance(hashes, dict)
        or set(hashes) != expected_files
        or any(_sha_file(path / name) != hashes[name] for name in expected_files)
    ):
        raise A5ArtifactError(f"A5 outer shard digest changed: {outer_domain}")
    return complete


def _bootstrap_rows(
    aggregation: A5Aggregation, config: A5Config
) -> list[dict[str, object]]:
    return [
        {
            "effect_id": value.effect_id,
            "point_estimate": value.point_estimate,
            "lower": value.lower,
            "upper": value.upper,
            "improved_domains": value.improved_domains,
            "domain_effects": json.dumps(value.domain_effects, separators=(",", ":")),
            "seed": config.bootstrap_seed,
            "resamples": config.bootstrap_resamples,
            "indices_sha256": value.indices_sha256,
        }
        for value in aggregation.bootstrap_effects
    ]


def _write_derived(
    output: Path, *, aggregation: A5Aggregation, config: A5Config
) -> None:
    pl.DataFrame(aggregation.curves, infer_schema_length=None).write_csv(
        output / "cai_curves.csv"
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


def aggregate_a5_shards(
    config_path: str | Path, *, project_root: str | Path
) -> Path:
    """Validate six outer shards and transactionally stage A5 evidence."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_a5_config(config_file, project_root=root)
    _base_path, base_config = _base_config(root, config)
    config_sha256 = _sha_file(config_file)
    shard_root = root / config.work_dir / "domains"
    states: list[pl.DataFrame] = []
    trajectories: list[pl.DataFrame] = []
    fit_audits: list[pl.DataFrame] = []
    teacher_index: list[pl.DataFrame] = []
    training: list[pl.DataFrame] = []
    outer_states: dict[str, dict[str, object]] = {}
    complete_by_domain: dict[str, dict[str, object]] = {}
    for domain in config.domain_order:
        leaf = shard_root / domain
        complete = _outer_complete(
            leaf, outer_domain=domain, config_sha256=config_sha256
        )
        complete_by_domain[domain] = complete
        outer_states[domain] = {
            name: complete[name]
            for name in (
                "evaluator_model_state_sha256",
                "initial_budget",
                "policy_state_sha256",
                "source_specimen_count",
                "state_rows",
                "target_specimen_count",
                "teacher_model_state_sha256",
                "teacher_state_rows",
                "trajectory_rows",
                "training_state_count",
            )
        }
        states.append(pl.read_parquet(leaf / "states.parquet"))
        trajectories.append(pl.read_parquet(leaf / "trajectories.parquet"))
        fit_audits.append(pl.read_csv(leaf / "teacher_fit_audits.csv"))
        teacher_index.append(pl.read_csv(leaf / "teacher_index.csv"))
        training.append(pl.read_csv(leaf / "policy_training.csv"))
    state_table = pl.concat(states, how="vertical_relaxed").sort(
        ["dataset_id", "specimen_id", "method", "nominal_checkpoint"]
    )
    trajectory_table = pl.concat(trajectories, how="vertical_relaxed").sort(
        ["dataset_id", "specimen_id", "method", "step"]
    )
    fit_table = pl.concat(fit_audits, how="vertical_relaxed").sort(
        ["held_out_target_domain", "query_source_domain", "stage", "pca_dimension"]
    )
    teacher_table = pl.concat(teacher_index, how="vertical_relaxed").sort(
        ["outer_domain", "dataset_id", "specimen_id"]
    )
    training_table = pl.concat(training, how="vertical_relaxed").sort(
        ["outer_domain", "epoch"]
    )
    a4_states = pl.read_parquet(
        root / "results/mva/a4_global_task_mask/state_metrics.parquet"
    )
    a2_states = pl.read_parquet(
        root / "results/mva/a2_oracle_value/state_metrics.parquet"
    )
    aggregation = aggregate_a5_tables(
        state_table,
        a4_states,
        a2_states,
        domain_order=config.domain_order,
        checkpoints=config.checkpoints,
        random_seeds=base_config.random_seeds,
        full_mae=base_config.full_mae,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    destination = root / config.work_dir / "aggregate"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        (temporary / "models").mkdir()
        for domain in config.domain_order:
            shutil.copyfile(
                shard_root / domain / "policy.npz",
                temporary / "models" / f"{domain}.npz",
            )
        fit_table.write_csv(temporary / "teacher_fit_audits.csv")
        teacher_table.write_csv(temporary / "teacher_index.csv")
        training_table.write_csv(temporary / "policy_training.csv")
        trajectory_table.write_parquet(
            temporary / "target_trajectories.parquet", compression="zstd"
        )
        state_table.write_parquet(
            temporary / "state_metrics.parquet", compression="zstd"
        )
        _write_derived(temporary, aggregation=aggregation, config=config)
        shutil.copyfile(config_file, temporary / "config.yaml")
        _write_json(
            temporary / "summary.json",
            {
                "a5_status": aggregation.gate.a5_status,
                "a6_status": aggregation.gate.a6_status,
                "aggregation_state_sha256": aggregation.state_sha256,
                "bootstrap_indices_sha256": aggregation.bootstrap_effects[0].indices_sha256,
                "gate": asdict(aggregation.gate),
                "outer_states": outer_states,
                "schema_version": 1,
                "scope": config.scope,
            },
        )
        _write_json(
            temporary / "WORK_COMPLETE.json",
            {
                "aggregation_state_sha256": aggregation.state_sha256,
                "file_sha256": {
                    name: _sha_file(temporary / name) for name in _WORK_FILES
                },
            },
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _same_scalar(observed: object, expected: object) -> bool:
    if observed is None or expected is None:
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


def _check_table(path: Path, expected: pl.DataFrame, keys: tuple[str, ...]) -> None:
    try:
        observed = pl.read_csv(path)
        left = observed.sort(list(keys)).to_dicts()
        right = expected.sort(list(keys)).to_dicts()
    except (OSError, pl.exceptions.PolarsError) as error:
        raise A5ArtifactError(f"A5 derived table changed: {path.name}") from error
    if (
        observed.columns != expected.columns
        or len(left) != len(right)
        or any(
            set(actual) != set(wanted)
            or any(
                not _same_scalar(actual[name], wanted[name]) for name in wanted
            )
            for actual, wanted in zip(left, right, strict=True)
        )
    ):
        raise A5ArtifactError(f"A5 derived table changed: {path.name}")


def _summary(path: Path, config: A5Config) -> dict[str, object]:
    try:
        value = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise A5ArtifactError("A5 summary is invalid") from error
    if (
        value.get("a5_status") not in config.a5_statuses
        or value.get("a6_status") not in config.a6_statuses
        or value.get("scope") != config.scope
    ):
        raise A5ArtifactError("A5 summary status changed")
    return value


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_raw_evidence(
    output: Path,
    *,
    config: A5Config,
    outer_states: Mapping[str, object],
) -> None:
    try:
        teacher = pl.read_csv(output / "teacher_index.csv")
        audits = pl.read_csv(output / "teacher_fit_audits.csv")
        training = pl.read_csv(output / "policy_training.csv")
        trajectories = pl.read_parquet(output / "target_trajectories.parquet")
        states = pl.read_parquet(output / "state_metrics.parquet")
    except (OSError, pl.exceptions.PolarsError) as error:
        raise A5ArtifactError("A5 raw evidence cannot be read") from error
    outer_keys = {
        "evaluator_model_state_sha256",
        "initial_budget",
        "policy_state_sha256",
        "source_specimen_count",
        "state_rows",
        "target_specimen_count",
        "teacher_model_state_sha256",
        "teacher_state_rows",
        "trajectory_rows",
        "training_state_count",
    }
    if any(
        not isinstance(outer_states.get(domain), Mapping)
        or set(outer_states[domain]) != outer_keys
        for domain in config.domain_order
    ):
        raise A5ArtifactError("A5 outer evidence schema changed")
    for domain in config.domain_order:
        item = outer_states[domain]
        if (
            any(
                not _is_sha256(item[name])
                for name in (
                    "evaluator_model_state_sha256",
                    "policy_state_sha256",
                    "teacher_model_state_sha256",
                )
            )
            or float(item["initial_budget"]) not in {0.015625, 0.03125}
            or any(
                isinstance(item[name], bool)
                or not isinstance(item[name], int)
                or int(item[name]) <= 0
                for name in (
                    "source_specimen_count",
                    "state_rows",
                    "target_specimen_count",
                    "teacher_state_rows",
                    "trajectory_rows",
                    "training_state_count",
                )
            )
        ):
            raise A5ArtifactError("A5 outer evidence values changed")
    required_teacher = {
        "cache_sha256",
        "candidate_count",
        "dataset_id",
        "decision_state_count",
        "outer_domain",
        "predictor_state_sha256",
        "specimen_id",
        "state_count",
        "trajectory_state_sha256",
    }
    required_audits = {
        "fit_domains",
        "held_out_target_domain",
        "pca_dimension",
        "predictor_state_sha256",
        "query_domains",
        "query_source_domain",
        "stage",
    }
    required_training = {
        "epoch",
        "outer_domain",
        "policy_state_sha256",
        "source_specimen_count",
        "teacher_model_state_sha256",
        "training_state_count",
        "weighted_pairwise_loss",
    }
    required_trajectory = {
        "budget_after",
        "budget_before",
        "dataset_id",
        "from_level",
        "method",
        "nominal_checkpoint",
        "outer_domain",
        "policy_state_sha256",
        "specimen_id",
        "step",
        "to_level",
        "trajectory_state_sha256",
    }
    required_states = {
        "dataset_id",
        "method",
        "nominal_checkpoint",
        "outer_domain",
        "policy_state_sha256",
        "specimen_id",
        "trajectory_state_sha256",
    }
    if (
        not required_teacher <= set(teacher.columns)
        or not required_audits <= set(audits.columns)
        or not required_training <= set(training.columns)
        or not required_trajectory <= set(trajectories.columns)
        or not required_states <= set(states.columns)
    ):
        raise A5ArtifactError("A5 raw evidence schema changed")
    if (
        teacher.height != config.specimen_count * 5
        or teacher.unique(subset=["outer_domain", "specimen_id"]).height
        != teacher.height
        or set(teacher["outer_domain"]) != set(config.domain_order)
        or teacher.filter(pl.col("outer_domain") == pl.col("dataset_id")).height
        or teacher.filter(
            (pl.col("state_count") <= 0)
            | (pl.col("decision_state_count") <= 0)
            | (pl.col("decision_state_count") > pl.col("state_count"))
            | (pl.col("candidate_count") < pl.col("state_count"))
        ).height
        or any(
            not _is_sha256(value)
            for name in (
                "predictor_state_sha256",
                "trajectory_state_sha256",
                "cache_sha256",
            )
            for value in teacher[name]
        )
    ):
        raise A5ArtifactError("A5 teacher index changed")
    specimen_use = teacher.group_by("specimen_id").agg(
        pl.len().alias("rows"),
        pl.col("dataset_id").n_unique().alias("dataset_count"),
    )
    if (
        specimen_use.height != config.specimen_count
        or specimen_use.filter(
            (pl.col("rows") != 5) | (pl.col("dataset_count") != 1)
        ).height
    ):
        raise A5ArtifactError("A5 teacher specimen reuse roster changed")
    for domain in config.domain_order:
        item = outer_states[domain]
        teacher_domain = teacher.filter(pl.col("outer_domain") == domain)
        if (
            teacher_domain.height != int(item["source_specimen_count"])
            or int(teacher_domain["state_count"].sum())
            != int(item["teacher_state_rows"])
            or int(teacher_domain["decision_state_count"].sum())
            != int(item["training_state_count"])
        ):
            raise A5ArtifactError("A5 teacher outer roster changed")
    expected_pairs = {
        (outer_domain, query_domain)
        for outer_domain in config.domain_order
        for query_domain in config.domain_order
        if query_domain != outer_domain
    }
    expected_stage_counts = Counter(
        {
            (outer_domain, query_domain, stage): count
            for outer_domain, query_domain in expected_pairs
            for stage, count in (
                ("inner", len(config.pca_dimensions) * 4),
                ("outer", 1),
            )
        }
    )
    observed_stage_counts: Counter[tuple[str, str, str]] = Counter()
    fit_barrier_changed = False
    for row in audits.select(
        "held_out_target_domain",
        "query_source_domain",
        "stage",
        "query_domains",
        "fit_domains",
        "pca_dimension",
        "predictor_state_sha256",
    ).iter_rows(named=True):
        outer_domain = str(row["held_out_target_domain"])
        query_domain = str(row["query_source_domain"])
        stage = str(row["stage"])
        observed_stage_counts[(outer_domain, query_domain, stage)] += 1
        allowed = set(config.domain_order) - {outer_domain, query_domain}
        fit_domains = set(str(row["fit_domains"]).split("|"))
        query_domains = set(str(row["query_domains"]).split("|"))
        inner_valid = (
            stage == "inner"
            and len(query_domains) == 1
            and fit_domains.isdisjoint(query_domains)
            and fit_domains | query_domains == allowed
        )
        outer_valid = (
            stage == "outer"
            and query_domains == {query_domain}
            and fit_domains == allowed
        )
        if (
            not (inner_valid or outer_valid)
            or int(row["pca_dimension"]) not in config.pca_dimensions
            or not _is_sha256(row["predictor_state_sha256"])
        ):
            fit_barrier_changed = True
            break
    if (
        audits.is_empty()
        or observed_stage_counts != expected_stage_counts
        or audits.unique(
            subset=[
                "held_out_target_domain",
                "query_source_domain",
                "stage",
                "query_domains",
                "pca_dimension",
            ]
        ).height
        != audits.height
        or fit_barrier_changed
    ):
        raise A5ArtifactError("A5 teacher fit barrier changed")
    if (
        training.height != len(config.domain_order) * config.epochs
        or training.unique(subset=["outer_domain", "epoch"]).height
        != training.height
        or set(training["outer_domain"]) != set(config.domain_order)
        or not bool(training.select(pl.col("weighted_pairwise_loss").is_finite().all()).item())
    ):
        raise A5ArtifactError("A5 policy training record changed")
    for domain in config.domain_order:
        selected = training.filter(pl.col("outer_domain") == domain).sort("epoch")
        item = outer_states[domain]
        if (
            tuple(int(value) for value in selected["epoch"])
            != tuple(range(1, config.epochs + 1))
            or set(selected["policy_state_sha256"])
            != {item["policy_state_sha256"]}
            or set(selected["teacher_model_state_sha256"])
            != {item["teacher_model_state_sha256"]}
            or {int(value) for value in selected["source_specimen_count"]}
            != {int(item["source_specimen_count"])}
            or {int(value) for value in selected["training_state_count"]}
            != {int(item["training_state_count"])}
        ):
            raise A5ArtifactError("A5 policy epoch or digest changed")
    if (
        trajectories.is_empty()
        or set(trajectories["method"]) != set(config.methods)
        or set(trajectories["nominal_checkpoint"]) != set(config.checkpoints)
        or trajectories.filter(
            (pl.col("outer_domain") != pl.col("dataset_id"))
            | (pl.col("budget_after") <= pl.col("budget_before"))
            | (pl.col("to_level") != pl.col("from_level") + 1)
        ).height
    ):
        raise A5ArtifactError("A5 target trajectory changed")
    if (
        states.height
        != config.specimen_count * len(config.methods) * len(config.checkpoints)
        or states.unique(
            subset=["specimen_id", "method", "nominal_checkpoint"]
        ).height
        != states.height
        or set(states["method"]) != set(config.methods)
        or set(states["nominal_checkpoint"]) != set(config.checkpoints)
        or states.filter(pl.col("outer_domain") != pl.col("dataset_id")).height
    ):
        raise A5ArtifactError("A5 target state roster changed")
    action_groups = trajectories.group_by(
        ["outer_domain", "specimen_id", "method"]
    ).agg(
        pl.len().alias("rows"),
        pl.col("step").n_unique().alias("unique_steps"),
        pl.col("step").min().alias("minimum_step"),
        pl.col("step").max().alias("maximum_step"),
        pl.col("trajectory_state_sha256").n_unique().alias("trajectory_hashes"),
    )
    if action_groups.filter(
        (pl.col("rows") != pl.col("unique_steps"))
        | (pl.col("minimum_step") != 0)
        | (pl.col("maximum_step") != pl.col("rows") - 1)
        | (pl.col("trajectory_hashes") != 1)
    ).height:
        raise A5ArtifactError("A5 target action sequence changed")
    state_hashes = states.group_by(
        ["outer_domain", "specimen_id", "method"]
    ).agg(
        pl.col("trajectory_state_sha256").n_unique().alias("state_hash_count"),
        pl.col("trajectory_state_sha256").first().alias("state_hash"),
    )
    action_hashes = trajectories.group_by(
        ["outer_domain", "specimen_id", "method"]
    ).agg(pl.col("trajectory_state_sha256").first().alias("action_hash"))
    joined = state_hashes.join(
        action_hashes,
        on=["outer_domain", "specimen_id", "method"],
        how="full",
        coalesce=True,
    )
    if (
        joined.height != config.specimen_count * len(config.methods)
        or joined.filter(
            (pl.col("state_hash_count") != 1)
            | pl.col("action_hash").is_null()
            | (pl.col("state_hash") != pl.col("action_hash"))
        ).height
    ):
        raise A5ArtifactError("A5 target trajectory digest changed")
    for domain in config.domain_order:
        item = outer_states[domain]
        domain_states = states.filter(pl.col("outer_domain") == domain)
        domain_actions = trajectories.filter(pl.col("outer_domain") == domain)
        if (
            domain_states.height != int(item["state_rows"])
            or domain_actions.height != int(item["trajectory_rows"])
            or domain_states["specimen_id"].n_unique()
            != int(item["target_specimen_count"])
            or set(
                domain_states.filter(pl.col("method") == "imitation_policy")[
                    "policy_state_sha256"
                ]
            )
            != {item["policy_state_sha256"]}
            or set(
                domain_actions.filter(pl.col("method") == "imitation_policy")[
                    "policy_state_sha256"
                ]
            )
            != {item["policy_state_sha256"]}
            or domain_states.filter(
                (pl.col("method") != "imitation_policy")
                & pl.col("policy_state_sha256").is_not_null()
            ).height
            or domain_actions.filter(
                (pl.col("method") != "imitation_policy")
                & pl.col("policy_state_sha256").is_not_null()
            ).height
        ):
            raise A5ArtifactError("A5 target outer evidence changed")


def _validate_derived(
    output: Path,
    *,
    root: Path,
    config: A5Config,
    base_config: MVAConfig,
    summary: Mapping[str, object],
) -> A5Aggregation:
    try:
        recomputed = aggregate_a5_tables(
            pl.read_parquet(output / "state_metrics.parquet"),
            pl.read_parquet(
                root / "results/mva/a4_global_task_mask/state_metrics.parquet"
            ),
            pl.read_parquet(root / "results/mva/a2_oracle_value/state_metrics.parquet"),
            domain_order=config.domain_order,
            checkpoints=config.checkpoints,
            random_seeds=base_config.random_seeds,
            full_mae=base_config.full_mae,
            bootstrap_seed=config.bootstrap_seed,
            bootstrap_resamples=config.bootstrap_resamples,
        )
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        raise A5ArtifactError("A5 state evidence cannot be recomputed") from error
    expected_summary = {
        "a5_status": recomputed.gate.a5_status,
        "a6_status": recomputed.gate.a6_status,
        "aggregation_state_sha256": recomputed.state_sha256,
        "bootstrap_indices_sha256": recomputed.bootstrap_effects[0].indices_sha256,
        "gate": json.loads(_json_bytes(asdict(recomputed.gate))),
        "outer_states": summary.get("outer_states"),
        "schema_version": 1,
        "scope": config.scope,
    }
    if dict(summary) != expected_summary:
        raise A5ArtifactError("A5 derived summary changed")
    outer_states = summary.get("outer_states")
    if not isinstance(outer_states, dict) or set(outer_states) != set(config.domain_order):
        raise A5ArtifactError("A5 outer summary roster changed")
    _validate_raw_evidence(output, config=config, outer_states=outer_states)
    for domain in config.domain_order:
        item = outer_states[domain]
        policy = load_policy_package(output / "models" / f"{domain}.npz")
        if (
            not isinstance(item, dict)
            or item.get("policy_state_sha256") != policy.state_sha256
            or set(policy.source_domains) != set(config.domain_order) - {domain}
        ):
            raise A5ArtifactError("A5 policy package source roster changed")
    expected_tables = (
        (
            "cai_curves.csv",
            pl.DataFrame(recomputed.curves, infer_schema_length=None),
            ("protocol", "method", "nominal_checkpoint"),
        ),
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
            pl.DataFrame(_bootstrap_rows(recomputed, config), infer_schema_length=None),
            ("effect_id",),
        ),
    )
    for name, expected, keys in expected_tables:
        _check_table(output / name, expected, keys)
    return recomputed


def _records(output: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise A5ArtifactError("A5 package cannot contain symlinks")
        if not path.is_file() or path.name in _METADATA_FILES:
            continue
        relative = path.relative_to(output).as_posix()
        payload = path.read_bytes()
        records[relative] = {"bytes": len(payload), "sha256": _sha(payload)}
    return records


def _tree_sha(records: Mapping[str, Mapping[str, object]]) -> str:
    return _sha(_json_bytes({name: dict(records[name]) for name in sorted(records)}))


def _required(output: Path) -> None:
    missing = [
        name
        for name in (*REQUIRED_A5_OUTPUTS, *_MODEL_FILES, "config.yaml")
        if not (output / name).is_file()
    ]
    if missing:
        raise A5ArtifactError(f"required A5 outputs are missing: {missing}")


def _privacy(output: Path, root: Path) -> None:
    forbidden = {root.as_posix().encode(), Path.home().as_posix().encode()}
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in {".csv", ".json", ".md", ".svg", ".yaml"}:
            payload = path.read_bytes()
            if any(value and value in payload for value in forbidden):
                raise A5ArtifactError("A5 package contains a private absolute path")


def publish_a5_manifest(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> A5PackageValidation:
    """Publish A5 manifest and checksums after all evidence is rendered."""

    root = Path(project_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_a5_config(config_file, project_root=root)
    _base_path, base_config = _base_config(root, config)
    _required(output)
    _privacy(output, root)
    if (output / "config.yaml").read_bytes() != config_file.read_bytes():
        raise A5ArtifactError("published A5 config differs from authority")
    summary = _summary(output, config)
    _validate_derived(
        output,
        root=root,
        config=config,
        base_config=base_config,
        summary=summary,
    )
    records = _records(output)
    manifest = {
        "a5_status": summary["a5_status"],
        "a6_status": summary["a6_status"],
        "aggregation_state_sha256": summary["aggregation_state_sha256"],
        "config_sha256": _sha(config_file.read_bytes()),
        "output_tree_sha256": _tree_sha(records),
        "outputs": records,
        "schema_version": 1,
        "scope": config.scope,
        "sources": {
            name: {"path": value.path.as_posix(), "sha256": value.sha256}
            for name, value in sorted(config.sources.items())
        },
    }
    manifest_bytes = _json_bytes(manifest)
    (output / "artifact_manifest.json").write_bytes(manifest_bytes)
    checksums = {
        **{name: str(value["sha256"]) for name, value in records.items()},
        "artifact_manifest.json": _sha(manifest_bytes),
    }
    (output / "CHECKSUMS.sha256").write_text(
        "".join(f"{checksums[name]}  {name}\n" for name in sorted(checksums)),
        encoding="ascii",
    )
    return validate_a5_package(output, project_root=root, config_path=config_file)


def validate_a5_package(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> A5PackageValidation:
    """Validate A5 hashes, policies, authority, statuses, and derived evidence."""

    root = Path(project_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_a5_config(config_file, project_root=root)
    _base_path, base_config = _base_config(root, config)
    _required(output)
    _privacy(output, root)
    if (output / "config.yaml").read_bytes() != config_file.read_bytes():
        raise A5ArtifactError("published A5 config differs from authority")
    summary = _summary(output, config)
    aggregation = _validate_derived(
        output,
        root=root,
        config=config,
        base_config=base_config,
        summary=summary,
    )
    try:
        manifest_payload = (output / "artifact_manifest.json").read_bytes()
        manifest = json.loads(manifest_payload)
        lines = (output / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise A5ArtifactError("A5 manifest or checksums cannot be read") from error
    records = _records(output)
    expected_manifest = {
        "a5_status": summary["a5_status"],
        "a6_status": summary["a6_status"],
        "aggregation_state_sha256": summary["aggregation_state_sha256"],
        "config_sha256": _sha(config_file.read_bytes()),
        "output_tree_sha256": _tree_sha(records),
        "outputs": records,
        "schema_version": 1,
        "scope": config.scope,
        "sources": {
            name: {"path": value.path.as_posix(), "sha256": value.sha256}
            for name, value in sorted(config.sources.items())
        },
    }
    expected_checksums = {
        **{name: str(value["sha256"]) for name, value in records.items()},
        "artifact_manifest.json": _sha(manifest_payload),
    }
    parsed: dict[str, str] = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise A5ArtifactError("A5 checksum syntax changed") from error
        parsed[name] = digest
    if manifest != expected_manifest or parsed != expected_checksums:
        raise A5ArtifactError("A5 manifest or file digest changed")
    return A5PackageValidation(
        a5_status=aggregation.gate.a5_status,
        a6_status=aggregation.gate.a6_status,
        output_tree_sha256=str(manifest["output_tree_sha256"]),
        manifest_sha256=_sha(manifest_payload),
        aggregation_state_sha256=aggregation.state_sha256,
        file_sha256=MappingProxyType(
            {name: str(value["sha256"]) for name, value in records.items()}
        ),
    )


def finalize_a5_package(
    staging_dir: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> A5PackageValidation:
    """Atomically publish a fully rendered A5 package."""

    root = Path(project_root).resolve(strict=True)
    staging = Path(staging_dir).resolve(strict=True)
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise A5ArtifactError("A5 destination already exists")
    temporary = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        for item in staging.iterdir():
            if item.name == "WORK_COMPLETE.json":
                continue
            target = temporary / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copyfile(item, target)
        validation = publish_a5_manifest(
            temporary, project_root=root, config_path=config_path
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    final = validate_a5_package(
        destination, project_root=root, config_path=config_path
    )
    if final.output_tree_sha256 != validation.output_tree_sha256:
        raise A5ArtifactError("published A5 package changed during finalization")
    return final


__all__ = [
    "REQUIRED_A5_OUTPUTS",
    "A5ArtifactError",
    "A5PackageValidation",
    "aggregate_a5_shards",
    "finalize_a5_package",
    "publish_a5_manifest",
    "validate_a5_package",
]
