"""Finalize and verify the MAVIS P1 state-bank artifact package."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from .authority import load_mavis_authority
from .config import load_mavis_config
from .state_bank_execution import _validate_existing_shard


class MAVISStateBankPackageError(RuntimeError):
    """Raised when P1 shards or their formal package are incomplete."""


_CODE_PATHS = (
    "src/cmc_bbdm/mavis/authority.py",
    "src/cmc_bbdm/mavis/config.py",
    "src/cmc_bbdm/mavis/contracts.py",
    "src/cmc_bbdm/mavis/reveal.py",
    "src/cmc_bbdm/mavis/state_bank.py",
    "src/cmc_bbdm/mavis/state_bank_artifacts.py",
    "src/cmc_bbdm/mavis/state_bank_execution.py",
    "src/cmc_bbdm/mavis/state_bank_package.py",
    "src/cmc_bbdm/mavis/state_candidates.py",
    "src/cmc_bbdm/mavis/teacher.py",
    "src/cmc_bbdm/mavis/trajectory_sources.py",
)
_MEASUREMENT_COLUMNS = (
    "revealed_rows",
    "revealed_columns",
    "revealed_red",
    "revealed_green",
    "revealed_blue",
)
_GIT_BLOB_LIMIT = 100 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _repository_state(root: Path) -> tuple[str, str, bool]:
    digest = hashlib.sha256()
    for relative in _CODE_PATHS:
        path = root / relative
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise MAVISStateBankPackageError(
                f"P1 runtime source is unavailable: {relative}"
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
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise MAVISStateBankPackageError("P1 Git provenance is unavailable") from error
    if len(git_sha) != 40:
        raise MAVISStateBankPackageError("P1 Git SHA is invalid")
    return git_sha, digest.hexdigest(), bool(status)


def _worker_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISStateBankPackageError("P1 worker summary is invalid") from error
    if type(payload) is not dict:
        raise MAVISStateBankPackageError("P1 worker summary is invalid")
    return payload


def _validate_tables(
    states: pl.DataFrame,
    actions: pl.DataFrame,
    audits: pl.DataFrame,
    *,
    specimen_count: int,
    authority_state_sha256: str,
) -> dict[str, int]:
    expected_states = specimen_count * 5 * 6
    if (
        states.height != expected_states
        or states.get_column("state_id").n_unique() != expected_states
        or states.get_column("specimen_id").n_unique() != specimen_count
        or states.get_column("authority_state_sha256").unique().to_list()
        != [authority_state_sha256]
        or actions.height <= 0
        or actions.get_column("authority_state_sha256").unique().to_list()
        != [authority_state_sha256]
        or audits.height != 30
    ):
        raise MAVISStateBankPackageError("P1 table roster is incomplete")
    per_specimen = states.group_by("specimen_id").len().get_column("len")
    per_method = states.group_by("specimen_id", "method").len().get_column("len")
    if per_specimen.unique().to_list() != [30] or per_method.unique().to_list() != [6]:
        raise MAVISStateBankPackageError("P1 state hierarchy changed")
    monotone = (
        states.sort(["trajectory_id", "step"])
        .group_by("trajectory_id")
        .agg(pl.col("exact_acquired_cost").is_sorted().alias("ok"))
        .get_column("ok")
    )
    list_contract = states.select(
        (
            (pl.col("context_features").list.len() == 34)
            & (pl.col("measurement_levels").list.len() == 64)
            & (
                pl.col("revealed_rows").list.len()
                == pl.col("exact_acquired_cost")
            )
            & (
                pl.col("revealed_columns").list.len()
                == pl.col("exact_acquired_cost")
            )
            & (pl.col("revealed_red").list.len() == pl.col("exact_acquired_cost"))
            & (
                pl.col("revealed_green").list.len()
                == pl.col("exact_acquired_cost")
            )
            & (
                pl.col("revealed_blue").list.len()
                == pl.col("exact_acquired_cost")
            )
            & (pl.col("acquired_action_cell_indices").list.len() == pl.col("step"))
            & (pl.col("teacher_outer_domains").list.len() == 5)
            & (pl.col("strict_oof_cai_predictions").list.len() == 5)
            & (pl.col("teacher_state_sha256").list.len() == 5)
            & (pl.col("teacher_predictor_state_sha256").list.len() == 5)
            & (
                pl.col("candidate_cell_indices").list.len()
                == pl.col("candidate_exact_added_costs").list.len()
            )
        ).all()
    ).item()
    if not monotone.all() or not list_contract:
        raise MAVISStateBankPackageError("P1 state causality contract failed")
    missing_states = actions.join(states.select("state_id"), on="state_id", how="anti")
    multiplicity = (
        actions.group_by("state_id", "candidate_index").len().get_column("len")
    )
    action_check = actions.join(
        states.select("state_id", "exact_acquired_cost", "native_count"),
        on="state_id",
        how="left",
    ).select(
        (
            (pl.col("outer_domain") != pl.col("domain_id"))
            & (pl.col("exact_added_cost") > 0)
            & (
                pl.col("candidate_exact_cost_after")
                == pl.col("exact_acquired_cost") + pl.col("exact_added_cost")
            )
            & (
                (
                    pl.col("candidate_effective_budget_after")
                    - pl.col("candidate_exact_cost_after") / pl.col("native_count")
                ).abs()
                <= 1.0e-15
            )
        ).all()
    ).item()
    if (
        missing_states.height != 0
        or multiplicity.unique().to_list() != [5]
        or not action_check
    ):
        raise MAVISStateBankPackageError("P1 state-action linkage failed")
    audit_check = audits.select(
        (
            (~pl.col("fit_domains").list.contains(pl.col("held_out_target_domain")))
            & (~pl.col("fit_domains").list.contains(pl.col("query_source_domain")))
            & (
                pl.col("fit_specimen_ids")
                .list.set_intersection(pl.col("query_specimen_ids"))
                .list.len()
                == 0
            )
            & (pl.col("fit_domains").list.len() == 4)
        ).all()
    ).item()
    if not audit_check:
        raise MAVISStateBankPackageError("P1 strict-OOF audit failed")
    numeric = actions.select(pl.selectors.numeric())
    if numeric.select(pl.any_horizontal(pl.all().is_nan().any())).item():
        raise MAVISStateBankPackageError("P1 action table contains NaN")
    terminal_states = states.join(
        actions.select("state_id").unique(),
        on="state_id",
        how="anti",
    ).height
    return {
        "state_count": states.height,
        "state_action_pair_count": actions.height,
        "terminal_state_count": terminal_states,
        "teacher_fit_count": audits.height,
    }


def write_compact_state_manifest(
    states: pl.DataFrame,
    output: str | Path,
    *,
    specimens_per_part: int = 8,
) -> None:
    """Write a compact state index plus lossless bounded measurement payloads."""

    root = Path(output)
    required = {"state_id", "domain_id", "specimen_id", *_MEASUREMENT_COLUMNS}
    if (
        not isinstance(states, pl.DataFrame)
        or states.height == 0
        or not required <= set(states.columns)
        or states.get_column("state_id").n_unique() != states.height
        or type(specimens_per_part) is not int
        or specimens_per_part <= 0
        or (root / "state_manifest.parquet").exists()
        or (root / "revealed_measurements").exists()
    ):
        raise MAVISStateBankPackageError("P1 compact state request is invalid")
    root.mkdir(parents=True, exist_ok=True)
    payload_root = root / "revealed_measurements"
    payload_root.mkdir()
    specimens = (
        states.select("domain_id", "specimen_id")
        .unique()
        .sort(["domain_id", "specimen_id"])
    )
    mappings: list[pl.DataFrame] = []
    for part_index, start in enumerate(
        range(0, specimens.height, specimens_per_part)
    ):
        roster = specimens.slice(start, specimens_per_part)
        selected = states.join(
            roster,
            on=["domain_id", "specimen_id"],
            how="semi",
        ).sort(["domain_id", "specimen_id", "method", "nominal_checkpoint"])
        if selected.height == 0:
            raise MAVISStateBankPackageError("P1 measurement payload is empty")
        relative = f"revealed_measurements/part-{part_index:04d}.parquet"
        destination = root / relative
        selected.select("state_id", *_MEASUREMENT_COLUMNS).write_parquet(
            destination,
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        if destination.stat().st_size >= _GIT_BLOB_LIMIT:
            raise MAVISStateBankPackageError(
                f"P1 measurement payload exceeds Git blob limit: {relative}"
            )
        mappings.append(
            selected.select("state_id").with_columns(
                pl.lit(relative).alias("measurement_payload_file")
            )
        )
    mapping = pl.concat(mappings)
    if (
        mapping.height != states.height
        or mapping.get_column("state_id").n_unique() != states.height
    ):
        raise MAVISStateBankPackageError("P1 measurement payload mapping is incomplete")
    compact = (
        states.drop(_MEASUREMENT_COLUMNS)
        .join(mapping, on="state_id", how="left", validate="1:1")
        .sort(["domain_id", "specimen_id", "method", "nominal_checkpoint"])
    )
    if compact.get_column("measurement_payload_file").null_count() != 0:
        raise MAVISStateBankPackageError("P1 compact state index is incomplete")
    compact.write_parquet(
        root / "state_manifest.parquet",
        compression="zstd",
        compression_level=12,
        statistics=True,
    )


def write_partitioned_action_pairs(actions: pl.DataFrame, output: str | Path) -> None:
    """Write state-action labels in bounded teacher/query-domain partitions."""

    root = Path(output)
    required = {"outer_domain", "domain_id", "state_id", "candidate_index"}
    action_root = root / "state_action_pairs"
    if (
        not isinstance(actions, pl.DataFrame)
        or actions.height == 0
        or not required <= set(actions.columns)
        or action_root.exists()
    ):
        raise MAVISStateBankPackageError("P1 partitioned action request is invalid")
    action_root.mkdir(parents=True)
    partition_count = 0
    for (outer_domain, query_domain), table in actions.group_by(
        "outer_domain",
        "domain_id",
        maintain_order=True,
    ):
        if (
            type(outer_domain) is not str
            or type(query_domain) is not str
            or not outer_domain.isalnum()
            or not query_domain.isalnum()
            or outer_domain == query_domain
        ):
            raise MAVISStateBankPackageError("P1 action partition identity is invalid")
        relative = f"state_action_pairs/{outer_domain}__{query_domain}.parquet"
        destination = root / relative
        table.sort(["specimen_id", "state_id", "candidate_index"]).write_parquet(
            destination,
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        if destination.stat().st_size >= _GIT_BLOB_LIMIT:
            raise MAVISStateBankPackageError(
                f"P1 action partition exceeds Git blob limit: {relative}"
            )
        partition_count += 1
    expected = (
        actions.select("outer_domain", "domain_id").unique().height
    )
    if partition_count != expected:
        raise MAVISStateBankPackageError("P1 action partitions are incomplete")


def load_state_action_pairs_package(path: str | Path) -> pl.DataFrame:
    """Load every registered P1 teacher/query action partition."""

    action_root = Path(path) / "state_action_pairs"
    parts = sorted(action_root.glob("*.parquet")) if action_root.is_dir() else []
    if not parts or any("__" not in part.stem for part in parts):
        raise MAVISStateBankPackageError("P1 action partition roster is invalid")
    try:
        actions = pl.read_parquet(parts)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise MAVISStateBankPackageError("P1 action partitions are invalid") from error
    if (
        actions.height == 0
        or not {"outer_domain", "domain_id", "state_id", "candidate_index"}
        <= set(actions.columns)
    ):
        raise MAVISStateBankPackageError("P1 action partitions are invalid")
    expected = {
        f"{outer}__{query}.parquet"
        for outer, query in actions.select("outer_domain", "domain_id")
        .unique()
        .iter_rows()
    }
    if {part.name for part in parts} != expected:
        raise MAVISStateBankPackageError("P1 action partition roster changed")
    return actions


def load_state_manifest_package(path: str | Path) -> pl.DataFrame:
    """Restore the full P1 state rows from a verified compact package."""

    root = Path(path)
    try:
        compact = pl.read_parquet(root / "state_manifest.parquet")
    except (OSError, pl.exceptions.PolarsError) as error:
        raise MAVISStateBankPackageError("P1 compact state manifest is invalid") from error
    if (
        compact.height == 0
        or "measurement_payload_file" not in compact.columns
        or compact.get_column("state_id").n_unique() != compact.height
        or compact.get_column("measurement_payload_file").null_count() != 0
    ):
        raise MAVISStateBankPackageError("P1 compact state manifest is invalid")
    relative_files = tuple(
        sorted(compact.get_column("measurement_payload_file").unique().to_list())
    )
    if any(
        type(value) is not str
        or not value.startswith("revealed_measurements/")
        or Path(value).is_absolute()
        or ".." in Path(value).parts
        for value in relative_files
    ):
        raise MAVISStateBankPackageError("P1 measurement payload path is invalid")
    expected = {Path(value).name for value in relative_files}
    payload_root = root / "revealed_measurements"
    actual = (
        {value.name for value in payload_root.glob("*.parquet")}
        if payload_root.is_dir()
        else set()
    )
    if actual != expected:
        raise MAVISStateBankPackageError("P1 measurement payload roster changed")
    try:
        payloads = pl.read_parquet([root / value for value in relative_files])
    except (OSError, pl.exceptions.PolarsError) as error:
        raise MAVISStateBankPackageError("P1 measurement payload is invalid") from error
    if (
        payloads.height != compact.height
        or payloads.get_column("state_id").n_unique() != compact.height
        or not {"state_id", *_MEASUREMENT_COLUMNS} <= set(payloads.columns)
    ):
        raise MAVISStateBankPackageError("P1 measurement payload linkage changed")
    restored = compact.drop("measurement_payload_file").join(
        payloads,
        on="state_id",
        how="left",
        validate="1:1",
    )
    if any(restored.get_column(name).null_count() for name in _MEASUREMENT_COLUMNS):
        raise MAVISStateBankPackageError("P1 restored measurement rows are incomplete")
    return restored


def _write_checksums(output: Path) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    lines = [f"{_sha256(path)}  {path.relative_to(output).as_posix()}" for path in files]
    (output / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def finalize_state_bank_package(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_project_root: str | Path,
) -> Path:
    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_mavis_config(config_file, project_root=root)
    authority = load_mavis_authority(
        config,
        source_project_root=source_project_root,
    )
    work = root / config.output_root / ".work/p1_state_bank"
    state_directory = work / "states"
    action_directory = work / "state_action_pairs"
    state_paths: list[Path] = []
    action_paths: list[Path] = []
    workers: list[dict[str, object]] = []
    audit_paths: list[Path] = []
    for domain_id in config.domain_order:
        summary_path = work / f"worker__{domain_id}.json"
        payload = _worker_payload(summary_path)
        expected_specimens = tuple(
            specimen_id
            for specimen_id, dataset_id in zip(
                authority.specimen_ids,
                authority.dataset_ids,
                strict=True,
            )
            if dataset_id == domain_id
        )
        if (
            payload.get("domain_id") != domain_id
            or payload.get("config_sha256") != config.config_sha256
            or payload.get("authority_state_sha256") != authority.state_sha256
            or payload.get("specimen_count") != len(expected_specimens)
            or [item["specimen_id"] for item in payload.get("specimens", [])]
            != list(expected_specimens)
        ):
            raise MAVISStateBankPackageError("P1 worker roster is incomplete")
        for specimen_id in expected_specimens:
            token = specimen_id.replace("/", "_")
            state_path = state_directory / f"{domain_id}__{token}.parquet"
            action_path = action_directory / f"{domain_id}__{token}.parquet"
            if not _validate_existing_shard(
                state_path,
                action_path,
                specimen_id=specimen_id,
                dataset_id=domain_id,
                config=config,
                authority=authority,
            ):
                raise MAVISStateBankPackageError("P1 specimen shard is invalid")
            state_paths.append(state_path)
            action_paths.append(action_path)
        workers.append(payload)
        audit_paths.append(work / f"teacher_fit_audits__{domain_id}.parquet")
    expected_state_names = {path.name for path in state_paths}
    expected_action_names = {path.name for path in action_paths}
    if (
        {path.name for path in state_directory.glob("*.parquet")}
        != expected_state_names
        or {path.name for path in action_directory.glob("*.parquet")}
        != expected_action_names
        or any(not path.is_file() for path in audit_paths)
        or len({_sha256(path) for path in audit_paths}) != 1
    ):
        raise MAVISStateBankPackageError("P1 work directory has missing or extra shards")
    try:
        states = pl.read_parquet(state_paths).sort(
            ["domain_id", "specimen_id", "method", "nominal_checkpoint"]
        )
        actions = pl.read_parquet(action_paths).sort(
            [
                "outer_domain",
                "domain_id",
                "specimen_id",
                "method",
                "nominal_checkpoint",
                "candidate_index",
            ]
        )
        audits = pl.read_parquet(audit_paths[0]).sort(
            ["held_out_target_domain", "query_source_domain"]
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise MAVISStateBankPackageError("P1 shards cannot be combined") from error
    counts = _validate_tables(
        states,
        actions,
        audits,
        specimen_count=authority.specimen_count,
        authority_state_sha256=authority.state_sha256,
    )
    output = root / config.output_root / "p1_state_bank"
    if output.exists():
        raise MAVISStateBankPackageError("P1 formal package already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p1_state_bank.", dir=output.parent))
    try:
        write_compact_state_manifest(states, temporary)
        write_partitioned_action_pairs(actions, temporary)
        audits.write_parquet(
            temporary / "teacher_fit_audits.parquet",
            compression="zstd",
            statistics=True,
        )
        shutil.copyfile(config_file, temporary / "config.yaml")
        _json(temporary / "worker_manifest.json", {"workers": workers})
        git_sha, code_state_sha256, git_dirty = _repository_state(root)
        state_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "schema": 1,
                    "config_sha256": config.config_sha256,
                    "authority_state_sha256": authority.state_sha256,
                    "state_manifest_sha256": _sha256(
                        temporary / "state_manifest.parquet"
                    ),
                    "state_action_pair_partitions": {
                        path.relative_to(temporary).as_posix(): _sha256(path)
                        for path in sorted(
                            (temporary / "state_action_pairs").glob("*.parquet")
                        )
                    },
                    "teacher_fit_audits_sha256": _sha256(
                        temporary / "teacher_fit_audits.parquet"
                    ),
                    "measurement_payloads": {
                        path.relative_to(temporary).as_posix(): _sha256(path)
                        for path in sorted(
                            (temporary / "revealed_measurements").glob("*.parquet")
                        )
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        per_domain = (
            states.group_by("domain_id")
            .agg(
                pl.col("specimen_id").n_unique().alias("specimen_count"),
                pl.len().alias("state_count"),
            )
            .join(
                actions.group_by("domain_id").len("state_action_pair_count"),
                on="domain_id",
            )
            .sort("domain_id")
            .to_dicts()
        )
        summary = {
            "schema_version": 1,
            "stage": "P1_STRICT_OOF_STATE_BANK",
            "status": "COMPLETE",
            "git_sha": git_sha,
            "git_worktree_dirty_at_run": git_dirty,
            "runtime_code_state_sha256": code_state_sha256,
            "config_sha256": config.config_sha256,
            "authority_state_sha256": authority.state_sha256,
            "state_bank_state_sha256": state_sha256,
            "specimen_count": authority.specimen_count,
            "domain_count": len(config.domain_order),
            "trajectory_methods": [
                "random",
                "uniform",
                "reconstruction_driven",
                "one_shot_mechanical_oracle",
                "sequential_mechanical_oracle",
            ],
            "checkpoint_count": len(config.checkpoints),
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "measurement_payload_part_count": len(
                list((temporary / "revealed_measurements").glob("*.parquet"))
            ),
            "state_action_pair_part_count": len(
                list((temporary / "state_action_pairs").glob("*.parquet"))
            ),
            **counts,
            "per_domain": per_domain,
        }
        _json(temporary / "summary.json", summary)
        report = (
            "# MAVIS P1 Strict-OOF Sequential State Bank\n\n"
            "Status: `COMPLETE`.\n\n"
            f"The package contains `{counts['state_count']}` causal states and "
            f"`{counts['state_action_pair_count']}` strict-OOF state-action labels "
            f"for `{authority.specimen_count}` physical specimens across six "
            "leave-one-domain-out domains. Five frozen trajectory families and six "
            "registered checkpoints are present for every specimen.\n\n"
            "Every teacher fit excludes both the held-out target domain and the "
            "query source domain. Source true CAI appears only in privileged "
            "state-action teacher rows; policy-visible state rows contain only "
            "initial context, actually revealed measurements, exact costs, legal "
            "candidate geometry/cost, and strict-OOF predictions.\n\n"
            f"`{counts['terminal_state_count']}` terminal checkpoint states have no "
            "legal next action under the exact 25% endpoint and therefore have no "
            "state-action rows. Statistical inference remains specimen/domain-level; "
            "state-action rows are training samples, not independent experimental "
            "replicates.\n"
        )
        (temporary / "REPORT.md").write_text(report, encoding="utf-8")
        scientific_files = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file() and path.name not in {"artifact_manifest.json"}
        )
        manifest = {
            "schema_version": 1,
            "artifact": "mavis_p1_state_bank",
            "state_bank_state_sha256": state_sha256,
            "config_sha256": config.config_sha256,
            "authority_state_sha256": authority.state_sha256,
            "runtime_code_state_sha256": code_state_sha256,
            "files": {
                name: {
                    "bytes": (temporary / name).stat().st_size,
                    "sha256": _sha256(temporary / name),
                }
                for name in scientific_files
            },
        }
        _json(temporary / "artifact_manifest.json", manifest)
        _write_checksums(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    verify_state_bank_package(output)
    return output


def verify_state_bank_package(path: str | Path) -> dict[str, object]:
    root = Path(path)
    try:
        manifest = json.loads(
            (root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        lines = (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISStateBankPackageError("P1 package metadata is invalid") from error
    expected_files = set(manifest.get("files", {})) | {"artifact_manifest.json"}
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if actual_files != expected_files or len(lines) != len(expected_files):
        raise MAVISStateBankPackageError("P1 package file roster changed")
    ledger: dict[str, str] = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise MAVISStateBankPackageError("P1 checksum ledger is invalid") from error
        if name in ledger or len(digest) != 64:
            raise MAVISStateBankPackageError("P1 checksum ledger is invalid")
        ledger[name] = digest
    if set(ledger) != expected_files:
        raise MAVISStateBankPackageError("P1 checksum coverage changed")
    for name, digest in ledger.items():
        if _sha256(root / name) != digest:
            raise MAVISStateBankPackageError(f"P1 checksum mismatch: {name}")
    for name, metadata in manifest["files"].items():
        file_path = root / name
        if (
            metadata.get("sha256") != _sha256(file_path)
            or metadata.get("bytes") != file_path.stat().st_size
        ):
            raise MAVISStateBankPackageError(f"P1 manifest mismatch: {name}")
    if any((root / name).stat().st_size >= _GIT_BLOB_LIMIT for name in expected_files):
        raise MAVISStateBankPackageError("P1 package contains an oversized Git blob")
    return manifest


__all__ = [
    "MAVISStateBankPackageError",
    "finalize_state_bank_package",
    "load_state_action_pairs_package",
    "load_state_manifest_package",
    "verify_state_bank_package",
    "write_compact_state_manifest",
    "write_partitioned_action_pairs",
]
