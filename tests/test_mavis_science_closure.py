from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest
import yaml


def test_science_closure_analysis_module_exists() -> None:
    assert importlib.util.find_spec("cmc_bbdm.mavis.science_closure") is not None


def test_science_closure_execution_module_exists() -> None:
    assert (
        importlib.util.find_spec("cmc_bbdm.mavis.science_closure_execution")
        is not None
    )


_DOMAIN_ORDER = ("d0", "d1", "d2")
_MODES = ("real", "positions_only", "shuffled", "static")


def _states() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "domain_id": ["d0", "d0"],
            "specimen_id": ["specimen-1", "specimen-1"],
            "trajectory_id": ["trajectory-1", "trajectory-1"],
            "method": ["uniform", "uniform"],
            "state_id": ["state-0", "state-1"],
            "inspection_state_sha256": ["0" * 64, "1" * 64],
            "step": [0, 1],
            "nominal_checkpoint": [0.05, 0.10],
            "exact_acquired_cost": [10, 20],
            "native_count": [200, 200],
            "effective_budget": [0.05, 0.10],
            "teacher_outer_domains": [["d1", "d2"], ["d1", "d2"]],
        }
    )


def _action_scores() -> pl.DataFrame:
    teacher = {
        "state-0": {0: 1.0, 1: 0.0, 2: 3.0},
        "state-1": {0: 0.0, 1: 2.0},
    }
    predicted = {
        "real": {
            "state-0": {0: 0.8, 1: 0.2, 2: 2.0},
            "state-1": {0: 0.1, 1: 1.2},
        },
        "positions_only": {
            "state-0": {0: 0.7, 1: 0.6, 2: 1.0},
            "state-1": {0: 0.8, 1: 0.7},
        },
        "shuffled": {
            "state-0": {0: 0.1, 1: 0.9, 2: 0.5},
            "state-1": {0: 0.2, 1: 0.9},
        },
        "static": {
            "state-0": {0: 1.0, 1: 0.0, 2: 2.0},
            "state-1": {0: 0.9, 1: 0.1},
        },
    }
    rows: list[dict[str, object]] = []
    for mode in _MODES:
        for state_id, candidates in predicted[mode].items():
            for candidate_index, (cell_index, score) in enumerate(candidates.items()):
                rows.append(
                    {
                        "outer_domain": "d0",
                        "domain_id": "d0",
                        "specimen_id": "specimen-1",
                        "state_id": state_id,
                        "mode": mode,
                        "candidate_index": candidate_index,
                        "cell_index": cell_index,
                        "from_level": 0,
                        "to_level": 1,
                        "exact_added_cost": 10,
                        "predicted_score": score,
                        "teacher_value": teacher[state_id][cell_index],
                        "teacher_fold_count": 2,
                        "dynamic_model_state_sha256": mode[0] * 64,
                    }
                )
    return pl.DataFrame(rows)


def _analysis_module():
    from cmc_bbdm.mavis import science_closure

    return science_closure


def _execution_module():
    return importlib.import_module("cmc_bbdm.mavis.science_closure_execution")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_state(path: Path) -> str:
    rows = [
        (item.relative_to(path).as_posix(), item.stat().st_size, _sha256(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _package_inputs(root: Path) -> Path:
    input_root = root / "inputs"
    input_root.mkdir()
    state_parts = []
    action_parts = []
    for domain_index, domain in enumerate(_DOMAIN_ORDER):
        suffix = str(domain_index)
        state_parts.append(
            _states().with_columns(
                pl.lit(domain).alias("domain_id"),
                (pl.col("specimen_id") + f"-{suffix}").alias("specimen_id"),
                (pl.col("trajectory_id") + f"-{suffix}").alias("trajectory_id"),
                (pl.col("state_id") + f"-{suffix}").alias("state_id"),
                pl.lit([item for item in _DOMAIN_ORDER if item != domain]).alias(
                    "teacher_outer_domains"
                ),
            )
        )
        action_parts.append(
            _action_scores().with_columns(
                pl.lit(domain).alias("outer_domain"),
                pl.lit(domain).alias("domain_id"),
                (pl.col("specimen_id") + f"-{suffix}").alias("specimen_id"),
                (pl.col("state_id") + f"-{suffix}").alias("state_id"),
            )
        )
    state_path = input_root / "states.parquet"
    action_path = input_root / "actions.parquet"
    pl.concat(state_parts).write_parquet(state_path)
    pl.concat(action_parts).write_parquet(action_path)
    p7 = input_root / "p7"
    p7.mkdir()
    (p7 / "frozen.txt").write_text("frozen\n", encoding="ascii")
    config = {
        "schema_version": 1,
        "audit_base_git_sha": "7" * 40,
        "domain_order": list(_DOMAIN_ORDER),
        "p1_state_manifest": "inputs/states.parquet",
        "p1_state_manifest_sha256": _sha256(state_path),
        "p3_action_scores": "inputs/actions.parquet",
        "p3_action_scores_sha256": _sha256(action_path),
        "p7_package": "inputs/p7",
        "p7_tree_state_sha256": _tree_state(p7),
        "modes": list(_MODES),
        "top_k": 1,
        "bootstrap_replicates": 20,
        "seed": 17,
    }
    config_path = root / "science_closure.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="ascii")
    return config_path


def test_value_evolution_uses_causal_state_only() -> None:
    module = _analysis_module()
    assert hasattr(module, "build_value_evolution")

    rows = module.build_value_evolution(
        _states(),
        _action_scores(),
        domain_order=_DOMAIN_ORDER,
        modes=_MODES,
    )

    assert rows.get_column("initial_state_id").unique().to_list() == ["state-0"]
    assert rows.get_column("current_state_id").unique().to_list() == ["state-1"]
    assert rows.get_column("acquired_cost_delta").unique().to_list() == [10]
    assert set(rows.get_column("cell_index")) == {0, 1}
    assert "future_unacquired_content" not in rows.columns


def test_value_evolution_never_reads_future_unacquired_content() -> None:
    module = _analysis_module()
    assert hasattr(module, "build_value_evolution")
    states = _states()
    actions = _action_scores()
    first = module.build_value_evolution(
        states.with_columns(pl.lit("first").alias("future_unacquired_content")),
        actions.with_columns(pl.lit(1.0).alias("future_unacquired_content")),
        domain_order=_DOMAIN_ORDER,
        modes=_MODES,
    )
    second = module.build_value_evolution(
        states.with_columns(pl.lit("changed").alias("future_unacquired_content")),
        actions.with_columns(pl.lit(-999.0).alias("future_unacquired_content")),
        domain_order=_DOMAIN_ORDER,
        modes=_MODES,
    )

    assert first.equals(second)


def test_value_evolution_teacher_is_strict_oof() -> None:
    module = _analysis_module()
    assert hasattr(module, "ValueEvolutionError")
    assert hasattr(module, "build_value_evolution")
    states = _states().with_columns(
        pl.when(pl.col("state_id") == "state-1")
        .then(pl.lit(["d0", "d2"]))
        .otherwise(pl.col("teacher_outer_domains"))
        .alias("teacher_outer_domains")
    )

    with pytest.raises(module.ValueEvolutionError, match="strict-OOF"):
        module.build_value_evolution(
            states,
            _action_scores(),
            domain_order=_DOMAIN_ORDER,
            modes=_MODES,
        )


def test_positions_and_shuffled_controls_share_identical_actions_costs() -> None:
    module = _analysis_module()
    assert hasattr(module, "build_value_evolution")
    rows = module.build_value_evolution(
        _states(),
        _action_scores(),
        domain_order=_DOMAIN_ORDER,
        modes=_MODES,
    )
    controls = rows.filter(
        pl.col("value_source").is_in(["real", "positions_only", "shuffled"])
    )
    rosters = controls.group_by("value_source").agg(
        pl.struct("cell_index", "from_level", "to_level", "exact_added_cost")
        .sort()
        .alias("roster")
    )
    assert rosters.get_column("roster").n_unique() == 1

    changed = _action_scores().with_columns(
        pl.when(
            (pl.col("mode") == "positions_only")
            & (pl.col("state_id") == "state-1")
            & (pl.col("cell_index") == 0)
        )
        .then(11)
        .otherwise(pl.col("exact_added_cost"))
        .alias("exact_added_cost")
    )
    with pytest.raises(module.ValueEvolutionError, match="action roster"):
        module.build_value_evolution(
            _states(),
            changed,
            domain_order=_DOMAIN_ORDER,
            modes=_MODES,
        )


def test_value_evolution_preserves_state_dependent_exact_action_cost() -> None:
    module = _analysis_module()
    changed = _action_scores().with_columns(
        pl.when(pl.col("state_id") == "state-1")
        .then(8)
        .otherwise(pl.col("exact_added_cost"))
        .alias("exact_added_cost")
    )

    rows = module.build_value_evolution(
        _states(),
        changed,
        domain_order=_DOMAIN_ORDER,
        modes=_MODES,
    )

    assert rows.get_column("initial_exact_added_cost").unique().to_list() == [10]
    assert rows.get_column("current_exact_added_cost").unique().to_list() == [8]


def test_value_evolution_metrics_match_known_rank_and_utility_case() -> None:
    module = _analysis_module()
    assert hasattr(module, "evaluate_value_evolution")
    rows = module.build_value_evolution(
        _states(),
        _action_scores(),
        domain_order=_DOMAIN_ORDER,
        modes=_MODES,
    )

    metrics = module.evaluate_value_evolution(rows, top_k=1).filter(
        pl.col("value_source") == "teacher"
    )

    assert metrics.height == 1
    row = metrics.row(0, named=True)
    assert row["common_candidate_count"] == 2
    assert row["rank_spearman"] == pytest.approx(-1.0)
    assert row["top_k_jaccard"] == pytest.approx(0.0)
    assert row["best_action_changed"] is True
    assert row["mean_absolute_value_shift"] == pytest.approx(1.5)
    assert row["dynamic_vs_initial_opportunity"] == pytest.approx(2.0)


def _pair_metrics() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        ("d0", "s0"): 1.0,
        ("d0", "s1"): 1.0,
        ("d1", "s2"): 5.0,
    }
    for (domain_id, specimen_id), opportunity in values.items():
        for method_index, method in enumerate(("uniform", "random")):
            for source_index, value_source in enumerate(
                ("teacher", "real", "positions_only", "shuffled", "static")
            ):
                rows.append(
                    {
                        "outer_domain": domain_id,
                        "domain_id": domain_id,
                        "specimen_id": specimen_id,
                        "trajectory_id": f"{specimen_id}-{method}",
                        "method": method,
                        "initial_state_id": f"{specimen_id}-{method}-0",
                        "current_state_id": f"{specimen_id}-{method}-1",
                        "initial_step": 0,
                        "current_step": 1 + method_index,
                        "initial_checkpoint": 0.03125,
                        "current_checkpoint": 0.0625,
                        "initial_acquired_cost": 10,
                        "current_acquired_cost": 20,
                        "acquired_cost_delta": 10,
                        "value_source": value_source,
                        "common_candidate_count": 2,
                        "initial_best_action": "0:0:1",
                        "current_best_action": "1:0:1",
                        "mean_absolute_value_shift": 0.5 + source_index,
                        "rank_spearman": 0.8 - source_index * 0.1,
                        "top_k_jaccard": 0.5,
                        "best_action_changed": True,
                        "dynamic_vs_initial_opportunity": opportunity
                        + source_index * 0.1,
                        "top_k": 1,
                    }
                )
    return pl.DataFrame(rows)


def test_value_evolution_aggregation_is_specimen_first_and_equal_domain() -> None:
    module = _analysis_module()
    assert hasattr(module, "aggregate_value_evolution")

    tables = module.aggregate_value_evolution(
        _pair_metrics(),
        domain_order=("d0", "d1"),
    )

    assert tables.per_specimen.height == 15
    teacher = tables.aggregate.filter(pl.col("value_source") == "teacher")
    assert teacher.height == 1
    assert teacher.item(0, "dynamic_vs_initial_opportunity") == pytest.approx(3.0)
    assert teacher.item(0, "domain_count") == 2
    assert teacher.item(0, "specimen_count") == 3


def test_value_evolution_bootstrap_is_paired_and_deterministic() -> None:
    module = _analysis_module()
    assert hasattr(module, "bootstrap_value_evolution")
    tables = module.aggregate_value_evolution(
        _pair_metrics(),
        domain_order=("d0", "d1"),
    )

    first = module.bootstrap_value_evolution(
        tables.per_specimen,
        domain_order=("d0", "d1"),
        replicates=50,
        seed=17,
    )
    second = module.bootstrap_value_evolution(
        tables.per_specimen,
        domain_order=("d0", "d1"),
        replicates=50,
        seed=17,
    )

    assert first.equals(second)
    contrast = first.filter(
        (pl.col("metric") == "dynamic_vs_initial_opportunity")
        & (pl.col("contrast") == "real_minus_positions_only")
    )
    assert contrast.height == 1
    assert contrast.item(0, "estimate") == pytest.approx(-0.1)
    assert contrast.item(0, "bootstrap_replicates") == 50


def test_p9_value_evolution_package_writes_required_outputs(tmp_path: Path) -> None:
    module = _execution_module()
    assert hasattr(module, "run_p9_value_evolution")
    assert hasattr(module, "verify_p9_value_evolution_package")
    config = _package_inputs(tmp_path)

    output = module.run_p9_value_evolution(
        config,
        project_root=tmp_path,
        output_root="results/p9_value_evolution",
    )
    manifest = module.verify_p9_value_evolution_package(output)

    assert {
        "value_evolution.parquet",
        "aggregate_metrics.csv",
        "domain_metrics.csv",
        "bootstrap.csv",
        "REPORT.md",
        "summary.json",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    } <= {item.name for item in output.iterdir()}
    assert manifest["stage"] == "P9_CONDITIONAL_VALUE_EVOLUTION"
    assert manifest["status"] == "COMPLETE"
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "Teacher Top-1 Jaccard" in report
    assert all(line == line.rstrip() for line in report.splitlines())


def test_science_closure_does_not_modify_p7_artifacts(tmp_path: Path) -> None:
    module = _execution_module()
    config = _package_inputs(tmp_path)
    p7 = tmp_path / "inputs/p7"
    before = _tree_state(p7)

    module.run_p9_value_evolution(
        config,
        project_root=tmp_path,
        output_root="results/p9_value_evolution",
    )

    assert _tree_state(p7) == before


def test_science_closure_manifest_hashes_match(tmp_path: Path) -> None:
    module = _execution_module()
    output = module.run_p9_value_evolution(
        _package_inputs(tmp_path),
        project_root=tmp_path,
        output_root="results/p9_value_evolution",
    )
    module.verify_p9_value_evolution_package(output)
    (output / "summary.json").write_text("{}\n", encoding="ascii")

    with pytest.raises(module.ScienceClosureExecutionError, match="checksum"):
        module.verify_p9_value_evolution_package(output)
