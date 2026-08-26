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


def _artifacts_module():
    return importlib.import_module("cmc_bbdm.mavis.science_closure_artifacts")


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


_MRIS_MODES = (
    "real",
    "positions_only",
    "shuffled",
    "static",
    "reconstruction",
)


def _mris_predictions() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    errors = {
        "d0": {
            "real": (4.0, 2.0),
            "positions_only": (3.0, 3.0),
            "shuffled": (4.5, 4.5),
            "static": (5.0, 5.0),
            "reconstruction": (1.5, 1.5),
        },
        "d1": {
            "real": (8.0, 6.0),
            "positions_only": (5.0, 5.0),
            "shuffled": (7.0, 7.0),
            "static": (10.0, 10.0),
            "reconstruction": (4.0, 4.0),
        },
    }
    for domain_index, domain in enumerate(("d0", "d1")):
        for specimen_index in range(2):
            specimen_id = f"{domain}-s{specimen_index}"
            for trajectory_index in range(2):
                for checkpoint_index, checkpoint in enumerate((0.1, 0.2)):
                    state_id = (
                        f"{specimen_id}-t{trajectory_index}-c{checkpoint_index}"
                    )
                    for mode in _MRIS_MODES:
                        prediction = (
                            errors[domain][mode][checkpoint_index]
                            + specimen_index * 0.2
                            + trajectory_index * 0.02
                        )
                        rows.append(
                            {
                                "outer_domain": domain,
                                "state_id": state_id,
                                "specimen_id": specimen_id,
                                "trajectory_id": f"{specimen_id}-t{trajectory_index}",
                                "method": "uniform",
                                "seed": trajectory_index,
                                "nominal_checkpoint": checkpoint,
                                "exact_acquired_cost": 10 * (checkpoint_index + 1),
                                "native_count": 100,
                                "effective_budget": checkpoint,
                                "mode": mode,
                                "target": 0.0,
                                "prediction": prediction,
                                "absolute_error": prediction,
                                "model_state_sha256": mode[0] * 64,
                            }
                        )
    return pl.DataFrame(rows)


def _full_field_predictions() -> pl.DataFrame:
    rows = []
    for domain, base in (("d0", 1.0), ("d1", 3.0)):
        for specimen_index in range(2):
            prediction = base + specimen_index * 0.2
            rows.append(
                {
                    "method": "I_field_selected",
                    "specimen_id": f"{domain}-s{specimen_index}",
                    "dataset_id": domain,
                    "target": 0.0,
                    "prediction": prediction,
                    "seed": 0,
                }
            )
    return pl.DataFrame(rows)


def test_mris_closure_reuses_frozen_state_predictions_when_available() -> None:
    module = _analysis_module()
    assert hasattr(module, "evaluate_mris_causal_closure")
    frozen = _mris_predictions()

    tables = module.evaluate_mris_causal_closure(
        frozen,
        _full_field_predictions(),
        domain_order=("d0", "d1"),
        full_field_method="I_field_selected",
        bootstrap_replicates=50,
        seed=17,
    )

    observed = tables.per_specimen_predictions.filter(
        (pl.col("outer_domain") == "d0")
        & (pl.col("specimen_id") == "d0-s0")
        & (pl.col("mode") == "real")
        & (pl.col("nominal_checkpoint") == 0.2)
    ).row(0, named=True)
    expected = frozen.filter(
        (pl.col("outer_domain") == "d0")
        & (pl.col("specimen_id") == "d0-s0")
        & (pl.col("mode") == "real")
        & (pl.col("nominal_checkpoint") == 0.2)
    )
    assert observed["mean_prediction"] == pytest.approx(
        expected.get_column("prediction").mean()
    )
    assert observed["mae"] == pytest.approx(
        expected.get_column("absolute_error").mean()
    )
    assert observed["source"] == "frozen_p2_state_predictions"
    assert tables.source_prediction_row_count == frozen.height


def test_mris_closure_requires_identical_state_cost_rosters() -> None:
    module = _analysis_module()
    changed = _mris_predictions().with_columns(
        pl.when(
            (pl.col("mode") == "positions_only")
            & (pl.col("state_id") == "d0-s0-t0-c0")
        )
        .then(11)
        .otherwise(pl.col("exact_acquired_cost"))
        .alias("exact_acquired_cost")
    )

    with pytest.raises(module.MRISCausalClosureError, match="same state/cost"):
        module.evaluate_mris_causal_closure(
            changed,
            _full_field_predictions(),
            domain_order=("d0", "d1"),
            full_field_method="I_field_selected",
            bootstrap_replicates=20,
            seed=17,
        )


def test_mris_closure_uses_lower_is_better_contrasts_and_equal_domains() -> None:
    module = _analysis_module()
    tables = module.evaluate_mris_causal_closure(
        _mris_predictions(),
        _full_field_predictions(),
        domain_order=("d0", "d1"),
        full_field_method="I_field_selected",
        bootstrap_replicates=50,
        seed=17,
    )

    contrast = tables.contrasts.filter(
        (pl.col("nominal_checkpoint") == 0.2)
        & (pl.col("control_mode") == "positions_only")
    ).row(0, named=True)
    assert contrast["equal_domain_real_minus_control_mae"] == pytest.approx(0.0)
    assert contrast["improved_domain_count"] == 1
    assert contrast["worst_domain_effect"] == pytest.approx(1.0)
    d0 = tables.domain_metrics.filter(
        (pl.col("outer_domain") == "d0")
        & (pl.col("nominal_checkpoint") == 0.2)
    ).row(0, named=True)
    assert d0["real_minus_positions_only_mae"] == pytest.approx(-1.0)
    assert d0["full_field_utility_recovery_fraction"] == pytest.approx(
        (5.11 - 2.11) / (5.11 - 1.1)
    )


def test_mris_closure_bootstrap_is_paired_and_deterministic() -> None:
    module = _analysis_module()
    first = module.evaluate_mris_causal_closure(
        _mris_predictions(),
        _full_field_predictions(),
        domain_order=("d0", "d1"),
        full_field_method="I_field_selected",
        bootstrap_replicates=50,
        seed=19,
    ).bootstrap
    second = module.evaluate_mris_causal_closure(
        _mris_predictions(),
        _full_field_predictions(),
        domain_order=("d0", "d1"),
        full_field_method="I_field_selected",
        bootstrap_replicates=50,
        seed=19,
    ).bootstrap

    assert first.equals(second)
    row = first.filter(
        (pl.col("scope") == "equal_domain")
        & (pl.col("metric") == "real_minus_control_mae")
        & (pl.col("control_mode") == "positions_only")
        & (pl.col("nominal_checkpoint") == 0.2)
    ).row(0, named=True)
    assert row["estimate"] == pytest.approx(0.0)
    assert row["bootstrap_replicates"] == 50


def _p10_package_inputs(root: Path) -> Path:
    inputs = root / "inputs"
    inputs.mkdir()
    p2_predictions = inputs / "p2_state_predictions.parquet"
    full_field_predictions = inputs / "full_field_predictions.csv"
    _mris_predictions().write_parquet(p2_predictions)
    _full_field_predictions().write_csv(full_field_predictions)
    p2_state = "2" * 64
    p2_manifest = inputs / "p2_artifact_manifest.json"
    p2_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "mavis_p2_mris",
                "p2_state_sha256": p2_state,
                "files": {
                    "state_predictions.parquet": {
                        "sha256": _sha256(p2_predictions)
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    full_field_manifest = inputs / "full_field_artifact_manifest.json"
    full_field_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "cpb_v3_p1_full_field_oracle",
                "files": {
                    "predictions.csv": {"sha256": _sha256(full_field_predictions)}
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    p7 = inputs / "p7"
    p7.mkdir()
    (p7 / "frozen.txt").write_text("frozen\n", encoding="ascii")
    config = {
        "schema_version": 1,
        "stage": "P10_MRIS_CAUSAL",
        "audit_base_git_sha": "7" * 40,
        "domain_order": ["d0", "d1"],
        "p2_state_predictions": "inputs/p2_state_predictions.parquet",
        "p2_state_predictions_sha256": _sha256(p2_predictions),
        "p2_artifact_manifest": "inputs/p2_artifact_manifest.json",
        "p2_artifact_manifest_sha256": _sha256(p2_manifest),
        "p2_state_sha256": p2_state,
        "full_field_predictions": "inputs/full_field_predictions.csv",
        "full_field_predictions_sha256": _sha256(full_field_predictions),
        "full_field_artifact_manifest": "inputs/full_field_artifact_manifest.json",
        "full_field_artifact_manifest_sha256": _sha256(full_field_manifest),
        "full_field_method": "I_field_selected",
        "p7_package": "inputs/p7",
        "p7_tree_state_sha256": _tree_state(p7),
        "bootstrap_replicates": 50,
        "seed": 19,
    }
    config_path = root / "p10_mris_causal.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="ascii")
    return config_path


def test_p10_mris_causal_package_is_hash_bound_and_deterministic(
    tmp_path: Path,
) -> None:
    module = _artifacts_module()
    assert hasattr(module, "run_p10_mris_causal")
    assert hasattr(module, "verify_p10_mris_causal_package")
    config = _p10_package_inputs(tmp_path)
    p7 = tmp_path / "inputs/p7"
    p7_before = _tree_state(p7)

    first = module.run_p10_mris_causal(
        config,
        project_root=tmp_path,
        output_root="results/p10_first",
    )
    second = module.run_p10_mris_causal(
        config,
        project_root=tmp_path,
        output_root="results/p10_second",
    )
    manifest = module.verify_p10_mris_causal_package(first)

    expected = {
        "state_cost_curve.csv",
        "per_specimen_predictions.parquet",
        "domain_metrics.csv",
        "contrasts.csv",
        "bootstrap.csv",
        "REPORT.md",
        "summary.json",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
    assert {item.name for item in first.iterdir()} == expected
    assert manifest["stage"] == "P10_MRIS_CAUSAL"
    assert manifest["status"] == "COMPLETE"
    assert _tree_state(p7) == p7_before
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_prediction_row_count"] == _mris_predictions().height
    assert summary["p7_modified"] is False
    assert summary["actual_content_beyond_geometry_supported"] is False


def test_p10_mris_causal_package_rejects_checksum_changes(tmp_path: Path) -> None:
    module = _artifacts_module()
    output = module.run_p10_mris_causal(
        _p10_package_inputs(tmp_path),
        project_root=tmp_path,
        output_root="results/p10",
    )
    (output / "contrasts.csv").write_text("changed\n", encoding="ascii")

    with pytest.raises(module.ScienceClosureArtifactError, match="checksum"):
        module.verify_p10_mris_causal_package(output)


_DYNAMIC_MODES = ("real", "positions_only", "shuffled")


def _dynamic_states() -> pl.DataFrame:
    rows = []
    for domain in ("d0", "d1"):
        for specimen_index in range(2):
            specimen_id = f"{domain}-s{specimen_index}"
            for checkpoint_index, checkpoint in enumerate((0.1, 0.2)):
                rows.append(
                    {
                        "domain_id": domain,
                        "specimen_id": specimen_id,
                        "trajectory_id": f"{specimen_id}-uniform",
                        "method": "uniform",
                        "state_id": f"{specimen_id}-c{checkpoint_index}",
                        "nominal_checkpoint": checkpoint,
                        "exact_acquired_cost": 10 * (checkpoint_index + 1),
                        "native_count": 100,
                        "effective_budget": checkpoint,
                    }
                )
    return pl.DataFrame(rows)


def _dynamic_action_scores() -> pl.DataFrame:
    score_map = {
        "real": ((3.0, 2.0, 1.0), (1.0, 3.0, 2.0)),
        "positions_only": ((2.0, 3.0, 1.0), (2.0, 3.0, 1.0)),
        "shuffled": ((1.0, 2.0, 3.0), (3.0, 1.0, 2.0)),
    }
    rows = []
    for state in _dynamic_states().iter_rows(named=True):
        checkpoint_index = 0 if state["nominal_checkpoint"] == 0.1 else 1
        for mode in _DYNAMIC_MODES:
            for candidate_index in range(3):
                teacher = 3.0 - candidate_index
                rows.append(
                    {
                        "outer_domain": state["domain_id"],
                        "domain_id": state["domain_id"],
                        "specimen_id": state["specimen_id"],
                        "state_id": state["state_id"],
                        "mode": mode,
                        "candidate_index": candidate_index,
                        "cell_index": candidate_index,
                        "from_level": checkpoint_index,
                        "to_level": checkpoint_index + 1,
                        "exact_added_cost": 5 + candidate_index,
                        "predicted_score": score_map[mode][checkpoint_index][
                            candidate_index
                        ],
                        "teacher_value": teacher,
                        "current_prediction": 0.5,
                        "candidate_prediction": 0.5 + teacher * 0.01,
                        "evaluation_true_cai": 0.7,
                        "teacher_fold_count": 1,
                        "dynamic_model_state_sha256": mode[0] * 64,
                    }
                )
    return pl.DataFrame(rows)


def _mvd_action_scores() -> pl.DataFrame:
    rows = []
    for domain in ("d0", "d1"):
        for specimen_index in range(2):
            specimen_id = f"{domain}-s{specimen_index}"
            for method, scores in (
                ("o2_global_candidate", (3.0, 2.0, 1.0)),
                ("o1_candidate_mlp_huber", (1.0, 3.0, 2.0)),
            ):
                for cell_index, score in enumerate(scores):
                    rows.append(
                        {
                            "outer_domain": domain,
                            "specimen_id": specimen_id,
                            "dataset_id": domain,
                            "method": method,
                            "cell_index": cell_index,
                            "predicted_value": score + specimen_index * 0.01,
                            "teacher_value": 0.0,
                            "candidate_cost": 99 + cell_index,
                        }
                    )
    return pl.DataFrame(rows)


def test_dynamic_valuation_aligns_all_scorers_on_same_legal_actions() -> None:
    module = _analysis_module()
    assert hasattr(module, "build_dynamic_valuation_alignment")

    aligned = module.build_dynamic_valuation_alignment(
        _dynamic_states(),
        _dynamic_action_scores(),
        _mvd_action_scores(),
        domain_order=("d0", "d1"),
        dynamic_modes=_DYNAMIC_MODES,
        mvd_o2_method="o2_global_candidate",
        candidate_only_method="o1_candidate_mlp_huber",
    )

    assert set(aligned.get_column("scorer").unique()) == {
        "dynamic_real",
        "dynamic_positions_only",
        "dynamic_shuffled",
        "static_m1_o2",
        "candidate_only_static",
    }
    rosters = aligned.group_by("scorer").agg(
        pl.struct(
            "state_id",
            "candidate_index",
            "cell_index",
            "from_level",
            "to_level",
            "exact_added_cost",
        )
        .sort()
        .alias("roster")
    )
    assert rosters.get_column("roster").n_unique() == 1
    static = aligned.filter(pl.col("scorer") == "static_m1_o2")
    assert (
        static.group_by("specimen_id", "cell_index")
        .agg(pl.col("predicted_score").n_unique().alias("n"))
        .get_column("n")
        .unique()
        .to_list()
        == [1]
    )
    assert static.get_column("exact_added_cost").max() == 7
    assert static.get_column("source_candidate_cost").min() == 99
    assert static.get_column("static_extension_to_current_state").all()


def test_dynamic_valuation_rejects_misaligned_dynamic_action_costs() -> None:
    module = _analysis_module()
    changed = _dynamic_action_scores().with_columns(
        pl.when(
            (pl.col("mode") == "positions_only")
            & (pl.col("state_id") == "d0-s0-c0")
            & (pl.col("candidate_index") == 0)
        )
        .then(100)
        .otherwise(pl.col("exact_added_cost"))
        .alias("exact_added_cost")
    )

    with pytest.raises(module.DynamicValuationClosureError, match="action roster"):
        module.build_dynamic_valuation_alignment(
            _dynamic_states(),
            changed,
            _mvd_action_scores(),
            domain_order=("d0", "d1"),
            dynamic_modes=_DYNAMIC_MODES,
            mvd_o2_method="o2_global_candidate",
            candidate_only_method="o1_candidate_mlp_huber",
        )


def test_dynamic_valuation_metrics_are_cost_stratified_and_paired() -> None:
    module = _analysis_module()
    aligned = module.build_dynamic_valuation_alignment(
        _dynamic_states(),
        _dynamic_action_scores(),
        _mvd_action_scores(),
        domain_order=("d0", "d1"),
        dynamic_modes=_DYNAMIC_MODES,
        mvd_o2_method="o2_global_candidate",
        candidate_only_method="o1_candidate_mlp_huber",
    )

    first = module.evaluate_dynamic_valuation_closure(
        aligned,
        domain_order=("d0", "d1"),
        recall_k=2,
        bootstrap_replicates=50,
        seed=23,
    )
    second = module.evaluate_dynamic_valuation_closure(
        aligned,
        domain_order=("d0", "d1"),
        recall_k=2,
        bootstrap_replicates=50,
        seed=23,
    )

    assert first.bootstrap.equals(second.bootstrap)
    real = first.per_state.filter(
        (pl.col("scorer") == "dynamic_real")
        & (pl.col("nominal_checkpoint") == 0.2)
    )
    assert real.get_column("next_action_regret").unique().to_list() == [1.0]
    assert real.get_column("one_step_cai_utility").unique().to_list() == [2.0]
    assert first.regret_by_cost.get_column("nominal_checkpoint").n_unique() == 2
    contrast = first.bootstrap.filter(
        (pl.col("scope") == "equal_domain")
        & (pl.col("nominal_checkpoint") == 0.2)
        & (pl.col("metric") == "real_minus_control_regret")
        & (pl.col("control_scorer") == "static_m1_o2")
    ).row(0, named=True)
    assert contrast["estimate"] == pytest.approx(1.0)
    assert contrast["bootstrap_replicates"] == 50


def _p11_package_inputs(root: Path) -> Path:
    inputs = root / "inputs"
    inputs.mkdir()
    states = inputs / "p1_states.parquet"
    actions = inputs / "p3_actions.parquet"
    mvd = inputs / "mvd_scores.parquet"
    _dynamic_states().write_parquet(states)
    _dynamic_action_scores().write_parquet(actions)
    _mvd_action_scores().write_parquet(mvd)
    p1_state = "1" * 64
    p3_state = "3" * 64
    mvd_authority = "4" * 64
    manifests = {
        "p1_manifest.json": {
            "artifact": "mavis_p1_state_bank",
            "state_bank_state_sha256": p1_state,
            "files": {"state_manifest.parquet": {"sha256": _sha256(states)}},
        },
        "p3_manifest.json": {
            "artifact": "mavis_p3_dynamic_voi",
            "p3_state_sha256": p3_state,
            "files": {"action_scores.parquet": {"sha256": _sha256(actions)}},
        },
        "mvd_manifest.json": {
            "package": "mvd_m1_observability",
            "authority_state_sha256": mvd_authority,
            "files": {
                "observability_predictions.parquet": {"sha256": _sha256(mvd)}
            },
        },
    }
    for name, payload in manifests.items():
        (inputs / name).write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="ascii"
        )
    p7 = inputs / "p7"
    p7.mkdir()
    (p7 / "frozen.txt").write_text("frozen\n", encoding="ascii")
    config = {
        "schema_version": 1,
        "stage": "P11_DYNAMIC_VALUATION",
        "audit_base_git_sha": "7" * 40,
        "domain_order": ["d0", "d1"],
        "p1_state_manifest": "inputs/p1_states.parquet",
        "p1_state_manifest_sha256": _sha256(states),
        "p1_artifact_manifest": "inputs/p1_manifest.json",
        "p1_artifact_manifest_sha256": _sha256(inputs / "p1_manifest.json"),
        "p1_state_sha256": p1_state,
        "p3_action_scores": "inputs/p3_actions.parquet",
        "p3_action_scores_sha256": _sha256(actions),
        "p3_artifact_manifest": "inputs/p3_manifest.json",
        "p3_artifact_manifest_sha256": _sha256(inputs / "p3_manifest.json"),
        "p3_state_sha256": p3_state,
        "mvd_action_scores": "inputs/mvd_scores.parquet",
        "mvd_action_scores_sha256": _sha256(mvd),
        "mvd_artifact_manifest": "inputs/mvd_manifest.json",
        "mvd_artifact_manifest_sha256": _sha256(inputs / "mvd_manifest.json"),
        "mvd_authority_state_sha256": mvd_authority,
        "dynamic_modes": list(_DYNAMIC_MODES),
        "mvd_o2_method": "o2_global_candidate",
        "candidate_only_method": "o1_candidate_mlp_huber",
        "recall_k": 2,
        "p7_package": "inputs/p7",
        "p7_tree_state_sha256": _tree_state(p7),
        "bootstrap_replicates": 50,
        "seed": 23,
    }
    config_path = root / "p11_dynamic_valuation.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="ascii")
    return config_path


def test_p11_dynamic_valuation_package_is_hash_bound_and_deterministic(
    tmp_path: Path,
) -> None:
    module = _artifacts_module()
    assert hasattr(module, "run_p11_dynamic_valuation")
    assert hasattr(module, "verify_p11_dynamic_valuation_package")
    config = _p11_package_inputs(tmp_path)
    p7 = tmp_path / "inputs/p7"
    before = _tree_state(p7)

    first = module.run_p11_dynamic_valuation(
        config,
        project_root=tmp_path,
        output_root="results/p11_first",
    )
    second = module.run_p11_dynamic_valuation(
        config,
        project_root=tmp_path,
        output_root="results/p11_second",
    )
    manifest = module.verify_p11_dynamic_valuation_package(first)

    expected = {
        "action_predictions.parquet",
        "regret_by_cost.csv",
        "one_step_utility.csv",
        "domain_metrics.csv",
        "bootstrap.csv",
        "REPORT.md",
        "summary.json",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
    assert {item.name for item in first.iterdir()} == expected
    assert manifest["stage"] == "P11_DYNAMIC_VALUATION"
    assert manifest["status"] == "COMPLETE"
    assert _tree_state(p7) == before
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["aligned_action_prediction_count"] == 120
    assert summary["p7_modified"] is False
    assert summary["target_data_used_for_selection"] is False


def test_p11_dynamic_valuation_package_rejects_checksum_changes(
    tmp_path: Path,
) -> None:
    module = _artifacts_module()
    output = module.run_p11_dynamic_valuation(
        _p11_package_inputs(tmp_path),
        project_root=tmp_path,
        output_root="results/p11",
    )
    (output / "regret_by_cost.csv").write_text("changed\n", encoding="ascii")

    with pytest.raises(module.ScienceClosureArtifactError, match="checksum"):
        module.verify_p11_dynamic_valuation_package(output)
