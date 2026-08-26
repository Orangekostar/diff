"""Small-budget joint-utility set-planning diagnosis for MAVIS P13."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from itertools import permutations
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from cmc_bbdm.mva.measurement_state import RefinementAction

from .authority import load_mavis_authority
from .config import load_mavis_config
from .dynamic_training import load_fitted_dynamic_checkpoint
from .mris_training import load_fitted_mris_checkpoint
from .policy import DeployedDynamicScorer
from .reveal import MAVISRevealError, reveal_action, reveal_uniform_scout
from .rollout import _candidate_descriptors, _scores
from .science_closure_planning import (
    planning_state_sha256,
    select_learned_lookahead_action,
)


class SetPlanningExecutionError(RuntimeError):
    """Raised when the P13 diagnostic or package violates its contract."""


_METHODS = (
    "current_greedy",
    "beam_width_2",
    "beam_width_4",
    "two_step_lookahead",
    "exact_learned_lookahead_reachable_pool",
    "retrospective_joint_near_oracle_reachable_pool",
)
_FILES = {
    "set_results.parquet",
    "enumerated_sets.parquet",
    "per_domain.csv",
    "bootstrap.csv",
    "REPORT.md",
    "summary.json",
    "artifact_manifest.json",
    "CHECKSUMS.sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_state(path: Path) -> str:
    rows = [
        (item.relative_to(path).as_posix(), item.stat().st_size, _sha256(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    if not rows:
        raise SetPlanningExecutionError("bound artifact tree is empty")
    return planning_state_sha256(rows)


def _bound(root: Path, value: object, *, directory: bool) -> Path:
    if type(value) is not str or not value:
        raise SetPlanningExecutionError("configured path is invalid")
    try:
        path = (root / value).resolve(strict=True)
    except OSError as error:
        raise SetPlanningExecutionError("configured artifact is unavailable") from error
    if root != path and root not in path.parents:
        raise SetPlanningExecutionError("configured artifact escapes project root")
    if path.is_dir() != directory:
        raise SetPlanningExecutionError("configured artifact type changed")
    return path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_checksums(path: Path) -> None:
    files = sorted(item for item in path.iterdir() if item.name != "CHECKSUMS.sha256")
    (path / "CHECKSUMS.sha256").write_text(
        "".join(f"{_sha256(item)}  {item.name}\n" for item in files),
        encoding="ascii",
    )


def _load_config(path: str | Path) -> dict[str, object]:
    try:
        source = Path(path).resolve(strict=True)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SetPlanningExecutionError("P13 config is unavailable") from error
    keys = {
        "schema_version",
        "stage",
        "audit_base_git_sha",
        "domain_order",
        "mavis_config",
        "mavis_config_sha256",
        "p2_checkpoint_root",
        "p2_checkpoint_tree_sha256",
        "p3_checkpoint_root",
        "p3_checkpoint_tree_sha256",
        "p7_package",
        "p7_tree_state_sha256",
        "diagnostic_checkpoint",
        "base_shortlist_width",
        "beam_widths",
        "lookahead_width",
        "set_size",
        "bootstrap_replicates",
        "seed",
    }
    hash_keys = (
        "mavis_config_sha256",
        "p2_checkpoint_tree_sha256",
        "p3_checkpoint_tree_sha256",
        "p7_tree_state_sha256",
    )
    if (
        type(payload) is not dict
        or set(payload) != keys
        or payload["schema_version"] != 1
        or payload["stage"] != "P13_SET_PLANNING"
        or type(payload["domain_order"]) is not list
        or len(payload["domain_order"]) != 6
        or len(set(payload["domain_order"])) != 6
        or any(type(item) is not str or not item for item in payload["domain_order"])
        or any(
            type(payload[key]) is not str
            or len(payload[key]) != 64
            or any(character not in "0123456789abcdef" for character in payload[key])
            for key in hash_keys
        )
        or type(payload["diagnostic_checkpoint"]) is not float
        or not 0.0 < payload["diagnostic_checkpoint"] <= 1.0
        or payload["base_shortlist_width"] != 8
        or payload["beam_widths"] != [2, 4]
        or payload["lookahead_width"] != 8
        or payload["set_size"] != 2
        or type(payload["bootstrap_replicates"]) is not int
        or payload["bootstrap_replicates"] < 2
        or type(payload["seed"]) is not int
        or isinstance(payload["seed"], bool)
    ):
        raise SetPlanningExecutionError("P13 config schema changed")
    payload["config_sha256"] = _sha256(source)
    return payload


def _action_key(action: RefinementAction) -> tuple[int, int, int]:
    return (action.cell_index, action.from_level, action.to_level)


def _greedy_two_action_plan(
    authority: object,
    state: object,
    *,
    endpoint_budget: float,
    scorer: DeployedDynamicScorer,
) -> tuple[tuple[RefinementAction, ...], object, float]:
    current = state
    selected: list[RefinementAction] = []
    point_sum = 0.0
    for _step in range(2):
        actions, candidates = _candidate_descriptors(
            current,
            endpoint_budget=endpoint_budget,
            action_budget=endpoint_budget,
        )
        if not actions:
            break
        scores = _scores(
            scorer.score_actions(current, candidates),
            len(candidates),
            objective="direct_cost_aware",
        )
        index = max(
            range(len(actions)),
            key=lambda item: (
                float(scores[item]),
                -actions[item].cell_index,
                -actions[item].to_level,
            ),
        )
        selected.append(actions[index])
        point_sum += float(scores[index])
        current = reveal_action(authority, current, actions[index])
    if len(selected) != 2:
        raise SetPlanningExecutionError("two-action greedy plan is infeasible")
    return tuple(selected), current, point_sum


def _lookahead_two_action_plan(
    authority: object,
    state: object,
    *,
    endpoint_budget: float,
    scorer: DeployedDynamicScorer,
    width: int,
) -> tuple[tuple[RefinementAction, ...], object, float]:
    first, _immediate, total = select_learned_lookahead_action(
        authority,
        state,
        endpoint_budget=endpoint_budget,
        action_budget=endpoint_budget,
        scorer=scorer,
        beam_width=width,
    )
    branch = reveal_action(authority, state, first)
    actions, candidates = _candidate_descriptors(
        branch,
        endpoint_budget=endpoint_budget,
        action_budget=endpoint_budget,
    )
    if not actions:
        raise SetPlanningExecutionError("lookahead second action is infeasible")
    scores = _scores(
        scorer.score_actions(branch, candidates),
        len(candidates),
        objective="direct_cost_aware",
    )
    index = max(
        range(len(actions)),
        key=lambda item: (
            float(scores[item]),
            -actions[item].cell_index,
            -actions[item].to_level,
        ),
    )
    second = actions[index]
    return (first, second), reveal_action(authority, branch, second), total


def _specimen_diagnostic(
    authority: object,
    *,
    specimen_id: str,
    domain: str,
    initial_budget: float,
    checkpoint: float,
    scorer: DeployedDynamicScorer,
    evaluator: object,
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    initial = reveal_uniform_scout(
        authority,
        authority.policy_context(specimen_id),
        initial_budget=initial_budget,
        checkpoint=checkpoint,
    )
    actions, candidates = _candidate_descriptors(
        initial,
        endpoint_budget=checkpoint,
        action_budget=checkpoint,
    )
    initial_scores = _scores(
        scorer.score_actions(initial, candidates),
        len(candidates),
        objective="direct_cost_aware",
    )
    ranking = sorted(
        range(len(actions)),
        key=lambda index: (
            -float(initial_scores[index]),
            actions[index].cell_index,
            actions[index].to_level,
        ),
    )[:8]
    plans: dict[str, tuple[tuple[RefinementAction, ...], object, float]] = {
        "current_greedy": _greedy_two_action_plan(
            authority,
            initial,
            endpoint_budget=checkpoint,
            scorer=scorer,
        ),
        "beam_width_2": _lookahead_two_action_plan(
            authority,
            initial,
            endpoint_budget=checkpoint,
            scorer=scorer,
            width=2,
        ),
        "beam_width_4": _lookahead_two_action_plan(
            authority,
            initial,
            endpoint_budget=checkpoint,
            scorer=scorer,
            width=4,
        ),
        "two_step_lookahead": _lookahead_two_action_plan(
            authority,
            initial,
            endpoint_budget=checkpoint,
            scorer=scorer,
            width=8,
        ),
    }
    reachable: dict[tuple[int, int, int], RefinementAction] = {
        _action_key(actions[index]): actions[index] for index in ranking
    }
    for plan, _state, _score in plans.values():
        for action in plan:
            reachable[_action_key(action)] = action
    reachable_pool = tuple(reachable[key] for key in sorted(reachable))
    initial_score_by_action = {
        _action_key(action): float(score)
        for action, score in zip(actions, initial_scores, strict=True)
    }
    pair_states: list[object] = []
    pair_meta: list[tuple[RefinementAction, RefinementAction, float]] = []
    for first in reachable_pool:
        try:
            branch = reveal_action(authority, initial, first)
        except MAVISRevealError:
            continue
        next_actions, next_candidates = _candidate_descriptors(
            branch,
            endpoint_budget=checkpoint,
            action_budget=checkpoint,
        )
        next_scores = _scores(
            scorer.score_actions(branch, next_candidates),
            len(next_candidates),
            objective="direct_cost_aware",
        )
        next_by_action = {
            _action_key(action): (action, float(score))
            for action, score in zip(next_actions, next_scores, strict=True)
        }
        for _first, second in permutations(reachable_pool, 2):
            if _first != first or _action_key(second) not in next_by_action:
                continue
            issued_second, second_score = next_by_action[_action_key(second)]
            try:
                state = reveal_action(authority, branch, issued_second)
            except MAVISRevealError:
                continue
            if state.exact_acquired_count > math.floor(checkpoint * state.native_count):
                continue
            pair_states.append(state)
            pair_meta.append(
                (
                    first,
                    issued_second,
                    initial_score_by_action[_action_key(first)] + second_score,
                )
            )
    if not pair_states:
        raise SetPlanningExecutionError("P13 exact reachable set is empty")
    target = float(authority.evaluation_view(specimen_id).true_cai)
    initial_prediction = float(
        evaluator.predict_inspection_state(initial, device=device)
    )
    initial_error = abs(target - initial_prediction)
    pair_predictions = evaluator.predict_inspection_states(
        tuple(pair_states), batch_size=len(pair_states), device=device
    )
    enumeration: list[dict[str, object]] = []
    for index, (state, prediction_raw, meta) in enumerate(
        zip(pair_states, pair_predictions, pair_meta, strict=True)
    ):
        first, second, point_sum = meta
        prediction = float(prediction_raw)
        final_error = abs(target - prediction)
        enumeration.append(
            {
                "outer_domain": domain,
                "specimen_id": specimen_id,
                "diagnostic_checkpoint": checkpoint,
                "set_index": index,
                "base_shortlist_width": 8,
                "reachable_pool_width": len(reachable_pool),
                "action_cell_indices": [first.cell_index, second.cell_index],
                "action_from_levels": [first.from_level, second.from_level],
                "action_to_levels": [first.to_level, second.to_level],
                "exact_added_cost": state.exact_acquired_count
                - initial.exact_acquired_count,
                "point_value_sum": point_sum,
                "initial_error": initial_error,
                "final_prediction": prediction,
                "final_error": final_error,
                "true_joint_utility": initial_error - final_error,
                "terminal_state_sha256": state.state_sha256,
            }
        )
    exact_index = max(
        range(len(enumeration)),
        key=lambda index: (
            float(enumeration[index]["point_value_sum"]),
            -int(enumeration[index]["exact_added_cost"]),
            tuple(-value for value in enumeration[index]["action_cell_indices"]),
        ),
    )
    oracle_index = max(
        range(len(enumeration)),
        key=lambda index: (
            float(enumeration[index]["true_joint_utility"]),
            -int(enumeration[index]["exact_added_cost"]),
            tuple(-value for value in enumeration[index]["action_cell_indices"]),
        ),
    )
    for method, index in (
        ("exact_learned_lookahead_reachable_pool", exact_index),
        ("retrospective_joint_near_oracle_reachable_pool", oracle_index),
    ):
        first, second, point_sum = pair_meta[index]
        plans[method] = ((first, second), pair_states[index], point_sum)
    enumeration_by_plan = {
        (
            tuple(row["action_cell_indices"]),
            tuple(row["action_from_levels"]),
            tuple(row["action_to_levels"]),
        ): row
        for row in enumeration
    }
    oracle_utility = float(enumeration[oracle_index]["true_joint_utility"])
    rows: list[dict[str, object]] = []
    for method in _METHODS:
        plan, state, point_sum = plans[method]
        plan_key = (
            tuple(action.cell_index for action in plan),
            tuple(action.from_level for action in plan),
            tuple(action.to_level for action in plan),
        )
        try:
            evaluated = enumeration_by_plan[plan_key]
        except KeyError as error:
            raise SetPlanningExecutionError(
                "learned plan escaped the reachable exact enumeration"
            ) from error
        prediction = float(evaluated["final_prediction"])
        final_error = float(evaluated["final_error"])
        utility = float(evaluated["true_joint_utility"])
        rows.append(
            {
                "outer_domain": domain,
                "specimen_id": specimen_id,
                "method": method,
                "diagnostic_checkpoint": checkpoint,
                "candidate_pool_width": len(reachable_pool),
                "set_size": 2,
                "action_cell_indices": [action.cell_index for action in plan],
                "action_from_levels": [action.from_level for action in plan],
                "action_to_levels": [action.to_level for action in plan],
                "exact_added_cost": state.exact_acquired_count
                - initial.exact_acquired_count,
                "exact_final_cost": state.exact_acquired_count,
                "native_count": state.native_count,
                "point_value_sum": point_sum,
                "initial_prediction": initial_prediction,
                "final_prediction": prediction,
                "target": target,
                "initial_error": initial_error,
                "final_error": final_error,
                "true_joint_utility": utility,
                "planning_regret": oracle_utility - utility,
                "terminal_state_sha256": state.state_sha256,
                "deployable": method
                != "retrospective_joint_near_oracle_reachable_pool",
            }
        )
    return rows, enumeration


def _bootstrap(
    results: pl.DataFrame,
    *,
    domains: tuple[str, ...],
    replicates: int,
    seed: int,
) -> pl.DataFrame:
    generator = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for method in _METHODS[:-1]:
        for replicate in range(replicates):
            domain_regrets: list[float] = []
            for domain in domains:
                values = results.filter(
                    (pl.col("outer_domain") == domain)
                    & (pl.col("method") == method)
                ).get_column("planning_regret").to_numpy()
                indices = generator.integers(0, len(values), len(values))
                domain_regrets.append(float(np.mean(values[indices])))
            rows.append(
                {
                    "method": method,
                    "reference_method": _METHODS[-1],
                    "replicate": replicate,
                    "oracle_minus_method_joint_utility": float(
                        np.mean(domain_regrets)
                    ),
                    "statistical_unit": "paired_physical_specimen_within_domain",
                }
            )
    return pl.DataFrame(rows).sort(["method", "replicate"])


def _report(summary: dict[str, object]) -> str:
    return (
        "# MAVIS P13 Set-Level Planning Diagnosis\n\n"
        "Status: `COMPLETE`.\n\n"
        "At the frozen 6.25% checkpoint, each method selects exactly two legal "
        "actions. The exact and retrospective rows enumerate every feasible "
        "ordered pair in the target-safe reachable pool: the initial learned "
        "top-8 plus any action reached by the registered greedy/beam plans. Final "
        "selection quality is evaluated with the true joint "
        "downstream CAI-error change of the complete set, never by summing point "
        "values. Beam widths 2 and 4 and lookahead width 8 were predeclared; no "
        "target outcome selected a width.\n\n"
        f"Current-greedy mean joint utility is `{summary['aggregate']['current_greedy']['joint_utility']:.10f}` "
        f"with planning regret `{summary['aggregate']['current_greedy']['planning_regret']:.10f}`. "
        f"The retrospective reachable-pool near-oracle utility is "
        f"`{summary['aggregate']['retrospective_joint_near_oracle_reachable_pool']['joint_utility']:.10f}`.\n\n"
        f"Interpretation: {summary['primary_conclusion']}\n\n"
        "This is a bounded diagnostic, not a deployable policy or an unrestricted "
        "oracle. It does not tune the P7 checkpoint and does not establish "
        "scanner-time reduction.\n"
    )


def run_p13_set_planning(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_project_root: str | Path,
    output_root: str | Path,
    device: str,
) -> Path:
    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        raise SetPlanningExecutionError("project root is unavailable") from error
    config = _load_config(config_path)
    mavis_path = _bound(root, config["mavis_config"], directory=False)
    p2_root = _bound(root, config["p2_checkpoint_root"], directory=True)
    p3_root = _bound(root, config["p3_checkpoint_root"], directory=True)
    p7_path = _bound(root, config["p7_package"], directory=True)
    if (
        _sha256(mavis_path) != config["mavis_config_sha256"]
        or _tree_state(p2_root) != config["p2_checkpoint_tree_sha256"]
        or _tree_state(p3_root) != config["p3_checkpoint_tree_sha256"]
        or _tree_state(p7_path) != config["p7_tree_state_sha256"]
    ):
        raise SetPlanningExecutionError("P13 frozen input hash changed")
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if root not in destination.parents or destination.exists():
        raise SetPlanningExecutionError("P13 output is invalid or already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p13_set_planning.", dir=destination.parent))
    p7_before = _tree_state(p7_path)
    try:
        mavis = load_mavis_config(mavis_path, project_root=root)
        domains = tuple(config["domain_order"])
        authority = load_mavis_authority(
            mavis, source_project_root=source_project_root
        )
        result_rows: list[dict[str, object]] = []
        enumeration_rows: list[dict[str, object]] = []
        checkpoint_hashes: list[tuple[str, str, str]] = []
        for domain in domains:
            p2 = load_fitted_mris_checkpoint(p2_root / f"{domain}__real.npz")
            p3 = load_fitted_dynamic_checkpoint(p3_root / f"{domain}__real.npz")
            scorer = DeployedDynamicScorer(
                mris_model=p2, dynamic_model=p3, device=device
            )
            checkpoint_hashes.append(
                (domain, p2.model_state_sha256, p3.model_state_sha256)
            )
            for specimen_id, dataset in zip(
                authority.specimen_ids, authority.dataset_ids, strict=True
            ):
                if dataset != domain:
                    continue
                rows, sets = _specimen_diagnostic(
                    authority,
                    specimen_id=specimen_id,
                    domain=domain,
                    initial_budget=mavis.initial_budget_by_domain[domain],
                    checkpoint=float(config["diagnostic_checkpoint"]),
                    scorer=scorer,
                    evaluator=p2,
                    device=device,
                )
                result_rows.extend(rows)
                enumeration_rows.extend(sets)
            p2.model.cpu()
            p3.model.cpu()
        results = pl.DataFrame(result_rows, infer_schema_length=None).sort(
            ["outer_domain", "specimen_id", "method"]
        )
        enumerated = pl.DataFrame(enumeration_rows, infer_schema_length=None).sort(
            ["outer_domain", "specimen_id", "set_index"]
        )
        per_domain = results.group_by("outer_domain", "method").agg(
            pl.col("specimen_id").n_unique().alias("specimen_count"),
            pl.col("true_joint_utility").mean().alias("mean_joint_utility"),
            pl.col("planning_regret").mean().alias("mean_planning_regret"),
            pl.col("final_error").mean().alias("mean_final_error"),
            pl.col("exact_added_cost").mean().alias("mean_exact_added_cost"),
        ).sort(["outer_domain", "method"])
        bootstrap = _bootstrap(
            results,
            domains=domains,
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["seed"]),
        )
        aggregate_rows = per_domain.group_by("method").agg(
            pl.col("mean_joint_utility").mean().alias("joint_utility"),
            pl.col("mean_planning_regret").mean().alias("planning_regret"),
            pl.col("mean_final_error").mean().alias("final_error"),
        ).sort("method")
        aggregate = {
            str(row["method"]): {
                "joint_utility": float(row["joint_utility"]),
                "planning_regret": float(row["planning_regret"]),
                "final_error": float(row["final_error"]),
            }
            for row in aggregate_rows.iter_rows(named=True)
        }
        intervals = {
            method: [
                float(
                    bootstrap.filter(pl.col("method") == method)
                    .get_column("oracle_minus_method_joint_utility")
                    .quantile(0.025)
                ),
                float(
                    bootstrap.filter(pl.col("method") == method)
                    .get_column("oracle_minus_method_joint_utility")
                    .quantile(0.975)
                ),
            ]
            for method in _METHODS[:-1]
        }
        current_interval = intervals["current_greedy"]
        conclusion = (
            "The bounded set-planning gap is supported because the paired "
            "current-greedy regret interval is strictly positive."
            if current_interval[0] > 0.0
            else "The bounded set-planning gap is not resolved because the paired current-greedy regret interval includes zero."
        )
        summary = {
            "schema_version": 1,
            "stage": "P13_SET_PLANNING",
            "audit_base_git_sha": config["audit_base_git_sha"],
            "config_sha256": config["config_sha256"],
            "domain_order": domains,
            "specimen_count": authority.specimen_count,
            "diagnostic_checkpoint": config["diagnostic_checkpoint"],
            "base_shortlist_width": config["base_shortlist_width"],
            "reachable_pool_width_range": [
                int(results.get_column("candidate_pool_width").min()),
                int(results.get_column("candidate_pool_width").max()),
            ],
            "beam_widths": config["beam_widths"],
            "lookahead_width": config["lookahead_width"],
            "set_size": config["set_size"],
            "method_order": _METHODS,
            "aggregate": aggregate,
            "planning_regret_intervals": intervals,
            "enumerated_set_count": enumerated.height,
            "policy_checkpoint_state_sha256": planning_state_sha256(
                checkpoint_hashes
            ),
            "p7_tree_state_sha256": p7_before,
            "oracle_deployable": False,
            "primary_conclusion": conclusion,
        }
        results.write_parquet(
            temporary / "set_results.parquet", compression="zstd", statistics=True
        )
        enumerated.write_parquet(
            temporary / "enumerated_sets.parquet",
            compression="zstd",
            statistics=True,
        )
        per_domain.write_csv(temporary / "per_domain.csv", float_scientific=False)
        bootstrap.write_csv(temporary / "bootstrap.csv", float_scientific=False)
        (temporary / "REPORT.md").write_text(_report(summary), encoding="utf-8")
        _write_json(temporary / "summary.json", summary)
        _write_json(
            temporary / "artifact_manifest.json",
            {
                "schema_version": 1,
                "stage": "P13_SET_PLANNING",
                "config_sha256": config["config_sha256"],
                "inputs": {
                    "p2_checkpoint_tree_sha256": config[
                        "p2_checkpoint_tree_sha256"
                    ],
                    "p3_checkpoint_tree_sha256": config[
                        "p3_checkpoint_tree_sha256"
                    ],
                    "p7_tree_state_sha256": p7_before,
                },
                "artifacts": sorted(_FILES - {"CHECKSUMS.sha256"}),
                "code_state_sha256": _sha256(Path(__file__)),
            },
        )
        _write_checksums(temporary)
        if _tree_state(p7_path) != p7_before:
            raise SetPlanningExecutionError("P13 modified frozen P7 artifacts")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_p13_set_planning_package(destination)
    return destination


def verify_p13_set_planning_package(path: str | Path) -> dict[str, object]:
    try:
        package = Path(path).resolve(strict=True)
    except OSError as error:
        raise SetPlanningExecutionError("P13 package is unavailable") from error
    if not package.is_dir() or {item.name for item in package.iterdir()} != _FILES:
        raise SetPlanningExecutionError("P13 package file roster changed")
    checksums: dict[str, str] = {}
    for line in (package / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    if set(checksums) != _FILES - {"CHECKSUMS.sha256"} or any(
        _sha256(package / name) != digest for name, digest in checksums.items()
    ):
        raise SetPlanningExecutionError("P13 package checksum mismatch")
    summary = json.loads((package / "summary.json").read_text(encoding="utf-8"))
    results = pl.read_parquet(package / "set_results.parquet")
    sets = pl.read_parquet(package / "enumerated_sets.parquet")
    domains = pl.read_csv(package / "per_domain.csv")
    bootstrap = pl.read_csv(package / "bootstrap.csv")
    if (
        summary.get("stage") != "P13_SET_PLANNING"
        or results.height != 276 * len(_METHODS)
        or results.get_column("method").unique().sort().to_list()
        != sorted(_METHODS)
        or results.filter(pl.col("set_size") != 2).height
        or results.filter(pl.col("exact_final_cost") / pl.col("native_count") > 0.0625).height
        or results.filter(pl.col("planning_regret") < -1.0e-12).height
        or results.filter(
            (pl.col("method") == _METHODS[-1]) & pl.col("deployable")
        ).height
        or sets.is_empty()
        or domains.height != 6 * len(_METHODS)
        or bootstrap.height != 5000 * (len(_METHODS) - 1)
    ):
        raise SetPlanningExecutionError("P13 scientific contract changed")
    return summary


__all__ = [
    "SetPlanningExecutionError",
    "run_p13_set_planning",
    "verify_p13_set_planning_package",
]
