"""Controlled planning substitutions for MAVIS science-closure diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path

import polars as pl
import yaml

from cmc_bbdm.mva.measurement_state import RefinementAction

from .authority import MAVISAuthority
from .closed_loop_execution import evaluate_inspection_curve
from .closed_loop_metrics import (
    bootstrap_closed_loop_contrasts,
    evaluate_closed_loop_predictions,
)
from .config import load_mavis_config
from .contracts import InspectionState
from .dynamic_training import load_fitted_dynamic_checkpoint
from .mris_training import load_fitted_mris_checkpoint
from .policy import DeployedDynamicScorer
from .reveal import reveal_action, reveal_action_history, reveal_uniform_scout
from .rollout import DeployableRolloutScorer, _candidate_descriptors, _scores


class ScienceClosurePlanningError(ValueError):
    """Raised when a diagnostic planner violates its frozen contract."""


_P12_FILES = {
    "substitution_matrix.csv",
    "per_specimen.csv",
    "per_domain.csv",
    "per_budget.csv",
    "bootstrap.csv",
    "REPORT.md",
    "summary.json",
    "artifact_manifest.json",
    "CHECKSUMS.sha256",
}


@dataclass(frozen=True, slots=True)
class PlanningCandidate:
    key: Hashable
    exact_added_cost: int
    point_value: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, Hashable)
            or type(self.exact_added_cost) is not int
            or self.exact_added_cost <= 0
            or isinstance(self.point_value, bool)
            or not math.isfinite(float(self.point_value))
        ):
            raise ScienceClosurePlanningError("planning candidate is invalid")


@dataclass(frozen=True, slots=True)
class JointSetSelection:
    keys: tuple[Hashable, ...]
    exact_cost: int
    joint_utility: float
    point_value_sum: float


@dataclass(frozen=True, slots=True)
class SubstitutionRow:
    row_id: str
    representation: str
    valuation: str
    planner: str
    deployable: bool
    cai_auebc: float
    policy_checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class LookaheadStep:
    step: int
    nominal_checkpoint: float
    action: RefinementAction
    exact_cost_before: int
    exact_cost_after: int
    immediate_score: float
    lookahead_score: float
    state_sha256_before: str
    state_sha256_after: str


@dataclass(frozen=True, slots=True)
class LookaheadCurve:
    specimen_id: str
    initial_budget: float
    checkpoints: tuple[float, ...]
    beam_width: int
    checkpoint_states: tuple[InspectionState, ...]
    steps: tuple[LookaheadStep, ...]
    state_sha256: str


def _stable_key(value: Hashable) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def select_joint_utility_set(
    candidates: tuple[PlanningCandidate, ...],
    *,
    exact_budget: int,
    set_size: int,
    joint_utility: Callable[[tuple[Hashable, ...]], float],
) -> JointSetSelection:
    """Enumerate a small frozen roster and optimize its actual joint utility."""

    if (
        type(candidates) is not tuple
        or not candidates
        or any(type(candidate) is not PlanningCandidate for candidate in candidates)
        or type(exact_budget) is not int
        or exact_budget <= 0
        or type(set_size) is not int
        or not 0 < set_size <= len(candidates)
        or not callable(joint_utility)
    ):
        raise ScienceClosurePlanningError("joint-set planning request is invalid")
    keys = tuple(candidate.key for candidate in candidates)
    if len(set(keys)) != len(keys):
        raise ScienceClosurePlanningError("planning candidate keys are duplicated")
    ordered = tuple(sorted(candidates, key=lambda candidate: _stable_key(candidate.key)))
    feasible = tuple(
        group
        for group in combinations(ordered, set_size)
        if sum(candidate.exact_added_cost for candidate in group) <= exact_budget
    )
    if not feasible:
        raise ScienceClosurePlanningError("no candidate set fits the exact budget")
    selections: list[JointSetSelection] = []
    for group in feasible:
        group_keys = tuple(candidate.key for candidate in group)
        utility = float(joint_utility(group_keys))
        if not math.isfinite(utility):
            raise ScienceClosurePlanningError("joint set utility is invalid")
        selections.append(
            JointSetSelection(
                keys=group_keys,
                exact_cost=sum(candidate.exact_added_cost for candidate in group),
                joint_utility=utility,
                point_value_sum=float(sum(candidate.point_value for candidate in group)),
            )
        )
    return max(
        selections,
        key=lambda selection: (
            selection.joint_utility,
            -selection.exact_cost,
            tuple(-ord(character) for character in "\0".join(map(_stable_key, selection.keys))),
        ),
    )


def build_registered_substitutions(
    *,
    current_auebc: float,
    learned_lookahead_auebc: float,
    true_greedy_auebc: float,
    true_set_auebc: float,
    policy_checkpoint_sha256: str,
) -> tuple[SubstitutionRow, ...]:
    values = (
        current_auebc,
        learned_lookahead_auebc,
        true_greedy_auebc,
        true_set_auebc,
    )
    if (
        any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values)
        or type(policy_checkpoint_sha256) is not str
        or len(policy_checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in policy_checkpoint_sha256)
    ):
        raise ScienceClosurePlanningError("substitution matrix request is invalid")
    checkpoint = policy_checkpoint_sha256
    return (
        SubstitutionRow("A", "R_learned", "V_learned", "P_current", True, float(current_auebc), checkpoint),
        SubstitutionRow("B", "current_causal_history", "V_true_conditional", "P_current", False, float(true_greedy_auebc), checkpoint),
        SubstitutionRow("C", "R_learned", "V_learned", "P_lookahead_beam2", True, float(learned_lookahead_auebc), checkpoint),
        SubstitutionRow("D", "current_causal_history", "V_true_conditional", "P_greedy_current", False, float(true_greedy_auebc), checkpoint),
        SubstitutionRow("E", "current_causal_history", "V_true_conditional", "P_set_near_oracle", False, float(true_set_auebc), checkpoint),
    )


def planning_state_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


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
        raise ScienceClosurePlanningError("bound artifact tree is empty")
    return planning_state_sha256(rows)


def _bound(root: Path, value: object, *, directory: bool) -> Path:
    if type(value) is not str or not value:
        raise ScienceClosurePlanningError("configured artifact path is invalid")
    try:
        path = (root / value).resolve(strict=True)
    except OSError as error:
        raise ScienceClosurePlanningError("configured artifact is unavailable") from error
    if root != path and root not in path.parents:
        raise ScienceClosurePlanningError("configured artifact escapes project root")
    if path.is_dir() != directory:
        raise ScienceClosurePlanningError("configured artifact type changed")
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


def _p12_config(path: str | Path) -> dict[str, object]:
    try:
        source = Path(path).resolve(strict=True)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ScienceClosurePlanningError("P12 config is unavailable") from error
    keys = {
        "schema_version",
        "stage",
        "audit_base_git_sha",
        "domain_order",
        "mavis_config",
        "mavis_config_sha256",
        "p4_predictions",
        "p4_predictions_sha256",
        "p2_checkpoint_root",
        "p2_checkpoint_tree_sha256",
        "p3_checkpoint_root",
        "p3_checkpoint_tree_sha256",
        "p7_package",
        "p7_tree_state_sha256",
        "learned_beam_width",
        "true_set_portfolio",
        "bootstrap_replicates",
        "seed",
    }
    hashes = (
        "mavis_config_sha256",
        "p4_predictions_sha256",
        "p2_checkpoint_tree_sha256",
        "p3_checkpoint_tree_sha256",
        "p7_tree_state_sha256",
    )
    if (
        type(payload) is not dict
        or set(payload) != keys
        or payload["schema_version"] != 1
        or payload["stage"] != "P12_RVP_ATTRIBUTION"
        or type(payload["domain_order"]) is not list
        or len(payload["domain_order"]) != 6
        or len(set(payload["domain_order"])) != 6
        or any(type(item) is not str or not item for item in payload["domain_order"])
        or any(
            type(payload[key]) is not str
            or len(payload[key]) != 64
            or any(character not in "0123456789abcdef" for character in payload[key])
            for key in hashes
        )
        or type(payload["audit_base_git_sha"]) is not str
        or len(payload["audit_base_git_sha"]) != 40
        or type(payload["learned_beam_width"]) is not int
        or payload["learned_beam_width"] <= 0
        or payload["true_set_portfolio"]
        != ["one_shot_mechanical_oracle", "sequential_mechanical_oracle"]
        or type(payload["bootstrap_replicates"]) is not int
        or payload["bootstrap_replicates"] < 2
        or type(payload["seed"]) is not int
        or isinstance(payload["seed"], bool)
    ):
        raise ScienceClosurePlanningError("P12 config schema changed")
    payload["config_sha256"] = _sha256(source)
    return payload


def _portfolio_near_oracle(predictions: pl.DataFrame) -> pl.DataFrame:
    methods = ["one_shot_mechanical_oracle", "sequential_mechanical_oracle"]
    keys = ["outer_domain", "specimen_id", "nominal_checkpoint"]
    selected = (
        predictions.filter(pl.col("method").is_in(methods))
        .sort([*keys, "absolute_error", "method"])
        .unique(subset=keys, keep="first", maintain_order=True)
        .with_columns(pl.lit("E").alias("method"))
    )
    expected = predictions.select("outer_domain", "specimen_id", "nominal_checkpoint").unique().height
    if selected.height != expected:
        raise ScienceClosurePlanningError("true-set planner portfolio is incomplete")
    return selected


def _substitution_report(summary: dict[str, object]) -> str:
    return (
        "# MAVIS P12 Representation-Valuation-Planning Attribution\n\n"
        "Status: `COMPLETE`.\n\n"
        "Rows A-E change only the registered diagnostic component shown in "
        "`substitution_matrix.csv`. B and D intentionally coincide: both are the "
        "frozen retrospective conditional mechanical-value trajectory under the "
        "current greedy planner. C is a causal two-step beam over the unchanged "
        "learned P2/P3 models. E is a retrospective joint downstream-error "
        "portfolio over the two pre-registered true-value set trajectories; it is "
        "a limited near-oracle, not an unrestricted set oracle.\n\n"
        f"Valuation substitution improvement (A-B): `{summary['valuation_improvement']:.10f}`.  "
        f"Learned planning substitution improvement (A-C): `{summary['learned_planning_improvement']:.10f}`.  "
        f"True-value planning substitution improvement (D-E): `{summary['true_planning_improvement']:.10f}`.  "
        f"Total limited-oracle improvement (A-E): `{summary['total_oracle_improvement']:.10f}`.\n\n"
        f"Paired 95% intervals for substitution-minus-reference AUEBC are "
        f"B-A `{summary['bootstrap_intervals']['B_minus_A']}`, C-A "
        f"`{summary['bootstrap_intervals']['C_minus_A']}`, E-D "
        f"`{summary['bootstrap_intervals']['E_minus_D']}`, and E-A "
        f"`{summary['bootstrap_intervals']['E_minus_A']}`.\n\n"
        "Lower CAI AUEBC is better. Positive improvement values favor the "
        "substituted row. Equal-domain aggregation follows specimen-first AUEBC. "
        "Rows B, D, and E use retrospective target outcomes and are explicitly "
        "non-deployable. P10 did not support adding optional row F: observed UT "
        "content did not improve over positions-only, so no new representation "
        "was trained. No P7 checkpoint or artifact was modified.\n"
    )


def run_p12_rvp_attribution(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_project_root: str | Path,
    output_root: str | Path,
    device: str,
) -> Path:
    """Execute and package the controlled P12 substitution matrix."""

    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        raise ScienceClosurePlanningError("project root is unavailable") from error
    config = _p12_config(config_path)
    mavis_config_path = _bound(root, config["mavis_config"], directory=False)
    p4_path = _bound(root, config["p4_predictions"], directory=False)
    p2_root = _bound(root, config["p2_checkpoint_root"], directory=True)
    p3_root = _bound(root, config["p3_checkpoint_root"], directory=True)
    p7_path = _bound(root, config["p7_package"], directory=True)
    if (
        _sha256(mavis_config_path) != config["mavis_config_sha256"]
        or _sha256(p4_path) != config["p4_predictions_sha256"]
        or _tree_state(p2_root) != config["p2_checkpoint_tree_sha256"]
        or _tree_state(p3_root) != config["p3_checkpoint_tree_sha256"]
        or _tree_state(p7_path) != config["p7_tree_state_sha256"]
    ):
        raise ScienceClosurePlanningError("P12 frozen input hash changed")
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if root not in destination.parents or destination.exists():
        raise ScienceClosurePlanningError("P12 output is invalid or already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p12_rvp_attribution.", dir=destination.parent))
    p7_before = _tree_state(p7_path)
    try:
        mavis_config = load_mavis_config(mavis_config_path, project_root=root)
        domains = tuple(config["domain_order"])
        if mavis_config.domain_order != domains:
            raise ScienceClosurePlanningError("P12 domain roster changed")
        from .authority import load_mavis_authority

        authority = load_mavis_authority(
            mavis_config,
            source_project_root=source_project_root,
        )
        frozen = pl.read_parquet(p4_path)
        generated_rows: list[dict[str, object]] = []
        curve_hashes: list[tuple[str, str, str]] = []
        checkpoint_hashes: list[tuple[str, str, str]] = []
        for domain_index, domain in enumerate(domains):
            p2 = load_fitted_mris_checkpoint(p2_root / f"{domain}__real.npz")
            p3 = load_fitted_dynamic_checkpoint(p3_root / f"{domain}__real.npz")
            scorer = DeployedDynamicScorer(
                mris_model=p2,
                dynamic_model=p3,
                device=device,
            )
            checkpoint_hashes.append(
                (domain, p2.model_state_sha256, p3.model_state_sha256)
            )
            specimen_ids = tuple(
                specimen
                for specimen, dataset in zip(
                    authority.specimen_ids,
                    authority.dataset_ids,
                    strict=True,
                )
                if dataset == domain
            )
            for specimen_id in specimen_ids:
                curve = rollout_learned_lookahead_curve(
                    authority,
                    specimen_id=specimen_id,
                    initial_budget=mavis_config.initial_budget_by_domain[domain],
                    checkpoints=mavis_config.checkpoints,
                    scorer=scorer,
                    beam_width=int(config["learned_beam_width"]),
                )
                curve_hashes.append((domain, specimen_id, curve.state_sha256))
                generated_rows.extend(
                    evaluate_inspection_curve(
                        authority,
                        outer_domain=domain,
                        method="C",
                        checkpoints=mavis_config.checkpoints,
                        states=curve.checkpoint_states,
                        cai_evaluator=p2,
                        device=device,
                    )
                )
            p2.model.cpu()
            p3.model.cpu()
        current = frozen.filter(pl.col("method") == "mavis_full").with_columns(
            pl.lit("A").alias("method")
        )
        true_greedy = frozen.filter(
            pl.col("method") == "sequential_mechanical_oracle"
        )
        row_b = true_greedy.with_columns(pl.lit("B").alias("method"))
        row_d = true_greedy.with_columns(pl.lit("D").alias("method"))
        row_e = _portfolio_near_oracle(frozen)
        combined = pl.concat(
            [current, row_b, pl.DataFrame(generated_rows), row_d, row_e],
            how="vertical_relaxed",
        ).sort(["outer_domain", "specimen_id", "method", "nominal_checkpoint"])
        metrics = evaluate_closed_loop_predictions(
            combined,
            domain_order=domains,
            method_order=("A", "B", "C", "D", "E"),
            checkpoints=mavis_config.checkpoints,
        )
        aggregate = {
            str(row["method"]): float(row["domain_balanced_cai_auebc"])
            for row in metrics.aggregate_auebc.iter_rows(named=True)
        }
        policy_hash = planning_state_sha256(checkpoint_hashes)
        registered = build_registered_substitutions(
            current_auebc=aggregate["A"],
            learned_lookahead_auebc=aggregate["C"],
            true_greedy_auebc=aggregate["B"],
            true_set_auebc=aggregate["E"],
            policy_checkpoint_sha256=policy_hash,
        )
        bootstrap_a = bootstrap_closed_loop_contrasts(
            metrics.specimen_auebc,
            reference_method="A",
            control_methods=("B", "C", "E"),
            domain_order=domains,
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["seed"]),
        )
        bootstrap_d = bootstrap_closed_loop_contrasts(
            metrics.specimen_auebc,
            reference_method="D",
            control_methods=("E",),
            domain_order=domains,
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["seed"]) + 1,
        )
        bootstrap = pl.concat([bootstrap_a, bootstrap_d]).with_columns(
            (
                pl.col("control_method")
                + pl.lit("_minus_")
                + pl.col("reference_method")
            ).alias("contrast_id")
        ).sort(["contrast_id", "replicate"])
        interval_rows = bootstrap.group_by("contrast_id").agg(
            pl.col("control_minus_reference_cai_auebc").mean().alias("mean"),
            pl.col("control_minus_reference_cai_auebc").quantile(0.025).alias("lower"),
            pl.col("control_minus_reference_cai_auebc").quantile(0.975).alias("upper"),
        ).sort("contrast_id")
        intervals = {
            str(row["contrast_id"]): [
                float(row["lower"]),
                float(row["upper"]),
            ]
            for row in interval_rows.iter_rows(named=True)
        }
        matrix = pl.DataFrame(
            [
                {
                    "row_id": row.row_id,
                    "representation": row.representation,
                    "valuation": row.valuation,
                    "planner": row.planner,
                    "deployable": row.deployable,
                    "cai_auebc": row.cai_auebc,
                    "substitution_minus_A": row.cai_auebc - aggregate["A"],
                    "A_minus_substitution_improvement": aggregate["A"] - row.cai_auebc,
                    "policy_checkpoint_sha256": row.policy_checkpoint_sha256,
                }
                for row in registered
            ]
        )
        component = matrix.select(
            "row_id", "representation", "valuation", "planner", "deployable"
        )
        per_domain = metrics.domain_auebc.rename(
            {"method": "row_id", "cai_auebc": "domain_cai_auebc"}
        ).join(component, on="row_id", how="left", validate="m:1")
        per_budget = metrics.aggregate_curve.rename(
            {"method": "row_id"}
        ).join(component, on="row_id", how="left", validate="m:1")
        summary = {
            "schema_version": 1,
            "stage": "P12_RVP_ATTRIBUTION",
            "audit_base_git_sha": config["audit_base_git_sha"],
            "config_sha256": config["config_sha256"],
            "domain_order": domains,
            "specimen_count": authority.specimen_count,
            "learned_beam_width": config["learned_beam_width"],
            "true_set_portfolio": config["true_set_portfolio"],
            "bootstrap_replicates": config["bootstrap_replicates"],
            "bootstrap_seed": config["seed"],
            "bootstrap_intervals": intervals,
            "optional_row_f": "OMITTED_P10_DID_NOT_SUPPORT_REPRESENTATION_BOTTLENECK",
            "policy_checkpoint_state_sha256": policy_hash,
            "lookahead_curve_state_sha256": planning_state_sha256(curve_hashes),
            "p7_tree_state_sha256": p7_before,
            "valuation_improvement": aggregate["A"] - aggregate["B"],
            "learned_planning_improvement": aggregate["A"] - aggregate["C"],
            "true_planning_improvement": aggregate["D"] - aggregate["E"],
            "total_oracle_improvement": aggregate["A"] - aggregate["E"],
            "row_auebc": aggregate,
            "lower_is_better": True,
            "oracle_rows_deployable": False,
        }
        matrix.write_csv(temporary / "substitution_matrix.csv", float_scientific=False)
        metrics.specimen_auebc.write_csv(
            temporary / "per_specimen.csv", float_scientific=False
        )
        per_domain.write_csv(temporary / "per_domain.csv", float_scientific=False)
        per_budget.write_csv(temporary / "per_budget.csv", float_scientific=False)
        bootstrap.write_csv(temporary / "bootstrap.csv", float_scientific=False)
        (temporary / "REPORT.md").write_text(
            _substitution_report(summary), encoding="utf-8"
        )
        _write_json(temporary / "summary.json", summary)
        manifest = {
            "schema_version": 1,
            "stage": "P12_RVP_ATTRIBUTION",
            "config_sha256": config["config_sha256"],
            "inputs": {
                "p4_predictions_sha256": config["p4_predictions_sha256"],
                "p2_checkpoint_tree_sha256": config["p2_checkpoint_tree_sha256"],
                "p3_checkpoint_tree_sha256": config["p3_checkpoint_tree_sha256"],
                "p7_tree_state_sha256": p7_before,
            },
            "artifacts": sorted(_P12_FILES - {"CHECKSUMS.sha256"}),
            "code_state_sha256": planning_state_sha256(
                [
                    (Path(__file__).name, _sha256(Path(__file__))),
                    ("closed_loop_metrics.py", _sha256(Path(__file__).with_name("closed_loop_metrics.py"))),
                ]
            ),
        }
        _write_json(temporary / "artifact_manifest.json", manifest)
        _write_checksums(temporary)
        if _tree_state(p7_path) != p7_before:
            raise ScienceClosurePlanningError("P12 modified frozen P7 artifacts")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_p12_rvp_attribution_package(destination)
    return destination


def verify_p12_rvp_attribution_package(path: str | Path) -> dict[str, object]:
    try:
        package = Path(path).resolve(strict=True)
    except OSError as error:
        raise ScienceClosurePlanningError("P12 package is unavailable") from error
    if not package.is_dir() or {item.name for item in package.iterdir()} != _P12_FILES:
        raise ScienceClosurePlanningError("P12 package file roster changed")
    checksums: dict[str, str] = {}
    for line in (package / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    expected = _P12_FILES - {"CHECKSUMS.sha256"}
    if set(checksums) != expected or any(
        _sha256(package / name) != digest for name, digest in checksums.items()
    ):
        raise ScienceClosurePlanningError("P12 package checksum mismatch")
    summary = json.loads((package / "summary.json").read_text(encoding="utf-8"))
    matrix = pl.read_csv(package / "substitution_matrix.csv")
    domains = pl.read_csv(package / "per_domain.csv")
    budgets = pl.read_csv(package / "per_budget.csv")
    specimens = pl.read_csv(package / "per_specimen.csv")
    bootstrap = pl.read_csv(package / "bootstrap.csv")
    if (
        summary.get("stage") != "P12_RVP_ATTRIBUTION"
        or matrix.get_column("row_id").to_list() != ["A", "B", "C", "D", "E"]
        or matrix.filter(pl.col("row_id").is_in(["B", "D", "E"]) & pl.col("deployable")).height
        or matrix.filter(pl.col("row_id").is_in(["A", "C"]) & ~pl.col("deployable")).height
        or domains.height != 30
        or budgets.height != 30
        or domains.get_column("outer_domain").n_unique() != 6
        or budgets.get_column("nominal_checkpoint").n_unique() != 6
        or specimens.height != 1380
        or specimens.get_column("specimen_id").n_unique() != 276
        or bootstrap.get_column("contrast_id").n_unique() != 4
        or bootstrap.group_by("contrast_id").len().get_column("len").unique().to_list()
        != [5000]
        or matrix.get_column("policy_checkpoint_sha256").n_unique() != 1
    ):
        raise ScienceClosurePlanningError("P12 package scientific contract changed")
    return summary


def _objective_scores(
    scorer: DeployableRolloutScorer,
    state: InspectionState,
    candidates: tuple[object, ...],
    *,
    objective: str,
) -> tuple[float, ...]:
    values = _scores(
        scorer.score_actions(state, candidates),
        len(candidates),
        objective=objective,
    )
    if objective == "value_per_exact_cost":
        values = values / [candidate.exact_added_cost for candidate in candidates]
    return tuple(float(value) for value in values)


def select_learned_lookahead_action(
    authority: MAVISAuthority,
    state: InspectionState,
    *,
    endpoint_budget: float,
    action_budget: float,
    scorer: DeployableRolloutScorer,
    objective: str = "direct_cost_aware",
    beam_width: int = 2,
) -> tuple[RefinementAction, float, float]:
    """Select the first action of a two-step causal learned-value beam."""

    if (
        type(authority) is not MAVISAuthority
        or type(state) is not InspectionState
        or not hasattr(scorer, "score_actions")
        or type(beam_width) is not int
        or beam_width <= 0
        or objective not in {"direct_cost_aware", "raw_score", "value_per_exact_cost"}
    ):
        raise ScienceClosurePlanningError("lookahead planning request is invalid")
    actions, candidates = _candidate_descriptors(
        state,
        endpoint_budget=float(endpoint_budget),
        action_budget=float(action_budget),
    )
    if not actions:
        raise ScienceClosurePlanningError("lookahead planner has no legal action")
    immediate = _objective_scores(scorer, state, candidates, objective=objective)
    ranked = sorted(
        range(len(actions)),
        key=lambda index: (
            -immediate[index],
            actions[index].cell_index,
            actions[index].to_level,
        ),
    )[:beam_width]
    evaluated: list[tuple[float, float, int]] = []
    for index in ranked:
        branch = reveal_action(authority, state, actions[index])
        _next_actions, next_candidates = _candidate_descriptors(
            branch,
            endpoint_budget=float(endpoint_budget),
            action_budget=float(action_budget),
        )
        future = 0.0
        if next_candidates:
            future = max(
                _objective_scores(scorer, branch, next_candidates, objective=objective)
            )
        evaluated.append((immediate[index] + future, immediate[index], index))
    total, first, selected = max(
        evaluated,
        key=lambda item: (
            item[0],
            item[1],
            -actions[item[2]].cell_index,
            -actions[item[2]].to_level,
        ),
    )
    return actions[selected], first, total


def rollout_learned_lookahead_curve(
    authority: MAVISAuthority,
    *,
    specimen_id: str,
    initial_budget: float,
    checkpoints: tuple[float, ...],
    scorer: DeployableRolloutScorer,
    beam_width: int = 2,
) -> LookaheadCurve:
    """Execute the learned two-step beam under the registered exact-cost caps."""

    if (
        type(authority) is not MAVISAuthority
        or type(specimen_id) is not str
        or not specimen_id
        or type(checkpoints) is not tuple
        or not checkpoints
        or any(
            isinstance(checkpoint, bool) or not math.isfinite(float(checkpoint))
            for checkpoint in checkpoints
        )
        or any(second <= first for first, second in pairwise(checkpoints))
    ):
        raise ScienceClosurePlanningError("lookahead curve request is invalid")
    endpoint = float(checkpoints[-1])
    current = reveal_uniform_scout(
        authority,
        authority.policy_context(specimen_id),
        initial_budget=float(initial_budget),
        checkpoint=endpoint,
    )
    snapshots: list[InspectionState] = []
    steps: list[LookaheadStep] = []
    for checkpoint_index, checkpoint_raw in enumerate(checkpoints):
        checkpoint = float(checkpoint_raw)
        if checkpoint_index or current.effective_budget <= checkpoint + 1.0e-15:
            current = reveal_action_history(
                authority,
                authority.policy_context(specimen_id),
                initial_budget=float(initial_budget),
                checkpoint=checkpoint,
                actions=current.action_history,
            )
        while True:
            actions, _candidates = _candidate_descriptors(
                current,
                endpoint_budget=endpoint,
                action_budget=checkpoint,
            )
            if not actions:
                break
            action, immediate, lookahead = select_learned_lookahead_action(
                authority,
                current,
                endpoint_budget=endpoint,
                action_budget=checkpoint,
                scorer=scorer,
                beam_width=beam_width,
            )
            previous = current
            current = reveal_action(authority, current, action)
            if current.effective_budget > checkpoint + 1.0e-15:
                raise ScienceClosurePlanningError("lookahead action exceeded exact budget")
            steps.append(
                LookaheadStep(
                    step=len(steps),
                    nominal_checkpoint=checkpoint,
                    action=action,
                    exact_cost_before=previous.exact_acquired_count,
                    exact_cost_after=current.exact_acquired_count,
                    immediate_score=immediate,
                    lookahead_score=lookahead,
                    state_sha256_before=previous.state_sha256,
                    state_sha256_after=current.state_sha256,
                )
            )
        snapshots.append(current)
    payload = {
        "schema": 1,
        "specimen_id": specimen_id,
        "initial_budget": float(initial_budget),
        "checkpoints": checkpoints,
        "beam_width": beam_width,
        "checkpoint_states": [state.state_sha256 for state in snapshots],
        "actions": [
            (step.action.cell_index, step.action.from_level, step.action.to_level)
            for step in steps
        ],
    }
    return LookaheadCurve(
        specimen_id=specimen_id,
        initial_budget=float(initial_budget),
        checkpoints=tuple(float(value) for value in checkpoints),
        beam_width=beam_width,
        checkpoint_states=tuple(snapshots),
        steps=tuple(steps),
        state_sha256=planning_state_sha256(payload),
    )


__all__ = [
    "JointSetSelection",
    "LookaheadCurve",
    "LookaheadStep",
    "PlanningCandidate",
    "ScienceClosurePlanningError",
    "SubstitutionRow",
    "build_registered_substitutions",
    "planning_state_sha256",
    "rollout_learned_lookahead_curve",
    "run_p12_rvp_attribution",
    "select_joint_utility_set",
    "select_learned_lookahead_action",
    "verify_p12_rvp_attribution_package",
]
