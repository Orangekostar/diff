#!/usr/bin/env python3
"""Run the registered pre-outer D8 exploration commands."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from cmc_bbdm.cpb_diffusion_marginalization.artifacts import (
    publish_d8_search_package,
    validate_d8_search_package,
)
from cmc_bbdm.cpb_diffusion_marginalization.baseline import (
    reproduce_internal_only_baseline,
)
from cmc_bbdm.cpb_diffusion_marginalization.config import (
    DOMAIN_ORDER,
    load_d8_config,
)
from cmc_bbdm.cpb_diffusion_marginalization.pilot import (
    build_pilot_escalation_evidence,
    run_registered_pilot,
)
from cmc_bbdm.cpb_diffusion_marginalization.residuals import (
    build_cross_fitted_p6_residual_bank,
)
from cmc_bbdm.cpb_diffusion_marginalization.tracking import TRIAL_INDEX_FIELDS
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTERED_CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_exploration.yaml"
REGISTERED_OUTPUT = PROJECT_ROOT / "results/d8_search"
_COMMANDS = ("baseline", "residual-bank", "pilot", "validate")
_SEARCH_SUMMARY_FIELDS = (
    "outer_domain",
    "initial_trial_count",
    "trial_count",
    "completed_count",
    "pruned_count",
    "failed_count",
    "best_objective",
    "best_candidate_sha256",
    "selection_state_sha256",
    "escalation_evidence_sha256",
)
_SOURCE_FILES = frozenset(
    {
        "trial_index.csv",
        "study.db",
        "residual_bank_manifest.json",
        "search_summary.csv",
        "selected_configs.json",
        "escalation_evidence.json",
        "pilot_report.md",
    }
)


class D8ExecutionError(RuntimeError):
    """Raised when a registered D8 execution stage is unavailable."""


def _load_authorities() -> tuple[object, object]:
    config = load_d8_config(REGISTERED_CONFIG, project_root=PROJECT_ROOT)
    p1 = load_v3_config(
        PROJECT_ROOT / config.sources["p1_config"].path,
        project_root=PROJECT_ROOT,
    )
    return config, load_data(p1, PROJECT_ROOT)


def execute_baseline() -> dict[str, object]:
    config, data = _load_authorities()
    result = reproduce_internal_only_baseline(
        data,
        config=config,
        project_root=PROJECT_ROOT,
        device="cuda:0",
    )
    if (
        not math.isclose(
            result.equal_domain_mae,
            config.baseline_mae,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or result.maximum_prediction_error > 1.0e-12
        or result.maximum_target_error > 1.0e-12
    ):
        raise D8ExecutionError("registered D8 baseline reproduction failed")
    return {
        "command": "baseline",
        "status": "PASS",
        "specimen_count": result.specimen_count,
        "equal_domain_mae": result.equal_domain_mae,
        "maximum_prediction_error": result.maximum_prediction_error,
        "maximum_target_error": result.maximum_target_error,
        "state_sha256": result.state_sha256,
    }


def execute_residual_bank() -> dict[str, object]:
    config, data = _load_authorities()
    bank = build_cross_fitted_p6_residual_bank(
        data,
        config=config,
        project_root=PROJECT_ROOT,
        device="cuda",
    )
    if (
        bank.specimen_count != 276
        or bank.draw_count != config.p6_draws
        or len(bank.records) != 276 * config.p6_draws
        or bank.maximum_mean_error > 1.0e-6
        or bank.maximum_variance_error > 1.0e-6
    ):
        raise D8ExecutionError("registered D8 residual-bank validation failed")
    return {
        "command": "residual-bank",
        "status": "PASS",
        "specimen_count": bank.specimen_count,
        "draw_count": bank.draw_count,
        "record_count": len(bank.records),
        "maximum_mean_error": bank.maximum_mean_error,
        "maximum_variance_error": bank.maximum_variance_error,
        "state_sha256": bank.state_sha256,
    }


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


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )


def _read_trial_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if tuple(reader.fieldnames or ()) != TRIAL_INDEX_FIELDS:
            raise D8ExecutionError("registered D8 trial schema changed")
        rows = tuple(dict(row) for row in reader)
    if not rows or any(
        None in row or any(value is None for value in row.values()) for row in rows
    ):
        raise D8ExecutionError("registered D8 trial rows are incomplete")
    return rows


def _load_selection_documents(root: Path) -> tuple[dict[str, object], ...]:
    documents = []
    for outer_domain in DOMAIN_ORDER:
        path = root / f"{outer_domain}.json"
        try:
            value = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise D8ExecutionError("registered D8 selection is unavailable") from error
        if (
            not isinstance(value, dict)
            or value.get("outer_domain") != outer_domain
            or type(value.get("state_sha256")) is not str
        ):
            raise D8ExecutionError("registered D8 selection authority changed")
        documents.append(value)
    return tuple(documents)


def _best_complete_trial(
    rows: tuple[dict[str, str], ...], outer_domain: str
) -> dict[str, str]:
    complete = tuple(
        row
        for row in rows
        if row["outer_fold"] == outer_domain and row["state"] == "COMPLETE"
    )
    try:
        ranked = tuple(
            sorted(
                complete,
                key=lambda row: (float(row["objective"]), row["candidate_sha256"]),
            )
        )
    except ValueError as error:
        raise D8ExecutionError("registered D8 objective is invalid") from error
    if not ranked or not math.isfinite(float(ranked[0]["objective"])):
        raise D8ExecutionError("registered D8 complete trial is unavailable")
    return ranked[0]


def _write_pilot_source(
    root: Path,
    *,
    config: object,
    bank: object,
    run: object,
) -> str:
    config_sha256 = getattr(config, "config_sha256", None)
    residual_sha256 = getattr(bank, "state_sha256", None)
    outer_runs = getattr(run, "outer_runs", None)
    if (
        getattr(run, "config_sha256", None) != config_sha256
        or getattr(run, "residual_bank_sha256", None) != residual_sha256
        or getattr(run, "outer_evaluation_count", None) != 0
        or type(outer_runs) is not tuple
        or tuple(getattr(item, "outer_domain", None) for item in outer_runs)
        != DOMAIN_ORDER
    ):
        raise D8ExecutionError("registered D8 pilot authority changed")
    rows = _read_trial_rows(root / "trial_index.csv")
    selection_root = root / "best_inner_configs"
    selections = _load_selection_documents(selection_root)
    decision = build_pilot_escalation_evidence(
        rows,
        selections=selections,
        bank=bank,
        config=config,
    )
    studies = getattr(decision, "studies", None)
    if (
        type(studies) not in {tuple, list}
        or tuple(getattr(item, "outer_domain", None) for item in studies)
        != DOMAIN_ORDER
    ):
        raise D8ExecutionError("registered D8 escalation roster changed")
    study_by_outer = {item.outer_domain: item for item in studies}
    summaries = []
    for outer_run, selection in zip(outer_runs, selections, strict=True):
        outer_domain = outer_run.outer_domain
        search = outer_run.search
        best = _best_complete_trial(rows, outer_domain)
        if selection["state_sha256"] != outer_run.selection.state_sha256:
            raise D8ExecutionError("registered D8 selection state changed")
        summaries.append(
            {
                "outer_domain": outer_domain,
                "initial_trial_count": search.initial_trial_count,
                "trial_count": search.trial_count,
                "completed_count": search.completed_count,
                "pruned_count": search.pruned_count,
                "failed_count": search.failed_count,
                "best_objective": float(best["objective"]),
                "best_candidate_sha256": best["candidate_sha256"],
                "selection_state_sha256": selection["state_sha256"],
                "escalation_evidence_sha256": study_by_outer[
                    outer_domain
                ].state_sha256,
            }
        )
    with (root / "search_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_SEARCH_SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerows(summaries)
    _write_json(
        root / "selected_configs.json",
        {
            "schema_version": 2,
            "scope": "d8_prospective_outer_selections",
            "config_sha256": config_sha256,
            "outer_evaluation_count": 0,
            "selections": list(selections),
            "state_sha256": _canonical_sha256(list(selections)),
        },
    )
    _write_json(root / "escalation_evidence.json", decision.to_payload())
    _write_json(
        root / "residual_bank_manifest.json",
        {
            "schema_version": 1,
            "scope": "d8_cross_fitted_p6_residual_bank",
            "specimen_count": bank.specimen_count,
            "draw_count": bank.draw_count,
            "record_count": len(bank.records),
            "maximum_mean_error": bank.maximum_mean_error,
            "maximum_variance_error": bank.maximum_variance_error,
            "state_sha256": residual_sha256,
        },
    )
    decision_name = getattr(decision, "decision", None)
    if type(decision_name) is not str:
        raise D8ExecutionError("registered D8 escalation decision is unavailable")
    (root / "pilot_report.md").write_text(
        "# D8 Pre-Outer Pilot\n\n"
        f"Decision: `{decision_name}`\n\n"
        "Outer evaluations: `0`.\n",
        encoding="ascii",
    )
    shutil.rmtree(selection_root)
    if {path.name for path in root.iterdir()} != _SOURCE_FILES:
        raise D8ExecutionError("registered D8 source package file set changed")
    return decision_name


def execute_pilot() -> dict[str, object]:
    config, data = _load_authorities()
    bank = build_cross_fitted_p6_residual_bank(
        data,
        config=config,
        project_root=PROJECT_ROOT,
        device="cuda",
    )
    REGISTERED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".d8-pilot-source-", dir=REGISTERED_OUTPUT.parent
    ) as directory:
        source = Path(directory)
        run = run_registered_pilot(
            data,
            config=config,
            bank=bank,
            project_root=PROJECT_ROOT,
            output=source,
            device="cuda:0",
        )
        decision = _write_pilot_source(
            source,
            config=config,
            bank=bank,
            run=run,
        )
        result = publish_d8_search_package(
            source,
            REGISTERED_OUTPUT,
            project_root=PROJECT_ROOT,
            config_path=REGISTERED_CONFIG,
            escalation_status=decision,
        )
    return {
        "command": "pilot",
        "status": "PASS",
        "outer_domains": list(result.outer_domains),
        "trial_count": result.trial_count,
        "outer_evaluation_count": result.outer_evaluation_count,
        "escalation_status": result.escalation_status,
        "scientific_digest": result.scientific_digest,
        "output_tree_sha256": result.output_tree_sha256,
    }


def execute_validate() -> dict[str, object]:
    result = validate_d8_search_package(
        REGISTERED_OUTPUT,
        project_root=PROJECT_ROOT,
        config_path=REGISTERED_CONFIG,
    )
    return {
        "command": "validate",
        "status": "PASS",
        "outer_domains": list(result.outer_domains),
        "trial_count": result.trial_count,
        "outer_evaluation_count": result.outer_evaluation_count,
        "escalation_status": result.escalation_status,
        "scientific_digest": result.scientific_digest,
        "output_tree_sha256": result.output_tree_sha256,
    }


def run_registered_command(command: str) -> dict[str, object]:
    functions = {
        "baseline": execute_baseline,
        "residual-bank": execute_residual_bank,
        "pilot": execute_pilot,
        "validate": execute_validate,
    }
    try:
        function = functions[command]
    except KeyError as error:
        raise D8ExecutionError("D8 command is not registered") from error
    return function()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the registered D8 pre-outer exploration stages."
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
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
