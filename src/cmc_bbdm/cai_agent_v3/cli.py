"""Command-line entry points for the staged CAI Agent v3 execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .actor_training import (
    precheck_policy_training,
    run_policy_expansion,
    run_policy_pilots,
)
from .cohort_export import export_cohort
from .diagnostics import export_policy_diagnostics
from .feature_bank import build_feature_bank
from .final_evaluation import run_internal_test_evaluation
from .gdfs_training import precheck_gdfs_training, run_gdfs_pilot
from .legacy_rescore import rescore_v2
from .predictor_training import (
    precheck_predictor_training,
    refresh_cost_precision_evaluations,
    run_oof_reward_predictors,
    run_predictor_candidates,
)
from .vlm_perception import run_vlm_perception


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="Repository root."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("rescore-v2", help="Correct legacy v2 trajectory statistics.")
    subparsers.add_parser("build-cohort", help="Build the 276-candidate grouped split.")
    subparsers.add_parser(
        "prepare", help="Run the non-training W0 rescore and W1 cohort export."
    )
    encode = subparsers.add_parser(
        "encode-features", help="Build/reuse the frozen sharded feature bank."
    )
    encode.add_argument("--source-root", type=Path, required=True)
    encode.add_argument("--device", default="cuda:0")
    predictors = subparsers.add_parser(
        "train-predictors",
        help="Run fixed Ridge diagnostics and three neural candidates.",
    )
    predictors.add_argument("--device", default="cuda:0")
    precheck = subparsers.add_parser(
        "precheck-predictors", help="Run one update per predictor candidate."
    )
    precheck.add_argument("--device", default="cuda:0")
    oof = subparsers.add_parser(
        "train-oof", help="Train three capture-group-excluded reward predictors."
    )
    oof.add_argument("--device", default="cuda:0")
    refresh = subparsers.add_parser(
        "refresh-cost-evaluation",
        help="Re-evaluate frozen W2 checkpoints with float64 native costs.",
    )
    refresh.add_argument("--device", default="cuda:0")
    vlm = subparsers.add_parser(
        "run-vlm", help="Extend the frozen Qwen surface cache for a gated scope."
    )
    vlm.add_argument("--source-root", type=Path, required=True)
    vlm.add_argument("--scope", choices=("fit", "test"), required=True)
    actors = subparsers.add_parser(
        "train-policy-pilots", help="Train the gated seed-1 policy matrix."
    )
    actors.add_argument("--device", default="cuda:0")
    expansion = subparsers.add_parser(
        "train-policy-expansion", help="Conditionally train policy seed panels 2 and 3."
    )
    expansion.add_argument("--device", default="cuda:0")
    actor_precheck = subparsers.add_parser(
        "precheck-policies", help="Run one update per policy architecture."
    )
    actor_precheck.add_argument("--device", default="cuda:0")
    gdfs_precheck = subparsers.add_parser(
        "precheck-gdfs", help="Run one grouped-Concrete selector update."
    )
    gdfs_precheck.add_argument("--device", default="cuda:0")
    gdfs = subparsers.add_parser(
        "train-gdfs-pilot", help="Train the one authorized frozen-predictor pilot."
    )
    gdfs.add_argument("--device", default="cuda:0")
    diagnostics = subparsers.add_parser(
        "export-diagnostics", help="Export fixed-hash policy traces and figures."
    )
    diagnostics.add_argument("--source-root", type=Path, required=True)
    diagnostics.add_argument("--scope", choices=("valid", "test"), default="valid")
    test_eval = subparsers.add_parser(
        "evaluate-test", help="Run the once-only conditionally unlocked internal TEST."
    )
    test_eval.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    payload: dict[str, Any]
    if args.command == "rescore-v2":
        payload = rescore_v2(project_root=args.project_root)
    elif args.command == "build-cohort":
        payload = export_cohort(project_root=args.project_root)
    elif args.command == "encode-features":
        payload = build_feature_bank(
            project_root=args.project_root,
            source_root=args.source_root,
            device=args.device,
        )
    elif args.command == "train-predictors":
        payload = run_predictor_candidates(
            project_root=args.project_root, device=args.device
        )
    elif args.command == "precheck-predictors":
        payload = precheck_predictor_training(
            project_root=args.project_root, device=args.device
        )
    elif args.command == "train-oof":
        payload = run_oof_reward_predictors(
            project_root=args.project_root, device=args.device
        )
    elif args.command == "refresh-cost-evaluation":
        payload = refresh_cost_precision_evaluations(
            project_root=args.project_root, device=args.device
        )
    elif args.command == "run-vlm":
        payload = run_vlm_perception(
            project_root=args.project_root,
            source_root=args.source_root,
            scope=args.scope,
        )
    elif args.command == "train-policy-pilots":
        payload = run_policy_pilots(project_root=args.project_root, device=args.device)
    elif args.command == "train-policy-expansion":
        payload = run_policy_expansion(
            project_root=args.project_root, device=args.device
        )
    elif args.command == "precheck-policies":
        payload = precheck_policy_training(
            project_root=args.project_root, device=args.device
        )
    elif args.command == "precheck-gdfs":
        payload = precheck_gdfs_training(
            project_root=args.project_root, device=args.device
        )
    elif args.command == "train-gdfs-pilot":
        payload = run_gdfs_pilot(project_root=args.project_root, device=args.device)
    elif args.command == "export-diagnostics":
        payload = export_policy_diagnostics(
            project_root=args.project_root,
            source_root=args.source_root,
            scope=args.scope,
        )
    elif args.command == "evaluate-test":
        payload = run_internal_test_evaluation(
            project_root=args.project_root, device=args.device
        )
    else:
        payload = {
            "legacy_v2_rescore": rescore_v2(project_root=args.project_root),
            "new_protocol": export_cohort(project_root=args.project_root),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
