"""Command-line entry points for staged MAVIS execution."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from .aggregation_execution import run_aggregation_outer_domain
from .aggregation_package import (
    finalize_aggregation_package,
    verify_aggregation_package,
)
from .authority import load_mavis_authority
from .closed_loop_execution import run_closed_loop_outer_domain
from .closed_loop_package import (
    finalize_closed_loop_package,
    verify_closed_loop_package,
)
from .config import load_mavis_config
from .dynamic_execution import run_dynamic_outer_domain
from .dynamic_package import finalize_dynamic_package, verify_dynamic_package
from .final_execution import run_final_outer_domain
from .final_package import (
    development_package_sha256,
    finalize_final_package,
    verify_final_package,
)
from .historical_sources import HistoricalPolicySource
from .mris_data import (
    build_mris_feature_bank,
    load_mris_feature_bank,
    save_mris_feature_bank,
)
from .mris_execution import run_mris_outer_domain
from .mris_package import finalize_mris_package, verify_mris_package
from .replay import run_final_replay, verify_replay_package
from .safety_execution import run_safety_outer_domain
from .safety_package import finalize_safety_package, verify_safety_package
from .state_bank_execution import run_state_bank_worker_group
from .state_bank_package import (
    finalize_state_bank_package,
    load_state_action_pairs_package,
    load_state_manifest_package,
    verify_state_bank_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="paper_v3/configs/mavis_development.yaml",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--source-project-root",
        default="/home/ww/paper3/cmc_damage_inference",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("p1-worker")
    worker.add_argument("--domains", nargs="+", required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--max-specimens-per-domain", type=int)
    commands.add_parser("p1-finalize")
    verify = commands.add_parser("p1-verify")
    verify.add_argument("--package", default="results/mavis/p1_state_bank")
    prepare_p2 = commands.add_parser("p2-prepare")
    prepare_p2.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    prepare_p2.add_argument(
        "--feature-bank",
        default="results/mavis/.work/p2_mris/feature_bank.npz",
    )
    prepare_p2.add_argument("--shuffle-seed", type=int, default=20260821)
    worker_p2 = commands.add_parser("p2-worker")
    worker_p2.add_argument("--outer-domain", required=True)
    worker_p2.add_argument("--device", required=True)
    worker_p2.add_argument(
        "--feature-bank",
        default="results/mavis/.work/p2_mris/feature_bank.npz",
    )
    worker_p2.add_argument(
        "--worker-root",
        default="results/mavis/.work/p2_mris/workers",
    )
    worker_p2.add_argument("--max-epochs", type=int, default=80)
    worker_p2.add_argument("--patience", type=int, default=10)
    worker_p2.add_argument("--batch-size", type=int, default=256)
    finalize_p2 = commands.add_parser("p2-finalize")
    finalize_p2.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    finalize_p2.add_argument(
        "--feature-bank",
        default="results/mavis/.work/p2_mris/feature_bank.npz",
    )
    finalize_p2.add_argument(
        "--worker-root",
        default="results/mavis/.work/p2_mris/workers",
    )
    finalize_p2.add_argument("--bootstrap-replicates", type=int, default=5000)
    verify_p2 = commands.add_parser("p2-verify")
    verify_p2.add_argument("--package", default="results/mavis/p2_mris")
    worker_p3 = commands.add_parser("p3-worker")
    worker_p3.add_argument("--outer-domain", required=True)
    worker_p3.add_argument("--device", required=True)
    worker_p3.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    worker_p3.add_argument("--p2-package", default="results/mavis/p2_mris")
    worker_p3.add_argument(
        "--feature-bank",
        default="results/mavis/.work/p2_mris/feature_bank.npz",
    )
    worker_p3.add_argument(
        "--worker-root",
        default="results/mavis/.work/p3_dynamic_voi/workers",
    )
    worker_p3.add_argument("--max-epochs", type=int, default=40)
    worker_p3.add_argument("--patience", type=int, default=5)
    worker_p3.add_argument("--batch-size", type=int, default=64)
    worker_p3.add_argument("--recall-k", type=int, default=5)
    finalize_p3 = commands.add_parser("p3-finalize")
    finalize_p3.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    finalize_p3.add_argument("--p2-package", default="results/mavis/p2_mris")
    finalize_p3.add_argument(
        "--feature-bank",
        default="results/mavis/.work/p2_mris/feature_bank.npz",
    )
    finalize_p3.add_argument(
        "--worker-root",
        default="results/mavis/.work/p3_dynamic_voi/workers",
    )
    finalize_p3.add_argument("--bootstrap-replicates", type=int, default=5000)
    verify_p3 = commands.add_parser("p3-verify")
    verify_p3.add_argument("--package", default="results/mavis/p3_dynamic_voi")
    worker_p4 = commands.add_parser("p4-worker")
    worker_p4.add_argument("--outer-domain", required=True)
    worker_p4.add_argument("--device", required=True)
    worker_p4.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    worker_p4.add_argument("--p2-package", default="results/mavis/p2_mris")
    worker_p4.add_argument("--p3-package", default="results/mavis/p3_dynamic_voi")
    worker_p4.add_argument(
        "--worker-root",
        default="results/mavis/.work/p4_closed_loop/workers",
    )
    finalize_p4 = commands.add_parser("p4-finalize")
    finalize_p4.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    finalize_p4.add_argument("--p2-package", default="results/mavis/p2_mris")
    finalize_p4.add_argument("--p3-package", default="results/mavis/p3_dynamic_voi")
    finalize_p4.add_argument(
        "--worker-root",
        default="results/mavis/.work/p4_closed_loop/workers",
    )
    finalize_p4.add_argument("--bootstrap-replicates", type=int, default=5000)
    verify_p4 = commands.add_parser("p4-verify")
    verify_p4.add_argument("--package", default="results/mavis/p4_closed_loop")
    worker_p5 = commands.add_parser("p5-worker")
    worker_p5.add_argument("--outer-domain", required=True)
    worker_p5.add_argument("--device", required=True)
    worker_p5.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    worker_p5.add_argument("--p2-package", default="results/mavis/p2_mris")
    worker_p5.add_argument("--p3-package", default="results/mavis/p3_dynamic_voi")
    worker_p5.add_argument(
        "--feature-bank",
        default="results/mavis/.work/p2_mris/feature_bank.npz",
    )
    worker_p5.add_argument(
        "--worker-root",
        default="results/mavis/.work/p5_aggregation/workers",
    )
    worker_p5.add_argument("--batch-size", type=int, default=64)
    finalize_p5 = commands.add_parser("p5-finalize")
    finalize_p5.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    finalize_p5.add_argument("--p2-package", default="results/mavis/p2_mris")
    finalize_p5.add_argument("--p3-package", default="results/mavis/p3_dynamic_voi")
    finalize_p5.add_argument(
        "--feature-bank",
        default="results/mavis/.work/p2_mris/feature_bank.npz",
    )
    finalize_p5.add_argument(
        "--worker-root",
        default="results/mavis/.work/p5_aggregation/workers",
    )
    verify_p5 = commands.add_parser("p5-verify")
    verify_p5.add_argument("--package", default="results/mavis/p5_aggregation")
    worker_p6 = commands.add_parser("p6-worker")
    worker_p6.add_argument("--outer-domain", required=True)
    worker_p6.add_argument("--device", required=True)
    worker_p6.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    worker_p6.add_argument("--p2-package", default="results/mavis/p2_mris")
    worker_p6.add_argument("--p3-package", default="results/mavis/p3_dynamic_voi")
    worker_p6.add_argument(
        "--worker-root",
        default="results/mavis/.work/p6_safety/workers",
    )
    finalize_p6 = commands.add_parser("p6-finalize")
    finalize_p6.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    finalize_p6.add_argument("--p2-package", default="results/mavis/p2_mris")
    finalize_p6.add_argument("--p3-package", default="results/mavis/p3_dynamic_voi")
    finalize_p6.add_argument(
        "--worker-root",
        default="results/mavis/.work/p6_safety/workers",
    )
    verify_p6 = commands.add_parser("p6-verify")
    verify_p6.add_argument("--package", default="results/mavis/p6_safety")
    development_p7 = commands.add_parser("p7-development-sha")
    development_p7.add_argument(
        "--p4-package", default="results/mavis/p4_closed_loop"
    )
    development_p7.add_argument(
        "--p5-package", default="results/mavis/p5_aggregation"
    )
    development_p7.add_argument("--p6-package", default="results/mavis/p6_safety")
    worker_p7 = commands.add_parser("p7-worker")
    worker_p7.add_argument("--outer-domain", required=True)
    worker_p7.add_argument("--device", required=True)
    worker_p7.add_argument("--p2-package", default="results/mavis/p2_mris")
    worker_p7.add_argument("--p4-package", default="results/mavis/p4_closed_loop")
    worker_p7.add_argument("--p5-package", default="results/mavis/p5_aggregation")
    worker_p7.add_argument("--p6-package", default="results/mavis/p6_safety")
    worker_p7.add_argument(
        "--worker-root",
        default="results/mavis/.work/p7_final_frozen_eval/workers",
    )
    finalize_p7 = commands.add_parser("p7-finalize")
    finalize_p7.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    finalize_p7.add_argument("--p4-package", default="results/mavis/p4_closed_loop")
    finalize_p7.add_argument("--p5-package", default="results/mavis/p5_aggregation")
    finalize_p7.add_argument("--p6-package", default="results/mavis/p6_safety")
    finalize_p7.add_argument(
        "--worker-root",
        default="results/mavis/.work/p7_final_frozen_eval/workers",
    )
    finalize_p7.add_argument("--bootstrap-replicates", type=int, default=5000)
    verify_p7 = commands.add_parser("p7-verify")
    verify_p7.add_argument(
        "--package", default="results/mavis/p7_final_frozen_eval"
    )
    replay_p7 = commands.add_parser("p7-replay")
    replay_p7.add_argument(
        "--formal-package", default="results/mavis/p7_final_frozen_eval"
    )
    replay_p7.add_argument("--replay-root", default="results/mavis/replay")
    replay_p7.add_argument("--p1-package", default="results/mavis/p1_state_bank")
    replay_p7.add_argument("--p4-package", default="results/mavis/p4_closed_loop")
    replay_p7.add_argument("--p5-package", default="results/mavis/p5_aggregation")
    replay_p7.add_argument("--p6-package", default="results/mavis/p6_safety")
    replay_p7.add_argument(
        "--worker-root",
        default="results/mavis/.work/p7_final_frozen_eval/workers",
    )
    replay_p7.add_argument("--bootstrap-replicates", type=int, default=5000)
    verify_replay_p7 = commands.add_parser("p7-replay-verify")
    verify_replay_p7.add_argument("--package", default="results/mavis/replay")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = Path(arguments.project_root).resolve(strict=True)
    config = (root / arguments.config).resolve(strict=True)
    if arguments.command == "p1-worker":
        outputs = run_state_bank_worker_group(
            config,
            project_root=root,
            source_project_root=arguments.source_project_root,
            domain_ids=tuple(arguments.domains),
            device=arguments.device,
            max_specimens_per_domain=arguments.max_specimens_per_domain,
        )
        print(json.dumps({"outputs": [str(path) for path in outputs]}), flush=True)
    elif arguments.command == "p1-finalize":
        output = finalize_state_bank_package(
            config,
            project_root=root,
            source_project_root=arguments.source_project_root,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p1-verify":
        package = (root / arguments.package).resolve(strict=True)
        manifest = verify_state_bank_package(package)
        print(
            json.dumps(
                {
                    "output": str(package),
                    "state_bank_state_sha256": manifest[
                        "state_bank_state_sha256"
                    ],
                }
            ),
            flush=True,
        )
    elif arguments.command == "p2-prepare":
        package = (root / arguments.p1_package).resolve(strict=True)
        verify_state_bank_package(package)
        config_value = load_mavis_config(config, project_root=root)
        authority = load_mavis_authority(
            config_value,
            source_project_root=arguments.source_project_root,
        )
        bank = build_mris_feature_bank(
            load_state_manifest_package(package),
            authority=authority,
            domain_order=config_value.domain_order,
            shuffle_seed=arguments.shuffle_seed,
        )
        destination = root / arguments.feature_bank
        checksum = destination.with_suffix(destination.suffix + ".sha256")
        if destination.exists() or checksum.exists():
            raise RuntimeError("P2 feature bank already exists")
        save_mris_feature_bank(bank, destination)
        print(
            json.dumps(
                {
                    "output": str(destination),
                    "row_count": bank.row_count,
                    "input_state_sha256": bank.input_state_sha256,
                    "target_state_sha256": bank.target_state_sha256,
                }
            ),
            flush=True,
        )
    elif arguments.command == "p2-worker":
        config_value = load_mavis_config(config, project_root=root)
        bank = load_mris_feature_bank(root / arguments.feature_bank)
        output = run_mris_outer_domain(
            bank,
            outer_domain=arguments.outer_domain,
            output_root=root / arguments.worker_root,
            hidden_dimension=config_value.mris_hidden_size,
            mris_dimension=config_value.mris_dimension,
            learning_rate=config_value.learning_rate,
            max_epochs=arguments.max_epochs,
            patience=arguments.patience,
            batch_size=arguments.batch_size,
            seed=config_value.seed,
            device=arguments.device,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p2-finalize":
        output = finalize_mris_package(
            config,
            project_root=root,
            feature_bank_path=root / arguments.feature_bank,
            worker_root=root / arguments.worker_root,
            p1_package=root / arguments.p1_package,
            bootstrap_replicates=arguments.bootstrap_replicates,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p2-verify":
        package = (root / arguments.package).resolve(strict=True)
        manifest = verify_mris_package(package)
        print(
            json.dumps(
                {
                    "output": str(package),
                    "p2_state_sha256": manifest["p2_state_sha256"],
                }
            ),
            flush=True,
        )
    elif arguments.command == "p3-worker":
        p1_package = (root / arguments.p1_package).resolve(strict=True)
        p2_package = (root / arguments.p2_package).resolve(strict=True)
        verify_state_bank_package(p1_package)
        verify_mris_package(p2_package)
        config_value = load_mavis_config(config, project_root=root)
        output = run_dynamic_outer_domain(
            load_mris_feature_bank(root / arguments.feature_bank),
            states=load_state_manifest_package(p1_package),
            actions=load_state_action_pairs_package(p1_package),
            outer_domain=arguments.outer_domain,
            p2_checkpoint_root=p2_package / "checkpoints",
            output_root=root / arguments.worker_root,
            hidden_dimension=config_value.mris_hidden_size,
            learning_rate=config_value.learning_rate,
            max_epochs=arguments.max_epochs,
            patience=arguments.patience,
            batch_size=arguments.batch_size,
            seed=config_value.seed,
            device=arguments.device,
            loss_weights=dict(config_value.loss_weights),
            recall_k=arguments.recall_k,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p3-finalize":
        output = finalize_dynamic_package(
            config,
            project_root=root,
            feature_bank_path=root / arguments.feature_bank,
            worker_root=root / arguments.worker_root,
            p1_package=root / arguments.p1_package,
            p2_package=root / arguments.p2_package,
            bootstrap_replicates=arguments.bootstrap_replicates,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p3-verify":
        package = (root / arguments.package).resolve(strict=True)
        manifest = verify_dynamic_package(package)
        print(
            json.dumps(
                {
                    "output": str(package),
                    "p3_state_sha256": manifest["p3_state_sha256"],
                }
            ),
            flush=True,
        )
    elif arguments.command == "p4-worker":
        config_value = load_mavis_config(config, project_root=root)
        p1_package = (root / arguments.p1_package).resolve(strict=True)
        p2_package = (root / arguments.p2_package).resolve(strict=True)
        p3_package = (root / arguments.p3_package).resolve(strict=True)
        verify_state_bank_package(p1_package)
        verify_mris_package(p2_package)
        verify_dynamic_package(p3_package)
        authority = load_mavis_authority(
            config_value,
            source_project_root=arguments.source_project_root,
        )
        sources = config_value.sources
        historical = HistoricalPolicySource(
            a4_path=root / sources["a4_fixed_trajectories"].path,
            a4_sha256=sources["a4_fixed_trajectories"].sha256,
            a5_path=root / sources["a5_target_trajectories"].path,
            a5_sha256=sources["a5_target_trajectories"].sha256,
            mvd_m1_path=root / sources["mvd_m1_predictions"].path,
            mvd_m1_sha256=sources["mvd_m1_predictions"].sha256,
            checkpoints=config_value.checkpoints,
        )
        output = run_closed_loop_outer_domain(
            authority,
            config_value,
            outer_domain=arguments.outer_domain,
            p1_states=load_state_manifest_package(p1_package),
            donor_mapping=pl.read_csv(p2_package / "donor_mapping.csv"),
            historical_source=historical,
            p2_checkpoint_root=p2_package / "checkpoints",
            p3_checkpoint_root=p3_package / "checkpoints",
            output_root=root / arguments.worker_root,
            device=arguments.device,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p4-finalize":
        output = finalize_closed_loop_package(
            config,
            project_root=root,
            worker_root=root / arguments.worker_root,
            p1_package=root / arguments.p1_package,
            p2_package=root / arguments.p2_package,
            p3_package=root / arguments.p3_package,
            bootstrap_replicates=arguments.bootstrap_replicates,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p4-verify":
        package = (root / arguments.package).resolve(strict=True)
        manifest = verify_closed_loop_package(package)
        print(
            json.dumps(
                {
                    "output": str(package),
                    "p4_state_sha256": manifest["p4_state_sha256"],
                }
            ),
            flush=True,
        )
    elif arguments.command == "p5-worker":
        config_value = load_mavis_config(config, project_root=root)
        p1_package = (root / arguments.p1_package).resolve(strict=True)
        p2_package = (root / arguments.p2_package).resolve(strict=True)
        p3_package = (root / arguments.p3_package).resolve(strict=True)
        verify_state_bank_package(p1_package)
        verify_mris_package(p2_package)
        verify_dynamic_package(p3_package)
        output = run_aggregation_outer_domain(
            load_mavis_authority(
                config_value,
                source_project_root=arguments.source_project_root,
            ),
            config_value,
            load_mris_feature_bank(root / arguments.feature_bank),
            states=load_state_manifest_package(p1_package),
            actions=load_state_action_pairs_package(p1_package),
            outer_domain=arguments.outer_domain,
            p2_checkpoint_root=p2_package / "checkpoints",
            p3_checkpoint_root=p3_package / "checkpoints",
            output_root=root / arguments.worker_root,
            project_root=root,
            encoder_project_root=arguments.source_project_root,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p5-finalize":
        output = finalize_aggregation_package(
            config,
            project_root=root,
            feature_bank_path=root / arguments.feature_bank,
            worker_root=root / arguments.worker_root,
            p1_package=root / arguments.p1_package,
            p2_package=root / arguments.p2_package,
            p3_package=root / arguments.p3_package,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p5-verify":
        package = (root / arguments.package).resolve(strict=True)
        manifest = verify_aggregation_package(package)
        print(
            json.dumps(
                {
                    "output": str(package),
                    "p5_state_sha256": manifest["p5_state_sha256"],
                }
            ),
            flush=True,
        )
    elif arguments.command == "p6-worker":
        config_value = load_mavis_config(config, project_root=root)
        p1_package = (root / arguments.p1_package).resolve(strict=True)
        p2_package = (root / arguments.p2_package).resolve(strict=True)
        p3_package = (root / arguments.p3_package).resolve(strict=True)
        verify_state_bank_package(p1_package)
        verify_mris_package(p2_package)
        verify_dynamic_package(p3_package)
        output = run_safety_outer_domain(
            load_mavis_authority(
                config_value,
                source_project_root=arguments.source_project_root,
            ),
            config_value,
            outer_domain=arguments.outer_domain,
            p1_states=load_state_manifest_package(p1_package),
            p2_checkpoint_root=p2_package / "checkpoints",
            p3_checkpoint_root=p3_package / "checkpoints",
            output_root=root / arguments.worker_root,
            device=arguments.device,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p6-finalize":
        output = finalize_safety_package(
            config,
            project_root=root,
            worker_root=root / arguments.worker_root,
            p1_package=root / arguments.p1_package,
            p2_package=root / arguments.p2_package,
            p3_package=root / arguments.p3_package,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p6-verify":
        package = (root / arguments.package).resolve(strict=True)
        manifest = verify_safety_package(package)
        print(
            json.dumps(
                {
                    "output": str(package),
                    "p6_state_sha256": manifest["p6_state_sha256"],
                }
            ),
            flush=True,
        )
    elif arguments.command == "p7-development-sha":
        p4_manifest = verify_closed_loop_package(root / arguments.p4_package)
        p5_manifest = verify_aggregation_package(root / arguments.p5_package)
        p6_manifest = verify_safety_package(root / arguments.p6_package)
        print(
            json.dumps(
                {
                    "development_package_sha256": development_package_sha256(
                        p4_manifest,
                        p5_manifest,
                        p6_manifest,
                    )
                }
            ),
            flush=True,
        )
    elif arguments.command == "p7-worker":
        config_value = load_mavis_config(config, project_root=root)
        config_value.require_finalized()
        p2_package = (root / arguments.p2_package).resolve(strict=True)
        p4_package = (root / arguments.p4_package).resolve(strict=True)
        p5_package = (root / arguments.p5_package).resolve(strict=True)
        p6_package = (root / arguments.p6_package).resolve(strict=True)
        verify_mris_package(p2_package)
        p4_manifest = verify_closed_loop_package(p4_package)
        p5_manifest = verify_aggregation_package(p5_package)
        p6_manifest = verify_safety_package(p6_package)
        if config_value.development_package_sha256 != development_package_sha256(
            p4_manifest,
            p5_manifest,
            p6_manifest,
        ):
            raise RuntimeError("frozen MAVIS development package changed")
        output = run_final_outer_domain(
            load_mavis_authority(
                config_value,
                source_project_root=arguments.source_project_root,
            ),
            config_value,
            outer_domain=arguments.outer_domain,
            p2_checkpoint_root=p2_package / "checkpoints",
            p5_checkpoint_root=p5_package / "checkpoints",
            selections=pl.read_csv(p6_package / "selections.csv"),
            output_root=root / arguments.worker_root,
            device=arguments.device,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p7-finalize":
        output = finalize_final_package(
            config,
            project_root=root,
            worker_root=root / arguments.worker_root,
            p1_package=root / arguments.p1_package,
            p4_package=root / arguments.p4_package,
            p5_package=root / arguments.p5_package,
            p6_package=root / arguments.p6_package,
            bootstrap_replicates=arguments.bootstrap_replicates,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    elif arguments.command == "p7-verify":
        package = (root / arguments.package).resolve(strict=True)
        manifest = verify_final_package(package)
        print(
            json.dumps(
                {
                    "output": str(package),
                    "p7_state_sha256": manifest["p7_state_sha256"],
                    "claim_tier": manifest["claim_tier"],
                }
            ),
            flush=True,
        )
    elif arguments.command == "p7-replay":
        output = run_final_replay(
            config,
            project_root=root,
            formal_package=root / arguments.formal_package,
            replay_root=root / arguments.replay_root,
            worker_root=root / arguments.worker_root,
            p1_package=root / arguments.p1_package,
            p4_package=root / arguments.p4_package,
            p5_package=root / arguments.p5_package,
            p6_package=root / arguments.p6_package,
            bootstrap_replicates=arguments.bootstrap_replicates,
        )
        print(json.dumps({"output": str(output)}), flush=True)
    else:
        package = (root / arguments.package).resolve(strict=True)
        manifest = verify_replay_package(package, project_root=root)
        print(
            json.dumps(
                {
                    "output": str(package),
                    "tree_state_sha256": manifest["tree_state_sha256"],
                    "byte_identical": manifest["byte_identical"],
                }
            ),
            flush=True,
        )
    return 0


__all__ = ["build_parser", "main"]
