from __future__ import annotations

from cmc_bbdm.mavis.cli import build_parser


def test_mavis_cli_registers_formal_stage_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["p1-finalize"]).command == "p1-finalize"
    assert parser.parse_args(["p2-prepare"]).command == "p2-prepare"
    worker = parser.parse_args(
        ["p2-worker", "--outer-domain", "d0", "--device", "cuda:0"]
    )
    assert worker.outer_domain == "d0"
    assert worker.max_epochs == 80
    assert parser.parse_args(["p2-finalize"]).command == "p2-finalize"
    assert parser.parse_args(["p2-verify"]).command == "p2-verify"
    p3_worker = parser.parse_args(
        ["p3-worker", "--outer-domain", "d0", "--device", "cuda:0"]
    )
    assert p3_worker.outer_domain == "d0"
    assert p3_worker.max_epochs == 40
    assert parser.parse_args(["p3-finalize"]).command == "p3-finalize"
    assert parser.parse_args(["p3-verify"]).command == "p3-verify"
    p4_worker = parser.parse_args(
        ["p4-worker", "--outer-domain", "d0", "--device", "cuda:0"]
    )
    assert p4_worker.outer_domain == "d0"
    assert p4_worker.p3_package == "results/mavis/p3_dynamic_voi"
    assert parser.parse_args(["p4-finalize"]).command == "p4-finalize"
    assert parser.parse_args(["p4-verify"]).command == "p4-verify"
    p5_worker = parser.parse_args(
        ["p5-worker", "--outer-domain", "d0", "--device", "cuda:0"]
    )
    assert p5_worker.outer_domain == "d0"
    assert p5_worker.p3_package == "results/mavis/p3_dynamic_voi"
    assert parser.parse_args(["p5-finalize"]).command == "p5-finalize"
    assert parser.parse_args(["p5-verify"]).command == "p5-verify"
    p6_worker = parser.parse_args(
        ["p6-worker", "--outer-domain", "d0", "--device", "cuda:0"]
    )
    assert p6_worker.outer_domain == "d0"
    assert p6_worker.p2_package == "results/mavis/p2_mris"
    assert parser.parse_args(["p6-finalize"]).command == "p6-finalize"
    assert parser.parse_args(["p6-verify"]).command == "p6-verify"
    assert parser.parse_args(["p7-development-sha"]).command == "p7-development-sha"
    p7_worker = parser.parse_args(
        ["p7-worker", "--outer-domain", "d0", "--device", "cuda:0"]
    )
    assert p7_worker.outer_domain == "d0"
    assert p7_worker.p5_package == "results/mavis/p5_aggregation"
    assert parser.parse_args(["p7-finalize"]).command == "p7-finalize"
    assert parser.parse_args(["p7-verify"]).command == "p7-verify"
    assert parser.parse_args(["p7-replay"]).command == "p7-replay"
    assert parser.parse_args(["p7-replay-verify"]).command == "p7-replay-verify"
