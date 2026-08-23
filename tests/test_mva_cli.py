from __future__ import annotations

from cmc_bbdm.mva.cli import build_parser


def test_mva_cli_registers_low_budget_and_publication_commands() -> None:
    parser = build_parser()
    base = ["--config", "paper_v3/configs/mva_a0_a3.yaml"]

    assert (
        parser.parse_args(["low-domain", *base, "--outer-domain", "d"]).command
        == "low-domain"
    )
    assert (
        parser.parse_args(["stability-domain", *base, "--outer-domain", "d"]).command
        == "stability-domain"
    )
    assert parser.parse_args(["figures", *base]).command == "figures"
    assert parser.parse_args(["finalize", *base]).command == "finalize"
    assert parser.parse_args(["validate", *base]).command == "validate"
    replay = parser.parse_args(
        ["replay", *base, "--source", "formal", "--destination", "replay"]
    )
    assert replay.source == "formal"
    assert replay.destination == "replay"
