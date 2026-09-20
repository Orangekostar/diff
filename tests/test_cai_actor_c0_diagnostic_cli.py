from scripts.cai_actor_c0_diagnostic.cli import build_parser


def test_cli_exposes_exact_stage_contract():
    parser = build_parser()
    config = "docs/cai/actor_c0_diagnostic/ACTOR_C0_DIAGNOSTIC_SCOPE.json"
    for command in (
        "prepare",
        "replay",
        "diagnose",
        "render",
        "verify",
        "publish",
        "status",
        "all",
    ):
        args = parser.parse_args([command, "--config", config])
        assert args.command == command
        assert str(args.config) == config

