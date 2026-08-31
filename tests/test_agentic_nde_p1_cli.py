from __future__ import annotations

import builtins
from types import SimpleNamespace

import scripts.run_agentic_nde as cli


def test_run_p1_cli_reports_formal_status(monkeypatch, capsys) -> None:
    observed: dict[str, object] = {}

    def fake_run(config, **kwargs):
        observed.update({"config": config, **kwargs})
        return SimpleNamespace(
            summary={"status": "P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO"}
        )

    monkeypatch.setattr(cli, "run_p1_visual_observability", fake_run)
    assert (
        cli.main(
            [
                "run-p1",
                "--config",
                "p1.yaml",
                "--research-root",
                "research",
                "--project-root",
                "project",
                "--feature-root",
                "features",
            ]
        )
        == 0
    )
    assert observed == {
        "config": "p1.yaml",
        "project_root": "project",
        "research_root": "research",
        "feature_root": "features",
        "device": "cuda:0",
        "notify": builtins.print,
    }
    assert capsys.readouterr().out.splitlines()[-1] == (
        "P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO"
    )


def test_replay_p1_cli_runs_full_scientific_replay(monkeypatch, capsys) -> None:
    observed: dict[str, object] = {}

    def fake_replay(path, **kwargs):
        observed.update({"path": path, **kwargs})
        return {"status": "P1_SPATIAL_VISUAL_OBSERVABILITY_GO"}

    monkeypatch.setattr(cli, "replay_p1_science", fake_replay)
    assert (
        cli.main(
            [
                "replay-p1",
                "--path",
                "package",
                "--project-root",
                "project",
                "--feature-root",
                "features",
                "--replay-output",
                "replay",
            ]
        )
        == 0
    )
    assert observed == {
        "path": "package",
        "project_root": "project",
        "feature_root": "features",
        "replay_output": "replay",
    }
    assert capsys.readouterr().out.strip() == "P1_SPATIAL_VISUAL_OBSERVABILITY_GO"
