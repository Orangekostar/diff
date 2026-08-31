from __future__ import annotations

from pathlib import Path

import scripts.run_agentic_nde as cli


def test_audit_p0r_cli_runs_qc_and_reports_status(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    package = tmp_path / "package"
    calls: list[tuple[str, object]] = []

    def fake_audit_p0r(**kwargs):
        calls.append(("audit", kwargs))
        return package

    def fake_qc(**kwargs):
        calls.append(("qc", kwargs))
        return tmp_path / "qc"

    def fake_replay(path, **kwargs):
        calls.append(("replay", (path, kwargs)))
        return {"status": "P0R_AUTHOR_REGISTRATION_GO"}

    monkeypatch.setattr(cli, "audit_p0r", fake_audit_p0r)
    monkeypatch.setattr(cli, "render_p0r_qc", fake_qc)
    monkeypatch.setattr(cli, "replay_p0r_package", fake_replay)

    result = cli.main(
        [
            "audit-p0r",
            "--config",
            "config.yaml",
            "--surface-root",
            "surface",
            "--output",
            str(package),
            "--project-root",
            "project",
            "--qc-output",
            str(tmp_path / "qc"),
        ]
    )

    assert result == 0
    assert [name for name, _ in calls] == ["audit", "qc", "replay"]
    assert capsys.readouterr().out.strip() == "P0R_AUTHOR_REGISTRATION_GO"


def test_replay_p0r_cli_uses_source_aware_replay_when_root_is_supplied(
    monkeypatch, capsys
) -> None:
    observed: dict[str, object] = {}

    def fake_replay(path, **kwargs):
        observed.update({"path": path, **kwargs})
        return {"status": "P0R_AUTHOR_REGISTRATION_GO"}

    monkeypatch.setattr(cli, "replay_p0r_package", fake_replay)

    assert cli.main(["replay-p0r", "--path", "package", "--surface-root", "surface"]) == 0
    assert observed["path"] == "package"
    assert observed["surface_root"] == "surface"
    assert observed["project_root"] == cli._PROJECT_ROOT
    assert capsys.readouterr().out.strip() == "P0R_AUTHOR_REGISTRATION_GO"


def test_replay_p0r_cli_returns_one_for_integrity_error(tmp_path: Path) -> None:
    assert cli.main(["replay-p0r", "--path", str(tmp_path / "missing")]) == 1
