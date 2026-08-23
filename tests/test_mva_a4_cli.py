from __future__ import annotations

import json
from pathlib import Path

from cmc_bbdm.mva import a4_cli

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mva_a4_global_mask.yaml"


def test_a4_cli_dispatches_outer_domain(monkeypatch, capsys) -> None:
    observed = {}

    def fake_worker(config_path, *, project_root, outer_domain, device):
        observed.update(
            config_path=Path(config_path),
            project_root=Path(project_root),
            outer_domain=outer_domain,
            device=device,
        )
        return ROOT / "results/mva/.work/a4_domains" / outer_domain

    monkeypatch.setattr(a4_cli, "run_a4_outer_worker", fake_worker)

    code = a4_cli.main(
        [
            "--project-root",
            str(ROOT),
            "--config",
            str(CONFIG),
            "domain",
            "--outer-domain",
            "d0",
            "--device",
            "cuda:2",
        ]
    )

    assert code == 0
    assert observed["outer_domain"] == "d0"
    assert observed["device"] == "cuda:2"
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "domain"
    assert payload["output"].endswith("d0")


def test_a4_cli_render_writes_figures_then_report(monkeypatch, capsys) -> None:
    calls = []

    def fake_figures(evidence, *, config_path, project_root):
        calls.append(("figures", Path(evidence)))
        return Path(evidence) / "figures"

    def fake_report(evidence, *, config_path, project_root):
        calls.append(("report", Path(evidence)))
        return Path(evidence) / "REPORT.md"

    monkeypatch.setattr(a4_cli, "render_a4_figures", fake_figures)
    monkeypatch.setattr(a4_cli, "render_a4_report", fake_report)

    code = a4_cli.main(
        [
            "--project-root",
            str(ROOT),
            "--config",
            str(CONFIG),
            "render",
        ]
    )

    assert code == 0
    assert [name for name, _ in calls] == ["figures", "report"]
    assert calls[0][1] == ROOT / "results/mva/.work/a4_aggregate"
    assert json.loads(capsys.readouterr().out)["command"] == "render"


def test_a4_runner_freezes_blas_before_cli_import() -> None:
    text = (ROOT / "scripts/run_mva_a4.py").read_text(encoding="utf-8")

    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        assert f'"{variable}"' in text
    assert text.index("os.environ[variable]") < text.index(
        "from cmc_bbdm.mva.a4_cli import main"
    )
