from __future__ import annotations

import csv
import importlib.util
import json
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cmc_bbdm.cpb_diffusion_marginalization.config import DOMAIN_ORDER
from cmc_bbdm.cpb_diffusion_marginalization.tracking import TRIAL_INDEX_FIELDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_d8_exploration.py"
SHELL = PROJECT_ROOT / "scripts/run_d8_exploration.sh"
CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_exploration.yaml"


def _module() -> object:
    spec = importlib.util.spec_from_file_location("run_d8_exploration", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load D8 CLI module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_exposes_only_registered_pre_outer_commands(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    calls: list[str] = []

    def fake_run(command: str) -> dict[str, object]:
        calls.append(command)
        return {"command": command, "status": "PASS"}

    monkeypatch.setattr(module, "run_registered_command", fake_run)
    for command in ("baseline", "residual-bank", "pilot", "validate"):
        assert module.main([command]) == 0
        assert json.loads(capsys.readouterr().out) == {
            "command": command,
            "status": "PASS",
        }
    assert calls == ["baseline", "residual-bank", "pilot", "validate"]
    for command in ("outer", "outer-evaluation", "formal", "final"):
        with pytest.raises(SystemExit):
            module.main([command])


def test_validate_command_uses_exact_registered_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[tuple[Path, dict[str, Path]]] = []
    result = type(
        "Result",
        (),
        {
            "outer_domains": ("a",),
            "trial_count": 432,
            "outer_evaluation_count": 0,
            "escalation_status": "FREEZE_PILOT_FOR_OUTER_EVALUATION",
            "scientific_digest": "a" * 64,
            "output_tree_sha256": "b" * 64,
        },
    )()

    def fake_validate(output: Path, **kwargs: Path) -> object:
        calls.append((output, kwargs))
        return result

    monkeypatch.setattr(module, "validate_d8_search_package", fake_validate)
    observed = module.execute_validate()
    assert observed["command"] == "validate"
    assert observed["outer_evaluation_count"] == 0
    assert calls == [
        (
            PROJECT_ROOT / "results/d8_search",
            {"project_root": PROJECT_ROOT, "config_path": CONFIG},
        )
    ]


def test_baseline_command_accepts_only_registered_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    config = SimpleNamespace(baseline_mae=0.08963580465761432)
    data = object()
    result = SimpleNamespace(
        specimen_count=276,
        equal_domain_mae=0.08963580465761434,
        maximum_prediction_error=0.0,
        maximum_target_error=0.0,
        state_sha256="1" * 64,
    )
    monkeypatch.setattr(module, "_load_authorities", lambda: (config, data))
    monkeypatch.setattr(
        module,
        "reproduce_internal_only_baseline",
        lambda *args, **kwargs: result,
    )

    observed = module.execute_baseline()

    assert observed["status"] == "PASS"
    assert observed["equal_domain_mae"] == result.equal_domain_mae


def test_shell_wrapper_freezes_runtime_config_and_command_boundary() -> None:
    assert SHELL.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(SHELL)], check=True)
    text = SHELL.read_text(encoding="utf-8")
    for variable in (
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUBLAS_WORKSPACE_CONFIG",
    ):
        assert variable in text
    assert "paper_v3/configs/d8_exploration.yaml" in text
    assert "run_d8_exploration.py" in text
    assert "outer-evaluation" not in text


def test_registered_config_cannot_be_overridden() -> None:
    module = _module()
    assert module.REGISTERED_CONFIG == CONFIG
    with pytest.raises(SystemExit):
        module.main(["validate", "--config", "/tmp/other.yaml"])


def test_pilot_command_builds_and_publishes_registered_source_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    config_sha256 = "a" * 64
    residual_sha256 = "b" * 64
    config = SimpleNamespace(config_sha256=config_sha256, p6_draws=8)
    data = object()
    bank = SimpleNamespace(
        specimen_count=276,
        draw_count=8,
        records=tuple(range(2208)),
        maximum_mean_error=0.0,
        maximum_variance_error=0.0,
        state_sha256=residual_sha256,
    )
    outer_runs = []
    decision_studies = []

    def fake_run(*args: object, **kwargs: object) -> object:
        assert args == (data,)
        assert kwargs["config"] is config
        assert kwargs["bank"] is bank
        assert kwargs["project_root"] == tmp_path
        assert kwargs["device"] == "cuda:0"
        root = Path(kwargs["output"])
        selections = root / "best_inner_configs"
        selections.mkdir(parents=True)
        rows = []
        for outer_index, outer in enumerate(DOMAIN_ORDER):
            selection_state = f"{outer_index + 1:064x}"
            selection = {
                "outer_domain": outer,
                "state_sha256": selection_state,
            }
            (selections / f"{outer}.json").write_text(
                json.dumps(selection), encoding="ascii"
            )
            study_state = f"{outer_index + 11:064x}"
            decision_studies.append(
                SimpleNamespace(outer_domain=outer, state_sha256=study_state)
            )
            outer_runs.append(
                SimpleNamespace(
                    outer_domain=outer,
                    search=SimpleNamespace(
                        initial_trial_count=72,
                        trial_count=2,
                        completed_count=2,
                        pruned_count=0,
                        failed_count=0,
                    ),
                    selection=SimpleNamespace(state_sha256=selection_state),
                )
            )
            for trial_id, control in enumerate(("B0", "B5")):
                row = {field: "" for field in TRIAL_INDEX_FIELDS}
                row.update(
                    {
                        "study_name": f"d8::{outer}",
                        "trial_id": trial_id,
                        "outer_fold": outer,
                        "state": "COMPLETE",
                        "objective": 0.10 - trial_id * 0.01,
                        "control_id": control,
                        "candidate_sha256": f"{outer_index * 2 + trial_id + 21:064x}",
                    }
                )
                rows.append(row)
        with (root / "trial_index.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TRIAL_INDEX_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        (root / "study.db").write_bytes(b"registered-study")
        return SimpleNamespace(
            config_sha256=config_sha256,
            residual_bank_sha256=residual_sha256,
            outer_runs=tuple(outer_runs),
            outer_evaluation_count=0,
        )

    decision = SimpleNamespace(
        decision="FREEZE_PILOT_FOR_OUTER_EVALUATION",
        studies=decision_studies,
        to_payload=lambda: {
            "scope": "d8_pilot_escalation_evidence",
            "decision": "FREEZE_PILOT_FOR_OUTER_EVALUATION",
        },
    )

    def fake_build_evidence(rows: object, **kwargs: object) -> object:
        assert len(rows) == 12
        assert kwargs == {
            "selections": tuple(
                {
                    "outer_domain": outer,
                    "state_sha256": f"{index + 1:064x}",
                }
                for index, outer in enumerate(DOMAIN_ORDER)
            ),
            "bank": bank,
            "config": config,
        }
        return decision

    published: dict[str, object] = {}

    def fake_publish(source: Path, output: Path, **kwargs: object) -> object:
        published["files"] = {path.name for path in source.iterdir()}
        published["selection"] = json.loads(
            (source / "selected_configs.json").read_text(encoding="ascii")
        )
        published["summary"] = tuple(
            csv.DictReader(
                (source / "search_summary.csv").read_text(encoding="utf-8").splitlines()
            )
        )
        published["report"] = (source / "pilot_report.md").read_text(
            encoding="ascii"
        )
        assert output == tmp_path / "results/d8_search"
        assert kwargs == {
            "project_root": tmp_path,
            "config_path": tmp_path / "paper_v3/configs/d8_exploration.yaml",
            "escalation_status": "FREEZE_PILOT_FOR_OUTER_EVALUATION",
        }
        return SimpleNamespace(
            outer_domains=DOMAIN_ORDER,
            trial_count=12,
            outer_evaluation_count=0,
            escalation_status="FREEZE_PILOT_FOR_OUTER_EVALUATION",
            scientific_digest="c" * 64,
            output_tree_sha256="d" * 64,
        )

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "REGISTERED_CONFIG",
        tmp_path / "paper_v3/configs/d8_exploration.yaml",
    )
    monkeypatch.setattr(module, "REGISTERED_OUTPUT", tmp_path / "results/d8_search")
    monkeypatch.setattr(module, "_load_authorities", lambda: (config, data))
    monkeypatch.setattr(module, "build_cross_fitted_p6_residual_bank", lambda *a, **k: bank)
    monkeypatch.setattr(module, "run_registered_pilot", fake_run, raising=False)
    monkeypatch.setattr(
        module, "build_pilot_escalation_evidence", fake_build_evidence, raising=False
    )
    monkeypatch.setattr(module, "publish_d8_search_package", fake_publish, raising=False)

    observed = module.execute_pilot()

    assert observed == {
        "command": "pilot",
        "status": "PASS",
        "outer_domains": list(DOMAIN_ORDER),
        "trial_count": 12,
        "outer_evaluation_count": 0,
        "escalation_status": "FREEZE_PILOT_FOR_OUTER_EVALUATION",
        "scientific_digest": "c" * 64,
        "output_tree_sha256": "d" * 64,
    }
    assert published["files"] == {
        "trial_index.csv",
        "study.db",
        "residual_bank_manifest.json",
        "search_summary.csv",
        "selected_configs.json",
        "escalation_evidence.json",
        "pilot_report.md",
    }
    selection = published["selection"]
    assert selection["outer_evaluation_count"] == 0
    assert [item["outer_domain"] for item in selection["selections"]] == list(
        DOMAIN_ORDER
    )
    assert len(published["summary"]) == 6
    assert "`FREEZE_PILOT_FOR_OUTER_EVALUATION`" in published["report"]
    assert "Outer evaluations: `0`" in published["report"]
