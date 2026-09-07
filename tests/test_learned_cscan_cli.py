from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REQUIRED_RESULTS = {
    "config.yaml",
    "inventory.json",
    "split_manifest.csv",
    "reference_manifest.csv",
    "perception_manifest.json",
    "surface_percepts.jsonl",
    "readout_validation.csv",
    "training_bank_manifest.json",
    "training_log.csv",
    "validation_selection.json",
    "model_manifest.json",
    "models",
    "test_evaluation_manifest.json",
    "per_episode_metrics.csv",
    "trajectories.parquet",
    "representative_replay.parquet",
    "replay_validation.json",
    "comparisons.csv",
    "failure_cases.csv",
    "summary.json",
    "CHECKSUMS.sha256",
}


def test_cli_runs_recoverable_stages_and_requires_explicit_test_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts/run_learned_cscan.py"
    specification = importlib.util.spec_from_file_location(
        "run_learned_cscan_test", script_path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)

    config = tmp_path / "config.yaml"
    config.write_text("stage: synthetic-test\n", encoding="utf-8")
    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "results"
    calls: list[tuple[str, str | None]] = []

    def stage(name: str):
        def invoke(**kwargs):
            assert kwargs["config_path"] == config
            assert kwargs["source_root"] == source_root
            assert kwargs["project_root"] == project_root
            calls.append((name, kwargs.get("split")))
            return {"stage": name}

        return invoke

    monkeypatch.setattr(module, "prepare_study", stage("prepare"))
    monkeypatch.setattr(module, "run_perception", stage("perception"))
    monkeypatch.setattr(module, "build_training_bank", stage("build-train-bank"))

    def train(**kwargs):
        stage("train")(**kwargs)
        model_dir = output / "models"
        model_dir.mkdir(parents=True)
        torch.save({"weight": torch.ones(2)}, model_dir / "l_ctg_0.pt")
        return {"stage": "train"}

    monkeypatch.setattr(module, "train_models", train)
    monkeypatch.setattr(module, "validate_models", stage("validate"))
    monkeypatch.setattr(module, "evaluate_study", stage("evaluate"))

    def replay(**kwargs):
        stage("replay-audit")(**kwargs)
        output.mkdir(exist_ok=True)
        (output / "representative_replay.parquet").write_bytes(
            b"PAR1syntheticPAR1"
        )
        (output / "replay_validation.json").write_text("{}\n", encoding="utf-8")
        return {"stage": "replay-audit"}

    monkeypatch.setattr(module, "run_representative_replay_audit", replay)

    def summarize(**kwargs):
        stage("summarize")(**kwargs)
        output.mkdir(exist_ok=True)
        for name in REQUIRED_RESULTS - {"trajectories.parquet"}:
            path = output / name
            if name == "models":
                path.mkdir(exist_ok=True)
            elif not path.exists():
                path.write_text("{}\n", encoding="utf-8")
        (output / "trajectories.parquet").write_bytes(b"PAR1syntheticPAR1")
        return {"stage": "summarize"}

    monkeypatch.setattr(module, "summarize_study", summarize)
    common = ["--config", str(config), "--source-root", str(source_root)]
    for command in (
        "prepare",
        "perception",
        "build-train-bank",
        "train",
        "validate",
    ):
        assert module.main([command, *common]) == 0
    assert module.main(["evaluate", *common, "--split", "test"]) == 0
    assert module.main(["replay-audit", *common]) == 0
    assert module.main(["summarize", *common]) == 0

    assert calls == [
        ("prepare", None),
        ("perception", None),
        ("build-train-bank", None),
        ("train", None),
        ("validate", None),
        ("evaluate", "test"),
        ("replay-audit", None),
        ("summarize", None),
    ]
    assert REQUIRED_RESULTS <= {path.name for path in output.iterdir()}
    state = torch.load(output / "models/l_ctg_0.pt", weights_only=True)
    assert torch.equal(state["weight"], torch.ones(2))
