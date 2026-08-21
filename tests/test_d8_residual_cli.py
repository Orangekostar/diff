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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_d8_residual_diffusion.py"
SHELL = PROJECT_ROOT / "scripts/run_d8_residual_diffusion.sh"
CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_residual_diffusion.yaml"
_WORKER_SOURCE_FILES = {
    "config.yaml",
    "candidate_index.csv",
    "training.csv",
    "inner_predictions.csv",
    "inner_metrics.csv",
    "checkpoint_index.csv",
    "selected_generators.json",
    "frozen_pipelines.json",
    "models",
    "REPORT.md",
}


def _module() -> object:
    spec = importlib.util.spec_from_file_location("run_d8_residual_diffusion", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load residual diffusion CLI module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker_source(
    root: Path,
    *,
    worker_index: int,
    outer_domains: tuple[str, ...],
    config: object,
) -> Path:
    root.mkdir()
    (root / "models").mkdir()
    (root / "config.yaml").write_bytes(CONFIG.read_bytes())
    (root / "candidate_index.csv").write_text(
        "candidate_id\nRD0\n", encoding="ascii"
    )
    with (root / "training.csv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("role", "outer_domain"),
            lineterminator="\n",
        )
        writer.writeheader()
        for outer_domain in outer_domains:
            writer.writerows(
                {"role": "stage_a", "outer_domain": outer_domain}
                for _ in range(40)
            )
            writer.writerows(
                {"role": "stage_b", "outer_domain": outer_domain}
                for _ in range(30)
            )
    for name in (
        "inner_predictions.csv",
        "inner_metrics.csv",
        "checkpoint_index.csv",
    ):
        (root / name).write_text("row\nvalue\n", encoding="ascii")
    common = {
        "schema_version": 1,
        "config_sha256": config.config_sha256,
        "outer_evaluation_count": 0,
        "test_scale_override": False,
        "selections": [
            {"outer_domain": outer_domain} for outer_domain in outer_domains
        ],
    }
    for name, scope in (
        ("selected_generators.json", "cpb_d8_residual_selected_generators"),
        ("frozen_pipelines.json", "cpb_d8_residual_frozen_pipelines"),
    ):
        (root / name).write_text(
            json.dumps({**common, "scope": scope}, sort_keys=True) + "\n",
            encoding="ascii",
        )
    (root / "models" / f"worker-{worker_index}.bin").write_bytes(
        f"worker:{worker_index}".encode("ascii")
    )
    (root / "REPORT.md").write_text(
        f"# Worker {worker_index}\n", encoding="ascii"
    )
    assert {path.name for path in root.iterdir()} == _WORKER_SOURCE_FILES
    return root


def test_cli_exposes_only_registered_preouter_commands(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[str] = []

    def fake_run(command: str) -> dict[str, object]:
        calls.append(command)
        return {"command": command, "status": "PASS", "outer_evaluation_count": 0}

    monkeypatch.setattr(module, "run_registered_command", fake_run)
    for command in ("smoke", "train", "validate", "replay"):
        assert module.main([command]) == 0
        assert json.loads(capsys.readouterr().out) == {
            "command": command,
            "status": "PASS",
            "outer_evaluation_count": 0,
        }
    assert calls == ["smoke", "train", "validate", "replay"]
    for arguments in (
        ["outer"],
        ["formal"],
        ["train", "--outer", DOMAIN_ORDER[0]],
        ["train", "--config", "/tmp/unregistered.yaml"],
    ):
        with pytest.raises(SystemExit):
            module.main(arguments)


def test_worker_plan_assigns_exactly_two_outers_to_each_gpu() -> None:
    module = _module()

    assignments = module.registered_worker_assignments()

    assert assignments == (
        (0, DOMAIN_ORDER[0:2]),
        (1, DOMAIN_ORDER[2:4]),
        (2, DOMAIN_ORDER[4:6]),
    )
    assert tuple(
        outer for _gpu, outer_domains in assignments for outer in outer_domains
    ) == DOMAIN_ORDER


def test_train_and_replay_require_exactly_three_visible_a40_gpus() -> None:
    module = _module()

    valid = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 3,
            get_device_name=lambda index: f"NVIDIA A40 GPU {index}",
        )
    )
    assert module.require_registered_gpu_inventory(valid) == (
        "NVIDIA A40 GPU 0",
        "NVIDIA A40 GPU 1",
        "NVIDIA A40 GPU 2",
    )
    for invalid in (
        SimpleNamespace(
            cuda=SimpleNamespace(
                device_count=lambda: 2,
                get_device_name=lambda index: "NVIDIA A40",
            )
        ),
        SimpleNamespace(
            cuda=SimpleNamespace(
                device_count=lambda: 3,
                get_device_name=lambda index: (
                    "NVIDIA A100" if index == 1 else "NVIDIA A40"
                ),
            )
        ),
    ):
        with pytest.raises(RuntimeError, match="three.*A40|A40.*three"):
            module.require_registered_gpu_inventory(invalid)


def test_registered_outputs_keep_production_and_replay_separate() -> None:
    module = _module()
    config = SimpleNamespace(
        output_dir="results/d8_residual_diffusion_search",
        replay_output_dir="results/replay/d8_residual_diffusion_search",
    )

    assert module.registered_output("train", config=config) == (
        PROJECT_ROOT / config.output_dir
    )
    assert module.registered_output("replay", config=config) == (
        PROJECT_ROOT / config.replay_output_dir
    )
    with pytest.raises(ValueError, match="output|command"):
        module.registered_output("smoke", config=config)


def test_shell_wrapper_freezes_runtime_and_has_no_outer_surface() -> None:
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
    assert "paper_v3/configs/d8_residual_diffusion.yaml" in text
    assert "run_d8_residual_diffusion.py" in text
    assert "outer-evaluation" not in text


def test_worker_manifest_binds_assignment_code_config_sources_and_tree(
    tmp_path: Path,
) -> None:
    module = _module()
    config = module._config()
    source = _worker_source(
        tmp_path / "worker-0",
        worker_index=0,
        outer_domains=DOMAIN_ORDER[:2],
        config=config,
    )

    manifest = module.write_worker_manifest(
        source,
        worker_index=0,
        gpu_index=0,
        outer_domains=DOMAIN_ORDER[:2],
        config=config,
        test_scale_override=False,
    )
    validated = module.validate_worker_manifest(
        source,
        worker_index=0,
        gpu_index=0,
        outer_domains=DOMAIN_ORDER[:2],
        config=config,
        test_scale_override=False,
    )

    assert validated == manifest
    assert manifest["outer_evaluation_count"] == 0
    assert manifest["outer_domains"] == list(DOMAIN_ORDER[:2])
    assert manifest["source_records"]
    assert manifest["code_records"][
        "scripts/run_d8_residual_diffusion.py"
    ]["sha256"]
    assert manifest["registered_sources"] == {
        key: {"path": value.path, "sha256": value.sha256}
        for key, value in config.sources.items()
    }

    with (source / "training.csv").open("a", encoding="ascii") as handle:
        handle.write("stage_a,wrong-outer\n")
    with pytest.raises(RuntimeError, match="manifest|source|tree|allocation"):
        module.validate_worker_manifest(
            source,
            worker_index=0,
            gpu_index=0,
            outer_domains=DOMAIN_ORDER[:2],
            config=config,
            test_scale_override=False,
        )


def test_parent_merges_only_three_validated_worker_sources(tmp_path: Path) -> None:
    module = _module()
    config = module._config()
    roots = []
    for worker_index, (gpu_index, outer_domains) in enumerate(
        module.registered_worker_assignments()
    ):
        source = _worker_source(
            tmp_path / f"worker-{worker_index}",
            worker_index=worker_index,
            outer_domains=outer_domains,
            config=config,
        )
        module.write_worker_manifest(
            source,
            worker_index=worker_index,
            gpu_index=gpu_index,
            outer_domains=outer_domains,
            config=config,
            test_scale_override=False,
        )
        roots.append(source)

    merged = module.merge_worker_sources(
        tuple(roots),
        tmp_path / "merged",
        config=config,
        test_scale_override=False,
    )

    assert {path.name for path in merged.iterdir()} == _WORKER_SOURCE_FILES
    with (merged / "training.csv").open(
        encoding="ascii", newline=""
    ) as handle:
        rows = tuple(csv.DictReader(handle))
    assert len(rows) == 6 * (40 + 30)
    assert tuple(dict.fromkeys(row["outer_domain"] for row in rows)) == DOMAIN_ORDER
    selected = json.loads(
        (merged / "selected_generators.json").read_text(encoding="ascii")
    )
    assert [row["outer_domain"] for row in selected["selections"]] == list(
        DOMAIN_ORDER
    )
    assert sorted(path.name for path in (merged / "models").iterdir()) == [
        "worker-0.bin",
        "worker-1.bin",
        "worker-2.bin",
    ]


def test_worker_runs_only_its_two_preouter_studies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    config = module._config()
    outer_domains = DOMAIN_ORDER[:2]
    context = SimpleNamespace(
        config=config,
        exploration=object(),
        data=object(),
        assets=object(),
        scaffolds={outer: f"scaffold:{outer}" for outer in outer_domains},
        candidates={outer: f"candidate:{outer}" for outer in outer_domains},
        incumbents={outer: (f"pilot:{outer}", f"b0:{outer}") for outer in outer_domains},
        encoder=object(),
    )
    searches: dict[str, object] = {}
    events: list[tuple[object, ...]] = []

    class FakeRecorder:
        def __init__(self, root: Path, **kwargs: object) -> None:
            self.root = root
            self.root.mkdir()
            events.append(("recorder", kwargs["config"], kwargs["config_path"]))

        def record_cell(self, run: object, *, retain_checkpoint: bool) -> None:
            events.append(("cell", run, retain_checkpoint))

        def record_final(self, result: object) -> None:
            events.append(("final", result))

        def record_outer(self, result: object) -> None:
            events.append(("outer", result.outer_domain))

        def finalize_source(self, **kwargs: object) -> Path:
            events.append(("finalize", kwargs))
            return self.root

    def fake_search(data: object, *, outer_domain: str, config: object) -> object:
        assert data is context.data
        assert config is context.exploration
        search = SimpleNamespace(outer_domain=outer_domain)
        searches[outer_domain] = search
        return search

    def fake_fold(search: object, *, query_domain: str) -> object:
        assert query_domain != search.outer_domain
        return SimpleNamespace(
            search_view=search,
            outer_domain=search.outer_domain,
            query_domain=query_domain,
        )

    def fake_outer_run(search: object, **kwargs: object) -> object:
        outer = search.outer_domain
        assert tuple(kwargs["folds"]) == tuple(
            domain for domain in DOMAIN_ORDER if domain != outer
        )
        assert kwargs["scaffold"] == f"scaffold:{outer}"
        assert kwargs["pilot_candidate"] == f"candidate:{outer}"
        assert kwargs["field_bank"] == f"field-bank:{outer}"
        assert kwargs["incumbents"] == (f"pilot:{outer}", f"b0:{outer}")
        assert kwargs["device"] == "cuda:0"
        assert kwargs["test_scale_override"] is True
        return SimpleNamespace(outer_domain=outer)

    monkeypatch.setattr(module, "_load_worker_context", lambda device: context)
    monkeypatch.setattr(module, "ResidualArtifactRecorder", FakeRecorder)
    monkeypatch.setattr(module, "issue_search_view", fake_search)
    monkeypatch.setattr(module, "issue_inner_fold", fake_fold)
    monkeypatch.setattr(
        module,
        "load_search_residual_field_bank",
        lambda search, **kwargs: f"field-bank:{search.outer_domain}",
    )
    monkeypatch.setattr(module, "run_residual_outer_search", fake_outer_run)
    monkeypatch.setattr(
        module,
        "write_worker_manifest",
        lambda *args, **kwargs: {"state_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        module,
        "validate_worker_manifest",
        lambda *args, **kwargs: {"state_sha256": "a" * 64},
    )

    result = module.run_worker_source(
        tmp_path / "worker-source",
        worker_index=0,
        gpu_index=0,
        outer_domains=outer_domains,
        test_scale_override=True,
    )

    assert result["outer_domains"] == list(outer_domains)
    assert result["outer_evaluation_count"] == 0
    assert [event for event in events if event[0] == "outer"] == [
        ("outer", outer_domains[0]),
        ("outer", outer_domains[1]),
    ]
    assert events[-1] == (
        "finalize",
        {
            "test_scale_override": True,
            "expected_outer_domains": outer_domains,
        },
    )


def test_train_validates_merges_and_publishes_registered_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    config = module._config()
    worker_roots = tuple(tmp_path / f"worker-{index}" for index in range(3))
    output = tmp_path / "registered-production"
    merged = tmp_path / "merged-source"
    calls: list[tuple[object, ...]] = []
    result = SimpleNamespace(
        outer_evaluation_count=0,
        pipeline_count=6,
        training_count=420,
        checkpoint_count=180,
        scientific_digest="a" * 64,
        output_tree_sha256="b" * 64,
    )

    monkeypatch.setattr(module, "_config", lambda: config)
    monkeypatch.setattr(
        module,
        "require_registered_gpu_inventory",
        lambda torch_module: ("NVIDIA A40",) * 3,
    )
    monkeypatch.setattr(
        module,
        "registered_output",
        lambda command, **kwargs: output,
    )
    monkeypatch.setattr(
        module,
        "run_isolated_workers",
        lambda root, **kwargs: worker_roots,
    )

    def fake_merge(roots: object, target: Path, **kwargs: object) -> Path:
        calls.append(("merge", roots, kwargs))
        assert target.name == "merged-source"
        return merged

    def fake_publish(source: Path, target: Path, **kwargs: object) -> object:
        calls.append(("publish", source, target, kwargs))
        return result

    monkeypatch.setattr(module, "merge_worker_sources", fake_merge)
    monkeypatch.setattr(module, "publish_residual_search_package", fake_publish)

    observed = module.execute_train()

    assert observed == {
        "command": "train",
        "status": "PASS",
        "outer_evaluation_count": 0,
        "pipeline_count": 6,
        "training_count": 420,
        "checkpoint_count": 180,
        "scientific_digest": "a" * 64,
        "output_tree_sha256": "b" * 64,
    }
    assert calls[0][0] == "merge"
    assert calls[1][0:3] == ("publish", merged, output)


def test_validate_requires_identical_checkpoint_scientific_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    config = module._config()
    production = tmp_path / "production"
    replay = tmp_path / "replay"
    validated = SimpleNamespace(
        scientific_digest="a" * 64,
        output_tree_sha256="b" * 64,
    )

    monkeypatch.setattr(module, "_config", lambda: config)
    monkeypatch.setattr(
        module,
        "registered_output",
        lambda command, **kwargs: production if command == "train" else replay,
    )
    monkeypatch.setattr(
        module,
        "validate_residual_search_package",
        lambda *args, **kwargs: validated,
    )
    monkeypatch.setattr(
        module,
        "checkpoint_scientific_records",
        lambda root: (("production",),) if root == production else (("replay",),),
    )

    with pytest.raises(RuntimeError, match="checkpoint.*differ"):
        module.execute_validate()
