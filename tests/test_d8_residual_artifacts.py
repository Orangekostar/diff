from __future__ import annotations

import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from cmc_bbdm.cpb_diffusion_marginalization import residual_artifacts
from cmc_bbdm.cpb_diffusion_marginalization.authority import (
    issue_inner_fold,
    issue_search_view,
)
from cmc_bbdm.cpb_diffusion_marginalization.config import (
    DOMAIN_ORDER,
    load_d8_config,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_artifacts import (
    ResidualArtifactError,
    ResidualArtifactRecorder,
    _publish_built_residual_package,
    build_residual_search_package,
    publish_residual_search_package,
    validate_residual_search_package,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_config import (
    load_residual_diffusion_config,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_model import (
    build_residual_unet,
    freeze_residual_checkpoint,
    load_residual_checkpoint,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_search import (
    ResidualCellEvaluation,
    ResidualCellRun,
    ResidualIncumbentEvidence,
    ResidualOuterSearchRun,
    ResidualSearchCell,
    promote_stage_a_outer,
    select_stage_b_pipeline,
    stage_a_cell_keys,
    stage_b_cell_keys,
    summarize_candidate_cells,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_training import (
    EpochLossRecord,
    ResidualFinalTrainingResult,
    ResidualTrainingResult,
)
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_residual_diffusion.yaml"


@lru_cache(maxsize=1)
def _authorities():
    config = load_residual_diffusion_config(CONFIG, project_root=PROJECT_ROOT)
    exploration = load_d8_config(
        PROJECT_ROOT / config.sources["exploration_config"].path,
        project_root=PROJECT_ROOT,
    )
    v3 = load_v3_config(
        PROJECT_ROOT / exploration.sources["p1_config"].path,
        project_root=PROJECT_ROOT,
    )
    data = load_data(v3, PROJECT_ROOT)
    search = issue_search_view(data, outer_domain=DOMAIN_ORDER[0], config=exploration)
    fold = issue_inner_fold(search, query_domain=DOMAIN_ORDER[1])
    return config, fold


@lru_cache(maxsize=1)
def _checkpoint():
    config, fold = _authorities()
    candidate = config.candidate("RD0")
    return freeze_residual_checkpoint(
        build_residual_unet(candidate),
        candidate=candidate,
        config_sha256=config.config_sha256,
        split_sha256=fold.state_sha256,
        training_seed=config.screening_seed,
    )


@lru_cache(maxsize=1)
def _final_checkpoint():
    config, fold = _authorities()
    candidate = config.candidate("RD0")
    return freeze_residual_checkpoint(
        build_residual_unet(candidate),
        candidate=candidate,
        config_sha256=config.config_sha256,
        split_sha256=fold.search_view.state_sha256,
        training_seed=config.screening_seed,
    )


def _run(stage: str) -> ResidualCellRun:
    config, fold = _authorities()
    cell = ResidualSearchCell(
        stage,
        fold.outer_domain,
        fold.query_domain,
        "RD0",
        config.screening_seed,
    )
    checkpoint = _checkpoint()
    training = ResidualTrainingResult(
        outer_domain=fold.outer_domain,
        query_domain=fold.query_domain,
        candidate_id="RD0",
        seed=config.screening_seed,
        epochs=1,
        fit_specimen_ids=fold.fit_specimen_ids,
        fit_dataset_ids=fold.fit_dataset_ids,
        target_state_sha256="1" * 64,
        split_sha256=fold.state_sha256,
        epoch_losses=(
            EpochLossRecord(
                1,
                1.0,
                1.0,
                0.0,
                0.0,
                len(fold.fit_specimen_ids),
                1,
            ),
        ),
        checkpoint=checkpoint,
        sample_count=len(fold.fit_specimen_ids),
        batch_count=1,
        response_read_count=0,
        test_scale_override=True,
    )
    target = np.asarray((0.25,), dtype=np.float64)
    evaluation = ResidualCellEvaluation(
        cell=cell,
        specimen_ids=(fold.query_specimen_ids[0],),
        targets=target,
        predictions=target + 0.01,
        accepted_proposals=1,
        proposed_variants=1,
        checkpoint_sha256=checkpoint.scientific_digest,
        prediction_sha256=hashlib.sha256(
            f"prediction:{stage}".encode("ascii")
        ).hexdigest(),
    )
    return ResidualCellRun(
        cell=cell,
        training=training,
        feature_bundle_sha256=hashlib.sha256(
            f"feature:{stage}".encode("ascii")
        ).hexdigest(),
        evaluation=evaluation,
    )


def _evaluation(cell: ResidualSearchCell, *, error: float) -> ResidualCellEvaluation:
    targets = np.asarray((0.25, 0.75), dtype=np.float64)
    return ResidualCellEvaluation(
        cell=cell,
        specimen_ids=(f"{cell.query_domain}-0", f"{cell.query_domain}-1"),
        targets=targets,
        predictions=targets + error,
        accepted_proposals=4,
        proposed_variants=4,
        checkpoint_sha256=hashlib.sha256(
            f"checkpoint:{cell.state_sha256}".encode("ascii")
        ).hexdigest(),
        prediction_sha256=hashlib.sha256(
            f"prediction:{cell.state_sha256}".encode("ascii")
        ).hexdigest(),
    )


def _outer_run_with_incumbent_selection() -> ResidualOuterSearchRun:
    config, _fold = _authorities()
    outer = DOMAIN_ORDER[0]
    stage_a_evaluations = tuple(
        _evaluation(cell, error=0.01 * int(cell.candidate_id[2:]))
        for cell in stage_a_cell_keys(config)
        if cell.outer_domain == outer
    )
    stage_a = promote_stage_a_outer(stage_a_evaluations, config=config)
    stage_b_evaluations = tuple(
        _evaluation(
            cell,
            error={"RD0": 0.10, "RD1": 0.11}[cell.candidate_id],
        )
        for cell in stage_b_cell_keys(
            config,
            finalists={domain: ("RD0", "RD1") for domain in DOMAIN_ORDER},
        )
        if cell.outer_domain == outer
    )
    reference = summarize_candidate_cells(
        tuple(
            row for row in stage_b_evaluations if row.cell.candidate_id == "RD0"
        ),
        config=config,
        stage="B",
    )
    incumbents = tuple(
        ResidualIncumbentEvidence(
            pipeline_id=pipeline_id,
            outer_domain=outer,
            specimen_ids=reference.oof_specimen_ids,
            domain_ids=reference.oof_domain_ids,
            targets=reference.oof_targets,
            predictions=reference.oof_targets + error,
            evidence_sha256=hashlib.sha256(
                f"{pipeline_id}:{outer}".encode("ascii")
            ).hexdigest(),
        )
        for pipeline_id, error in (("PILOT", 0.09), ("B0", 0.12))
    )
    selection = select_stage_b_pipeline(
        stage_b_evaluations,
        incumbents=incumbents,
        finalists=stage_a.finalists,
        config=config,
    )
    return ResidualOuterSearchRun(
        outer_domain=outer,
        stage_a=stage_a,
        stage_a_run_sha256=tuple(
            hashlib.sha256(f"stage-a:{index}".encode("ascii")).hexdigest()
            for index in range(40)
        ),
        stage_b_run_sha256=tuple(
            hashlib.sha256(f"stage-b:{index}".encode("ascii")).hexdigest()
            for index in range(30)
        ),
        selection=selection,
        final_training_sha256=(),
        outer_evaluation_count=0,
    )


def _write_summary_row(path: Path, summary) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        fields = tuple(csv.DictReader(handle).fieldnames or ())
    row = {
        "stage": summary.stage,
        "outer_domain": summary.outer_domain,
        "candidate_id": summary.candidate_id,
        "training_seeds": json.dumps(list(summary.training_seeds), separators=(",", ":")),
        "domain_mae": json.dumps(
            dict(summary.domain_mae), sort_keys=True, separators=(",", ":")
        ),
        "mean_mae": summary.mean_mae,
        "worst_mae": summary.worst_mae,
        "domain_sd": summary.domain_sd,
        "objective": summary.objective,
        "overall_acceptance": summary.overall_acceptance,
        "domain_acceptance": json.dumps(
            dict(summary.domain_acceptance), sort_keys=True, separators=(",", ":")
        ),
        "eligible": summary.eligible,
        "failed_domains": json.dumps(list(summary.failed_domains), separators=(",", ":")),
        "oof_count": len(summary.oof_specimen_ids),
        "target_sha256": hashlib.sha256(
            np.ascontiguousarray(summary.oof_targets).dtype.str.encode("ascii")
            + b"\0"
            + repr(np.ascontiguousarray(summary.oof_targets).shape).encode("ascii")
            + b"\0"
            + np.ascontiguousarray(summary.oof_targets).tobytes(order="C")
        ).hexdigest(),
        "prediction_sha256": hashlib.sha256(
            np.ascontiguousarray(summary.oof_predictions).dtype.str.encode("ascii")
            + b"\0"
            + repr(np.ascontiguousarray(summary.oof_predictions).shape).encode("ascii")
            + b"\0"
            + np.ascontiguousarray(summary.oof_predictions).tobytes(order="C")
        ).hexdigest(),
        "cell_state_sha256": json.dumps(
            list(summary.cell_state_sha256), separators=(",", ":")
        ),
        "state_sha256": summary.state_sha256,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def _complete_stage_a_source(tmp_path: Path) -> Path:
    config, first_fold = _authorities()
    candidate = config.candidate("RD0")
    model = build_residual_unet(candidate)
    recorder = ResidualArtifactRecorder(
        tmp_path / "source", config=config, config_path=CONFIG
    )
    evaluations = []
    for index, query_domain in enumerate(DOMAIN_ORDER[1:], start=1):
        fold = issue_inner_fold(first_fold.search_view, query_domain=query_domain)
        cell = ResidualSearchCell(
            "A", DOMAIN_ORDER[0], query_domain, "RD0", config.screening_seed
        )
        checkpoint = freeze_residual_checkpoint(
            model,
            candidate=candidate,
            config_sha256=config.config_sha256,
            split_sha256=fold.state_sha256,
            training_seed=config.screening_seed,
        )
        training = ResidualTrainingResult(
            outer_domain=cell.outer_domain,
            query_domain=cell.query_domain,
            candidate_id=cell.candidate_id,
            seed=cell.training_seed,
            epochs=1,
            fit_specimen_ids=fold.fit_specimen_ids,
            fit_dataset_ids=fold.fit_dataset_ids,
            target_state_sha256=hashlib.sha256(
                f"target:{query_domain}".encode("ascii")
            ).hexdigest(),
            split_sha256=fold.state_sha256,
            epoch_losses=(
                EpochLossRecord(
                    1,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    len(fold.fit_specimen_ids),
                    1,
                ),
            ),
            checkpoint=checkpoint,
            sample_count=len(fold.fit_specimen_ids),
            batch_count=1,
            response_read_count=0,
            test_scale_override=True,
        )
        target = np.asarray((0.25 + index / 100.0,), dtype=np.float64)
        evaluation = ResidualCellEvaluation(
            cell=cell,
            specimen_ids=(fold.query_specimen_ids[0],),
            targets=target,
            predictions=target + index / 1000.0,
            accepted_proposals=4,
            proposed_variants=4,
            checkpoint_sha256=checkpoint.scientific_digest,
            prediction_sha256=hashlib.sha256(
                f"prediction:{query_domain}".encode("ascii")
            ).hexdigest(),
        )
        run = ResidualCellRun(
            cell=cell,
            training=training,
            feature_bundle_sha256=hashlib.sha256(
                f"feature:{query_domain}".encode("ascii")
            ).hexdigest(),
            evaluation=evaluation,
        )
        recorder.record_cell(run, retain_checkpoint=False)
        evaluations.append(evaluation)
    summary = summarize_candidate_cells(
        tuple(evaluations), config=config, stage="A"
    )
    recorder.finalize_source(test_scale_override=True)
    _write_summary_row(recorder.root / "inner_metrics.csv", summary)
    return recorder.root


def _rewrite_checksums(root: Path) -> None:
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    (root / "CHECKSUMS.sha256").write_text(
        "".join(
            f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ),
        encoding="ascii",
        newline="\n",
    )


def _simple_source(tmp_path: Path, name: str = "source") -> Path:
    config, _fold = _authorities()
    recorder = ResidualArtifactRecorder(
        tmp_path / name, config=config, config_path=CONFIG
    )
    recorder.record_cell(_run("A"), retain_checkpoint=False)
    recorder.finalize_source(test_scale_override=True)
    return recorder.root


def test_regular_file_uses_one_descriptor_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact.bin"
    replacement = tmp_path / "replacement.bin"
    artifact.write_bytes(b"trusted!")
    replacement.write_bytes(b"changed!")
    original_read_bytes = Path.read_bytes
    swapped = False

    def replace_before_path_read(path: Path) -> bytes:
        nonlocal swapped
        if path == artifact and not swapped:
            swapped = True
            path.unlink()
            path.symlink_to(replacement)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", replace_before_path_read)
    payload = residual_artifacts._regular_file(artifact)

    assert payload == b"trusted!"
    assert swapped is False


def test_recorder_streams_only_full_rerank_checkpoint_files(tmp_path: Path) -> None:
    config, fold = _authorities()
    recorder = ResidualArtifactRecorder(
        tmp_path / "source", config=config, config_path=CONFIG
    )

    recorder.record_cell(_run("A"), retain_checkpoint=False)
    recorder.record_cell(_run("B"), retain_checkpoint=True)

    assert len(recorder.training_rows) == 2
    assert len(recorder.prediction_rows) == 2
    assert len(recorder.checkpoint_rows) == 1
    assert tuple(sorted(path.name for path in (tmp_path / "source/models").iterdir())) == (
        f"stage_b__{fold.outer_domain}__{fold.query_domain}__RD0__20260823.json",
        f"stage_b__{fold.outer_domain}__{fold.query_domain}__RD0__20260823.safetensors",
    )
    row = recorder.checkpoint_rows[0]
    loaded = load_residual_checkpoint(
        tmp_path / "source" / row["weights_path"],
        tmp_path / "source" / row["metadata_path"],
        candidate=config.candidate("RD0"),
        config_sha256=config.config_sha256,
        split_sha256=fold.state_sha256,
    )
    assert loaded.scientific_digest == _checkpoint().scientific_digest
    assert row["role"] == "stage_b"
    assert row["weights_sha256"] == hashlib.sha256(
        (tmp_path / "source" / row["weights_path"]).read_bytes()
    ).hexdigest()
    assert row["metadata_sha256"] == hashlib.sha256(
        (tmp_path / "source" / row["metadata_path"]).read_bytes()
    ).hexdigest()


def test_worker_source_finalization_rejects_an_incomplete_outer_roster(
    tmp_path: Path,
) -> None:
    config, _fold = _authorities()
    recorder = ResidualArtifactRecorder(
        tmp_path / "worker-source",
        config=config,
        config_path=CONFIG,
    )

    with np.testing.assert_raises_regex(
        ResidualArtifactError,
        "worker.*roster|roster.*worker",
    ):
        recorder.finalize_source(
            test_scale_override=True,
            expected_outer_domains=DOMAIN_ORDER[:2],
        )


def test_recorder_binds_one_complete_five_domain_final_checkpoint(
    tmp_path: Path,
) -> None:
    config, fold = _authorities()
    checkpoint = _final_checkpoint()
    result = ResidualFinalTrainingResult(
        outer_domain=fold.outer_domain,
        candidate_id="RD0",
        seed=config.screening_seed,
        epochs=1,
        fit_specimen_ids=fold.search_view.specimen_ids,
        fit_dataset_ids=fold.search_view.dataset_ids,
        target_state_sha256="2" * 64,
        split_sha256=fold.search_view.state_sha256,
        epoch_losses=(EpochLossRecord(1, 1.0, 1.0, 0.0, 0.0, 1, 1),),
        checkpoint=checkpoint,
        sample_count=fold.search_view.specimen_count,
        batch_count=1,
        response_read_count=0,
        test_scale_override=True,
    )
    recorder = ResidualArtifactRecorder(
        tmp_path / "source", config=config, config_path=CONFIG
    )

    recorder.record_final(result)

    assert len(recorder.training_rows) == 1
    assert len(recorder.prediction_rows) == 0
    assert len(recorder.checkpoint_rows) == 1
    row = recorder.checkpoint_rows[0]
    assert row["role"] == "final"
    assert row["query_domain"] == ""
    assert row["split_sha256"] == fold.search_view.state_sha256
    assert row["checkpoint_scientific_digest"] == checkpoint.scientific_digest
    with np.testing.assert_raises_regex(ValueError, "duplicate"):
        recorder.record_final(result)


def test_recorder_serializes_outer_metrics_and_frozen_selection(
    tmp_path: Path,
) -> None:
    config, _fold = _authorities()
    recorder = ResidualArtifactRecorder(
        tmp_path / "source", config=config, config_path=CONFIG
    )
    recorder.record_cell(_run("A"), retain_checkpoint=False)
    recorder.record_cell(_run("B"), retain_checkpoint=True)
    outer_run = _outer_run_with_incumbent_selection()

    recorder.record_outer(outer_run)
    recorder.finalize_source(test_scale_override=True)

    with (recorder.root / "inner_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metrics = tuple(csv.DictReader(handle))
    selected = json.loads(
        (recorder.root / "selected_generators.json").read_text(encoding="ascii")
    )
    frozen = json.loads(
        (recorder.root / "frozen_pipelines.json").read_text(encoding="ascii")
    )
    assert len(metrics) == 10
    assert [row["stage"] for row in metrics].count("A") == 8
    assert [row["stage"] for row in metrics].count("B") == 2
    assert len(selected["selections"]) == 1
    assert len(frozen["selections"]) == 1
    assert selected["selections"][0]["selected_pipeline"] == "INCUMBENT"
    assert frozen["selections"][0]["outer_evaluation_started"] is False
    assert frozen["selections"][0]["outer_run_sha256"] == outer_run.state_sha256


def test_test_scale_package_has_exact_tree_manifest_and_checksums(
    tmp_path: Path,
) -> None:
    config, _fold = _authorities()
    recorder = ResidualArtifactRecorder(
        tmp_path / "source", config=config, config_path=CONFIG
    )
    recorder.record_cell(_run("A"), retain_checkpoint=False)
    recorder.record_cell(_run("B"), retain_checkpoint=True)
    final = ResidualFinalTrainingResult(
        outer_domain=DOMAIN_ORDER[0],
        candidate_id="RD0",
        seed=config.screening_seed,
        epochs=1,
        fit_specimen_ids=_authorities()[1].search_view.specimen_ids,
        fit_dataset_ids=_authorities()[1].search_view.dataset_ids,
        target_state_sha256="2" * 64,
        split_sha256=_authorities()[1].search_view.state_sha256,
        epoch_losses=(EpochLossRecord(1, 1.0, 1.0, 0.0, 0.0, 1, 1),),
        checkpoint=_final_checkpoint(),
        sample_count=_authorities()[1].search_view.specimen_count,
        batch_count=1,
        response_read_count=0,
        test_scale_override=True,
    )
    recorder.record_final(final)
    recorder.finalize_source(test_scale_override=True)

    validated = build_residual_search_package(
        tmp_path / "package",
        source_dir=recorder.root,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    )

    assert validated.outer_evaluation_count == 0
    assert validated.test_scale_override
    assert validated.training_count == 3
    assert validated.checkpoint_count == 2
    assert {path.name for path in (tmp_path / "package").iterdir()} == {
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
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
    assert validate_residual_search_package(
        tmp_path / "package",
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    ) == validated


def test_package_recomputes_inner_metrics_from_prediction_rows(
    tmp_path: Path,
) -> None:
    source = _complete_stage_a_source(tmp_path)
    build_residual_search_package(
        tmp_path / "valid-package",
        source_dir=source,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    )
    metrics_path = source / "inner_metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
        fields = tuple(rows[0])
    rows[0]["mean_mae"] = "0.0"
    rows[0]["objective"] = "0.0"
    rows[0]["state_sha256"] = "0" * 64
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with np.testing.assert_raises_regex(
        ResidualArtifactError, "metric summary"
    ):
        build_residual_search_package(
            tmp_path / "tampered-package",
            source_dir=source,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
        )


def test_manifest_binds_runtime_sources_code_and_split_provenance(
    tmp_path: Path,
) -> None:
    config, _fold = _authorities()
    recorder = ResidualArtifactRecorder(
        tmp_path / "source", config=config, config_path=CONFIG
    )
    recorder.record_cell(_run("A"), retain_checkpoint=False)
    recorder.finalize_source(test_scale_override=True)
    package = tmp_path / "package"
    build_residual_search_package(
        package,
        source_dir=recorder.root,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    )
    manifest_path = package / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    provenance = manifest["execution_provenance"]
    assert set(provenance) == {"runtime", "sources", "code", "splits"}
    assert provenance["runtime"] == dict(config.runtime)
    assert provenance["sources"] == {
        key: {"path": source.path, "sha256": source.sha256}
        for key, source in config.sources.items()
    }
    assert "src/cmc_bbdm/cpb_diffusion_marginalization/residual_artifacts.py" in (
        provenance["code"]
    )
    assert len(provenance["splits"]) == 1

    provenance["code"].pop(
        "src/cmc_bbdm/cpb_diffusion_marginalization/residual_artifacts.py"
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    _rewrite_checksums(package)
    with np.testing.assert_raises_regex(
        ResidualArtifactError, "execution provenance"
    ):
        validate_residual_search_package(
            package,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
        )


def test_transaction_publication_commits_validated_package_without_residue(
    tmp_path: Path,
) -> None:
    config, _fold = _authorities()
    recorder = ResidualArtifactRecorder(
        tmp_path / "source", config=config, config_path=CONFIG
    )
    recorder.record_cell(_run("A"), retain_checkpoint=False)
    recorder.finalize_source(test_scale_override=True)
    output = tmp_path / "published"

    published = _publish_built_residual_package(
        recorder.root,
        output,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    )

    assert published == validate_residual_search_package(
        output,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    )
    assert not tuple(tmp_path.glob(".published.transaction-*"))


def test_double_rename_failure_preserves_previous_for_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _simple_source(tmp_path)
    output = tmp_path / "published"
    original = _publish_built_residual_package(
        source,
        output,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    )
    real_replace = residual_artifacts._atomic_replace

    def fail_commit_and_rollback(source_path: Path, target_path: Path) -> None:
        if target_path == output and source_path.name in {"staged", "previous"}:
            raise OSError(f"injected rename failure: {source_path.name}")
        real_replace(source_path, target_path)

    monkeypatch.setattr(
        residual_artifacts, "_atomic_replace", fail_commit_and_rollback
    )
    with np.testing.assert_raises_regex(OSError, "staged|previous"):
        _publish_built_residual_package(
            source,
            output,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
        )
    transactions = tuple(tmp_path.glob(".published.transaction-*"))
    assert not output.exists()
    assert len(transactions) == 1
    assert validate_residual_search_package(
        transactions[0] / "previous",
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    ) == original

    monkeypatch.setattr(residual_artifacts, "_atomic_replace", real_replace)
    recovered = residual_artifacts._recover_residual_publication_unlocked(
        output,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    )
    assert recovered == original
    assert not transactions[0].exists()


def test_interrupted_first_publication_recovers_valid_staged_package(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _simple_source(tmp_path)
    output = tmp_path / "published"
    real_replace = residual_artifacts._atomic_replace

    def fail_first_commit(source_path: Path, target_path: Path) -> None:
        if source_path.name == "staged" and target_path == output:
            raise OSError("injected first publication failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(residual_artifacts, "_atomic_replace", fail_first_commit)
    with np.testing.assert_raises_regex(OSError, "first publication"):
        _publish_built_residual_package(
            source,
            output,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
        )
    transaction = next(iter(tmp_path.glob(".published.transaction-*")))
    assert (transaction / "staged").is_dir()
    assert not output.exists()

    monkeypatch.setattr(residual_artifacts, "_atomic_replace", real_replace)
    recovered = residual_artifacts._recover_residual_publication_unlocked(
        output,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    )
    assert recovered == validate_residual_search_package(
        output,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    )
    assert not transaction.exists()


def test_publication_lock_and_registered_leaf_are_fail_closed(
    tmp_path: Path,
) -> None:
    source = _simple_source(tmp_path)
    output = tmp_path / "published"
    with (
        residual_artifacts._publication_lock(output),
        np.testing.assert_raises_regex(ResidualArtifactError, "already active"),
    ):
        _publish_built_residual_package(
            source,
            output,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
        )
    with np.testing.assert_raises_regex(
        ResidualArtifactError, "registered safe leaf"
    ):
        publish_residual_search_package(
            source,
            tmp_path / "unsafe",
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
        )


def test_package_rejects_self_declared_selection_record(
    tmp_path: Path,
) -> None:
    source = _simple_source(tmp_path)
    for name in ("selected_generators.json", "frozen_pipelines.json"):
        path = source / name
        payload = json.loads(path.read_text(encoding="ascii"))
        payload["selections"] = [{"outer_domain": DOMAIN_ORDER[0]}]
        path.write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
            encoding="ascii",
            newline="\n",
        )

    with np.testing.assert_raises_regex(
        ResidualArtifactError, "selection record"
    ):
        build_residual_search_package(
            tmp_path / "package",
            source_dir=source,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
        )


def test_package_rederives_nested_split_authority_from_registered_data(
    tmp_path: Path,
) -> None:
    source = _simple_source(tmp_path)
    build_residual_search_package(
        tmp_path / "valid-package",
        source_dir=source,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
    )
    training_path = source / "training.csv"
    with training_path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
        fields = tuple(rows[0])
    fit_ids = json.loads(rows[0]["fit_specimen_ids"])
    rows[0]["fit_specimen_ids"] = json.dumps(
        list(reversed(fit_ids)), separators=(",", ":")
    )
    rows[0]["split_sha256"] = "f" * 64
    with training_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with np.testing.assert_raises_regex(
        ResidualArtifactError, "split authority"
    ):
        build_residual_search_package(
            tmp_path / "tampered-package",
            source_dir=source,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
        )


def test_package_rejects_selection_transplanted_without_its_cell_evidence(
    tmp_path: Path,
) -> None:
    source = _simple_source(tmp_path, "source")
    config, _fold = _authorities()
    evidence = ResidualArtifactRecorder(
        tmp_path / "selection-source", config=config, config_path=CONFIG
    )
    evidence.record_cell(_run("A"), retain_checkpoint=False)
    evidence.record_outer(_outer_run_with_incumbent_selection())
    evidence.finalize_source(test_scale_override=True)
    for name in ("selected_generators.json", "frozen_pipelines.json"):
        (source / name).write_bytes((evidence.root / name).read_bytes())

    with np.testing.assert_raises_regex(
        ResidualArtifactError, "selection evidence"
    ):
        build_residual_search_package(
            tmp_path / "package",
            source_dir=source,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
        )
