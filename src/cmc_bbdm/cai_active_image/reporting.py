"""Final figures, verification evidence, and handoff artifacts for CAI v2."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

from .figures import generate_preselected_figures
from .protocol import EXTERNAL_SOURCE_ROOT_BINDING, CAIActiveImageProtocol


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"JSON artifact is not a mapping: {path}")
    return value


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _run(command: list[str], *, cwd: Path) -> dict[str, object]:
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": process.returncode,
        "output": process.stdout.strip(),
    }


def _verification(root: Path) -> dict[str, object]:
    checks = [
        _run(
            [
                "python",
                "-m",
                "ruff",
                "check",
                "src/cmc_bbdm/cai_active_image",
                "scripts/run_cai_active_image_v2.py",
                "tests/test_cai_active_image_v2.py",
            ],
            cwd=root,
        ),
        _run(
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_cai_active_image_v2.py",
                "tests/test_learned_cscan_cli.py",
                "tests/test_learned_cscan_controls.py",
                "tests/test_learned_cscan_perception_readout.py",
                "tests/test_learned_cscan_rollouts_metrics.py",
                "tests/test_learned_cscan_runtime.py",
                "tests/test_learned_cscan_training.py",
            ],
            cwd=root,
        ),
        _run(["git", "diff", "--check"], cwd=root),
    ]
    if any(check["returncode"] != 0 for check in checks):
        raise ValueError("final scoped verification failed")
    protected = [
        "results/learned_cscan_same_perception",
        "results/multiview",
        "src/cmc_bbdm/learned_cscan",
        "src/cmc_bbdm/vlm_cscan",
        "src/cmc_bbdm/mavis",
    ]
    frozen = _run(
        ["git", "diff", "--name-only", "29b3249610c821e8f9c4f58d740588181c443dd5", "--", *protected],
        cwd=root,
    )
    if frozen["returncode"] != 0 or frozen["output"]:
        raise ValueError("frozen source or prior-result paths changed")
    return {
        "schema_version": 2,
        "checks": checks,
        "frozen_paths": protected,
        "frozen_diff": frozen,
        "status": "P6_SCOPED_VERIFICATION_PASS",
    }


def _format_metric(value: object) -> str:
    return f"{float(value):.4f}"


def _write_checksums(directory: Path) -> Path:
    checksum_path = directory / "CHECKSUMS.sha256"
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path != checksum_path:
            rows.append(f"{_sha256_file(path)}  {path.relative_to(directory).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return checksum_path


def finalize_study(
    protocol: CAIActiveImageProtocol,
    *,
    project_root: str | Path,
    source_root: str | Path,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    summary = _read_json(protocol.results_dir / "summary.json")
    predictor_manifest = _read_json(
        protocol.results_dir / "predictor_training_manifest.json"
    )
    actor_manifest = _read_json(protocol.results_dir / "actor_training_manifest.json")
    feature_manifest = _read_json(protocol.results_dir / "feature_bank_manifest.json")
    vlm_manifest = _read_json(protocol.results_dir / "vlm_prior_manifest.json")
    vlm_records = vlm_manifest.get("records")
    if type(vlm_records) is not list or any(type(row) is not dict for row in vlm_records):
        raise ValueError("VLM prior records are missing")
    compute_accounting = {
        "schema_version": 2,
        "wall_clock_limit_hours": protocol.wall_clock_limit_hours,
        "formal_retained_optimizer_updates": int(
            predictor_manifest["optimizer_updates"]
        )
        + int(actor_manifest["optimizer_updates"]),
        "vlm_new_calls": int(vlm_manifest["new_calls"]),
        "vlm_historical_call_count": int(vlm_manifest["original_calls"]),
        "vlm_historical_latency_seconds": sum(
            float(row["latency_seconds"])
            for row in vlm_records
        ),
        "encoder_elapsed_seconds": float(feature_manifest["encoder_elapsed_seconds"]),
        "predictor_elapsed_seconds": None,
        "predictor_timing_status": "FORMAL_RUN_DURATION_NOT_PERSISTED",
        "actor_elapsed_seconds": float(actor_manifest["elapsed_hours"]) * 3600.0,
        "exact_cumulative_elapsed_seconds": None,
        "wall_clock_assessment": (
            "FORMAL_EXECUTION_COMPLETED_BELOW_LIMIT; "
            "EXACT_CUMULATIVE_SECONDS_NOT_PERSISTED"
        ),
        "invalidated_development_updates": {
            "predictor": 8000,
            "actor": 13500,
        },
    }
    _write_json(
        protocol.results_dir / "compute_accounting.json", compute_accounting
    )
    performance = _read_csv(protocol.results_dir / "absolute_cai_performance.csv")
    effects = _read_csv(protocol.results_dir / "vlm_ablation_effects.csv")
    figures = generate_preselected_figures(
        protocol, project_root=root, source_root=source_root
    )
    verification = _verification(root)
    verification_path = protocol.results_dir / "verification.json"
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    main_endpoint = next(
        row
        for row in performance
        if row["method"] == "VLM_CAI_FEEDBACK_AGENT"
        and float(row["budget"]) == protocol.endpoint_budget
    )
    main_zero = next(
        row
        for row in performance
        if row["method"] == "VLM_CAI_FEEDBACK_AGENT" and float(row["budget"]) == 0.0
    )
    full = next(
        row for row in performance if row["method"] == "FULL_INPUT_COMMON_PREDICTOR"
    )
    endpoint_by_method = {
        row["method"]: row
        for row in performance
        if float(row["budget"]) == protocol.endpoint_budget
    }
    no_vlm_endpoint = endpoint_by_method["NO_VLM_FEEDBACK"]
    open_loop_endpoint = endpoint_by_method["VLM_OPEN_LOOP"]
    fixed_endpoint = endpoint_by_method[str(summary["best_nonadaptive"])]
    effect_lines = "\n".join(
        f"| {row['claim']} | {_format_metric(row['estimate_mpa'])} | "
        f"[{_format_metric(row['ci_lower_mpa'])}, {_format_metric(row['ci_upper_mpa'])}] | "
        f"{row['evidence_status']} |"
        for row in effects
    )
    source_lines = "\n".join(
        f"| {item.name} | `{item.path.relative_to(root)}` | `{item.sha256}` |"
        for item in protocol.sources
    )
    figure_lines = "\n".join(
        f"- `{row['file']}` (`{row['sha256']}`)"
        for row in figures["records"]  # type: ignore[index]
    )
    handoff = f"""# Codex Handoff: CAI VLM-Guided Agent v2

## Repository identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Branch: `research/cai-vlm-guided-agent-v2`
- Evidence base: `{protocol.repository_base_sha}`
- Protocol SHA-256: `{protocol.config_sha256}`
- Output roots: `results/cai_active_image/v2/`, `artifacts/cai_active_image/v2/`

## Executed design

- Main method: `VLM_CAI_FEEDBACK_AGENT`.
- Frozen VLM: `Qwen/Qwen2.5-VL-7B-Instruct` revision `{protocol.vlm_revision}`.
- VLM cache: 60/60 compatible hits; 0 new calls; prior execution recorded 62 calls including 2 format repairs.
- VLM cells are consumed in the once-registered 8x8 frame without a second rotation.
- First action uses only the highest available reliable VLM confidence level in C0; later actions unlock all affordable unmeasured cells.
- C0 fallback reasons are explicit in every trajectory and initialization row.
- Action: `NATIVE_8X8_FULL_CELL_V1`; endpoint cost {protocol.endpoint_budget:.2f} by exact native pixels.
- Actor state protocol `{protocol.actor_state_protocol}` includes spent/remaining cost and ordered acquired-cell geometry history.
- Common predictor: frozen ResNet18 surface/cell tokens plus only acquired C-scan tokens; one `P_all` checkpoint is shared by every method.
- The persisted feature bank contains TRAIN/VALID MPa targets but redacts all 24 TEST targets; TEST MPa is joined from the hash-bound authority only in P5 scoring.
- Learned controls: independently trained `NO_VLM_FEEDBACK`, `VLM_OPEN_LOOP`, and `LEARNED_STATIC`.
- Fixed controls: `SERPENTINE`, `CENTER_FIRST`, `GEOMETRY_SPREAD`, and `RANDOM`.
- `BEST_NONADAPTIVE` was selected on VALID as `{summary['best_nonadaptive']}` before TEST comparison.
- VALID main/no-VLM/best-nonadaptive AUEC: `{_format_metric(summary['validation']['main_normalized_error_area_mpa'])}` / `{_format_metric(summary['validation']['no_vlm_normalized_error_area_mpa'])}` / `{_format_metric(summary['validation']['best_nonadaptive_normalized_error_area_mpa'])}` MPa; directional gates main-beats-no-VLM=`{summary['validation']['main_beats_no_vlm']}`, main-beats-best-nonadaptive=`{summary['validation']['main_beats_best_nonadaptive']}`.
- Cached VLM region-cell counts by confidence: `{vlm_manifest['confidence_cell_counts']}`; no-reliable-cue fraction: `{float(vlm_manifest['no_reliable_cue_fraction']):.4f}`.
- Frozen CNN encoding took `{float(feature_manifest['encoder_elapsed_seconds']):.3f}` s for the registered cohort. Historical VLM latency and per-step Actor/predictor latency are recorded separately; none is presented as scanner acquisition time.
- Domain-level MAE and each domain's gap to the common full-input predictor are exported in `domain_cai_performance.csv`.

## Primary results

| Quantity | MAE (MPa) | RMSE (MPa) | R2 |
|---|---:|---:|---:|
| Main, zero C-scan | {_format_metric(main_zero['mae_mpa'])} | {_format_metric(main_zero['rmse_mpa'])} | {_format_metric(main_zero['r2'])} |
| Main, cost 0.25 | {_format_metric(main_endpoint['mae_mpa'])} | {_format_metric(main_endpoint['rmse_mpa'])} | {_format_metric(main_endpoint['r2'])} |
| Common predictor, full 64 cells | {_format_metric(full['mae_mpa'])} | {_format_metric(full['rmse_mpa'])} | {_format_metric(full['r2'])} |

Primary effects are comparator error minus main error; positive values favor the main method. Intervals are 98.333333% two-sided, 5000-replicate, paired hierarchical bootstrap intervals with equal-domain aggregation.

| Claim | Estimate (MPa) | 98.333333% CI | Evidence |
|---|---:|---:|---|
{effect_lines}

Engineering status remains `ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED`; no engineering-readiness threshold was invented.

At cost 0.25, the main MAE is `{_format_metric(main_endpoint['mae_mpa'])}` MPa: `{_format_metric(float(main_endpoint['mae_mpa']) - float(no_vlm_endpoint['mae_mpa']))}` MPa worse than `NO_VLM_FEEDBACK`, `{_format_metric(float(open_loop_endpoint['mae_mpa']) - float(main_endpoint['mae_mpa']))}` MPa better than `VLM_OPEN_LOOP`, and `{_format_metric(float(fixed_endpoint['mae_mpa']) - float(main_endpoint['mae_mpa']))}` MPa better than `{summary['best_nonadaptive']}`. These endpoint contrasts are descriptive and do not override the unsupported primary intervals.

## Compute and chronology

- Formal registered predictor updates: `{predictor_manifest['optimizer_updates']}`.
- Formal registered actor updates: `{actor_manifest['optimizer_updates']}`.
- Formal registered total: `{int(predictor_manifest['optimizer_updates']) + int(actor_manifest['optimizer_updates'])}` / 21,500.
- Measured encoder / Actor time: `{float(feature_manifest['encoder_elapsed_seconds']):.3f}` s / `{float(actor_manifest['elapsed_hours']) * 3600.0:.3f}` s. The exact formal predictor duration was not persisted, so no fabricated cumulative runtime is reported; the formal execution completed below the 12-hour limit.
- During implementation, an earlier 8,000-update predictor development execution was invalidated before Actor training because it omitted 32/64-cell states required for the registered full-input report. It was regenerated under the corrected pre-Actor protocol. This was a coverage correction, not outcome-driven hyperparameter or seed selection.
- A preceding 13,500-update Actor execution was invalidated during final contract audit because C0 combined medium/high candidates instead of retaining only the highest reliable confidence level. The retained Actor run follows `{protocol.initial_proposal_rule}`. No result was inspected to choose this correction.
- No VLM, ResNet, source image, CAI label, prior learned-C-scan result, or old scientific code was modified.
- No TEST target is stored in the training feature bank or trajectory log.

## Source bindings

| Source | Path | SHA-256 |
|---|---|---|
{source_lines}

## Generated figures

{figure_lines}

The three cases were fixed in the protocol before TEST evaluation; full C-scans are not shown in these acquisition figures.

## Verification

- Ruff: `{verification['checks'][0]['output']}`
- Tests: `{verification['checks'][1]['output']}`
- `git diff --check`: pass
- Frozen prior-result/shared-code diff from evidence base: empty
- Git push and local/upstream/remote SHA equality are recorded in the final Codex response after push.
"""
    protocol.artifacts_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = protocol.artifacts_dir / "CODEX_HANDOFF_CAI_VLM_GUIDED_AGENT_V2.md"
    handoff_path.write_text(handoff, encoding="utf-8")

    bindings = f"""# CAI Active Image v2 Source Bindings

Protocol: `{protocol.config_path.relative_to(root)}` (`{protocol.config_sha256}`)

| Source | Path | SHA-256 |
|---|---|---|
{source_lines}

External image root binding: `{EXTERNAL_SOURCE_ROOT_BINDING}`; supply its local path with `--source-root`.
The external ResNet18 weight copy was checked against the registered `{protocol.source('resnet18_weights').relative_to(root)}` digest before encoding.
"""
    bindings_path = protocol.artifacts_dir / "CAI_ACTIVE_IMAGE_SOURCE_BINDINGS_V2.md"
    bindings_path.write_text(bindings, encoding="utf-8")

    go_nogo = "# CAI VLM-Guided Agent v2 Status\n\n" + "\n".join(
        f"- {row['claim']}: {row['evidence_status']} "
        f"(effect {_format_metric(row['estimate_mpa'])} MPa; "
        f"98.333333% CI [{_format_metric(row['ci_lower_mpa'])}, {_format_metric(row['ci_upper_mpa'])}])"
        for row in effects
    )
    go_nogo += (
        "\n- CAI_ABSOLUTE_PERFORMANCE: REPORTED; "
        "ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED\n"
    )
    go_nogo_path = protocol.artifacts_dir / "GO_NOGO.md"
    go_nogo_path.write_text(go_nogo, encoding="utf-8")

    completion = f"""# CAI VLM-Guided Agent v2 Completion Audit

- [x] 60 frozen specimens retain TRAIN/VALID/TEST = 24/12/24.
- [x] Author CAI strength is evaluated in MPa.
- [x] All 24 TEST targets are absent from the training feature bank and joined only in P5 scoring.
- [x] Frozen Qwen cache is bound by model, prompt, schema, render, and image identities.
- [x] No second cell rotation is applied.
- [x] C0 uses only the highest reliable confidence level, constrains only the first main/open-loop action, and has a defined fallback.
- [x] Hidden C-scan tokens are masked from predictor and feedback actor.
- [x] `VLM_OPEN_LOOP` cannot read acquired C-scan content or current prediction.
- [x] `NO_VLM_FEEDBACK` cannot read VLM features or C0.
- [x] One common `P_all` predictor is used across all TEST methods.
- [x] Exact native-raster cost and absolute-zero state are logged.
- [x] Initial/legal action masks and per-step Actor/predictor latency are logged.
- [x] Actor state includes explicit spent/remaining cost and ordered action geometry history.
- [x] Formal optimizer schedule is 21,500 updates.
- [x] VALID selects `{summary['best_nonadaptive']}` as `BEST_NONADAPTIVE`.
- [x] Three Bonferroni-adjusted primary comparisons use 5,000 paired bootstrap replicates.
- [x] Absolute and domain-level CAI performance are exported.
- [x] Three cases were selected before TEST evaluation and exported.
- [x] Prior results and shared learned-C-scan/VLM/MAVIS code have an empty diff from base.
- [x] Scoped tests, Ruff, and `git diff --check` pass.

Resource disclosure: one 8,000-update development predictor execution was invalidated before Actor training due to missing full-input state coverage. A subsequent 13,500-update Actor execution was invalidated during final contract audit due to the broader-than-specified C0 rule. The formal retained run follows the corrected frozen 21,500-update schedule. Neither correction was result-driven.
"""
    completion_path = protocol.artifacts_dir / "CAI_VLM_GUIDED_AGENT_V2_COMPLETION_AUDIT.md"
    completion_path.write_text(completion, encoding="utf-8")

    readme = """# Start Here

Main method: `VLM_CAI_FEEDBACK_AGENT`, with frozen VLM-guided initialization and a feedback Actor that reselects after every acquired C-scan cell.

- Protocol: `paper_v3/configs/cai_active_image_v2.yaml`
- Results: `results/cai_active_image/v2/summary.json`
- Primary effects: `results/cai_active_image/v2/vlm_ablation_effects.csv`
- Absolute metrics: `results/cai_active_image/v2/absolute_cai_performance.csv`
- Handoff: `CODEX_HANDOFF_CAI_VLM_GUIDED_AGENT_V2.md`

Run a stage with `python scripts/run_cai_active_image_v2.py <stage> --source-root /path/to/cmc_damage_inference`.
"""
    readme_path = protocol.artifacts_dir / "README_START.md"
    readme_path.write_text(readme, encoding="utf-8")

    result_checksums = _write_checksums(protocol.results_dir)
    result_files = [path for path in protocol.results_dir.rglob("*") if path.is_file()]
    artifact_manifest = {
        "schema_version": 2,
        "status": "P7_HANDOFF_COMPLETE",
        "repository_base_sha": protocol.repository_base_sha,
        "protocol_sha256": protocol.config_sha256,
        "result_files": {
            path.relative_to(root).as_posix(): _sha256_file(path)
            for path in sorted(result_files)
        },
        "handoff": str(handoff_path.relative_to(root)),
        "completion_audit": str(completion_path.relative_to(root)),
        "scientific_integrity": {
            "old_scientific_paths_changed": False,
            "vlm_new_calls": 0,
            "formal_optimizer_updates": 21_500,
            "engineering_accuracy_criterion_specified": False,
            "test_targets_stored_in_feature_bank": False,
            "test_target_join_stage": "P5_SCORING_SIDE_ONLY",
            "initial_proposal_rule": protocol.initial_proposal_rule,
            "actor_state_protocol": protocol.actor_state_protocol,
        },
    }
    manifest_path = protocol.artifacts_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_checksums = _write_checksums(protocol.artifacts_dir)
    return {
        "status": "P7_HANDOFF_COMPLETE",
        "handoff": str(handoff_path),
        "completion_audit": str(completion_path),
        "go_nogo": str(go_nogo_path),
        "manifest": str(manifest_path),
        "result_checksums": str(result_checksums),
        "artifact_checksums": str(artifact_checksums),
        "figures": figures,
    }


__all__ = ["finalize_study"]
