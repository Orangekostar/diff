from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import cmc_bbdm.cpb_diffusion_marginalization.artifacts as artifact_module
from cmc_bbdm.cpb_diffusion_marginalization.artifacts import (
    D8ArtifactError,
    build_d8_search_package,
    recover_interrupted_d8_publication,
    validate_d8_search_package,
)
from cmc_bbdm.cpb_diffusion_marginalization.authority import issue_search_view
from cmc_bbdm.cpb_diffusion_marginalization.config import DOMAIN_ORDER, load_d8_config
from cmc_bbdm.cpb_diffusion_marginalization.pilot import (
    D8PilotStudyEvidence,
    decide_pilot_escalation,
)
from cmc_bbdm.cpb_diffusion_marginalization.regression import (
    CandidatePrediction,
    CandidateSpec,
)
from cmc_bbdm.cpb_diffusion_marginalization.search import (
    D8Candidate,
    robust_inner_objective,
)
from cmc_bbdm.cpb_diffusion_marginalization.selection import (
    evaluate_finalists,
    fit_nonnegative_ensemble,
    freeze_outer_selection,
    rerank_candidates,
)
from cmc_bbdm.cpb_diffusion_marginalization.tracking import TRIAL_INDEX_FIELDS
from cmc_bbdm.cpb_diffusion_marginalization.variants import MorphologyThresholds
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_exploration.yaml"
_HASH = "1" * 64
_STATUS = "CLOSE_DIFFUSION_SPECIFIC_ROUTE"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _selection_candidate(config, index: int) -> D8Candidate:
    control = "B0" if index == 0 else "B5"
    return D8Candidate(
        control_id=control,
        decomposition_family="gaussian",
        band="low" if control == "B0" else "high",
        decomposition_parameters={"sigma": 1.0 + 0.1 * index},
        alpha=0.0 if control == "B0" else 0.05 + 0.01 * index,
        K_train=1,
        K_test=1,
        thresholds=MorphologyThresholds(
            area_relative_deviation=0.10,
            width_relative_deviation=0.10,
            height_relative_deviation=0.10,
            centroid_shift_mm=2.0,
            low_frequency_correlation_minimum=0.95,
            radial_spearman_minimum=0.90,
        ),
        feature_layer="global",
        feature_aggregation="mean",
        prediction_aggregation="mean",
        morphology_beta=None,
        consistency="none",
        consistency_weight=0.0,
        regressor_spec=CandidateSpec(
            pca_dimension=8,
            regressor="ridge",
            parameters={"alpha": 10.0},
            seed=config.seed + index,
        ),
        seed=config.seed + index,
        config_sha256=config.config_sha256,
    )


def _selection_prediction(candidate, fold) -> CandidatePrediction:
    indices = np.asarray(fold.query_indices, dtype=np.int64)
    targets = np.asarray(fold.search_view.data_view.cai_ratio[indices], dtype=np.float64)
    candidate_index = round((candidate.alpha - 0.05) / 0.01)
    predictions = targets + (0.03 - 0.001 * candidate_index)
    fitted = hashlib.sha256(
        candidate.state_sha256.encode("ascii") + fold.state_sha256.encode("ascii")
    ).hexdigest()
    state = hashlib.sha256(
        fitted.encode("ascii") + targets.tobytes() + predictions.tobytes()
    ).hexdigest()
    return CandidatePrediction(
        fit_specimen_ids=fold.fit_specimen_ids,
        query_specimen_ids=fold.query_specimen_ids,
        targets=targets,
        predictions=predictions,
        fit_state_sha256=fitted,
        state_sha256=state,
    )


@lru_cache(maxsize=1)
def _registered_selection_documents() -> tuple[
    dict[str, dict[str, object]], tuple[D8Candidate, ...]
]:
    config = load_d8_config(CONFIG, project_root=PROJECT_ROOT)
    p1 = load_v3_config(
        PROJECT_ROOT / config.sources["p1_config"].path,
        project_root=PROJECT_ROOT,
    )
    data = load_data(p1, PROJECT_ROOT)
    candidates = tuple(_selection_candidate(config, index) for index in range(12))
    documents: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory() as directory:
        for outer in DOMAIN_ORDER:
            view = issue_search_view(data, outer_domain=outer, config=config)
            reranked = rerank_candidates(
                candidates,
                view=view,
                seeds=config.rerank_seeds,
                evaluator=_selection_prediction,
            )
            finalists = evaluate_finalists(
                tuple(row.candidate for row in reranked.finalists),
                view=view,
                seeds=config.rerank_seeds,
                K_test_values=(8, 16),
                evaluator=_selection_prediction,
            )
            ensemble = fit_nonnegative_ensemble(
                np.vstack([row.oof_predictions for row in finalists.selected]),
                finalists.selected[0].oof_targets,
                specimen_ids=view.specimen_ids,
                domain_ids=view.dataset_ids,
                candidate_sha256=tuple(
                    row.candidate.state_sha256 for row in finalists.selected
                ),
                minimum_j_gain=1.0e-4,
            )
            output = Path(directory) / f"{outer}.json"
            freeze_outer_selection(
                reranked,
                finalists=finalists,
                ensemble=ensemble,
                view=view,
                output=output,
            )
            documents[outer] = json.loads(output.read_text(encoding="ascii"))
    return documents, candidates


def _write_source(root: Path) -> Path:
    documents, registered_candidates = _registered_selection_documents()
    config = load_d8_config(CONFIG, project_root=PROJECT_ROOT)
    source = root / "source"
    source.mkdir()
    config_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    escalation_studies: list[D8PilotStudyEvidence] = []
    database = sqlite3.connect(source / "study.db")
    database.executescript(
        "CREATE TABLE studies (study_id INTEGER PRIMARY KEY, study_name TEXT NOT NULL);"
        "CREATE TABLE trials (trial_id INTEGER PRIMARY KEY, number INTEGER, "
        "study_id INTEGER, state TEXT NOT NULL, datetime_start TEXT, "
        "datetime_complete TEXT);"
    )
    trial_id = 0
    for study_id, outer in enumerate(DOMAIN_ORDER, start=1):
        selection_document = documents[outer]
        search_view_sha256 = str(selection_document["search_view_sha256"])
        study_name = f"d8::{outer}"
        database.execute(
            "INSERT INTO studies(study_id, study_name) VALUES (?, ?)",
            (study_id, study_name),
        )
        best_objective = math.inf
        best_hash = ""
        for number in range(72):
            inner = tuple(domain for domain in DOMAIN_ORDER if domain != outer)
            values = np.asarray(
                [0.080 + 0.001 * DOMAIN_ORDER.index(domain) + number * 1.0e-6 for domain in inner],
                dtype=np.float64,
            )
            objective = robust_inner_objective(values)
            candidate_hash = (
                registered_candidates[number].state_sha256
                if number < len(registered_candidates)
                else hashlib.sha256(f"{outer}:{number}".encode("ascii")).hexdigest()
            )
            if objective < best_objective:
                best_objective = objective
                best_hash = candidate_hash
            row: dict[str, object] = {field: "" for field in TRIAL_INDEX_FIELDS}
            row.update(
                {
                    "study_name": study_name,
                    "trial_id": number,
                    "outer_fold": outer,
                    "state": "COMPLETE",
                    "mean_mae": math.fsum(values.tolist()) / 5,
                    "worst_mae": float(np.max(values)),
                    "domain_sd": float(np.std(values)),
                    "objective": objective,
                    "accepted_proposals": 50,
                    "proposed_variants": 50,
                    "acceptance_rate": 1.0,
                    "acceptance_by_domain": json.dumps(
                        {
                            domain: {
                                "accepted_proposals": 10,
                                "proposed_variants": 10,
                                "acceptance_rate": 1.0,
                            }
                            for domain in inner
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "control_id": (
                        registered_candidates[number].control_id
                        if number < len(registered_candidates)
                        else "B5"
                    ),
                    "decomposition_family": "gaussian",
                    "band": "high",
                    "decomposition_parameters": '{"sigma":2.0}',
                    "alpha": 0.1,
                    "K_train": 4,
                    "K_test": 1,
                    "feature_layer": "global",
                    "feature_aggregation": "mean",
                    "prediction_aggregation": "mean",
                    "consistency": "none",
                    "consistency_weight": 0.0,
                    "pca_dimension": 8,
                    "regressor": "ridge",
                    "regressor_parameters": '{"alpha":10.0}',
                    "seed": 20260820 + number,
                    "runtime_seconds": 0.01,
                    "config_sha256": config_sha,
                    "search_view_sha256": search_view_sha256,
                    "candidate_sha256": candidate_hash,
                    "evidence_sha256": _HASH,
                }
            )
            for domain, value in zip(inner, values, strict=True):
                row[f"inner_mae__{domain}"] = value
            rows.append(row)
            trial_id += 1
            database.execute(
                "INSERT INTO trials(trial_id, number, study_id, state) VALUES (?, ?, ?, ?)",
                (trial_id, number, study_id, "COMPLETE"),
            )
        selection_state = str(selection_document["state_sha256"])
        escalation_study = D8PilotStudyEvidence(
            outer_domain=outer,
            baseline_candidate_sha256=registered_candidates[0].state_sha256,
            diffusion_candidate_sha256=registered_candidates[1].state_sha256,
            baseline_objective=robust_inner_objective(
                np.asarray(
                    [
                        0.080 + 0.001 * DOMAIN_ORDER.index(domain)
                        for domain in inner
                    ],
                    dtype=np.float64,
                )
            ),
            diffusion_objective=robust_inner_objective(
                np.asarray(
                    [
                        0.080 + 0.001 * DOMAIN_ORDER.index(domain) + 1.0e-6
                        for domain in inner
                    ],
                    dtype=np.float64,
                )
            ),
            improved_inner_domains=(),
            low_band_energy_fraction=0.49,
            maximum_alpha_point_one_acceptance=1.0,
            selected_diffusion_weight=1.0,
            selection_state_sha256=selection_state,
            residual_bank_sha256=_HASH,
        )
        escalation_studies.append(escalation_study)
        summaries.append(
            {
                "outer_domain": outer,
                "initial_trial_count": 72,
                "trial_count": 72,
                "completed_count": 72,
                "pruned_count": 0,
                "failed_count": 0,
                "best_objective": best_objective,
                "best_candidate_sha256": best_hash,
                "selection_state_sha256": selection_state,
                "escalation_evidence_sha256": escalation_study.state_sha256,
            }
        )
        selections.append(selection_document)
    database.commit()
    database.close()
    with (source / "trial_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRIAL_INDEX_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    summary_fields = tuple(summaries[0])
    with (source / "search_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_fields))
        writer.writeheader()
        writer.writerows(summaries)
    selection_payload = {
        "schema_version": 2,
        "scope": "d8_prospective_outer_selections",
        "config_sha256": config_sha,
        "outer_evaluation_count": 0,
        "selections": selections,
        "state_sha256": _canonical_hash(selections),
    }
    (source / "selected_configs.json").write_text(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    decision = decide_pilot_escalation(tuple(escalation_studies), config=config)
    assert decision.decision == _STATUS
    (source / "escalation_evidence.json").write_text(
        json.dumps(
            decision.to_payload(), sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="ascii",
    )
    residual_payload = {
        "schema_version": 1,
        "scope": "d8_cross_fitted_p6_residual_bank",
        "specimen_count": 276,
        "draw_count": 8,
        "record_count": 2208,
        "maximum_mean_error": 0.0,
        "maximum_variance_error": 0.0,
        "state_sha256": _HASH,
    }
    (source / "residual_bank_manifest.json").write_text(
        json.dumps(residual_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    (source / "pilot_report.md").write_text(
        f"# D8 Pilot\n\nDecision: `{_STATUS}`\n\nOuter evaluations: `0`.\n",
        encoding="ascii",
    )
    return source


def test_pilot_package_contains_required_tracking_and_selections(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    output = tmp_path / "d8_search"
    built = build_d8_search_package(
        output,
        source_dir=source,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
        escalation_status=_STATUS,
    )
    validated = validate_d8_search_package(
        output, project_root=PROJECT_ROOT, config_path=CONFIG
    )
    assert built == validated
    assert validated.outer_domains == DOMAIN_ORDER
    assert validated.initial_trial_count == 72 * 6
    assert validated.trial_count == 72 * 6
    assert validated.outer_evaluation_count == 0
    assert validated.escalation_status == _STATUS
    assert {
        "trial_index.csv",
        "study.db",
        "residual_bank_manifest.json",
        "search_summary.csv",
        "selected_configs.json",
        "escalation_evidence.json",
        "pilot_report.md",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    } == validated.required_files
    with (output / "search_summary.csv").open(encoding="utf-8", newline="") as handle:
        raw_best = tuple(row["best_candidate_sha256"] for row in csv.DictReader(handle))
    selections = json.loads((output / "selected_configs.json").read_text())[
        "selections"
    ]
    assert all(
        best
        not in {
            candidate["state_sha256"]
            for candidate in selection["selected_candidates"]
        }
        for best, selection in zip(raw_best, selections, strict=True)
    )


def test_package_validation_recomputes_trial_objective_and_hashes(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    output = tmp_path / "d8_search"
    build_d8_search_package(
        output,
        source_dir=source,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
        escalation_status=_STATUS,
    )
    trial = output / "trial_index.csv"
    trial.write_text(trial.read_text(encoding="utf-8").replace("0.101", "0.102", 1))
    with pytest.raises(D8ArtifactError, match="hash|objective|trial"):
        validate_d8_search_package(
            output, project_root=PROJECT_ROOT, config_path=CONFIG
        )


def test_package_rejects_tampered_morphology_acceptance_counts(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    trial_index = source / "trial_index.csv"
    with trial_index.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, strict=True))
    rows[0]["accepted_proposals"] = "49"
    with trial_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRIAL_INDEX_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(D8ArtifactError, match="acceptance"):
        build_d8_search_package(
            tmp_path / "d8_search",
            source_dir=source,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
            escalation_status=_STATUS,
        )


def test_package_rejects_tampered_pruned_acceptance_counts(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    trial_index = source / "trial_index.csv"
    with trial_index.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, strict=True))
    row = rows[71]
    inner = tuple(domain for domain in DOMAIN_ORDER if domain != DOMAIN_ORDER[0])
    row["state"] = "PRUNED"
    row["failure_reason"] = f"morphology_acceptance:{inner[0]}"
    row["objective"] = ""
    row["accepted_proposals"] = "44"
    row["proposed_variants"] = "50"
    row["acceptance_rate"] = "0.9"
    row["acceptance_by_domain"] = json.dumps(
        {
            domain: {
                "accepted_proposals": 5 if index == 0 else 10,
                "proposed_variants": 10,
                "acceptance_rate": 0.5 if index == 0 else 1.0,
            }
            for index, domain in enumerate(inner)
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    with trial_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRIAL_INDEX_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    with sqlite3.connect(source / "study.db") as database:
        database.execute(
            "UPDATE trials SET state='PRUNED' WHERE study_id=1 AND number=71"
        )
    summary_path = source / "search_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summaries = list(csv.DictReader(handle, strict=True))
    summaries[0]["completed_count"] = "71"
    summaries[0]["pruned_count"] = "1"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with pytest.raises(D8ArtifactError, match="acceptance"):
        build_d8_search_package(
            tmp_path / "d8_search",
            source_dir=source,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
            escalation_status=_STATUS,
        )


def test_package_rejects_tampered_rerank_oof_even_with_root_state_rehashed(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    path = source / "selected_configs.json"
    document = json.loads(path.read_text(encoding="ascii"))
    selection = document["selections"][0]
    selection["rerank"]["rows"][0]["seeds"][0]["oof_predictions"][0] += 0.01
    selection_without_state = dict(selection)
    selection_without_state.pop("state_sha256")
    selection["state_sha256"] = _canonical_hash(selection_without_state)
    document["state_sha256"] = _canonical_hash(document["selections"])
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(D8ArtifactError, match="selection|configuration|evidence"):
        build_d8_search_package(
            tmp_path / "d8_search",
            source_dir=source,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
            escalation_status=_STATUS,
        )


def test_package_rejects_synchronized_escalation_trend_tampering(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    path = source / "escalation_evidence.json"
    payload = json.loads(path.read_text(encoding="ascii"))
    studies: list[D8PilotStudyEvidence] = []
    for index, value in enumerate(payload["studies"]):
        study = dict(value)
        study.pop("state_sha256")
        study["improved_inner_domains"] = tuple(study["improved_inner_domains"])
        if index == 0:
            study["improved_inner_domains"] = (DOMAIN_ORDER[1],)
        studies.append(D8PilotStudyEvidence(**study))
    config = load_d8_config(CONFIG, project_root=PROJECT_ROOT)
    decision = decide_pilot_escalation(tuple(studies), config=config)
    assert decision.decision == _STATUS
    path.write_text(
        json.dumps(decision.to_payload(), sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="ascii",
    )
    summary_path = source / "search_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summaries = list(csv.DictReader(handle, strict=True))
    summaries[0]["escalation_evidence_sha256"] = studies[0].state_sha256
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with pytest.raises(D8ArtifactError, match="escalation"):
        build_d8_search_package(
            tmp_path / "d8_search",
            source_dir=source,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
            escalation_status=_STATUS,
        )


def test_double_rename_failure_preserves_previous_for_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(tmp_path)
    output = tmp_path / "d8_search"
    build_d8_search_package(
        output,
        source_dir=source,
        project_root=PROJECT_ROOT,
        config_path=CONFIG,
        escalation_status=_STATUS,
    )
    before = (output / "artifact_manifest.json").read_bytes()
    real_replace = artifact_module._atomic_replace

    def fail_commit_and_rollback(source_path: Path, target_path: Path) -> None:
        if source_path.name in {"staged", "previous"} and target_path == output:
            raise OSError("injected rename failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(artifact_module, "_atomic_replace", fail_commit_and_rollback)
    with pytest.raises(OSError, match="rename failure"):
        artifact_module._publish_built_package(
            source,
            output,
            project_root=PROJECT_ROOT,
            config_path=CONFIG,
            escalation_status=_STATUS,
        )
    transactions = tuple(tmp_path.glob(".d8_search.transaction-*"))
    assert len(transactions) == 1
    assert (transactions[0] / "previous" / "artifact_manifest.json").read_bytes() == before
    monkeypatch.setattr(artifact_module, "_atomic_replace", real_replace)
    recovered = recover_interrupted_d8_publication(
        output, project_root=PROJECT_ROOT, config_path=CONFIG
    )
    assert recovered.scientific_digest
    assert not tuple(tmp_path.glob(".d8_search.transaction-*"))
