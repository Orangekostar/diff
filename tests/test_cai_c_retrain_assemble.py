from __future__ import annotations

import copy
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import ClassVar

import pytest

CONTROL_METHODS = (
    "CENTER_FIRST",
    "GEOMETRY_SPREAD",
    "SERPENTINE",
    "RANDOM",
    "LEARNED_STATIC_TRUE",
    "NO_VLM_SPATIAL_FEEDBACK",
)
C_METHODS = (
    "VLM_SPATIAL_FEEDBACK",
    "VLM_SPATIAL_OPEN_LOOP",
    "VLM_MEAN_FEEDBACK",
)


def _row(method: str, specimen: int, run: int = 0) -> dict[str, str]:
    return {
        "specimen_key": f"d{specimen % 6}:s{specimen}",
        "dataset_id": f"d{specimen % 6}",
        "capture_group_id": f"g{specimen % 48}",
        "method": method,
        "run": str(run),
        "target_mpa": str(100.0 + specimen),
        "payload": f"{method}:{specimen}:{run}",
    }


def _inputs() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    old: list[dict[str, str]] = []
    for method in CONTROL_METHODS:
        repeats = 5 if method == "RANDOM" else 1
        for specimen in range(50):
            for run in range(repeats):
                old.append(_row(method, specimen, run))
    for method in C_METHODS:
        old.extend(_row(method, specimen) for specimen in range(50))
    new = []
    for method in C_METHODS:
        new.extend(_row(method, specimen) for specimen in range(50))
    return old, new


def _write_gzip_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_assemble_rows_freezes_500_controls_and_adds_150_c_rows():
    from scripts.cai_c_retrain.assemble import assemble_rows

    old_rows, new_rows = _inputs()
    old_before = copy.deepcopy(old_rows)
    primary, provenance, historical = assemble_rows(old_rows, new_rows)

    assert len(primary) == 650
    assert len(provenance) == 650
    assert len(historical) == 150
    assert Counter(row["method"] for row in primary) == Counter(
        {
            "CENTER_FIRST": 50,
            "GEOMETRY_SPREAD": 50,
            "SERPENTINE": 50,
            "RANDOM": 250,
            "LEARNED_STATIC_TRUE": 50,
            "NO_VLM_SPATIAL_FEEDBACK": 50,
            "VLM_SPATIAL_FEEDBACK": 50,
            "VLM_SPATIAL_OPEN_LOOP": 50,
            "VLM_MEAN_FEEDBACK": 50,
        }
    )
    assert all(row["method"] not in C_METHODS for row in primary[:500])
    assert all(row["method"] in C_METHODS for row in primary[500:])
    assert all(row["method"] in C_METHODS for row in historical)
    assert all("provenance" not in row for row in primary)
    assert old_rows == old_before
    assert provenance[0]["provenance"] == "REUSED_FROZEN_CONTROL"
    assert provenance[499]["prior_version"] == "UNCHANGED_CONTROL"
    assert provenance[500]["provenance"] == "NEW_C_RETRAIN"
    assert provenance[500]["prior_version"] == "C_P0_R1_GLOBAL_V1"


def test_assemble_rows_rejects_missing_control_row():
    from scripts.cai_c_retrain.assemble import assemble_rows

    old_rows, new_rows = _inputs()
    old_rows.pop(0)
    with pytest.raises(ValueError, match="frozen controls method counts"):
        assemble_rows(old_rows, new_rows)


def test_assemble_rows_rejects_target_drift():
    from scripts.cai_c_retrain.assemble import assemble_rows

    old_rows, new_rows = _inputs()
    new_rows[0]["target_mpa"] = "999.0"
    with pytest.raises(ValueError, match="target_mpa drift"):
        assemble_rows(old_rows, new_rows)


def test_historical_a_rows_are_not_entered_into_primary_matrix():
    from scripts.cai_c_retrain.assemble import assemble_rows

    old_rows, new_rows = _inputs()
    primary, _, historical = assemble_rows(old_rows, new_rows)
    primary_methods = {row["method"] for row in primary}
    assert primary_methods == set(CONTROL_METHODS) | set(C_METHODS)
    assert {row["method"] for row in historical} == set(C_METHODS)
    assert (
        len([row for row in primary if row["method"] == "VLM_SPATIAL_FEEDBACK"]) == 50
    )


def test_assemble_stage_writes_atomic_outputs_and_manifest(tmp_path):
    from scripts.cai_c_retrain.assemble import assemble_stage, read_episodes
    from scripts.cai_c_retrain.context import canonical_json, sha256_bytes, sha256_file

    old_rows, new_rows = _inputs()
    old_root = tmp_path / "old_w3"
    old_artifacts = tmp_path / "old_artifacts"
    data = tmp_path / "data"
    vlm = tmp_path / "vlm"
    w3 = tmp_path / "w3"
    old_path = old_root / "policy_validation_episodes.csv.gz"
    new_path = w3 / "new_selected_policy_episodes.csv.gz"
    _write_gzip_csv(old_path, old_rows)
    _write_gzip_csv(new_path, new_rows)
    data.mkdir()
    vlm.mkdir()
    files = {}
    for key, name in (
        ("feature_bank_manifest", "feature_bank_manifest.json"),
        ("feature_bank_index", "feature_bank_index.csv"),
        ("oof_fold_manifest", "oof_fold_manifest.csv"),
        ("split_manifest", "split_manifest.csv"),
        ("candidate_queue", "candidate_queue.csv"),
        ("cost_precision_audit", "cost_precision_audit.json"),
        ("feature_bank_shards", "feature_bank_shards.csv"),
    ):
        path = data / name
        path.write_text(key, encoding="utf-8")
        files[key] = sha256_file(path)
    shard_rows = []
    for index in range(6):
        path = data / f"shard_{index}.npz"
        path.write_bytes(f"shard-{index}".encode())
        digest = sha256_file(path)
        files[f"feature_shard_d{index}"] = digest
        shard_rows.append(
            {
                "dataset_id": f"d{index}",
                "shard_path": str(path.relative_to(tmp_path)),
                "sha256": digest,
            }
        )
    vlm_manifest = vlm / "vlm_manifest_fit.json"
    vlm_features = vlm / "vlm_actor_features_fit.csv"
    vlm_manifest.write_text("{}", encoding="utf-8")
    vlm_features.write_text("features", encoding="utf-8")
    files["vlm_manifest"] = sha256_file(vlm_manifest)
    files["vlm_features"] = sha256_file(vlm_features)
    binding = {"files": files}
    (w3 / "input_bindings.json").write_text(json.dumps(binding), encoding="utf-8")
    input_signature = sha256_bytes(canonical_json(binding))
    predictor = tmp_path / "predictor.pt"
    predictor.write_bytes(b"predictor")
    predictor_relative = str(predictor.relative_to(tmp_path))
    old_artifacts.mkdir()
    old_binding = {
        "data_metadata": {
            name: files[key]
            for name, key in (
                ("feature_bank_manifest.json", "feature_bank_manifest"),
                ("feature_bank_index.csv", "feature_bank_index"),
                ("oof_fold_manifest.csv", "oof_fold_manifest"),
                ("split_manifest.csv", "split_manifest"),
            )
        },
        "feature_shards": shard_rows,
        "predictors": [
            {
                "checkpoint_path": predictor_relative,
                "checkpoint_sha256": sha256_file(predictor),
            }
        ],
    }
    (old_artifacts / "INPUT_BINDINGS.json").write_text(
        json.dumps(old_binding), encoding="utf-8"
    )
    (w3 / "actor_manifests.json").parent.mkdir(parents=True, exist_ok=True)
    (w3 / "actor_manifests.json").write_text(
        json.dumps(
            {
                "status": "THREE_C_ACTORS_COMPLETE",
                "selected_episode_count": 150,
                "actor_manifests": [
                    {"input_signature": input_signature} for _ in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )

    class DummyContext:
        root = tmp_path
        task_id = "task"
        scope: ClassVar[dict[str, object]] = {
            "prior_version": "C_P0_R1_GLOBAL_V1",
            "predictors": {
                "common_checkpoint": predictor_relative,
                "common_checkpoint_sha256": sha256_file(predictor),
            },
        }

        def path(self, name: str) -> Path:
            return {
                "old_w3": old_root,
                "old_w3_artifacts": old_artifacts,
                "data": data,
                "vlm": vlm,
                "w3": w3,
            }[name]

        def transition(self, phase: str, status: str, **details: object) -> None:
            self.transitioned = (phase, status, details)

    context = DummyContext()
    manifest = assemble_stage(context)
    assert manifest["status"] == "PRIMARY_MATRIX_COMPLETE"
    assert manifest["primary_count"] == 650
    assert len(read_episodes(w3 / "policy_validation_episodes.csv.gz")) == 650
    assert (
        len(
            read_episodes(
                w3 / "historical_A_comparison/policy_validation_episodes.csv.gz"
            )
        )
        == 150
    )
    assert context.transitioned[0:2] == ("assemble", "COMPLETE")
    assert not list(w3.rglob("*.tmp"))
