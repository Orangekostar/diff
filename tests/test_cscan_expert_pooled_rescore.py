from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from cmc_bbdm.learned_cscan.expert_pooled_rescore import (
    ExpectedReference,
    ReferenceCandidate,
    classify_positive_effect,
    inherit_proxy_inputs,
    planner_absolute_rows,
    pool_reference_files,
    summarize_expert_rescore,
    validate_episode_invariants,
    verify_output_only_config,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "paper_v3/configs/bc_cscan_frozen_process_analysis.yaml"
DERIVED_CONFIG = ROOT / "paper_v3/configs/bc_cscan_expert_pooled_rescore_v1.yaml"


def _reference_payload(specimen_key: str, reviewer_alias: str) -> dict[str, object]:
    return {
        "specimen_key": specimen_key,
        "source_image_sha256": "a" * 64,
        "reference_type": "AUTHOR_PROVIDED",
        "review_state": "reviewed",
        "reviewer_alias": reviewer_alias,
        "frame": "registered_cscan",
        "regions": [
            {
                "certainty": "certain",
                "polygon": [[0.1, 0.1], [0.8, 0.1], [0.5, 0.8]],
            }
        ],
        "uncertain_regions": [],
        "notes": "confirmed",
    }


def _write_reference(path: Path, specimen_key: str, alias: str) -> bytes:
    raw = (
        json.dumps(_reference_payload(specimen_key, alias), indent=2) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def test_pool_references_preserves_bytes_polygons_and_aliases(tmp_path: Path) -> None:
    first = tmp_path / "exports/ww/session-1/references/reference-001.json"
    second = tmp_path / "exports/phl/session-2/references/reference-002.json"
    first_raw = _write_reference(first, "domain-a:s1", "ww")
    second_raw = _write_reference(second, "domain-b:s2", "phl")
    candidates = (
        ReferenceCandidate(first, "packet-1", "ww", "CONFIRMED"),
        ReferenceCandidate(second, "packet-2", "phl", "CONFIRMED"),
    )
    expected = {
        "domain-a:s1": ExpectedReference("domain-a", "a" * 64, (10, 12)),
        "domain-b:s2": ExpectedReference("domain-b", "a" * 64, (10, 12)),
    }

    result = pool_reference_files(candidates, tmp_path / "pooled", expected)

    assert result["physical_specimen_count"] == 2
    assert (tmp_path / "pooled/reference-001.json").read_bytes() == first_raw
    assert (tmp_path / "pooled/reference-002.json").read_bytes() == second_raw
    assert {row["reviewer_alias"] for row in result["provenance"]} == {
        "ww",
        "phl",
    }
    assert all(row["certain_pixel_count"] > 0 for row in result["provenance"])


@pytest.mark.parametrize("failure", ("duplicate", "unknown"))
def test_pool_references_rejects_duplicate_or_unknown_specimens(
    tmp_path: Path, failure: str
) -> None:
    first = tmp_path / "one/reference-001.json"
    second = tmp_path / "two/reference-002.json"
    _write_reference(first, "domain-a:s1", "ww")
    specimen_key = "domain-a:s1" if failure == "duplicate" else "domain-x:s9"
    _write_reference(second, specimen_key, "phl")
    candidates = (
        ReferenceCandidate(first, "packet-1", "ww", "CONFIRMED"),
        ReferenceCandidate(second, "packet-2", "phl", "CONFIRMED"),
    )
    expected = {"domain-a:s1": ExpectedReference("domain-a", "a" * 64, (10, 12))}

    with pytest.raises(ValueError, match=failure):
        pool_reference_files(candidates, tmp_path / "pooled", expected)


def test_derived_config_changes_only_output_paths() -> None:
    changed = verify_output_only_config(SOURCE_CONFIG, DERIVED_CONFIG)

    assert changed == {
        "paths.output_root": "results/bc_cscan_expert_pooled_rescore/v1",
        "paths.artifact_root": "artifacts/bc_cscan_expert_pooled_rescore/v1",
    }


def test_inherit_proxy_inputs_copies_only_three_registered_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy"
    source.mkdir()
    expected = {
        "analysis_summary.json": b"{}\n",
        "recovery_manifest.json": b'{"reference_version":"PROXY_LEGACY"}\n',
        "report_stability_and_delay.csv": b"task,value\nLOCATE,1\n",
    }
    for name, raw in expected.items():
        (source / name).write_bytes(raw)
    (source / "do_not_copy.csv").write_text("x\n", encoding="utf-8")

    manifest = inherit_proxy_inputs(source, tmp_path / "output")

    assert {path.name for path in (tmp_path / "output").iterdir()} == {
        *expected,
        "inherited_proxy_inputs.json",
    }
    assert all(
        (tmp_path / "output" / name).read_bytes() == raw
        for name, raw in expected.items()
    )
    assert len(manifest["files"]) == 3
    assert manifest["analysis_scope"] == "PROXY_LEGACY_CONTEXT_ONLY"


def test_planner_absolute_rows_do_not_double_count_stop_systems_or_seeds() -> None:
    rows = []
    values = {
        ("R_BALANCED_P8", 8): (0.2, 0.4),
        ("BC_S1", 1): (0.4, 0.6),
        ("BC_S2", 2): (0.6, 0.8),
        ("BC_S3", 3): (0.8, 1.0),
    }
    for (method, seed), per_specimen in values.items():
        for index, value in enumerate(per_specimen, start=1):
            for stop_system in ("S_BC_CAL", "S_RULE"):
                rows.append(
                    {
                        "dataset_id": f"domain-{index}",
                        "specimen_key": f"domain-{index}:s{index}",
                        "task": "LOCATE",
                        "method": method,
                        "seed": seed,
                        "stop_system": stop_system,
                        "planner_ausc": value,
                    }
                )

    summary = planner_absolute_rows(tuple(rows))
    by_method = {row["method"]: row for row in summary}

    assert by_method["R_BALANCED_P8"]["planner_ausc"] == pytest.approx(0.3)
    assert by_method["BC_S1"]["planner_ausc"] == pytest.approx(0.5)
    assert by_method["BC_3SEED"]["planner_ausc"] == pytest.approx(0.7)
    assert by_method["BC_3SEED"]["physical_specimen_count"] == 2


@pytest.mark.parametrize(
    ("lower", "physical_n", "expected", "status"),
    (
        (0.01, 24, 24, "SUPPORTED"),
        (0.0, 24, 24, "NOT_SUPPORTED"),
        (-0.1, 24, 24, "NOT_SUPPORTED"),
        (0.2, 23, 24, "INSUFFICIENT_PRECISION"),
    ),
)
def test_claim_support_is_computed_from_coverage_and_interval(
    lower: float, physical_n: int, expected: int, status: str
) -> None:
    assert classify_positive_effect(lower, physical_n, expected) == status


def test_episode_invariants_cover_stop_and_failure_cost_identities() -> None:
    rows = (
        {
            "completion": True,
            "false_stop": False,
            "exhausted": False,
            "autonomous_ausc": 0.75,
            "failure_penalized_cost": 0.25,
        },
        {
            "completion": False,
            "false_stop": True,
            "exhausted": False,
            "autonomous_ausc": 0.0,
            "failure_penalized_cost": 1.0,
        },
        {
            "completion": False,
            "false_stop": False,
            "exhausted": True,
            "autonomous_ausc": 0.0,
            "failure_penalized_cost": 1.0,
        },
    )

    assert validate_episode_invariants(rows) == {
        "episode_count": 3,
        "outcome_partition_valid": True,
        "cost_identity_valid": True,
    }


def test_summarize_entrypoint_has_no_recovery_call() -> None:
    source = inspect.getsource(summarize_expert_rescore)

    assert "execute_frozen_finalize" not in source
    assert "recover_reviewed_report_scores" not in source


def test_committed_planner_summary_matches_reviewed_effect() -> None:
    tables = ROOT / "results/bc_cscan_expert_pooled_rescore/v1"
    absolute = {
        (row["task"], row["method"]): float(row["planner_ausc"])
        for row in _csv_rows(tables / "tables/planner_absolute_means.csv")
    }
    effects = [
        row
        for row in _csv_rows(tables / "reviewed/planner_effects.csv")
        if float(row["confidence_level"]) == 0.975
    ]

    for row in effects:
        task = row["task"]
        assert float(row["estimate"]) == pytest.approx(
            absolute[(task, "BC_3SEED")] - absolute[(task, "R_BALANCED_P8")]
        )


def _csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))
