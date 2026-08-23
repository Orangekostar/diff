from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from cmc_bbdm.mva.a4_config import load_a4_config
from cmc_bbdm.mva.a4_report import A4ReportError, render_a4_report

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mva_a4_global_mask.yaml"
EFFECTS = (
    "uniform_minus_global_mechanical_auebc",
    "global_reconstruction_minus_global_mechanical_auebc",
    "global_appearance_minus_global_mechanical_auebc",
    "global_mechanical_minus_mechanical_oracle_auebc",
)
METHODS = (
    "uniform",
    "global_appearance_mask",
    "global_reconstruction_mask",
    "global_mechanical_mask",
    "mechanical_oracle",
    "random_median",
)


def _evidence(path: Path) -> None:
    config = load_a4_config(CONFIG, project_root=ROOT)
    path.mkdir()
    summary = {
        "global_mask_status": "MVA_A4_GLOBAL_GO",
        "a5_status": "MVA_A5_AUTHORIZED",
        "gate": {"relative_adaptive_gap": 0.07125},
    }
    (path / "summary.json").write_text(
        json.dumps(summary) + "\n", encoding="ascii"
    )
    pl.DataFrame(
        [
            {
                "effect_id": effect,
                "point_estimate": 0.004 + index * 0.001,
                "lower": 0.002 + index * 0.001,
                "upper": 0.006 + index * 0.001,
                "improved_domains": 6 - index % 2,
                "domain_effects": "[0.1,0.1,0.1,0.1,0.1,0.1]",
                "seed": config.bootstrap_seed,
                "resamples": config.bootstrap_resamples,
                "indices_sha256": "a" * 64,
            }
            for index, effect in enumerate(EFFECTS)
        ]
    ).write_csv(path / "bootstrap.csv")
    pl.DataFrame(
        [
            {
                "method": method,
                "auebc": 0.02 + index * 0.001,
                "b_2p5": 0.25,
                "b_5": 0.1875,
                "b_7p5": 0.125,
                "saving_vs_uniform_b5": 0.0 if index == 0 else 0.0625,
            }
            for index, method in enumerate(METHODS)
        ]
    ).write_csv(path / "budget_metrics.csv")
    pl.DataFrame(
        [
            {
                "outer_domain": outer,
                "method": method,
                "removed_domain": removed,
                "top1_agreement": True,
                "top10_overlap": 0.8,
                "spearman": 0.9,
                "rbo_p0_9": 0.7,
            }
            for outer in config.domain_order
            for method in config.methods
            for removed in config.domain_order
            if removed != outer
        ]
    ).write_csv(path / "ranking_stability.csv")


def test_a4_report_is_derived_and_path_private(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _evidence(evidence)

    report = render_a4_report(
        evidence,
        config_path=CONFIG,
        project_root=ROOT,
    )
    text = report.read_text(encoding="utf-8")

    assert "MVA_A4_GLOBAL_GO" in text
    assert "MVA_A5_AUTHORIZED" in text
    assert "7.125%" in text
    assert "0.004000" in text
    assert "source-only" in text
    assert str(ROOT) not in text


def test_a4_report_rejects_incomplete_bootstrap_roster(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _evidence(evidence)
    bootstrap = pl.read_csv(evidence / "bootstrap.csv").slice(1)
    bootstrap.write_csv(evidence / "bootstrap.csv")

    with pytest.raises(A4ReportError, match="bootstrap roster"):
        render_a4_report(
            evidence,
            config_path=CONFIG,
            project_root=ROOT,
        )
