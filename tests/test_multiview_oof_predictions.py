from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cmc_bbdm.aei_multiview_regression.oof_predictions import (
    evaluate_independent_views,
    load_authoritative_inputs,
)
from cmc_bbdm.aei_multiview_regression.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/aei_multiview_regression.yaml"


def _frozen_full_predictions() -> dict[str, float]:
    path = ROOT / "paper_v3/experiments/P1_full_field_oracle/predictions.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, strict=True)
        return {
            row["specimen_id"]: float(row["prediction"])
            for row in rows
            if row["method"] == "I_frozen"
        }


def test_full_expert_reproduces_frozen_p1_and_all_fits_exclude_queries() -> None:
    protocol = load_protocol(CONFIG, project_root=ROOT)
    inputs = load_authoritative_inputs(protocol, project_root=ROOT)
    events = []

    result = evaluate_independent_views(
        inputs, protocol=protocol, fit_hook=events.append
    )
    frozen = _frozen_full_predictions()
    expected = np.asarray([frozen[item] for item in result.specimen_ids])

    assert np.max(np.abs(result.predictions[:, 0] - expected)) <= 1e-12
    assert result.predictions.shape == (276, 3)
    assert result.predictions.flags.writeable is False
    assert result.cai_strength_mpa.flags.writeable is False
    assert result.intact_strength_mpa.flags.writeable is False
    np.testing.assert_allclose(
        result.cai_strength_mpa,
        result.targets * result.intact_strength_mpa,
        rtol=0.0,
        atol=1e-10,
    )
    assert np.all(np.isfinite(result.predictions))
    assert len(result.selections) == 18
    assert len(result.source_oof) == 6
    assert events
    for event in events:
        assert set(event.fit_ids).isdisjoint(event.query_ids)
        assert event.outer_domain not in event.fit_domains
        if event.inner_domain is not None:
            assert event.inner_domain not in event.fit_domains
