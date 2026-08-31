from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl
import pytest

from cmc_bbdm.agentic_nde.p1_pipeline import (
    P1RunError,
    load_p1_score_freeze,
    write_p1_score_freeze,
)
from cmc_bbdm.agentic_nde.visual_observability import (
    FrozenOuterScores,
    VisualExamples,
    freeze_outer_scores,
)


def _inference() -> VisualExamples:
    return VisualExamples.create(
        outer_domain="f",
        role="outer_inference",
        specimen_ids=("s-1",),
        dataset_ids=("f",),
        initial_embeddings=np.zeros((1, 512)),
        current_predictions=np.zeros(1),
        candidate_features=np.zeros((1, 64, 8)),
        global_embeddings=np.zeros((1, 512), dtype=np.float32),
        local_embeddings=np.zeros((1, 64, 512), dtype=np.float32),
        mechanical_values=None,
        feature_control="correct_registration",
    )


def _frozen(inference: VisualExamples) -> FrozenOuterScores:
    return freeze_outer_scores(
        inference,
        scores=MappingProxyType(
            {
                "c0_mvd_m1_o2": -np.arange(64, dtype=np.float64)[None, :],
                "proposed": np.arange(64, dtype=np.float64)[None, :],
            }
        ),
        model_state_sha256=MappingProxyType(
            {"c0_mvd_m1_o2": "1" * 64, "proposed": "2" * 64}
        ),
        selection_state_sha256="3" * 64,
    )


def test_score_freeze_is_atomic_label_free_and_replayable(tmp_path: Path) -> None:
    inference = _inference()
    frozen = _frozen(inference)
    selection = pl.DataFrame(
        {
            "outer_domain": ["f"],
            "stage": ["FINAL_FIT"],
            "candidate_id": ["ridge_alpha_0.1"],
        }
    )
    path = write_p1_score_freeze(
        tmp_path / "freeze", frozen, selection_audit=selection
    )
    assert {item.name for item in path.iterdir()} == {
        "scores.parquet",
        "selection.csv",
        "manifest.json",
    }
    score_table = pl.read_parquet(path / "scores.parquet")
    assert "mechanical_value" not in score_table.columns
    assert "target" not in score_table.columns
    loaded, audit = load_p1_score_freeze(path, inference=inference)
    assert loaded.state_sha256 == frozen.state_sha256
    assert loaded.selection_state_sha256 == frozen.selection_state_sha256
    assert np.array_equal(loaded.scores["proposed"], frozen.scores["proposed"])
    assert audit.equals(selection)
    with pytest.raises(P1RunError, match="already exists"):
        write_p1_score_freeze(path, frozen, selection_audit=selection)


def test_score_freeze_rejects_tampered_scores(tmp_path: Path) -> None:
    inference = _inference()
    path = write_p1_score_freeze(
        tmp_path / "freeze",
        _frozen(inference),
        selection_audit=pl.DataFrame(
            {"outer_domain": ["f"], "stage": ["FINAL_FIT"]}
        ),
    )
    with (path / "scores.parquet").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(P1RunError, match="hash changed"):
        load_p1_score_freeze(path, inference=inference)
