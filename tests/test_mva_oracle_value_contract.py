from types import SimpleNamespace

import numpy as np

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.interpolation import RefinementPatchCache
from cmc_bbdm.mva.measurement_state import budget_record, initial_state
from cmc_bbdm.mva.oracle_execution import _oracle_trajectory


class _Encoder:
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        output = np.zeros((len(images), 512), dtype=np.float64)
        output[:, 0] = [float(np.mean(image)) / 255.0 for image in images]
        return output


class _Predictor:
    state_sha256 = "a" * 64

    def predict(self, metadata: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        assert metadata.shape[0] == embeddings.shape[0]
        return embeddings[:, 0]


def test_mechanical_oracle_candidate_rows_preserve_full_transition_contract() -> None:
    image = np.arange(64 * 64 * 3, dtype=np.uint16).reshape(64, 64, 3)
    image = np.asarray(image % 256, dtype=np.uint8)
    authority = SimpleNamespace(
        images=(image,),
        specimen_ids=("specimen",),
        dataset_ids=("domain",),
        targets=np.asarray([1.0]),
        metadata13=np.zeros((1, 13), dtype=np.float64),
    )
    grid = build_acquisition_grid(64, 64, initial_budget=0.015625)

    oracle = _oracle_trajectory(
        method="mechanical_oracle",
        authority=authority,
        specimen_index=0,
        grid=grid,
        encoder=_Encoder(),
        p_a_model=_Predictor(),
        initial_embedding=np.zeros(512, dtype=np.float64),
        checkpoints=(0.023,),
        patch_cache=RefinementPatchCache(image=image, grid=grid),
    )

    required = {
        "budget_before",
        "candidate",
        "value",
        "budget_after",
        "current_prediction",
        "new_prediction",
        "current_error",
        "new_error",
    }
    assert oracle.values
    assert all(required <= row.keys() for row in oracle.values)
    initial_budget = budget_record(grid, initial_state(grid)).effective_budget
    assert all(row["budget_before"] == initial_budget for row in oracle.values)
    assert all(row["candidate"] == row["cell_index"] for row in oracle.values)
    assert all(row["value"] == row["primary_value"] for row in oracle.values)
    assert all(row["budget_after"] == row["effective_budget"] for row in oracle.values)
    assert all(row["current_prediction"] == 0.0 for row in oracle.values)
    assert all(row["new_prediction"] is not None for row in oracle.values)
    assert all(row["current_error"] == row["error_before"] for row in oracle.values)
    assert all(row["new_error"] == row["error_after"] for row in oracle.values)
