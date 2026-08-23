from __future__ import annotations

import hashlib
import inspect

import numpy as np

from cmc_bbdm.mva.a5_deployment import (
    run_deployable_trajectory,
    select_deployable_action,
)
from cmc_bbdm.mva.measurement_state import measurement_mask
from cmc_bbdm.mva.ranking_policy import RankingExample, train_ranking_policy


class _FixtureEncoder:
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        output = np.zeros((len(images), 512), dtype=np.float64)
        for index, image in enumerate(images):
            output[index] = np.resize(image.astype(np.float64).ravel(), 512) / 255.0
        return output

    def validate(self) -> None:
        return None


class _FixturePredictor:
    fit_domains = ("d1", "d2", "d3", "d4", "d5")
    state_sha256 = hashlib.sha256(b"p-a").hexdigest()

    def predict(self, metadata: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        return 0.6 + 0.01 * metadata[:, 0] + 0.001 * embeddings[:, 0]


def _image() -> np.ndarray:
    rows, columns = np.indices((41, 43))
    return np.stack(
        (
            (7 * rows + 3 * columns) % 256,
            (2 * rows + 11 * columns) % 256,
            (13 * rows + 5 * columns) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def _policy():
    rng = np.random.default_rng(11)
    examples = tuple(
        RankingExample(
            dataset_id=f"d{1 + index // 2}",
            specimen_id=f"s{index}",
            global_features=rng.normal(size=579),
            candidate_features=rng.normal(size=(4, 8)),
            selected_index=index % 4,
        )
        for index in range(6)
    )
    return train_ranking_policy(examples, seed=7, epochs=1, batch_states=3)


def test_deployable_selector_api_cannot_receive_forbidden_evidence() -> None:
    parameters = set(inspect.signature(select_deployable_action).parameters)

    assert not parameters & {
        "target",
        "true_cai",
        "full_image",
        "source_image",
        "oracle_values",
        "unmeasured_pixels",
    }


def test_deployable_methods_restore_measurements_and_respect_budgets() -> None:
    image = _image()
    for method in (
        "center_first",
        "observed_gradient",
        "observed_uncertainty",
        "imitation_policy",
    ):
        result = run_deployable_trajectory(
            specimen_id="target-0",
            dataset_id="d0",
            image=image,
            metadata=np.asarray([1.0, 2.0, 3.0]),
            initial_budget=0.03125,
            checkpoints=(0.0625, 0.125),
            predictor=_FixturePredictor(),
            encoder=_FixtureEncoder(),
            method=method,
            policy=_policy() if method == "imitation_policy" else None,
        )

        budgets = [snapshot.effective_budget for snapshot in result.snapshots]
        assert budgets == sorted(budgets)
        assert len(result.snapshots) == 2
        assert result.actions
        for snapshot in result.snapshots:
            mask = measurement_mask(result.grid, snapshot.state)
            assert np.array_equal(snapshot.image[mask], image[mask])
            assert snapshot.effective_budget <= snapshot.checkpoint
        for action in result.actions:
            assert action.budget_after > action.budget_before
            assert action.from_level + 1 == action.to_level


def test_deployable_trajectory_is_deterministic() -> None:
    arguments = {
        "specimen_id": "target-0",
        "dataset_id": "d0",
        "image": _image(),
        "metadata": np.asarray([1.0, 2.0, 3.0]),
        "initial_budget": 0.03125,
        "checkpoints": (0.0625,),
        "predictor": _FixturePredictor(),
        "encoder": _FixtureEncoder(),
        "method": "observed_uncertainty",
    }

    first = run_deployable_trajectory(**arguments)
    second = run_deployable_trajectory(**arguments)

    assert first.state_sha256 == second.state_sha256
    assert first.actions == second.actions
