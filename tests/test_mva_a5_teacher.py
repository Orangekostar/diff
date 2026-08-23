from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.mva.a5_teacher import (
    TeacherTrajectoryInput,
    fit_outer_safe_teacher_models,
    generate_teacher_trajectories,
    generate_teacher_trajectory,
    load_teacher_trajectory,
    save_teacher_trajectory,
)

DOMAINS = tuple(f"d{index}" for index in range(6))


class _ConstantEncoder:
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        return np.zeros((len(images), 512), dtype=np.float64)

    def validate(self) -> None:
        return None


class _ConstantPredictor:
    fit_domains = DOMAINS[2:]
    state_sha256 = hashlib.sha256(b"constant-predictor").hexdigest()

    def predict(self, metadata: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        assert metadata.shape[0] == embeddings.shape[0]
        return np.full(metadata.shape[0], 0.6, dtype=np.float64)


def _fit_fixture() -> tuple[
    tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(20260823)
    specimen_ids = tuple(f"{domain}-{row}" for domain in DOMAINS for row in range(9))
    dataset_ids = tuple(domain for domain in DOMAINS for _ in range(9))
    metadata = rng.normal(size=(54, 3))
    embeddings = rng.normal(size=(54, 24))
    targets = 0.6 + 0.02 * metadata[:, 0] + 0.01 * embeddings[:, 0]
    return specimen_ids, dataset_ids, targets, metadata, embeddings


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


def test_teacher_models_exclude_outer_and_each_query_domain() -> None:
    specimen_ids, dataset_ids, targets, metadata, embeddings = _fit_fixture()
    bundle = fit_outer_safe_teacher_models(
        outer_domain="d0",
        domain_order=DOMAINS,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=targets,
        metadata=metadata,
        full_embeddings=embeddings,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
    )

    assert set(bundle.models) == set(DOMAINS) - {"d0"}
    for query_domain, model in bundle.models.items():
        assert set(model.fit_domains) == set(DOMAINS) - {"d0", query_domain}
    for audit in bundle.fit_audits:
        assert "d0" not in audit.fit_domains
        assert audit.query_source_domain not in audit.fit_domains
        assert set(audit.query_domains).isdisjoint(audit.fit_domains)


def test_teacher_trajectory_records_complete_nested_candidates() -> None:
    trajectory = generate_teacher_trajectory(
        specimen_id="source-0",
        dataset_id="d1",
        image=_image(),
        target=0.7,
        metadata=np.asarray([1.0, 2.0, 3.0]),
        initial_budget=0.03125,
        checkpoints=(0.25,),
        predictor=_ConstantPredictor(),
        encoder=_ConstantEncoder(),
    )

    assert trajectory.states
    assert trajectory.states[0].actions[0].cell_index == 0
    assert trajectory.states[0].selected_index == 0
    assert any(
        action.from_level == 1 and action.to_level == 2
        for state in trajectory.states
        for action in state.actions
    )
    for state in trajectory.states:
        assert state.global_features.shape == (579,)
        assert state.candidate_features.shape == (len(state.actions), 8)
        assert state.values.shape == (len(state.actions),)
        assert state.selected_index == max(
            range(len(state.actions)),
            key=lambda index: (
                state.values[index],
                -state.actions[index].cell_index,
                -state.actions[index].to_level,
            ),
        )
        assert state.budget_after > state.budget_before
    assert len(trajectory.selected_actions) == len(trajectory.states)


def test_teacher_trajectory_is_deterministic_and_content_bound() -> None:
    arguments = {
        "specimen_id": "source-0",
        "dataset_id": "d1",
        "image": _image(),
        "target": 0.7,
        "metadata": np.asarray([1.0, 2.0, 3.0]),
        "initial_budget": 0.03125,
        "checkpoints": (0.0625,),
        "predictor": _ConstantPredictor(),
        "encoder": _ConstantEncoder(),
    }

    first = generate_teacher_trajectory(**arguments)
    second = generate_teacher_trajectory(**arguments)

    assert first.state_sha256 == second.state_sha256
    assert first.selected_actions == second.selected_actions
    assert all(
        np.array_equal(left.values, right.values)
        and np.array_equal(left.global_features, right.global_features)
        and np.array_equal(left.candidate_features, right.candidate_features)
        for left, right in zip(first.states, second.states, strict=True)
    )


def test_teacher_trajectory_package_round_trip(tmp_path) -> None:
    trajectory = generate_teacher_trajectory(
        specimen_id="source-0",
        dataset_id="d1",
        image=_image(),
        target=0.7,
        metadata=np.asarray([1.0, 2.0, 3.0]),
        initial_budget=0.03125,
        checkpoints=(0.0625,),
        predictor=_ConstantPredictor(),
        encoder=_ConstantEncoder(),
    )

    path = save_teacher_trajectory(tmp_path / "teacher.npz", trajectory)
    loaded = load_teacher_trajectory(path)

    assert loaded.state_sha256 == trajectory.state_sha256
    assert loaded.selected_actions == trajectory.selected_actions
    assert all(
        np.array_equal(left.candidate_features, right.candidate_features)
        and np.array_equal(left.values, right.values)
        for left, right in zip(loaded.states, trajectory.states, strict=True)
    )


def test_lockstep_teacher_batch_matches_single_trajectory_generation() -> None:
    inputs = tuple(
        TeacherTrajectoryInput(
            specimen_id=f"source-{index}",
            dataset_id="d1",
            image=_image(),
            target=0.7 + 0.01 * index,
            metadata=np.asarray([1.0, 2.0, 3.0]),
            predictor=_ConstantPredictor(),
            initial_embedding=np.zeros(512, dtype=np.float64),
        )
        for index in range(2)
    )

    batched = generate_teacher_trajectories(
        inputs,
        initial_budget=0.03125,
        checkpoints=(0.0625,),
        encoder=_ConstantEncoder(),
    )
    singles = tuple(
        generate_teacher_trajectory(
            specimen_id=value.specimen_id,
            dataset_id=value.dataset_id,
            image=value.image,
            target=value.target,
            metadata=value.metadata,
            initial_budget=0.03125,
            checkpoints=(0.0625,),
            predictor=value.predictor,
            encoder=_ConstantEncoder(),
            initial_embedding=value.initial_embedding,
        )
        for value in inputs
    )

    assert tuple(value.specimen_id for value in batched) == (
        "source-0",
        "source-1",
    )
    assert tuple(value.state_sha256 for value in batched) == tuple(
        value.state_sha256 for value in singles
    )
