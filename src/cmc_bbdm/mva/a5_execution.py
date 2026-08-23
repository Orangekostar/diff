"""Resumable outer-domain execution for the formal MVA A5 study."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from .a4_execution import (
    _global_ssim,
    _load_uniform_embeddings,
    fit_outer_evaluation_models,
)
from .a5_config import A5Config, load_a5_config
from .a5_deployment import run_deployable_trajectory
from .a5_teacher import (
    TeacherTrajectory,
    TeacherTrajectoryInput,
    fit_outer_safe_teacher_models,
    generate_teacher_trajectories,
    load_teacher_trajectory,
    save_teacher_trajectory,
)
from .authority import MVAAuthority, load_mva_authority
from .config import load_mva_config
from .encoder_session import MVAEncoderSession
from .oracle_execution import _initial_embeddings
from .pipeline import _encoder
from .ranking_policy import (
    RankingExample,
    save_policy_package,
    train_ranking_policy,
)
from .reconstruction_value import normalized_rgb_mse


class A5ExecutionError(ValueError):
    """Raised when an A5 outer worker is incomplete or inconsistent."""


class _ValidatedEncoder:
    def __init__(self, session: MVAEncoderSession) -> None:
        session.validate()
        self._session = session

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        return self._session.encode(images)

    def validate(self) -> None:
        return None


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _base_authority(
    root: Path, config: A5Config
) -> tuple[Path, MVAAuthority]:
    config_path = root / config.sources["a0_a3_config"].path
    base_config = load_mva_config(config_path, project_root=root)
    return config_path, load_mva_authority(base_config, project_root=root)


def _initial_budget(root: Path, outer_domain: str) -> float:
    table = pl.read_parquet(root / "results/mva/a4_global_task_mask/state_metrics.parquet")
    values = table.filter(pl.col("outer_domain") == outer_domain)[
        "initial_budget"
    ].unique()
    if len(values) != 1 or float(values[0]) not in {0.015625, 0.03125}:
        raise A5ExecutionError("A4 initial budget authority changed")
    return float(values[0])


def _a2_predictor_hashes(
    root: Path,
    *,
    outer_domain: str,
    checkpoints: tuple[float, ...],
) -> tuple[str, dict[float, str]]:
    table = pl.read_parquet(root / "results/mva/a2_oracle_value/state_metrics.parquet")
    selected = table.filter(
        (pl.col("dataset_id") == outer_domain)
        & pl.col("nominal_checkpoint").is_in(list(checkpoints))
    )
    p_a = tuple(str(value) for value in selected["p_a_predictor_state_sha256"].unique())
    p_b: dict[float, str] = {}
    for checkpoint in checkpoints:
        values = tuple(
            str(value)
            for value in selected.filter(
                pl.col("nominal_checkpoint") == checkpoint
            )["p_b_predictor_state_sha256"].unique()
        )
        if len(values) != 1:
            raise A5ExecutionError("A2 P-B predictor authority changed")
        p_b[checkpoint] = values[0]
    if len(p_a) != 1:
        raise A5ExecutionError("A2 P-A predictor authority changed")
    return p_a[0], p_b


def _teacher_cache_path(
    root: Path, outer_domain: str, specimen_index: int
) -> Path:
    return (
        root
        / "results/mva/.work/a5/teacher"
        / outer_domain
        / f"{specimen_index:03d}.npz"
    )


def _valid_cached_teacher(
    path: Path,
    *,
    specimen_id: str,
    dataset_id: str,
    predictor_state_sha256: str,
) -> TeacherTrajectory | None:
    if not path.is_file():
        return None
    try:
        trajectory = load_teacher_trajectory(path)
    except ValueError:
        return None
    if (
        trajectory.specimen_id != specimen_id
        or trajectory.dataset_id != dataset_id
        or trajectory.predictor_state_sha256 != predictor_state_sha256
    ):
        return None
    return trajectory


def _examples(
    trajectory: TeacherTrajectory,
) -> list[RankingExample]:
    return [
        RankingExample(
            dataset_id=trajectory.dataset_id,
            specimen_id=trajectory.specimen_id,
            global_features=state.global_features,
            candidate_features=state.candidate_features,
            selected_index=state.selected_index,
        )
        for state in trajectory.states
        if len(state.actions) >= 2
    ]


def _fit_audit_rows(bundle) -> list[dict[str, object]]:
    return [
        {
            "stage": value.stage,
            "held_out_target_domain": value.held_out_target_domain,
            "query_source_domain": value.query_source_domain,
            "query_domains": "|".join(value.query_domains),
            "fit_domains": "|".join(value.fit_domains),
            "query_specimen_ids": "|".join(value.query_specimen_ids),
            "fit_specimen_ids": "|".join(value.fit_specimen_ids),
            "pca_dimension": value.pca_dimension,
            "predictor_state_sha256": value.predictor_state_sha256,
        }
        for value in bundle.fit_audits
    ]


def _target_rows(
    *,
    authority: MVAAuthority,
    specimen_index: int,
    outer_domain: str,
    initial_budget: float,
    trajectory,
    p_b_models,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    target = float(authority.targets[specimen_index])
    metadata = authority.metadata13[specimen_index : specimen_index + 1]
    states: list[dict[str, object]] = []
    for snapshot in trajectory.snapshots:
        model = p_b_models[snapshot.checkpoint]
        p_b = float(
            model.predict(metadata, snapshot.embedding.reshape(1, -1))[0]
        )
        states.append(
            {
                "specimen_id": trajectory.specimen_id,
                "dataset_id": trajectory.dataset_id,
                "outer_domain": outer_domain,
                "method": trajectory.method,
                "initial_budget": initial_budget,
                "nominal_checkpoint": snapshot.checkpoint,
                "measured_count": snapshot.measured_count,
                "native_count": snapshot.native_count,
                "effective_budget": snapshot.effective_budget,
                "cumulative_actions": snapshot.cumulative_actions,
                "target": target,
                "p_a_prediction": snapshot.p_a_prediction,
                "p_a_absolute_error": abs(target - snapshot.p_a_prediction),
                "p_b_prediction": p_b,
                "p_b_absolute_error": abs(target - p_b),
                "normalized_rgb_mse": normalized_rgb_mse(
                    authority.images[specimen_index], snapshot.image
                ),
                "ssim": _global_ssim(
                    authority.images[specimen_index], snapshot.image
                ),
                "p_a_predictor_state_sha256": trajectory.predictor_state_sha256,
                "p_b_predictor_state_sha256": model.state_sha256,
                "policy_state_sha256": trajectory.policy_state_sha256,
                "trajectory_state_sha256": trajectory.state_sha256,
            }
        )
    actions = [
        {
            "specimen_id": trajectory.specimen_id,
            "dataset_id": trajectory.dataset_id,
            "outer_domain": outer_domain,
            "method": trajectory.method,
            "step": value.step,
            "nominal_checkpoint": value.checkpoint,
            "cell_index": value.cell_index,
            "from_level": value.from_level,
            "to_level": value.to_level,
            "selector_score": value.selector_score,
            "budget_before": value.budget_before,
            "budget_after": value.budget_after,
            "p_a_prediction_before": value.p_a_prediction_before,
            "p_a_prediction_after": value.p_a_prediction_after,
            "policy_state_sha256": trajectory.policy_state_sha256,
            "trajectory_state_sha256": trajectory.state_sha256,
        }
        for value in trajectory.actions
    ]
    return states, actions


def _published_shard_valid(path: Path, config_sha256: str, outer_domain: str) -> bool:
    if not path.is_dir() or not (path / "complete.json").is_file():
        return False
    try:
        complete = json.loads((path / "complete.json").read_text(encoding="utf-8"))
        hashes = complete["file_sha256"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        return False
    return bool(
        complete.get("outer_domain") == outer_domain
        and complete.get("config_sha256") == config_sha256
        and isinstance(hashes, dict)
        and set(hashes)
        == {
            "policy.npz",
            "policy_training.csv",
            "states.parquet",
            "teacher_fit_audits.csv",
            "teacher_index.csv",
            "trajectories.parquet",
        }
        and all(_sha_file(path / name) == digest for name, digest in hashes.items())
    )


def run_a5_outer_worker(
    config_path: str | Path,
    *,
    project_root: str | Path,
    outer_domain: str,
    device: str,
) -> Path:
    """Generate teacher data, train, and evaluate one held-out A5 domain."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_a5_config(config_file, project_root=root)
    if outer_domain not in config.domain_order:
        raise A5ExecutionError("outer domain is not registered")
    config_sha256 = _sha_file(config_file)
    destination = root / config.work_dir / "domains" / outer_domain
    if _published_shard_valid(destination, config_sha256, outer_domain):
        return destination
    _base_path, authority = _base_authority(root, config)
    if authority.specimen_count != config.specimen_count:
        raise A5ExecutionError("A5 cohort authority changed")
    initial_budget = _initial_budget(root, outer_domain)
    initial_embeddings = _initial_embeddings(root, initial_budget)
    teacher_models = fit_outer_safe_teacher_models(
        outer_domain=outer_domain,
        domain_order=config.domain_order,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        targets=authority.targets,
        metadata=authority.metadata13,
        full_embeddings=authority.full_embeddings,
        pca_dimensions=config.pca_dimensions,
        ridge_alpha=config.ridge_alpha,
        tie_tolerance=config.tie_tolerance,
    )
    encoder = _ValidatedEncoder(MVAEncoderSession(_encoder(root, device)))
    dataset_array = np.asarray(authority.dataset_ids, dtype=object)
    source_indices = np.flatnonzero(dataset_array != outer_domain)
    examples: list[RankingExample] = []
    teacher_index: list[dict[str, object]] = []
    batch_size = config.teacher_trajectory_batch_size
    for batch_start in range(0, len(source_indices), batch_size):
        chunk = tuple(
            int(value) for value in source_indices[batch_start : batch_start + batch_size]
        )
        trajectories: dict[int, TeacherTrajectory] = {}
        cached_indices: set[int] = set()
        missing_indices: list[int] = []
        requests: list[TeacherTrajectoryInput] = []
        for index in chunk:
            specimen_id = authority.specimen_ids[index]
            dataset_id = authority.dataset_ids[index]
            model = teacher_models.models[dataset_id]
            trajectory = _valid_cached_teacher(
                _teacher_cache_path(root, outer_domain, index),
                specimen_id=specimen_id,
                dataset_id=dataset_id,
                predictor_state_sha256=model.state_sha256,
            )
            if trajectory is not None:
                trajectories[index] = trajectory
                cached_indices.add(index)
            else:
                missing_indices.append(index)
                requests.append(
                    TeacherTrajectoryInput(
                        specimen_id=specimen_id,
                        dataset_id=dataset_id,
                        image=authority.images[index],
                        target=float(authority.targets[index]),
                        metadata=authority.metadata13[index],
                        predictor=model,
                        initial_embedding=initial_embeddings[index],
                    )
                )
        if requests:
            generated = generate_teacher_trajectories(
                tuple(requests),
                initial_budget=initial_budget,
                checkpoints=config.checkpoints,
                encoder=encoder,
            )
            for index, trajectory in zip(
                missing_indices, generated, strict=True
            ):
                path = _teacher_cache_path(root, outer_domain, index)
                save_teacher_trajectory(path, trajectory)
                trajectories[index] = trajectory
        for offset, index in enumerate(chunk):
            trajectory = trajectories[index]
            specimen_id = authority.specimen_ids[index]
            dataset_id = authority.dataset_ids[index]
            model = teacher_models.models[dataset_id]
            path = _teacher_cache_path(root, outer_domain, index)
            decision_examples = _examples(trajectory)
            examples.extend(decision_examples)
            teacher_index.append(
                {
                    "outer_domain": outer_domain,
                    "specimen_id": specimen_id,
                    "dataset_id": dataset_id,
                    "predictor_state_sha256": model.state_sha256,
                    "trajectory_state_sha256": trajectory.state_sha256,
                    "cache_sha256": _sha_file(path),
                    "state_count": len(trajectory.states),
                    "decision_state_count": len(decision_examples),
                    "candidate_count": sum(
                        len(state.actions) for state in trajectory.states
                    ),
                }
            )
            print(
                f"A5 teacher {outer_domain}: "
                f"{batch_start + offset + 1}/{len(source_indices)} "
                f"{specimen_id} "
                f"({'cache' if index in cached_indices else 'generated'})",
                flush=True,
            )
    if (
        not examples
        or {example.dataset_id for example in examples}
        != set(config.domain_order) - {outer_domain}
    ):
        raise A5ExecutionError("A5 policy training roster is incomplete")
    seed = config.seed + config.domain_order.index(outer_domain)
    policy = train_ranking_policy(
        tuple(examples),
        seed=seed,
        epochs=config.epochs,
        batch_states=config.batch_states,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
    )

    uniform_embeddings = _load_uniform_embeddings(
        root,
        authority,
        initial_budget=initial_budget,
        checkpoints=config.checkpoints,
    )
    evaluators = fit_outer_evaluation_models(
        outer_domain=outer_domain,
        domain_order=config.domain_order,
        checkpoints=config.checkpoints,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        targets=authority.targets,
        metadata=authority.metadata13,
        full_embeddings=authority.full_embeddings,
        uniform_embeddings=uniform_embeddings,
        pca_dimensions=config.pca_dimensions,
        ridge_alpha=config.ridge_alpha,
        tie_tolerance=config.tie_tolerance,
    )
    expected_p_a, expected_p_b = _a2_predictor_hashes(
        root, outer_domain=outer_domain, checkpoints=config.checkpoints
    )
    if (
        evaluators.p_a_model.state_sha256 != expected_p_a
        or any(
            evaluators.p_b_models[checkpoint].state_sha256
            != expected_p_b[checkpoint]
            for checkpoint in config.checkpoints
        )
    ):
        raise A5ExecutionError("A5 evaluator does not reproduce A2 predictor hashes")
    target_indices = np.flatnonzero(dataset_array == outer_domain)
    state_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    for position, specimen_index in enumerate(target_indices, start=1):
        index = int(specimen_index)
        for method in config.methods:
            trajectory = run_deployable_trajectory(
                specimen_id=authority.specimen_ids[index],
                dataset_id=authority.dataset_ids[index],
                image=authority.images[index],
                metadata=authority.metadata13[index],
                initial_budget=initial_budget,
                checkpoints=config.checkpoints,
                predictor=evaluators.p_a_model,
                encoder=encoder,
                method=method,
                policy=policy if method == "imitation_policy" else None,
            )
            states, actions = _target_rows(
                authority=authority,
                specimen_index=index,
                outer_domain=outer_domain,
                initial_budget=initial_budget,
                trajectory=trajectory,
                p_b_models=evaluators.p_b_models,
            )
            state_rows.extend(states)
            action_rows.extend(actions)
        print(
            f"A5 target {outer_domain}: {position}/{len(target_indices)} "
            f"{authority.specimen_ids[index]}",
            flush=True,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{outer_domain}.")
    )
    try:
        save_policy_package(temporary / "policy.npz", policy)
        pl.DataFrame(_fit_audit_rows(teacher_models), infer_schema_length=None).write_csv(
            temporary / "teacher_fit_audits.csv"
        )
        pl.DataFrame(teacher_index, infer_schema_length=None).write_csv(
            temporary / "teacher_index.csv"
        )
        pl.DataFrame(
            [
                {
                    "outer_domain": outer_domain,
                    "seed": seed,
                    "epoch": epoch,
                    "weighted_pairwise_loss": loss,
                    "policy_state_sha256": policy.state_sha256,
                    "teacher_model_state_sha256": teacher_models.state_sha256,
                    "training_state_count": len(examples),
                    "source_specimen_count": len(source_indices),
                }
                for epoch, loss in enumerate(policy.loss_trace, start=1)
            ],
            infer_schema_length=None,
        ).write_csv(temporary / "policy_training.csv")
        pl.DataFrame(state_rows, infer_schema_length=None).sort(
            ["dataset_id", "specimen_id", "method", "nominal_checkpoint"]
        ).write_parquet(temporary / "states.parquet")
        pl.DataFrame(action_rows, infer_schema_length=None).sort(
            ["dataset_id", "specimen_id", "method", "step"]
        ).write_parquet(temporary / "trajectories.parquet")
        data_files = {
            "policy.npz",
            "policy_training.csv",
            "states.parquet",
            "teacher_fit_audits.csv",
            "teacher_index.csv",
            "trajectories.parquet",
        }
        file_sha256 = {
            name: _sha_file(temporary / name) for name in sorted(data_files)
        }
        _write_json(
            temporary / "complete.json",
            {
                "authority_state_sha256": authority.state_sha256,
                "config_sha256": config_sha256,
                "evaluator_model_state_sha256": evaluators.state_sha256,
                "file_sha256": file_sha256,
                "initial_budget": initial_budget,
                "outer_domain": outer_domain,
                "policy_state_sha256": policy.state_sha256,
                "source_specimen_count": len(source_indices),
                "state_rows": len(state_rows),
                "target_specimen_count": len(target_indices),
                "teacher_model_state_sha256": teacher_models.state_sha256,
                "teacher_state_rows": sum(
                    int(row["state_count"]) for row in teacher_index
                ),
                "trajectory_rows": len(action_rows),
                "training_state_count": len(examples),
            },
        )
        if not _published_shard_valid(temporary, config_sha256, outer_domain):
            raise A5ExecutionError("staged A5 outer shard validation failed")
        if destination.exists():
            raise A5ExecutionError("invalid existing A5 outer shard requires audit")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


__all__ = ["A5ExecutionError", "run_a5_outer_worker"]
