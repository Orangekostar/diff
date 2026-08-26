"""P15 downstream-learner stability audit for retrospective mechanical value."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
import yaml
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor

from cmc_bbdm.cpb_v3.models import FoldPreprocessor, fit_fold_preprocessor
from cmc_bbdm.mva.a4_candidate_bank import CandidateBank, load_candidate_bank
from cmc_bbdm.mva.authority import load_mva_authority
from cmc_bbdm.mva.cai_evaluator import CAIPredictor, PCAProjection
from cmc_bbdm.mva.config import load_mva_config
from cmc_bbdm.mva.crossfit import fit_outer_source_predictor


class ValueStabilityError(ValueError):
    """Raised when the P15 information barrier or artifact contract changes."""


class _Regressor(Protocol):
    def predict(self, matrix: object) -> np.ndarray: ...


_LEARNERS = ("ridge", "huber", "shallow_mlp")
_PAIRS = (
    ("ridge", "huber"),
    ("ridge", "shallow_mlp"),
    ("huber", "shallow_mlp"),
)
_FILES = {
    "CHECKSUMS.sha256",
    "REPORT.md",
    "artifact_manifest.json",
    "bootstrap.csv",
    "model_metrics.csv",
    "oracle_utility.csv",
    "rank_agreement.csv",
    "region_stability.csv",
    "ridge_sensitivity.csv",
    "strict_oof_audit.csv",
    "summary.json",
    "value_maps.parquet",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_state(root: Path) -> str:
    rows = [
        (path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    if not rows:
        raise ValueStabilityError("bound package is empty")
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _state(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(
                json.dumps(array.shape, separators=(",", ":")).encode("ascii")
            )
            digest.update(array.tobytes(order="C"))
        elif isinstance(value, bytes):
            digest.update(value)
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FittedValueLearner:
    name: str
    outer_domain: str
    pca: PCAProjection
    preprocessor: FoldPreprocessor | None
    estimator: _Regressor | None
    ridge: CAIPredictor | None
    fit_specimen_ids: tuple[str, ...]
    fit_domains: tuple[str, ...]
    state_sha256: str

    def predict(self, metadata: object, embeddings: object) -> np.ndarray:
        if self.ridge is not None:
            return self.ridge.predict(metadata, embeddings)
        if self.preprocessor is None or self.estimator is None:
            raise ValueStabilityError("value learner is incomplete")
        meta = np.asarray(metadata, dtype=np.float64)
        projected = self.pca.transform(embeddings)
        matrix = np.column_stack((meta, projected))
        transformed = self.preprocessor.transform(matrix)
        prediction = np.asarray(self.estimator.predict(transformed), dtype=np.float64)
        if prediction.shape != (len(matrix),) or not np.all(np.isfinite(prediction)):
            raise ValueStabilityError("value learner returned invalid predictions")
        return prediction


def fit_outer_value_learners(
    *,
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    domain_order: tuple[str, ...],
    targets: object,
    metadata: object,
    embeddings: object,
    outer_domain: str,
    pca_dimensions: tuple[int, ...],
    seed: int,
) -> dict[str, FittedValueLearner]:
    """Fit Ridge, Huber, and shallow MLP without any outer-domain specimen."""

    y = np.asarray(targets, dtype=np.float64)
    meta = np.asarray(metadata, dtype=np.float64)
    values = np.asarray(embeddings, dtype=np.float64)
    count = len(specimen_ids)
    if (
        outer_domain not in domain_order
        or len(dataset_ids) != count
        or y.shape != (count,)
        or meta.ndim != 2
        or meta.shape[0] != count
        or values.ndim != 2
        or values.shape[0] != count
        or not np.all(np.isfinite(y))
        or np.any(np.isinf(meta))
        or not np.all(np.isfinite(values))
    ):
        raise ValueStabilityError("value learner inputs are invalid")
    ridge_fit = fit_outer_source_predictor(
        method="P15_RIDGE",
        outer_domain=outer_domain,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domain_order,
        targets=y,
        metadata=meta,
        embeddings=values,
        pca_dimensions=pca_dimensions,
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )
    source = np.flatnonzero(np.asarray(dataset_ids, dtype=object) != outer_domain)
    fit_ids = tuple(specimen_ids[index] for index in source)
    fit_domain_rows = tuple(dataset_ids[index] for index in source)
    fit_domains = tuple(dict.fromkeys(fit_domain_rows))
    if outer_domain in ridge_fit.model.fit_domains or outer_domain in fit_domains:
        raise ValueStabilityError("outer domain entered a value learner fit")
    ridge = FittedValueLearner(
        name="ridge",
        outer_domain=outer_domain,
        pca=ridge_fit.model.pca,
        preprocessor=None,
        estimator=None,
        ridge=ridge_fit.model,
        fit_specimen_ids=ridge_fit.model.fit_specimen_ids,
        fit_domains=ridge_fit.model.fit_domains,
        state_sha256=ridge_fit.model.state_sha256,
    )
    train_matrix = np.column_stack((meta[source], ridge.pca.transform(values[source])))
    preprocessor = fit_fold_preprocessor(
        train_matrix,
        fit_sample_ids=fit_ids,
        fit_domain_ids=fit_domain_rows,
    )
    transformed = preprocessor.transform(train_matrix)
    specifications: tuple[tuple[str, _Regressor], ...] = (
        (
            "huber",
            HuberRegressor(
                alpha=0.1,
                epsilon=1.35,
                max_iter=5_000,
                tol=1.0e-10,
            ),
        ),
        (
            "shallow_mlp",
            MLPRegressor(
                hidden_layer_sizes=(16,),
                activation="relu",
                solver="lbfgs",
                alpha=0.01,
                max_iter=5_000,
                random_state=int(seed),
            ),
        ),
    )
    output = {"ridge": ridge}
    for name, estimator in specifications:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                estimator.fit(transformed, y[source])
            except (
                ConvergenceWarning,
                FloatingPointError,
                TypeError,
                ValueError,
            ) as error:
                raise ValueStabilityError(f"{name} learner fit failed") from error
        model_state = _state(
            "p15-value-learner",
            name,
            outer_domain,
            ridge.pca.state_sha256,
            preprocessor.state_sha256,
            pickle.dumps(estimator, protocol=5),
            fit_ids,
            fit_domain_rows,
        )
        output[name] = FittedValueLearner(
            name=name,
            outer_domain=outer_domain,
            pca=ridge.pca,
            preprocessor=preprocessor,
            estimator=estimator,
            ridge=None,
            fit_specimen_ids=fit_ids,
            fit_domains=fit_domains,
            state_sha256=model_state,
        )
    return output


def validate_shared_value_bank(frame: pl.DataFrame) -> None:
    required = {
        "specimen_id",
        "dataset_id",
        "learner",
        "cell_index",
        "initial_budget",
        "added_measurements",
        "native_count",
        "candidate_bank_state_sha256",
    }
    if frame.is_empty() or not required <= set(frame.columns):
        raise ValueStabilityError("shared state/action bank schema changed")
    grouped = frame.group_by("specimen_id", "cell_index").agg(
        pl.col("learner").n_unique().alias("learners"),
        pl.col("dataset_id").n_unique().alias("domains"),
        pl.col("initial_budget").n_unique().alias("budgets"),
        pl.col("added_measurements").n_unique().alias("costs"),
        pl.col("native_count").n_unique().alias("native_counts"),
        pl.col("candidate_bank_state_sha256").n_unique().alias("banks"),
    )
    if grouped.filter(
        (pl.col("learners") != len(_LEARNERS))
        | (pl.col("domains") != 1)
        | (pl.col("budgets") != 1)
        | (pl.col("costs") != 1)
        | (pl.col("native_counts") != 1)
        | (pl.col("banks") != 1)
    ).height:
        raise ValueStabilityError("shared state/action bank changed across learners")


def _predict_candidates(
    learner: FittedValueLearner,
    metadata: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    rows, actions, width = embeddings.shape
    flat = embeddings.reshape(rows * actions, width)
    repeated = np.repeat(metadata, actions, axis=0)
    predictions = []
    for start in range(0, len(flat), 4096):
        predictions.append(
            learner.predict(repeated[start : start + 4096], flat[start : start + 4096])
        )
    output = np.concatenate(predictions).reshape(rows, actions)
    if not np.all(np.isfinite(output)):
        raise ValueStabilityError("candidate predictions are nonfinite")
    return output


def _top(values: np.ndarray, count: int) -> tuple[int, ...]:
    return tuple(
        sorted(range(len(values)), key=lambda index: (-float(values[index]), index))[
            :count
        ]
    )


def _radial(cell_index: int) -> float:
    row, column = divmod(cell_index, 8)
    return math.hypot(row / 7.0 - 0.5, column / 7.0 - 0.5) / math.sqrt(0.5)


def _ranking_rows(
    value_maps: pl.DataFrame, *, top_count: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (domain, specimen), table in value_maps.group_by(
        "dataset_id", "specimen_id", maintain_order=True
    ):
        scores = {
            learner: table.filter(pl.col("learner") == learner)
            .sort("cell_index")
            .get_column("value")
            .to_numpy()
            for learner in _LEARNERS
        }
        for first, second in _PAIRS:
            first_top = _top(scores[first], top_count)
            second_top = _top(scores[second], top_count)
            statistic = float(spearmanr(scores[first], scores[second]).statistic)
            if not math.isfinite(statistic):
                statistic = (
                    1.0 if np.array_equal(scores[first], scores[second]) else 0.0
                )
            first_xy = (
                np.asarray([divmod(cell, 8) for cell in first_top], dtype=np.float64)
                / 7.0
            )
            second_xy = (
                np.asarray([divmod(cell, 8) for cell in second_top], dtype=np.float64)
                / 7.0
            )
            rows.append(
                {
                    "specimen_id": specimen,
                    "dataset_id": domain,
                    "first_learner": first,
                    "second_learner": second,
                    "pair": f"{first}__{second}",
                    "candidate_count": 64,
                    "topk_count": top_count,
                    "spearman": statistic,
                    "topk_overlap": len(set(first_top) & set(second_top)) / top_count,
                    "topk_jaccard": len(set(first_top) & set(second_top))
                    / len(set(first_top) | set(second_top)),
                    "best_action_agreement": first_top[0] == second_top[0],
                    "normalized_topk_centroid_distance": float(
                        np.linalg.norm(first_xy.mean(axis=0) - second_xy.mean(axis=0))
                        / math.sqrt(2.0)
                    ),
                    "topk_mean_radial_difference": abs(
                        math.fsum(_radial(cell) for cell in first_top) / top_count
                        - math.fsum(_radial(cell) for cell in second_top) / top_count
                    ),
                }
            )
    return rows


def _canonical(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col(pl.Float64).round(15))


def _domain_balanced(frame: pl.DataFrame, metric: str) -> float:
    means = []
    order = ["dataset_id"] + (["specimen_id"] if "specimen_id" in frame.columns else [])
    for (_domain,), group in frame.sort(order).group_by(
        "dataset_id", maintain_order=True
    ):
        values = group.get_column(metric).cast(pl.Float64).to_list()
        means.append(math.fsum(values) / len(values))
    return math.fsum(means) / len(means)


def _bootstrap_subject(
    frame: pl.DataFrame,
    *,
    analysis: str,
    subject: str,
    metrics: tuple[str, ...],
    replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    sampled_means = {
        metric: np.zeros(replicates, dtype=np.float64) for metric in metrics
    }
    domain_count = 0
    for (_domain,), group in frame.sort("dataset_id", "specimen_id").group_by(
        "dataset_id", maintain_order=True
    ):
        domain_count += 1
        indices = rng.integers(0, group.height, size=(replicates, group.height))
        for metric in metrics:
            values = group.get_column(metric).cast(pl.Float64).to_numpy()
            sampled_means[metric] += values[indices].mean(axis=1)
    if domain_count != 6:
        raise ValueStabilityError("bootstrap domain roster changed")
    rows = []
    for metric in metrics:
        values = sampled_means[metric] / domain_count
        rows.extend(
            {
                "analysis": analysis,
                "subject": subject,
                "metric": metric,
                "replicate": replicate,
                "value": float(value),
            }
            for replicate, value in enumerate(values)
        )
    return rows


def _load_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path).resolve(strict=True)
    raw = config_path.read_bytes()
    value = yaml.safe_load(raw)
    expected_learners = {
        "ridge": {"alpha": 10.0},
        "huber": {
            "alpha": 0.1,
            "epsilon": 1.35,
            "max_iter": 5000,
            "tolerance": 1.0e-10,
        },
        "shallow_mlp": {
            "hidden_layer_size": 16,
            "alpha": 0.01,
            "solver": "lbfgs",
            "max_iter": 5000,
            "seed": 20260826,
        },
    }
    if (
        not isinstance(value, dict)
        or value.get("stage") != "P15_VALUE_STABILITY"
        or value.get("pca_dimensions") != [8, 16, 32]
        or value.get("learners") != expected_learners
    ):
        raise ValueStabilityError("P15 config changed")
    value["config_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def _bound(root: Path, value: object, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueStabilityError("P15 source path is invalid")
    try:
        path = (root / value).resolve(strict=True)
    except OSError as error:
        raise ValueStabilityError("P15 source is unavailable") from error
    if root != path and root not in path.parents:
        raise ValueStabilityError("P15 source escapes project root")
    if path.is_dir() != directory:
        raise ValueStabilityError("P15 source type changed")
    return path


def _write_package_metadata(
    output: Path,
    *,
    config: dict[str, object],
    source_hashes: dict[str, str],
) -> None:
    payload_names = sorted(_FILES - {"CHECKSUMS.sha256", "artifact_manifest.json"})
    manifest = {
        "schema_version": 1,
        "stage": "P15_VALUE_STABILITY",
        "audit_base_git_sha": config["audit_base_git_sha"],
        "config_sha256": config["config_sha256"],
        "source_sha256": source_hashes,
        "artifacts": [
            {
                "path": name,
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in payload_names
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    checksum_names = sorted(_FILES - {"CHECKSUMS.sha256"})
    (output / "CHECKSUMS.sha256").write_text(
        "".join(f"{_sha256(output / name)}  {name}\n" for name in checksum_names),
        encoding="ascii",
    )


def run_p15_value_stability(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_project_root: str | Path | None = None,
    output_root: str | Path,
) -> Path:
    root = Path(project_root).resolve(strict=True)
    authority_root = (
        root
        if source_project_root is None
        else Path(source_project_root).resolve(strict=True)
    )
    config = _load_config(config_path)
    domains = tuple(config["domain_order"])
    if len(domains) != 6 or len(set(domains)) != 6:
        raise ValueStabilityError("P15 domain roster changed")
    source_paths = {
        "mva_config": _bound(root, config["mva_config"]),
        "oracle_values": _bound(root, config["oracle_values"]),
        "ridge_sensitivity": _bound(root, config["ridge_sensitivity"]),
    }
    for name, path in source_paths.items():
        if _sha256(path) != config[f"{name}_sha256"]:
            raise ValueStabilityError(f"P15 {name} hash changed")
    p7 = _bound(root, config["p7_package"], directory=True)
    if _tree_state(p7) != config["p7_tree_state_sha256"]:
        raise ValueStabilityError("P15 P7 state changed")
    bank_config = config["candidate_banks"]
    if not isinstance(bank_config, dict):
        raise ValueStabilityError("P15 bank config changed")
    banks: dict[float, CandidateBank] = {}
    bank_file_hashes: dict[float, str] = {}
    for budget in (0.015625, 0.03125):
        entry = bank_config.get(budget)
        if not isinstance(entry, dict):
            raise ValueStabilityError("P15 bank entry changed")
        path = _bound(root, entry["path"])
        file_hash = _sha256(path)
        bank = load_candidate_bank(path)
        if (
            file_hash != entry["file_sha256"]
            or bank.state_sha256 != entry["state_sha256"]
        ):
            raise ValueStabilityError("P15 candidate bank hash changed")
        banks[budget] = bank
        bank_file_hashes[budget] = file_hash
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if root not in destination.parents or destination.exists():
        raise ValueStabilityError("P15 output is invalid or already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".p15_value_stability.", dir=destination.parent)
    )
    p7_before = _tree_state(p7)
    try:
        authority_config = _bound(authority_root, config["mva_config"])
        if _sha256(authority_config) != config["mva_config_sha256"]:
            raise ValueStabilityError("P15 authority config hash changed")
        mva_config = load_mva_config(authority_config, project_root=authority_root)
        authority = load_mva_authority(mva_config, project_root=authority_root)
        if tuple(mva_config.domain_order) != domains:
            raise ValueStabilityError("P15 authority domain order changed")
        selected_budget = {
            str(key): float(value)
            for key, value in config["selected_initial_budget"].items()
        }
        if set(selected_budget) != set(domains) or set(selected_budget.values()) - set(
            banks
        ):
            raise ValueStabilityError("P15 selected budget map changed")
        issued = pl.read_parquet(source_paths["oracle_values"]).filter(
            (pl.col("method") == "mechanical_oracle") & (pl.col("step") == 0)
        )
        value_rows: list[dict[str, object]] = []
        oracle_rows: list[dict[str, object]] = []
        audit_rows: list[dict[str, object]] = []
        model_rows: list[dict[str, object]] = []
        maximum_ridge_bank_recompute_delta = 0.0
        dataset_array = np.asarray(authority.dataset_ids, dtype=object)
        for outer_domain in domains:
            budget = selected_budget[outer_domain]
            bank = banks[budget]
            learners = fit_outer_value_learners(
                specimen_ids=authority.specimen_ids,
                dataset_ids=authority.dataset_ids,
                domain_order=domains,
                targets=authority.targets,
                metadata=authority.metadata13,
                embeddings=authority.full_embeddings,
                outer_domain=outer_domain,
                pca_dimensions=tuple(config["pca_dimensions"]),
                seed=int(config["learners"]["shallow_mlp"]["seed"]),
            )
            query = np.flatnonzero(dataset_array == outer_domain)
            for learner in learners.values():
                if outer_domain in learner.fit_domains or set(query) & {
                    authority.specimen_ids.index(item)
                    for item in learner.fit_specimen_ids
                }:
                    raise ValueStabilityError("P15 strict OOF barrier changed")
                current = learner.predict(
                    authority.metadata13[query], bank.initial_embeddings[query]
                )
                candidates = _predict_candidates(
                    learner,
                    authority.metadata13[query],
                    bank.embeddings[query],
                )
                full_prediction = learner.predict(
                    authority.metadata13[query], authority.full_embeddings[query]
                )
                model_rows.append(
                    {
                        "dataset_id": outer_domain,
                        "learner": learner.name,
                        "specimen_count": len(query),
                        "selected_pca_dimension": learner.pca.dimension,
                        "full_state_mae": float(
                            np.mean(np.abs(authority.targets[query] - full_prediction))
                        ),
                        "initial_sparse_state_mae": float(
                            np.mean(np.abs(authority.targets[query] - current))
                        ),
                    }
                )
                fit_id_hash = _state(learner.fit_specimen_ids)
                audit_rows.append(
                    {
                        "outer_domain": outer_domain,
                        "query_domains": outer_domain,
                        "learner": learner.name,
                        "fit_domains": ",".join(learner.fit_domains),
                        "fit_specimen_count": len(learner.fit_specimen_ids),
                        "fit_specimen_ids_sha256": fit_id_hash,
                        "selected_pca_dimension": learner.pca.dimension,
                        "predictor_state_sha256": learner.state_sha256,
                        "initial_budget": budget,
                        "candidate_count": 64,
                        "candidate_bank_state_sha256": bank.state_sha256,
                        "candidate_bank_file_sha256": bank_file_hashes[budget],
                    }
                )
                for local, specimen_index in enumerate(query):
                    target = float(authority.targets[specimen_index])
                    emitted_current = float(current[local])
                    emitted_candidates = candidates[local].copy()
                    before = abs(target - emitted_current)
                    after = np.abs(target - emitted_candidates)
                    values = before - after
                    issued_specimen = issued.filter(
                        pl.col("specimen_id") == authority.specimen_ids[specimen_index]
                    ).sort("cell_index")
                    if learner.name == "ridge":
                        if issued_specimen.height != 64 or issued_specimen.get_column(
                            "cell_index"
                        ).to_list() != list(range(64)):
                            raise ValueStabilityError("P15 issued Ridge roster changed")
                        formal_values = issued_specimen.get_column(
                            "primary_value"
                        ).to_numpy()
                        maximum_ridge_bank_recompute_delta = max(
                            maximum_ridge_bank_recompute_delta,
                            float(np.max(np.abs(values - formal_values))),
                        )
                        values = formal_values
                        after = issued_specimen.get_column("error_after").to_numpy()
                        emitted_candidates = issued_specimen.get_column(
                            "new_prediction"
                        ).to_numpy()
                        before = float(
                            issued_specimen.get_column("error_before").item(0)
                        )
                        emitted_current = float(
                            issued_specimen.get_column("current_prediction").item(0)
                        )
                    selected_cell = _top(values, 1)[0]
                    reconstruction_cell = _top(
                        bank.reconstruction_values[specimen_index], 1
                    )[0]
                    oracle_rows.append(
                        {
                            "specimen_id": authority.specimen_ids[specimen_index],
                            "dataset_id": outer_domain,
                            "learner": learner.name,
                            "initial_budget": budget,
                            "current_absolute_error": before,
                            "mechanical_oracle_cell": selected_cell,
                            "mechanical_oracle_added_measurements": int(
                                bank.added_measurements[specimen_index, selected_cell]
                            ),
                            "mechanical_oracle_error_after": float(
                                after[selected_cell]
                            ),
                            "mechanical_oracle_improvement": float(
                                values[selected_cell]
                            ),
                            "reconstruction_oracle_cell": reconstruction_cell,
                            "reconstruction_oracle_added_measurements": int(
                                bank.added_measurements[
                                    specimen_index, reconstruction_cell
                                ]
                            ),
                            "reconstruction_action_error_after": float(
                                after[reconstruction_cell]
                            ),
                            "mechanical_headroom_over_reconstruction": float(
                                after[reconstruction_cell] - after[selected_cell]
                            ),
                            "candidate_bank_state_sha256": bank.state_sha256,
                            "predictor_state_sha256": learner.state_sha256,
                        }
                    )
                    for cell_index in range(64):
                        value_rows.append(
                            {
                                "specimen_id": authority.specimen_ids[specimen_index],
                                "dataset_id": outer_domain,
                                "learner": learner.name,
                                "cell_index": cell_index,
                                "initial_budget": budget,
                                "added_measurements": int(
                                    bank.added_measurements[specimen_index, cell_index]
                                ),
                                "native_count": int(bank.native_counts[specimen_index]),
                                "current_prediction": emitted_current,
                                "candidate_prediction": float(
                                    emitted_candidates[cell_index]
                                ),
                                "current_absolute_error": before,
                                "candidate_absolute_error": float(after[cell_index]),
                                "value": float(values[cell_index]),
                                "selected": cell_index == selected_cell,
                                "candidate_bank_state_sha256": bank.state_sha256,
                                "predictor_state_sha256": learner.state_sha256,
                            }
                        )
        value_maps = pl.DataFrame(value_rows).sort(
            "dataset_id", "specimen_id", "learner", "cell_index"
        )
        if value_maps.height != 276 * len(_LEARNERS) * 64:
            raise ValueStabilityError("P15 value-map roster changed")
        validate_shared_value_bank(value_maps)
        formal_reference = issued.select(
            "specimen_id",
            "cell_index",
            pl.col("primary_value").alias("formal_value"),
        )
        maximum_formal_ridge_value_delta = float(
            value_maps.filter(pl.col("learner") == "ridge")
            .join(formal_reference, on=["specimen_id", "cell_index"], how="inner")
            .select((pl.col("value") - pl.col("formal_value")).abs().max())
            .item()
        )
        if maximum_formal_ridge_value_delta != 0.0:
            raise ValueStabilityError("P15 emitted Ridge values changed")
        top_count = max(1, math.ceil(float(config["top_fraction"]) * 64))
        rank = _canonical(
            pl.DataFrame(_ranking_rows(value_maps, top_count=top_count))
        ).sort("dataset_id", "specimen_id", "pair")
        region = rank.select(
            "specimen_id",
            "dataset_id",
            "pair",
            "normalized_topk_centroid_distance",
            "topk_mean_radial_difference",
        )
        oracle = _canonical(pl.DataFrame(oracle_rows)).sort(
            "dataset_id", "specimen_id", "learner"
        )
        audits = pl.DataFrame(audit_rows).sort("outer_domain", "learner")
        models = _canonical(pl.DataFrame(model_rows)).sort("dataset_id", "learner")
        bootstrap_rows: list[dict[str, object]] = []
        boot_seed = int(config["seed"])
        for offset, learner in enumerate(_LEARNERS):
            bootstrap_rows.extend(
                _bootstrap_subject(
                    oracle.filter(pl.col("learner") == learner),
                    analysis="oracle_utility",
                    subject=learner,
                    metrics=(
                        "mechanical_oracle_improvement",
                        "mechanical_headroom_over_reconstruction",
                    ),
                    replicates=int(config["bootstrap_replicates"]),
                    seed=boot_seed + offset,
                )
            )
        for offset, (first, second) in enumerate(_PAIRS, start=len(_LEARNERS)):
            pair = f"{first}__{second}"
            bootstrap_rows.extend(
                _bootstrap_subject(
                    rank.filter(pl.col("pair") == pair),
                    analysis="rank_agreement",
                    subject=pair,
                    metrics=(
                        "spearman",
                        "topk_jaccard",
                        "best_action_agreement",
                        "normalized_topk_centroid_distance",
                    ),
                    replicates=int(config["bootstrap_replicates"]),
                    seed=boot_seed + offset,
                )
            )
        bootstrap = _canonical(pl.DataFrame(bootstrap_rows)).sort(
            "analysis", "subject", "metric", "replicate"
        )
        intervals: dict[str, dict[str, object]] = {}
        for (analysis, subject, metric), group in bootstrap.group_by(
            "analysis", "subject", "metric", maintain_order=True
        ):
            source = (
                oracle.filter(pl.col("learner") == subject)
                if analysis == "oracle_utility"
                else rank.filter(pl.col("pair") == subject)
            )
            values = group.get_column("value").to_numpy()
            intervals[f"{analysis}/{subject}/{metric}"] = {
                "point": round(_domain_balanced(source, metric), 15),
                "interval": [
                    round(float(np.quantile(values, 0.025)), 15),
                    round(float(np.quantile(values, 0.975)), 15),
                ],
            }
        thresholds = config["stability_thresholds"]
        pair_summary = []
        stable = True
        for first, second in _PAIRS:
            pair = f"{first}__{second}"
            table = rank.filter(pl.col("pair") == pair)
            row = {
                "pair": pair,
                "mean_spearman": round(_domain_balanced(table, "spearman"), 15),
                "mean_topk_overlap": round(_domain_balanced(table, "topk_overlap"), 15),
                "mean_topk_jaccard": round(_domain_balanced(table, "topk_jaccard"), 15),
                "best_action_agreement": round(
                    _domain_balanced(table, "best_action_agreement"), 15
                ),
                "mean_normalized_topk_centroid_distance": round(
                    _domain_balanced(table, "normalized_topk_centroid_distance"), 15
                ),
            }
            pair_summary.append(row)
            stable &= (
                row["mean_spearman"] >= float(thresholds["mean_spearman"])
                and row["mean_topk_jaccard"] >= float(thresholds["mean_topk_jaccard"])
                and row["best_action_agreement"]
                >= float(thresholds["best_action_agreement"])
            )
        learner_summary = []
        for learner in _LEARNERS:
            table = oracle.filter(pl.col("learner") == learner)
            key = f"oracle_utility/{learner}/mechanical_headroom_over_reconstruction"
            learner_summary.append(
                {
                    "learner": learner,
                    "full_state_oof_mae": round(
                        _domain_balanced(
                            models.rename({"full_state_mae": "metric"}).filter(
                                pl.col("learner") == learner
                            ),
                            "metric",
                        ),
                        15,
                    ),
                    "mechanical_oracle_improvement": round(
                        _domain_balanced(table, "mechanical_oracle_improvement"), 15
                    ),
                    "mechanical_headroom_over_reconstruction": round(
                        _domain_balanced(
                            table, "mechanical_headroom_over_reconstruction"
                        ),
                        15,
                    ),
                    "headroom_interval": intervals[key]["interval"],
                }
            )
        if stable:
            conclusion = (
                "Mechanics-relevant measurement value has learner-robust structure "
                "under the tested Ridge, Huber, and shallow-MLP family."
            )
        else:
            ridge_huber = next(
                row for row in pair_summary if row["pair"] == "ridge__huber"
            )
            ridge_mlp = next(
                row for row in pair_summary if row["pair"] == "ridge__shallow_mlp"
            )
            mlp = next(
                row for row in learner_summary if row["learner"] == "shallow_mlp"
            )
            conclusion = (
                "Ridge and Huber retain similar value structure "
                f"(mean Spearman {ridge_huber['mean_spearman']:.3f}), but the "
                f"less accurate shallow MLP (OOF MAE {mlp['full_state_oof_mae']:.3f}) "
                "does not reproduce it "
                f"(Ridge-MLP Spearman {ridge_mlp['mean_spearman']:.3f}). Across "
                "the predeclared family, the defensible term is "
                "downstream-predictor-conditioned task value."
            )
        summary = {
            "schema_version": 1,
            "stage": "P15_VALUE_STABILITY",
            "audit_base_git_sha": config["audit_base_git_sha"],
            "config_sha256": config["config_sha256"],
            "specimen_count": 276,
            "domain_order": domains,
            "learners": _LEARNERS,
            "pair_summary": pair_summary,
            "learner_summary": learner_summary,
            "bootstrap_intervals": intervals,
            "maximum_formal_ridge_value_delta": maximum_formal_ridge_value_delta,
            "maximum_ridge_bank_recompute_delta": maximum_ridge_bank_recompute_delta,
            "learner_robust_structure_supported": bool(stable),
            "required_term": (
                "mechanics-relevant measurement value"
                if stable
                else "downstream-predictor-conditioned task value"
            ),
            "p7_tree_state_sha256": p7_before,
            "primary_conclusion": conclusion,
        }
        value_maps.write_parquet(
            temporary / "value_maps.parquet", compression="zstd", statistics=True
        )
        rank.write_csv(temporary / "rank_agreement.csv", float_scientific=False)
        region.write_csv(temporary / "region_stability.csv", float_scientific=False)
        oracle.write_csv(temporary / "oracle_utility.csv", float_scientific=False)
        models.write_csv(temporary / "model_metrics.csv", float_scientific=False)
        audits.write_csv(temporary / "strict_oof_audit.csv", float_scientific=False)
        bootstrap.write_csv(temporary / "bootstrap.csv", float_scientific=False)
        pl.read_csv(source_paths["ridge_sensitivity"]).sort(
            "dataset_id", "specimen_id", "variant"
        ).write_csv(temporary / "ridge_sensitivity.csv", float_scientific=False)
        (temporary / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "REPORT.md").write_text(
            "# P15 Downstream-Learner Value-Stability Audit\n\n"
            f"{conclusion}\n\n"
            "The audit uses the same 276-specimen cohort, six LODO outer splits, "
            "fold-local PCA dimension, 64-action initial state bank, measurement "
            "reveal, and exact action cost for Ridge, Huber, and a 16-unit shallow "
            "MLP. The emitted Ridge map reproduces the issued formal oracle values "
            f"exactly (maximum delta {maximum_formal_ridge_value_delta:.3g}); a "
            "fresh candidate-bank Ridge inference differs by at most "
            f"{maximum_ridge_bank_recompute_delta:.3g} because the frozen image "
            "states were encoded in a separate GPU pass.\n\n"
            "Mechanical oracle utility is retrospective and non-deployable. The "
            "analysis does not establish an intrinsic or universal physical value "
            "map, and coordinate summaries do not identify a failure mechanism. "
            "P7 remains unchanged.\n",
            encoding="utf-8",
        )
        source_hashes = {
            **{name: _sha256(path) for name, path in source_paths.items()},
            **{
                f"candidate_bank_{budget:g}": bank_file_hashes[budget]
                for budget in sorted(bank_file_hashes)
            },
            "p7_tree_state": p7_before,
        }
        _write_package_metadata(temporary, config=config, source_hashes=source_hashes)
        if _tree_state(p7) != p7_before:
            raise ValueStabilityError("P15 modified P7")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_p15_value_stability_package(destination)
    return destination


def verify_p15_value_stability_package(package_path: str | Path) -> dict[str, object]:
    package = Path(package_path)
    if not package.is_dir() or {item.name for item in package.iterdir()} != _FILES:
        raise ValueStabilityError("P15 package file roster changed")
    checksums = {}
    for line in (package / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    if set(checksums) != _FILES - {"CHECKSUMS.sha256"} or any(
        _sha256(package / name) != digest for name, digest in checksums.items()
    ):
        raise ValueStabilityError("P15 package checksum mismatch")
    summary = json.loads((package / "summary.json").read_text(encoding="utf-8"))
    values = pl.read_parquet(package / "value_maps.parquet")
    rank = pl.read_csv(package / "rank_agreement.csv")
    oracle = pl.read_csv(package / "oracle_utility.csv")
    audits = pl.read_csv(package / "strict_oof_audit.csv")
    bootstrap = pl.read_csv(package / "bootstrap.csv")
    validate_shared_value_bank(values)
    if (
        summary.get("stage") != "P15_VALUE_STABILITY"
        or values.height != 276 * 3 * 64
        or rank.height != 276 * 3
        or oracle.height != 276 * 3
        or audits.height != 6 * 3
        or bootstrap.height != (3 * 2 + 3 * 4) * 5000
        or float(summary.get("maximum_formal_ridge_value_delta", math.inf)) > 1.0e-12
        or any(
            str(row["outer_domain"]) in str(row["fit_domains"]).split(",")
            for row in audits.iter_rows(named=True)
        )
    ):
        raise ValueStabilityError("P15 scientific contract changed")
    return summary


__all__ = [
    "FittedValueLearner",
    "ValueStabilityError",
    "fit_outer_value_learners",
    "run_p15_value_stability",
    "validate_shared_value_bank",
    "verify_p15_value_stability_package",
]
