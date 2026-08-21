"""Nested-domain Optuna search control plane for D8."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import optuna

from .authority import D8InnerFold, D8SearchView, issue_inner_fold, validate_search_view
from .config import D8Config
from .regression import CandidateSpec
from .tracking import write_trial_index
from .variants import MorphologyThresholds, evaluate_candidate_acceptance


class D8TrialFailure(RuntimeError):
    """Raised when one trial fails while the study remains valid."""


@dataclass(frozen=True, slots=True)
class InnerEvaluation:
    """One query-domain candidate score with proposal-acceptance evidence."""

    query_domain: str
    mae: float
    accepted_proposals: int
    proposed_variants: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.query_domain) is not str or not self.query_domain:
            raise ValueError("inner evaluation domain is invalid")
        if type(self.mae) is not float or not math.isfinite(self.mae) or self.mae < 0.0:
            raise ValueError("inner evaluation MAE is invalid")
        if (
            type(self.accepted_proposals) is not int
            or type(self.proposed_variants) is not int
            or self.proposed_variants < 1
            or not 0 <= self.accepted_proposals <= self.proposed_variants
        ):
            raise ValueError("inner evaluation acceptance counts are invalid")
        if (
            type(self.evidence_sha256) is not str
            or len(self.evidence_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.evidence_sha256)
        ):
            raise ValueError("inner evaluation evidence SHA-256 is invalid")


@dataclass(frozen=True, slots=True)
class D8Candidate:
    """Canonical control/decomposition/gate/feature/regressor search candidate."""

    control_id: str
    decomposition_family: str
    band: str
    decomposition_parameters: Mapping[str, object]
    alpha: float
    K_train: int
    K_test: int
    thresholds: MorphologyThresholds
    feature_layer: str
    feature_aggregation: str
    prediction_aggregation: str
    morphology_beta: float | None
    consistency: str
    consistency_weight: float
    regressor_spec: CandidateSpec
    seed: int
    config_sha256: str
    marginalization_stage: str = "feature"
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if self.marginalization_stage not in {"feature", "prediction"}:
            raise ValueError("marginalization stage is not registered")
        payload = self.to_payload(include_state=False)
        state = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
        ).hexdigest()
        object.__setattr__(
            self,
            "decomposition_parameters",
            MappingProxyType(dict(self.decomposition_parameters)),
        )
        object.__setattr__(self, "state_sha256", state)

    @property
    def canonical_sha256(self) -> str:
        return self.state_sha256

    def to_payload(self, *, include_state: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "control_id": self.control_id,
            "decomposition_family": self.decomposition_family,
            "band": self.band,
            "decomposition_parameters": dict(self.decomposition_parameters),
            "alpha": self.alpha,
            "K_train": self.K_train,
            "K_test": self.K_test,
            "thresholds": asdict(self.thresholds),
            "feature_layer": self.feature_layer,
            "feature_aggregation": self.feature_aggregation,
            "prediction_aggregation": self.prediction_aggregation,
            "marginalization_stage": self.marginalization_stage,
            "morphology_beta": self.morphology_beta,
            "consistency": self.consistency,
            "consistency_weight": self.consistency_weight,
            "regressor_spec": {
                "pca_dimension": self.regressor_spec.pca_dimension,
                "regressor": self.regressor_spec.regressor,
                "parameters": dict(self.regressor_spec.parameters),
                "seed": self.regressor_spec.seed,
            },
            "seed": self.seed,
            "config_sha256": self.config_sha256,
        }
        if include_state:
            payload["state_sha256"] = self.state_sha256
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> D8Candidate:
        if not isinstance(payload, Mapping):
            raise TypeError("candidate payload is invalid")
        values = dict(payload)
        state = values.pop("state_sha256", None)
        thresholds = values.pop("thresholds", None)
        regressor = values.pop("regressor_spec", None)
        if not isinstance(thresholds, Mapping) or not isinstance(regressor, Mapping):
            raise TypeError("candidate payload is incomplete")
        candidate = cls(
            thresholds=MorphologyThresholds(**dict(thresholds)),
            regressor_spec=CandidateSpec(**dict(regressor)),
            **values,
        )
        if state is not None and state != candidate.state_sha256:
            raise ValueError("candidate payload state changed")
        return candidate


@dataclass(frozen=True, slots=True)
class SearchResult:
    outer_domain: str
    initial_trial_count: int
    trial_count: int
    completed_count: int
    pruned_count: int
    failed_count: int
    selected_candidates: tuple[D8Candidate, ...]
    study_database: str


def robust_inner_objective(domain_mae: np.ndarray) -> float:
    """Compute the frozen five-domain robust search objective."""

    if np.iscomplexobj(domain_mae):
        raise ValueError("inner MAEs must be real")
    values = np.asarray(domain_mae, dtype=np.float64)
    if values.shape != (5,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("the robust objective requires five finite domain MAEs")
    mean = math.fsum(float(value) for value in values) / len(values)
    return float(mean + 0.25 * float(np.max(values)) + 0.10 * float(np.std(values)))


def _range(mapping: Mapping[str, object]) -> tuple[float, float, bool]:
    return (
        float(mapping["minimum"]),
        float(mapping["maximum"]),
        mapping["distribution"] == "log_uniform",
    )


def _regressor_spec(
    trial: optuna.Trial, *, seed: int, config: D8Config
) -> CandidateSpec:
    spaces = config.search_space
    regressors = spaces["regressors"]
    name = trial.suggest_categorical("regressor", tuple(regressors["registered"]))
    pca = trial.suggest_categorical(
        "pca_dimension", tuple(regressors["pca_dimensions"])
    )
    if name == "ridge":
        parameters = {"alpha": trial.suggest_float("ridge_alpha", 1e-3, 1e3, log=True)}
    elif name == "elastic_net":
        parameters = {
            "alpha": trial.suggest_float("elastic_net_alpha", 1e-4, 1.0, log=True),
            "l1_ratio": trial.suggest_float("elastic_net_l1_ratio", 0.0, 1.0),
        }
    elif name == "pls":
        parameters = {"n_components": trial.suggest_int("pls_components", 1, 8)}
    elif name == "huber":
        parameters = {
            "alpha": trial.suggest_float("huber_alpha", 1e-6, 1.0, log=True),
            "epsilon": trial.suggest_float("huber_epsilon", 1.0, 2.5),
        }
    elif name == "kernel_ridge":
        parameters = {
            "alpha": trial.suggest_float("kernel_ridge_alpha", 1e-4, 1e2, log=True),
            "gamma": trial.suggest_float("kernel_ridge_gamma", 1e-4, 1e2, log=True),
        }
    elif name == "svr":
        parameters = {
            "C": trial.suggest_float("svr_C", 1e-3, 1e3, log=True),
            "epsilon": trial.suggest_float("svr_epsilon", 0.0, 0.25),
            "gamma": trial.suggest_float("svr_gamma", 1e-4, 1e2, log=True),
        }
    elif name == "hist_gradient_boosting":
        parameters = {
            "l2_regularization": trial.suggest_float("hgb_l2", 0.0, 10.0),
            "learning_rate": trial.suggest_float(
                "hgb_learning_rate", 0.01, 0.3, log=True
            ),
            "max_leaf_nodes": trial.suggest_int("hgb_max_leaf_nodes", 4, 31),
        }
    else:
        parameters = {
            "alpha": trial.suggest_float("mlp_alpha", 1e-5, 1.0, log=True),
            "hidden_layer_size": trial.suggest_categorical(
                "mlp_hidden_layer_size", (8, 16, 32, 64)
            ),
        }
    return CandidateSpec(
        pca_dimension=int(pca),
        regressor=str(name),
        parameters=parameters,
        seed=seed,
    )


def suggest_candidate(trial: optuna.Trial, config: D8Config) -> D8Candidate:
    """Suggest one valid registered D8 search candidate."""

    if type(config) is not D8Config:
        raise TypeError("candidate suggestion requires exact D8Config")
    spaces = config.search_space
    controls = spaces["controls"]
    decomposition = spaces["decomposition"]
    gate = spaces["morphology_gate"]
    features = spaces["features"]
    control = str(
        trial.suggest_categorical("control_id", tuple(controls["registered"]))
    )
    seed = int(config.seed + trial.number)
    if control == "B0":
        return D8Candidate(
            control_id=control,
            decomposition_family="gaussian",
            band="low",
            decomposition_parameters={"sigma": 2.0},
            alpha=0.0,
            K_train=1,
            K_test=1,
            thresholds=MorphologyThresholds(
                area_relative_deviation=0.10,
                width_relative_deviation=0.10,
                height_relative_deviation=0.10,
                centroid_shift_mm=2.0,
                low_frequency_correlation_minimum=0.95,
                radial_spearman_minimum=0.90,
            ),
            feature_layer="global",
            feature_aggregation="mean",
            prediction_aggregation="mean",
            morphology_beta=None,
            consistency="none",
            consistency_weight=0.0,
            regressor_spec=CandidateSpec(
                pca_dimension=8,
                regressor="ridge",
                parameters={"alpha": 10.0},
                seed=seed,
            ),
            seed=seed,
            config_sha256=config.config_sha256,
            marginalization_stage="feature",
        )

    family = (
        "gaussian"
        if control == "B1"
        else str(
            trial.suggest_categorical(
                "decomposition_family", tuple(decomposition["families"])
            )
        )
    )
    band = (
        "low"
        if control == "B1"
        else str(trial.suggest_categorical("band", tuple(decomposition["bands"])))
    )
    if family == "gaussian":
        minimum, maximum, log = _range(decomposition["gaussian_sigma_pixels"])
        parameters: dict[str, object] = {
            "sigma": trial.suggest_float("gaussian_sigma", minimum, maximum, log=log)
        }
    elif family == "fourier":
        low, high, log = _range(decomposition["fourier_cutoff_fraction"])
        t_low, t_high, t_log = _range(decomposition["fourier_transition_fraction"])
        parameters = {
            "cutoff": trial.suggest_float("fourier_cutoff", low, high, log=log),
            "transition": trial.suggest_float(
                "fourier_transition", t_low, t_high, log=t_log
            ),
        }
    else:
        parameters = {
            "wavelet": trial.suggest_categorical(
                "wavelet", tuple(decomposition["wavelets"])
            ),
            "level": trial.suggest_categorical(
                "wavelet_level", tuple(decomposition["wavelet_levels"])
            ),
        }
    if control == "B1":
        alpha = 1.0
    else:
        low, high, log = _range(decomposition["alpha"])
        alpha = trial.suggest_float("alpha", low, high, log=log)
    thresholds = MorphologyThresholds(
        area_relative_deviation=float(
            trial.suggest_categorical(
                "area_relative_deviation", tuple(gate["area_relative_deviation"])
            )
        ),
        width_relative_deviation=float(
            trial.suggest_categorical(
                "width_relative_deviation", tuple(gate["width_relative_deviation"])
            )
        ),
        height_relative_deviation=float(
            trial.suggest_categorical(
                "height_relative_deviation", tuple(gate["height_relative_deviation"])
            )
        ),
        centroid_shift_mm=float(
            trial.suggest_categorical(
                "centroid_shift_mm", tuple(gate["centroid_shift_mm"])
            )
        ),
        low_frequency_correlation_minimum=float(
            trial.suggest_categorical(
                "low_frequency_correlation_minimum",
                tuple(gate["low_frequency_correlation_minimum"]),
            )
        ),
        radial_spearman_minimum=float(
            trial.suggest_categorical(
                "radial_spearman_minimum", tuple(gate["radial_spearman_minimum"])
            )
        ),
        low_frequency_sigma_pixels=float(gate["low_frequency_sigma_pixels"]),
        radial_profile_bins=int(gate["radial_profile_bins"]),
    )
    if control == "B1":
        k_train = 1
        k_test = 1
    else:
        k_train = int(
            trial.suggest_categorical("K_train", tuple(features["K_train"]))
        )
        k_test = (
            1
            if control in {"B5", "B6"}
            else int(trial.suggest_categorical("K_test", tuple(features["K_test"])))
        )
    feature_layer = str(
        trial.suggest_categorical("feature_layer", tuple(features["layers"]))
    )
    if control in {"B1", "B5", "B6"}:
        marginalization_stage = "feature"
    elif control == "B7":
        marginalization_stage = "prediction"
    else:
        marginalization_stage = str(
            trial.suggest_categorical(
                "marginalization_stage",
                tuple(features["marginalization_stage"]),
            )
        )
    if marginalization_stage == "feature":
        feature_aggregation = str(
            trial.suggest_categorical(
                "feature_aggregation", tuple(features["aggregation"])
            )
        )
        prediction_aggregation = "mean"
    else:
        feature_aggregation = "mean"
        prediction_aggregation = str(
            trial.suggest_categorical(
                "prediction_aggregation",
                tuple(features["prediction_aggregation"]),
            )
        )
    if prediction_aggregation == "morphology_weighted":
        low, high, log = _range(features["morphology_weight_beta"])
        beta: float | None = trial.suggest_float("morphology_beta", low, high, log=log)
    else:
        beta = None
    if control in {"B6", "B8"}:
        consistency = str(
            trial.suggest_categorical("consistency", tuple(features["consistency"]))
        )
        if control == "B6" and consistency == "none":
            trial.set_user_attr(
                "failure_reason", "invalid_combination:B6_requires_consistency"
            )
            raise optuna.TrialPruned("B6 requires consistency")
    else:
        consistency = "none"
    if (
        marginalization_stage == "feature"
        and consistency not in {"none", "feature_variance"}
    ) or (
        marginalization_stage == "prediction" and consistency == "feature_variance"
    ):
        trial.set_user_attr(
            "failure_reason", "invalid_combination:consistency_stage_mismatch"
        )
        raise optuna.TrialPruned("consistency stage mismatch")
    if consistency == "none":
        consistency_weight = 0.0
    else:
        low, high, log = _range(features["consistency_weight"])
        consistency_weight = trial.suggest_float(
            "consistency_weight", low, high, log=log
        )
    return D8Candidate(
        control_id=control,
        decomposition_family=family,
        band=band,
        decomposition_parameters=parameters,
        alpha=float(alpha),
        K_train=k_train,
        K_test=k_test,
        thresholds=thresholds,
        feature_layer=feature_layer,
        feature_aggregation=feature_aggregation,
        prediction_aggregation=prediction_aggregation,
        morphology_beta=beta,
        consistency=consistency,
        consistency_weight=float(consistency_weight),
        regressor_spec=_regressor_spec(trial, seed=seed, config=config),
        seed=seed,
        config_sha256=config.config_sha256,
        marginalization_stage=marginalization_stage,
    )


def _warm_trials() -> tuple[dict[str, object], ...]:
    common: dict[str, object] = {
        "area_relative_deviation": 0.1,
        "width_relative_deviation": 0.1,
        "height_relative_deviation": 0.1,
        "centroid_shift_mm": 2.0,
        "low_frequency_correlation_minimum": 0.95,
        "radial_spearman_minimum": 0.90,
        "K_train": 4,
        "K_test": 8,
        "feature_layer": "global",
        "feature_aggregation": "mean",
        "prediction_aggregation": "mean",
        "pca_dimension": 8,
        "regressor": "ridge",
        "ridge_alpha": 10.0,
    }
    rows: list[dict[str, object]] = [{"control_id": "B0"}]
    specifications = (
        ("B1", "gaussian", "low", None, "feature"),
        ("B2", "gaussian", "high", 0.10, "feature"),
        ("B3", "fourier", "high", 0.10, "feature"),
        ("B4", "wavelet", "high", 0.10, "feature"),
        ("B5", "gaussian", "high", 0.10, "feature"),
        ("B5", "fourier", "high", 0.10, "feature"),
        ("B5", "wavelet", "high", 0.10, "feature"),
        ("B6", "gaussian", "high", 0.10, "feature"),
        ("B7", "fourier", "mid", 0.10, "prediction"),
        ("B8", "wavelet", "mid+high", 0.10, "feature"),
        ("B8", "gaussian", "high", -0.10, "prediction"),
    )
    for control, family, band, alpha, stage in specifications:
        row = dict(common)
        row.update({"control_id": control, "decomposition_family": family})
        if control in {"B2", "B3", "B4", "B8"}:
            row["marginalization_stage"] = stage
        if stage == "prediction":
            row.pop("feature_aggregation", None)
        else:
            row.pop("prediction_aggregation", None)
        if control != "B1":
            row["band"] = band
            row["alpha"] = alpha
        if family == "gaussian":
            row["gaussian_sigma"] = 2.0
        elif family == "fourier":
            row["fourier_cutoff"] = 0.20
            row["fourier_transition"] = 0.05
        else:
            row["wavelet"] = "db2"
            row["wavelet_level"] = 2
        if control in {"B5", "B6"}:
            row.pop("K_test", None)
            row.pop("prediction_aggregation", None)
        if control == "B6":
            row["consistency"] = "feature_variance"
            row["consistency_weight"] = 0.10
        elif control == "B8":
            row["consistency"] = "pairwise_ranking"
            row["consistency_weight"] = 0.10
        rows.append(row)
    return tuple(rows)


def _study_attributes(view: D8SearchView, config: D8Config) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "d8_inner_search",
        "outer_domain": view.outer_domain,
        "config_sha256": config.config_sha256,
        "search_view_sha256": view.state_sha256,
        "forced_trials": config.forced_trials,
        "optuna_trials": config.optuna_trials,
    }


def _candidate_from_trial(trial: optuna.trial.FrozenTrial) -> D8Candidate:
    return D8Candidate.from_payload(trial.user_attrs["candidate"])


def _validate_trial_candidates(
    study: optuna.Study, *, config: D8Config
) -> None:
    finished_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
    }
    for trial in study.trials:
        if trial.state not in finished_states:
            continue
        replay = optuna.trial.FixedTrial(dict(trial.params), number=trial.number)
        try:
            expected = suggest_candidate(replay, config)
        except optuna.TrialPruned as error:
            expected_reason = replay.user_attrs.get("failure_reason")
            if (
                trial.state != optuna.trial.TrialState.PRUNED
                or "candidate" in trial.user_attrs
                or not isinstance(expected_reason, str)
                or trial.user_attrs.get("failure_reason") != expected_reason
            ):
                raise ValueError("recorded parameters do not match pruned candidate") from error
            continue
        recorded = trial.user_attrs.get("candidate")
        if not isinstance(recorded, Mapping):
            raise TypeError("recorded parameters are missing their candidate")
        if dict(recorded) != expected.to_payload():
            raise ValueError("candidate differs from recorded parameters")


def run_outer_search(
    view: D8SearchView,
    *,
    config: D8Config,
    output: Path,
    evaluator: Callable[[D8Candidate, D8InnerFold], InnerEvaluation],
) -> SearchResult:
    """Run or resume one five-domain inner search without an outer evaluation view."""

    validate_search_view(view)
    if type(config) is not D8Config or view.config_sha256 != config.config_sha256:
        raise ValueError("search config differs from the issued search view")
    if not callable(evaluator):
        raise TypeError("search evaluator must be callable")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    database = (root / "study.db").resolve()
    storage = f"sqlite:///{database}"
    study_name = f"d8::{view.outer_domain}"
    sampler = optuna.samplers.TPESampler(seed=config.seed, multivariate=False)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    expected_attributes = _study_attributes(view, config)
    if study.user_attrs:
        if study.user_attrs != expected_attributes:
            raise ValueError("existing D8 study authority changed")
    else:
        for key, value in expected_attributes.items():
            study.set_user_attr(key, value)
    _validate_trial_candidates(study, config=config)
    if not study.trials:
        warm = _warm_trials()
        if len(warm) != config.forced_trials:
            raise ValueError("registered warm-start count changed")
        for parameters in warm:
            study.enqueue_trial(parameters)

    query_domains = tuple(
        domain for domain in config.outer_domains if domain != view.outer_domain
    )

    def objective(trial: optuna.Trial) -> float:
        started = time.perf_counter()
        trial.set_user_attr("outer_fold", view.outer_domain)
        trial.set_user_attr("config_sha256", config.config_sha256)
        trial.set_user_attr("search_view_sha256", view.state_sha256)
        try:
            candidate = suggest_candidate(trial, config)
        except optuna.TrialPruned:
            trial.set_user_attr("runtime_seconds", time.perf_counter() - started)
            raise
        trial.set_user_attr("candidate", candidate.to_payload())
        evaluations: list[InnerEvaluation] = []
        try:
            for domain in query_domains:
                fold = issue_inner_fold(view, query_domain=domain)
                evaluation = evaluator(candidate, fold)
                if (
                    type(evaluation) is not InnerEvaluation
                    or evaluation.query_domain != domain
                ):
                    raise ValueError("inner evaluator returned the wrong domain")
                evaluations.append(evaluation)
        except Exception as error:
            trial.set_user_attr("failure_reason", f"{type(error).__name__}: {error}")
            trial.set_user_attr("runtime_seconds", time.perf_counter() - started)
            raise D8TrialFailure("D8 inner evaluation failed") from error
        counts = {
            item.query_domain: (item.accepted_proposals, item.proposed_variants)
            for item in evaluations
        }
        acceptance = evaluate_candidate_acceptance(
            counts, minimum_overall=0.80, minimum_domain=0.60
        )
        accepted_total = sum(accepted for accepted, _ in counts.values())
        proposed_total = sum(proposed for _, proposed in counts.values())
        acceptance_by_domain = {
            domain: {
                "accepted_proposals": accepted,
                "proposed_variants": proposed,
                "acceptance_rate": accepted / proposed,
            }
            for domain, (accepted, proposed) in counts.items()
        }
        inner = {item.query_domain: item.mae for item in evaluations}
        values = np.asarray(
            [inner[domain] for domain in query_domains], dtype=np.float64
        )
        mean = math.fsum(float(value) for value in values) / len(values)
        worst = float(np.max(values))
        deviation = float(np.std(values))
        score = robust_inner_objective(values)
        evidence = hashlib.sha256(
            "\0".join(item.evidence_sha256 for item in evaluations).encode("ascii")
        ).hexdigest()
        trial.set_user_attr("inner_mae", inner)
        trial.set_user_attr("mean_mae", mean)
        trial.set_user_attr("worst_mae", worst)
        trial.set_user_attr("domain_sd", deviation)
        trial.set_user_attr("accepted_proposals", accepted_total)
        trial.set_user_attr("proposed_variants", proposed_total)
        trial.set_user_attr("acceptance_rate", acceptance.overall_rate)
        trial.set_user_attr("acceptance_by_domain", acceptance_by_domain)
        trial.set_user_attr("evidence_sha256", evidence)
        trial.set_user_attr("runtime_seconds", time.perf_counter() - started)
        if not acceptance.eligible:
            trial.set_user_attr(
                "failure_reason",
                "morphology_acceptance:" + ",".join(acceptance.failed_domains),
            )
            raise optuna.TrialPruned("morphology acceptance gate failed")
        return score

    finished_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
    }
    finished = sum(trial.state in finished_states for trial in study.trials)
    target = config.forced_trials + config.optuna_trials
    if finished < target:
        study.optimize(
            objective,
            n_trials=target - finished,
            catch=(D8TrialFailure,),
            gc_after_trial=True,
            show_progress_bar=False,
        )
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    studies = tuple(
        optuna.load_study(study_name=item.study_name, storage=storage)
        for item in summaries
        if item.study_name.startswith("d8::")
    )
    write_trial_index(studies, root / "trial_index.csv")
    study = optuna.load_study(study_name=study_name, storage=storage)
    _validate_trial_candidates(study, config=config)
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    ranked = sorted(
        completed,
        key=lambda trial: (
            float(trial.value),
            str(trial.user_attrs["candidate"]["state_sha256"]),
        ),
    )
    selected: list[D8Candidate] = []
    observed: set[str] = set()
    for trial in ranked:
        candidate = _candidate_from_trial(trial)
        if candidate.state_sha256 in observed:
            continue
        observed.add(candidate.state_sha256)
        selected.append(candidate)
        if len(selected) == 12:
            break
    if len(selected) < 12:
        raise ValueError("D8 search produced fewer than twelve valid candidates")
    return SearchResult(
        outer_domain=view.outer_domain,
        initial_trial_count=target,
        trial_count=sum(trial.state in finished_states for trial in study.trials),
        completed_count=sum(
            trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
        ),
        pruned_count=sum(
            trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials
        ),
        failed_count=sum(
            trial.state == optuna.trial.TrialState.FAIL for trial in study.trials
        ),
        selected_candidates=tuple(selected[:12]),
        study_database=database.relative_to(root.resolve()).as_posix(),
    )


__all__ = [
    "D8Candidate",
    "D8TrialFailure",
    "InnerEvaluation",
    "SearchResult",
    "robust_inner_objective",
    "run_outer_search",
    "suggest_candidate",
]
