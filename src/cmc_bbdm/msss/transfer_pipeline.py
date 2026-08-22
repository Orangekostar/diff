"""Authorization, execution, bootstrap, and gating for MSSS S2."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import S1PackageValidation, validate_s1_package
from .authority import MSSSAuthority
from .protocol import MSSSProtocol
from .scale_features import ScaleFeatureBank, build_condition_registry
from .statistics import CommonBootstrap, common_stratified_bootstrap
from .transfer_metrics import S2Gate, s2_gate, transfer_gain
from .transfer_tasks import (
    TransferTaskEvaluation,
    build_task_registry,
    evaluate_transfer_task,
)


class TransferPipelineError(ValueError):
    """Raised when S2 lacks authorization or reproducible transfer evidence."""


@dataclass(frozen=True, slots=True)
class AuthorizedS1:
    validation: S1PackageValidation
    bank: ScaleFeatureBank
    package_dir: Path


@dataclass(frozen=True, slots=True)
class TransferMetricRow:
    task_family: str
    task_id: str
    target_label: str
    comparator: str
    condition_id: str
    specimen_count: int
    domain_count: int
    mae: float
    equal_domain_mae: float
    worst_domain_mae: float
    full_equal_domain_mae: float
    tg: float
    rtg: float
    nonworse: bool
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class S2Run:
    authorization: S1PackageValidation
    evaluations: tuple[TransferTaskEvaluation, ...]
    metrics: tuple[TransferMetricRow, ...]
    bootstraps: tuple[tuple[str, CommonBootstrap], ...]
    gate: S2Gate
    state_sha256: str


StatusHook = Callable[[str], None]


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TransferPipelineError(f"invalid authorized S1 file: {path.name}") from error
    if type(value) is not dict:
        raise TransferPipelineError(f"invalid authorized S1 file: {path.name}")
    return value


def load_authorized_sampling_bank(
    s1_package: str | Path,
    *,
    protocol: MSSSProtocol,
    authority: MSSSAuthority,
    project_root: str | Path,
    config_path: str | Path,
) -> AuthorizedS1:
    """Validate formal S1 authorization and restore its sampling feature bank."""

    if type(protocol) is not MSSSProtocol or type(authority) is not MSSSAuthority:
        raise TransferPipelineError("issued protocol and authority are required")
    root = Path(s1_package).resolve(strict=True)
    validation = validate_s1_package(
        root, project_root=project_root, config_path=config_path
    )
    if validation.test_only or validation.gate_status not in {"GO", "STRONG_GO"}:
        raise TransferPipelineError("validated formal S1 GO is required for S2")
    index = _read_json(root / "feature_index.json")
    conditions_payload = index.get("conditions")
    provenance = index.get("encoder_provenance")
    if (
        index.get("specimen_ids") != list(authority.specimen_ids)
        or index.get("dataset_ids") != list(authority.dataset_ids)
        or type(conditions_payload) is not dict
        or type(provenance) is not dict
    ):
        raise TransferPipelineError("authorized S1 sampling roster changed")
    conditions = tuple(
        item for item in build_condition_registry(protocol) if item.axis == "sampling"
    )
    if set(conditions_payload) != {item.condition_id for item in conditions}:
        raise TransferPipelineError("authorized sampling condition registry changed")
    features: dict[str, np.ndarray] = {}
    transform_hashes: dict[str, str] = {}
    try:
        with np.load(root / "sampling_features.npz", allow_pickle=False) as archive:
            expected_keys = {
                str(conditions_payload[item.condition_id]["archive_key"])
                for item in conditions
            }
            if set(archive.files) != expected_keys:
                raise TransferPipelineError("authorized sampling archive keys changed")
            for condition in conditions:
                record = conditions_payload[condition.condition_id]
                if type(record) is not dict:
                    raise TransferPipelineError("authorized sampling index is invalid")
                key = record.get("archive_key")
                if type(key) is not str:
                    raise TransferPipelineError("authorized sampling archive key is invalid")
                features[condition.condition_id] = np.asarray(archive[key], dtype=np.float64)
                state = record.get("transform_state_sha256")
                if type(state) is not str:
                    raise TransferPipelineError("authorized transform state is invalid")
                transform_hashes[condition.condition_id] = state
    except TransferPipelineError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise TransferPipelineError("authorized sampling archive is unreadable") from error
    bank = ScaleFeatureBank.issue(
        conditions=conditions,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        features=features,
        transform_state_sha256=transform_hashes,
        encoder_provenance=provenance,
    )
    for condition in conditions:
        expected = conditions_payload[condition.condition_id].get("feature_sha256")
        if bank.feature_sha256[condition.condition_id] != expected:
            raise TransferPipelineError("authorized sampling feature hash changed")
    return AuthorizedS1(validation=validation, bank=bank, package_dir=root)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise TransferPipelineError("transfer aggregation is empty")
    result = float(math.fsum(float(value) for value in values) / len(values))
    if not math.isfinite(result):
        raise TransferPipelineError("transfer aggregation is non-finite")
    return result


def summarize_s2(
    protocol: MSSSProtocol,
    *,
    authorization: S1PackageValidation,
    evaluations: Sequence[TransferTaskEvaluation],
    bootstrap_resamples: int | None = None,
) -> S2Run:
    """Aggregate target predictions, bootstrap TG, and apply the frozen S2 gate."""

    registry = tuple(evaluations)
    if (
        type(protocol) is not MSSSProtocol
        or authorization.test_only
        or authorization.gate_status not in {"GO", "STRONG_GO"}
        or len(registry) != 11
        or tuple(sum(item.task.family == family for item in registry) for family in ("domain", "ply", "layup")) != (6, 3, 2)
    ):
        raise TransferPipelineError("S2 summary roster or authorization is invalid")
    resamples = protocol.bootstrap_resamples if bootstrap_resamples is None else bootstrap_resamples
    metric_rows: list[TransferMetricRow] = []
    bootstraps: list[tuple[str, CommonBootstrap]] = []
    source_msss_tg: dict[str, list[float]] = {"domain": [], "ply": [], "layup": []}
    for evaluation in registry:
        by_comparator: dict[str, dict[str, object]] = {}
        for row in evaluation.predictions:
            record = by_comparator.setdefault(
                row.comparator,
                {"condition_id": row.condition_id, "errors": {}, "datasets": {}},
            )
            if record["condition_id"] != row.condition_id:
                raise TransferPipelineError("comparator condition changed within task")
            errors = record["errors"]
            datasets = record["datasets"]
            if row.specimen_id in errors:
                raise TransferPipelineError("duplicate transfer prediction")
            errors[row.specimen_id] = row.absolute_error
            datasets[row.specimen_id] = row.dataset_id
        if set(by_comparator) != {"FULL", "FIXED_25", "SOURCE_MSSS", "OVER_COARSE"}:
            raise TransferPipelineError("transfer comparator roster is incomplete")
        specimen_order = tuple(by_comparator["FULL"]["errors"])
        dataset_values = tuple(by_comparator["FULL"]["datasets"][item] for item in specimen_order)
        domain_order = tuple(dict.fromkeys(dataset_values))
        full_errors = np.asarray(
            [by_comparator["FULL"]["errors"][item] for item in specimen_order], dtype=np.float64
        )
        effects: dict[str, np.ndarray] = {}
        for comparator, record in by_comparator.items():
            if tuple(record["errors"]) != specimen_order or tuple(record["datasets"][item] for item in specimen_order) != dataset_values:
                raise TransferPipelineError("transfer comparator prediction rosters differ")
            errors = np.asarray([record["errors"][item] for item in specimen_order], dtype=np.float64)
            effects[comparator] = full_errors - errors
        bootstrap = common_stratified_bootstrap(
            effects,
            groups=dataset_values,
            group_order=domain_order,
            seed=protocol.bootstrap_seed,
            resamples=resamples,
            quantiles=(0.025, 0.975),
        )
        bootstraps.append((evaluation.task.task_id, bootstrap))
        domain_indices = {
            group: np.flatnonzero(np.asarray(dataset_values, dtype=str) == group)
            for group in domain_order
        }
        full_domain = tuple(float(np.mean(full_errors[indices])) for indices in domain_indices.values())
        full_equal = _mean(full_domain)
        for comparator, record in by_comparator.items():
            errors = np.asarray([record["errors"][item] for item in specimen_order], dtype=np.float64)
            domain_mae = tuple(float(np.mean(errors[indices])) for indices in domain_indices.values())
            equal_mae = _mean(domain_mae)
            gain = transfer_gain(full_mae=full_equal, candidate_mae=equal_mae)
            interval = bootstrap.effects[comparator]
            if not math.isclose(gain.tg, interval.estimate, rel_tol=1.0e-11, abs_tol=1.0e-13):
                raise TransferPipelineError("bootstrap and direct transfer gains disagree")
            metric_rows.append(
                TransferMetricRow(
                    task_family=evaluation.task.family,
                    task_id=evaluation.task.task_id,
                    target_label=evaluation.task.target_label,
                    comparator=comparator,
                    condition_id=str(record["condition_id"]),
                    specimen_count=len(errors),
                    domain_count=len(domain_order),
                    mae=float(np.mean(errors)),
                    equal_domain_mae=equal_mae,
                    worst_domain_mae=max(domain_mae),
                    full_equal_domain_mae=full_equal,
                    tg=gain.tg,
                    rtg=gain.rtg,
                    nonworse=gain.nonworse,
                    ci_low=interval.low,
                    ci_high=interval.high,
                )
            )
            if comparator == "SOURCE_MSSS":
                source_msss_tg[evaluation.task.family].append(gain.tg)
    gate = s2_gate(
        domain_tg=source_msss_tg["domain"],
        ply_tg=source_msss_tg["ply"],
        layup_tg=source_msss_tg["layup"],
    )
    state = hashlib.sha256(
        json.dumps(
            {
                "authorization": authorization.scientific_digest,
                "evaluations": [item.state_sha256 for item in registry],
                "metrics": [
                    (item.task_id, item.comparator, item.equal_domain_mae, item.tg)
                    for item in metric_rows
                ],
                "draws": [item.draws_sha256 for _, item in bootstraps],
                "gate": gate.status,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return S2Run(
        authorization=authorization,
        evaluations=registry,
        metrics=tuple(metric_rows),
        bootstraps=tuple(bootstraps),
        gate=gate,
        state_sha256=state,
    )


def run_s2_experiment(
    protocol: MSSSProtocol,
    authority: MSSSAuthority,
    *,
    s1_package: str | Path,
    project_root: str | Path,
    config_path: str | Path,
    status_hook: StatusHook | None = None,
) -> tuple[AuthorizedS1, S2Run]:
    """Execute the eleven registered transfer tasks after formal S1 authorization."""

    if status_hook is not None:
        status_hook("validating S1 authorization and sampling feature archive")
    authorized = load_authorized_sampling_bank(
        s1_package,
        protocol=protocol,
        authority=authority,
        project_root=project_root,
        config_path=config_path,
    )
    tasks = build_task_registry(
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        ply_count=authority.ply_count,
        layup_family=authority.layup_family,
        domain_order=protocol.domain_order,
    )
    evaluations = []
    for task in tasks:
        if status_hook is not None:
            status_hook(f"evaluating transfer task {task.task_id}")
        evaluations.append(
            evaluate_transfer_task(
                authorized.bank,
                targets=authority.targets,
                metadata13=authority.metadata13,
                task=task,
                pca_dimensions=protocol.pca_dimensions,
                primary_margin=protocol.primary_margin,
                margins=protocol.noninferiority_margins,
            )
        )
    if status_hook is not None:
        status_hook(f"aggregating S2 with {protocol.bootstrap_resamples} bootstrap resamples")
    return authorized, summarize_s2(
        protocol,
        authorization=authorized.validation,
        evaluations=tuple(evaluations),
    )


__all__ = [
    "AuthorizedS1",
    "S2Run",
    "TransferMetricRow",
    "TransferPipelineError",
    "load_authorized_sampling_bank",
    "run_s2_experiment",
    "summarize_s2",
]
