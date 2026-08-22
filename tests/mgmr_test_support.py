from __future__ import annotations

import hashlib
from types import MappingProxyType

import numpy as np

from cmc_bbdm.mgmr.evaluation import PredictionRecord
from cmc_bbdm.mgmr.formal_outer import M0FormalResult
from cmc_bbdm.mgmr.protocol import MGMRProtocol
from cmc_bbdm.mgmr.statistics import (
    decide_m0,
    paired_domain_bootstrap,
    prediction_metrics,
)


def synthetic_formal(protocol: MGMRProtocol) -> M0FormalResult:
    domain_errors = {
        "B0": (0.08,) * 6,
        "B1": (0.10,) * 6,
        "B2": (0.12,) * 6,
        "B3": (0.09, 0.09, 0.09, 0.09, 0.11, 0.11),
        "B4": (0.07,) * 6,
        "R_coarse": (0.09, 0.09, 0.09, 0.09, 0.11, 0.11),
        "R_full": (0.07,) * 6,
        "P3_20260831": (0.099,) * 6,
        "P3_20260901": (0.0995,) * 6,
        "P3_20260902": (0.0998,) * 6,
    }
    specimen_ids: list[str] = []
    dataset_ids: list[str] = []
    targets: list[float] = []
    for domain_index, domain in enumerate(protocol.domain_order):
        for row in range(46):
            specimen_ids.append(f"s{domain_index}-{row:02d}")
            dataset_ids.append(domain)
            targets.append(0.5 + 0.01 * row + 0.02 * domain_index)
    target_array = np.asarray(targets, dtype=np.float64)
    records: dict[str, tuple[PredictionRecord, ...]] = {}
    for method, errors in domain_errors.items():
        rows = tuple(
            PredictionRecord(
                method,
                specimen_ids[index],
                dataset_ids[index],
                float(target_array[index]),
                float(target_array[index] + errors[protocol.domain_order.index(dataset_ids[index])]),
                (8,),
            )
            for index in range(276)
        )
        records[method] = rows
    signal_pairs = {
        "S_R_coarse": ("B1", "R_coarse"),
        "S_R_full": ("B0", "R_full"),
        "S_P3_20260831": ("B1", "P3_20260831"),
        "S_P3_20260901": ("B1", "P3_20260901"),
        "S_P3_20260902": ("B1", "P3_20260902"),
    }
    for method, (baseline, corrected) in signal_pairs.items():
        rows: list[PredictionRecord] = []
        for index, (base_row, corrected_row) in enumerate(
            zip(records[baseline], records[corrected], strict=True)
        ):
            rows.append(
                PredictionRecord(
                    method,
                    specimen_ids[index],
                    dataset_ids[index],
                    base_row.target - base_row.prediction,
                    corrected_row.prediction - base_row.prediction,
                    (8,),
                )
            )
        records[method] = tuple(rows)
    metrics = {
        method: prediction_metrics(rows, domain_order=protocol.domain_order)
        for method, rows in records.items()
    }
    shuffled = {
        seed: metrics[f"P3_{seed}"] for seed in protocol.specificity_seeds
    }
    decision = decide_m0(
        direct={method: metrics[method] for method in ("B1", "B2", "B3")},
        coarse_baseline=metrics["B1"],
        coarse_corrected=metrics["R_coarse"],
        full_baseline=metrics["B0"],
        full_corrected=metrics["R_full"],
        shuffled=shuffled,
        required_gates=protocol.gate_required,
        minimum_positive_domains=protocol.minimum_positive_domains,
    )
    def effect(reference: str, candidate: str) -> tuple[float, ...]:
        return tuple(
            left.mae - right.mae
            for left, right in zip(
                metrics[reference].domain_metrics,
                metrics[candidate].domain_metrics,
                strict=True,
            )
        )

    effects = {
        "B1_minus_B3": effect("B1", "B3"),
        "B2_minus_B3": effect("B2", "B3"),
        "B0_minus_B4": effect("B0", "B4"),
        "B1_minus_R_coarse": effect("B1", "R_coarse"),
        "B0_minus_R_full": effect("B0", "R_full"),
    }
    for seed in protocol.specificity_seeds:
        name = f"P3_{seed}"
        effects[f"B1_minus_{name}"] = effect("B1", name)
        effects[f"real_minus_{name}"] = effect(name, "R_coarse")
    bootstrap = paired_domain_bootstrap(
        effects,
        domain_order=protocol.domain_order,
        seed=protocol.bootstrap_seed,
        resamples=protocol.bootstrap_resamples,
        quantiles=protocol.bootstrap_quantiles,
    )
    digest = hashlib.sha256(b"synthetic-mgmr-formal").hexdigest()
    return M0FormalResult(
        specimen_ids=tuple(specimen_ids),
        dataset_ids=tuple(dataset_ids),
        prediction_records=MappingProxyType(records),
        metrics=MappingProxyType(metrics),
        bootstrap=bootstrap,
        decision=decision,
        source_residuals=(),
        component_state_sha256="1" * 64,
        residual_state_sha256="2" * 64,
        state_sha256=digest,
    )
