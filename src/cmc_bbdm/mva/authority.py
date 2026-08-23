"""Cross-bound P1/MGMR authority and fresh FULL baseline reproduction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmc_bbdm.mgmr.authority import MGMRM0Authority, load_authority
from cmc_bbdm.mgmr.evaluation import FitRecord, nested_lodo_predictions
from cmc_bbdm.mgmr.m0_components import RegisteredB0, load_registered_b0
from cmc_bbdm.mgmr.protocol import MGMRProtocol, load_protocol

from .config import MVAConfig


class MVAAuthorityError(ValueError):
    """Raised when the frozen cohort or FULL prediction authority drifts."""


def _readonly(value: object, *, dtype: object) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    output.setflags(write=False)
    return output


def _state(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(
                json.dumps(value.shape, separators=(",", ":")).encode("ascii")
            )
            digest.update(value.tobytes(order="C"))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MVAAuthority:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    images: tuple[np.ndarray, ...]
    image_sha256: tuple[str, ...]
    targets: np.ndarray
    metadata13: np.ndarray
    full_embeddings: np.ndarray
    state_sha256: str
    _protocol: MGMRProtocol
    _authority: MGMRM0Authority
    _registered_b0: RegisteredB0

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)


@dataclass(frozen=True, slots=True)
class FullBaselineReproduction:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    targets: np.ndarray
    predictions: np.ndarray
    registered_predictions: np.ndarray
    selected_pca_dimensions: tuple[int, ...]
    domain_mae: tuple[float, ...]
    equal_domain_mae: float
    maximum_prediction_delta: float
    fit_records: tuple[FitRecord, ...]
    state_sha256: str


def load_mva_authority(config: MVAConfig, *, project_root: str | Path) -> MVAAuthority:
    """Load the exact P1 cohort through its existing MGMR source bindings."""

    if type(config) is not MVAConfig:
        raise MVAAuthorityError("issued MVAConfig is required")
    root = Path(project_root).resolve(strict=True)
    protocol = load_protocol(
        root / config.sources["mgmr_config"].path, project_root=root
    )
    authority = load_authority(protocol, project_root=root)
    registered = load_registered_b0(protocol, authority, project_root=root)
    if (
        authority.specimen_count != config.specimen_count
        or protocol.domain_order != config.domain_order
        or abs(registered.equal_domain_mae - config.full_mae)
        > config.baseline_tolerance
    ):
        raise MVAAuthorityError("P1/MGMR authority changed")
    state = _state(
        "mva-authority",
        authority.state_sha256,
        registered.state_sha256,
        config.domain_order,
        config.specimen_count,
    )
    return MVAAuthority(
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        images=authority.images,
        image_sha256=authority.image_sha256,
        targets=authority.targets,
        metadata13=authority.metadata13,
        full_embeddings=authority.full_global,
        state_sha256=state,
        _protocol=protocol,
        _authority=authority,
        _registered_b0=registered,
    )


def reproduce_full_baseline(
    config: MVAConfig, authority: MVAAuthority
) -> FullBaselineReproduction:
    """Freshly recompute nested LODO and stop if P1 changes beyond tolerance."""

    if type(config) is not MVAConfig or type(authority) is not MVAAuthority:
        raise MVAAuthorityError("issued config and authority are required")
    run = nested_lodo_predictions(
        method="MVA_FULL_P_A",
        metadata=authority.metadata13,
        blocks={"internal_cscan": authority.full_embeddings},
        targets=authority.targets,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        domain_order=config.domain_order,
        pca_dimensions=config.pca_dimensions,
        ridge_alpha=config.ridge_alpha,
        tie_tolerance=1.0e-12,
    )
    domains = np.asarray(authority.dataset_ids, dtype=object)
    domain_mae = tuple(
        float(
            np.mean(
                np.abs(
                    authority.targets[domains == domain]
                    - run.predictions[domains == domain]
                ),
                dtype=np.float64,
            )
        )
        for domain in config.domain_order
    )
    equal_mae = float(sum(domain_mae) / len(domain_mae))
    maximum_delta = float(
        np.max(np.abs(run.predictions - authority._registered_b0.predictions))
    )
    if (
        abs(equal_mae - config.full_mae) > config.baseline_tolerance
        or maximum_delta > config.baseline_tolerance
    ):
        raise MVAAuthorityError("fresh FULL baseline reproduction failed")
    dimensions = tuple(
        int(run.selection_by_domain[domain].dimensions[0])
        for domain in config.domain_order
    )
    targets = _readonly(authority.targets, dtype="<f8")
    predictions = _readonly(run.predictions, dtype="<f8")
    registered_predictions = _readonly(
        authority._registered_b0.predictions, dtype="<f8"
    )
    state = _state(
        "mva-full-baseline",
        authority.state_sha256,
        dimensions,
        domain_mae,
        equal_mae,
        maximum_delta,
        predictions,
    )
    return FullBaselineReproduction(
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        targets=targets,
        predictions=predictions,
        registered_predictions=registered_predictions,
        selected_pca_dimensions=dimensions,
        domain_mae=domain_mae,
        equal_domain_mae=equal_mae,
        maximum_prediction_delta=maximum_delta,
        fit_records=run.fit_records,
        state_sha256=state,
    )


__all__ = [
    "FullBaselineReproduction",
    "MVAAuthority",
    "MVAAuthorityError",
    "load_mva_authority",
    "reproduce_full_baseline",
]
