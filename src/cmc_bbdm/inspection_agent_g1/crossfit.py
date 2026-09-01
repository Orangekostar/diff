"""Strict outer/source cross-fitting for G1 teacher dependencies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.inspection_agent.cai_assessor import (
    StateCAIAssessor,
    StateFeatureRow,
    fit_state_cai_assessor,
)
from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    SourceBackgroundPrior,
)
from cmc_bbdm.mavis.authority import MAVISAuthority


class G1CrossfitError(ValueError):
    """Raised when a G1 split or dependency violates dual exclusion."""


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _valid_domain_order(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) == 6
        and len(set(value)) == 6
        and all(type(domain) is str and domain for domain in value)
    )


@dataclass(frozen=True, slots=True)
class CrossfitRoster:
    domain_order: tuple[str, ...]
    outer_target: str
    labeled_domain: str
    fit_domains: tuple[str, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class CrossfitPriorFit:
    roster: CrossfitRoster
    prior: SourceBackgroundPrior
    state_sha256: str


@dataclass(frozen=True, slots=True)
class CrossfitCAIAssessorFit:
    roster: CrossfitRoster
    assessor: StateCAIAssessor
    state_sha256: str


def build_crossfit_roster(
    domain_order: tuple[str, ...],
    *,
    outer_target: str,
    labeled_domain: str,
) -> CrossfitRoster:
    if (
        not _valid_domain_order(domain_order)
        or type(outer_target) is not str
        or type(labeled_domain) is not str
        or outer_target not in domain_order
        or labeled_domain not in domain_order
        or outer_target == labeled_domain
    ):
        raise G1CrossfitError("crossfit domain roles are invalid")
    fit_domains = tuple(
        domain
        for domain in domain_order
        if domain not in {outer_target, labeled_domain}
    )
    payload = {
        "schema": 1,
        "kind": "g1-crossfit-roster",
        "domain_order": domain_order,
        "outer_target": outer_target,
        "labeled_domain": labeled_domain,
        "fit_domains": fit_domains,
    }
    return CrossfitRoster(
        domain_order=domain_order,
        outer_target=outer_target,
        labeled_domain=labeled_domain,
        fit_domains=fit_domains,
        state_sha256=_json_sha(payload),
    )


def _border_median(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise G1CrossfitError("source scan is invalid")
    border = np.concatenate(
        (value[0], value[-1], value[1:-1, 0], value[1:-1, -1]),
        axis=0,
    )
    return np.median(border.astype(np.float64), axis=0)


def _authority_domain_order(authority: MAVISAuthority) -> tuple[str, ...]:
    if type(authority) is not MAVISAuthority:
        raise G1CrossfitError("issued source authority is required")
    order = tuple(dict.fromkeys(authority.dataset_ids))
    if not _valid_domain_order(order):
        raise G1CrossfitError("source authority must contain six domains")
    return order


def fit_crossfit_source_prior(
    authority: MAVISAuthority,
    *,
    outer_target: str,
    labeled_domain: str,
) -> CrossfitPriorFit:
    domain_order = _authority_domain_order(authority)
    roster = build_crossfit_roster(
        domain_order,
        outer_target=outer_target,
        labeled_domain=labeled_domain,
    )
    fit_ids = tuple(
        specimen
        for specimen, domain in zip(
            authority.specimen_ids, authority.dataset_ids, strict=True
        )
        if domain in roster.fit_domains
    )
    if not fit_ids:
        raise G1CrossfitError("crossfit source prior has no fit specimens")
    domain_rows: list[np.ndarray] = []
    identity_rows: list[object] = []
    for domain in roster.fit_domains:
        specimen_rows = [
            specimen
            for specimen, specimen_domain in zip(
                authority.specimen_ids, authority.dataset_ids, strict=True
            )
            if specimen_domain == domain
        ]
        if not specimen_rows:
            raise G1CrossfitError("crossfit source-prior domain is empty")
        medians = []
        for specimen in specimen_rows:
            view = authority.source_teacher_view(specimen)
            medians.append(_border_median(view.full_scan))
            decoded_sha = hashlib.sha256()
            decoded_sha.update(view.full_scan.dtype.str.encode("ascii"))
            decoded_sha.update(
                json.dumps(view.full_scan.shape, separators=(",", ":")).encode("ascii")
            )
            decoded_sha.update(view.full_scan.tobytes(order="C"))
            identity_rows.append(
                (
                    domain,
                    specimen,
                    view.source_image_sha256,
                    decoded_sha.hexdigest(),
                )
            )
        domain_rows.append(np.mean(np.asarray(medians), axis=0, dtype=np.float64))
    domain_medians = np.asarray(domain_rows, dtype=np.float64)
    background = np.rint(
        np.mean(domain_medians, axis=0, dtype=np.float64)
    ).clip(0, 255).astype(np.uint8)
    fit_authority_sha = _json_sha(
        {
            "schema": 1,
            "kind": "g1-crossfit-prior-source-authority",
            "fit_domains": roster.fit_domains,
            "specimens": identity_rows,
        }
    )
    prior = SourceBackgroundPrior(
        outer_domain=outer_target,
        source_domains=roster.fit_domains,
        fit_specimen_ids=fit_ids,
        source_authority_sha256=fit_authority_sha,
        domain_border_medians=domain_medians,
        background_rgb=background,
    )
    return CrossfitPriorFit(
        roster=roster,
        prior=prior,
        state_sha256=_json_sha(
            {
                "schema": 1,
                "kind": "g1-crossfit-source-prior-fit",
                "roster": roster.state_sha256,
                "prior": prior.state_sha256,
            }
        ),
    )


def fit_crossfit_cai_assessor(
    rows: tuple[StateFeatureRow, ...],
    *,
    domain_order: tuple[str, ...],
    outer_target: str,
    labeled_domain: str,
    pca_dimension: int,
    ridge_alpha: float,
) -> CrossfitCAIAssessorFit:
    roster = build_crossfit_roster(
        domain_order,
        outer_target=outer_target,
        labeled_domain=labeled_domain,
    )
    if (
        type(rows) is not tuple
        or not rows
        or any(type(row) is not StateFeatureRow for row in rows)
        or tuple(dict.fromkeys(row.dataset_id for row in rows)) != roster.fit_domains
        or {row.dataset_id for row in rows} != set(roster.fit_domains)
    ):
        raise G1CrossfitError("CAI rows do not match the exact fit-domain roster")
    assessor = fit_state_cai_assessor(
        rows,
        outer_domain=outer_target,
        pca_dimension=pca_dimension,
        ridge_alpha=ridge_alpha,
    )
    if assessor.fit_domains != roster.fit_domains:
        raise G1CrossfitError("CAI assessor fit-domain order changed")
    return CrossfitCAIAssessorFit(
        roster=roster,
        assessor=assessor,
        state_sha256=_json_sha(
            {
                "schema": 1,
                "kind": "g1-crossfit-cai-assessor-fit",
                "roster": roster.state_sha256,
                "assessor": assessor.model_state_sha256,
            }
        ),
    )


__all__ = [
    "CrossfitCAIAssessorFit",
    "CrossfitPriorFit",
    "CrossfitRoster",
    "G1CrossfitError",
    "build_crossfit_roster",
    "fit_crossfit_cai_assessor",
    "fit_crossfit_source_prior",
]
