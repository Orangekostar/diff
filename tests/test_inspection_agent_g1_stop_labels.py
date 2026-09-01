from __future__ import annotations

import hashlib
import inspect

import pytest
import torch

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.stopping import ReferenceEndpoint
from cmc_bbdm.inspection_agent_g1.crossfit import build_crossfit_roster
from cmc_bbdm.inspection_agent_g1.stopping_policy import (
    GATE_ELIGIBLE_FIXED_METHODS,
    G1StoppingError,
    build_source_stop_label,
    observable_stop_loss,
    select_source_fixed_reference,
)
from cmc_bbdm.inspection_agent_g1.teacher import authorize_source_teacher

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _authorization():
    return authorize_source_teacher(
        build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1"),
        query_domain="d1",
    )


def _reference_rows(*, contaminated_domain: str | None = None):
    rows = []
    for domain in ("d2", "d3", "d4", "d5"):
        actual = contaminated_domain if domain == "d5" and contaminated_domain else domain
        for method in GATE_ELIGIBLE_FIXED_METHODS:
            rows.append(
                ReferenceEndpoint(
                    method=method,
                    dataset_id=actual,
                    specimen_id=f"{actual}-sample",
                    task_loss=0.5 if method == "CENTER_FIRST" else 1.0,
                )
            )
    return tuple(rows)


def test_stop_reference_uses_only_the_other_four_source_domains() -> None:
    reference = select_source_fixed_reference(_authorization(), _reference_rows())
    assert reference.method == "CENTER_FIRST"
    assert reference.fit_domains == ("d2", "d3", "d4", "d5")
    assert reference.authorization_sha256 == _authorization().state_sha256


@pytest.mark.parametrize("domain", ["d1", "d6"])
def test_stop_reference_rejects_labeled_or_outer_domain_metrics(domain: str) -> None:
    with pytest.raises(G1StoppingError, match="four source domains"):
        select_source_fixed_reference(
            _authorization(),
            _reference_rows(contaminated_domain=domain),
        )


def test_source_stop_label_uses_privileged_loss_only_as_a_target() -> None:
    reference = select_source_fixed_reference(_authorization(), _reference_rows())
    sufficient = build_source_stop_label(
        _authorization(),
        reference,
        source_domain="d1",
        specimen_sha256=_sha("sufficient"),
        task=InspectionTask.CAI,
        policy_state_sha256=_sha("policy-state"),
        current_true_loss=1.04,
        reference_true_loss=1.0,
    )
    insufficient = build_source_stop_label(
        _authorization(),
        reference,
        source_domain="d1",
        specimen_sha256=_sha("insufficient"),
        task=InspectionTask.CAI,
        policy_state_sha256=_sha("other-policy-state"),
        current_true_loss=1.06,
        reference_true_loss=1.0,
    )
    assert sufficient.is_sufficient is True
    assert insufficient.is_sufficient is False

    parameters = set(inspect.signature(observable_stop_loss).parameters)
    assert parameters == {"stop_logits", "sufficient_labels", "sample_weights"}
    loss = observable_stop_loss(
        torch.tensor([0.0, 1.0]),
        torch.tensor([True, False]),
    )
    assert torch.isfinite(loss)
