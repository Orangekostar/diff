from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1 import stop_execution as stop_module
from cmc_bbdm.inspection_agent_g1.crossfit import build_crossfit_roster
from cmc_bbdm.inspection_agent_g1.formal import FIXED_BASELINE_METHODS
from cmc_bbdm.inspection_agent_g1.stop_execution import (
    G1FixedEndpointBankFile,
    G1FixedEndpointBuild,
    G1FixedEndpointRecord,
    build_g1_all_source_fixed_endpoint_banks,
    build_g1_all_source_stop_banks,
    read_fixed_endpoint_bank,
    select_g1_source_fixed_reference,
    stop_bank_path,
    write_fixed_endpoint_bank,
)
from cmc_bbdm.inspection_agent_g1.teacher import authorize_source_teacher

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _record(source: str, method: str, *, loss: float) -> G1FixedEndpointRecord:
    return G1FixedEndpointRecord(
        outer_target="d6",
        source_domain=source,
        specimen_id=f"{source}-sample",
        specimen_sha256=_sha(f"specimen-{source}"),
        task=InspectionTask.FIELD,
        method=method,
        fit_domains=tuple(
            domain for domain in DOMAINS if domain not in {"d6", source}
        ),
        dependency_sha256=_sha(f"dependency-{source}"),
        action_history_sha256=_sha(f"actions-{source}-{method}"),
        endpoint_observation_sha256=_sha(f"observation-{source}-{method}"),
        effective_budget=0.249,
        exact_acquired_count=249,
        native_count=1000,
        task_loss=loss,
    )


def _records() -> tuple[G1FixedEndpointRecord, ...]:
    return tuple(
        _record(
            source,
            method,
            loss=(
                0.01
                if source == "d1" and method == "RANDOM"
                else 0.5
                if method == "CENTER_FIRST"
                else 1.0
            ),
        )
        for source in DOMAINS[:-1]
        for method in FIXED_BASELINE_METHODS
    )


def test_source_fixed_reference_excludes_outer_and_labeled_source_endpoints() -> None:
    authorization = authorize_source_teacher(
        build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1"),
        query_domain="d1",
    )

    selected = select_g1_source_fixed_reference(
        authorization,
        _records(),
        task=InspectionTask.FIELD,
    )

    assert selected.method == "CENTER_FIRST"
    assert selected.fit_domains == ("d2", "d3", "d4", "d5")


def test_fixed_endpoint_bank_round_trip_is_exact(tmp_path: Path) -> None:
    records = tuple(row for row in _records() if row.source_domain == "d2")
    path = tmp_path / "fixed.parquet"

    identity = write_fixed_endpoint_bank(path, records)
    loaded_identity, loaded = read_fixed_endpoint_bank(path)

    assert loaded_identity == identity
    assert loaded == records
    assert identity.row_count == len(FIXED_BASELINE_METHODS)


def test_fixed_endpoint_batch_resume_follows_directed_fold_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    protocol = SimpleNamespace(domain_order=DOMAINS)

    def fake_dependencies(
        _runtime: object,
        _protocol: object,
        *,
        outer_target: str,
        labeled_domain: str,
        **_kwargs: object,
    ) -> object:
        calls.append((outer_target, labeled_domain))
        return SimpleNamespace(
            roster=SimpleNamespace(
                outer_target=outer_target,
                labeled_domain=labeled_domain,
            ),
            state_sha256=_sha(f"dependency-{outer_target}-{labeled_domain}"),
        )

    def fake_bank(
        _runtime: object,
        _protocol: object,
        dependencies: object,
        **_kwargs: object,
    ) -> G1FixedEndpointBuild:
        outer = dependencies.roster.outer_target
        source = dependencies.roster.labeled_domain
        return G1FixedEndpointBuild(
            path=tmp_path / outer / f"{source}.parquet",
            outer_target=outer,
            source_domain=source,
            specimen_count=1,
            dependency_sha256=dependencies.state_sha256,
            bank=G1FixedEndpointBankFile(
                row_count=10,
                parquet_sha256=_sha(f"parquet-{outer}-{source}"),
                records_sha256=_sha(f"records-{outer}-{source}"),
                manifest_sha256=_sha(f"manifest-{outer}-{source}"),
            ),
        )

    monkeypatch.setattr(stop_module, "build_g1_source_dependencies", fake_dependencies)
    monkeypatch.setattr(stop_module, "build_g1_source_fixed_endpoint_bank", fake_bank)
    results = build_g1_all_source_fixed_endpoint_banks(
        object(),
        protocol,
        encoder=object(),
        work_root=tmp_path,
        start_fold=3,
    )
    pairs = tuple(
        (outer, source)
        for outer in DOMAINS
        for source in DOMAINS
        if source != outer
    )
    assert tuple(calls) == pairs[2:]
    assert tuple((row.outer_target, row.source_domain) for row in results) == pairs[2:]


def test_stop_bank_path_is_bound_to_outer_and_source() -> None:
    assert stop_bank_path("stop", "outer", "source").as_posix() == (
        "stop/outer/source.parquet"
    )


def test_stop_bank_batch_resume_uses_the_same_directed_fold_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    protocol = SimpleNamespace(domain_order=DOMAINS)

    def fake_dependencies(
        _runtime: object,
        _protocol: object,
        *,
        outer_target: str,
        labeled_domain: str,
        **_kwargs: object,
    ) -> object:
        calls.append((outer_target, labeled_domain))
        return SimpleNamespace(
            roster=SimpleNamespace(
                outer_target=outer_target,
                labeled_domain=labeled_domain,
            )
        )

    monkeypatch.setattr(stop_module, "build_g1_source_dependencies", fake_dependencies)
    def fake_stop_bank(
        _runtime: object,
        _protocol: object,
        dependencies: object,
        **_kwargs: object,
    ) -> object:
        return SimpleNamespace(
            outer_target=dependencies.roster.outer_target,
            source_domain=dependencies.roster.labeled_domain,
        )

    monkeypatch.setattr(stop_module, "build_g1_source_stop_bank", fake_stop_bank)
    results = build_g1_all_source_stop_banks(
        object(),
        protocol,
        encoder=object(),
        teacher_bank_root=tmp_path / "teacher",
        fixed_endpoint_root=tmp_path / "fixed",
        work_root=tmp_path / "stop",
        start_fold=4,
    )
    pairs = tuple(
        (outer, source)
        for outer in DOMAINS
        for source in DOMAINS
        if source != outer
    )
    assert tuple(calls) == pairs[3:]
    assert tuple((row.outer_target, row.source_domain) for row in results) == pairs[3:]
