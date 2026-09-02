from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.crossfit import build_crossfit_roster
from cmc_bbdm.inspection_agent_g1.formal import FIXED_BASELINE_METHODS
from cmc_bbdm.inspection_agent_g1.metrics import build_engineering_curve
from cmc_bbdm.inspection_agent_g1.source_bridge import (
    G1SourceBridgeError,
    G1SourceBridgeRecord,
    materialize_source_bridge_for_world,
    read_source_bridge_bank,
    select_source_fixed_bridge,
    write_source_bridge_bank,
)
from cmc_bbdm.inspection_agent_g1.teacher import authorize_source_teacher
from cmc_bbdm.inspection_agent_g1.warm_start import build_deployment_grid
from cmc_bbdm.mavis.authority import MAVISAuthority

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _record(
    source: str,
    method: str,
    loss: float,
    *,
    task: InspectionTask = InspectionTask.FIELD,
    specimen_token: str | None = None,
) -> G1SourceBridgeRecord:
    specimen = f"{source}-specimen"
    fit_domains = tuple(
        domain for domain in DOMAINS if domain not in {"d6", source}
    )
    state_hashes = tuple(_sha(f"{source}-{method}-{index}") for index in range(5))
    curve = build_engineering_curve(
        method=method,
        target_domain=source,
        specimen_sha256=_sha(specimen if specimen_token is None else specimen_token),
        task=task,
        grid_sha256=_sha(f"grid-{source}"),
        evaluator_sha256=_sha(f"evaluator-{source}"),
        warm_start_sha256=_sha(f"warm-{source}-{task.value}"),
        state_budgets=(0.0, 0.05, 0.1, 0.18, 0.24),
        state_losses=(loss,) * 5,
        state_sha256=state_hashes,
    )
    return G1SourceBridgeRecord(
        outer_target="d6",
        source_domain=source,
        specimen_id=specimen,
        fit_domains=fit_domains,
        dependency_sha256=_sha(f"dependencies-{source}"),
        action_history_sha256=_sha(f"actions-{source}-{method}-{task.value}"),
        curve=curve,
    )


def test_source_fixed_bridge_selection_excludes_validation_domain() -> None:
    records = []
    for source in DOMAINS[:-1]:
        for method_index, method in enumerate(FIXED_BASELINE_METHODS):
            loss = 1.0 + method_index
            if method == "SURFACE_FOCUS":
                loss = 0.5 if source != "d1" else 100.0
            records.append(_record(source, method, loss))

    selection = select_source_fixed_bridge(
        tuple(records),
        validation_domain="d1",
        task=InspectionTask.FIELD,
    )

    assert selection.method == "SURFACE_FOCUS"
    assert selection.fit_domains == ("d2", "d3", "d4", "d5")
    assert tuple(domain for domain, _value in selection.domain_auebc) == (
        "d2",
        "d3",
        "d4",
        "d5",
    )
    assert selection.equal_domain_auebc == 0.125


def test_source_bridge_bank_round_trips_curve_evidence(tmp_path: Path) -> None:
    records = tuple(
        _record("d1", method, 1.0 + index, task=task)
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
        for index, method in enumerate(
            (
                *FIXED_BASELINE_METHODS,
                f"ORACLE_{task.value}",
            )
        )
    )
    path = tmp_path / "bridge.parquet"

    identity = write_source_bridge_bank(path, records)
    replay_identity, replay = read_source_bridge_bank(path)

    assert replay_identity == identity
    ordered = tuple(
        sorted(
            records,
            key=lambda row: (
                row.outer_target,
                row.source_domain,
                row.specimen_id,
                row.curve.task.value,
                row.curve.method,
            ),
        )
    )
    assert tuple(row.state_sha256 for row in replay) == tuple(
        row.state_sha256 for row in ordered
    )
    for expected, actual in zip(ordered, replay, strict=True):
        assert actual.curve.state_sha256 == expected.curve.state_sha256
        assert np.array_equal(actual.curve.exact_budgets, expected.curve.exact_budgets)
        assert np.array_equal(actual.curve.task_losses, expected.curve.task_losses)


def test_source_bridge_bank_rejects_cross_method_specimen_mixing(
    tmp_path: Path,
) -> None:
    records = [
        _record("d1", method, 1.0 + index, task=task)
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
        for index, method in enumerate(
            (*FIXED_BASELINE_METHODS, f"ORACLE_{task.value}")
        )
    ]
    records[0] = _record(
        "d1",
        records[0].curve.method,
        1.0,
        task=records[0].curve.task,
        specimen_token="other-physical-specimen",
    )

    with pytest.raises(G1SourceBridgeError, match="specimen identity"):
        write_source_bridge_bank(tmp_path / "mixed.parquet", tuple(records))


class _Assessor:
    outer_domain = "d6"
    fit_domains = ("d2", "d3", "d4", "d5")
    model_state_sha256 = _sha("assessor")

    def predict(self, embeddings: object, scalars: object) -> np.ndarray:
        del scalars
        return np.full(len(np.asarray(embeddings)), 0.4, dtype=np.float64)


class _Encoder:
    def encode(self, images: object) -> np.ndarray:
        return np.zeros((len(tuple(images)), 512), dtype=np.float64)


def _world_fixture(task: InspectionTask):
    rows, columns = np.indices((41, 43))
    full_scan = np.stack((3 * rows, 4 * columns, rows + columns), axis=2).astype(
        np.uint8
    )
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("d1-sample",),
        dataset_ids=("d1",),
        images=(full_scan,),
        targets=np.asarray([0.4]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )
    grid = build_deployment_grid(full_scan.shape[:2])
    surface_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    world = CausalInspectionWorld(
        authority,
        specimen_id="d1-sample",
        task=task,
        surface_rgb=surface_rgb,
        surface_sha256=_sha("surface"),
        grid=grid,
        endpoint_budget=0.25,
    )
    hypothesis = SurfaceHypothesis(
        scores=np.linspace(0.0, 1.0, 64),
        top_cells=tuple(range(63, 55, -1)),
        border_median_rgb=np.zeros(3),
        state_sha256=_sha("hypothesis"),
    )
    prior = SourceBackgroundPrior(
        outer_domain="d6",
        source_domains=("d2", "d3", "d4", "d5"),
        fit_specimen_ids=("s2", "s3", "s4", "s5"),
        source_authority_sha256=_sha("source-authority"),
        domain_border_medians=np.zeros((4, 3)),
        background_rgb=np.zeros(3, dtype=np.uint8),
    )
    authorization = authorize_source_teacher(
        build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1"),
        query_domain="d1",
    )
    return world, grid, hypothesis, prior, authorization, full_scan


@pytest.mark.parametrize("task", (InspectionTask.FIELD, InspectionTask.CAI))
def test_source_world_materializes_fixed_and_task_oracle_bridge(
    task: InspectionTask,
) -> None:
    world, grid, hypothesis, prior, authorization, full_scan = _world_fixture(task)

    records = materialize_source_bridge_for_world(
        world,
        grid,
        hypothesis,
        prior,
        authorization,
        assessor=_Assessor(),
        encoder=_Encoder(),
        outer_target="d6",
        source_domain="d1",
        specimen_id="d1-sample",
        specimen_sha256=_sha("d1-sample"),
        dependency_sha256=_sha("dependencies"),
        full_scan=full_scan,
        true_cai=0.4,
        random_seed=2026090101,
    )

    assert {row.curve.method for row in records} == {
        *FIXED_BASELINE_METHODS,
        f"ORACLE_{task.value}",
    }
    assert all(row.curve.task is task for row in records)
    assert all(row.fit_domains == ("d2", "d3", "d4", "d5") for row in records)
    assert all(np.isfinite(row.curve.auebc) for row in records)
