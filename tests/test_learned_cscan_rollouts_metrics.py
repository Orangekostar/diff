from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cmc_bbdm.learned_cscan.contracts import (
    ReferenceEvidence,
    ReferenceStatus,
    ReviewState,
    Task,
)
from cmc_bbdm.learned_cscan.metrics import (
    MetricRecord,
    StepSnapshot,
    exact_step_integral,
    paired_physical_specimen_bootstrap,
    scope_effect,
)
from cmc_bbdm.learned_cscan.rollouts import (
    cost_to_go_targets,
    run_candidate_branches,
    run_visible_episode,
)


def test_exact_step_integral_handles_endpoint_duplicates_and_regression() -> None:
    snapshots = (
        StepSnapshot(0.0, False, 1.0, "r0"),
        StepSnapshot(0.2, True, 0.2, "r1"),
        StepSnapshot(0.2, False, 0.8, "r2"),
        StepSnapshot(0.6, True, 0.1, "r3"),
        StepSnapshot(0.8, False, 0.9, "r4"),
    )

    success_area = exact_step_integral(
        snapshots, field="success", start_cost=0.0, end_cost=1.0
    )
    failure_area = exact_step_integral(
        snapshots, field="failure", start_cost=0.0, end_cost=1.0
    )
    loss_area = exact_step_integral(
        snapshots, field="task_loss", start_cost=0.0, end_cost=1.0
    )

    assert np.isclose(success_area, 0.2)
    assert np.isclose(failure_area, 0.8)
    assert np.isclose(success_area + failure_area, 1.0)
    assert np.isclose(loss_area, 0.72)


class _BranchWorld:
    def __init__(self, identity: int) -> None:
        self.identity = identity
        self.revealed: set[int] = set()

    def replay(self, history: tuple[int, ...]) -> frozenset[int]:
        self.revealed = set(history)
        return frozenset(self.revealed)

    def step(self, action: int) -> frozenset[int]:
        self.revealed.add(action)
        return frozenset(self.revealed)


def test_cost_to_go_uses_future_steps_only_queried_actions_and_isolated_branches() -> None:
    created: list[_BranchWorld] = []

    def factory() -> _BranchWorld:
        world = _BranchWorld(len(created))
        created.append(world)
        return world

    branch_results = run_candidate_branches(
        world_factory=factory,
        base_history=(1,),
        candidates=(3, 7),
        continue_branch=lambda world, _observation: world.step(9),
    )
    targets = cost_to_go_targets(
        current_cost=0.1,
        candidate_snapshots={
            3: (
                StepSnapshot(0.1, False, 1.0, "a0"),
                StepSnapshot(0.5, True, 0.0, "a1"),
                StepSnapshot(0.9, False, 1.0, "a2"),
            ),
            7: (
                StepSnapshot(0.1, False, 1.0, "b0"),
                StepSnapshot(0.2, True, 0.0, "b1"),
            ),
        },
        temperature=0.10,
        auxiliary_weight=0.05,
    )

    assert created[0] is not created[1]
    assert branch_results[3] == frozenset({1, 3, 9})
    assert branch_results[7] == frozenset({1, 7, 9})
    assert created[0].revealed == {1, 3, 9}
    assert created[1].revealed == {1, 7, 9}
    assert targets.queried_actions == (3, 7)
    assert len(targets.probabilities) == 2
    assert np.isclose(targets.probabilities.sum(), 1.0)
    assert np.allclose(targets.main_costs, (0.5, 0.1))
    assert targets.probabilities[1] > targets.probabilities[0]


@dataclass(frozen=True)
class _EpisodeWorld:
    limit: int
    step_calls: int = 0

    def reset(self) -> int:
        object.__setattr__(self, "step_calls", 0)
        return 0

    def step(self, _action: int) -> int:
        object.__setattr__(self, "step_calls", self.step_calls + 1)
        return self.step_calls


def test_true_stop_breaks_without_post_stop_reveal_and_freezes_report() -> None:
    def snapshot(state: int) -> StepSnapshot:
        return StepSnapshot(
            cost=state / 4.0,
            success=state == 2,
            task_loss=abs(2 - state) / 2.0,
            report_digest=f"report-{state}",
        )

    full_world = _EpisodeWorld(limit=4)
    stopped_world = _EpisodeWorld(limit=4)
    full = run_visible_episode(
        full_world,
        select_action=lambda state: state,
        make_snapshot=snapshot,
        should_stop=lambda _snapshot: False,
        max_actions=4,
    )
    stopped = run_visible_episode(
        stopped_world,
        select_action=lambda state: state,
        make_snapshot=snapshot,
        should_stop=lambda current: current.cost >= 0.5,
        max_actions=4,
    )

    assert stopped.stopped is True
    assert stopped_world.step_calls == 2
    assert stopped.terminal_snapshot.report_digest == "report-2"
    assert tuple(full.snapshots[:3]) == stopped.snapshots
    assert full_world.step_calls == 4
    assert full.terminal_snapshot.report_digest == "report-4"


def test_tasks_and_seeds_do_not_expand_physical_n_and_proxy_effect_stays_nonformal() -> None:
    rows = []
    for domain_index, domain in enumerate(("d0", "d1")):
        specimen = f"{domain}:sample-{domain_index}"
        for task in Task:
            for seed in (1, 2, 3):
                rows.extend(
                    (
                        MetricRecord(specimen, domain, "RULE", task, seed, 0.4),
                        MetricRecord(specimen, domain, "LEARNED", task, seed, 0.6),
                    )
                )
    result = paired_physical_specimen_bootstrap(
        tuple(rows),
        treatment="LEARNED",
        comparator="RULE",
        replicates=100,
        seed=5,
    )
    scoped = scope_effect(
        result,
        references=(
            ReferenceEvidence(
                status=ReferenceStatus.ALGORITHM_DERIVED_NOT_REVIEWED,
                review_state=ReviewState.PENDING,
                reviewer_alias=None,
            ),
        ),
    )

    assert result.physical_specimen_count == 2
    assert np.isclose(result.estimate, 0.2)
    assert np.isclose(result.ci_lower, 0.2)
    assert np.isclose(result.ci_upper, 0.2)
    assert scoped.proxy_effect == result
    assert scoped.formal_effect is None
