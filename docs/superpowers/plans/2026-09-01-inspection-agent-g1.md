# G1 Observable Inspection Policy Implementation Plan

> Execute test-first in the isolated G1 worktree. The primary agent owns all
> scientific, architectural, debugging, review, and integration decisions.

## 1. Freeze Evidence and Protocol

Create the configuration, protocol/design, literature ledger, 30-question audit,
repository binding map, privilege/split contract, and deployment geometry decision.
Verify source hashes, formal/replay G0 identity, base/remote identity, all 276 roster
rows, three geometries, K8 order/cost/action counts, and frozen-path diff. Commit as:

```text
audit: freeze G1 evidence and deployment geometry
```

## 2. Contracts, Geometry, and Crossfit (TDD)

First add failing tests for base identity, frozen G0, geometry audit, warm start,
policy state privilege rejection, dual-exclusion prior, and dual-exclusion CAI
assessor. Then implement `contracts.py`, `warm_start.py`, and `crossfit.py`. Tests
must prove complete scouting removes BROADEN, K8 preserves 56 BROADEN, and every
outer/labeled pair uses only the other four domains. Commit the fold-safe crossfit
and observable state changes separately.

## 3. Observable Features and Teachers (TDD)

Add failing policy-state, no-privilege, candidate-completeness, no-target, and
teacher-distribution tests. Implement `features.py`, `teacher.py`, and
`teacher_bank.py`; build all 192 slots, exact masks/costs, separated namespaces,
13 label-independent plus four oracle-checkpoint states, robust utility targets, and
hash-bound work manifests. Run a one-specimen/two-domain smoke before any full bank.

## 4. Models and Supervised Training (TDD)

Add masking, parameter-cap, hard-BC, and soft-distillation tests. Implement
`SharedActionMLP`, `StructuredInspectionPolicy`, deterministic training/checkpoints,
and utility loss. Prove illegal probability is unavailable, actor inputs contain no
teacher fields, exact trainable count is below one million, and a synthetic batch
decreases each registered loss.

## 5. DAgger, Conditional AAWR, and STOP (TDD)

Add source-only DAgger, AAWR authorization, STOP label, and STOP threshold tests.
Implement source-world guards and deterministic quantile caps; run iterations 0/1/2
inside inner source validation. Evaluate the AAWR condition before any target access;
run it only when authorized. Train STOP separately and retain the fixed-endpoint
fallback.

## 6. Rollout, Metrics, Statistics, and Gates (TDD)

Add task token, surface control, target freeze, closed-loop, statistics, and final
gate tests. Implement target freeze state machine, all fixed/learned/control rollouts,
same-geometry bridge curves, gap closure, synchronized 100k bootstrap, component
gates, and exact final status vocabulary. A pre-seal target truth request must fail.

## 7. Formal Six-Fold Execution

Run in prompt order: dependency construction; teacher banks; inner source selection;
DAgger/conditional AAWR decision; final five-source refit/freeze; causal target
rollouts; trajectory seal; truth evaluation; bootstrap/gates. Do not alter the
registered roster, K8, objectives, model candidates, seeds, or gates from target
results. Preserve negative results.

## 8. Artifact Publication and Replay (TDD)

Add artifact/replay tests before implementing the G1 package validator. Publish the
exact required 20 result files plus manifest/checksums. Rebuild the independent
replay package and require byte identity, or the preregistered strict model
hash/tolerance contract if hardware prevents floating byte identity. Run Git LFS
checks for any tracked Parquet files.

## 9. Completion Review and Push

Create all required result audits and handoff, inventory every changed file, and
personally review the full diff. Run:

```bash
git diff --check
python -m ruff check src/cmc_bbdm/inspection_agent_g1 tests/test_inspection_agent_g1*.py scripts/run_inspection_agent_g1.py
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_inspection_agent*.py
git diff --name-only 7a10cd425de582fa158bf6639285731ccd8ff7a7 -- <all frozen paths>
git lfs fsck
```

Validate formal/replay packages through the new CLI, verify all explicit prompt
outputs/tests/gates requirement-by-requirement, commit documentation, push without
force, and require local/upstream/remote SHA equality plus a clean worktree.
