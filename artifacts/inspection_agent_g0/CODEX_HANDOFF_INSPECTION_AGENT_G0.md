# Codex Handoff: Inspection Agent G0

## Repository Identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Branch: `research/agentic-nde-inspection-reasoning-g0`
- Worktree: `/home/ww/diff/.worktrees/agentic-nde-inspection-reasoning-g0`
- Exact base: `892d92ea4979d9ca8ceeafef3348cd43266ed1b8`
- Hasebe source project: `/home/ww/paper3/cmc_damage_inference`
- Hasebe dataset: `/home/ww/paper3/cmc_damage_inference/data/public/hasebe`
- Frozen prompt SHA-256:
  `5e5ad7bdf871f445b6cb60476540e01b267b42d4b3194c4463ca4a273e83f8bf`
- Frozen config SHA-256:
  `cb056637f88294876c27a4ae4094b4802a125f43b2d625dae93b28accb084cbc`
- Verified evidence commit, local/upstream/remote:
  `9581be641d50913c0415ab93820fdf87b8e2dc5a`

The enclosing handoff commit cannot contain its own content-dependent SHA. The
final branch tip, upstream, remote, and clean status are therefore verified with
the commands under `GitHub Verification` and reported outside this file.

## Registered Decision

Final status: `G0_ACTIVE_INSPECTION_OPPORTUNITY_GO`.

G0 establishes privileged opportunity for zero-ultrasound active inspection. It
does not contain a learned planner and does not establish deployment readiness.
No VLM, LLM, policy Transformer, behavior-cloning model, PPO, GRPO, or AAWR
policy was trained.

Component gates:

| Component | Result |
|---|---|
| Zero-state compatibility and causal reveal | PASS |
| Metadata-free CAI assessor | PASS |
| Initialization headroom | FAIL |
| FIELD hierarchical headroom | PASS |
| CAI hierarchical headroom | PASS |
| Task conditioning | PASS |
| FIELD stopping | PASS |
| CAI stopping | PASS |

The final status remains ACTIVE rather than the strongest TASK_CONDITIONED
status because the preregistered final logic also requires initialization
headroom.

## Historical P0R and P1

- P0R remains `P0R_AUTHOR_REGISTRATION_GO`.
- All 276/276 specimens retain the author-attested global `ROT90` registration.
- P1 remains `P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO`.
- All six P1 outer folds selected surface-fusion lambda zero.
- Frozen old CAI AUEBC: `0.09202411704871961`.
- Frozen proposed CAI AUEBC: `0.09202411587521975`.
- Mechanical-oracle gap closure was approximately `9.8e-08`.

P1 rejects registered surface RGB as stable incremental information for frozen
post-scout mechanical-value ranking. It does not reject surface as a provisional
inspection hypothesis followed by causally acquired ultrasound evidence. G0
preserves P1 and uses the surface only for transparent saliency, FOCUS/BROADEN
labels, and misleading-surface analysis.

## Literature and Novelty Boundary

`LITERATURE_AND_NOVELTY_LEDGER.md` records primary-source precedents including
Fuentes et al. autonomous ultrasonic BO, adaptive inspection planning, POMDP
robotics, task-driven experimental design, active perception, and LLM/VLM tool
planning. G0 makes no "first" claim. A Fuentes-like GP/BO comparator is required
for G2 and was intentionally not allowed to block G0.

## Code Reuse and Ownership

Frozen code was reused without modification:

- `MAVISAuthority` supplies the private offline hidden world;
- MVA acquisition grids and `uniform_cell_order()` supply exact geometry and
  geometry-spread order;
- MVA interpolation semantics, FIELD loss, registered ResNet18 encoder, and
  CAI regression patterns are reused;
- P0R surface-cell registration remains the spatial authority;
- frozen MAVIS actions are replayed only by the historical B4 comparator.

The new `cmc_bbdm.inspection_agent` namespace owns generalized state, causal
world, source-safe reconstruction, surface hypothesis, metadata-free assessor,
privileged teachers, evaluation, stopping, statistics, artifact binding, and G0
execution. Existing MVA/MVD/MAVIS files were not edited.

## Input and Privilege Contract

Primary policy-visible inputs are surface RGB/hash, task, scanner lattice,
levels, acquired positions and values, exact budget, remaining budget, surface
hypothesis, and structured belief. They exclude true CAI, full/future C-scan,
domain/specimen identity, impact history, laminate, ply count, `metadata13`, and
`profile_stats21`.

`FIXED_UNIFORM_THEN_MAVIS` is labeled
`METADATA_AUGMENTED_UPPER_BOUND`. It is not gate eligible, and the label denotes
input privilege rather than performance superiority. Full scans and true CAI
are teacher/evaluation privilege only.

## Zero-State Audit

- 276 specimens across six Hasebe domains passed.
- Every initial state has 0 acquired pixels, 0 exact cost, 0 budget, and all 64
  levels equal to -1.
- All `-1->0` unions exactly match the old MVA scout masks and budgets.
- All level-1 states match the old all-level-1 masks.
- All level-2 states equal the full native raster.
- Prior values remain byte-identical across transitions.
- Sentinel tests reject hidden/future measurement and CAI leakage.
- Formal/replay zero-state tables are byte-identical.

Zero-state table SHA-256:
`2445b4015ce43438f01ba3290cead4c2ad239078baba48733467658f75207aa2`.

## Metadata-Free CAI Assessor

Each outer fold uses source-only PCA-32 and Ridge alpha 10 on the frozen 512-D
reconstruction embedding plus three observable acquisition scalars. Every
source specimen contributes one zero anchor and 18 label-independent states.
No CAI-oracle state enters fitting.

| Metric | Equal-domain result |
|---|---:|
| Zero-state CAI MAE | 0.182643120536 |
| Fixed 25% CAI MAE | 0.100215374569 |
| Improvement | 0.0824277459671 |
| 95% CI | [0.0696398759474, 0.0951896340019] |
| Improving domains | 6/6 |

All folds record outer-target exclusion and deterministic prediction replay.
CAI planning is authorized.

## G0-A Initialization

ORACLE_DISCOVERY minus ZERO_UNIFORM capture AUC is
`0.000740212343696`, 95% CI
`[0.000622894718166, 0.000867124273196]`, with 6/6 domains improving. The
relative improvement is `8.163795%` and capture-budget reduction is zero. Both
miss the alternative 10% magnitude gates, so
`INITIALIZATION_HEADROOM_NO_GO` is retained.

Surface/internal strata are 1 AGREE, 114 PARTIAL, and 161 MISLEADING. G1 should
use a simple geometry-spread initialization rather than claim sophisticated
zero-start prioritization.

## G0-B Hierarchical Allocation

| Task | Fixed AUEBC | Oracle AUEBC | Effect | Relative | 95% CI | Domains |
|---|---:|---:|---:|---:|---|---:|
| FIELD | 0.001162681470 | 0.001072049934 | 0.0000906315358 | 7.795044% | [0.0000826309228, 0.0000994948969] | 6/6 |
| CAI | 0.026682472404 | 0.005833683797 | 0.020848788607 | 78.136645% | [0.019150257953, 0.022589405744] | 6/6 |

FIELD passes through its alternative 48.24% reference-quality budget reduction;
CAI passes directly through relative AUEBC improvement. FIELD largely surveys
before refinement, while CAI repeatedly alternates BROADEN and REFINE. These are
privileged trajectory teachers, not a learned policy.

Historical `FIXED_UNIFORM_THEN_MAVIS` AUEBC is 0.001226035237 for FIELD and
0.027918914020 for CAI. It is not gate eligible.

## G0-C Task Conditioning

Wrong-task minus correct-task AUEBC:

| Objective | Advantage | 95% CI | Domains |
|---|---:|---|---:|
| FIELD | 0.009724283073 | [0.009285352589, 0.010169558857] | 6/6 |
| CAI | 0.020280014444 | [0.018388785185, 0.022221862075] | 6/6 |

Mean action Jaccard is 0.3444, cell Jaccard 0.5027, high-level overlap 0.2248,
and normalized edit distance 0.9405. The component gate passes because swapped
performance is worse on both objectives, not merely because paths differ.

## G0-D Stopping

| Task | Measurement saving | 95% CI | Loss ratio | Reached | Domains |
|---|---:|---|---:|---:|---:|
| FIELD | 48.241088% | [47.021426%, 49.480970%] | 1.022851 | 276/276 | 6/6 |
| CAI | 97.476454% | [96.166695%, 98.574421%] | 0.679090 | 276/276 | 6/6 |

Seventy-three CAI specimens meet the fixed-reference threshold at zero budget.
Stopping uses privileged task loss and is an opportunity upper bound. G1 must
learn STOP from observable state. Savings are normalized measurements, never
scanner time.

## Decision and Roadmap

- G1: learn structured FOCUS/BROADEN/REFINE/STOP decisions with geometry-spread
  initialization and no privileged policy input.
- G2: evaluate the closed loop against fixed, Fuentes-like GP/BO, and clearly
  privileged historical MAVIS references.
- G3: evaluate learned task conditioning because G0-C passed.
- G4: add a VLM/tool interface only after a deterministic planner is validated.

## Formal Execution and Replay

Formal command:

```bash
python scripts/run_inspection_agent.py run \
  --config paper_v3/configs/inspection_agent_g0.yaml \
  --source-project-root /home/ww/paper3/cmc_damage_inference \
  --output results/inspection_agent/g0 \
  --project-root /home/ww/diff/.worktrees/agentic-nde-inspection-reasoning-g0
```

Independent replay used the same command with output
`results/inspection_agent/replay/g0`.

The first formal attempt completed target computation but failed before package
publication because Polars inferred a nullable trajectory hash column from only
the first 100 rows. It left no formal output. Commit `40d08dd` added full-schema
inference and a 101-null-then-string regression test. It changed no task, gate,
model, action, metric, or target result. Formal and replay were then recomputed
from scratch.

Validation and comparison:

```bash
python scripts/run_inspection_agent.py validate \
  --config paper_v3/configs/inspection_agent_g0.yaml \
  --path results/inspection_agent/g0 \
  --project-root /home/ww/diff/.worktrees/agentic-nde-inspection-reasoning-g0
python scripts/run_inspection_agent.py validate \
  --config paper_v3/configs/inspection_agent_g0.yaml \
  --path results/inspection_agent/replay/g0 \
  --project-root /home/ww/diff/.worktrees/agentic-nde-inspection-reasoning-g0
python scripts/run_inspection_agent.py compare \
  --config paper_v3/configs/inspection_agent_g0.yaml \
  --formal results/inspection_agent/g0 \
  --replay results/inspection_agent/replay/g0 \
  --project-root /home/ww/diff/.worktrees/agentic-nde-inspection-reasoning-g0
```

Actual result:

- status: `G0_ACTIVE_INSPECTION_OPPORTUNITY_GO`;
- byte-identical: true;
- manifest SHA-256:
  `a85a62f14bd05d69c684deab1673e01a1a84d7ebf9c3e7805760c2898eacc179`;
- output-tree SHA-256:
  `e1441d847eaf187eb98de7eb84e93b708225924b5263d99560354120a7f30b0a`;
- formal/replay package SHA-256:
  `429f829b60bc9f520a41814ae2b6d34d05ef07cdfa188f39e2d9dbac93c45eca`.

The 170,851,318-byte trajectory Parquet is stored through Git LFS. Formal and
replay paths share one LFS object with SHA-256
`6b286eab5080a66cec804677944de14417f6f5c1d7b645df25348abe40bee49f`.

## Tests and Static Checks

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  $(rg --files tests | rg '/test_inspection_agent_.*\.py$' | sort)
```

Result: `67 passed in 61.88s`.

Historical regression:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  $(rg --files tests | rg '/test_agentic_nde.*\.py$' | sort)
```

Result: `196 passed in 50.89s`.

```bash
python -m ruff check src/cmc_bbdm/inspection_agent \
  scripts/run_inspection_agent.py \
  $(rg --files tests | rg '/test_inspection_agent_.*\.py$' | sort)
python -m ruff check src/cmc_bbdm/agentic_nde scripts/run_agentic_nde.py \
  $(rg --files tests | rg '/test_agentic_nde.*\.py$' | sort)
git diff --check 892d92ea4979d9ca8ceeafef3348cd43266ed1b8
git lfs fsck
```

Result: all passed.

## Frozen-Path Audit

The exact frozen-path diff from base is empty for `results/mva`, `results/mvd`,
`results/mavis`, `results/mavis_science_closure`, `src/cmc_bbdm/{mva,mvd,mavis}`,
historical P0/P0R/P1 formal and replay packages, and the four frozen P0/P0R/P1
decision artifacts. `tests/test_inspection_agent_frozen_paths.py` also passed.

No result was rescued after formal execution. No domain was dropped, no target
selected a threshold, no task or gate changed, and no new planner was trained.

## Commit History

```text
615a188ce945930d385995d8a1886da21eebe578 audit: map inspection-agent reuse and privilege boundaries
09da78c3a978245be2deb0e8c8338615b6decc08 feat: add zero-ultrasound inspection state
f9230881172ac2b763346e3507d9d5f47b11f080 feat: add causal zero-start inspection world
d9cd34adf2e29c18b00a61d404d910ee98efc87a feat: add generalized field reconstruction
a1d4f47eae958fb05ccd30a63c8982945ffabb12 audit: register source zero-state assessor anchor
ff8fcf5c70a4eb8de0005c00b402bb3585e66d68 feat: add metadata-free CAI state assessor
e05c15d75d2c55861f15a0ae53937207946bca1d exp: add privileged inspection opportunity teachers
3c95825502084913ebb00a8b236dbef3ec539d7b exp: add G0 inference stopping and decision gates
be7aef91cb22148ff1a22c74625e332e5e6d4a27 perf: preserve exact G0 candidate semantics at scale
50998dbbdc5da5e091240b5fa86194ff2a2ed40c feat: bind deterministic G0 artifacts and replay
7e5af7d55ca8f5b6e3c84ff4bbe2e84244071062 perf: streamline causal trajectory transitions
f272dfddb3676365b306c52556303f7ba99b3640 exp: add formal G0 execution and CLI
40d08dd84a7fa015c677703294b31e8cd83719cc fix: infer complete G0 trajectory schema
9581be641d50913c0415ab93820fdf87b8e2dc5a exp: register G0 active inspection opportunity
```

## Complete Changed-File Inventory

Governance and docs:

```text
.gitattributes
artifacts/inspection_agent_g0/CODEX_HANDOFF_INSPECTION_AGENT_G0.md
artifacts/inspection_agent_g0/G0_CAI_ASSESSOR_AUDIT.md
artifacts/inspection_agent_g0/G0_DECISION.md
artifacts/inspection_agent_g0/G0_HIERARCHICAL_OPPORTUNITY.md
artifacts/inspection_agent_g0/G0_INITIALIZATION_OPPORTUNITY.md
artifacts/inspection_agent_g0/G0_INPUT_PRIVILEGE_CONTRACT.md
artifacts/inspection_agent_g0/G0_PREFLIGHT_20_QUESTION_AUDIT.md
artifacts/inspection_agent_g0/G0_REPOSITORY_CODE_MAP.md
artifacts/inspection_agent_g0/G0_STOPPING_OPPORTUNITY.md
artifacts/inspection_agent_g0/G0_TASK_CONDITIONING_AUDIT.md
artifacts/inspection_agent_g0/G0_ZERO_STATE_AUDIT.md
artifacts/inspection_agent_g0/LITERATURE_AND_NOVELTY_LEDGER.md
docs/INSPECTION_AGENT_G0_PROTOCOL.md
docs/superpowers/plans/2026-08-31-inspection-agent-g0.md
docs/superpowers/specs/2026-08-31-inspection-agent-g0-design.md
paper_v3/configs/inspection_agent_g0.yaml
scripts/run_inspection_agent.py
```

Runtime:

```text
src/cmc_bbdm/inspection_agent/__init__.py
src/cmc_bbdm/inspection_agent/artifacts.py
src/cmc_bbdm/inspection_agent/cai_assessor.py
src/cmc_bbdm/inspection_agent/contracts.py
src/cmc_bbdm/inspection_agent/evaluation.py
src/cmc_bbdm/inspection_agent/field_task.py
src/cmc_bbdm/inspection_agent/g0.py
src/cmc_bbdm/inspection_agent/generalized_reconstruction.py
src/cmc_bbdm/inspection_agent/oracle.py
src/cmc_bbdm/inspection_agent/state.py
src/cmc_bbdm/inspection_agent/state_bank.py
src/cmc_bbdm/inspection_agent/statistics.py
src/cmc_bbdm/inspection_agent/stopping.py
src/cmc_bbdm/inspection_agent/surface_hypothesis.py
src/cmc_bbdm/inspection_agent/world.py
```

Tests:

```text
tests/test_inspection_agent_artifacts.py
tests/test_inspection_agent_cai_assessor.py
tests/test_inspection_agent_cli.py
tests/test_inspection_agent_exact_cost.py
tests/test_inspection_agent_frozen_paths.py
tests/test_inspection_agent_g0_execution.py
tests/test_inspection_agent_gates.py
tests/test_inspection_agent_generalized_reconstruction.py
tests/test_inspection_agent_mva_equivalence.py
tests/test_inspection_agent_oracle.py
tests/test_inspection_agent_replay.py
tests/test_inspection_agent_source_prior.py
tests/test_inspection_agent_state_bank.py
tests/test_inspection_agent_state_transitions.py
tests/test_inspection_agent_statistics.py
tests/test_inspection_agent_stopping.py
tests/test_inspection_agent_surface_hypothesis.py
tests/test_inspection_agent_task_swap.py
tests/test_inspection_agent_world_no_leakage.py
tests/test_inspection_agent_zero_state.py
```

Formal and replay each contain the following exact roster under respectively
`results/inspection_agent/g0/` and `results/inspection_agent/replay/g0/`:

```text
CHECKSUMS.sha256
REPORT.md
artifact_manifest.json
authorized_roster.csv
bootstrap.csv
cai_assessor_metrics.csv
config.yaml
decision_summary.json
domain_metrics.csv
hierarchical_trajectories.parquet
initialization_curves.csv
state_bank_manifest.csv
stopping_results.csv
task_swap.csv
zero_state_audit.csv
```

Machine-verifiable inventory:

```bash
git diff --name-only 892d92ea4979d9ca8ceeafef3348cd43266ed1b8..HEAD | sort
```

## GitHub Verification

Evidence push:

```bash
git push -u origin research/agentic-nde-inspection-reasoning-g0
git lfs push --all origin research/agentic-nde-inspection-reasoning-g0
```

At evidence commit `9581be6`, local, upstream, and remote were identical and the
single 171 MB LFS object was uploaded. Final acceptance uses:

```bash
git rev-parse HEAD
git rev-parse '@{upstream}'
git ls-remote origin refs/heads/research/agentic-nde-inspection-reasoning-g0
git status --short
```

Required result: identical local/upstream/remote SHAs and an empty worktree. No
force push, PR creation, merge, or history rewrite is part of this handoff.
