# BC C-scan Frozen Process Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver frozen W0-W5 trajectory/process evidence now and a single W6-W8 finalize path that imports later user-supplied reviewed references, blind reviews, and human sessions without retraining or new Actor/STOP/VLM inference.

**Architecture:** A strict config binds the immutable source artifacts. One analysis module performs table-only W0-W3/W5 computations, one recovery module replays only stored actions to recover Reader reports and W4 spatial summaries, and one thin finalize module scores optional external evidence and builds the claim matrix. A small CLI exposes `analyze` and `finalize`; all outputs go to the new result/artifact roots.

**Tech Stack:** Python 3.10+, Polars, NumPy, SciPy, Matplotlib, PyYAML, pytest, existing `cmc_bbdm.learned_cscan` runtime/statistics/artifact helpers.

**Spec:** `/home/ww/paper3/CODEX_BC_FINAL_EVIDENCE_CLOSURE_V3.md` (authoritative), with `/home/ww/paper3/BC_LAST_ROUND_SCOPE_AUDIT.md` as scope rationale.

## Global Constraints

- Exact base: `fe58de39298c412d0829580cfda40b7c7c4c53e9`; branch: `research/bc-cscan-frozen-process-analysis`.
- Training updates, new weights, VLM calls, Actor forward calls, STOP forward calls, threshold scans, and counterfactual action searches: all zero.
- Reader, Actor, STOP, 0.99 thresholds, task criteria, cached percepts, actions, geometry, split, and all historical result files remain frozen.
- W0-W5 use all 336 episodes for scalar analysis; W4 recovery is limited to P8 plus BC seeds 1/2/3, 192 episodes and at most 36,864 stored-action steps.
- Without reviewed references the total recovery cap is 40,000 steps; with real references, frozen ablation rescoring may raise the conditional cap to 60,000.
- CPU only, at most four processes; no GPU requirement; no more than three new PNG figure groups and six physical case studies.
- Statistics use physical specimens as the unit, average BC seeds within specimen, weight six domains equally, and reuse seed `2026090801` with 5,000 bootstrap draws.
- Proxy and reviewed reference versions never share a formal mean. Missing external evidence remains `PENDING_INPUT`; no synthetic human or expert data is created.
- No historical code/result/config file is reformatted or overwritten. New functionality writes only under `results/bc_cscan_frozen_process_analysis/` and `artifacts/bc_cscan_frozen_process_analysis/`.

---

### Task 1: Pure Frozen-Trajectory Semantics

**Files:**
- Create: `tests/test_bc_cscan_frozen_process_analysis.py`
- Create: `src/cmc_bbdm/learned_cscan/frozen_process_analysis.py`

**Interfaces:**
- Consumes: ordered trajectory rows with key `(specimen_key, task, method, seed, step)`.
- Produces: `stop_decomposition(rows, stop_system)`, `report_stability(rows, stop_step)`, `planning_stop_accounting(rows, stop_step)`, `derive_action_events(rows, stop_steps)`, `surface_cell_partition(cue_cells)`, and `overlap_metrics(target, cue)`.

- [ ] **Step 1: Write failing tests for the six W1-W4 semantic groups**

  Cover: state/action offset and zero-cost transitions; `0->1->0->1` report stability; all three signed accounting examples; duplicate-cost left-step integration; V/R/O partitions and empty masks; compound-key isolation and pending inputs. Tests assert that stopping at row `k` excludes action `a_k`, `G=A-U` and `G=P-F`, terminal actions are absent, and partition counts are exact.

- [ ] **Step 2: Run the new tests and verify RED**

  Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_frozen_process_analysis.py`

  Expected: collection fails because `frozen_process_analysis` does not exist.

- [ ] **Step 3: Implement the pure functions and typed constants**

  Use existing `StepSnapshot` and `exact_step_integral`. Validate ordered unique steps and nondecreasing costs. Treat `S_BC_CAL` as the first `calibrated_stop_trigger`, `S_RULE` as the first `rule_stop`, and never inspect future rows to change the stop. Return null delay values when their conditions are undefined.

- [ ] **Step 4: Run the six semantic groups and verify GREEN**

  Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_frozen_process_analysis.py -k 'not reviewed and not external'`

  Expected: all selected tests pass.

### Task 2: Config, W0 Validation, and W1-W3 Outputs

**Files:**
- Create: `paper_v3/configs/bc_cscan_frozen_process_analysis.yaml`
- Modify: `src/cmc_bbdm/learned_cscan/frozen_process_analysis.py`
- Modify: `tests/test_bc_cscan_frozen_process_analysis.py`

**Interfaces:**
- Consumes: exact hashes for `report_manifest.json`, `trajectories.parquet`, `per_episode_metrics.csv`, stop calibration, source summaries, percept cache, and parent configs.
- Produces: `FrozenProcessConfig`, `load_frozen_process_config()`, `validate_frozen_inputs()`, and `analyze_tables()` returning W0-W3 rows plus diagnostic effects.

- [ ] **Step 1: Add a failing config/input identity test**

  Assert exact base, zero inference/training budgets, separate source/output roots, seven frozen methods, two tasks, threshold 0.99, and rejection of a changed source SHA.

- [ ] **Step 2: Run the config test and verify RED**

  Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_frozen_process_analysis.py -k config`

- [ ] **Step 3: Implement strict config loading and matrix validation**

  Validate 24 specimens, six domains, seven methods, two tasks, 336 episode keys, 193 ordered states per episode, 64,848 total rows, terminal `action_cell=-1`, frozen model identity, and selected thresholds. Hash only required files and never call old write-producing entry points.

- [ ] **Step 4: Implement W1-W3 table assembly**

  Generate one W1 row per episode/STOP system, W2 stability and signed accounting rows, action events with successor deltas, per-scope cost allocation, and BC-seed/P8 common-prefix divergence. Keep `FULL_PLANNER_DIAGNOSTIC`, `AUTONOMOUS_PREFIX`, and `POST_STOP_FROZEN_SUFFIX` labels explicit.

- [ ] **Step 5: Add paired exploratory effects**

  Reuse `make_domain_bootstrap_draws()` and `paired_domain_bootstrap()` for `Delta_A`, `Delta_U`, and `Delta_G`, averaging BC seeds within each specimen and using each P8 specimen once. Preserve 97.5% historical Path B results as referenced anchors; use 95% only for diagnostic effects.

- [ ] **Step 6: Run focused tests and Ruff**

  Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_frozen_process_analysis.py`

  Run: `python -m ruff check src/cmc_bbdm/learned_cscan/frozen_process_analysis.py tests/test_bc_cscan_frozen_process_analysis.py`

### Task 3: Stored-Action Reader Recovery and W4 Spatial Evidence

**Files:**
- Create: `src/cmc_bbdm/learned_cscan/frozen_process_recovery.py`
- Modify: `tests/test_bc_cscan_frozen_process_analysis.py`

**Interfaces:**
- Consumes: `FrozenProcessConfig`, source root, frozen trajectories, cached `SurfacePercept`, existing `_full_reference`, `read_visible_task_report`, `_report_digest`, and stored actions.
- Produces: `recover_spatial_analysis()` with surface agreement, action-level spatial records, episode summaries, report cache metadata, and exact replay counts.

- [ ] **Step 1: Add failing recovery helper tests**

  On a synthetic grid, assert V/R/O exclusivity, exact added-position allocation despite shared cell boundaries, distinction between action-cell region, newly measured target positions, and report support, and `NOT_APPLICABLE` when V is never entered.

- [ ] **Step 2: Run recovery tests and verify RED**

  Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_frozen_process_analysis.py -k spatial`

- [ ] **Step 3: Implement one-pass sequential replay**

  Load context/percepts once, form each specimen/task full Reader proxy target once, then replay P8 and BC S1/S2/S3 actions in row order. At each state use `read_visible_task_report()` and verify `_report_digest`; at each action use `action_added_positions_from_mask()` before `world.step()`. Never call Actor, STOP, VLM, route compilation, `_run_planner_episode`, or old rescore entry points.

- [ ] **Step 4: Implement W4 summaries and recovery manifest**

  Record continuous V/D coverage, precision proxy and IoU; fixed `EMPTY_TARGET/NO_CUE/CUE_OVERLAP/CUE_DISJOINT` groups; first target measurement, first target support, LOCATE support criterion timing, V/R/O added cost, and first post-V outside action. Record report-only hash validation and exact replay steps/cache reuse.

- [ ] **Step 5: Run recovery unit tests and Ruff**

  Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_frozen_process_analysis.py -k spatial`

  Run: `python -m ruff check src/cmc_bbdm/learned_cscan/frozen_process_recovery.py`

### Task 4: External Reference, Blind Review, and Human Session Finalization

**Files:**
- Create: `src/cmc_bbdm/learned_cscan/frozen_evidence_finalize.py`
- Modify: `src/cmc_bbdm/learned_cscan/frozen_process_recovery.py`
- Modify: `tests/test_bc_cscan_frozen_process_analysis.py`

**Interfaces:**
- Consumes: optional reference JSON directory, blind-review CSV, human-session CSV/JSON, frozen trajectory/report identities, and W0-W5 outputs.
- Produces: `finalize_frozen_evidence()` and the complete W6-W8 reviewed/human/claim outputs.

- [ ] **Step 1: Add failing reviewed-reference tests**

  Build synthetic frozen report scores in which a new reference changes success while stop step, cost, action count, and report hash remain fixed. Assert proxy rows never enter reviewed estimates, partial coverage is explicit, and a task is supported only when both 97.5% lower-bound conditions pass.

- [ ] **Step 2: Add failing external-result association tests**

  Assert two reviewers of one report remain one physical specimen; raw reviewer decisions are preserved without invented consensus; unmatched human sessions produce `UNMATCHED_DATA` and no paired row; absent tracks are `PENDING_USER_INPUT` while supplied tracks still complete.

- [ ] **Step 3: Run W6-W8 tests and verify RED**

  Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_frozen_process_analysis.py -k 'reviewed or external or human or path_b'`

- [ ] **Step 4: Implement reviewed reference import and frozen report rescoring**

  Validate `specimen_key`, image SHA, registered coordinates, review provenance, certain/uncertain masks, duplicates, and frozen episode availability. Replay only matching stored actions for P8, BC S1/S2/S3, and, when references really exist, the two seed-1 ablations. Score with `adapt_task_report_v2()` plus `evaluate_task_report()`. Recompute full-input, planner AUsC, first STOP results, risk/cost, ablations, and per-task Path B decisions by reference version.

- [ ] **Step 5: Implement blind-review and human-session adapters**

  Accept explicit IDs or full compound keys, retain every reviewer/session row, report missing provenance fields as `NOT_RECORDED`, deduplicate physical specimens only in denominators, and create pairs only when specimen, task, geometry, scoring version, and a frozen model trajectory match. Never infer costs, actions, report edits, or consensus.

- [ ] **Step 6: Implement the W8 claim ledger**

  Emit the eight required claims with `SUPPORTED`, `NOT_SUPPORTED`, `INSUFFICIENT_PRECISION`, `PENDING_INPUT`, or `NOT_EVALUATED`, and include exact evidence tables, version, N, seed count, permitted wording, and prohibited overstatement.

- [ ] **Step 7: Run all new tests and Ruff**

  Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_frozen_process_analysis.py`

  Run: `python -m ruff check src/cmc_bbdm/learned_cscan/frozen_evidence_finalize.py src/cmc_bbdm/learned_cscan/frozen_process_recovery.py tests/test_bc_cscan_frozen_process_analysis.py`

### Task 5: CLI, Artifacts, Figures, and End-to-End Execution

**Files:**
- Create: `scripts/analyze_bc_cscan_frozen_process.py`
- Modify: `src/cmc_bbdm/learned_cscan/frozen_process_analysis.py`
- Create: `results/bc_cscan_frozen_process_analysis/*` through execution
- Create: `artifacts/bc_cscan_frozen_process_analysis/*` through execution
- Modify: `tests/test_bc_cscan_frozen_process_analysis.py`

**Interfaces:**
- Consumes: `analyze --config --source-root` and `finalize --config --source-root [--references] [--blind-reviews] [--human-sessions]`.
- Produces: every v3-required CSV/Parquet/JSON/PNG/MD artifact, deterministic checksums, and a rerunnable command for later external inputs.

- [ ] **Step 1: Add failing CLI parser and pending-finalize tests**

  Assert both subcommands exist, external paths are optional for `finalize`, no training/evaluation subcommand exists, and empty external input produces concrete pending coverage tables rather than fabricated rows.

- [ ] **Step 2: Run CLI tests and verify RED**

  Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_frozen_process_analysis.py -k cli`

- [ ] **Step 3: Implement CLI and atomic writers**

  `analyze` writes W0-W5 in a deterministic order; `finalize` independently updates only supplied evidence tracks and always regenerates the claim ledger. Write empty Parquet files with explicit schemas and all CSV files with fixed headers.

- [ ] **Step 4: Implement three bounded figures and case manifest**

  Render planning-to-stop accounting, action-cost allocation, and surface/proxy process cases. Select at most six specimens using predeclared category/domain/hash ordering and `(domain_rank mod 3)+1` BC seed rotation. Save PNG only and verify dimensions, nonblank pixels, labels, and cropping.

- [ ] **Step 5: Execute frozen analysis**

  Run: `PYTHONPATH=src python scripts/analyze_bc_cscan_frozen_process.py analyze --config paper_v3/configs/bc_cscan_frozen_process_analysis.yaml --source-root /home/ww/paper3/cmc_damage_inference`

  Expected: 336 scalar episodes, 672 stop rows, 192 spatial recovery episodes, no more than 36,864 replay steps, zero model/VLM calls, and all W0-W5 outputs present.

- [ ] **Step 6: Execute pending finalize**

  Run: `PYTHONPATH=src python scripts/analyze_bc_cscan_frozen_process.py finalize --config paper_v3/configs/bc_cscan_frozen_process_analysis.yaml --source-root /home/ww/paper3/cmc_damage_inference`

  Expected: reviewed, blind-review, and human-session tracks explicitly `PENDING_USER_INPUT`; proxy process claims remain available; completion dimension is `COMPUTATION_COMPLETE_INPUT_PENDING`.

- [ ] **Step 7: Write provenance, input, results, claim, and handoff documents**

  Generate `INPUT_AND_CODE_BINDINGS.md`, `RESULTS_AND_LIMITATIONS.md`, `INPUTS_FROM_USER.md`, `SUBMISSION_EVIDENCE_MATRIX.md`, `FINAL_RESULTS_AND_CLAIM_BOUNDARIES.md`, and `CODEX_HANDOFF_FROZEN_PROCESS_ANALYSIS.md` from actual outputs and hashes. State observed facts, supported associations, and not-established claims separately.

- [ ] **Step 8: Generate and verify deterministic checksums**

  Write `results/bc_cscan_frozen_process_analysis/CHECKSUMS.sha256` over all result files except itself and `_work`, then run `sha256sum -c CHECKSUMS.sha256` from the result root.

### Task 6: Integrity Audit, Final Verification, Commit, and Push

**Files:**
- Inspect all new files and the full base diff.
- Do not modify historical scientific roots.

**Interfaces:**
- Consumes: completed implementation and generated outputs.
- Produces: verified clean branch synchronized to `origin/research/bc-cscan-frozen-process-analysis`.

- [ ] **Step 1: Run focused test and lint gates**

  Run: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_bc_cscan_frozen_process_analysis.py`

  Run: `python -m ruff check src/cmc_bbdm/learned_cscan/frozen_process_analysis.py src/cmc_bbdm/learned_cscan/frozen_process_recovery.py src/cmc_bbdm/learned_cscan/frozen_evidence_finalize.py scripts/analyze_bc_cscan_frozen_process.py tests/test_bc_cscan_frozen_process_analysis.py`

- [ ] **Step 2: Verify outputs and immutable history**

  Run source checksum verification, row/schema assertions, figure pixel checks, recovery caps, and `git diff --name-only fe58de39298c412d0829580cfda40b7c7c4c53e9 -- results/bc_cscan_path_b_supplement results/learned_cscan_same_perception src/cmc_bbdm/learned_cscan/bc_supplement.py src/cmc_bbdm/learned_cscan/supplement_reporting.py`.

  Expected: the frozen-path diff is empty.

- [ ] **Step 3: Review the complete diff and secret/large-file scope**

  Run: `git status --short`, `git diff --check`, `git diff --stat`, and inspect every changed path. Confirm no model files, raw images, personal identities, unrelated formatting, or unbounded caches were added.

- [ ] **Step 4: Commit implementation and evidence**

  Stage only intended files. Use one implementation commit and one results/handoff commit if that keeps generated evidence auditable.

- [ ] **Step 5: Push without force and verify three SHAs**

  Run: `git push -u origin research/bc-cscan-frozen-process-analysis`, then compare `git rev-parse HEAD`, `git rev-parse @{upstream}`, and `git ls-remote origin refs/heads/research/bc-cscan-frozen-process-analysis`.

  Expected: all three SHAs are identical and `git status --short` is empty.
