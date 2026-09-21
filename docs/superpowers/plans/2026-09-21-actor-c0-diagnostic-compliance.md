# Actor C0 Diagnostic Compliance Revision Plan

> **Task:** `CAI_ACTOR_C0_MECHANISM_VIS_R2_9e765b04`
>
> **Input:** The 2026-09-21 package is byte-identical to the completed
> 2026-09-20 package. Reuse the published diagnostic cache and cumulative
> resource ledger. Do not run any Actor, predictor, CNN, Qwen, OOF, training,
> TEST, bootstrap, or paper-writing computation.

**Goal:** Close the report and provenance gaps found by an independent review
of the cached delivery, then republish the corrected offline report and
machine-verifiable manifest on the bound branch.

**Architecture:** Add a cache-only reporting module between diagnostic outputs
and rendering. It derives state provenance and hash bindings from existing
trajectory JSON, physical-state NPZ, score CSV, and frozen VLM feature CSV.
Rendering consumes those enriched records to build the required Chinese
native-trajectory and fixed-state comparison views, plus complete action and
intervention summaries. Validation enforces the new report contract without
calling model code.

**Stack:** Python 3, NumPy, Matplotlib/Pillow, stdlib CSV/JSON/HTML, pytest.

---

## Task 1: Lock Cache-Only Provenance Semantics

**Files:**
- Create: `scripts/cai_actor_c0_diagnostic/reporting.py`
- Create: `tests/test_cai_actor_c0_diagnostic_reporting.py`

1. Add failing tests for canonical array hashing, C-prior hashing over all four
   cached channels, policy-input hashing, source-labelled state origins,
   ordered prefix-cell reconstruction, and cache-only enforcement.
2. Implement deterministic pure helpers that read the existing C/N native
   trajectories and frozen VLM feature CSV, never model loaders or checkpoints.
3. Enrich every state row and `metadata.json` with `state_origins`,
   `prefix_cells_in_order`, `t`, `policy_input_sha256`, and `c_prior_sha256`.
4. Preserve the existing physical-state hash definition and prove that C/N
   policy-input hashes differ while the shared physical hash remains identical.

## Task 2: Complete Cached Visual Evidence and HTML Navigation

**Files:**
- Modify: `scripts/cai_actor_c0_diagnostic/render.py`
- Modify: `tests/test_cai_actor_c0_diagnostic_render.py`

1. Add failing tests for two top-level Chinese views, C-source/N-source
   fixed-state tabs, distinct toggle labels, source-metadata links, action
   annotations, and a trajectory-difference table.
2. Extend group A with legal centered-logit/rank panels and explicit
   `a_C0`, `a_free`, `a_N`, blocked-mass annotations from cached score tables.
3. Extend group F with C-native, N-native, and C-no-C0 action sequences plus a
   deterministic step/action/cost-difference table.
4. Build Chinese case/state navigation with separate `原生轨迹对照` and
   `相同状态对照` views; split the latter into `C来源状态` and `N来源状态`.
5. Link each state to its PNG/CSV/NPZ and `metadata.json`,
   `physical_state.npz`, `zero_prior_sensitivity.json`, and
   `query_checks.json` sources. Keep all assets repository-relative and local.
6. Ensure every figure title carries specimen, state source, time/prefix,
   exact cost, and target-action context where applicable.

## Task 3: Strengthen Contract Validation

**Files:**
- Modify: `scripts/cai_actor_c0_diagnostic/validate.py`
- Modify: `tests/test_cai_actor_c0_diagnostic_validate.py`

1. Add failing tests that reject missing state-provenance/hash fields and
   missing HTML view/tab/source-link contracts.
2. Validate every enriched metadata record against trajectory prefixes,
   physical-state files, policy score inputs, and cached C-prior channels.
3. Validate required group A/F assets, action annotations, local links, and
   equal-weight six-case t0 summary data if emitted.
4. Keep the resource ledger unchanged and continue enforcing all Q1-Q8 gates.

## Task 4: Render, Inspect, and Republish

**Files:**
- Update: `results/cai_agent_v3/actor_c0_diagnostic/r2_9e765b04/**`
- Update: `artifacts/cai_agent_v3/actor_c0_diagnostic/r2_9e765b04/**`

1. Run the cache-only render stage, then the strengthened verify stage. Do not
   run `prepare`, `replay`, `diagnose`, or the old two-commit `publish` stage.
2. Confirm the resource-count JSON is byte-identical before and after render
   and verification.
3. Inspect representative group A/F images and desktop/mobile offline HTML
   screenshots; run the local-link audit and panel-alignment checks.
4. Run the targeted diagnostic suite, `git diff --check`, scope audit, and
   manifest hash audit.
5. Commit the corrective result/report changes once, push normally to the same
   branch, and confirm local/upstream/`ls-remote` equality plus tracked sample
   HTML, CSV, NPZ, PNG, handoff, and delivery records.

## Required Commands

```bash
/home/ww/miniconda3/bin/python3.13 -m pytest -q \
  tests/test_cai_actor_c0_diagnostic_reporting.py \
  tests/test_cai_actor_c0_diagnostic_render.py \
  tests/test_cai_actor_c0_diagnostic_validate.py

/home/ww/miniconda3/bin/python3.13 \
  scripts/cai_actor_c0_diagnostic/cli.py render \
  --config docs/cai/actor_c0_diagnostic/ACTOR_C0_DIAGNOSTIC_SCOPE.json

/home/ww/miniconda3/bin/python3.13 \
  scripts/cai_actor_c0_diagnostic/cli.py verify \
  --config docs/cai/actor_c0_diagnostic/ACTOR_C0_DIAGNOSTIC_SCOPE.json

git diff --check
git status --short
```
