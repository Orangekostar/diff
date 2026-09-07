# VLM C-scan Success-Efficiency Implementation Plan

> **For Codex:** Execute this plan in the isolated
> `research/vlm-cscan-success-efficiency` worktree. Follow strict TDD for
> production behavior. Only bounded mechanical leaves may be delegated.

**Goal:** Deliver and execute the frozen real-RGB, frozen-VLM, causal C-scan
pilot that separates surface-plan, deterministic-feedback, and VLM-replanning
contributions while preserving null formal success when reviewed references are
absent.

**Architecture:** A new `cmc_bbdm.vlm_cscan` package wraps the existing P0R
surface authority, acquisition grid, MAVIS authority, and causal world. Typed
contracts isolate policy-visible state from hidden reference evaluation. One
shared reader, route compiler, reporter, and stop checker serve B0-B7. A thin
CLI prepares resumable manifests, runs cached VLM calls and episodes, then
evaluates/saves compact artifacts.

**Tech Stack:** Python 3.13, NumPy, Pillow, SciPy, scikit-learn, Polars,
PyYAML, PyTorch, Transformers 4.49, Qwen2.5-VL-7B-Instruct.

---

### Task 1: Core contracts, references, reader, and route

**Files:**
- Create: `tests/test_vlm_cscan_core.py`
- Create: `tests/test_vlm_cscan_evaluation.py`
- Create: `src/cmc_bbdm/vlm_cscan/contracts.py`
- Create: `src/cmc_bbdm/vlm_cscan/references.py`
- Create: `src/cmc_bbdm/vlm_cscan/reader.py`
- Create: `src/cmc_bbdm/vlm_cscan/route.py`

1. Write literal synthetic tests for three-state evidence, unknown preservation,
   polygon/reference provenance, LOCATE full-frame rejection, CHARACTERIZE
   uncertain-band scoring, shared-boundary de-duplication, and hand-computed
   normalized route distances.
2. Run both test files and confirm failure from the missing package.
3. Implement the smallest typed contracts and functions that satisfy the tests.
4. Run both files green and run Ruff on only these files.
5. Commit: `feat: add reviewed-reference benchmark and real surface adapter`.

### Task 2: Real runtime and surface rendering

**Files:**
- Extend: `tests/test_vlm_cscan_core.py`
- Create: `src/cmc_bbdm/vlm_cscan/runtime.py`

1. Add a synthetic asymmetric-corner test proving clockwise ROT90 and non-1x1
   output, plus a real six-domain smoke manifest test.
2. Confirm the new tests fail because runtime is missing.
3. Implement strict YAML/source hash loading, deterministic pilot selection,
   true-PNG decode, full-FOV rotation/resize, grid rendering, MAVIS/world setup,
   and policy-visible snapshots.
4. Run targeted tests and a six-image prepare smoke.

### Task 3: Cached VLM, initial plans, and feedback policies

**Files:**
- Create: `tests/test_vlm_cscan_planning.py`
- Create: `src/cmc_bbdm/vlm_cscan/vlm.py`
- Create: `src/cmc_bbdm/vlm_cscan/planning.py`

1. Add tests for strict JSON/cell bounds, abstention fallback, exact B3/B5/B6
   initial-plan identity, legal-menu-only replanning, format retry limit, and a
   cache hit preserving original latency metadata.
2. Confirm the tests fail from missing behavior.
3. Implement deterministic baseline orders, shared initial-plan compiler,
   deterministic feedback, observed-only RBF adaptation, event/menu generation,
   Qwen Transformers backend, and JSONL cache.
4. Run targeted tests and Ruff.
5. Load the exact model revision once and run the six-image real JSON smoke.
6. Commit: `feat: add VLM initial planning and causal feedback tools`.

### Task 4: Reporting, episode runner, metrics, artifacts, and CLI

**Files:**
- Extend: `tests/test_vlm_cscan_evaluation.py`
- Extend: `tests/test_vlm_cscan_planning.py`
- Create: `src/cmc_bbdm/vlm_cscan/reporting.py`
- Create: `src/cmc_bbdm/vlm_cscan/benchmark.py`
- Create: `src/cmc_bbdm/vlm_cscan/metrics.py`
- Create: `src/cmc_bbdm/vlm_cscan/artifacts.py`
- Create: `scripts/run_vlm_cscan.py`

1. Add tests for common reporting/stop behavior, STOP cost freeze, anytime vs
   autonomous curves, missing-reference null success, resume without duplicate
   reveal/cost, and equal-domain paired bootstrap on a literal fixture.
2. Confirm those tests fail.
3. Implement reporting, B0-B7 episode execution, two evaluation modes, compact
   rows, metrics/bootstrap, deterministic serialization/checksums, and the
   prepare/export/infer/run/evaluate CLI.
4. Run the three test files and Ruff.
5. Run one real non-VLM micro episode and inspect reveal/cost invariants.
6. Commit: `feat: add common route costs and completion metrics`.

### Task 5: Bounded real pilot

**Files:**
- Generate only: `results/vlm_cscan_efficiency/**`

1. Run `prepare` and `export-annotations` for the fixed 60-specimen pilot.
2. Run/verify the six-domain VLM smoke, then cache initial plans for all 60.
3. Run B0-B7 for LOCATE and CHARACTERIZE, with real C-scan reveal and no formal
   success claims. Resume rather than restart after interruption.
4. Run evaluation once; generate aggregate proxy diagnostics, comparisons,
   annotation queue, 12 examples, summary, and checksums.
5. Verify formal success is null and coverage is zero while proxy columns are
   explicitly labeled.
6. Commit: `exp: run bounded VLM C-scan pilot`.

### Task 6: Final audit and delivery

**Files:**
- Create: `artifacts/vlm_cscan_efficiency/ANNOTATION_GUIDE.md`
- Create: `artifacts/vlm_cscan_efficiency/BENCHMARK_PROTOCOL.md`
- Create: `artifacts/vlm_cscan_efficiency/CODEX_HANDOFF_VLM_CSCAN_EFFICIENCY.md`

1. Record actual data/model revisions, reference coverage, methods/tasks,
   runtime/calls, results, limitations, and exact resume/import commands.
2. Read and apply `verification-before-completion` and
   `finishing-a-development-branch`.
3. Run only new-package Ruff, the 2-3 focused test files, one real micro episode,
   `git diff --check`, result checksum validation, and one base-to-HEAD frozen
   path diff. Do not rerun old suites.
4. Inspect the complete diff and generated sizes; reject leaked originals,
   weights, hidden reference inputs, or unrelated changes.
5. Commit: `docs: add VLM C-scan handoff`.
6. Push without force/PR/merge and verify local HEAD equals upstream and remote;
   require a clean worktree.
