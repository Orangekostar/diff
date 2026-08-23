# MVA A0-A3 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement and execute the preregistered A0-A3 retrospective
Mechanical-Value Acquisition experiment, validate/replay its artifacts, and
stop at a deterministic GO/NO-GO decision.

**Architecture:** A new `cmc_bbdm.mva` package binds the existing MGMR/P1 data
and feature authorities, extends the P5 endpoint-preserving raster semantics to
a nested 8 x 8 refinement grid, constructs strict OOF retrospective oracles,
and evaluates every acquisition method with common source-only P-B heads.
Deterministic tables are the authority; figures, reports, checksums, replay, and
the A3 decision are derived from those tables.

**Tech Stack:** Python 3.13, NumPy, PyTorch/Torchvision, Pillow, Polars Parquet,
SciPy, scikit-learn-compatible local ridge/PCA APIs, Matplotlib, PyYAML, pytest.

---

### Task 1: Freeze protocol and executable configuration

**Files:**
- Create: `paper_v3/configs/mva_a0_a3.yaml`
- Create: `src/cmc_bbdm/mva/__init__.py`
- Create: `src/cmc_bbdm/mva/config.py`
- Test: `tests/test_mva_config.py`

**Steps:**

1. Write failing tests requiring exact schema, cohort, budgets, methods, seeds,
   baseline tolerance, P5 endpoint, bootstrap, gates, output layout, and source
   hashes. Reject unknown keys, absolute paths, duplicates, nonfinite numbers,
   changed constants, and A4-A7 entries.
2. Run `python -m pytest -q tests/test_mva_config.py` and confirm failure due to
   the missing MVA package.
3. Implement an immutable validated config loader and frozen YAML using only
   repository-relative source paths.
4. Re-run the test and require it to pass.

### Task 2: Bind A0 authority and reproduce FULL

**Files:**
- Create: `src/cmc_bbdm/mva/authority.py`
- Test: `tests/test_mva_baseline_reproduction.py`

**Steps:**

1. Write failing tests that load exactly 276 ordered specimens, six registered
   domains, RGB bytes, targets, `metadata13`, and FULL 512-D features through
   the existing MGMR authority.
2. Require a fresh nested-LODO calculation using `metadata13`, fold-local PCA
   dimensions `(8,16,32)`, and Ridge alpha 10 to match the registered P1
   predictions within `1e-12` and MAE `0.08963580465761432` within `1e-12`.
3. Implement the authority wrapper and baseline stop without weakening existing
   P1/MGMR validation.
4. Run `python -m pytest -q tests/test_mva_baseline_reproduction.py`.

### Task 3: Implement the nested acquisition grid

**Files:**
- Create: `src/cmc_bbdm/mva/acquisition_grid.py`
- Create: `src/cmc_bbdm/mva/measurement_state.py`
- Test: `tests/test_acquisition_grid_nested.py`
- Test: `tests/test_measurements_only_added.py`
- Test: `tests/test_budget_counts_unique_measurements.py`

**Steps:**

1. Write parameterized failing tests for all native shapes `(674,675)`,
   `(338,352)`, `(338,340)` and initial survey candidates.
2. Assert 64 row-major cells, endpoint-preserving initial axes, cell-boundary
   alignment, `level0 subset level1 subset level2`, unique masks, legal one-step
   actions, monotonic measured counts, and exact measured/native budget.
3. Implement immutable grid/state/action types, mask generation, transition,
   checkpoint caps, and deterministic lower-index tie order.
4. Run the three focused test files.

### Task 4: Implement interpolation and P5 equivalence

**Files:**
- Create: `src/cmc_bbdm/mva/interpolation.py`
- Create: `src/cmc_bbdm/mva/refinement_simulator.py`
- Test: `tests/test_measured_values_restored_exactly.py`
- Test: `tests/test_mva_p5_equivalence.py`

**Steps:**

1. Write failing tests for deterministic bilinear mixed-cell reconstruction,
   exact RGB restoration, shape/dtype preservation, no removal, and input
   immutability.
2. Require an all-level-1 state to have exactly the P5 25% mask and byte-exact
   bilinear reconstruction for every registered shape and representative real
   specimens.
3. Implement cell-local interpolation with exact restoration and the global
   P5 fast path; add nearest/bicubic diagnostic modes without changing primary
   bilinear behavior.
4. Run the focused tests plus `tests/test_cpb_sparse_scan_sampling.py`.

### Task 5: Implement value functions and information barriers

**Files:**
- Create: `src/cmc_bbdm/mva/reconstruction_value.py`
- Create: `src/cmc_bbdm/mva/appearance_value.py`
- Create: `src/cmc_bbdm/mva/mechanical_value.py`
- Test: `tests/test_reconstruction_value_definition.py`
- Test: `tests/test_mechanical_value_definition.py`
- Test: `tests/test_oracle_candidate_does_not_access_future_state.py`
- Test: `tests/test_policy_never_reads_unobserved_pixels.py`
- Test: `tests/test_policy_never_reads_true_cai.py`

**Steps:**

1. Write failing unit tests with small synthetic images and spy/capability
   objects. Confirm normalized RGB MSE reduction, border-median appearance
   score, absolute CAI-error reduction, and squared-error secondary value.
2. Require deployable control APIs to accept only current state, legal action,
   observed values, seed, and public geometry. Restrict full image/true CAI to
   explicit retrospective oracle capabilities.
3. Implement pure value functions and capability-separated candidate views.
4. Run the five focused test files.

### Task 6: Implement strict cross-fitted CAI evaluators

**Files:**
- Create: `src/cmc_bbdm/mva/cai_evaluator.py`
- Create: `src/cmc_bbdm/mva/crossfit.py`
- Test: `tests/test_mva_oracle_uses_oof_predictor.py`
- Test: `tests/test_outer_domain_not_used_for_oracle_training.py`
- Test: `tests/test_outer_domain_not_used_for_policy_selection.py`

**Steps:**

1. Write failing tests on synthetic six-domain embeddings that inspect every
   PCA/Ridge fit roster and prove query/outer domains are absent.
2. Implement P-A FULL-trained strict OOF models with nested source-only PCA
   selection and batch prediction of reconstructed-image embeddings.
3. Implement one P-B model per outer domain/checkpoint, trained only on uniform
   source states and applied unchanged to all target methods.
4. Implement source-only initial-survey selection using the exact preregistered
   upper/headroom rule and a terminal failure when no candidate is viable.
5. Run the three leakage test files and baseline reproduction.

### Task 7: Build trajectories and deterministic controls

**Files:**
- Create: `src/cmc_bbdm/mva/oracle.py`
- Create: `src/cmc_bbdm/mva/oracle_trajectory.py`
- Test: `tests/test_mva_oracle_trajectory.py`
- Test: `tests/test_budget_curve_monotonic_measurement_count.py`

**Steps:**

1. Write failing tests for legal greedy actions, actual cap enforcement,
   deterministic ties, farthest-spread uniform order, PCG64 random replay, and
   checkpoint serialization.
2. Implement uniform, 100-seed random, appearance, reconstruction, and
   mechanical trajectory builders over the common simulator.
3. Store every considered oracle candidate value and every selected action,
   including fit-state IDs and actual measured counts.
4. Run the focused tests twice and require identical records.

### Task 8: Implement curves, budget metrics, bootstrap, and A3 gate

**Files:**
- Create: `src/cmc_bbdm/mva/budget_metrics.py`
- Create: `src/cmc_bbdm/mva/evaluation.py`
- Create: `src/cmc_bbdm/mva/statistics.py`
- Test: `tests/test_auebc.py`
- Test: `tests/test_b5_metric.py`
- Test: `tests/test_mva_statistics.py`
- Test: `tests/test_mva_gate.py`

**Steps:**

1. Write failing tests for trapezoidal AUEBC on `[0.0625,0.25]`, null
   sufficiency, savings, equal-domain aggregation, random quantiles, synchronized
   bootstrap reuse, and each H1-H4 boundary.
2. Implement the pure table calculations and one fixed 100000 x 6 PCG64 domain
   bootstrap matrix.
3. Require missing/adverse values to fail their gate and return exactly one
   terminal status.
4. Run the four focused test files.

### Task 9: Implement validated artifacts and replay

**Files:**
- Create: `src/cmc_bbdm/mva/artifacts.py`
- Create: `src/cmc_bbdm/mva/replay.py`
- Test: `tests/test_mva_artifacts.py`
- Test: `tests/test_mva_replay.py`

**Steps:**

1. Write failing tests for exact A0/A1/A2 layouts, Parquet schemas, CSV/JSON
   canonicalization, repository-relative provenance, checksum coverage,
   transactional publication, tamper rejection, and byte-identical replay.
2. Implement Polars Parquet writers plus independent recomputation validators.
3. Ensure manifests exclude lock files, temporary paths, timestamps, and local
   absolute paths.
4. Run the two focused test files.

### Task 10: Implement publication figures and report

**Files:**
- Create: `src/cmc_bbdm/mva/figures.py`
- Test: `tests/test_mva_figures.py`

**Steps:**

1. Write failing tests requiring O1-O5 and error-budget files, readable PDF/PNG,
   deterministic source-data tables, complete method legends, checkpoint axes,
   domain visibility, and no figure-to-gate dependency.
2. Implement figures from validated result tables only, using the fixed `c8-2`
   example rather than outcome-selected specimens.
3. Generate `REPORT.md` from validated summaries with failed gates, adverse
   domains, P-A/P-B distinction, and raster-simulation limitations.
4. Run the figure tests.

### Task 11: Wire CLI and execute A0-A3

**Files:**
- Create: `src/cmc_bbdm/mva/pipeline.py`
- Create: `src/cmc_bbdm/mva/cli.py`
- Create: `scripts/run_mva_a0_a3.py`
- Test: `tests/test_mva_cli.py`

**Steps:**

1. Write failing CLI smoke tests for `audit`, `simulate`, `run`, `validate`, and
   `replay`, including nonzero exit on baseline/gate-package invalidity.
2. Implement ordered A0-A3 execution with stage locks, caches, transactional
   output, and no import or call path to A4-A7.
3. Run all MVA tests, then execute:
   `python scripts/run_mva_a0_a3.py run --config paper_v3/configs/mva_a0_a3.yaml`.
4. Validate and replay with the same CLI. Compare formal/replay file hashes and
   record the terminal `MVA_ORACLE_GO` or `MVA_ORACLE_NO_GO` without rescue.

### Task 12: Final verification and Git publication

**Files:**
- Modify only as required by test-discovered defects in the files above.
- Copy the validated code/docs/results into the compact repository, preserving
  its existing compact-repository layout and exclusions.

**Steps:**

1. Run `python -m pytest -q` for the full relevant suite and record inherited
   failures separately from new failures.
2. Run MVA validation/replay, scan outputs for absolute local paths, verify no
   A4-A7 implementation exists, and inspect all figure files.
3. Inspect the complete source and compact-repository diffs; verify results are
   consistent with `summary.json` and `REPORT.md`.
4. Commit on the compact repository's `main` branch with an MVA A0-A3 message and push to
   `origin/main`.
