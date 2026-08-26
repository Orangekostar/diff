# MAVIS Science Closure Implementation Plan

**Goal:** Complete P8-P16, bind every scientific claim to machine-readable
evidence, and publish a deterministic science-closure package without changing
the frozen P7 endpoint.

**Architecture:** Add reuse-first analysis, diagnostic planner, strict-OOF
learner-stability, packaging, and replay modules under the current MAVIS
namespace. New results use a separate namespace and hash-bind all historical
inputs.

**Runtime:** `PYTHONPATH=src /home/ww/miniconda3/bin/python` with NumPy, Polars,
PyTorch, scikit-learn, SciPy, Pytest, and Ruff.

## Task 1: P8 reverse audit and literature boundary

**Files:**

- Create `artifacts/mavis_science_closure/P8_CURRENT_CODE_MAP.md`
- Create `artifacts/mavis_science_closure/P8_STAGE_RESULT_MAP.md`
- Create `artifacts/mavis_science_closure/P8_CLAIM_EVIDENCE_AUDIT.md`
- Create `artifacts/mavis_science_closure/P8_OPEN_SCIENTIFIC_GAPS.md`
- Create `artifacts/mavis_science_closure/P8_TIER_B_DECOMPOSITION.csv`
- Create `artifacts/mavis_science_closure/LITERATURE_LEDGER.md`

- [x] Record the exact 20-role code map and P1-P7 stage map.
- [x] Decompose B1-B4 using frozen aggregate, domain, and bootstrap rows.
- [x] Verify the six required literature boundaries from primary sources.
- [x] Verify MAVIS baseline tests and commit `audit: reverse-map MAVIS 716de19 code/results/claims`.

## Task 2: Closure contracts and P9 conditional value evolution

**Files:**

- Create `src/cmc_bbdm/mavis/science_closure.py`
- Create `src/cmc_bbdm/mavis/science_closure_artifacts.py`
- Create `tests/test_mavis_science_closure.py`
- Create `paper_v3/configs/mavis_science_closure.yaml`
- Create `results/mavis_science_closure/p9_value_evolution/*`

- [x] Write RED tests for frozen input hashes, state/action identity, strict-OOF
  provenance, conditional value changes, equal action-cost controls, and
  deterministic bootstrap.
- [x] Implement rank stability, top-K overlap, best-action turnover, value shift,
  and dynamic-vs-static opportunity from P1/P3 rows.
- [x] Write the required Parquet/CSV/report/summary package and verify hashes.
- [x] Commit `analysis: add conditional value evolution`.

## Task 3: P10 MRIS causal closure

**Files:**

- Modify `src/cmc_bbdm/mavis/science_closure.py`
- Modify `src/cmc_bbdm/mavis/science_closure_artifacts.py`
- Modify `tests/test_mavis_science_closure.py`
- Create `results/mavis_science_closure/p10_mris_causal/*`

- [x] Test reuse of frozen P2 predictions, identical cohort/cost/action rosters,
  and no use of future target measurements.
- [x] Produce state-cost curves, per-specimen predictions, domain contrasts,
  bootstrap, and information-accumulation summaries.
- [x] Commit `analysis: close MRIS causal informativeness`.

## Task 4: P11 dynamic valuation closure

**Files:**

- Modify `src/cmc_bbdm/mavis/science_closure.py`
- Modify `tests/test_mavis_science_closure.py`
- Create `results/mavis_science_closure/p11_dynamic_valuation/*`

- [ ] Test candidate-only, MVD O2, static M1, positions, shuffled, and dynamic
  scorer alignment on the same legal action rows.
- [ ] Compute regret, one-step utility, rank metrics, domain effects, cost strata,
  and paired bootstrap.
- [ ] Commit `analysis: close dynamic valuation comparisons`.

## Task 5: P12 representation-value-planning attribution

**Files:**

- Create `src/cmc_bbdm/mavis/science_closure_planning.py`
- Create `tests/test_mavis_science_closure_planning.py`
- Create `results/mavis_science_closure/p12_rvp_attribution/*`

- [ ] Write RED tests that oracle substitutions are non-deployable, preserve
  checkpoint hashes, and respect exact budgets.
- [ ] Evaluate the registered substitution matrix while changing one component
  at a time; record per-domain and per-budget attribution.
- [ ] Commit `analysis: add representation-valuation-planning substitution matrix`.

## Task 6: P13 set-level planning diagnosis

**Files:**

- Modify `src/cmc_bbdm/mavis/science_closure_planning.py`
- Modify `tests/test_mavis_science_closure_planning.py`
- Create `results/mavis_science_closure/p13_set_planning/*`

- [ ] Test joint utility, deterministic ties, no duplicate action, exact budget,
  and distinction from sums of point values.
- [ ] Compare greedy, beam/lookahead, and true-value diagnostic planners using
  frozen action candidates and downstream CAI utility.
- [ ] Commit `analysis: add set-level planning diagnosis`.

## Task 7: P14 task specificity

**Files:**

- Modify `src/cmc_bbdm/mavis/historical_sources.py`
- Modify `src/cmc_bbdm/mavis/science_closure.py`
- Modify `tests/test_mavis_science_closure.py`
- Create `results/mavis_science_closure/p14_task_specificity/*`

- [ ] Add hash-bound reading of the frozen A4 global reconstruction mask.
- [ ] Test same cohort/cost and frozen reconstruction metric.
- [ ] Evaluate the 2x2 reconstruction/mechanics objective-policy comparison and
  spatial overlap.
- [ ] Commit `analysis: add reconstruction-vs-mechanics task specificity`.

## Task 8: P15 downstream learner value stability

**Files:**

- Create `src/cmc_bbdm/mavis/value_stability.py`
- Create `tests/test_mavis_value_stability.py`
- Create `results/mavis_science_closure/p15_value_stability/*`

- [ ] Write RED tests that every learner uses identical strict-OOF splits and
  the same frozen state bank.
- [ ] Fit the registered simple source-only learner families with fixed settings.
- [ ] Compare action-rank, top-K, best-action, region, and oracle-acquisition
  agreement; report learner dependence without selecting on target domains.
- [ ] Commit `analysis: add downstream learner value-stability audit`.

## Task 9: P16 feedback mechanism and final evidence

**Files:**

- Modify `src/cmc_bbdm/mavis/science_closure.py`
- Create `results/mavis_science_closure/p16_feedback_mechanism/*`
- Create `artifacts/mavis_science_closure/MANUSCRIPT_CLAIM_MAP.md`
- Create `artifacts/mavis_science_closure/COMPLETION_AUDIT.md`

- [ ] Join action turnover/value shifts to frozen feedback/no-feedback outcomes.
- [ ] Build the claim map with allowed and forbidden wording.
- [ ] Audit every prompt requirement and historical P7 file hash.

## Task 10: Deterministic replay and integration

**Files:**

- Create `src/cmc_bbdm/mavis/science_closure_replay.py`
- Create `tests/test_mavis_science_closure_replay.py`
- Create `results/mavis_science_closure/replay/*`

- [ ] Write RED tests for deterministic replay and complete manifest hashes.
- [ ] Regenerate all derived packages and require byte identity.
- [ ] Run Ruff, all MAVIS/MVD regressions, scoped MVA regression, package
  verifiers, checksum checks, `git diff --check`, and P7 immutability audit.
- [ ] Commit `evidence: add manuscript claim map and science-closure replay`.
- [ ] Integrate the branch into `main`, push `origin/main`, and verify the remote
  commit and artifact paths.
