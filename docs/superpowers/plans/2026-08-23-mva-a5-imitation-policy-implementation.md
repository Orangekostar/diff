# MVA A5 Oracle-Imitation Policy Implementation Plan

> Execute task-by-task with test-driven development and review checkpoints.

**Goal:** Train and evaluate a leakage-safe supervised oracle-imitation policy,
publish a validated/replayed A5 evidence package, and issue A5/A6 decisions.

**Architecture:** Outer-specific source teachers regenerate full nested oracle
states with four-domain P-A fits. Immutable feature caches feed a small
float64 shared scorer. Target workers execute policy and deployable heuristics
using current observations only; aggregation reuses A2/A4 reference evidence.

**Tech stack:** Python 3.13, NumPy, PyTorch, Polars/Parquet, Matplotlib, SciPy,
PyYAML, pytest, Ruff.

### Task 1: Freeze A5 authority and configuration

- [ ] Add immutable A5 config bound to A2 and formal A4 manifests/statuses.
- [ ] Reject source drift, unknown keys, nonfinite settings, model growth above
  50k parameters, and any A6/A7 implementation entry.
- [ ] Add config authorization and tamper tests.

### Task 2: Implement observed-only policy state and candidate features

- [ ] Test exact 579/8 shapes, finite values, stable ordering, and budget fields.
- [ ] Test that changing unmeasured true pixels or true CAI cannot change policy
  tensors when the current reconstruction/state is held fixed.
- [ ] Implement `policy_state.py` and `candidate_features.py` without full-image
  or target arguments in public feature APIs.

### Task 3: Regenerate outer-safe teacher datasets

- [ ] Test the `outer d` / `query q` four-domain barrier and nested PCA audits.
- [ ] Test complete feasible-action values, deterministic teacher tie breaks,
  nested 0->1/1->2 actions, exact budgets, and content hashes.
- [ ] Implement transactionally saved outer teacher caches and independent load
  validation. Never reuse A2 oracle rows as training labels.

### Task 4: Implement and train the ranking policy

- [ ] Test parameter count, forward shapes, candidate masking, pairwise loss,
  equal-domain/specimen/state weights, deterministic training, and model load.
- [ ] Implement the fixed float64 shared scorer and 50-epoch full-weighted
  gradient-accumulation training loop.
- [ ] Serialize normalization statistics, weights, training digest, loss trace,
  and source roster in a canonical model package.

### Task 5: Implement deployable trajectories and heuristics

- [ ] Test that policy/center/appearance/reconstruction selectors accept no
  target or full-image evidence and always select a feasible action.
- [ ] Test observation updates, exact restoration, monotonic unique budgets,
  checkpoint snapshots, and deterministic tie breaks.
- [ ] Implement target execution with current reconstruction encoding and P-A
  prediction updates after each selected action.

### Task 6: Execute outer evaluation workers

- [ ] Fit one policy and outer-safe P-A/P-B evaluator bundle per target domain.
- [ ] Evaluate imitation, center-first, observed-gradient, and
  observed-uncertainty on every held-out specimen.
- [ ] Transactionally publish fit audits, model, training trace, decisions,
  trajectories, states, and shard completion hashes.

### Task 7: Aggregate and issue A5/A6 gates

- [ ] Validate all six shards, exact A2 P-B hashes, A4 reference binding, method
  roster, specimen/checkpoint completeness, and target-information barriers.
- [ ] Compute equal-domain curves, AUEBC, B metrics, specimen/domain metrics,
  synchronized bootstrap effects, gap closure, `MVA_A5_POLICY_*`, and
  `MVA_A6_*` statuses.
- [ ] Add row-order determinism and evidence-tamper tests.

### Task 8: Figures, report, artifacts, and replay

- [ ] Freeze a visual contract before reading A5 outcomes.
- [ ] Render policy error-budget curves, per-domain effects/gap closure, and
  teacher-imitation diagnostics from validated tables only.
- [ ] Generate REPORT.md, source-data CSV, manifest/checksums, and atomic formal
  package; independently validate and byte-replay it.

### Task 9: Formal execution and stop decision

- [ ] Run all six teacher/policy/evaluation workers with one-thread deterministic
  linear algebra and available GPU scheduling.
- [ ] Run full A5 tests, Ruff, package validation, and replay comparison.
- [ ] If A5/A6 gate fails, stop without rescue search. If authorized, freeze a
  separate A6 protocol before any laminate-conditioned implementation.

