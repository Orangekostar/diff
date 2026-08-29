# Spatial Neural Probe Implementation Plan

> Execute stage gates in order. Stop on any integration failure. Scientific
> outputs are written only to the new neural-probe namespace.

**Goal:** Implement and evaluate the fixed `spatial_grid_cnn_v1` encoder without
changing frozen MAVIS evidence or runtime contracts.

**Architecture:** A parallel package provides the spatial encoder, P2 wrapper,
training/checkpoint logic, nested-LODO execution, unchanged-P3 integration, and
rollout adapter. Existing feature-bank, metric, bootstrap, reveal, and rollout
functions remain authoritative.

**Stack:** Python, PyTorch, NumPy, pandas/Parquet, pytest, Ruff, Git.

---

### Task 1: Freeze N0 protocol

**Create:**
- `artifacts/mavis_neural_probe/N0_INTEGRATION_AUDIT.md`
- `artifacts/mavis_neural_probe/NEURAL_PROBE_PROTOCOL.md`
- `results/mavis_neural_probe/n0/*`

Record repository identity, frozen hashes, legal-state data flow, parameter
counts, split proof, resource estimate, and the `N1_AUTHORIZED` decision.
Validate with `git diff --check` and commit the documents separately.

### Task 2: Test and implement the spatial encoder

**Create:**
- `tests/test_mavis_neural_probe_state_encoder.py`
- `src/cmc_bbdm/mavis/neural_probe/__init__.py`
- `src/cmc_bbdm/mavis/neural_probe/state_encoder.py`

Write failing tests for row-major reshape, mask channel, shapes, parameter count,
empty/static behavior, finite output, gradient flow, and fixed-seed determinism.
Implement only enough code for those tests, then run the test file and Ruff.

### Task 3: Test and implement P2 model/checkpoints

**Create:**
- `tests/test_mavis_neural_probe_training.py`
- `src/cmc_bbdm/mavis/neural_probe/mechanics.py`
- `src/cmc_bbdm/mavis/neural_probe/training.py`
- `src/cmc_bbdm/mavis/neural_probe/artifacts.py`

Test the 64-D embedding/scalar prediction interface, source-only normalizer,
deterministic training, early stopping, checkpoint round trip, schema rejection,
and provenance/hash validation. Reuse existing fold and metric data structures.

### Task 4: Test and implement N1 execution

**Create:**
- `tests/test_mavis_neural_probe_execution.py`
- `src/cmc_bbdm/mavis/neural_probe/execution.py`

Test synthetic nested-LODO isolation, four-mode execution, target exclusion,
manifest creation, and deterministic summaries. Execute the real frozen bank,
write `n1_spatial_p2`, cross-check frozen DeepSets metrics at runtime, calculate
the registered contrast, and assign Gate 1.

### Task 5: Integrate unchanged P3 and execute N2

Extend execution tests first. Add a spatial-P2 embedding adapter while importing
the existing `DynamicActionScorer`, candidate features, loss, training groups,
metrics, and bootstrap. Execute the same six-domain nested LODO, write
`n2_dynamic_p3`, calculate `DeepSets_regret - Spatial_regret`, and assign Gate 2.

### Task 6: Execute N3 content attribution

Use the already trained four P2/P3 modes. Report full-bank and fixed
`CLEAN_NONPRIV = {uniform, random}` results with unified
`control_minus_real` signs. Preserve existing metric values and record only the
explicit exploratory conversion. Write `n3_content_attribution` and Gate 3.

### Task 7: Test rollout adapter and execute N4

**Create:**
- `tests/test_mavis_neural_probe_policy.py`
- `src/cmc_bbdm/mavis/neural_probe/policy.py`

Write failing tests for the `score_actions` protocol, identical candidate roster,
deterministic scores, legal-state-only access, exact-cost behavior, and P2 closed-
loop evaluator interface. Implement `SpatialProbeDeployedScorer`; reuse
`rollout_scout_and_focus_curve` and existing closed-loop metrics. Execute all 276
specimens, read frozen learned/static baselines from artifacts, write N4, and
assign Gate 4.

### Task 8: Final audit and Git synchronization

Run new tests, relevant MAVIS regressions, full requested suite, Ruff,
`git diff --check`, frozen tree diff, canonical SHA, action/cost roster checks,
artifact checksum verification, and deterministic rerun checks. Create
`artifacts/mavis_neural_probe/FINAL_GO_NOGO.md` using the claim authorization
matrix. Do not edit the paper. Commit in auditable stage order and push the new
branch without force.
