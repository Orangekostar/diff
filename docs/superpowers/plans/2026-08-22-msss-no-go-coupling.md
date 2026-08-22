# MSSS NO-GO Coupling Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the controlling prompt's S1 `NO_GO` branch as a reproducible post-hoc configuration-dependent scale diagnostic without running S2 or promoting MSSS claims.

**Architecture:** A pure diagnostic module consumes the validated formal S1 cross-fitted candidate errors and immutable structural authorities, computes group-balanced scale curves and deterministic local selections, then publishes a separate checksum-bound package. The formal S1 protocol, configuration, package, and replay remain byte-identical.

**Tech Stack:** Python 3.13, NumPy, standard-library CSV/JSON/hashlib, existing `cmc_bbdm.msss` authorities and artifact validation, pytest.

---

### Task 1: Group Curves And Local Selection

**Files:**
- Create: `src/cmc_bbdm/msss/coupling.py`
- Create: `tests/test_msss_coupling.py`

- [ ] **Step 1: Write failing tests**

Test stable rank-balanced tertiles, equal-domain rather than specimen-weighted
MAE, the coarsest 5%-eligible candidate, and rejection of incomplete candidate
rosters.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_msss_coupling.py`

Expected: import failure for `cmc_bbdm.msss.coupling`.

- [ ] **Step 3: Implement the minimal pure diagnostic**

Define immutable curve, selection, trend, and result records. Accept explicit
specimen IDs, domains, structural labels, physical damage values, candidate
registry, and absolute errors. Create domain/ply/layup and three physical
tertile groupings, calculate equal-domain curves, and reuse the frozen 5%
fine-to-coarse selection semantics.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_msss_coupling.py`

Expected: all deterministic aggregation and selection tests pass.

### Task 2: Direction Audit

**Files:**
- Modify: `src/cmc_bbdm/msss/coupling.py`
- Modify: `tests/test_msss_coupling.py`

- [ ] **Step 1: Add failing trend tests**

Cover increasing, decreasing, equal, and non-monotonic three-level sequences;
signed layup contrasts; and the requirement that at least two axes agree before
reporting `CROSS_AXIS_ALIGNED`.

- [ ] **Step 2: Verify RED**

Run the new test nodes and require the missing trend API to fail.

- [ ] **Step 3: Implement minimal trend classification**

Classify coarse-rank sequences without fitting or significance claims. Emit
factor-level consistency counts and keep validation status permanently
`NOT_VALIDATED_POST_HOC`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_msss_coupling.py`

Expected: all trend cases pass.

### Task 3: Parent Loading, Atomic Artifacts, And Replay

**Files:**
- Create: `src/cmc_bbdm/msss/coupling_artifacts.py`
- Create: `scripts/run_msss_coupling.py`
- Create: `tests/test_msss_coupling_artifacts.py`

- [ ] **Step 1: Write failing package tests**

Require an immutable formal parent with `NO_GO`, the exact registered digest,
complete prediction/authority rosters, all eight output files, atomic
non-overwrite publication, checksums, and prediction-level replay.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_msss_coupling_artifacts.py`

Expected: missing package implementation.

- [ ] **Step 3: Implement loading and publication**

Validate `results/msss/s1_scale_discovery`, parse only primary candidate rows,
bind them by specimen to `MSSSAuthority`, write canonical CSV/JSON/report
artifacts in a same-parent staging directory, validate them, and atomically
rename. Replay recomputes every curve, selection, and trend from the parent
before comparing package digests.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_msss_coupling_artifacts.py`

Expected: clean and replayed packages validate; corruption, parent-gate drift,
and overwrite attempts fail closed.

### Task 4: Formal Execution And Claim Audit

**Files:**
- Create: `results/msss/s1_no_go_coupling/*`
- Modify: `docs/MSSS_CLAIM_EVIDENCE_RESULT.md`
- Create: `docs/MSSS_COMPLETION_AUDIT.md`

- [ ] **Step 1: Run focused tests and lint**

Run the coupling tests, the full MSSS-focused suite, Ruff, and compileall.

- [ ] **Step 2: Execute the diagnostic**

Run: `PYTHONPATH=src python scripts/run_msss_coupling.py --project-root .`

Expected: a validated immutable package derived from the formal S1 `NO_GO`
parent, with no S2 directory created.

- [ ] **Step 3: Replay and audit claims**

Run the CLI with `--replay` to a temporary output and compare scientific
digests. Update only the post-run claim ledger and completion audit; do not
modify any hash-bound preregistration source or the formal S1 package.

- [ ] **Step 4: Sync and verify the compact Git repository**

Copy the exact new source, tests, docs, scripts, and formal diagnostic package
to the compact Git repository root, run focused validation there, commit, push
`main`, and confirm `HEAD == origin/main` with a clean worktree.
