# MVD Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether initial one-shot Mechanical Value headroom exists and, only if it does, whether that value is observable from deployable coarse observations.

**Architecture:** A new `cmc_bbdm.mvd` package binds immutable MVA authority without editing historical MVA code. M0 builds exact-cost frozen rankings and evaluates held-out target and interaction states; conditionally, M1 builds a leak-closed initial-value dataset and runs nested source-domain model selection. Each formal result is derived from compact raw evidence, hash-validated, and byte-replayed.

**Tech Stack:** Python 3.13, NumPy, Polars, SciPy, scikit-learn, PyTorch, pytest, Ruff.

---

### Task 1: Freeze authority and protocol

**Files:**
- Create: `docs/MVD_REPOSITORY_AUTHORITY_AUDIT.md`
- Create: `docs/MVD_M0_PROTOCOL.md`
- Create: `docs/MVD_M1_OBSERVABILITY_PROTOCOL.md`
- Create: `docs/MVD_CLAIM_EVIDENCE_MATRIX.md`
- Create: `paper_v3/configs/mvd_feasibility.yaml`

- [ ] Record the fresh FULL reproduction, candidate-bank states, historical hashes, exact split barriers, and frozen M0/M1 gates.
- [ ] Add a fail-closed config loader test that rejects any changed source hash, threshold, method, or output path.
- [ ] Run `python -m pytest -q tests/test_mvd_config.py` and require PASS.

### Task 2: Bind immutable initial values

**Files:**
- Create: `src/cmc_bbdm/mvd/authority.py`
- Create: `src/cmc_bbdm/mvd/initial_value_dataset.py`
- Create: `tests/test_mvd_candidate_bank_binding.py`
- Create: `tests/test_mvd_source_labels_strict_oof.py`

- [ ] Test rejection of changed CandidateBank state, writable arrays, roster drift, and any source-label fit containing the outer or query domain.
- [ ] Implement a readonly dataset containing only identity, domain, initial embedding, current prediction, eight candidate features, complete strict-OOF values, costs, and authority hashes.
- [ ] Run the two focused tests and require PASS.

### Task 3: Implement exact-cost one-shot selection

**Files:**
- Create: `src/cmc_bbdm/mvd/action_cost_audit.py`
- Create: `src/cmc_bbdm/mvd/one_shot_oracle.py`
- Create: `tests/test_mvd_reuses_authoritative_grid.py`
- Create: `tests/test_mvd_action_cost_exact.py`
- Create: `tests/test_mvd_one_shot_scores_once.py`
- Create: `tests/test_mvd_no_sequential_recompute.py`

- [ ] Write synthetic unequal-cost tests proving deterministic descending-score traversal, exact unique-location caps, skip-not-break behavior, frozen rankings, and no scoring callback after S0.
- [ ] Implement action-cost rows and immutable checkpoint plans using only MVA grid/state/action APIs.
- [ ] Run the four focused tests and require PASS.

### Task 4: Evaluate M0 and interaction

**Files:**
- Create: `src/cmc_bbdm/mvd/evaluation.py`
- Create: `src/cmc_bbdm/mvd/interaction_audit.py`
- Create: `src/cmc_bbdm/mvd/statistics.py`
- Create: `src/cmc_bbdm/mvd/m0_execution.py`
- Create: `tests/test_mvd_m0_evaluation.py`

- [ ] Test that outer P-A/P-B models contain no target rows, one-shot/reconstruction orders are specimen-specific, and all final predictions use the shared checkpoint P-B heads.
- [ ] Materialize fixed plans, encode checkpoint states, compute interaction diagnostics, aggregate equal-domain curves/AUEBC/B5, bootstrap both primary effects, and apply the frozen M0 gate.
- [ ] Run all six outer folds, aggregate formal evidence, and stop immediately if M0 is NO-GO.

### Task 5: Implement conditional M1 observability

**Files:**
- Create: `src/cmc_bbdm/mvd/observability_dataset.py`
- Create: `src/cmc_bbdm/mvd/observability_models.py`
- Create: `src/cmc_bbdm/mvd/observability_metrics.py`
- Create: `src/cmc_bbdm/mvd/m1_execution.py`
- Create: `tests/test_mvd_student_never_reads_true_cai.py`
- Create: `tests/test_mvd_student_never_reads_candidate_embedding.py`
- Create: `tests/test_mvd_student_never_reads_full_scan.py`
- Create: `tests/test_mvd_student_never_reads_unobserved_rgb.py`
- Create: `tests/test_mvd_outer_domain_never_trains_observability_model.py`

- [ ] Write leak tests that fail on every forbidden field or outer-domain row.
- [ ] Implement O0/O1/O2/O3, uncertainty, and random scorers with source-only nested selection and the frozen Ridge/MLP/loss grid.
- [ ] Implement per-specimen and equal-domain Spearman, NDCG, Recall, regret, value capture, bootstrap, and evaluation-only AUEBC advantage capture.
- [ ] Run M1 only when the verified M0 summary authorizes it, then apply the frozen M1 gate and stop before M2/M3.

### Task 6: Publish and replay evidence

**Files:**
- Create: `src/cmc_bbdm/mvd/artifacts.py`
- Create: `src/cmc_bbdm/mvd/replay.py`
- Create: `src/cmc_bbdm/mvd/report.py`
- Create: `src/cmc_bbdm/mvd/cli.py`
- Create: `scripts/run_mvd.py`
- Create: `tests/test_mvd_replay.py`

- [ ] Validate required files, schemas, derived-table recomputation, privacy, source hashes, stop status, manifest tree, and checksum ledger.
- [ ] Copy only a validated formal package into replay and require byte identity.
- [ ] Render reports that answer every M0/M1 question in the controlling prompt.

### Task 7: Audit external data without performance

**Files:**
- Create: `src/cmc_bbdm/mvd/external_data.py`
- Create: `src/cmc_bbdm/mvd/cranfield.py`
- Create: `docs/EXTERNAL_CAI_DATA_FEASIBILITY_AUDIT.md`
- Create: `docs/CRANFIELD_RAW_PA_ACQUISITION_AUDIT.md`
- Create: `tests/test_external_manifest_has_no_model_results.py`
- Create: `tests/test_cranfield_raw_pairing.py`

- [ ] Download official DOI records, hash archives/files, unpack outside Git, and derive specimen/file pairing without invoking any predictor or acquisition method.
- [ ] Publish the four required manifests, grid schema, example mapping, external manifest JSON, and checksum ledgers.
- [ ] Fail validation if any external artifact contains method predictions, MAE, AUEBC, or model results.

### Task 8: Completion verification and integration

**Files:**
- Modify: `README.md`
- Modify: `docs/MVD_CLAIM_EVIDENCE_MATRIX.md`

- [ ] Run the complete focused MVD test suite with `python -m pytest -q tests/test_mvd_*.py tests/test_external_manifest_has_no_model_results.py tests/test_cranfield_raw_pairing.py`.
- [ ] Run `ruff check src/cmc_bbdm/mvd scripts/run_mvd.py tests/test_mvd_*.py tests/test_external_manifest_has_no_model_results.py tests/test_cranfield_raw_pairing.py`.
- [ ] Validate formal/replay packages and all checksum ledgers; compare formal/replay recursively.
- [ ] Inspect the full Git diff, verify no historical MVA file changed, commit, push, and require a clean worktree with local `main == origin/main`.
