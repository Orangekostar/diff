# MAVIS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a causal, strict-OOF, closed-loop mechanics-aware ultrasonic acquisition system without changing frozen MVA/MVD evidence.

**Architecture:** A typed privileged authority reveals only action-bound native-raster measurements into immutable deployable states. A permutation-invariant MRIS supports a mechanics head and conditional action preference model; exact-cost rollout, source-only aggregation, and source-selected fallback are evaluated once under frozen nested LODO.

**Tech Stack:** Python 3.10+, NumPy, Polars, PyTorch, scikit-learn, SciPy, PyYAML, Pytest, Ruff.

---

## File map

| File | Responsibility |
|---|---|
| `src/cmc_bbdm/mavis/contracts.py` | Immutable policy-visible, teacher-only, and evaluation-only data types. |
| `src/cmc_bbdm/mavis/config.py` | Strict YAML schema, source bindings, development/final freeze hashes. |
| `src/cmc_bbdm/mavis/authority.py` | Load the 276 upstream specimens and write/read a hash-bound MAVIS manifest. |
| `src/cmc_bbdm/mavis/reveal.py` | Uniform scout and exact action-bound reveal with budget/duplicate guards. |
| `src/cmc_bbdm/mavis/state_bank.py` | Strict-OOF source trajectories, state manifests, and state-action pairs. |
| `src/cmc_bbdm/mavis/state_encoder.py` | Static, position-only, real, shuffled, and reconstruction MRIS encoders. |
| `src/cmc_bbdm/mavis/mechanics_head.py` | Source-only normalization, fitting, CAI prediction, and fold audits. |
| `src/cmc_bbdm/mavis/dynamic_voi.py` | Conditional teacher labels and pairwise/listwise/value learning. |
| `src/cmc_bbdm/mavis/policy.py` | Legal exact-cost scoring, confidence, frozen-ranking ablation. |
| `src/cmc_bbdm/mavis/rollout.py` | Scout-and-focus loop and complete baseline roster. |
| `src/cmc_bbdm/mavis/aggregation.py` | Source-only on-policy visited-state aggregation. |
| `src/cmc_bbdm/mavis/fallback.py` | Source-selected robust baseline and threshold. |
| `src/cmc_bbdm/mavis/metrics.py` | CAI/reconstruction curves, AUEBC, regret, ranking, risk-coverage. |
| `src/cmc_bbdm/mavis/evaluation.py` | Nested LODO development and one-time frozen outer evaluation. |
| `src/cmc_bbdm/mavis/artifacts.py` | Machine-readable package, manifest, checksums, figures, report. |
| `src/cmc_bbdm/mavis/replay.py` | Deterministic package regeneration and byte/hash comparison. |
| `src/cmc_bbdm/mavis/cli.py` | Stage-specific build/train/evaluate/replay commands. |

## Task 1: P0 audit and design freeze

**Files:**
- Create: `artifacts/mavis/P0_LITERATURE_LEDGER.md`
- Create: `artifacts/mavis/P0_REPO_CODE_MAP.md`
- Create: `artifacts/mavis/P0_DATA_FLOW.md`
- Create: `artifacts/mavis/P0_FROZEN_EVIDENCE_LEDGER.md`
- Create: `artifacts/mavis/P0_MVD_AUTHORITY_SCHEMA.md`
- Create: `artifacts/mavis/P0_AUTHORITY_ARRAYS.csv`
- Create: `artifacts/mavis/MANUSCRIPT_EVIDENCE_MAP.md`
- Create: `docs/superpowers/specs/2026-08-25-mavis-design.md`

- [x] Re-verify repository, literature, authority arrays, frozen metrics, replay, and upstream raw loader.
- [x] Record privilege boundaries and the post-scout meaning of historical MVD initial embeddings.
- [x] Run `ruff check .`, focused MVD tests, historical MVA regression, and `git diff --check`.
- [x] Commit as `audit: map MAVIS authority and frozen evidence`.

## Task 2: Typed causal authority and reveal

**Files:**
- Create: `src/cmc_bbdm/mavis/__init__.py`
- Create: `src/cmc_bbdm/mavis/contracts.py`
- Create: `src/cmc_bbdm/mavis/config.py`
- Create: `src/cmc_bbdm/mavis/authority.py`
- Create: `src/cmc_bbdm/mavis/reveal.py`
- Create: `paper_v3/configs/mavis_development.yaml`
- Create: `paper_v3/configs/mavis_final.yaml`
- Create: `tests/test_mavis_authority.py`
- Create: `tests/test_mavis_reveal.py`
- Create: `tests/test_mavis_exact_cost.py`

- [ ] Write failing tests for a policy-visible API with no `full_scan` or `true_cai`, exact authoritative reveal, future-content invariance, duplicate rejection, budget rejection, and legacy MVA cost equality.

```python
context = authority.policy_context(specimen_id)
state = begin_inspection(context)
scouted = reveal_uniform_scout(authority, state, budget=0.015625)
next_state = reveal_action(authority, scouted, RefinementAction(7, 0, 1))
assert not hasattr(context, "true_cai")
assert np.array_equal(next_state.values[-added:], full_scan[mask])
assert next_state.acquired_count == budget_record(grid, legacy_state).measured_count
```

- [ ] Run `pytest -q tests/test_mavis_authority.py tests/test_mavis_reveal.py tests/test_mavis_exact_cost.py` and verify collection fails because `cmc_bbdm.mavis` is absent.
- [ ] Implement frozen dataclasses `PolicyContext`, `InspectionState`, `SourceTeacherView`, and `EvaluationView`; all numeric arrays are finite, copied, contiguous, read-only, and shape-checked.
- [ ] Implement `load_mavis_authority`, `policy_context`, private privileged lookup, and `source_teacher_view`; load the registered MGMR authority and require exact 276 IDs, six domains, scan hashes, metadata13, profile_stats21, and authority state hash.
- [ ] Implement `begin_inspection`, `reveal_uniform_scout`, and `reveal_action` by reusing `build_acquisition_grid`, `measurement_mask`, `apply_action`, and `budget_record`; reveal only newly added native positions and RGB values.
- [ ] Add strict development/final YAML loaders. Bind the upstream MVA config SHA, authority SHA, domain order, 276 cohort, scout budgets `(0.015625, 0.03125, 0.0625, 0.125)`, checkpoint `0.25`, and seed roster.
- [ ] Re-run the three focused tests, all existing MVA/MVD tests, Ruff, and `git diff --check`; expect PASS.
- [ ] Write `artifacts/mavis_authority/manifest.json`, `arrays.npz`, `CHECKSUMS.sha256`, and `REPORT.md` through code, verify checksums, and commit `data: add MAVIS authority and causal reveal`.

## Task 3: Strict-OOF sequential state bank

**Files:**
- Create: `src/cmc_bbdm/mavis/state_bank.py`
- Create: `src/cmc_bbdm/mavis/teacher.py`
- Create: `tests/test_mavis_state_bank.py`
- Create: `tests/test_mavis_teacher_oof.py`

- [ ] Write failing tests that every teacher row excludes the outer target domain and query specimen, every state cost is monotone, and target labels cannot change policy-visible rows.

```python
assert held_out_domain not in audit.fit_domains
assert query_specimen_id not in audit.fit_specimen_ids
assert state_rows.group_by("trajectory_id").agg(pl.col("exact_cost").is_sorted()).item()
```

- [ ] Verify RED with `pytest -q tests/test_mavis_state_bank.py tests/test_mavis_teacher_oof.py`.
- [ ] Generate random, uniform, reconstruction, one-shot oracle, and sequential oracle source trajectories at registered checkpoints. Teacher values are strict-OOF reductions in absolute CAI error and remain unavailable through target rollout types.
- [ ] Serialize `state_manifest.parquet`, `state_action_pairs.parquet`, fit audits, manifest, checksums, and report under `results/mavis/p1_state_bank/`; inference remains specimen/domain-level.
- [ ] Validate exact counts, no duplicate acquisition, source-only fit rosters, deterministic row order, and package hashes; commit `data: add strict-OOF MAVIS state bank`.

## Task 4: MRIS and mechanics head

**Files:**
- Create: `src/cmc_bbdm/mavis/state_encoder.py`
- Create: `src/cmc_bbdm/mavis/mechanics_head.py`
- Create: `tests/test_mavis_state_encoder.py`
- Create: `tests/test_mavis_mechanics_head.py`

- [ ] Write failing tests for permutation invariance, position-only value absence, deterministic stratified shuffled donors, and outer-domain normalization isolation.

```python
z1 = encoder(context, positions, values)
z2 = encoder(context, positions[permutation], values[permutation])
torch.testing.assert_close(z1, z2)
assert positions_only.measurement_values is None
```

- [ ] Verify RED, then implement a two-layer token MLP, sum/mean pooling, context fusion, and linear CAI head. Default hidden size is 64 and MRIS size is 64; no Transformer.
- [ ] Fit normalization and early stopping strictly on inner source folds. Evaluate S0 static, S1 positions, S2 real, S3 shuffled, and S4 existing reconstruction state at identical costs.
- [ ] Save per-state predictions, cost curves, domain/aggregate metrics, bootstrap, model-selection audit, checkpoints, hashes, report, and figures to `results/mavis/p2_mris/`; commit `model: add MRIS and mechanics head`.

## Task 5: Dynamic conditional mechanical VoI

**Files:**
- Create: `src/cmc_bbdm/mavis/dynamic_voi.py`
- Create: `src/cmc_bbdm/mavis/losses.py`
- Create: `tests/test_mavis_dynamic_voi.py`

- [ ] Write failing tests that target CAI and unacquired target pixels cannot affect scores, source teacher labels equal strict-OOF one-step CAI loss reduction, and pair ordering follows teacher utility.
- [ ] Verify RED, then implement a shared action scorer over MRIS, candidate geometry, incremental exact cost, and remaining budget. Optimize `L_cai + lambda_p L_pair + lambda_l L_list + lambda_v L_value`; all weights come from source-only selection.
- [ ] Compare MVD O2, static candidate-only, dynamic real, positions-only, and shuffled variants using regret, one-step downstream utility, Spearman, NDCG, and Recall@K.
- [ ] Save formal development artifacts to `results/mavis/p3_dynamic_voi/`; commit `model: add dynamic mechanical VoI`.

## Task 6: Exact-cost closed-loop rollout

**Files:**
- Create: `src/cmc_bbdm/mavis/policy.py`
- Create: `src/cmc_bbdm/mavis/rollout.py`
- Create: `tests/test_mavis_policy.py`
- Create: `tests/test_mavis_rollout.py`

- [ ] Write failing tests for budget limits, no duplicates, deterministic action ties, scorer calls after every reveal, and no-feedback ranking frozen immediately after scout.
- [ ] Verify RED, then implement legal-action filtering, value/cost and direct cost-aware variants, checkpoint guards, and the scout-and-focus loop.
- [ ] Run identical-cost Full, Uniform, Random, Reconstruction, Global Mechanical, MVA A5, MVD O2, no-feedback, positions-only, shuffled, MAVIS full, one-shot oracle, and sequential oracle trajectories.
- [ ] Save CAI/reconstruction curves, trajectories, AUEBC, domain/bootstrap tables, feedback and task-specificity ablations under `results/mavis/p4_closed_loop/`; commit `policy: add cost-aware closed-loop rollout`.

## Task 7: Source-only on-policy aggregation

**Files:**
- Create: `src/cmc_bbdm/mavis/aggregation.py`
- Create: `tests/test_mavis_aggregation.py`

- [ ] Write a failing test proving visited target states are never teacher-labeled or appended and each aggregation round records source-only specimen/domain rosters.
- [ ] Verify RED, then implement deterministic rounds: train, source rollout, source oracle label, deduplicate by state hash, append, retrain. Default maximum is three rounds with source-only early selection.
- [ ] Record round deltas, visited-state coverage, fit rosters, model hashes, and convergence diagnostics; commit `train: add source-only on-policy aggregation`.

## Task 8: Safe fallback

**Files:**
- Create: `src/cmc_bbdm/mavis/fallback.py`
- Create: `tests/test_mavis_fallback.py`

- [ ] Write failing tests that target outcomes cannot select the fallback baseline or confidence threshold and that low-confidence actions exactly match the source-selected baseline.
- [ ] Verify RED, then choose uniform or reconstruction baseline and a confidence threshold using the source lexicographic criterion only.
- [ ] Save risk-coverage, fallback frequency, CAI AUEBC, worst-domain error, selected threshold, and selection audit to `results/mavis/p5_safe_policy/`; commit `safety: add source-selected MAVIS fallback`.

## Task 9: Complete experiment and frozen evaluation

**Files:**
- Create: `src/cmc_bbdm/mavis/metrics.py`
- Create: `src/cmc_bbdm/mavis/evaluation.py`
- Create: `src/cmc_bbdm/mavis/artifacts.py`
- Create: `src/cmc_bbdm/mavis/report.py`
- Create: `src/cmc_bbdm/mavis/figures.py`
- Create: `src/cmc_bbdm/mavis/cli.py`
- Create: `tests/test_mavis_evaluation.py`
- Create: `tests/test_mavis_artifacts.py`

- [ ] Write failing tests for the complete E1-E10 roster, identical cohort/cost/endpoint, specimen/domain statistical units, one-time frozen config hash, and report values traceable to tables.
- [ ] Verify RED, implement nested source selection and one final frozen outer evaluation. Random specimen splits remain diagnostics only.
- [ ] Classify Tier S/A/B from the preregistered aggregate/domain/bootstrap and safe-system criteria without changing method or topic.
- [ ] Generate all tables/figures/report from machine-readable data under `results/mavis/p6_final_frozen_eval/`; update only evidence-map status and artifact references; commit `eval: complete MAVIS baselines and ablations`.

## Task 10: Replay, completion audit, integration, and push

**Files:**
- Create: `src/cmc_bbdm/mavis/replay.py`
- Create: `tests/test_mavis_replay.py`
- Create: `artifacts/mavis/MAVIS_COMPLETION_AUDIT.md`

- [ ] Write failing tests for deterministic replay, manifest coverage, checksum rejection of missing/unlisted/changed files, and unchanged historical MVA/MVD package hashes.
- [ ] Verify RED, implement replay into a temporary directory, compare the complete formal package byte-for-byte, and write a root replay summary.
- [ ] Run the fifteen named barrier tests and full MAVIS suite in the worktree. Run the complete historical suite from `/home/ww/paper3/cmc_damage_inference`, whose untracked 50GB authority, registered weights, and sibling prompt paths are required by legacy tests; also run Ruff, `git diff --check`, every checksum verifier, and formal/replay directory comparisons.
- [ ] Audit every prompt requirement against current files and command evidence; record PASS/FAIL and unresolved external-validation boundaries without weakening the objective.
- [ ] Commit `evidence: freeze MAVIS final evaluation and replay`, fast-forward `main`, verify local/remote history, push `main`, and verify uploaded commit/artifact paths using `git ls-tree -r origin/main`.
