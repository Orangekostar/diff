# Damage-to-Failure Response P1 Implementation Plan

> **Execution rule:** P0 is `P0_GO` at commit
> `93f5f3b4ef9a89e96dad72965ca3219d4820759c`. Implement every task test-first.
> No P2 feature model or training is authorized by this plan.

**Goal:** Decide whether the raw pre-peak CAI response contains a reproducible,
physically interpretable non-strength target that is not nearly determined by
ultimate CAI strength under strict leave-one-domain-out evaluation.

**P0 authority:** `results/damage_to_failure_response/p0_data_audit/summary.json`,
SHA-256 `9d44ead975119db2181a91efbf14b74165671a9d25b7b576d90f6e104757a633`.
The primary cohort is fixed at 276. Strain status is
`STRAIN_UNIT_UNRESOLVED`, so no gauge/JIS endpoint is permitted.

## Frozen P1 definitions

### Pre-peak stress-extension extraction

For each canonical primary specimen:

1. Reuse the P0 raw peak row, `max(abs(stress))`, with measured width and
   thickness. Use rows from acquisition start through that peak, inclusive.
2. Compute extension and stress offsets as the median of the first 50 samples
   (one second at 50 Hz). This rule is global and not specimen-selected.
3. Orient extension and stress independently by the sign of their offset-
   corrected peak value, so compression and crosshead travel are positive.
4. Keep finite pre-peak points with oriented extension in
   `[0, extension_peak]`. Sort by extension and replace exact duplicate
   extension values by the median stress at that extension. This resolves
   instrument quantization only; it does not smooth, clip, or enforce monotonic
   stress.
5. Fix the anchors `(0, 0)` and `(extension_peak, stress_peak)`. Interpolate
   linearly on the fixed grid `u = 0.00, 0.01, ..., 1.00`, where
   `extension = u * extension_peak` and `q(u) = stress / stress_peak`.
6. Fail extraction if the peak occurs before sample 50, either peak component
   is nonpositive/nonfinite, fewer than 50 unique extension positions remain,
   any grid value is nonfinite, or the two fixed anchors cannot be reproduced.

No post-peak row enters a primary target. No strain column enters extraction.

### Primary non-strength descriptors

The P1 family contains exactly three endpoints:

- `extension_peak_mm`: offset-corrected oriented crosshead extension at raw
  peak load.
- `slope_u20_u60_mpa_per_mm`: ordinary least-squares slope of interpolated
  physical stress against physical extension on grid points
  `0.20 <= u <= 0.60`.
- `normalized_prepeak_auc`: trapezoidal integral of `q(u)` over `[0, 1]`.

`q(0.5)` and full `q(u)` may be reported as diagnostics but are not additional
primary gate endpoints. The extraction range tolerances are fixed at
`0.001 mm`, `1.0 MPa/mm`, and `0.0001`, respectively. A descriptor range passes
only if it exceeds ten times its fixed tolerance.

### Strength-redundancy screen

Each endpoint uses six outer leave-one-domain-out folds. Fit only source-domain
rows in each fold. Two fixed estimators are reported:

- strength-only: standardized polynomial features of published CAI strength,
  degrees 1-3, followed by Ridge with `alpha=1e-6`;
- strength-plus-design: the same strength basis plus source-standardized known
  design/geometry variables and fixed one-hot laminate/impactor variables,
  followed by the same Ridge.

There is no hyperparameter search. All transforms are fit separately on the
five source domains. OOF predictions retain specimen and held-out-domain
identity. The gate uses the **strength-only** estimator; the design model is a
reported sensitivity.

An endpoint passes P1 only when all conditions hold:

1. valid coverage is at least 90% of 276;
2. every one of the six domains has valid rows;
3. pooled OOF R-squared is below 0.90 **or** at least four of six domains have
   absolute OOF prediction-vs-target Spearman below 0.95;
4. its numerical range exceeds ten times the fixed extraction tolerance;
5. extraction and artifact replay are byte-identical.

If at least one primary endpoint passes, status is `P1_GO`. Otherwise status is
`RESPONSE_BEYOND_STRENGTH_NO_GO` and P2/P3 stop.

### Representative-pair rule

For each specimen, choose the other specimen with minimum absolute published
CAI-strength difference; ties are resolved by canonical specimen ID. Deduplicate
unordered pairs, rank them by descending RMS distance between their 101-point
normalized curves, then by pair IDs, and retain the first 12. No specimen,
domain, condition, or visual result is manually selected.

## Task 1: Freeze P1 config and extraction contracts

**Files:**

- Create: `paper_v3/configs/damage_to_failure_response_p1.yaml`
- Create: `src/cmc_bbdm/damage_response/response_extraction.py`
- Create: `tests/test_damage_response_response_extraction.py`

1. Write failing tests for baseline offsets, peak orientation, duplicate-
   extension aggregation, 101-point anchors, the three formulas, and every QC
   failure above.
2. Run the new test file and record RED from the missing module.
3. Implement only the frozen extraction and descriptor functions.
4. Run the extraction tests and focused Ruff; expected GREEN.

## Task 2: Bind design metadata and leakage roles

**Files:**

- Extend: `src/cmc_bbdm/damage_response/sources.py`
- Create: `src/cmc_bbdm/damage_response/feature_views.py`
- Create: `tests/test_damage_response_feature_views.py`
- Create: `tests/test_damage_response_no_leakage.py`

1. Add a path-free, hash-bound design-metadata reader for the existing spatial
   manifest and cross-check its 276 primary identities/domains.
2. Define P1 target/reference/design roles. True raw traces, derived response,
   true peak strength as an inference input, post-CAI images, and privileged
   context in a deployable view must fail closed.
3. Test that target-domain sentinels cannot enter fit data or transformation
   state.

## Task 3: Implement strict source-only LODO redundancy evaluation

**Files:**

- Create: `src/cmc_bbdm/damage_response/nested_eval.py`
- Create: `tests/test_damage_response_nested_eval.py`

1. Write failing synthetic tests with a target-domain sentinel and assert six
   exact held-out folds, one OOF prediction per specimen/model/endpoint, source-
   only transformer statistics, deterministic predictions, pooled R-squared,
   and per-domain Spearman.
2. Implement the two fixed Ridge pipelines with no search API.
3. Verify target permutation cannot alter a fold's fitted source state.

## Task 4: Freeze gate and representative-pair selection

**Files:**

- Create: `src/cmc_bbdm/damage_response/p1_gate.py`
- Create: `src/cmc_bbdm/damage_response/representative_pairs.py`
- Create: `tests/test_damage_response_p1_gates.py`
- Create: `tests/test_damage_response_representative_pairs.py`

Test each coverage/domain/redundancy/range/replay reason independently, NO-GO
when all endpoints fail, GO when any endpoint passes, deterministic nearest-
strength pairing, tie handling, deduplication, and top-12 ordering.

## Task 5: Generalize deterministic stage artifacts

**Files:**

- Extend: `src/cmc_bbdm/damage_response/artifacts.py`
- Extend: `tests/test_damage_response_artifacts.py`

Add a generic exact-membership writer/replayer while preserving every P0 byte
and P0 replay test. P1 required payloads are:

```text
descriptor_table.csv
descriptor_qc.csv
domain_summary.csv
strength_redundancy_oof.csv
response_curve_manifest.csv
representative_pair_manifest.csv
summary.json
REPORT.md
artifact_manifest.json
CHECKSUMS.sha256
```

Two independently written packages from identical records must be byte-
identical. Missing, extra, changed, or symlinked members fail replay.

## Task 6: Orchestrate and execute P1

**Files:**

- Extend: `src/cmc_bbdm/damage_response/pipeline.py`
- Extend: `scripts/run_damage_response.py`
- Create: `tests/test_damage_response_p1_pipeline.py`
- Create: `results/damage_to_failure_response/p1_response_richness/*`
- Create: `artifacts/damage_to_failure_response/P1_RESPONSE_RICHNESS_DECISION.md`

1. Add `audit-p1` and `replay-p1`. Require the committed P0 package and its
   exact summary SHA before reading external raw sources.
2. Re-extract all 276 primary responses from official-SHA-verified files. Write
   the three descriptors, QC, all OOF predictions/metrics, 101-point compact
   curve rows, deterministic representative pairs, decision, and downstream
   authorization state.
3. Execute once, replay, run `sha256sum -c`, and independently recompute gate
   metrics from committed CSVs.
4. If status is not `P1_GO`, write P2-P5 as `NOT_RUN_NOT_AUTHORIZED`, create the
   final negative-result handoff, and stop. If status is `P1_GO`, write a
   separate P2 implementation plan before any P2 estimator code.

## Verification and commit

Run:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_damage_response_*.py
python -m ruff check src/cmc_bbdm/damage_response tests/test_damage_response_*.py scripts/run_damage_response.py
git diff --check
git diff --name-only 3951f71f28b6efdf8c74eea0fe274b2a78a9cd57 -- \
  results/p1_full_field_oracle results/p3_spatial_specificity results/p5_sparse_scan \
  results/mvd results/mva results/mavis results/mavis_science_closure \
  artifacts/aei_information_hierarchy
```

Expected: all focused tests and Ruff pass; frozen-path diff is empty; no raw
data, image, feature array, model weight, or cache is staged. Commit with:

```text
feat: add standards-grounded CAI response extraction
```
