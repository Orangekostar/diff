# Damage-to-Failure Response P2 Implementation Plan

> **Execution rule:** P1 is `P1_GO` at commit
> `69b67c72e7462cfd5e27f55f875e25a3c78a45e7`. Implement every task
> test-first. This plan authorizes only low-capacity P2 baselines. It does not
> authorize any P3 curve estimator, neural model, manuscript rewrite, or new
> experiment.

**Goal:** Test whether pre-CAI information predicts a preregistered
non-strength CAI response endpoint across six held-out experimental domains,
and whether frozen full-field C-scan information adds engineering-relevant
value beyond measured scalar damage size.

**P1 authority:**
`results/damage_to_failure_response/p1_response_richness/summary.json`,
SHA-256 `37da95962395a0915f586820ab03f06d8d859856e8637d975bc302b1d555ebc7`.
The cohort is fixed at 276 specimens in domain counts `45, 49, 43, 59, 42,
38`. The three P2 targets are fixed by P1:

- `extension_peak_mm`;
- `slope_u20_u60_mpa_per_mm`;
- `normalized_prepeak_auc`.

No gauge-strain or JIS endpoint is permitted because strain units remain
unresolved.

## Frozen P2 authority

P2 must not import the absent compact-repository `cpb_v3.data` stack. It will
derive a compact, target-free feature authority from direct hash-bound sources
and cross-check it against the P1 identities and domains.

| Authority | Registered SHA-256 | Use |
|---|---|---|
| `results/aei_selective_invariance/a2_paired_features/paired_features.npz` | `f2a69f0da75e20880202d7fc4a6a92f979978406ec21f9d83e4bc8db07fb72a8` | Existing 512-d frozen ResNet18 bank; use the `FULL` view only |
| `paper_v3/configs/aei_multiview_regression.yaml` | `8b46986c23ccd87d5e79f2a6f94f034a90d10e91ec21d39c40a1a58e3b41a357` | Embedding semantics and source registration |
| external `results/public_recoverability/feature_cache.npz` | `6d581c77e0db18782406bd55dd2c0716d048c16c64c8df4423d5b331c84aeb52` | Existing 21 surface-profile statistics |
| external `results/cpb_cfrp_revision/physical_descriptors.csv` | `2f2c3b3a16cbe1b48364b76ad10999a91c01de000b81525ba56c8ec90399e078` | Projected area, damage height, and damage width |
| external `paper_v2/data/provenance/specimens.csv` | `84617cb12012ac9acc1b756b59161738a2a0a1089bd3eb432b7d01a066e65f9f` | Specimen-level provenance cross-check |
| official v3 LVI workbook | `e6d98c968f57ac5748e104dc1da5e112114d25d77c03c7222ba3a0d93ac23cf1` | F5-only total impact energy and impactor authority |
| external `src/cmc_bbdm/cpb_v3/data.py` | `4d294d3b047fc32540adb499402b7171700bfec152dac71263d10f92808b8e03` | Read-only equivalence reference; not a runtime import |
| external `src/cmc_bbdm/cpb_v3/config.py` | `be07e8313127bb76b31b2b498a131fda151268e7c0d86a0ced36d65f4676f2e3` | Read-only feature-name reference; not a runtime import |
| external `src/cmc_bbdm/hasebe_cai.py` | `e788f77e01fd534b14168f06d3177ba81c24abc3fe53ca8ab7feebeb0d0dbf07` | Read-only workbook semantic reference; not a runtime import |

The derived authority contains exactly the P1 cohort and only feature values,
identities, domains, source hashes, and feature provenance. It must contain no
response target, raw trace, post-CAI image, private absolute path, or model
prediction. Surface and scalar values must be elementwise identical to the
registered source arrays after specimen-ID alignment. Embedding IDs and domains
must exactly equal the P1 cohort.

Privileged total impact energy and impactor shape are recovered only for F5.
The LVI workbook's registered E/F semantic reversal must be preserved: original
column E is total energy and original column F is energy per thickness. Values
must be independently cross-checked against the historical metadata authority.

## Frozen feature views

- **F0, design-only:** layup family, ply count, width, and thickness.
- **F1, surface:** F0 plus the 21 registered surface profile statistics.
- **F2, scalar damage:** F0 plus projected damage area, damage height, and
  damage width.
- **F3, full C-scan spatial:** F0 plus the registered `FULL` 512-d embedding.
- **F4, multimodal:** F0 plus the 21 surface statistics and full embedding.
- **F5, privileged sensitivity:** F4 plus true total impact energy and impactor
  shape.

F5 is never a primary deployable view. True impact context, response targets,
published/derived CAI strength, raw CAI traces, post-CAI images, target-domain
statistics, and target-domain model-selection scores are forbidden from F0-F4.

## Frozen nested LODO protocol

For each of the three P1 endpoints and each F0-F5 view:

1. Hold out exactly one complete domain as the outer query fold.
2. On the remaining five domains, select hyperparameters with inner
   leave-one-source-domain-out evaluation and equal-domain raw-unit MAE.
3. Fit all numeric imputation, scaling, categorical one-hot state, and PCA only
   on the current fit domains. Fixed category registries may define columns,
   but no target-domain value may affect fitted state.
4. Use a Ridge estimator only in this first round. The fixed alpha grid is
   `[0.1, 1.0, 10.0, 100.0]`.
5. F0-F2 select alpha only. F3-F5 jointly select alpha and PCA dimension from
   `[8, 16, 32]`, with PCA fitted to the embedding block only.
6. At score ties within `1e-12`, choose the lower PCA dimension, then the
   stronger regularization (larger alpha). This order is fixed and reported.
7. Refit the selected pipeline on all five outer-source domains and query only
   the held-out domain. Retain one OOF prediction per specimen, target, and
   view, plus all inner selection rows and fold-state provenance.

Targets are fit in raw units. In addition to raw errors, report the absolute
query error divided by the target standard deviation fitted on the five outer
source domains. This standardized value is a sensitivity only and does not
enter the primary gate.

## Frozen metrics and inference

For every endpoint and view, report:

- equal-domain MAE as the primary absolute metric;
- each held-out-domain MAE;
- pooled RMSE and pooled R-squared as secondary metrics;
- source-fold-standardized absolute error sensitivity;
- deterministic OOF coverage and identity checks.

Primary contrasts are F3 minus F2 and F4 minus F2 for each of three endpoints,
giving six familywise-controlled tests. Improvement is represented as
`abs_error_F2 - abs_error_candidate`, so positive values favor the candidate.

Use a paired synchronized within-domain specimen bootstrap:

- NumPy `PCG64`, seed `20260830`;
- `100000` replicates;
- within each domain, sample that domain's specimen indices with replacement;
- reuse the exact sampled indices for every view and endpoint;
- compute each replicate's mean contrast within each domain, then average the
  six domain means;
- report ordinary 95% percentiles and familywise two-sided 95% percentiles at
  `0.025 / 6` and `1 - 0.025 / 6`;
- retain improved-domain count from the observed six domain MAE contrasts.

F4 versus F3 is a prespecified secondary complementarity contrast and never
substitutes for the F2 primary reference.

## Frozen P2 decision

An endpoint/view pair passes only when F3 or F4 versus F2 satisfies all three:

1. observed relative equal-domain MAE improvement is at least `0.10`;
2. at least four of six held-out domains improve;
3. the familywise bootstrap lower bound for raw MAE improvement is strictly
   greater than zero.

P2 is `P2_GO` when at least one non-strength endpoint/view pair passes. If all
three non-strength endpoints fail, the decision is
`MACK_EXTENSION_NO_GO`; P3-P5 remain `NOT_RUN_NOT_AUTHORIZED`. A positive F3
with negative F4 still authorizes the spatial route but does not support a
multimodal-superiority claim.

## Task 1: Freeze config and derived feature authority

**Files:**

- Create: `paper_v3/configs/damage_to_failure_response_p2.yaml`
- Create: `src/cmc_bbdm/damage_response/p2_features.py`
- Create: `tests/test_damage_response_p2_features.py`

1. Write failing tests for exact source hashes, 276-ID/domain alignment,
   `FULL`-only embedding selection, surface/scalar elementwise equivalence,
   F5 workbook E/F semantics, finite shapes, target-free serialization, and
   rejection of private paths or unexpected members.
2. Run the new test file and record RED from the missing module/config.
3. Implement only the strict source readers, validation, and deterministic
   derived authority serialization.
4. Run the new tests and focused Ruff; expected GREEN.

## Task 2: Enforce view and fold-local preprocessing contracts

**Files:**

- Create: `src/cmc_bbdm/damage_response/p2_views.py`
- Create: `tests/test_damage_response_p2_views.py`
- Extend: `tests/test_damage_response_no_leakage.py`

Test exact F0-F5 membership and ordering, F5 privilege labeling, target/input
role rejection, source-only imputation/scaling/one-hot/PCA state, embedding-only
PCA, fixed output dimensions, and target-domain sentinel invariance. Implement
no estimator selection in this task.

## Task 3: Implement strict nested LODO Ridge evaluation

**Files:**

- Create: `src/cmc_bbdm/damage_response/p2_evaluation.py`
- Create: `tests/test_damage_response_p2_evaluation.py`

1. Write synthetic failing tests for six exact outer folds, five inner source
   folds, fixed grids, deterministic tie resolution, target permutation
   isolation, one OOF row per specimen/endpoint/view, source-state hashes, raw
   and standardized errors, and metric formulas.
2. Implement the fixed Ridge/PCA protocol without a general search API.
3. Prove that changing any held-out-domain feature or target cannot alter a
   fold's fitted state or selected hyperparameters.

## Task 4: Implement synchronized inference and P2 gate

**Files:**

- Create: `src/cmc_bbdm/damage_response/p2_statistics.py`
- Create: `src/cmc_bbdm/damage_response/p2_gate.py`
- Create: `tests/test_damage_response_p2_statistics.py`
- Create: `tests/test_damage_response_p2_gates.py`

Test synchronized within-domain resampling, equal-domain aggregation, exact
seed/replicate count, ordinary and six-test familywise quantiles, contrast
sign, domain direction count, 10% boundary behavior, strict positive lower
bound, F3-only authorization wording, and the all-fail
`MACK_EXTENSION_NO_GO` path.

## Task 5: Generalize deterministic P2 artifacts

**Files:**

- Extend: `src/cmc_bbdm/damage_response/artifacts.py`
- Extend: `tests/test_damage_response_artifacts.py`

Preserve all existing P0 and P1 package bytes and replay behavior. The exact P2
payload set is:

```text
config.yaml
feature_authority.csv
feature_provenance.json
inner_selection.csv
oof_predictions.csv
aggregate_metrics.csv
domain_metrics.csv
bootstrap_contrasts.csv
summary.json
REPORT.md
artifact_manifest.json
CHECKSUMS.sha256
```

Two independently written packages from identical records must be byte-
identical. Missing, extra, changed, nonregular, or symlinked members fail
replay. Feature arrays, model weights, caches, and private external paths are
not package members.

## Task 6: Orchestrate and execute P2

**Files:**

- Extend: `src/cmc_bbdm/damage_response/pipeline.py`
- Extend: `scripts/run_damage_response.py`
- Create: `tests/test_damage_response_p2_pipeline.py`
- Create: `results/damage_to_failure_response/p2_response_baselines/*`
- Create: `artifacts/damage_to_failure_response/P2_DAMAGE_TO_RESPONSE_DECISION.md`

1. Add `audit-p2` and `replay-p2`. Require exact committed P0 and P1 package
   membership, replay, and registered summary hashes before reading any P2
   feature source.
2. Verify every registered source hash, derive the compact feature authority
   twice, execute all 108 outer model fits plus inner selections, and write the
   complete deterministic package and decision.
3. Execute once, replay, run `sha256sum -c`, independently recompute metrics,
   contrasts, intervals, and decision from committed CSVs, and compare a second
   independently written package byte-for-byte.
4. If status is not `P2_GO`, mark P3-P5 `NOT_RUN_NOT_AUTHORIZED`, write the
   negative-result handoff, and stop scientific execution. If status is
   `P2_GO`, create `PAPER_POSITIONING_DRAFT.md` and `CLAIM_EVIDENCE_MAP.md`, then
   write and commit a separate P3 implementation plan before any curve-model
   code.

## Verification and commit sequence

Run after every task's focused tests, then run the complete P0-P2 set:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_damage_response_*.py
python -m ruff check src/cmc_bbdm/damage_response tests/test_damage_response_*.py scripts/run_damage_response.py
git diff --check
git diff --name-only 3951f71f28b6efdf8c74eea0fe274b2a78a9cd57 -- \
  results/p1_full_field_oracle results/p3_spatial_specificity results/p5_sparse_scan \
  results/mvd results/mva results/mavis results/mavis_science_closure \
  artifacts/aei_information_hierarchy
```

Expected: all focused tests and Ruff pass; frozen old-science path diff is
empty; no raw data, image, feature array, model weight, cache, or private
absolute path is staged. Commit this plan before Task 1 implementation with:

```text
docs: plan damage-response P2 baselines
```
