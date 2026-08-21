# D8 Morphology-Preserving Diffusion Marginalization Design

**Frozen design date:** 2026-08-17  
**Target venue:** Composites Part B: Engineering  
**Method label:** D8-MPDM  
**Code authority:** the complete local project; the public Git repository receives only validated code and privacy-safe result exports.

## 1. Scientific question

D8 tests whether cross-domain CAI prediction improves when the stable low-/mid-frequency morphology of a measured C-scan is retained while fine-scale, acquisition-like variation is marginalized. Diffusion is used as a nuisance-residual proposal mechanism, not as a full-field reconstruction model and never receives CAI labels.

The immutable primary comparator is P1 `I_frozen`:

```text
measured internal C-scan
-> frozen ImageNet ResNet18 global embedding
-> fold-local PCA
-> Ridge(alpha=10)
-> CAI damaged-to-intact ratio
```

Its equal-domain MAE is `0.08963580465761432` on 276 specimens from six datasets. No surface feature enters D8.

## 2. Considered approaches

### Approach A: cross-fitted P6-residual pilot, then conditional escalation

Regenerate the eight deterministic P6 draws for every specimen with the checkpoint that held out that specimen's entire dataset. Band-limit the cross-fitted draw-minus-measurement residuals, apply morphology gates, and search augmentation/marginalization/regression choices using only nested inner-domain scores. Train a new residual diffusion model only when the frozen pilot trigger fires.

This is selected because it tests the new mechanism cheaply, uses existing out-of-domain diffusion proposals, and separates a useful invariance mechanism from a diffusion-specific claim.

### Approach B: train residual diffusion immediately

Train `p(R|S)` independently inside every inner and outer fit split before testing the pilot. This is scientifically clean but multiplies training cost across 30 inner folds and risks spending most compute on a mechanism that simple frequency controls may already explain.

This is rejected as the first action but retained as the registered escalation path.

### Approach C: non-diffusion spectral invariance only

Search Gaussian, phase-randomized, and empirical residual perturbations without diffusion. This is an essential control and may win, but it cannot by itself answer whether a learned diffusion residual prior adds value.

This is retained as B1-B4 controls, not selected as the proposed method.

## 3. Cohort, response, and split authority

- Cohort: the exact 276 P1 specimens.
- Domain order: `74t7kcdgkr`, `cgtnjyggtm`, `w68dtmpfyf`, `xcmzfsbd9t`, `yfxyg8jm46`, `ykhs7s2dck`.
- Domain sizes: 45, 49, 43, 59, 42, and 38.
- Response: published damaged-to-intact CAI strength ratio, unit `1`.
- Outer protocol: leave one complete dataset out; outer fit sizes are 231, 227, 233, 217, 234, and 238.
- Inner protocol: leave one of the five outer-fit datasets out. Candidate generation, preprocessing, PCA, regressor fitting, hyperparameter search, reranking, and ensemble weights are fit without the inner query domain whenever they learn from data.
- Inferential unit: held-out dataset. Variants, seeds, residual draws, and specimens are never treated as independent confirmatory units.

Global source preflight may hash all registered files, but each search component receives only a loader-issued five-domain view. No outer image array, response value, embedding, residual, or derived statistic enters the search object graph. The selected configuration record is serialized and hash-frozen before the outer evaluation view is issued. Evaluation is one-way: no selected configuration can be changed after an outer result exists.

## 4. Baseline reconstruction gate

Before any D8 search, the implementation must reproduce all 276 P1 `I_frozen` predictions, targets, specimen identities, domain MAEs, equal-domain MAE, and fold PCA dimensions at absolute tolerance `1e-12`.

The registered outer PCA dimensions are:

```text
74t7kcdgkr: 8
cgtnjyggtm: 32
w68dtmpfyf: 8
xcmzfsbd9t: 8
yfxyg8jm46: 8
ykhs7s2dck: 8
```

The six baseline domain MAEs are:

```text
74t7kcdgkr: 0.052003607090763716
cgtnjyggtm: 0.12486849958988917
w68dtmpfyf: 0.09660573834513045
xcmzfsbd9t: 0.07511512755876239
yfxyg8jm46: 0.12743074387350148
ykhs7s2dck: 0.06179111148763877
```

Search is blocked unless this gate passes.

## 5. P6 residual authority

The formal P6 package contains six fold-local diffusion checkpoints, each trained for 120 epochs on the five non-held-out datasets. The package saves posterior mean and variance for all 276 specimens but not the individual eight fields.

D8 therefore regenerates the eight draws using the P6 checkpoint, specimen IDs, DDIM schedule, and deterministic seed derivation. For every specimen, the checkpoint held out its complete dataset. The regenerated mean and variance must match `uncertainty_source_data.npz` at the registered numerical tolerance before any residual is issued.

The residual bank stores, per specimen and draw:

```text
Delta = D_p6_draw_64 - D_measured_64
```

It also binds specimen ID, dataset ID, P6 checkpoint scientific digest, source-field hash, draw seed, array hash, and cross-fitted-domain status. A residual generated by a checkpoint that trained on its specimen's domain is invalid.

## 6. Decomposition families

All decomposition operates on separately resized 64x64 RGB computational frames in `[-1,1]`. It is not a physical common pixel grid.

### Gaussian

- high residual: `Delta - G_sigma(Delta)`;
- mid residual: `G_sigma_low(Delta) - G_sigma_high(Delta)`;
- sigma is log-uniform on `[0.5, 8.0]` with `sigma_low < sigma_high` for mid bands.

### Fourier

- normalized radial cutoff in `[0.04, 0.50]` of Nyquist;
- high, mid, and mid+high bands;
- raised-cosine transition width in `[0.01, 0.10]` to avoid ringing-driven selection.

### Wavelet

- `haar`, `db2`, `db4`, and `sym4`;
- levels 1, 2, or 3 when supported by the 64x64 frame;
- retained residual bands: high only, mid only, or mid+high;
- reconstruction shape, finite values, and energy accounting must be exact within `1e-6`.

`PyWavelets==1.8.0` is the registered wavelet implementation. `Optuna==4.9.0` is the registered TPE search implementation.

## 7. Variant construction and morphology gate

Variants are constructed as:

```text
D_tilde = clip(D + alpha * R_band, -1, 1)
```

with `alpha` sampled continuously on `[-0.5, 1.0]`. The original registered C-scan footprint rule and physical calibration are rerun on the original and variant images. A variant is accepted only when all selected inner-frozen conditions pass:

The residual arithmetic remains on the separately resized 64x64 frame. Before feature extraction, the accepted 64x64 delta is bicubically lifted channel-wise to the specimen's native registered frame and added to the native normalized C-scan. The `alpha=0` path must reproduce the native P1 encoder input byte-for-byte. The physical footprint and radial-profile gates are evaluated on this actual native encoder image; the fixed Gaussian low-frequency correlation is evaluated on the paired 64x64 fields. The native image must independently resize back to the registered 64x64 source before any variant is accepted.

- absolute relative area, width, and height deviations each at most one of `2.5%, 5%, 7.5%, 10%`;
- centroid displacement at most one of `0.5, 1.0, 2.0` mm;
- Gaussian low-frequency Pearson correlation at least one of `0.95, 0.97, 0.98, 0.99`;
- radial-profile Spearman correlation at least one of `0.90, 0.95, 0.98`;
- finite RGB values and unchanged frame shape.

The gate never reads CAI. Candidate generation tries residual draws in deterministic order. If fewer than the requested `K` pass after 32 proposals, the original field fills the remaining slots and the fallback is recorded. A candidate is ineligible when overall acceptance is below 0.80 or any inner query domain acceptance is below 0.60.

## 8. Baseline and ablation matrix

| ID | Variant | Mechanism tested |
| --- | --- | --- |
| B0 | raw `I_frozen` | immutable comparator |
| B1 | morphology component only | low-/mid-frequency sufficiency |
| B2 | variance-matched Gaussian residual | generic noise invariance |
| B3 | spectrum-matched phase randomization | spectral but non-learned nuisance |
| B4 | fit-domain empirical residual bootstrap | empirical nuisance without diffusion prior |
| B5 | cross-fitted P6 diffusion residual augmentation | diffusion proposal value |
| B6 | B5 plus prediction/feature consistency | consistency contribution |
| B7 | B5 plus test-time marginalization | marginalization contribution |
| B8 | selected diffusion augmentation, consistency, marginalization, and optional ensemble | proposed pipeline |

B4 donors come only from the current inner-fit domains during inner scoring and only from the five outer-fit domains during final fitting. The query specimen and query domain cannot donate a residual.

## 9. Feature and prediction marginalization

Every candidate records `marginalization_stage` as either `feature` or
`prediction`. In the feature stage, the registered mean, median, trimmed, or
mean-plus-log-variance operator maps the K variant embeddings to one feature
vector before fold-local PCA/regression; prediction aggregation is fixed to a
single mean prediction. In the prediction stage, the training representation is
the mean variant feature, the fitted fold-local model is applied separately to
all K query variants, and the registered mean, median, trimmed, or
morphology-weighted operator returns one prediction per specimen. This keeps
feature dimensions aligned and makes both prompt-required marginalization
families executable rather than redundant search labels.

The frozen ResNet18 weights and preprocessing remain unchanged. Searchable feature sources are global average pooling, pooled `layer3`, or concatenated pooled `layer1`--`layer4` features. Every PCA and feature scaler is fit within the relevant inner/outer fit split.

For `K_train in {1,2,4,8}` and `K_test in {1,2,4,8,16}`, search:

- mean, coordinate median, or 10% trimmed feature;
- mean feature plus log diagonal variance after fold-local variance-feature selection;
- mean, median, 10% trimmed, or morphology-distance-weighted prediction;
- morphology weight `exp(-beta*d_morph)` with `beta` log-uniform on `[0.1,100]`.

If variants are used as separate training rows, every physical specimen has total weight one, regardless of `K_train`.

## 10. Regressor and consistency search

Candidate regressors are Ridge, ElasticNet, PLS, Huber, kernel Ridge, RBF-SVR, HistGradientBoosting, and a shallow MLP. Candidate-specific ranges are bounded for the 276-specimen cohort; any solver failure or nonfinite prediction marks the trial failed rather than silently changing method.

Consistency choices are none, prediction variance penalty, PCA-subspace feature variance penalty, or pairwise ranking penalty. Penalties are computed only from fit-domain variants. Search never uses outer targets or outer prediction dispersion.

## 11. Search and selection

Each outer fold has an independent Optuna TPE study with seed derived from `20260820` and the outer-domain ID.

The inner objective is fixed as:

```text
J = mean_inner_domain_MAE
  + 0.25 * worst_inner_domain_MAE
  + 0.10 * SD_inner_domain_MAE
```

Selection stages:

1. Twelve forced warm-start trials cover B0-B5 and all three decomposition families.
2. Sixty TPE trials search the pilot space.
3. The best twelve eligible configurations are rerun with seeds `20260820`, `20260821`, and `20260822`.
4. The best four are evaluated at `K_test` 8 and 16 and checked for complementarity.
5. A nonnegative inner-OOF prediction ensemble is allowed only when it improves `J` by at least `1e-4`; weights sum to one and are learned without the outer domain.
6. Ranking is by repeated-seed `J`, mean MAE, worst-domain MAE, model complexity, then canonical configuration hash.

The deterministic complexity tie-break is lexicographic and favors smaller `K_train`, smaller `K_test`, global before layer-3 before multi-layer features, mean before median before trimmed before mean-plus-variance feature aggregation, mean before median before trimmed before morphology-weighted prediction aggregation, no consistency before prediction/feature/ranking consistency, lower PCA dimension, and finally the registered regressor order Ridge, Huber, ElasticNet, PLS, kernel Ridge, SVR, HistGradientBoosting, shallow MLP. It is consulted only after repeated-seed `J`, mean MAE, and worst-domain MAE tie exactly.

Ensemble acceptance is itself cross-fitted across the five inner domains: for each query domain, simplex-constrained least-squares weights are fit on OOF rows from the other four domains and used only to predict that query domain. The concatenated cross-fitted predictions define the ensemble `J` and the `1e-4` acceptance gate. Only after acceptance are prospective deployment weights refit on all five-domain inner-OOF rows; both the five cross-fitted weight vectors and the prospective weight vector are serialized.

All 72 initial trials and every rerun are logged, including failures.

## 12. Residual-diffusion escalation

Training a new `p_theta(R|S)` is authorized when either condition is met before outer evaluation:

1. a P6-residual diffusion candidate improves mean inner-domain MAE by at least 1% over B0 in at least three of five inner domains for at least three outer-fold studies; or
2. P6 residuals are demonstrably mismatched, defined as at least 50% of residual energy below the selected nuisance cutoff or morphology-gate acceptance below 50% at `alpha=0.1` in at least three outer-fold studies.

The decision priority is frozen as `TRAIN_RESIDUAL_DIFFUSION`, then
`FREEZE_PILOT_FOR_OUTER_EVALUATION`, then
`CLOSE_DIFFUSION_SPECIFIC_ROUTE`. If neither training trigger fires, the pilot
is frozen for prospective outer evaluation only when at least three of the six
outer-fold studies both (i) improve the best eligible diffusion-candidate inner
objective over B0 by at least `1e-4` and (ii) retain at least `0.05` prospective
weight on B5--B8 in the frozen selected pipeline. Otherwise the
diffusion-specific route closes; non-diffusion controls may still be reported.
These thresholds are fixed before any D8 pilot result is generated.

The new model is a fold-local 64x64 conditional residual U-Net with base channels 32 or 64, limited attention, and DDPM/DDIM or EDM-style schedules. The search covers epsilon, v, or direct residual prediction and a frozen combination of diffusion, spectral, and low-frequency morphology losses. It is trained separately for each inner fit split used for selection and once on all five outer-fit domains after selection. It never receives response values.

## 13. Formal evaluation and statuses

The selected per-outer configuration is hash-frozen before an outer evaluation view is issued. The formal evaluator then produces one prediction per specimen and is never rerun for selection.

Primary statuses use the prompt's frozen thresholds:

- `MINIMUM_IMPROVEMENT`: equal-domain MAE `< 0.08963580465761432`;
- `PRIMARY_POSITIVE`: MAE `<= 0.085154` and at least 4/6 domains improve;
- `STRONG_POSITIVE`: MAE `<= 0.082465` and at least 5/6 domains improve;
- `STRETCH_POSITIVE`: MAE `< 0.080000` and all six domains improve.

`ROBUSTNESS_ONLY` requires all of: aggregate MAE no worse than B0, at least 5/6 domains improve, worst-domain MAE decreases by at least 5%, and the SD of six domain MAEs decreases by at least 10%.

Bootstrap uses 100,000 common PCG64 six-domain resamples. It reports ordinary and simultaneous intervals for B0-minus-D8 and best-non-diffusion-control-minus-D8. Diffusion-specific promotion additionally requires D8 to beat the best non-diffusion control in at least 4/6 domains with a strictly positive simultaneous lower bound. Otherwise the conclusion is augmentation/invariance value, not diffusion-specific value.

If full internal-only D8 reaches `PRIMARY_POSITIVE`, the exact selected mechanism is replicated at the P5 25% sparse internal-only setting without reopening the full-field search.

## 14. Reproducibility and publication

Search outputs live under `results/d8_search`; formal outputs live under `results/d8_final`; independent replay lives under `results/replay/d8_final`. Every package binds configuration, data/splits, source code, dependencies, model weights, residual banks, trial records, selected configurations, predictions, metrics, bootstrap draws, runtime, and output hashes.

Formal and replay scientific outputs must be byte-identical after excluding registered timing fields. Publication uses lock, owner-marked staging, validation-before-commit, rollback, and crash recovery. Raw images, personal absolute paths, and private source documents are excluded from the public export.

No result is claimed in this design. The outer-domain result remains unknown until the frozen formal evaluation runs.
