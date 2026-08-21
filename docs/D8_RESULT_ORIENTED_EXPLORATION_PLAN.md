# D8 Result-Oriented Exploration Plan

**Status:** frozen before formal D8 training or outer-domain evaluation  
**Date:** 2026-08-17  
**Design authority:** `docs/superpowers/specs/2026-08-17-d8-morphology-preserving-diffusion-marginalization-design.md`  
**Primary target:** beat internal-only P1 `I_frozen` MAE `0.08963580465761432` without outer-domain leakage.

## 1. Baseline reconstruction

Rebuild the exact P1 internal-only path: measured C-scan, frozen ImageNet ResNet18 global embedding, fold-local PCA, and Ridge alpha 10. The gate requires exact 276-specimen identities, targets, predictions, six domain MAEs, equal-domain MAE, and outer PCA dimensions `(8,32,8,8,8,8)` within `1e-12`.

No D8 trial starts until this reproduction passes. The baseline artifact manifest, ResNet weight SHA-256, source image hashes, and split hashes are copied into the D8 search authority.

## 2. Exact data split

The fixed domain order and counts are:

| Domain | Specimens | Outer-fit count when held out |
| --- | ---: | ---: |
| `74t7kcdgkr` | 45 | 231 |
| `cgtnjyggtm` | 49 | 227 |
| `w68dtmpfyf` | 43 | 233 |
| `xcmzfsbd9t` | 59 | 217 |
| `yfxyg8jm46` | 42 | 234 |
| `ykhs7s2dck` | 38 | 238 |

For each outer fold, the other five datasets form the exploration pool. Inner LODO uses four datasets for data-dependent fitting and the fifth only for scoring. Donor residuals, preprocessing, PCA, regressors, consistency weights, and ensemble weights obey this split. Global preflight may hash every registered source, but search receives only a loader-issued five-domain view. The outer evaluation view is issued only after the selected configuration is serialized and frozen.

## 3. Available P6 checkpoint audit

The formal P6 package is present and validated at `results/cpb_spatial/p6_diffusion_reconstruction`:

- six diffusion checkpoints, each 9.7 MiB, trained 120 epochs;
- six deterministic checkpoints, each 9.7 MiB, retained only as a control;
- diffusion checkpoints are domain-held-out and contain training IDs/domains, state SHA-256, training-data SHA-256, and scientific digest;
- `uncertainty_source_data.npz` contains posterior mean and variance for all 276 specimens but not individual draws;
- eight individual P6 draws are reproducibly regenerable through `sample_diffusion_fields` using the registered checkpoint and specimen/draw seeds;
- regenerated mean and variance must match the saved arrays before residual use.

The P6 package is 157 MiB; production and replay copies are byte-identical. D8 uses only cross-fitted draws: each specimen is processed by the P6 checkpoint that held out its complete dataset.

## 4. Decomposition candidates

| Family | Search variables | Candidate bands |
| --- | --- | --- |
| Gaussian | sigma `[0.5,8.0]`, log scale; ordered low/high sigma for mid band | high, mid, mid+high |
| Fourier | radial cutoff `[0.04,0.50]` Nyquist; transition `[0.01,0.10]` | high, mid, mid+high |
| Wavelet | `haar/db2/db4/sym4`; level `1/2/3` | high, mid, mid+high |

Every decomposition records source/residual energy, reconstruction error, finite/range checks, and state hash. Wavelet/Fourier operations are applied channel-wise on the 64x64 computational frame. Accepted deltas are lifted channel-wise to each native registered C-scan before the frozen encoder; `alpha=0` must recover the exact P1 input bytes. Physical footprint and radial gates run on that native encoder image, while the fixed low-frequency correlation runs on the paired 64x64 fields.

## 5. Residual candidates

| ID | Residual source | Role |
| --- | --- | --- |
| B0 | none/raw | primary baseline |
| B1 | none/low-pass morphology | morphology-only control |
| B2 | variance-matched Gaussian noise | generic stochastic control |
| B3 | amplitude-spectrum-preserving phase randomization | spectral control |
| B4 | fit-domain empirical P6 residual donor | empirical-prior control |
| B5 | specimen-specific cross-fitted P6 residual | diffusion-only augmentation |
| B6 | B5 plus consistency | consistency ablation |
| B7 | B5 plus test-time marginalization | marginalization ablation |
| B8 | selected diffusion pipeline and optional inner-OOF ensemble | proposed method |

Variants use `clip(D + alpha*R, -1, 1)`, `alpha in [-0.5,1.0]`, and the frozen morphology gate. Query-domain residuals never enter an empirical donor bank.

## 6. Optuna search space

`Optuna==4.9.0` TPE runs independently per outer fold with 12 forced warm starts and 60 adaptive trials.

Search dimensions include:

- residual/control family B0-B7;
- Gaussian/Fourier/wavelet parameters above;
- alpha, including negative symmetric perturbations;
- area/width/height tolerance `2.5/5/7.5/10%`;
- centroid tolerance `0.5/1/2 mm`;
- low-frequency correlation `0.95/0.97/0.98/0.99`;
- radial-profile correlation `0.90/0.95/0.98`;
- `K_train=1/2/4/8`, `K_test=1/2/4/8/16`;
- ResNet global/intermediate/multi-layer pooled feature source;
- explicit feature-level versus prediction-level marginalization stage;
- mean/median/trimmed/mean-plus-variance feature aggregation;
- mean/median/trimmed/morphology-weighted prediction aggregation;
- consistency type and bounded penalty weight;
- PCA dimension from `4/8/16/32/64`, limited by fit rank;
- Ridge, ElasticNet, PLS, Huber, kernel Ridge, SVR, HistGradientBoosting, or shallow MLP and their bounded parameters.

Every suggestion is serialized before evaluation. Invalid combinations are pruned with an explicit reason; exceptions are logged and never converted to a different model silently.

## 7. Objective function

For the five inner query domains:

```text
J = mean(domain_MAE) + 0.25*max(domain_MAE) + 0.10*SD(domain_MAE)
```

Candidate eligibility additionally requires overall morphology acceptance at least 0.80 and per-inner-domain acceptance at least 0.60. Trial ordering is `J`, mean MAE, worst MAE, complexity, then canonical configuration hash.

The search objective does not include any outer-domain value. Specimens, variants, draws, and seeds are not independent scoring units.

## 8. Model candidates

Frozen ResNet18 remains the default encoder. Global, pooled `layer3`, and concatenated pooled `layer1`--`layer4` features may compete, but no encoder weight is fine-tuned.

Regressors are bounded to the small cohort:

- Ridge and kernel Ridge;
- ElasticNet and Huber;
- PLS;
- RBF-SVR;
- HistGradientBoosting;
- one hidden-layer MLP with width no larger than 64 and strong regularization.

All scaling, PCA, hyperparameters, early stopping, and calibration occur within the relevant fit domains.

## 9. Ensemble strategy

After three-seed reranking, the best four configurations are checked using their inner-OOF predictions. Ranking first uses repeated-seed `J`, mean MAE, and worst-domain MAE; exact ties use the preregistered lexicographic complexity key and then configuration hash. A nonnegative weight vector summing to one is fit by constrained least squares. Ensemble acceptance uses five domain-cross-fitted weight vectors, each fit on the other four domains, and requires a cross-fitted `J` improvement of at least `1e-4` over the best member. Prospective weights are then refit on all five-domain OOF rows, and every cross-fitted and prospective weight is serialized.

Feature-stage candidates aggregate K embeddings before PCA/regression.
Prediction-stage candidates fit on the mean variant feature and apply that
fold-local model to each query variant before the registered prediction
aggregation. `mean_variance` is therefore restricted to feature-stage
candidates, while morphology-weighted prediction is restricted to
prediction-stage candidates.

An ensemble with total diffusion-member weight below 0.05 cannot support a diffusion-specific claim. Outer predictions never train or change ensemble weights.

## 10. Compute budget

Available compute is three NVIDIA A40 46 GiB GPUs, 64 CPU threads, and 251 GiB RAM.

Planned maximum before escalation:

| Stage | Budget |
| --- | --- |
| Baseline/P6 audit | one full baseline replay plus 6-fold, 8-draw P6 regeneration |
| Pilot search | 72 trials x 6 outer folds; one GPU worker per two folds |
| Reranking | top 12 x 3 seeds x 6 folds |
| Final candidate check | top 4 x `K_test` 8/16 x 6 folds |
| Bootstrap | 100,000 shared six-domain resamples |

GPU feature extraction is cached by source-field hash, residual-bank hash, configuration hash, and encoder-layer hash. CPU transforms use bounded worker pools. Search stops early only for invalidity, not because a favorable result appears.

If residual diffusion escalation fires, the additional cap is 30 inner-fold compact models plus six outer-fit models, with base channels 32/64 and no foundation-scale model.

## 11. Pilot stage

1. Validate P1/P6/P5 authorities.
2. Reproduce `I_frozen=0.08963580465761432` exactly.
3. Regenerate cross-fitted P6 draws and validate posterior mean/variance.
4. Build Gaussian/Fourier/wavelet residual banks.
5. Run B0-B5 warm starts and morphology audits.
6. Execute 60 TPE trials per outer fold.
7. Publish `results/d8_search/trial_index.csv`, `study.db`, residual-bank manifest, best inner configurations, and pilot report.

Residual-diffusion training fires only under the two objective escalation conditions frozen in the design. A negative pilot that does not trigger escalation closes the diffusion-specific route but still reports non-diffusion controls.

Decision priority is `TRAIN_RESIDUAL_DIFFUSION` ->
`FREEZE_PILOT_FOR_OUTER_EVALUATION` ->
`CLOSE_DIFFUSION_SPECIFIC_ROUTE`. When neither training trigger fires, freezing
the pilot requires at least three outer studies in which the best eligible
diffusion trial improves B0's inner objective by at least `1e-4` and the frozen
prospective selection assigns at least `0.05` total weight to B5--B8. Fewer
than three such studies closes the diffusion-specific route. This rule is
frozen before pilot execution.

## 12. Formal stage

For each outer fold:

1. rerank the top twelve using three seeds;
2. evaluate the best four at larger K and test ensemble feasibility;
3. serialize the selected configuration and all fit-domain evidence;
4. hash-freeze the configuration before decoding the outer data;
5. fit once on all five outer-fit domains;
6. produce one untouched outer prediction per specimen;
7. prohibit any return from outer results to search.

After all folds, compute equal-domain MAE, six domain MAEs, worst-domain MAE, domain SD, domain-level bootstrap, and exact statuses. Sparse 25% replication runs only after full internal-only `PRIMARY_POSITIVE`.

## 13. Failure modes

- Baseline mismatch: abort search.
- P6 draw/mean/variance mismatch: invalidate the residual bank.
- Held-out identity in any learned fit/donor/PCA/regressor state: abort that package.
- Low morphology acceptance: candidate ineligible; do not relax after results.
- Diffusion ties or loses to B2-B4: report augmentation/invariance, not diffusion-specific value.
- Aggregate gain below 5% but robustness thresholds pass: report `ROBUSTNESS_ONLY`.
- Neither accuracy nor robustness passes: report a negative D8 result and stop model escalation.
- Training/runtime failure: implementation failure, not scientific evidence.
- Any nonfinite, schema, hash, transaction, or replay mismatch: publication failure.

## 14. Leakage audit

The implementation must prove:

- global preflight may verify outer source bytes, but no outer array, response, embedding, residual, or statistic enters the search object graph before candidate freeze;
- inner query responses score candidates but do not fit transformations, residual donors, PCA, regressors, consistency weights, or ensemble weights;
- P6 residuals are cross-fitted by complete dataset;
- CAI never enters diffusion or residual generation;
- empirical donors come only from current fit domains;
- all variant rows retain physical specimen grouping and total specimen weight one;
- bootstrap resamples domains, not specimens, variants, draws, or seeds;
- outer results cannot mutate the search database or selected configuration;
- production/replay inputs, code, dependencies, checkpoints, and outputs are hash-bound;
- public export contains no raw images, private documents, personal paths, or machine-specific identities.

## Formal result package

The final validated package will contain exactly the prompt-required scientific tables:

```text
results/d8_final/
    aggregate_metrics.csv
    domain_metrics.csv
    bootstrap.csv
    selected_configs.csv
    ablation.csv
    morphology_audit.csv
    search_summary.csv
    REPORT.md
```

Additional model, split, prediction, provenance, figure-source, manifest, and runtime files may be present but cannot replace these required files.

No outer-domain D8 result has been generated in this plan. Success remains unproven until the frozen formal evaluation and independent replay validate.
