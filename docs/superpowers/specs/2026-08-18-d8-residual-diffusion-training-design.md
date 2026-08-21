# D8 Fold-Local Residual Diffusion Training Design

Date: 2026-08-18

## 1. Authorization and scope

The validated D8 Pilot decision is `TRAIN_RESIDUAL_DIFFUSION`. Four of six
prospective outer studies met the registered positive-trend trigger, and all
six met the P6-residual mismatch trigger because 0.889--0.942 of residual
energy lay below the selected nuisance cutoff. The formal outer evaluation has
not started (`outer_evaluation_count=0`).

This stage trains and selects a response-free conditional residual generator
inside each prospective outer-fit view. It publishes a pre-outer package and
six frozen pipelines. It cannot issue or score an outer-domain evaluation
view.

## 2. Chosen residual target

Three approaches were considered:

1. Train on the original P6 reconstruction errors. This is rejected as the
   formal target because the registered mismatch gate failed in all six
   studies. It would also require a prohibitively deep nested P6 refit to keep
   every prospective outer domain outside the residual-target generator.
2. Fine-tune the P6 full-image reconstruction model. This is rejected because
   P6 already established `NO_MECHANICAL_GAIN`, and the target prompt forbids
   returning to full-image reconstruction.
3. Model the measured field's selected nuisance band conditional on its
   complementary morphology component. This is selected because it directly
   implements `p(R|S)`, needs no outer-domain or CAI information, and preserves
   an exact decomposition of every fit-domain field.

For each prospective outer study, the best eligible Pilot diffusion candidate
freezes one decomposition family, band, and parameter set. For a normalized
measured field `D`:

```text
R_observed  = BandPass(D; frozen Pilot decomposition)
S_raw       = D - R_observed
S_condition = clip(S_raw, -1, 1)
```

The model learns `p_theta(R_observed / 2 | S_condition)`. `S_raw` is retained
unchanged so that `D = S_raw + R_observed` remains exact. Some registered
wavelet decompositions overshoot `[-1,1]`; the fixed clip is therefore applied
only to the model condition. The residual scale is fixed globally, and every
fit authority must pass a registered preflight confirming that
`R_observed / 2` lies in `[-1,1]`. No statistic from an inner-query or outer
domain determines either transform.

At inference, one sample gives `R_sample`, and the existing Pilot scaffold is
applied as:

```text
D_variant = clip(D + alpha * (R_sample - R_observed), -1, 1)
```

This is the residual-replacement form of the Pilot perturbation
`D + alpha*(D_generated-D)`. The original field remains the deterministic
fallback. The registered native-frame morphology gate is rerun on every
variant.

## 3. Fold isolation

For prospective outer domain `O`, only the other five domains enter its search
view. For inner query domain `Q`, the generator, residual normalization,
downstream PCA, regressor, and consistency terms use only the remaining four
fit domains. Query `S_condition` is supplied only after the generator
checkpoint is frozen. Query CAI is used only to compute the inner selection
objective; it is never passed to the generator.

After selection, three final generator checkpoints are trained on all five
outer-fit domains. The outer domain remains unopened. Model input APIs accept
only process-local `D8InnerFold` or frozen five-domain search authorities.

## 4. Frozen model candidates

All models use `diffusers.UNet2DModel`, 64x64 inputs, three noisy-residual
channels, three morphology-condition channels, three output channels, one
layer per block, group normalization, and no response input. The eight
registered candidates are:

| ID | Base channels | Prediction | Beta schedule | Bottleneck attention | Spectral weight | Low-pass weight |
| --- | ---: | --- | --- | --- | ---: | ---: |
| RD0 | 32 | epsilon | squared-cosine | no | 0.00 | 0.00 |
| RD1 | 32 | epsilon | squared-cosine | no | 0.05 | 0.10 |
| RD2 | 32 | v | squared-cosine | no | 0.05 | 0.10 |
| RD3 | 32 | direct residual | squared-cosine | no | 0.05 | 0.10 |
| RD4 | 64 | epsilon | squared-cosine | bottleneck only | 0.05 | 0.10 |
| RD5 | 64 | v | squared-cosine | bottleneck only | 0.05 | 0.10 |
| RD6 | 32 | epsilon | linear | no | 0.05 | 0.10 |
| RD7 | 32 | v | linear | no | 0.05 | 0.10 |

The base objective is diffusion MSE. Auxiliary spectral loss is the mean L1
difference between channel-wise orthonormal FFT magnitudes of predicted and
target clean residuals. The low-pass term is the mean L1 difference after the
registered Gaussian sigma-2 filter. For epsilon and v prediction, clean
residuals are reconstructed analytically from the noisy sample and scheduler
alpha product before auxiliary losses are evaluated.

Training uses AdamW, batch size 32, learning rate `2e-4`, weight decay `1e-4`,
1,000 train timesteps, deterministic final-epoch selection, and no early
stopping. DDIM sampling uses 25 steps and `eta=1.0` with identity-derived
seeds.

## 5. Staged inner search

Each prospective outer study is independent.

Stage A trains all eight candidates for 24 epochs with seed `20260823` in all
five inner folds. The two candidates with lowest

```text
J = mean(inner-domain MAE) + 0.25*worst MAE + 0.10*domain SD
```

advance. Ranking then uses mean MAE, worst MAE, parameter count, and candidate
ID. Every candidate is evaluated through the frozen Pilot decomposition,
alpha, `K_train`, `K_test`, feature layer, aggregation, consistency, PCA, and
regressor scaffold for that prospective outer study.

Stage B retrains both finalists from scratch for 120 epochs with seeds
`20260823`, `20260824`, and `20260825` in all five inner folds. One prediction
per specimen is first formed for each seed using the registered `K`; the three
seed predictions are then averaged before domain MAE and `J` are computed.
Seeds and residual draws are not inferential units.

All candidates must retain the Pilot morphology gates: overall acceptance at
least 0.80 and every inner-query-domain acceptance at least 0.60.

## 6. Pipeline selection and fallback

For each prospective outer study, the eligible choices are:

- the best Stage-B residual diffusion candidate;
- the already frozen D8 Pilot selection;
- raw `I_frozen`.

The residual model replaces the Pilot selection only when its repeated-seed
`J` improves the best incumbent by at least `1e-4`. A two-member nonnegative
cross-fitted ensemble is accepted only when it adds a further `1e-4` gain.
Otherwise the simpler incumbent remains frozen. This rule prevents the
training authorization from forcing an inferior residual model into formal
evaluation.

Three final checkpoints of the selected residual architecture are trained on
the complete five-domain outer-fit view. If the incumbent is retained, no
residual checkpoint is used by that fold. All six pipeline documents are
hash-frozen before any outer evaluation authority can be issued.

## 7. Compute and execution

The registered run uses three NVIDIA A40 GPUs. A parent process assigns two
prospective outer studies to each GPU. Worker outputs are isolated and merged
only after all three worker manifests validate. Screening trains 240 models;
full reranking trains 180 models; at most 18 final outer-fit checkpoints are
trained. The run stores every full-rerank and final checkpoint. Screening
models are reproducible from their exact training records and state digests
but are not publication models.

## 8. Pre-outer result package

The output leaf is `results/d8_residual_diffusion_search` and contains exactly:

```text
config.yaml
candidate_index.csv
training.csv
inner_predictions.csv
inner_metrics.csv
checkpoint_index.csv
selected_generators.json
frozen_pipelines.json
models/
REPORT.md
artifact_manifest.json
CHECKSUMS.sha256
```

The package binds the Pilot package, selected Pilot scaffolds, measured source
fields, split authorities, model tensor bytes, runtime, code, candidate rows,
predictions, metrics, and final pipeline states. It records
`outer_evaluation_count=0`. Publication uses the existing lock, owner-marked
staging, validation-before-commit, rollback, and crash-recovery pattern.

## 9. Gate to formal evaluation

Formal outer evaluation remains blocked until:

1. all 30 inner folds per Stage-B candidate have complete predictions;
2. all package hashes, row counts, split memberships, checkpoints, scores, and
   selection decisions independently recompute;
3. all six frozen pipeline documents validate under a fresh process;
4. the registered wrapper reports `outer_evaluation_count=0`;
5. an independent replay of the pre-outer training package has the same
   canonical scientific digest after excluding only registered timing fields.

No accuracy or diffusion-specific claim is unlocked by this design document.
