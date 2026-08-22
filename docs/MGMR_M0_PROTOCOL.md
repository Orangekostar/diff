# MGMR M0 Complementarity Protocol

Date frozen: 2026-08-22
Status: frozen before M0 feature extraction and result inspection

## Scope and hypothesis

M0 tests whether a coarse morphology component and a directional/boundary
component contain stable complementary CAI information under six-domain LODO.
It does not test a graph, laminate-aware message passing, diffusion, or a new
trainable image encoder.

The cohort is the issued 276-specimen V3 authority. The response is the damaged
to intact CAI strength ratio. The independent outer unit is released dataset.
All reported outer domains have prior project exposure, so this is a registered
post-hoc follow-up rather than independent external confirmation.

## Frozen inputs

- Exact B0: P1 `I_frozen`, equal-domain MAE `0.08963580465761432`.
- Coarse input: P5-compatible 25% endpoint-preserving digital-grid sampling,
  bilinear reconstruction to native size, and exact measured-point restoration.
- Encoder: hash-bound frozen ResNet18 with unchanged grayscale, resize,
  normalization, checkpoint, batch size, and deterministic runtime.
- Primary spatial layer: layer3, shape `N x 256 x 14 x 14`.
- Secondary sensitivity: layer2 only if the primary outcome requires diagnosis;
  it cannot override the primary gate.
- DWT: one level, `db2`, `periodization`, over the final two map axes.
- Wavelet sensitivity: `haar`; it cannot override the primary gate.
- Directional order: PyWavelets `(cH, cV, cD)`, serialized as separate blocks.
- P3 control: 8x8 patch shuffle, seeds 20260831, 20260901, 20260902.

## Feature definitions

```text
f_full = registered 512-D global frozen embedding
f_coarse = GAP(layer3(P5_25_percent_reconstruction))
f_boundary = GAP(cH) || GAP(cV) || GAP(cD)
```

The DWT is applied to FULL layer3 maps, not raw RGB. Reconstruction identity,
band orientation, dtype, border mode, output shape, and repeatability must pass
before any target is loaded by the evaluator.

## Direct models

```text
B0 = metadata13 + PCA(f_full)
B1 = metadata13 + PCA(f_coarse)
B2 = metadata13 + PCA(f_boundary)
B3 = metadata13 + PCA(f_coarse) + PCA(f_boundary)
B4 = metadata13 + PCA(f_full) + PCA(f_boundary)
```

PCA is fitted on raw branch values using only the current training rows. Allowed
dimensions are 8, 16, and 32. Each PCA block is independent. B3 and B4 search
the 3x3 dimension product. Selection minimizes equal-domain inner LODO MAE;
ties within `1e-12` prefer lower total dimension and then lexicographically
smaller dimension tuples. After concatenation, the exact existing mean imputer,
standard scaler, and Ridge alpha 10 pattern is used. Metadata13 occurs once.

B0 must reproduce every registered P1 prediction and the PCA tuple `(8, 32, 8,
8, 8, 8)` within `1e-12`. No MGMR result may replace or reinterpret B0.

## Strict residual audit

Within each outer fold, generate source-domain OOF baseline predictions by
holding out each of the five source domains in turn and selecting the baseline
only on the remaining source domains. Define `r = y - y_hat_oof` on those source
rows. Select and fit a boundary-only PCA/Ridge residual model on those residuals,
then predict one correction for the untouched outer domain. Run this separately
for coarse and exact FULL baselines. The residual model excludes metadata13.

Report baseline residual MAE (zero correction), corrected residual MAE, Pearson
and Spearman residual correlations, aggregate equal-domain effects, each domain,
and the worst domain.

For spatial specificity, recompute the directional feature after each registered
P3 shuffle and repeat the identical coarse-residual procedure. Define benefit as
`MAE_coarse - MAE_corrected`.

## Statistics and artifacts

Use raw specimen MAE per domain and equal weighting over the six domain MAEs.
Report maximum domain MAE as worst-domain error. Use synchronized PCG64 paired
domain bootstrap with seed 20260822 and 100000 resamples for descriptive 95%
intervals of registered method effects. Domains, not specimens or shuffle seeds,
are the inferential units for cross-configuration claims.

The formal directory must contain `config.yaml`, `aggregate_metrics.csv`,
`domain_metrics.csv`, `bootstrap.csv`, `summary.json`, `REPORT.md`,
`artifact_manifest.json`, and `CHECKSUMS.sha256`. A replay must reproduce all
scientific CSV/JSON/Markdown bytes and hashes.

## Gate

Gate A passes when B3 equal-domain MAE is strictly below both B1 and B2, and B3
improves over B1 in at least four of six domains.

Gate B passes when coarse residual correction has positive aggregate benefit and
strictly improves at least four of six domain MAEs.

Gate C is a strong positive diagnostic when FULL residual correction has positive
aggregate benefit and improves at least four of six domains. It is not mandatory.

Gate D passes when the real coarse-residual benefit is strictly greater than
every seed-specific shuffled benefit and the real per-domain benefit exceeds the
mean shuffled benefit in at least four of six domains.

M0 GO requires A, B, and D. Otherwise issue `MGMR_NO_GO` and stop before all M1
code and experiments. B4 thresholds are descriptive: below `0.0896358` is an
improvement, at most `0.08515` is positive, and at most `0.08247` is strong.
