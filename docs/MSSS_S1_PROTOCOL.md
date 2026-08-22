# MSSS S1 Scale Discovery Protocol

Date: 2026-08-22
Status: frozen before formal S1 execution

## Objective and Authorities

S1 tests whether a compact mesoscale representation preserves CAI performance
and remains dependent on spatial organization. The independent unit is the
specimen. The cohort is the immutable 276-specimen, six-domain V3 cohort. Full
images, targets, metadata, frozen ResNet18 weights, and domain identities are
bound to their existing hashes. No image-level split is permitted.

FULL is the existing `FULL` slice of the immutable A2 paired feature bank and
must reproduce equal-domain MAE `0.08963580465761432` within `1e-12`. Newly
encoded identity conditions must match that feature slice within numerical
tolerance before their axes are interpreted.

## Scale Registries

Sampling is primary bilinear reconstruction at requested densities
`[1, .75, .625, .5, .375, .25, .1875, .125, .0625]`. It reuses P5's ties-to-even
coordinate count, endpoint-inclusive `linspace`, coordinate rounding,
`align_corners=True`, no antialiasing, and exact measured-point restoration.
Every specimen records requested and effective density, row/column counts,
mean grid spacing in pixels and millimetres, and measured-point count.

Gaussian candidates are sigma
`[0, .5, 1, 1.5, 2, 3, 4, 6, 8]` pixels. Filtering is channel-independent,
reflect-padded, float64, rounded and clipped back to uint8. Native image shape,
0--255 intensity semantics, frozen ResNet preprocessing, and output shape remain
unchanged. No sigma in millimetres is reported because the registered crop
authority does not establish one common native pixel pitch for all screenshots.

Wavelet primary candidates are cumulative `db2` low-pass reconstructions at
levels `[0, 1, 2, 3]`, using PyWavelets `wavedec2`/`waverec2` with
`periodization`, original-shape crop, rounding, clipping, and uint8 restoration.
`haar` and `db4` repeat levels 1--3 as sensitivity analyses. For every family
and level, `low_plus_boundary_details` keeps the approximation and the detail
triplet at the decomposition boundary while removing finer details. These
detail variants diagnose edge/detail dependence and are not eligible for the
primary MSSS boundary. No band is labelled mechanically useful in advance.

The Fourier registry `[1, .75, .5, .35, .25, .15, .10]` is optional and cannot
change or block the S1 decision.

## Frozen Predictor

Every eligible candidate uses exactly the same frozen ImageNet ResNet18 and the
registered estimator:

```text
RGB condition -> frozen 512-D embedding
-> fold-local SVD-PCA in {8, 16, 32}
-> metadata13 + PCA scores
-> fold-local mean imputation and StandardScaler
-> Ridge(alpha=10)
-> CAI ratio
```

PCA dimension is selected by equal-domain MAE on source-only inner-domain
holdouts. Ties within `1e-12` retain registry order. No encoder fine-tuning,
target change, additional CNN, or scale-specific regressor is allowed.

## Source-Only Selection

For each outer domain and axis, all scale and PCA decisions use only the other
five domains. Candidate inner OOF scores define

```text
S_MS = {s: source_MAE(s) <= (1 + margin) * source_MAE(FULL)}.
```

The primary margin is 5%; 2.5% and 7.5% are sensitivity margins. The selected
candidate is the coarsest eligible scale: lowest sampling density, largest
Gaussian sigma, or highest primary db2 low-pass level. Its prediction on the
outer domain is made only after selection is frozen. Aggregate fixed-candidate
outer curves are descriptive and never select the candidate tested on that
same outer domain.

Boundary stability passes when at least four of six source-only selections fall
inside one contiguous two-candidate window. A plateau requires at least two
adjacent non-FULL candidates in the aggregate 5% mechanically sufficient set.
An over-coarse boundary exists only when a registered coarser candidate exceeds
the 5% margin; absence of such a candidate is reported rather than inferred.

## Spatial Specificity

For each outer-selected candidate, the registered P3 8x8 patch shuffle is
applied after the scale transform with seeds `[20260831, 20260901, 20260902]`.
The scale stays fixed; each shuffled predictor selects only PCA dimension from
source data. Define

```text
SSG = MAE(shuffled selected scale) - MAE(selected scale).
```

An axis passes specificity when aggregate SSG is strictly positive and at least
four of six outer-domain effects are positive. A family-wise simultaneous lower
bound above zero is reported as stronger evidence but is not required for the
primary promotion rule. Pixel shuffle is secondary and is run only if the
registered patch control is technically invalid.

## Statistics and Gate

Primary metrics are per-domain MAE and equal-domain MAE. Confidence intervals
use 100,000 synchronized PCG64 specimen resamples stratified within each of the
six domains, seed `20260822`; the same draws are used for all compared methods.
Ordinary 95% and three-axis Bonferroni-family intervals are reported.

An axis passes when it has a plateau, a confirmed over-coarse boundary, stable
source-only selections, cross-fitted selected MAE within the 5% FULL margin,
and the spatial-specificity rule. S1 is `GO` for at least two passing primary
axes, `STRONG_GO` for all three, and `NO_GO` otherwise. Normalized candidate
rank is visualization-only and never equates density, sigma, and DWT level.

If S1 is `NO_GO`, S2 is not executed. Ply count, layup, domain, and damage-size
associations with selected scale may be reported only as exploratory
scale-laminate coupling.
