# MSSS S2 Structured Transfer Protocol

Date: 2026-08-22
Status: frozen before formal S1 execution; execution conditional on S1 GO

## Scope

S2 tests the deployable sampling-axis MSSS because sampling density has a direct
sensing interpretation and an existing fixed-25% engineering comparator.
Gaussian and wavelet results corroborate S1 but are not mixed with sampling in
a cross-axis hyperparameter search. This prevents incomparable notions of
coarseness from becoming one post-hoc model selector.

Every task first selects sampling density using only its source specimens and
source-domain inner holdouts, freezes the density and PCA dimension, and then
predicts the unseen target group. Target values cannot affect density,
non-inferiority, PCA, Ridge, or the over-coarse comparator.

## Transfer Tasks and Authorities

- Six-domain LODO: the existing six registered dataset IDs, one target at a
  time.
- Leave-one-ply-count-out: authoritative metadata values 8, 16, and 24, giving
  `16+24 -> 8`, `8+24 -> 16`, and `8+16 -> 24`.
- Leave-one-layup-family-out: `cross_ply -> quasi_isotropic` and
  `quasi_isotropic -> cross_ply`, derived from the registered binary laminate
  metadata and cross-checked against specimen provenance.

Impact-condition shifts are omitted from the primary protocol. They require a
separate frozen sample-size audit and cannot affect S2 status.

Within each source set, inner selection holds out one represented dataset at a
time. The source set contains five domains for ordinary LODO, four domains for
each ply-count task, and three domains for each layup-family task.

## Comparators

- `FULL`: immutable full C-scan embedding.
- `FIXED_25`: exact P5-semantics bilinear density 0.25.
- `SOURCE_MSSS`: coarsest source-noninferior sampling candidate at the frozen
  5% margin.
- `OVER_COARSE`: first coarser source candidate outside the 5% sufficient set.
  If none exists, the registry endpoint is reported with
  `boundary_confirmed=false` and is not described as a validated over-coarse
  control.

Each comparator independently selects PCA dimension from source-only inner
folds. The encoder, metadata, StandardScaler, Ridge alpha, response, and specimen
roster are otherwise identical.

## Metrics and Decision

For every target group and comparator:

```text
TG = MAE(FULL) - MAE(candidate)
RTG = TG / MAE(FULL)
```

Positive transfer means `TG >= 0`. The report includes group MAE, equal-group
MAE, worst-group MAE, selected density, sufficient set, source domains, PCA
dimension, and boundary status. Synchronized stratified specimen bootstraps use
100,000 PCG64 resamples and seed `20260822`.

S2 is `GO` when SOURCE_MSSS is non-worse than FULL in at least three of the five
ply/layup shifts. It is `STRONG_GO` when TG is strictly positive in at least two
of three ply shifts and at least one of two layup shifts. Six-domain results are
reported as ordinary-transfer support and require at least four of six
non-negative effects for that support label.

Selections concentrated within one adjacent density step support a transferable
mesoscale band. Systematic shifts with ply count or layup are labelled
`EXPLORATORY_SCALE_LAMINATE_COUPLING`; they do not authorize a scale-adaptive
network in this protocol.
