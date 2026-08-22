# MSSS NO-GO Coupling Diagnostic Design

Date: 2026-08-22
Status: approved under the controlling prompt's no-question recommended-configuration authorization

## Objective

Complete the mandatory S1 `NO_GO` branch by checking whether descriptive
mechanically sufficient scales vary with domain, ply count, layup family, or
damage size. This is a post-hoc diagnostic, not S2, not a new MSSS promotion,
and not validation of Scale-Laminate Coupling.

## Alternatives

The selected design derives group curves from the immutable formal S1
cross-fitted candidate predictions. It preserves the already executed models,
requires no target-guided refit, and can inspect all requested factors.

Re-training separate subgroup models was rejected because several structural
groups have too little independent domain coverage and the formal S1 predictor
would no longer be held fixed. Reporting only the six source-selected scales
was rejected because those scales are functions of complementary source sets
and cannot diagnose specimen damage size.

## Authorities And Data Flow

The diagnostic first validates the formal S1 package and requires
`gate_status=NO_GO`, `test_only=false`, and the registered scientific digest.
It then loads the hash-bound 276-specimen V3 authority. Ply count and layup come
from the cross-checked MSSS authority; damage size uses the immutable physical
descriptor triplet in `V3Data.scalar_internal3`: projected damage area,
damage height, and damage width.

Only primary candidates are eligible: all sampling and Gaussian registry
conditions, plus `db2` cumulative low-pass wavelet levels. Per-specimen errors
come from S1 `candidate_predictions.csv`; each prediction was made by a model
that excluded its dataset domain.

```text
validated formal S1 NO_GO package + immutable V3 authority
  -> cross-fitted per-specimen candidate errors
  -> deterministic diagnostic groups
  -> group-balanced candidate curves
  -> local descriptive 5% sufficient set and coarsest candidate
  -> cross-axis direction audit
  -> immutable post-hoc diagnostic package
```

## Group Definitions

- `domain`: the six registered dataset IDs.
- `ply_count`: 8, 16, and 24.
- `layup_family`: cross-ply and quasi-isotropic.
- `damage_area`, `damage_height`, `damage_width`: stable rank-balanced
  tertiles. Values are ordered by `(physical value, specimen_id)` and split
  into three groups of 92 specimens. This avoids unstable quantile edge
  handling when physical measurements tie.

For groups spanning multiple domains, the score is the mean of within-domain
MAEs so large domains cannot dominate. A domain group uses its specimen MAE.
Every group must contain specimens and every expected condition exactly once.

## Selection And Trend Rules

Within each axis and diagnostic group, FULL is the first fine-to-coarse
candidate. The descriptive sufficient set is

```text
group_MAE(s) <= 1.05 * group_MAE(FULL).
```

The selected scale is the coarsest eligible candidate. The first registered
coarser ineligible candidate is the descriptive over-coarse boundary, when it
exists. This result may describe a local curve but cannot promote the failed S1
axis.

Ply and damage-tertile sequences are labelled `COARSER`, `FINER`, `SAME`, or
`NON_MONOTONIC`. Layup is a signed two-group contrast or `SAME`. A factor has
`CROSS_AXIS_ALIGNED` exploratory evidence only when at least two of three axes
have the same non-neutral direction. Otherwise it is
`NO_CROSS_AXIS_ALIGNMENT`. No p-value, causal interpretation, or universal
scale claim is attached to this small, post-hoc factorial audit.

## Outputs And Claims

The separate `results/msss/s1_no_go_coupling/` package contains:

- `group_scale_curves.csv`
- `group_scale_selection.csv`
- `damage_size_bins.csv`
- `factor_trends.csv`
- `summary.json`
- `REPORT.md`
- `artifact_manifest.json`
- `CHECKSUMS.sha256`

The package is atomic, non-overwriting, checksum-bound, and replay-validated
from the formal S1 predictions. Its summary always records
`validation_status=NOT_VALIDATED_POST_HOC` and
`s2_status=NOT_RUN_NOT_AUTHORIZED`. A cross-axis signal authorizes only a new
independent preregistration; it never authorizes S2 or a scale-adaptive model.

## Failure Handling And Tests

The run fails closed on an invalid or non-`NO_GO` parent, parent digest drift,
roster mismatch, duplicated/missing predictions, non-finite physical values,
incomplete groups, unknown conditions, overwrite attempts, or checksum/replay
failure. Unit tests cover deterministic tertiles, equal-domain aggregation,
coarsest selection, trend classification, parent-gate rejection, required
artifacts, and corruption detection.
