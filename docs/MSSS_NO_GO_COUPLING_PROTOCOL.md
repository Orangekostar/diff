# MSSS S1 NO-GO Coupling Diagnostic Protocol

Date: 2026-08-22
Status: frozen after formal S1 `NO_GO` and before coupling execution

## Scope And Parent

This post-hoc diagnostic implements the controlling prompt's required S1
`NO_GO` branch. Its only parent is the immutable formal S1 package with
scientific digest
`6ac389b0a4e09487202f5a8a9273dfdf5b338ef40de705661c5877e3e9bd0152`
and output-tree digest
`e41c42bdd8cb022b2d7d3c286685ae2530f2c302a236cf2e0a77d76ecf6a365b`.
The parent must validate as formal `NO_GO` before this analysis runs.

This diagnostic is not S2. It cannot change the S1 decision, select a target
deployment scale, validate a transferable MSSS, or authorize an adaptive
network.

## Inputs And Grouping

The only performance input is the complete primary-candidate roster in formal
S1 `candidate_predictions.csv`. Those predictions are already cross-fitted by
dataset domain. Structural values come from the immutable MSSS authority:
registered domain, ply count, layup family, and the three physical descriptors
projected damage area, damage height, and damage width.

Damage descriptors are split independently into stable rank-balanced tertiles.
Ordering is `(value, specimen_id)` and labels are `low`, `middle`, `high`; the
276 specimens therefore give 92 specimens per bin. No CAI target enters the
binning rule.

## Curve And Selection Rule

Only sampling, Gaussian, and primary `db2` low-only wavelet candidates are
eligible. For domain, ply-count, layup-family, and each damage-tertile group,
candidate MAE is averaged equally across represented dataset domains. A
single-domain group reduces to ordinary specimen MAE.

At the frozen primary margin, a descriptive local candidate is sufficient when

```text
group_MAE(s) <= 1.05 * group_MAE(FULL).
```

The diagnostic selected scale is the coarsest sufficient candidate. A boundary
is present only when a registered coarser ineligible candidate exists.

## Trend Rule

For ply count and damage tertiles, selected coarse ranks are classified as
`COARSER`, `FINER`, `SAME`, or `NON_MONOTONIC` in their registered order.
Layup uses the order cross-ply then quasi-isotropic. A factor is
`CROSS_AXIS_ALIGNED` only when at least two of the three axes share the same
non-neutral direction. Otherwise it is `NO_CROSS_AXIS_ALIGNMENT`.

Any aligned result is only `EXPLORATORY_SIGNAL`. All outputs retain
`validation_status=NOT_VALIDATED_POST_HOC` and
`s2_status=NOT_RUN_NOT_AUTHORIZED`. Independent pre-registration and new
evidence are required before a Scale-Laminate Coupling claim can be validated.

## Artifacts And Replay

The formal package is `results/msss/s1_no_go_coupling`; replay is
`results/msss/replay/s1_no_go_coupling`. Publication is atomic and
non-overwriting. CSV schemas, canonical JSON, parent digests, source hashes,
scientific state, output-tree digest, and SHA-256 checksums are validated.
Replay recomputes the diagnostic from the parent and must reproduce the same
scientific digest.
