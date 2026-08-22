# Mechanically Sufficient Spatial Scale Design

Date: 2026-08-22
Status: approved by the controlling prompt and no-question recommended-configuration authorization

## Objective

Discover whether CAI prediction has a stable compact spatial-information
plateau across sampling, Gaussian, and wavelet interventions, require retained
spatial specificity, and conditionally test the sampling-axis boundary under
dataset, ply-count, and layup transfer without target-guided scale selection.

## Architecture

The system has four boundaries. `authority.py` freezes sources, cohort, groups,
and configuration. Axis modules transform immutable native RGB crops and emit
hash-bound records. `scale_features.py` encodes each unique condition once with
the existing frozen ResNet18. `scale_evaluator.py` fits the existing standardized
metadata-plus-PCA Ridge expert under source-only nested domain folds.

S1 evaluates all fixed candidates for curves but promotes only cross-fitted
source-selected scales. It then materializes the exact P3 8x8 control after the
selected scale transform and evaluates its own source-fitted predictor. S2 is a
conditional second executable that consumes the validated S1 package and uses
only sampling candidates.

## Alternatives Considered

Reusing only the old FULL/50/25 feature bank cannot test Gaussian, wavelet, or a
boundary finer than the P5 grid. Re-encoding inside every fold is equivalent but
wasteful and makes provenance harder to audit. A one-time immutable feature bank
followed by pure fold-local fitting isolates image scale, supports replay, and
keeps outer-target isolation explicit, so it is the registered approach.

Cross-axis selection was rejected because density, sigma, and DWT level have no
common compactness ordering. Sampling is the S2 operational representation;
Gaussian and wavelet are independent S1 corroboration axes.

## Core Data Flow

```text
V3 authorities + registered crops
  -> deterministic axis conditions and transform records
  -> frozen ResNet18 embeddings, one condition at a time
  -> fixed-candidate nested predictions and source inner scores
  -> per-outer source-only non-inferiority selection
  -> selected-scale 8x8 shuffled embeddings and SSG
  -> S1 gate
  -> conditional sampling-only S2 transfer tasks
  -> atomic artifacts, figures, manifest, checksum replay
```

FULL, sampling 50%, and sampling 25% are cross-checked against the immutable A2
feature-bank slices. A mismatch aborts rather than silently changing baseline.

## Components

- `protocol.py`: strict YAML registry and exact source hashes.
- `authority.py`: V3 crops, A2 feature bank, ply and layup identities.
- `sampling_scale.py`, `gaussian_scale.py`, `wavelet_scale.py`: deterministic
  transforms and validation records.
- `scale_features.py`: condition registry, frozen encoding, cache serialization.
- `scale_evaluator.py`: nested source-only PCA/Ridge evaluation.
- `noninferiority.py`, `msss_selector.py`: margins, coarseness, plateau,
  boundary, and stability.
- `spatial_specificity.py`: post-scale P3 control and SSG.
- `transfer_tasks.py`, `source_only_selection.py`, `transfer_metrics.py`: S2.
- `statistics.py`: synchronized stratified specimen bootstrap.
- `artifacts.py`, `figures.py`, `replay.py`: atomic outputs and verification.
- `scripts/run_msss_s1.py`, `scripts/run_msss_s2.py`: strict entry points.

## Failure Handling

Any source hash, roster, identity reconstruction, feature-bank reproduction,
fold isolation, non-finite value, incomplete Cartesian output, or target leakage
aborts. Formal output directories are immutable: publishing uses a same-parent
staging directory, validates the staged package, and atomically renames it only
when the registered destination does not exist. S2 rejects any S1 status other
than `GO` or `STRONG_GO`.

## Decision Boundary

The exact margins, axis registries, stability rule, spatial-specificity rule,
bootstrap, and S1/S2 statuses are defined in `MSSS_S1_PROTOCOL.md` and
`MSSS_S2_TRANSFER_PROTOCOL.md`. No adaptive network, learned scale fusion,
frequency diffusion, or target-informed scale change is part of this design.
