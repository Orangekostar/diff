# P1 Real Internal Observability

This is a real-data, retrospective seven-domain LODO benchmark over the registered 292-specimen CFRP cohort.

## Contract

- Physical targets are read by exact `sample_id` from the P0-registered physical C-scan table.
- Each physical `source_sha256` is checked against the manifest `target_sha256`.
- The legacy `feature_cache.targets` array is not used as a target or for model selection.
- Primary metric: equal-domain standardized MAE; physical-unit MAE/RMSE are secondary.
- P1 primary comparison: nested-selected surface model; each target is selected by inner LODO across all four surface families.
- ExtraTrees predictions are averaged over frozen model seeds 20260811, 20260812, and 20260813 before metrics.
- Negative controls: synchronized profile+RGB surface shuffle with unchanged metadata/test features, and target shuffle within each outer training fold.
- Domain bootstrap: seed 20260811, 100000 resamples, simultaneous family size six.

## Feature mapping

- `train_mean`: no covariates; mean of outer-training physical targets
- `metadata_only`: feature_cache.metadata (10 registered impact/layup fields)
- `profile_ridge`: feature_cache.profile_stats (21 registered profile statistics) + Ridge
- `profile_extra_trees`: feature_cache.profile_stats (21 registered profile statistics) + ExtraTrees
- `frozen_rgb`: feature_cache.rgb (512-D frozen ResNet18 ImageNet embedding) + Ridge
- `combined`: metadata (10) + profile_stats (21) + frozen_rgb (512) + Ridge

## G1

G1 status: **FAIL** (nested-selected surface vs metadata-only).

- `projected_damage_area`: **FAIL**; relative improvement=`0.06213525709712912`, improved domains=`3`, simultaneous lower=`-0.164873`.
- `damage_height`: **FAIL**; relative improvement=`0.0659890251427522`, improved domains=`6`, simultaneous lower=`-0.0522272`.
- `damage_width`: **FAIL**; relative improvement=`0.07046980698342199`, improved domains=`5`, simultaneous lower=`-0.13131`.

Secondary targets are reported for completeness and are never used to decide G1.
Surface-shuffle and target-shuffle paired comparisons are formal outputs: 36 surface-shuffle rows and 36 target-shuffle rows with simultaneous CIs.

Secondary holdouts (leave-layup-out, leave-impactor-shape-out, and registered energy-band holdout) were not executed, were not used for G1, and were stopped after the frozen primary G1 failed. No secondary result is inferred.

All numeric values in this report are generated from the production run artifacts; no result is filled or inferred when a model fails.
