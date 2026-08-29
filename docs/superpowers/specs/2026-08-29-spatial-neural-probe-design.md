# Spatial Neural Probe Design

**Date:** 2026-08-29
**Base:** `9794d53a9549f2e3501fe482e8db8735f468ba20`
**Branch:** `mavis-spatial-neural-probe`

## Objective

Test one pre-registered hypothesis: replacing the frozen DeepSets partial-state
encoder with a parameter-matched 8x8 spatial CNN improves legal-state
representation, state-conditioned value estimation, and closed-loop acquisition.
The scientific endpoint, folds, controls, metrics, action space, rollout, and
paper remain unchanged.

## Chosen Integration

Implement a parallel `cmc_bbdm.mavis.neural_probe` namespace. It reuses the
frozen feature bank and existing split, metric, bootstrap, candidate, reveal,
and rollout contracts, while using new checkpoint and result roots.

Rejected alternatives:

- Modifying `MRISStateEncoder` would change a frozen class and invalidate old
  checkpoint assumptions.
- Generalizing the shared P2/P3 runtime around a new protocol would increase the
  regression surface without testing the spatial hypothesis.
- Native-raster CNN input would violate the deployable legal-state contract.

## Legal Input

The only spatial input is the existing `MRISTokenSummary`:

- context: 34 values;
- token features: 64 row-major cells by 6 features;
- token mask: 64 booleans;
- exact-cost features: 3 values.

Tokens reshape from `B x 64 x 6` to `B x 6 x 8 x 8`. The boolean mask is a
seventh channel. No full C-scan, native-raster image, target CAI, future
measurement, or unrevealed content enters inference.

## Fixed Architecture

`spatial_grid_cnn_v1` is fixed before results:

```text
Conv2d(7, 16, 3, padding=1), ReLU
Conv2d(16, 32, 3, padding=1), ReLU
Conv2d(32, 32, 3, padding=1), ReLU
global average + global maximum -> 64

Linear(34 + 3, 32), ReLU
Linear(32, 32), ReLU

Linear(64 + 32, 64), ReLU
Linear(64, 64)
```

The encoder has 27,552 parameters. With the scalar P2 head it has 27,617,
which is 160 fewer than the frozen DeepSets P2 model. The output remains 64-D,
so `DynamicActionScorer` remains unchanged.

## Training Protocol

- seed: `20260825`;
- optimizer learning rate: `0.001`;
- P2: batch 256, maximum 80 epochs, patience 10;
- P3: batch 64, maximum 40 epochs, patience 5;
- same six-domain nested LODO roster and source-only normalization;
- same inner validation and median selected-epoch final refit;
- target domains are evaluated only after selection.

P2 modes are `real`, `positions_only`, `shuffled`, and `static`.
Reconstruction remains the existing independent control. N3 reports both the
full frozen bank and the pre-registered `CLEAN_NONPRIV = {uniform, random}`.

## Stages And Gates

- N1 compares equal-domain P2 AUEBC using `DeepSets_real - Spatial_real`.
- N2 compares next-action regret using `DeepSets - Spatial` with the unchanged
  P3 scorer.
- N3 reports `positions/history - real` and `shuffled - real`, where positive
  means measured content helps.
- N4 compares closed-loop AUEBC using `static_reference - new_candidate` and
  secondarily compares with the frozen learned implementation.

All confidence intervals use existing specimen-first, equal-domain bootstrap
code. Strong gates require a positive point estimate, positive lower 95% CI,
and favorable direction in at least four of six domains. Promising gates use a
positive point estimate and at least four favorable domains with a CI including
zero. N3 requires both clean-control contrasts to pass for a strong claim.

## Artifacts And Provenance

New artifacts live only under `results/mavis_neural_probe/` and
`artifacts/mavis_neural_probe/`. Each stage records a report, JSON summary,
domain metrics, bootstrap results, manifest, and checksums. New checkpoints bind
the legal schema, architecture, base commit, fold rosters, normalizer,
feature-bank hashes, hyperparameters, seed, selected epoch, model state hash,
and configuration hash.

## Non-Goals

No paper edit, new rollout, new action descriptor, new loss, tuning, model
selection from target domains, training-data expansion, Transformer, GNN, RL,
diffusion, ResNet, or N5 candidate-local readout is authorized.
