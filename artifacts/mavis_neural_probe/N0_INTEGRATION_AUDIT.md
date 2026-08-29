# N0 Integration Audit

## Decision

`N1_AUTHORIZED`

Audit date: 2026-08-29. No training was run during N0.

## Repository Identity

- branch: `mavis-spatial-neural-probe`
- base and audit HEAD: `9794d53a9549f2e3501fe482e8db8735f468ba20`
- remote authority: `origin/aei-main-method-reframe` at the same SHA
- isolated worktree: `/home/ww/diff/.worktrees/mavis-spatial-neural-probe`
- initial worktree: clean; `git diff --check` clean

## Frozen Evidence

| Authority | Git tree/blob |
| --- | --- |
| `results/mavis/p2_mris` | `f73038ab616710c3953af82569241819fafb96d7` |
| `results/mavis/p3_dynamic_voi` | `36866c1f4f351ef9d0d3915a24c71dade6807b65` |
| `results/mavis/p7_final_frozen_eval` | `b7fb24ff2d808db6fd8ec4f6571daef55016b96c` |
| `results/mavis_science_closure` | `f57c92f5b629745746decd6e005d7b7b152b491a` |
| canonical metrics CSV | `ba4229ca00b3298444d43d41a6d63a91d1fd11c0` |

- canonical SHA-256: `f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`
- frozen config: `paper_v3/configs/mavis_final.yaml`
- frozen config SHA-256: `e99b47e161663fdaefe28719d16321a010f95b4ad8cf8f506a6e18d1d7f57b9d`
- P7 config snapshot has the same SHA-256.
- registered probe config: `artifacts/mavis_neural_probe/NEURAL_PROBE_CONFIG.json`
- probe config SHA-256: `26299353b1870081660af452cb7e8ca3ba0b4fe87666f26c3922ef1a9ef7fc66`

The old result roots, canonical metrics, paper evidence code, paper sources, and
all native scientific result paths are read-only for this probe.

## Frozen Feature Bank

The existing P2 bank was inspected without rebuilding P1:

- file SHA-256: `280c608d43be164cce8617aea1cc24bf3152d537c94a48d40b61ea15085d6467`
- rows: 8,280; specimens: 276; domains: 6
- context: `(8280, 34)` float64
- real/positions tokens: `(8280, 64, 6)` float32
- masks: `(8280, 64)` bool
- costs: `(8280, 3)` float64
- shuffled: `(6, 8280, 64, 6)` float32
- reconstruction control: `(6, 8280)` float64
- target vector: `(8280,)`
- each specimen contributes five trajectory methods at six registered checkpoints

The trajectory roster is `uniform`, `random`, `reconstruction_driven`,
`sequential_mechanical_oracle`, and `one_shot_mechanical_oracle`.

## Architecture And Checkpoints

The current `MRISStateEncoder` consumes `B x 64 x 6` tokens, a `B x 64` mask,
34 context features, and three cost features. Its masked DeepSets pooling uses
mean and sum divided by square-root count. It has 27,712 parameters and emits
64-D embeddings; the scalar P2 model has 27,777 parameters.

`DynamicActionScorer` consumes a 64-D state and 8-D candidate descriptor. Its
state, action, and fusion MLPs have 21,442 parameters. It can be reused unchanged.

Existing P2 and P3 checkpoints use schema v1 NPZ files with JSON metadata,
normalizer arrays, named weights, selected epoch, model state SHA, fold roster,
and audit fields. They are bound to old concrete types/weight names. The probe
therefore uses a parallel schema and never overwrites or silently loads them.

The fixed spatial encoder has 27,552 parameters, and 27,617 with its P2 head.
It is parameter-matched and slightly smaller than the DeepSets baseline.

## Legal-State And Leakage Audit

`InspectionState` contains specimen context, shape/count metadata, acquisition
grid identity, registered levels/checkpoints, sorted acquired positions and
their revealed measurements, exact cost, and action history. Full scans and CAI
targets live in separate source-teacher/evaluation views.

The deployable flow is:

```text
InspectionState
  -> build_mris_input
  -> summarize_mris_input
  -> MRISFeatureBank/model_inputs
  -> P2 64-D embedding and prediction
  -> unchanged P3 DynamicActionScorer
  -> existing score_actions rollout protocol
```

`summarize_mris_input` maps normalized acquired coordinates to row-major bins by
`row_bin * 8 + column_bin`. The legal reshape is `B x 64 x 6` to
`B x 6 x 8 x 8`, with the mask as channel seven. No native-raster input is
needed. `reveal_action` is the only causal path that reveals new values after an
action and validates monotonic measurements and exact budget.

The current nested LODO implementations fit normalizers and select epochs from
source domains only. Target domains are evaluated after selection. The probe
will preserve these rosters and validate them in tests and checkpoint metadata.

## Integration Choice

Create `src/cmc_bbdm/mavis/neural_probe/`. Do not change the frozen DeepSets
class, old checkpoint reader, dynamic scorer, candidate descriptor, teacher,
loss, metric, bootstrap, action roster, reveal contract, or rollout loop.

The new P2 model retains `encode`, `predict`, `predict_inspection_state`, and
`model_state_sha256` capabilities. A small deployed-scorer adapter will satisfy
the existing `score_actions(state, candidates)` protocol.

## Fixed Compute Contract

- architecture: `spatial_grid_cnn_v1`
- seed: `20260825`; learning rate: `0.001`
- P2: 80 epochs maximum, patience 10, batch 256
- P3: 40 epochs maximum, patience 5, batch 64
- P2 fits: 144; P3 fits: 144; N4 rollouts: 276 specimens x 6 checkpoints
- hardware: 3 NVIDIA A40 GPUs, 64 CPU cores, 225 GiB available RAM
- free disk at N0: 63 GiB; expected new output: 0.2--0.5 GiB

## Baseline Validation

The state encoder, P2/P3 data/training/execution, dynamic VoI, rollout, and
closed-loop metric tests completed with `30 passed in 7.19s`. This is a software
regression check, not a scientific rerun.

## Files And Boundaries

Only new design/plan documents, `cmc_bbdm.mavis.neural_probe`, new neural-probe
tests, and new `artifacts/mavis_neural_probe` / `results/mavis_neural_probe`
paths are authorized. No existing source file is planned for modification.
