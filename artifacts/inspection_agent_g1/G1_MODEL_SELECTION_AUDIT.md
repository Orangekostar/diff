# G1 Model-Selection Audit

Status: `SOURCE_ONLY_SELECTION_FROZEN_BEFORE_TARGET_EVALUATION`

Authority: `results/inspection_agent/g1/outer_selection.json` and
`results/inspection_agent/g1/model_manifest.csv`

## Registered Search Space

- Actors: `SharedActionMLP` (260,098 parameters) and
  `StructuredInspectionPolicy` (424,754 parameters), both below the 1,000,000
  parameter cap.
- Training routes: hard behavioral cloning and soft utility distillation.
- Soft temperatures: 0.25, 0.5, 1.0, 2.0.
- CAI context: task-specific masked or shared observable-state context.
- Learning rates: `1e-4`, `3e-4`; weight decays: `1e-4`, `1e-3`.
- DAgger rounds: 0, 1, 2, using outer-source worlds only.
- Conditional AAWR: expectiles 0.7/0.8 and beta 1.0/3.0, only after the
  preregistered source-observability authorization gate.
- STOP thresholds: 0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99.

All normalization, early stopping, hyperparameter comparison, DAgger relabeling,
AAWR authorization, fixed-comparator choice, and STOP-threshold selection used
only the five outer-source domains with inner leave-one-source-domain-out
validation. `target_outcomes_opened=false` is sealed in all six selections.

## Selected Action Policy

Every outer fold selected `SharedActionMLP`, `SOFT_UTILITY_DISTILL`, tau 0.5,
learning rate `3e-4`, weight decay `1e-4`, shared observable CAI context, and the
correct task token. Five folds selected DAgger round 2; outer `yfxyg8jm46`
selected round 0.

| Outer target | DAgger | Overall relative AUEBC | FIELD relative | CAI relative | Improved source domains | Mean gap closure |
|---|---:|---:|---:|---:|---:|---:|
| `74t7kcdgkr` | 2 | 0.994113974 | 0.999794698 | 0.988433250 | 3 | -0.073209732 |
| `cgtnjyggtm` | 2 | 1.005961820 | 1.027150502 | 0.984773138 | 2 | -0.228983254 |
| `w68dtmpfyf` | 2 | 0.999420372 | 1.027704351 | 0.971136394 | 2 | -0.245829004 |
| `xcmzfsbd9t` | 2 | 1.003435853 | 1.030176110 | 0.976695596 | 2 | -0.390263535 |
| `yfxyg8jm46` | 0 | 1.039519217 | 1.060746643 | 1.018291791 | 1 | -0.722451573 |
| `ykhs7s2dck` | 2 | 0.996096029 | 1.010339175 | 0.981852883 | 3 | -0.107337044 |

Across outer folds, source-only mean relative AUEBC was 1.006424544 overall,
1.025985247 for FIELD, and 0.986863842 for CAI. This was selection evidence,
not held-out-target evidence.

## Conditional Stages

The source-only AAWR prerequisite authorized CAI, but not FIELD, in five folds
(`74t7kcdgkr`, `cgtnjyggtm`, `w68dtmpfyf`, `xcmzfsbd9t`, `ykhs7s2dck`).
AAWR candidates were evaluated in those folds but did not beat the selected
soft-distillation actor. Outer `yfxyg8jm46` was `NOT_RUN_NOT_AUTHORIZED`.
Consequently every final model has an empty AAWR field and no privileged actor
input. STOP was `STOP_NOT_AUTHORIZED` for FIELD and CAI in all six folds, so the
frozen fallback is endpoint 0.25.

The six action-model hashes, six STOP-model hashes, selection hashes, and final
dependency hashes are recorded row-by-row in `model_manifest.csv`. No target
result changed the model family, route, DAgger round, AAWR state, comparator, or
STOP decision.
