# P8 Open Scientific Gaps

## Priority gaps

| ID | Open question | Why P7 cannot answer it | Available frozen inputs | Required result | Risk |
|---|---|---|---|---|---|
| Q1 | Does the true value of the same candidate change as UT evidence accumulates? | P3 reports state-averaged ranking metrics, not longitudinal same-action evolution | P1 state/action teacher values; P3 scores; five trajectories; six checkpoints | P9 value-evolution rows, rank stability, top-K overlap, best-action turnover, value shift | Moderate join/provenance risk; no new training required |
| Q2 | Does specimen-specific measured content enrich MRIS beyond positions and shuffled content? | P2 aggregate results are contradictory and can hide checkpoint/domain structure | P2 41,400 frozen predictions and common state IDs | P10 state-cost and per-specimen paired contrasts with bootstrap | Low compute; interpretation likely remains heterogeneous |
| Q3 | Does dynamic MRIS improve useful-action identification over static M1/candidate-only/MVD O2? | P3 omits a fully aligned candidate-only/MVD comparison and current shuffled result is adverse | P3 1,725,772 scores; MVD O2 predictions; P1 legal action rows | P11 aligned regret/utility/rank tables and cost strata | Candidate-only source fit may be required; must remain strict OOF |
| Q4 | Which of representation, valuation, or planning limits final CAI AUEBC? | P4 changes multiple components across methods | Frozen representations, predicted/true values, exact-cost rollout, downstream CAI endpoint | P12 one-component substitution matrix | Oracle semantics must be explicitly non-deployable |
| Q5 | Is greedy point-value planning the limiting abstraction? | Sequential point oracle also underperforms, but no joint set objective is tested | P1 counterfactual values, action masks, downstream teacher predictions | P13 greedy vs joint beam/lookahead under identical budgets | Joint utility must be defined without silently summing point values |
| Q6 | Are mechanics-optimal and reconstruction-optimal measurements different? | P4 reports both outcomes but not a controlled 2x2 objective-policy analysis | MVA A2 oracle, A4 global reconstruction mask, MAVIS mechanical trajectories | P14 same-cost task matrix and spatial overlap | Historical source schemas differ and require a hash-bound adapter |
| Q7 | Is action value stable across reasonable CAI predictors? | Formal value uses one PCA+Ridge family | Frozen state bank/splits; existing simple learner implementations | P15 multi-learner rank/top-K/best-action/region/utility agreement | Highest compute and leakage risk; target selection is forbidden |
| Q8 | Why does feedback hurt? | P4 only records aggregate feedback/no-feedback outcome | P4 trajectories plus P9 value/rank evolution | P16 turnover/outcome association by checkpoint/domain | Explanatory association only; not causal intervention evidence |

## Reuse map

- P9: `results/mavis/p1_state_bank/state_manifest.parquet`, partitioned
  `state_action_pairs/`, and P3 action scores.
- P10: P2 `state_predictions.parquet`, state/domain metrics, and bootstrap.
- P11: P3 action/state tables plus `results/mvd/` O2 observability outputs.
- P12-P13: current exact-cost candidate construction, reveal, teacher, scorer,
  policy, and rollout code; no P7 checkpoint may be replaced.
- P14: historical MVA A2 reconstruction-oracle trajectories, A4 global
  reconstruction mask, and registered MAVIS mechanical trajectories.
- P15: registered P1 state bank, source folds, CAI labels, and existing simple
  learner code where its protocol can be made identical.
- P16: analysis-only join of P4 trajectories and P9 value-evolution rows.

## Missing implementation

1. A hash-bound science-closure contract and package writer.
2. Longitudinal same-candidate state/action joins and value-evolution metrics.
3. Aligned candidate-only and historical MVD valuation adapters.
4. Non-deployable substitution execution with explicit component labels.
5. A deterministic joint-utility set planner with exact-budget guards.
6. An A4 global reconstruction-mask adapter in `HistoricalPolicySource`.
7. Same-split source-only learner adapters for the value-stability audit.
8. Closure replay and a complete historical-P7 non-modification verifier.

## Recommended execution order

Execute P9-P11 first because they reuse frozen predictions and determine whether
representation or valuation has any defensible signal. Then execute P12-P13 to
localize actionability/planning limitations, P14-P15 to establish task and
predictor specificity, and P16 last because it consumes P9 and P4. No new
representation model should be trained before attribution justifies a targeted
extension.

## Resource envelope

The current repository contains about 1.1 GB of MAVIS results. P9-P16 should
reference the 846 MB revealed-measurement payload and 149 MB state-action shards
in place. Expected new tables/checkpoints are approximately 2-8 GB, with P15 the
dominant runtime. Available local capacity at audit time was sufficient (about
84 GB free, 64 CPUs, three A40 GPUs), but package writers must reject accidental
payload duplication.
