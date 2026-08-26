# P8 MAVIS P1-P7 Stage Result Map

Audit base: `716de19c4fe6f742eddf40a95f8cc3ed45192217`. All
reported aggregate metrics weight the six held-out domains equally after
reducing to physical specimens.

| Stage | Frozen purpose | Cohort / split | Selected model or policy | Primary output | Actual result | State hash |
|---|---|---|---|---|---|---|
| P1 | Materialize causal sequential states and strict-OOF action values | 276 specimens, six LODO outer domains; every teacher excludes outer and query domains/specimen | Five frozen trajectories: random, uniform, reconstruction-driven, one-shot mechanical oracle, sequential mechanical oracle; six checkpoints | 8,280 states; 2,157,215 state-action rows; 1,380 terminal states | Complete causal state bank; state/action rows are training samples, not statistical replicates | `1cfc565b7c45ab558b4ae795c342b227f177c0a626c8064ece02368bb948cf18` |
| P2 | Test MRIS informativeness for partial-state CAI | Same cohort and nested LODO source-only selection | Shared MRIS/CAI protocol for real, positions-only, shuffled, static, reconstruction modes | 41,400 predictions; CAI AUEBC by mode/domain | Real `0.1250432019430529`; real beats static but loses to positions-only and reconstruction; shuffled contrast is uncertain | `d3dba6ec3621575c0eb20f58c9d041fb00050d11c23a0426b3da764bfcd09768` |
| P3 | Learn dynamic conditional mechanical action value | 6,900 decision states; outer target scores only after source-only selection | MRIS-conditioned action scorer with CAI, pairwise, listwise, and value losses | 1,725,772 action scores; regret, utility, Spearman, NDCG, Recall@K | Real regret `0.0122191085`; selected utility `-0.0000277945`; shuffled has significantly lower regret | `a16465bebc28f70a7662327bffd6f0a7efa2044daf91ff077e1e890620626f85` |
| P4 | Evaluate exact-cost scout-and-focus development rollouts | Same 276 specimens/checkpoints; target outcomes excluded from selection | Fourteen methods, including feedback/no-feedback, state controls, frozen baselines, and oracles | 159,140 trajectory rows; 23,184 predictions | MAVIS full `0.1250722401900199`; strongest deployable baseline MVD M1/O2 `0.12499204011570479`; no-feedback `0.1250572774283045` | `97325b81a4b3fe443ae9c099401b1f47a26bb12752e9af1a335faee7d08ef356` |
| P5 | Reduce state-distribution shift through on-policy aggregation | Three source-only rounds for each outer domain; zero target states | Aggregated dynamic policy checkpoint per outer domain | 129,072 source trajectory rows; 55,250 final states | Complete; `target_state_count=0`, `target_data_used_for_training=false` | `1fb711dcf1fc360c5db628b757c8384ae9adc02fa9d43ae10a07007df80cab0e` |
| P6 | Select confidence fallback without target outcomes | 1,380 nested source-specimen calibration pairs | Per-outer-domain baseline and first-decision confidence threshold selected on double-held-out source curves | selections, thresholds, calibration predictions | Complete; `target_outcomes_used_for_selection=false` | `78424f120583228ca11d8c9c0edaefb1342a60d0ac6d51f61665b4819b34233f` |
| P7 | Frozen outer evaluation and conservative claim tier | Configuration frozen; six outer domains evaluated once | Aggregated MAVIS, safe routing, fallback, deployable baselines, diagnostic oracles | 28,152 predictions and formal 31-file replayable package | MAVIS `0.12505318220938968`; MVD M1/O2 `0.12499204011570479`; MAVIS improves 2/6 domains; Tier B | `6bf473e811314b5c2897cdf789f71854ea9899bcbe13c8e16664ae32c03ed78c` |

## Frozen protocol

- Domain order: `74t7kcdgkr`, `cgtnjyggtm`, `w68dtmpfyf`,
  `xcmzfsbd9t`, `yfxyg8jm46`, `ykhs7s2dck`.
- Domain specimen counts: 45, 49, 43, 59, 42, and 38.
- Registered checkpoints: six exact native-raster budget checkpoints from the
  domain-specific quantized scout through the 25% endpoint.
- Model selection, early stopping, aggregation, baseline choice, and routing
  thresholds use source domains only.
- P2/P3/P4/P7 inference first aggregates within physical specimen; paired
  bootstrap resamples specimens within domains; domains receive equal weight.
- Historical outputs record dirty worktrees at generation time but bind runtime
  code/config/input state hashes. P7 is subsequently frozen by commit `716de19`.

## Required frozen observations

| Observation | Frozen result | Authority |
|---|---:|---|
| P2 positions-only minus real AUEBC | `-0.01782883542434792` | `results/mavis/p2_mris/summary.json` |
| P2 reconstruction minus real AUEBC | `-0.03736133017387293` | same |
| P2 shuffled minus real AUEBC | `+0.004081627987189113` | same |
| P2 static minus real AUEBC | `+0.012232558209541416` | same |
| P3 real next-action regret | `0.0122191085` | `results/mavis/p3_dynamic_voi/REPORT.md` |
| P3 shuffled minus real regret | `-0.00018358324056058466` | `results/mavis/p3_dynamic_voi/summary.json` |
| P4 no-feedback minus feedback AUEBC | `-0.0000149627617154` | `results/mavis/p4_closed_loop/aggregate_auebc.csv` |
| P7 baseline minus MAVIS AUEBC | `-0.00006114209368489` | `results/mavis/p7_final_frozen_eval/claim_evidence.csv` |
| P7 baseline-minus-MAVIS 95% CI | `[-0.00008460772463426673, -0.00003776607801288311]` | same |
| P7 safe minus fallback AUEBC | `+0.00000029143235326` error for safe vs fallback | same |
| P7 safe-control-minus-reference 95% CI | `[-0.000012679892274931746, +0.00001230794123002471]` | same |

## Replay status

`results/mavis/replay/summary.json` records 31/31 files and 7,891,077 bytes as
byte-identical. The replay tree SHA-256 is
`931dc86c26caf1c7246709c4706a7cd0428e3a1533b6ff1ad3c2ad8f9517d1e4`.
