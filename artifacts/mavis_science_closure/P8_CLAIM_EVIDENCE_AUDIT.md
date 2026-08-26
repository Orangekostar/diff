# P8 Claim-Evidence Audit

## Audit conclusion

The frozen Tier-B label is a conservative package status, not affirmative
evidence for every MAVIS component. B1 has mixed evidence, B2 and B3 are not
supported by their direct controls, and B4 is strongly supported as a boundary
finding. The defensible paper must therefore lead with a scientific
representation-valuation-planning diagnosis rather than an end-to-end
performance claim.

## Existing claims

| Claim | Direct evidence | Finding | Current status | Allowed wording | Forbidden wording |
|---|---|---|---|---|---|
| B1: real partial UT enriches CAI-relevant state | P2 equal-domain CAI AUEBC and paired bootstrap | Real beats static by `0.0122325582` AUEBC (static minus real; CI `[0.00407538, 0.02063618]`) but loses to positions-only by `0.0178288354` and reconstruction by `0.0373613302`; real beats shuffled only weakly (`+0.0040816280`, CI crosses zero) | `WEAK` | Real acquired content adds signal relative to a static state, but the benefit is not robust to stronger position/reconstruction controls | Real partial UT is broadly more informative; MRIS causally captures specimen mechanics |
| B2: dynamic state improves valuation over static M1 | P3 regret/utility with identical legal action rows | Static minus real regret is `-0.0000414755`, CI `[-0.0001854751, 0.0000947653]`; shuffled minus real is `-0.0001835832`, CI entirely below zero | `UNSUPPORTED` | The current dynamic scorer does not improve next-action valuation over static and shuffled controls | Dynamic MRIS improves measurement valuation |
| B3: feedback improves decisions | P4 feedback/no-feedback same-cost rollouts | No-feedback AUEBC `0.1250572774` is lower than feedback `0.1250722402`; no-feedback-minus-feedback is `-0.0000149628`, bootstrap CI approximately `[-0.0000194057, -0.0000107118]`; feedback is lower-error in only 2/6 domains | `UNSUPPORTED` | Feedback changes rankings, but the frozen implementation does not convert those changes into lower CAI error | Closed-loop feedback improves acquisition |
| B4: final performance is weak and heterogeneous | P7 frozen outer evaluation | MAVIS error exceeds strongest deployable baseline by `0.0000611421`; baseline-minus-MAVIS CI is entirely negative; MAVIS improves only 2/6 domains | `BOUNDARY_ONLY` with strong evidence | Frozen cross-domain closed-loop results expose an actionability/planning boundary | MAVIS outperforms the strongest baseline; MAVIS generalizes externally |

## Leakage and protocol audit

| Boundary | Evidence | Status |
|---|---|---|
| Policy cannot read target true CAI | Typed `PolicyContext`/`InspectionState`; P7 workers record `target_true_cai_used_by_policy=false`; invariance tests | PASS |
| Policy cannot read unacquired target UT | Action-bound reveal API; dynamic scorer accepts only `InspectionState`; future-content invariance tests | PASS |
| Teacher is strict OOF | P1 `teacher_fit_audits.parquet`; 30 registered fits exclude outer and query domains/specimen | PASS |
| Target domains cannot select models | P2/P3 model-selection audits and worker metadata | PASS |
| Target states cannot enter aggregation | P5 records `target_state_count=0` in all outer folds | PASS |
| Target outcomes cannot select fallback | P6 records `target_outcomes_used_for_selection=false` | PASS |
| Exact acquisition cost is shared | Native-grid action deltas and budget tests; all P4 methods use the same checkpoint contract | PASS |
| Shuffled control matches positions/cost | Registered donor mapping retains recipient positions, action history, and exact cost | PASS |
| Historical P7 is replayable | 31 files are byte-identical; checksums and tree state verified | PASS |
| Closure non-modification barrier | No closure package exists yet; must be implemented and tested | OPEN |

## Claim gaps that P9-P16 must close

| Proposed manuscript claim | Existing evidence | Missing evidence | Closure stage |
|---|---|---|---|
| Conditional mechanical value evolves with acquired evidence | P1 has conditional one-step values at multiple states | Same-action longitudinal rank/value change and controls | P9 |
| Real measured content enriches MRIS | P2 aggregate modes are mixed | Cost-resolved, specimen-paired accumulation and causal controls | P10 |
| Dynamic valuation identifies useful measurements | P3 shows no current advantage | Candidate-only/MVD/static comparison on one aligned action roster and cost strata | P11 |
| Representation, valuation, and planning are distinct bottlenecks | P4/P7 only expose final failure | One-component-at-a-time oracle substitution matrix | P12 |
| Individual values are not sufficient for measurement-set planning | Sequential oracle itself is weak | Joint-utility planner compared with greedy/current planner | P13 |
| Mechanics-oriented acquisition differs from reconstruction-oriented acquisition | Historical reconstruction and mechanics trajectories coexist | Same-cohort/same-cost 2x2 task outcomes and spatial overlap | P14 |
| Mechanical value is predictor-conditioned but scientifically stable | Only the frozen ridge CAI teacher is formal | Same splits/state bank across reasonable CAI learners | P15 |
| Feedback failure has a measurable mechanism | P4 only gives end-to-end contrast | Link value shifts/action turnover to beneficial and harmful trajectory changes | P16 |

## P7 claim boundary

The P7 package supports retrospective normalized-raster feasibility and a strong
negative/heterogeneous boundary result. It does not establish scanner-time
reduction, physical robot path feasibility, industrial deployment, prospective
benefit, or external-domain generalization. Science-closure diagnostics cannot
retroactively upgrade P7 deployable performance.
