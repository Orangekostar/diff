# MAVIS Science Closure Design

## Objective

Turn the frozen Tier-B MAVIS endpoint at commit `716de19` into a defensible
science package by separating representation, conditional valuation, and
planning evidence. The work is diagnostic and claim-closing, not a retuning of
P7 and not a generic performance search.

## Immutable boundary

All historical MAVIS, MVA, MVD, and authority artifacts are read-only. New
outputs live only under `results/mavis_science_closure/` and
`artifacts/mavis_science_closure/`. Every execution records the P7 tree state,
input hashes, cohort, domain order, and source-only fit roster. Oracle rows are
explicitly non-deployable and cannot replace any frozen checkpoint.

The policy-visible side may use only acquired ultrasonic values, acquired
positions, deployable context, legal action geometry, exact incremental cost,
and remaining budget. True CAI, future ultrasonic content, reconstruction
targets, and counterfactual teacher values remain evaluation- or teacher-only.

## Scientific decomposition

The closure preserves three separately testable layers:

1. Representation: whether measured specimen-specific UT enriches an evolving
   mechanics-relevant information state beyond position, budget, and shuffled
   content.
2. Valuation: whether conditional action values change as evidence accumulates
   and whether dynamic MRIS identifies useful actions better than static M1 and
   candidate-only controls.
3. Planning: whether stronger set-level planning converts fixed learned or true
   values into lower downstream CAI error under the exact acquisition budget.

P9-P11 close the first two layers. P12 performs controlled representation/value/
planner substitutions. P13 tests a joint-utility set planner. P14 contrasts
reconstruction- and mechanics-oriented acquisition. P15 tests whether value is
stable across reasonable strict-OOF CAI learners. P16 explains the observed
feedback result by linking action turnover and value evolution.

## Reuse-first implementation

P9 reuses P1 state-action teacher rows and P3 action scores. P10 reuses frozen
P2 state predictions. P11 reuses P3 and MVD O2 outputs. P14 reuses MVA A2
reconstruction-oracle, MVA A4 global reconstruction, and MAVIS mechanical
trajectories. P16 is an analysis join over frozen and closure-stage tables.

New code is limited to analysis/package modules in the existing MAVIS namespace.
It may add source-only diagnostic learners and planners, but cannot alter P7
training, selection, routing, or output code. Large P1 measurement payloads are
referenced by hash rather than copied.

## Statistics

Physical specimen is the resampling unit and held-out domain is the external
generalization unit. Aggregate metrics first reduce within specimen and then
weight the six domains equally. Paired bootstrap samples specimens within each
domain. State/action rows are never treated as independent replicates.

Primary directions are fixed before execution: lower CAI MAE/AUEBC/regret is
better; higher one-step utility, rank correlation, NDCG, and overlap are better.
All tables state the subtraction direction. Multiple diagnostic comparisons are
reported transparently rather than converted into an acceptance gate.

## Packages and replay

Each P9-P16 package contains machine-readable primary tables, domain summaries,
paired bootstrap or explicitly justified deterministic diagnostics, `REPORT.md`,
`summary.json`, `artifact_manifest.json`, and `CHECKSUMS.sha256`. The final
replay regenerates every derived file into `results/mavis_science_closure/replay/`
and requires byte identity. A completion audit verifies that every historical
P7 file hash is unchanged.

## Claim discipline

The final manuscript claim map may support only what the tables establish. The
work cannot claim first adaptive ultrasound, first ultrasound VoI, scanner-time
reduction, industrial deployment, causal field validation, or external
generalization. Oracle and substitution analyses diagnose ceilings and
bottlenecks; they do not describe deployable performance.
