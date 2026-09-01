# Inspection Agent G1 Protocol

## Registration

The controlling configuration is
`paper_v3/configs/inspection_agent_g1.yaml`, frozen against prompt SHA-256
`37b7e4b9860dd338589cf2d9dd8f2cc8d202451a84eeafbc4961b2c5a28fcda5` and base
`7a10cd425de582fa158bf6639285731ccd8ff7a7`. This protocol cannot be revised from
outer-target outcomes.

## Question and Inputs

G1 asks whether privileged G0 decisions are observable to a deployable structured
policy. Actor inputs are limited to the frozen 512-D embedding of a source-safe
current reconstruction, observable acquisition scalars, current-cell measurements,
surface hypothesis, task token, scanner geometry, and legal candidate costs.
Forbidden actor inputs include IDs, metadata context, full/future C-scan, true CAI,
true loss, teacher utilities, target outcomes, and authority/evaluation objects.

## Geometry and Actions

All 276 specimens use `build_acquisition_grid(height,width,0.015625)`. Every primary
trajectory begins with cells `0,63,7,56,27,38,3,24` at level 0, then selects among
192 canonical primitive slots until 0.25. FOCUS is unmeasured surface-top-eight,
BROADEN is another unmeasured cell, REFINE is a measured cell transition, and STOP
comes only from the separately authorized sufficiency head. K4/K16 are fixed
sensitivity runs and never select K.

## Splits

The outer protocol holds out one of six Hasebe domains. For each labeled source
inside an outer fold, both teacher prior and CAI assessor exclude the outer target
and labeled source. Inner leave-one-source-domain-out validation controls every
algorithmic selection. Final dependencies, model, and threshold are refit on all
five sources and frozen before the target rollout.

## State and Teacher Generation

Generate 13 label-independent states per specimen/task: post-K8 plus three
action-count snapshots from UNIFORM_CONTINUE, SURFACE_FOCUS_CONTINUE,
RANDOM_CONTINUE, and ALTERNATE_BROADEN_REFINE. Add the last fitting cross-fitted
oracle state at or below 0.0625, 0.125, 0.1875, and 0.25, collapsing exact
duplicates. Query every legal candidate for FIELD or CAI and retain cost, raw and
objective value, post-action loss, state hash, and decision type. Generate source
STOP labels against a fixed method selected on the other four domains.

Actor tensors and privileged teacher columns are physically and logically
separated. Manifests bind all fold rosters, source scan hashes, dependency/state
hashes, candidate counts, software versions, and generator command.

## Learning

Compare SharedActionMLP with the under-one-million-parameter structured Transformer,
hard BC with soft utility distillation, teacher temperatures, registered learning
rates/weight decays, and DAgger iterations 0/1/2. DAgger visits only outer-source
worlds and adds at most 16 deterministic quantile states per specimen/task/iteration.
AAWR-style training runs only if its source-validation authorization rule passes.
STOP is independently thresholded on source validation and otherwise disabled.

## Controls and Baselines

Fixed gate-eligible methods are RANDOM, ZERO_UNIFORM, CENTER_FIRST, SURFACE_FOCUS,
and SURVEY_THEN_REFINE_FIXED. Learned controls are HARD_BC, SharedActionMLP,
NO_TASK, NO_SURFACE, and SHUFFLED_SURFACE. WRONG_TASK is evaluated without tuning.
Privileged ORACLE_FIELD/ORACLE_CAI bound the bridge gap. Historical
FIXED_UNIFORM_THEN_MAVIS is metadata-augmented and not gate eligible.

## Formal Target Order

For every target: load frozen dependencies/model/threshold; create a causal world;
apply K=8; issue only observable policy state; mask illegal actions; choose STOP or
one legal action; reveal only the new native positions; repeat; seal trajectories,
scores, STOP scores, states, acquired values, and hashes; only then open target full
scan/true CAI for evaluation. Pre-seal truth access is an error.

## Metrics and Inference

Primary outcomes are FIELD normalized-RGB-MSE AUEBC and CAI absolute-error AUEBC
from zero/warm through 0.25. Fixed, learned, and oracle use identical geometry,
warm start, checkpoints, carry-forward, and evaluators. Gap closure is
`(L_fixed-L_learned)/(L_fixed-L_oracle)` when the denominator is positive.
Diagnostics include top-1/top-5/NDCG/regret and transition proportions but are not
primary evidence.

Pair methods per physical specimen, synchronously bootstrap specimens within each
domain 100,000 times with seed 2026090104, then average six domain effects equally.
Report task/surface controls and AGREE/PARTIAL/MISLEADING strata. STOP reports saving,
loss ratio, premature rate, and domain direction.

## Gates and Replay

Each task must beat its strongest source-selected fixed baseline with positive
paired CI lower bound, at least 4/6 improved domains, at least 20% oracle-gap closure,
no leakage, and valid deterministic replay. Task conditioning and STOP use their
separate registered gates. Formal/replay packages rebuild dependencies, teacher/model
identities, selection, rollouts, metrics, bootstrap, and decision. Negative results
are retained; target-triggered rescue is forbidden.
