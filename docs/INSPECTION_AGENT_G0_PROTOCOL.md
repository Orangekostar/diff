# Inspection Agent G0 Protocol

Status: preregistered before formal target evaluation.

## Question and scope

G0 is an opportunity audit for zero-ultrasound autonomous inspection. It asks
whether initialization, allocation between BROADEN and REFINE, task identity,
and specimen-specific STOP decisions contain measurable headroom. G0 does not
train a planner, VLM, LLM, policy Transformer, or reinforcement-learning agent.
The only learned object permitted is a metadata-free `StateCAIAssessor`.

The primary observation contains surface RGB, task, scanner lattice, acquired
positions and values, exact normalized budget, and structured belief. It excludes
true CAI, full/future C-scan, domain and specimen identity, impact history,
laminate metadata, ply count, `metadata13`, and `profile_stats21`.

## Cohort and split

The authorized P0R roster has 276 physical specimens in six Hasebe domains and
uses the author-attested ROT90 correspondence. Every learned transform and
parameter is fitted with leave-one-domain-out source data. The outer target is
absent from fitting, normalization, source-prior construction, and selection.
The inferential unit is the physical specimen followed by equal-weight domain
aggregation.

## Zero state and exact cost

`GeneralizedMeasurementState` has 64 levels in `{-1,0,1,2}`. State `-1` is never
measured; 0, 1, and 2 reuse the frozen MVA cell lattices. The initial state is
all `-1`. Legal actions are only `-1->0`, `0->1`, and `1->2`. Cost is the change
in the union of unique native-raster positions, including shared cell boundaries.
Executing all 64 `-1->0` actions must exactly reproduce the frozen MVA scout
mask. All level 1 and all level 2 must reproduce the existing level-1 mask and
the full native raster respectively.

## Surface hypothesis and belief

For each P0R surface cell, saliency is mean absolute RGB deviation from the full
surface image border median, min-max normalized within specimen. The eight
highest cells form the fixed surface hypothesis. No C-scan or CAI label enters
this computation. It is an inspection hypothesis, not damage probability.

The structured belief stores task, hypothesis cells/scores, observed cells and
levels, internal evidence summaries, current estimate/uncertainty, unexplored
fraction, decision, confidence, reason code, and state hash. Free-form reasoning
or chain-of-thought is never stored.

## Source-safe reconstruction

For each outer fold, compute each source specimen's full-C-scan border median,
average within source domain, then average the five source-domain means equally.
Round to uint8 using NumPy round-to-even and clip to `[0,255]`. Unmeasured cells
use this constant background. Measured cells use frozen MVA bilinear lattice
interpolation with deterministic half-open cell ownership; the last row/column
is closed. Every observed pixel is restored exactly. The reconstruction API does
not accept a target full scan.

## State CAI assessor

Each state is represented by the frozen 512-D ResNet18 reconstruction embedding,
effective budget, observed-cell fraction, and mean observed level. PCA dimension
32 and Ridge alpha 10 are fixed. Training uses one zero-state source anchor plus
18 label-independent acquired states per source specimen: three action-count
snapshots from each of `UNIFORM_BROADEN`,
`CENTER_BROADEN`, `RANDOM_BROADEN`, `SURFACE_FOCUS`,
`UNIFORM_THEN_REFINE`, and `ALTERNATE_BROADEN_REFINE`. Thus every specimen has
19 rows and equal total weight. The zero anchor prevents zero-budget performance
from being an extrapolation artifact. CAI-oracle states are forbidden from
assessor training.

CAI planning is authorized only when the fixed 25% state improves over zero-state
CAI MAE, the synchronized paired 95% CI lower bound is positive, at least four of
six domains improve, replay is valid, and every outer-target exclusion check
passes. Otherwise CAI planning and task-swap analysis are marked
`NOT_RUN_NOT_AUTHORIZED`.

## Policies and oracles

Initialization compares deterministic specimen-seeded RANDOM, geometry-spread
ZERO_UNIFORM, CENTER_FIRST, SURFACE_FOCUS, and privileged ORACLE_DISCOVERY.
Discovery uses cumulative acquired internal-signal saliency, defined as RGB
absolute deviation from the hidden full-C-scan border median divided by total
full-field saliency. It is not damage ground truth.

The fixed hierarchical comparator is `SURVEY_THEN_REFINE_FIXED`: all fitting
`-1->0` actions in uniform order, then `0->1`, then `1->2`; a nonfitting action
is skipped without changing order. `FIXED_UNIFORM_THEN_MAVIS` prepends the exact
zero-to-scout uniform sequence to the frozen historical `mavis_full` trajectory.
It is a metadata-augmented historical baseline, not the primary deployable input.

ORACLE_FIELD scores every legal fitting candidate by normalized-RGB-MSE reduction
per exact added pixel. Conditional ORACLE_CAI scores absolute-CAI-error reduction
per exact added pixel. Ties use objective, raw value, lower cell ID, then lower
target level. Full C-scan and true CAI are teacher/evaluation privilege only.

## Endpoints and tests

G0-A ends at the exact old-scout union. Its primary measure is capture AUC over
the exact normalized budget axis divided by that specimen's scout endpoint.
G0-B through G0-D end at normalized budget 0.25. Task AUEBC uses nominal points
`0, 0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25`; each point uses the last
fitting state and trapezoidal integration on `[0,0.25]`.

If FIELD and CAI are authorized, wrong-task swap evaluates FIELD trajectories on
CAI and CAI trajectories on FIELD at common budgets. Task conditioning requires
the correct trajectory to beat the wrong trajectory for both tasks, with positive
paired lower CI and at least four improving domains per task.

Stopping selects the strongest fixed nonprivileged 25% reference per outer fold
using source-only endpoint loss. Sufficiency is the earliest trajectory state
whose task loss is at most 1.05 times the same specimen's fixed-reference loss.
Reported saving is normalized measurement saving, never scanner-time saving.

## Statistics and gates

All contrasts pair methods on the same specimen. For each of 100,000 bootstrap
replicates, specimens are resampled within every domain using seed 2026083102;
the six resampled domain means are then equally averaged. Actions, cells, states,
timesteps, and random seeds are not independent samples.

The exact initialization, hierarchical, task-conditioning, stopping, and final
decision gates are those encoded in
`paper_v3/configs/inspection_agent_g0.yaml`. Gates and task definitions may not
change after target results are generated. A negative result terminates the
corresponding route and remains a committed result.

## Determinism and replay

The formal package records source hashes, P0R authority, roster, source priors,
outer splits, assessor identities, candidate rosters, selected actions, state
hashes, exact budgets, task losses, bootstrap effects, and decision. Replay must
recompute the package and match every deterministic formal artifact byte-for-byte.
Frozen MVA, MVD, MAVIS, P0R, and P1 paths are read-only throughout G0.
