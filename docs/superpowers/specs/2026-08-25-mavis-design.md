# MAVIS Causal Closed-Loop Inspection Design

## Objective

Build Mechanics-Aware Value-of-Information State Learning (MAVIS) as a new,
independent namespace over the frozen MVA/MVD evidence. MAVIS maintains an MRIS
from information actually acquired, estimates conditional mechanical utility of
legal measurements, and updates acquisition decisions after each reveal.

## Chosen approach

Use an upstream-regenerated, typed causal authority plus a modular supervised
closed-loop stack. Two rejected approaches are: reusing counterfactual MVD
candidate embeddings as observations, which leaks future scan content; and
placing full scans in a generic state object, which makes policy leakage
structurally possible. Reinforcement learning is excluded until supervised
dynamic imitation demonstrates a genuine long-horizon ceiling.

## Architecture

### Authority and contracts

`PrivilegedSpecimen` owns the hash-bound full scan and source/evaluation label.
It is never accepted by policy or encoder APIs. `MAVISAuthority` owns the cohort,
split metadata, and action-bound reveal operation. `InspectionState` contains
only deployable context, acquired native indices and RGB values, action history,
exact acquired count, and remaining count. Teacher APIs receive an explicit
`SourceTeacherView`; final metrics receive an `EvaluationView` only after a
rollout is complete.

The initial state contains no ultrasonic values. A registered geometry-neutral
uniform scout produces the first measurement state. Historical MVD initial
embeddings are treated as post-scout controls and all comparisons pay scout cost.

### State and mechanics model

The first MRIS encoder is a small permutation-invariant DeepSets model. Each
measurement token contains normalized native coordinates and RGB values; a
separate context vector contains registered deployable metadata, action mask,
acquired fraction, and remaining fraction. Sum/mean pooled measurement features
are fused with context. The mechanics head predicts registered CAI and is fitted
strictly within source-domain folds.

Five state variants share the same split and training code: static context,
position-only, real partial measurements, within-stratum shuffled measurements,
and the existing reconstruction-state control. Position-only objects cannot
store RGB arrays. Shuffled content retains positions and cost but records a
different donor specimen.

### Dynamic teacher and policy

For a source state and legal action, conditional teacher value is the reduction
in strict-OOF CAI loss after revealing that action. Full source scans and source
true CAI are permitted only here. Each specimen's evaluator is trained without
that specimen and without the current outer domain.

The scorer consumes MRIS, candidate geometry, exact incremental cost, and
remaining budget. Training combines CAI auxiliary loss, pairwise preference,
listwise/top-k ranking, and continuous value auxiliary loss. Selection is among
legal actions only, with exact cost guards; candidate count is never treated as
cost. A no-feedback ablation freezes the post-scout ranking.

### Rollout and on-policy aggregation

The rollout sequence is context -> uniform scout -> encode -> score legal
actions -> acquire -> reveal -> update -> repeat until the checkpoint or no
action fits. Source state banks initially combine random, uniform,
reconstruction-driven, one-shot mechanical-oracle, and sequential
mechanical-oracle trajectories. Later rounds roll out the learned source policy,
label visited source states with the privileged source teacher, append them, and
retrain. Outer target states are never relabeled or used for selection.

### Safe policy

Source-only validation selects both the fallback baseline and uncertainty
threshold. Low-confidence states use the strongest source-selected robust
baseline; no target result may change either choice. Report risk-coverage,
fallback frequency, aggregate/worst-domain CAI AUEBC, and per-domain effects.

## Data products

`artifacts/mavis_authority/` contains immutable manifests, hashes, compressed
measurement payloads or upstream-bound references, and no policy model output.
State manifests and state-action pairs use Parquet with array payloads stored
once by content hash. Every row binds specimen, domain, state, exact cost,
trajectory, authority, split, evaluator, and model hashes.

Each formal result package records git/config/authority/state-bank/checkpoint
hashes, cohort and splits, normalization, model-selection audit, per-state and
per-specimen outputs, domain and aggregate metrics, bootstrap, runtime, exact
costs, action trajectories, manifest, checksums, report, and deterministic
replay.

## Evaluation protocol

Development and hyperparameter selection are nested inside the five source
domains for each outer target. The lexicographic criterion is source validation
CAI AUEBC, improved source domains, worst-source-domain AUEBC, then model
simplicity. The final outer evaluation is run once after configuration freeze.

Experiments E1-E10 follow the controlling prompt. Primary endpoints are CAI MAE
versus exact cost for MRIS, next-action regret/one-step utility for dynamic VoI,
and CAI AUEBC for closed-loop acquisition. Statistical units remain physical
specimen and held-out domain. Claim strength is Tier S, A, or B; model
performance never changes the paper identity.

## Error handling and hard barriers

Loading fails closed on missing/hash-mismatched inputs, cohort drift, duplicate
specimens, invalid shapes, non-finite values, split contamination, or artifact
manifest drift. Reveal rejects illegal, repeated, or over-budget actions.
Policy-visible types do not expose full scans or labels. Invariance tests mutate
all unacquired target pixels and target CAI and require identical current state,
scores, and next action.

Implementation stops only for abnormal repository state, missing authoritative
input, inability to establish causal reveal, or leakage that cannot be excluded.
Scientific performance affects claim tier and diagnostics, not project identity.

## Verification

Development uses strict red-green-refactor TDD. The fifteen named MAVIS barrier
tests are mandatory, existing MVA/MVD regression remains green, and every stage
ends with Ruff, scoped/full Pytest as appropriate, `git diff --check`, checksum
verification, replay comparison, and a non-squashed scientific-stage commit.
