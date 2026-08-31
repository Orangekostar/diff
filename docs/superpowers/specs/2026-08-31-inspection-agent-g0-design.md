# Inspection Agent G0 Design

## Objective

Build a deterministic, replayable opportunity audit for zero-ultrasound active
inspection. Preserve all historical science, expose a deployable observation
boundary, use privileged teachers only for opportunity measurement, and stop the
route if the registered gates fail.

## Components

`state.py` owns immutable levels, actions, union masks, exact cost, legal fitting
actions, and MVA equivalence. `contracts.py` owns task/decision vocabulary,
policy-visible observations, structured belief, and trajectory records.
`world.py` is the only causal reveal boundary and privately wraps
`MAVISAuthority`. It never returns authority/evaluation records.

`surface_hypothesis.py` computes the transparent P0R surface score.
`generalized_reconstruction.py` owns fold-source priors and reconstruction from
observed values only. `field_task.py` owns internal-signal and FIELD metrics.
`state_bank.py` owns deterministic label-independent state sequences.
`cai_assessor.py` owns fold-local state representations, PCA/Ridge, exclusion
audits, and authorization metrics.

`oracle.py` owns privileged discovery/FIELD/conditional-CAI candidate scoring
and records every candidate roster before selecting. `stopping.py` owns fixed
reference selection and sufficiency. `evaluation.py` maps trajectories to fixed
checkpoints, AUC/AUEBC, task swap, and domain rows. `statistics.py` owns paired
specimen-within-domain bootstrap. `artifacts.py` owns canonical serialization,
hashes, manifests, and replay comparison. `g0.py` orchestrates phases without
allowing later phases to mutate configuration or gates.

## Data flow

P0R surface authority and MAVIS hidden authority are joined by the exact 276-row
specimen roster inside orchestration. A policy receives a surface image/hash,
task, observation, and belief; it never receives the join key. The evaluator
retains the private key to obtain teacher labels after an action sequence is
frozen.

For each outer fold, source full scans fit one background prior. Label-independent
source policies create 19 rows per specimen: one zero anchor and 18 acquired-state
snapshots. Reconstructions are encoded by the frozen ResNet18; PCA-32/Ridge-10
fits repeated state rows with state-unique IDs.
Target fixed-policy states are then predicted. Only a complete assessor gate can
enable the CAI oracle.

Discovery and FIELD oracles use optimized local reconstruction deltas, but tests
compare them against full generalized reconstruction. The CAI oracle batches all
candidate reconstructions through one validated encoder session. Selected actions
alone pass through `CausalInspectionWorld.step`.

## Determinism and failure behavior

All orders, seeds, checkpoints, tie breaks, serialization columns, and bootstrap
settings live in the frozen YAML. Invalid hashes, target-source overlap, changed
old trajectory costs, leakage, nonfinite values, duplicate rows, or replay drift
raise an error and prevent a decision artifact. Conditional CAI outputs use the
literal status `NOT_RUN_NOT_AUTHORIZED` when the gate fails.

No result may trigger a new model, threshold, task, domain subset, or gate. The
final status is a pure function of registered summaries.
