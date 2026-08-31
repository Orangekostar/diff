# Agentic Task-Driven NDE Research Design

Date: 2026-08-31

## Authority and objective

The external operator specification, SHA-256
`44debf1e9d98e4dd56e77409a01790d2eb08da5c3cb9936ad006e2fd3143764e`,
is the approved design authority. The exact repository base is
`15db6edad14ef36364fbda17945ccc924f600e47` on the isolated branch
`research/agentic-task-driven-nde`.

The research question is whether a released impacted-surface image improves
strict held-out-domain observability of where ultrasound should be acquired,
when combined with the frozen downstream task-value engine. All evaluation is
offline replay. Hidden full experimental C-scans are the replay world, not
deployable inputs. Measurement cost is the frozen normalized native-raster
cost. No result may be described as hardware, scanner-time, or robot evidence.

The route is new. It does not revise, rescue, or reinterpret the frozen MVA,
MVD, MAVIS, AEI manuscript, or damage-to-failure-response science. A negative
gate is a complete result and permanently blocks its dependent stages.

## Options considered

1. Start from visual-model training and infer a surface-to-C-scan alignment.
   Rejected because it would make the scientific target choose the coordinate
   system and would invalidate the P0 leakage boundary.
2. Treat independently normalized surface and C-scan coordinates as aligned.
   Rejected because common normalized coordinates are not evidence of physical
   correspondence, scale, orientation, or offset.
3. First recover specimen authority and a source-supported, deterministic
   surface-to-action-space transform; train nothing unless that gate passes.
   Selected because it is the only route compatible with the prompt and the
   existing impact-center audit.

## Stage architecture

The implementation is a fail-closed stage-gated pipeline under the new
namespace `cmc_bbdm.agentic_nde`.

- P0 recovers exact surface/C-scan/CAI identities, hashes every authority,
  audits surface geometry, resolves orientation and physical mapping without
  reading hidden C-scan content or CAI, renders legal 8x8 actions, audits
  external repositories and literature, and decides registration authority.
- P1, only after `P0_GO`, tests whether surface evidence improves prediction of
  the frozen 64-cell mechanical value target after the geometry-neutral scout.
- P2, only after P1 authorization, tests task conditioning between CAI-error
  reduction and normalized-RGB-MSE field-reconstruction reduction.
- P3, only after P1 authorization and subject to the P2 claim scope, wraps the
  frozen acquisition engine and tests closed-loop rank fusion.
- P4, only after `P3_GO`, tests typed legal tool calls. Frozen or few-shot
  prompting precedes any SFT or reinforcement learning.

The first failed gate emits the complete stage package and records every
dependent stage as `NOT_RUN_NOT_AUTHORIZED`. Thresholds, controls, model
families, splits, and metrics are not relaxed after observing results.

## Namespace and ownership boundaries

New implementation files are restricted to:

- `src/cmc_bbdm/agentic_nde/`
- `scripts/run_agentic_nde.py`
- `paper_v3/configs/agentic_nde_p0.yaml` through
  `paper_v3/configs/agentic_nde_p4.yaml`
- `docs/AGENTIC_NDE_*.md`
- `results/agentic_task_driven_nde/`
- `artifacts/agentic_task_driven_nde/`
- `tests/test_agentic_nde_*.py`

Existing MVA, MVD, and MAVIS APIs may be imported. Their source and frozen
results are read-only. Raw surface images, external repositories, model
weights, caches, and large arrays remain outside Git; committed manifests bind
their logical dataset-relative paths, revisions, licenses, sizes, and SHA-256
identities.

## P0 authority model

Every primary row is keyed by exact `(domain_id, specimen_id)`. It must bind:

1. released impacted-surface relative path, file size, and SHA-256;
2. source internal-damage/C-scan relative path and SHA-256;
3. frozen registered C-scan crop identity and SHA-256;
4. frozen CAI row identity;
5. dataset ID and version;
6. an explicit surface-to-action-space transform record.

Order-based, visual, manual, fuzzy, and numeric-only pairing are forbidden.
External absolute paths are runtime inputs and are never serialized.

Surface QC records image format, pixel dimensions, channel mode, boundary
status, annotation status, orientation evidence, and physical extent evidence.
The transform hierarchy is:

1. direct source metadata or export transform;
2. deterministic geometry-only transform supported by source metadata;
3. source-only learned registration, isolated from the target domain, only if
   the first two are impossible.

No transform may use C-scan pixel values, C-scan damage masks, CAI values,
oracle values, target-domain labels, damage centroids, or manual target
alignment. A typed transform contains source and destination frames,
orientation, scale, offset, dimensions, physical extents, evidence class,
evidence hashes, status, and a deterministic hash of canonical parameters. The
same object maps points, boxes, cells, and the legal 8x8 grid.

P0 passes only if identity and hashes are exact, orientation is resolved, the
mapping is deterministic and deployable without hidden C-scan/CAI evidence,
each primary domain reaches at least 90% coverage, total coverage is at least
240 specimens, and the package is deterministically replayable. The preferred
coverage is all 276. Failure to establish a defensible transform yields
`P0_SPATIAL_REGISTRATION_NO_GO`; P1-P4 are not authorized.

## P1 observability design

P1 binds the frozen initial-state target rows in
`results/mva/a2_oracle_value/oracle_values.parquet` (SHA-256
`6b289f2f6f74ac75dde47ea7cbfefcda1c49f025e74227bfb34ef269182ff963`).
The target is the post-scout 64-cell mechanical task value, not a
zero-ultrasound target. Surface images are the only visual input; filename,
specimen ID, domain, CAI, impact metadata, C-scan, and target are prohibited.

The preregistered candidates are center, the old static representation, small
frozen dense encoders with low-capacity heads, and a frozen legal-grid VLM
diagnostic. Controls are the strongest compatible old static baseline selected
before targets, a deterministic cross-specimen shuffled surface donor, 8x8
patch permutation, and center. Rank fusion weights are selected from
`0, 0.25, 0.5, 0.75, 1` using source domains only.

Primary outcomes are next-action regret, one-step CAI utility, and static-plan
CAI AUEBC. Statistical comparisons use synchronized specimen-within-domain
bootstrap resampling with equal-domain aggregation. P1 requires all prompt
conditions, including lower AUEBC, a positive confidence interval for
improvement, at least four of six domain directions, at least 20% oracle-gap
closure, a positive comparison with shuffled surfaces, and failure of the
center control to reproduce the effect. Descriptive ranking alone is not an
authorization signal.

## Conditional P2-P4 design

P2 uses the same action and cost semantics for CAI-error reduction and
normalized-RGB-MSE reconstruction reduction. Correct-task, wrong-task,
task-agnostic, shared, and separate variants differ only in the registered task
condition. A multi-task claim requires the complete P2 gate; otherwise any
later route is explicitly CAI-specific or stops.

P3 is a wrapper around the frozen acquisition engine and its exact
`rollout_scout_and_focus_curve` semantics. It compares uniform, center, the
strongest frozen static reference, original task-only MAVIS, visual-only,
surface plus old static, surface plus dynamic, and oracle policies, with
correct/shuffled/no-surface controls. It must beat the strongest baseline,
task-only, visual-only, and shuffled controls, attain at least four of six
domain directions, and close at least 20% of the oracle gap.

P4 exposes only typed `scan_ultrasound`, `get_state`, and `stop_report` tools.
The agent sees the surface image, task text, deployable partial state, budget,
and legal tools. It never sees specimen/domain identity, true CAI, full C-scan,
oracle values, or hidden chain of thought. Frozen/few-shot P4A must satisfy
legality, task correctness, preregistered non-inferiority, no target tuning, and
correct-over-shuffled controls before any P4B training is authorized.

## Evidence, statistics, and replay

The specimen is the prediction row and the held-out domain is the inferential
unit. All learned preprocessing, model fitting, hyperparameter selection,
threshold selection, and rank-fusion selection are source-domain only.
Bootstrap comparisons are synchronized within domain and aggregated with equal
domain weight. Cells, time points, seeds, and folds are never treated as
independent evidence.

Every executed stage has exact package membership, a canonical config copy,
source hashes, `summary.json`, `REPORT.md`, `artifact_manifest.json`, and
`CHECKSUMS.sha256`. Replay rejects symlinks, missing/extra files, schema drift,
size drift, hash drift, or status inconsistency. A scientific rerun must be
byte-identical for deterministic records and numerically identical under the
fixed floating-point tolerance for arrays explicitly declared non-byte-stable.

## Failure behavior and manuscript boundary

No manuscript directory is created before `P3_GO`. A P0 or P1 failure permits
only audit artifacts and the final handoff. It does not authorize a larger VLM,
manual registration, new targets, new thresholds, alternate domain splits, or
paper language that implies a positive agentic system.

The following roots remain untouched relative to the base commit:

```text
results/mva/
results/mvd/
results/mavis/
results/mavis_science_closure/
results/p1_full_field_oracle/
results/p3_spatial_specificity/
results/p5_sparse_scan/
artifacts/mavis/
artifacts/mavis_science_closure/
artifacts/mvd_authority/
artifacts/mavis_authority/
artifacts/aei_information_hierarchy/
results/damage_to_failure_response/
artifacts/damage_to_failure_response/
paper_aei_information_hierarchy/
src/cmc_bbdm/mva/
src/cmc_bbdm/mvd/
src/cmc_bbdm/mavis/
```

## Testing and execution policy

Implementation follows red-green-refactor cycles. P0 tests cover authority
snapshots, exact pairing, surface QC, registration contracts, orientation,
boundaries, transform inversion, point/box/cell/grid rendering, leakage
sentinels, grid legality, artifact membership, checksum replay, and gate
transitions. Conditional stages add tests before their production code.

Baseline evidence at the exact base is:

- damage-response suite: 300 passed;
- reusable MVA/MVD/MAVIS contracts: 42 passed;
- frozen-path diff: empty.

Final verification includes focused tests, Ruff, `git diff --check`, package
replay, frozen-path diff gates, stage authorization review, local/upstream/
remote SHA equality, and a clean worktree.

## Self-review

- Placeholders: none.
- Scientific ambiguity: fail closed; no transform is inferred from outcome
  evidence.
- Stage transitions: every downstream stage has one explicit authorization
  source and one stop status.
- Scope: all frozen science and the AEI manuscript are excluded.
- Unsupported novelty claims: none; the literature ledger will report searched
  evidence and explicit unknowns.
- Unauthorized training: none before P1, and P1 itself requires P0 GO plus a
  hash-frozen protocol.
