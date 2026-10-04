# CAI Order Mechanism Analysis Design

## Authority

This design implements `CODEX_CAI_ORDER_MECHANISM_FINAL.md` from
`CAI_ORDER_MECHANISM_CODEX_FINAL_PACKAGE.zip`. The packaged
`ORDER_MECHANISM_SCOPE.json` is copied verbatim and is the machine-readable
authority. The execution is bound to source commit
`37b3c40414c00c6633b64656c4bbb2b178ef9848` and the existing
`research/cai-vlm-agent-v3-controlled-reuse` branch.

The task is frozen-policy analysis, not training or model selection. The actor
checkpoint is hash-checked but never loaded or called. The only permitted
model forward is the selected W2 `MEAN_SC` predictor at update 1750, and only
for Track B VALID-prefix replay and its prescribed endpoint checks.

## Outcome

The branch gains one isolated, resumable analysis pipeline with two explicitly
separate result families:

1. `ARCHIVED_METHODS` derives action-event and four-stage contribution evidence
   for Proposed (`NO_VLM_SPATIAL_FEEDBACK`) and five fixed controls from the
   immutable 500-row episode archive, with no new forwards.
2. `FIXED_SET_ORDERS` holds each Proposed final cell set fixed and recomputes
   every prefix under native, reverse, and five deterministic label-free
   permutations using the frozen common predictor.

The pipeline produces all required tables, figures, three fixed case studies,
offline HTML, Chinese and bilingual handoff documents, acceptance evidence,
and a release manifest. Scientific effect direction is reported as observed
and never gates technical completion.

## Architecture

`python -m scripts.cai_order_mechanism.run` is the only public entry point. It
supports `prepare`, `derive`, `plan`, `infer`, `analyze`, `report`, `verify`,
and `all`, plus `--resume`. Git publication remains an external Codex step
after local verification.

The implementation is divided by responsibility:

- `run.py`: CLI, validated phase ordering, phase signatures, atomic task state,
  bounded resume, thread/runtime locking, and resource ledger orchestration.
- `inputs.py`: scope validation, single-pass source hashing, 50-VALID cohort
  locking, archive parsing and cross-file identity checks, six-shard thin
  feature loading, checkpoint binding, and fixed bootstrap reindex validation.
- `orders.py`: pure final-set extraction, reverse order, namespace-bound SHA256
  permutations, plan locking, mask/pixel clocks, and cache-key construction.
- `replay.py`: trial reproduction, common batched prefix engine, atomic cache
  commits, evaluated-row accounting, endpoint invariance, and uncached QA.
- `analysis.py`: float64 archive/order metrics, event identities, repeat and
  domain aggregation, frozen paired bootstrap, cap/event curves, and matched
  quality calculations.
- `reporting.py`: deterministic CSV/NPZ/JSON artifacts, at most five figure
  families, three prescribed cases, offline HTML, and handoff documents.
- `verify.py`: Q1-Q10 acceptance checks, output inventory and hashes, resource
  reconciliation, figure/HTML checks, and release manifest generation.

No existing scientific module is modified. Pure established contracts such as
`NativeCellGrid`, `MeanSCPredictor`, and checkpoint loading remain authoritative;
task-specific aggregation and safety rules live only in the new package.

## Data Flow And Scientific Semantics

`prepare` validates the branch ancestry, exact scope, immutable source hashes,
actor and predictor locks, job selection, the 50 VALID identities, 48 capture
groups, six domain counts, feature bindings, and runtime. Large immutable files
are hashed once and their bindings are persisted for downstream phases.

`derive` filters exactly the six authorized methods to 500 rows, parses all
semicolon trajectories, validates action uniqueness and trajectory invariants,
and emits the self-contained archived cohort and events. Stage membership uses
right-closed intervals via `searchsorted(edges, completion, side="left") - 1`.
For each action, raw gain is `e_before - e_after` and weighted gain is
`(1 - completion_cost / 0.25) * raw_gain`. Per-trajectory raw and weighted
identities are checked to `1e-8`. RANDOM losses and metrics are averaged across
its five repeats within specimen before domain-equal aggregation.

`plan` extracts each Proposed terminal set without labels, then atomically locks
native order, reverse order, and five SHA256-sorted permutations under namespace
`CAI_ORDER_MECHANISM_R1_37B3C404|PERMUTED_V1`. Duplicate permutation orders are
retained without redraw. The order plan and its sidecar SHA256 exist before any
predictor call.

`infer` uses integer cumulative native pixels divided by total pixels as the
Track B clock. It first recomputes three fixed native trajectories with the
default batch engine and requires archive prediction agreement within `1e-4`
and cost agreement within `1e-12`. One batch-size-one diagnostic attempt is
permitted only after failure; a second mismatch blocks the run. The chosen
engine is then locked and all 50 native trajectories must reproduce before
order effects are analyzed.

Prefix features keep the fixed row-major 64-token tensor and vary only the
boolean measured mask plus float32 cost. Cache identity binds protocol hash,
predictor hash, feature-shard hash, engine signature, specimen key, 64-bit mask,
and raw float32 cost bytes. Cache commits are atomic per completed batch/case.
The resource ledger records intended rows before each forward and completion
afterward, counting retries, padding, and QA against the 7000-row ceiling. The
nominal 350 scenarios contain at most 5950 prefix requests. Endpoint invariance
is checked for all 350 trajectories, followed by 12 uncached native/reverse
terminal calls selected by the fixed per-domain hash rule.

`analyze` keeps archive and intervention clocks separate. The primary order
effect is mean-of-five permutation area minus native area; the secondary effect
is reverse area minus native area, so positive values favor native order. Stage
contribution contrasts use Proposed minus control, so positive values favor
Proposed. Error contrasts use comparator minus Proposed/native, so positive
values favor Proposed/native. Every output row carries an explicit
`difference_definition`.

Area and stage estimates aggregate repeat mean, then specimen mean within each
domain, then equal mean across six domains. Endpoint MAE/RMSE/R2 retain their
specified physical-specimen estimands. Confidence intervals reuse and strictly
reindex the fixed 5000 capture-group-within-domain paired bootstrap matrix;
missing or inconsistent weights are fatal and are never regenerated. Quality
targets are integers 41 through 61 plus the fixed full-reference anchor, with
separate five-cap and full-event grids, first crossing, recrossing, unreached,
and negative-savings states preserved.

## State, Recovery, And Failure Policy

Each phase signature includes the canonical scope hash and hashes of all direct
inputs. A phase is reusable only when its signature and recorded output hashes
match. Output writes use temporary siblings, flush/fsync, and atomic replace.
`--resume` may recover one incomplete inference attempt; completed scientific
records are immutable. Reports may be rendered initially and once more only for
a cache correction, and both passes are cache-only.

Forbidden access or compute, source drift, TEST identity exposure, plan mutation,
bootstrap mismatch, prefix reproduction failure, endpoint disagreement, resource
cap exhaustion, or missing required output produces a hard failure. No fallback
changes model selection, creates bootstrap weights, substitutes archived
predictions for Track B, or silently drops a specimen.

## Reporting And Verification

All tables include the applicable identity family, cohort, clock, method/display
label, variant/repeat, units, aggregation, and difference definition. Figure
families compare only scientifically compatible estimands; case figures show
native, reverse, and permutation 0 for the three fixed specimens and never
select a visually favorable permutation. PNG files are rendered at at least
300 dpi and paired with editable SVG files. HTML is fully offline and links
every authoritative table and figure.

Development uses focused synthetic tests with no real model calls. Q1-Q10 cover
scope/input locks, archive parsing, stage identities, deterministic orders,
pixel clocks, cache identity, trial/fallback state, aggregation/bootstrap,
report inventory, and acceptance/resource invariants. Final verification runs
the single authorized test file once, Ruff on the new module and test, local
artifact checks, offline browser rendering, a contact sheet plus three key
figure inspections, and Git diff/scope review. Publication then uses selective
Git add, a content commit and push, remote SHA/sample checks, followed by a
separate publication receipt commit and push without PR, merge, force push,
or tag.
