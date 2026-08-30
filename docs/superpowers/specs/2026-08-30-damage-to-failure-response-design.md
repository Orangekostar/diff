# Damage-to-Failure Response Research Design

Date: 2026-08-30

## Authority and objective

The external operator specification, SHA-256
`bbdc4e26e70dcf22cc6a34186064b3924804c40ac3e78cd113cad72f9cffe44d`,
is the approved design authority. The exact repository base is
`3951f71f28b6efdf8c74eea0fe274b2a78a9cd57`.

The research question is whether pre-CAI post-impact observations contain
strictly cross-domain information about later compression response beyond
ultimate CAI strength. A positive outcome must be earned through fixed stage
gates. A negative outcome is a complete, publishable audit result and stops all
unauthorized downstream work.

## Options considered

1. Import the historical full code tree. Rejected because the available full
   tree has no Git metadata and cannot be treated as a code authority wholesale.
2. Build a focused `cmc_bbdm.damage_response` package in the compact repository,
   reading raw data through explicit external paths and SHA-256 manifests.
   Selected because it isolates new logic, preserves frozen Paper 1 paths, and
   makes every stage replayable.
3. Run ad hoc notebooks against the raw tree. Rejected because notebook state
   would weaken leakage controls, stage gating, and deterministic replay.

## Architecture

The implementation is a stage-gated command-line pipeline. P0 contains only
authority discovery, exact identity pairing, raw trace decoding, unit audit,
published-peak reconciliation, source hashing, and artifact replay. Model code
is not created or executed until P0 has answered the ten required authority
questions and emitted `P0_GO`.

If P0 passes, P1 derives preregistered response endpoints and tests response
richness under strict leave-one-domain-out evaluation. If P1 passes, P2 uses
fold-local feature processing and low-capacity models to test the incremental
value of spatial observations. P3 functional-curve work is conditional on P2.
P4 is not authorized by default; P5 is conditional on a P2/P3 signal.

The package boundaries are:

- `authority`: regular-file snapshots, SHA-256 identities, external root
  declarations, and fail-closed source validation.
- `pairing`: canonical specimen/domain identity joins only; no row-order,
  image-order, or fuzzy matching.
- `raw_cai`: raw channel decoding and trace quality control without scientific
  interpretation of unresolved units.
- `targets`: physical conversion, published-peak reconciliation, endpoint
  derivation, and scope labels.
- `feature_views`: explicit deployable, privileged, target, and forbidden input
  roles.
- `nested_eval` and `functional`: conditional P2/P3 estimators with source-only
  fitting.
- `gates`: immutable stage decisions and authorization transitions.
- `artifacts`: atomic result packages with exact membership, checksums, and
  byte-identical replay.
- `pipeline` and `scripts/run_damage_response.py`: orchestration only.

## Data flow and leakage boundary

External raw files remain outside Git. P0 snapshots their identities and emits
compact CSV/JSON/Markdown records. A row enters the primary 276-specimen cohort
only when canonical specimen ID, source domain, and raw-file SHA are all
available. Published workbooks supply specimen dimensions and reference peak
strength. Raw load is converted using the documented global calibration;
stress additionally requires measured width and thickness. Strain-dependent
endpoints remain disabled unless an independent unit audit resolves both unit
and sign.

Pre-CAI surface, C-scan, laminate, and geometry variables are eligible inputs.
Impact energy and impactor are privileged sensitivity inputs only. CAI traces,
derived targets, true peak strength, post-CAI images, and target-domain-fitted
transformations are prohibited inputs. Sentinel tests must fail if any of these
boundaries are crossed.

## Statistical and gate design

The specimen is the prediction row and the held-out source domain is the
inferential unit. Primary aggregation is equal-domain. Comparisons use
synchronized within-domain bootstrap samples. Time points, gauges, seeds, and
folds are never treated as independent evidence.

P0 fails closed if exact pairing is impossible, identity is guessed, published
peak strength cannot be reproduced under one global tolerance, a primary domain
is unavailable, or source/version identity cannot be hash-bound. A domain with
fewer than 20 exact pairs or more than 20% missing primary raw channels yields
`P0_REQUIRES_HUMAN_REVIEW` and blocks P1.

P1, P2, and P3 use exactly the thresholds in the approved operator
specification. Thresholds, endpoints, feature views, model families, and
multiple-comparison treatment are never changed in response to observed
results.

## Artifacts and failure behavior

Each executed stage writes only beneath `results/damage_to_failure_response/`
and `artifacts/damage_to_failure_response/`. Required files are written
atomically, bound by `artifact_manifest.json` and `CHECKSUMS.sha256`, and
verified by replay. A failed or review-required gate still produces its full
audit package and explicitly marks later stages `NOT_RUN_NOT_AUTHORIZED`.

Raw data, images, model weights, caches, and large arrays are never committed.
Frozen scientific result roots and `artifacts/aei_information_hierarchy/` must
remain byte-for-byte untouched relative to the base commit.

## Testing and execution

Development follows test-first red-green-refactor cycles. P0 tests cover exact
pairing, raw decoding, global physical conversion, unresolved strain units,
post-CAI exclusion, source hashes, atomic artifacts, replay, and fail-closed
gates. Conditional stages add feature-view, leakage-sentinel, nested-evaluation,
functional-basis, and authorization tests before any large run.

The final verification includes focused and full tests, Ruff, checksum replay,
frozen-path diff gates, Git diff checks, local/remote SHA equality, and a clean
working tree.

## Self-review

- Placeholders: none.
- Conflicting stage transitions: none; every downstream stage is conditional.
- Scope: Paper 1 science and manuscript files are excluded.
- Ambiguity: unresolved strain units explicitly disable strain endpoints rather
  than permitting inference.
- Unsupported claims: none; literature novelty and empirical outcomes remain
  unknown until their respective audits complete.
