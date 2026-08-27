# AEI Nature-Style Review Fixes Design

## Objective

Revise the six-section AEI manuscript without retraining, changing frozen
historical results, or presenting a small public dataset as external
validation. The revision addresses metric-definition consistency, evidence
chronology, closest-work positioning, predictor-claim calibration, and
transfer logic.

## Scientific decisions

- Keep `MANUSCRIPT_ONLY_PRIMARY`; the external micro-pilot is `NO-GO` because
  no locally available candidate passes every pairing, endpoint, sample-size,
  confounding, and pre-specification gate.
- Treat the normalized implementation in `closed_loop_metrics._auebc` as the
  AUEBC authority. Correct prose and notation without recomputing results.
- Classify every canonical claim as pre-freeze evidence, the frozen outer
  endpoint, or a post-freeze diagnostic. Post-freeze diagnostics never modify
  the frozen endpoint.
- Position novelty as the joint operational test of Useful, Observable, and
  Actionable claims under one causal acquisition contract, not as a new generic
  taxonomy or first adaptive/VoI/task-driven design.
- Preserve the predictor index in `U_f`. Bound learner dependence by the
  substantial OOF-performance gap between the shallow MLP and Ridge/Huber.
- State five methodological transfer conditions while bounding empirical
  support to the six held-out domains in the present data program.

## Artifact architecture

Paper-only authority files live under `artifacts/aei_information_hierarchy/`.
The closest-work matrix is generated as Table 1; the existing protocol and
hierarchy tables become Tables 2 and 3. Validation checks semantic coverage,
forbidden wording, chronology invariants, exact six-section structure, and
frozen-tree integrity. The package includes all three tables and regenerates
PDF, supplement, flat source, ZIP, manifests, and checksums deterministically.

## Immutable boundary

All historical P1, MVA, MVD, MAVIS, science-closure, and external-data
namespaces listed in the master prompt remain read-only. No learner, scorer,
planner, endpoint, canonical number, or historical source hash changes.

## Acceptance

Completion requires the new semantic tests, all paper/MAVIS/MVD/authority-MVA
regressions, Ruff, `git diff --check`, three LaTeX builds, font/warning/overfull
checks, deterministic replay, frozen-path verification, primary-agent review,
and remote branch/main verification.
