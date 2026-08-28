# AEI Figure Augmentation Plan

## Scope and frozen contracts

- Branch: `aei-main-method-reframe`
- Planning baseline: `b0aeba3f5c2eedda25e5b1c64cfd44d6ae0f4f2c`
- Historical scientific reference: `a21f84f583a3767f727aeace4c38ae7be3f880ee`
- Primary method: Task-Relevant Information Acquisition.
- Part I: Task-Relevant Information Characterization.
- Part II: State-Conditioned Task-Oriented Acquisition.
- MAVIS remains an implementation; `mvd_m1_o2` remains a static reference.
- No training, model selection, endpoint recomputation, canonical metric change, or
  frozen-science-root change is permitted.

## Current figure inventory

1. Figure 1 presents the two-part framework and causal information flow.
2. Figure 2 presents aggregate Part-I evidence without specimen context.
3. Figure 3 presents aggregate state-conditioned valuation and source controls
   without state maps or acquisition paths.
4. Figure 4 presents valuation substitution and bounded set realization.
5. The supplement currently contains no figures.

## Diagnosed visual weaknesses

- Spatial and sparse information are communicated only through aggregate plots.
- No main-text panel shows a real CFRP reconstruction, registered measurement mask,
  or specimen-level priority overlay.
- State evolution is summarized numerically but not shown as a change in the
  reconstruction, priority map, and acquired-cell history.
- Task-specific CAI and reconstruction priorities are not spatially contrasted on
  the same specimen and legal action grid.
- Cross-domain qualitative breadth is absent.

## Evidence-qualified plan

### Figure 1: Task-Relevant Information Acquisition

- Placement: main text.
- Decision: `GO` for Nature-style visual cleanup only.
- Role: preserve the positive two-part method identity and causal flow.
- Source: `PAPER_POSITIVE_NARRATIVE_MAP.csv`.
- Boundary: remove internal audit vocabulary from visible labels; do not add results.

### Figure 2: Spatial task-relevant ultrasonic information and sparse retention

- Placement: main text.
- Decision: `GO`.
- Role: join the registered U1/U2 performance evidence to an auditable real
  specimen, initial scout reconstruction, 25% uniform sparse reconstruction, and
  measured-position mask while retaining the U3/U4, O2, and U5 Part-I summary.
- Representative specimen: `c8-2`, already registered by the MVA figure package.
- Sources: canonical metrics and their bound artifacts, P1 compact state package,
  MVD candidate-bank metadata, and P9 summary.
- Boundary: the raw original full raster is not available in the versioned clean
  clone. The full field therefore remains a quantitative reference; no synthetic
  full-raster image will be drawn.

### Figure 3: State-conditioned measurement value evolves with evidence

- Placement: main text.
- Decision: `GO`.
- Role: preserve O4/O1 and O3 aggregate evidence while adding `c8-2` initial and
  18.75% strict-OOF teacher-priority overlays plus the registered acquisition path.
- Sources: P1 compact state package, P3 action scores, P9/P10/P11 results, and
  canonical metrics.
- Boundary: overlays show within-state percentiles on the registered 8x8 legal
  action grid. They do not imply a universal or causal material map.

### Figure 4: Valuation, planning, and set realization

- Placement: main text.
- Decision: `GO` for Nature-style re-rendering only.
- Role: preserve the A1/A2 component and reachable-set evidence without adding a
  new scientific comparison.
- Source: canonical metrics and their bound P12/P13 artifacts.

### Figure 5: Task-specific priorities on a CFRP specimen

- Placement: main text.
- Decision: `GO`.
- Role: compare CAI-mechanical and normalized-RGB-reconstruction oracle priorities
  at the same initial state, on the same `c8-2` specimen and 8x8 action grid, with
  an explicit percentile-difference view and the frozen aggregate cross-objective
  contrasts.
- Sources: `oracle_values.parquet`, P1 initial state, MVD candidate-bank metadata,
  and P14 task-specificity summary.
- Boundary: this is a representative retrospective oracle visualization, not a
  learned-policy superiority claim.

### Figure 6: Progressive evidence ladder across domains

- Placement: not implemented.
- Decision: `NO_GO`.
- Reason: the six stages use non-commensurate MAE, AUEBC, rank, turnover,
  one-step-regret, and set-regret endpoints. No frozen cross-endpoint normalization
  contract exists. A new common visual magnitude would be analytically arbitrary,
  while a text-only ladder would duplicate Figure 1 and Table 2.
- Resolution: the progression is carried by Figure 1 and the ordered Figure 2-5
  evidence sequence.

### Supplementary Figure S1: Cross-domain state-priority gallery

- Placement: supplement.
- Decision: `GO`.
- Role: show initial and 18.75% state-conditioned teacher-priority overlays for one
  deterministic specimen from each of six domains.
- Selection rule: lexicographically first specimen within each sorted domain; no
  result or effect value is used for selection.
- Sources: P1 compact state package, P3 action scores, and MVD candidate-bank
  metadata.
- Boundary: all panels use the same registered trajectory method and within-state
  percentile scale.

## Nature-style rendering contract

- Python/Matplotlib backend, 180 mm main-figure width, white page background.
- Sans-serif 7-9 pt source typography; every exported PDF glyph at least 5 pt.
- Editable SVG text (`svg.fonttype = none`), PDF, and 300 dpi PNG for every figure.
- Restrained neutral/blue/teal palette; red only for adverse directional cues.
- Shared legends and color bars; no decorative boxes or dashboard cards.
- Every multi-panel figure receives strict 1.5 pt panel-alignment audit output.
- Every final PDF receives text-size and collision audits plus whole-figure visual
  inspection at final physical size.

## Package and test implications

- Main figure contract changes from four to five figures.
- Supplement figure contract changes from zero to one figure.
- Figure source and checksum manifests must bind every new output.
- Tests must cover exact state reconstruction hashes, 64-cell value-map integrity,
  deterministic gallery selection, editable SVG text, and package inclusion.
- Main and supplementary LaTeX, flat submission source, deterministic ZIP, and
  manifest must be rebuilt.
