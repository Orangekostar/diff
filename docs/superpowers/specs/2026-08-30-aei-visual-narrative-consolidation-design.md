# AEI Visual Narrative Consolidation Design

## Objective and authority

Reorganize the paper into a reviewer-facing `WHY -> WHAT -> WHEN -> HOW`
sequence while preserving the exact 39-row canonical evidence authority. The
task starts from `9794d53a9549f2e3501fe482e8db8735f468ba20` on
`aei-main-method-reframe` and uses the immutable canonical metrics SHA-256
`f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`.
No training, model selection, endpoint recomputation, or frozen science-path
change is permitted.

## Approved approach

Use source-row redistribution with four live renderers and stable Figure 1-4
filenames. Merge Figure 5's real state and oracle-map bindings into Figure 2,
move O2/U5 evidence into Figure 3, move O3/O4 controls into Figure 4, and add
A4 only as the subordinate Figure 4d calibration panel. Retain the internal
deterministic Table 2 generator as provenance, but remove Table 2 and Figure 5
from the manuscript, materialized paper tree, working package, and flat source.

Rejected alternatives:

- Embedding or cropping the old Figure 5 inside Figure 2 would leave duplicate
  source ownership and an unreachable renderer.
- Composing panels in LaTeX would bypass the deterministic source-row,
  alignment, SVG, and checksum contracts.
- Normalizing the six evidence stages to one visual scale would introduce a new
  scientific transformation with no frozen authority.

## Final visual contract

### Figure 1: WHY

One conceptual framework figure with no performance value. It explains the
engineering problem, Part I information characterization, Part II
state-conditioned acquisition, the legal-state boundary, and the distinction
between retrospective teacher/oracle evidence and deployable decisions.

### Figure 2: WHAT

A 2-by-3 full-width figure:

- (a) matched scalar, full spatial field, and registered 25% sparse-field CAI
  MAE; 32.1% reduction and 89.9% retention remain explicit.
- (b) hash-verified `c8-2` initial legal state at 3.13%, with measured native
  positions and no implied raw full raster.
- (c) retrospective mechanical-oracle opportunity versus uniform and the
  field-content reference, including headroom retention.
- (d) CAI-task within-map priority percentiles and top-five cells.
- (e) C-scan-content within-map priority percentiles, operationalized by the
  registered normalized-RGB-MSE reconstruction objective.
- (f) paired CAI-minus-field-content percentile difference; this is not a raw
  utility difference or causal map.

Figure 2 source rows own U1/U2/U3/U4, the state manifest, the oracle-value
parquet, and P14-bound canonical source hashes. O2 and U5 leave Figure 2.

### Figure 3: WHEN

A 2-by-3 full-width figure:

- (a,b) initial and 18.75% strict-OOF priority maps for the same specimen and
  trajectory, using one percentile scale.
- (c) registered acquisition history with white initial and red latest markers.
- (d) O2 turnover, rank agreement, and top-five overlap across acquired cost.
- (e) O4 dynamic-minus-static regret with the O1 static-rank annotation and the
  unchanged sign convention.
- (f) U5 predictor-conditioned rank agreement with full-state MAE context.

Figure 3 source rows own O2/U5/O4-dynamic/O1 and the state/action-source rows.
O3 controls leave Figure 3.

### Figure 4: HOW

A 2-by-2 full-width figure:

- (a) measured state, acquired-position/history, field-content, and
  shuffled-content controls, preserving all adverse directions.
- (b) A1 retrospective valuation and planning substitutions.
- (c) A2 greedy and beam-4 regret in the registered two-action reachable pool.
- (d) A4 signed deployment calibration; negative continues to favor the static
  reference and does not define framework performance.

A3 remains supplement-only. A4 remains `MAIN_SYSTEM_DIAGNOSTIC`, is mapped to
`figure4d`, and is not promoted in the abstract, Introduction, contributions,
or Conclusions.

## Terminology boundary

Reviewer-visible roles use `CAI-task priority`, `C-scan-content priority`,
`field-content reference`, `field-content control`, and `CAI-specific excess
priority`. The exact operational definition `registered normalized-RGB-MSE
reconstruction objective` appears at least once in the main paper and remains
in the supplement. Frozen identifiers, claim IDs, source fields,
`control_mode="reconstruction"`, and `reconstruction_*` dataclass fields remain
unchanged.

## Data flow and traceability

`PAPER_CANONICAL_METRICS.csv` and hash-bound result files feed source-row
builders. Source CSVs feed renderers. Each live renderer emits editable SVG,
vector PDF, nonblank 300-dpi PNG, caption, and alignment artifacts where
applicable. `FIGURE_CHECKSUMS.csv` binds those deliverables. Governance CSVs
remain in canonical order and change only human-facing roles plus figure/table
placement; the visibility partition stays 12/15/1/11.

The main package copies Figure 1-4 and Table 1 only. Supplementary Figure S1
and all supplement evidence remain. The internal Table 2 CSV/TeX generator may
continue to run under `results/` but its output is not materialized or submitted.
Stale Figure 5 and Table 2 files are explicitly removed by materialization.

## Independent panel PNG deliverable

For manual composition, export formal panels without horizontal stretching:
Figure 2 (a-f), Figure 3 (a-f), Figure 4 (a-d), and Supplementary Figure S1
(a-l), plus the one-panel Figure 1. Crops are derived from the final 300-dpi
PNG and audited alignment geometry, preserving each grid cell's native aspect
ratio. They live under `paper_aei_information_hierarchy/panel_pngs/` with a
SHA-256 manifest and are deliberately excluded from the submission package.
This is a presentation derivative only; it introduces no new data or metric.

## Error handling

Rendering fails closed when a canonical claim, source hash, state
reconstruction, legal 8-by-8 priority map, or alignment contract is missing or
invalid. Panel export fails if a declared panel is absent, crop geometry is
outside the raster, output is blank, or aspect ratio would require resizing.
Materialization removes only enumerated stale paper artifacts.

## Verification strategy

Use test-first contract changes. RED tests must first require four source sets,
the new claim distribution, Figure 2's merged bindings, Figure 4d A4, 4/1
manuscript/package counts, stale cleanup, bounded terminology, and panel PNG
exports. GREEN implementation then runs the focused paper suite, Ruff and
format checks, two-directory byte replay, source/alignment/font/collision QA,
main/supplement/flat LaTeX builds, deterministic package replay, canonical SHA,
frozen-path diff, and final GitHub local/remote verification.
