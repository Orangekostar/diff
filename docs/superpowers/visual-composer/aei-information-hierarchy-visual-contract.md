# AEI Information Hierarchy Visual Contract

Target venue / format: Advanced Engineering Informatics, full-width journal
figures and booktabs main tables.

## Shared encoding

- Useful: Okabe-Ito orange `#E69F00`, circles, solid line.
- Observable: Okabe-Ito blue `#0072B2`, squares, dashed line where needed.
- Actionable: Okabe-Ito green `#009E73`, diamonds, solid line.
- Adverse or unsupported boundary: vermillion `#D55E00` plus hatch/marker.
- Reference/control: neutral gray `#666666`; uncertainty: `#A6A6A6`.
- Text/background: `#222222` / white; grid: `#D9D9D9`.

Color never carries the only distinction. Markers, hatches, direct labels, and
zero/reference lines must remain legible in grayscale. All vector text remains
editable; PNG is preview only.

## Figure 1

Artifact: Information hierarchy.

Core claim: usefulness, conditional observability, and actionability are
distinct validation questions.

Reviewer question: What is the paper's engineering-information construct, and
where do retrospective and deployable information diverge?

Evidence layer: conceptual main framework; no performance numbers.

Source data: generated `figure1_hierarchy.csv` defining nodes, roles, and gate
states; fixed wording from the paper scope ledger.

Statistics / uncertainty: none.

Figure prototype: left-to-right staged flow with a lower evidence-lane
distinction between retrospective teacher/oracle, legal deployable state, and
policy/planner.

Panel map: one unframed framework panel. Each gate is visually distinct; arrows
after Useful are not depicted as automatically successful.

Caption role: define the hierarchy and its validation logic without implying
all gates pass.

Manuscript placement: Section 3, after formal problem definition.

Output formats: SVG, PDF, PNG at 300 dpi.

Traceability: `src/cmc_bbdm/mavis/aei_paper_figures.py`, source CSV, visual
contract, caption file, and checksums.

## Figure 2

Artifact: What is useful?

Core claim: matched spatial morphology improves CAI prediction, sparse
measurements retain most of that gain, and oracle-optimal design depends on the
task; learned global masks do not reproduce the oracle separation.

Reviewer question: Is the usefulness claim registered, sparse, and genuinely
task-specific rather than a generic reconstruction effect?

Evidence layer: main result plus limitation.

Source data: `PAPER_CANONICAL_METRICS.csv`, P1 domain metrics, P5 retention and
bootstrap rows, and bound P14 task matrix.

Statistics / uncertainty: registered simultaneous intervals for P1/P5 effects;
synchronized specimen-bootstrap intervals for P14 oracle contrasts; six-domain
equal weighting.

Figure prototype: three coordinated panels.

Panel map:

- (a) matched scalar versus selected B-family spatial MAE with the registered
  effect and simultaneous interval;
- (b) surface/full/sparse MAE and explicit 89.9% gain retention, with the
  sparse-minus-full boundary visible;
- (c) CAI and reconstruction objective-specific oracle outcomes, with the
  learned global-mask failure explicitly marked.

Caption role: answer RQ1 and mark all oracle rows non-deployable.

Manuscript placement: first part of Section 5.

Output formats and traceability: as Figure 1, using `figure2_usefulness.csv`.

## Figure 3

Artifact: Why usefulness is not observability.

Core claim: static value observability is weak while true conditional value
evolves; real-content representation does not establish value beyond matched
positions/reconstruction controls, and shuffled content remains an adverse
dynamic control.

Reviewer question: Does the legally available state robustly identify future
measurement value?

Evidence layer: main mechanism and central adverse controls.

Source data: canonical metrics, bound MVD M1 rows, P9 aggregate checkpoint
metrics, P10 matched-control contrasts, and P11 endpoint contrasts.

Statistics / uncertainty: strict-OOF Spearman interval; teacher checkpoint
descriptives over 276 specimens/six domains; synchronized bootstrap intervals
for P10/P11 contrasts.

Figure prototype: three panels.

Panel map:

- (a) static Spearman estimate/CI plus exact-budget regret controls;
- (b) teacher turnover, rank agreement, and top-k overlap across checkpoints;
- (c) real/positions/reconstruction MAE and dynamic real-minus-static/shuffled
  regret contrasts, with adverse direction explicit.

Caption role: answer RQ2 without claiming information-theoretic impossibility.

Manuscript placement: middle of Section 5.

Output formats and traceability: as Figure 1, using
`figure3_observability.csv`.

## Figure 4

Artifact: Why observability is not actionability.

Core claim: retrospective valuation and bounded set-planning gaps exist, but
feedback is adverse and the frozen policy does not beat the strongest
deployable baseline.

Reviewer question: Is retrospective value converted into a better
cost-constrained sensing decision?

Evidence layer: main boundary and central adverse controls.

Source data: canonical metrics, bound P12 substitution matrix, P13 planning
summary, P16 feedback summary, and frozen P7 claim evidence.

Statistics / uncertainty: synchronized specimen-bootstrap intervals;
specimen-first equal-domain aggregation; retrospective planners marked
non-deployable.

Figure prototype: three panels.

Panel map:

- (a) valuation, learned-planning, and true-value-planning substitution gains;
- (b) current/beam/lookahead planning regret versus bounded near-oracle;
- (c) feedback benefit and baseline-minus-MAVIS effects centered on zero.

Caption role: answer RQ3 as a boundary, not an end-to-end success claim.

Manuscript placement: final evidence block of Section 5.

Output formats and traceability: as Figure 1, using
`figure4_actionability.csv`.

## Table 1

Artifact: Multi-domain case study and protocol.

Core claim: evidence is evaluated on physical specimens under strict domain
holdout and exact normalized-raster cost semantics.

Reviewer question: What is the cohort, information boundary, cost definition,
and statistical unit?

Source data: canonical cohort/protocol fields plus frozen domain rosters.

Table map: cohort; domain counts; modalities; LODO; checkpoints; exact cost;
teacher/oracle versus deployable information; aggregation; bootstrap.

Output: CSV and booktabs LaTeX. No vertical rules and no invented scanner-time
equivalence.

## Table 2

Artifact: Information hierarchy evidence table.

Core claim: each hierarchy layer has a distinct question, comparison,
uncertainty statement, evidence type, and bounded conclusion.

Reviewer question: Which results support or bound each layer?

Source data: `PAPER_CANONICAL_METRICS.csv` only.

Table map: Layer; Question; Key comparison; Effect; 95% CI; Domains; Evidence
type; Conclusion. Central adverse controls remain in the main table.

Output: CSV and booktabs LaTeX with consistent precision and direction notes.
