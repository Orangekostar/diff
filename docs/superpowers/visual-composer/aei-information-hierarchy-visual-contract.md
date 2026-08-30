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

Artifact: WHY task-relevant acquisition is needed.

Core claim: a complete sensing field must be reduced under exact acquisition
cost according to task-relevant, state-conditioned value.

Reviewer question: Why is full-field sensing not the decision object, and how
do retrospective evidence and deployable acquisition differ?

Evidence layer: conceptual main framework; no performance numbers.

Source data: generated `figure1_task_relevant_acquisition_framework.csv`
defining four stages and the frozen evidence lanes.

Statistics / uncertainty: none.

Figure prototype: one continuous left-to-right flow: complete sensing field,
limited sensing under exact cost, task-relevant value, and state-conditioned
acquisition loop. Part I and Part II bands separate characterization from
realization; the lower lanes distinguish retrospective teacher/oracle evidence
from legal deployable state and bounded planning.

Panel map: one unframed framework panel exported as one full panel.

Caption role: define the acquisition problem and evidence boundary without
implying that every downstream gate passes.

Manuscript placement: Section 3, after formal problem definition.

Output formats: SVG, PDF, PNG at 300 dpi.

Traceability: `src/cmc_bbdm/mavis/aei_paper_figures.py`, source CSV, visual
contract, caption file, and checksums.

## Figure 2

Artifact: WHAT ultrasonic information matters for CAI.

Core claim: spatial and sparse fields improve CAI prediction, opportunity is
heterogeneous, and retrospective CAI-task priorities are not reproduced by the
preregistered task-agnostic C-scan appearance-saliency reference.

Reviewer question: Are CAI-priority regions merely those ranked highly by a
simple task-agnostic appearance-saliency heuristic?

Evidence layer: main result plus limitation.

Source data: `PAPER_CANONICAL_METRICS.csv`, P1/P5 metrics, the registered real
state manifest, and frozen A2 bootstrap, map-similarity, and oracle-value
artifacts.

Statistics / uncertainty: registered simultaneous intervals for P1/P5 effects;
the frozen 100000-resample synchronized domain-bootstrap interval for the A2
appearance-minus-mechanical contrast; descriptive means over 276 paired
initial-state maps; six-domain equal weighting.

Figure prototype: six coordinated panels in a 2-by-3 layout.

Panel map:

- (a) matched spatial and sparse gains with registered boundaries;
- (b) one hash-verified compact legal state;
- (c) retrospective CAI-oracle versus appearance-saliency AUEBC;
- (d) CAI-task priority on that state;
- (e) task-agnostic C-scan appearance-saliency priority on the same state;
- (f) paired CAI-minus-saliency priority-percentile difference.

Caption role: answer RQ-A while stating that the appearance metric uses no CAI
outcome, both oracles are retrospective/nondeployable, and the difference is
neither raw utility nor a causal material map.

Manuscript placement: first part of Section 5.

Output formats and traceability: as Figure 1, using
`figure2_information_characterization.csv` plus alignment geometry.

## Figure 3

Artifact: WHEN measurement value changes.

Core claim: useful measurement value changes with acquired evidence and depends
on the predictor; state-conditioned valuation improves next-action estimation.

Reviewer question: Does the legally available state robustly identify future
measurement value?

Evidence layer: main mechanism and central adverse controls.

Source data: canonical metrics, bound MVD M1 rows, P9 aggregate checkpoint
metrics, registered state images/actions, and P11 endpoint contrasts.

Statistics / uncertainty: strict-OOF Spearman interval; teacher checkpoint
descriptives over 276 specimens/six domains; synchronized bootstrap intervals
for P10/P11 contrasts.

Figure prototype: six coordinated panels in a 2-by-3 layout.

Panel map:

- (a) initial legal-state priority;
- (b) updated priority after acquired evidence;
- (c) acquisition history on the same trajectory;
- (d) best-action turnover, rank agreement, and top-five overlap across states;
- (e) dynamic-versus-static next-action regret;
- (f) predictor-conditioned rank agreement.

Caption role: answer RQ2 without claiming information-theoretic impossibility.

Manuscript placement: middle of Section 5.

Output formats and traceability: as Figure 1, using
`figure3_state_conditioned_value.csv` plus alignment geometry.

## Figure 4

Artifact: HOW state-conditioned value becomes a bounded decision.

Core claim: matched source controls identify what the valuation uses, component
substitutions localize headroom, and exact-cost planning remains bounded by the
frozen end-to-end result.

Reviewer question: Is retrospective value converted into a better
cost-constrained sensing decision?

Evidence layer: main boundary and central adverse controls.

Source data: canonical metrics, P10 matched-control contrasts, bound P12
substitution matrix, P13 planning summary, and frozen P7 claim evidence.

Statistics / uncertainty: synchronized specimen-bootstrap intervals;
specimen-first equal-domain aggregation; retrospective planners marked
non-deployable.

Figure prototype: four coordinated panels in a 2-by-2 layout.

Panel map:

- (a) real-state, acquired-position/history, and shuffled-content evidence with
  adverse directions explicit;
- (b) valuation, learned-planning, and true-value-planning substitutions;
- (c) current, beam, and lookahead exact-cost planning regret;
- (d) frozen method-minus-strongest-baseline boundary.

Caption role: answer RQ-B as a boundary, not an end-to-end success claim. The
legacy reconstruction-derived control remains traceable in the supplement and
is not relabeled as saliency.

Manuscript placement: final evidence block of Section 5.

Output formats and traceability: as Figure 1, using
`figure4_valuation_planning_realization.csv` plus alignment geometry.

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

## Retired Table 2

The former information-hierarchy evidence table is not a manuscript or
submission asset. Its internal generator may remain as provenance-only support,
but its scientific content is carried by Figures 2--4, their captions, and the
corresponding Section 5 text. No `table2_task_relevant_results.tex` file is
materialized into the paper or deterministic package.
