# MVA A5 Visual Contract

Status: frozen before formal A5 target outcomes are aggregated
Mode: standard, full visual composition
Target format: publication-width SVG plus 300 dpi PNG

## Artifact 1: Deployable CAI error-budget curves

- Core claim: compare the frozen imitation policy with three observed-only
  heuristics and the registered A2/A4 references at equal simulated budgets.
- Evidence layer: primary result and limitation.
- Source data: validated `cai_curves.csv`, P-B rows only.
- Statistics: equal-domain MAE and the registered random 5th-95th seed band.
- Prototype: one common-axis line plot with stable markers and line styles.
- Traceability: export every plotted point and random-band bound to
  `figures/source_data.csv`.

## Artifact 2: Held-out-domain policy effects

- Core claim: show where the imitation policy improves or degrades AUEBC
  relative to the global mechanical mask.
- Evidence layer: gate heterogeneity.
- Source data: validated `domain_metrics.csv`.
- Statistics: exact per-domain `global - policy` P-B AUEBC; no fitted trend or
  post-hoc domain ordering.
- Prototype: six bars in the frozen domain order with a zero reference line.
- Traceability: export all six effects to `figures/source_data.csv`.

## Artifact 3: Training trace and oracle-gap context

- Core claim: expose deterministic source-only optimization traces and place
  policy AUEBC between the global baseline and retrospective oracle.
- Evidence layer: optimization audit and headroom limitation.
- Source data: validated `policy_training.csv` and `budget_metrics.csv`.
- Statistics: all 50 epochs for all six outer folds; exact P-B AUEBC values.
- Prototype: paired panels for loss traces and global/policy/oracle AUEBC.
- Traceability: export every epoch and all three area values to
  `figures/source_data.csv`.

## Shared visual grammar

- Imitation policy: blue `#0072B2`, solid line, circle marker.
- Global mechanical: vermillion `#D55E00`, solid line, plus marker.
- Uniform: neutral gray `#666666`, dashed line, diamond marker.
- Mechanical oracle: black, long-dashed line, star marker.
- Other deployable heuristics and random use distinct Okabe-Ito colors,
  markers, and line styles.
- Domain labels remain anonymized as D1-D6 in the frozen order.
- SVG text remains editable; all PNG and SVG outputs are inspected at final
  publication size.

## No-fabrication boundary

Figures consume only validated formal tables. Axis truncation chosen after
seeing results, domain reordering, smoothing, selective epochs, invented
uncertainty, significance marks, and representative-specimen selection are
forbidden.
