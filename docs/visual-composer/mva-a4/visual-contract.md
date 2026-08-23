# MVA A4 Visual Contract

Status: frozen before formal A4 outcomes are read
Mode: standard, full visual composition
Target format: publication-width SVG plus 300 dpi PNG

## Artifact 1: Global ranking maps

- Core claim: the three source-learned objectives induce inspectable normalized
  8 x 8 source scores whose six outer-fold consensus patterns can be compared.
- Reviewer question: where do appearance, reconstruction, and mechanical value
  place early measurements, and are their patterns visibly distinct?
- Evidence layer: mechanism.
- Source data: validated `rankings.csv`; no target image or target CAI.
- Statistics: for each cell, equal-weight mean of the six outer-fold normalized
  source scores in [0, 1]; this consensus is a diagnostic summary, not a single
  deployed mask. Complete fold-specific acquisition orders remain in
  `rankings.csv`.
- Prototype: three aligned consensus-score heatmaps with a common sequential
  scale.
- Panel map: (a) appearance, (b) reconstruction, (c) mechanical.
- Caption role: define cell orientation, score direction, equal-fold
  aggregation, source-only construction, and the distinction between the
  consensus diagnostic and fold-specific deployed orders.
- Placement: full-width near the first description of global masks.
- Traceability: export every plotted cell to `figures/source_data.csv`.

## Artifact 2: CAI error-budget curves

- Core claim: compare global mechanical acquisition with uniform, random median,
  global appearance, global reconstruction, and the specimen oracle.
- Reviewer question: does the global mechanical mask lower equal-domain P-B MAE
  throughout the registered budget interval, and how much oracle gap remains?
- Evidence layer: main result and limitation.
- Source data: validated `cai_curves.csv`, P-B rows only.
- Statistics: equal-domain MAE; random 5th-95th seed band; FULL MAE horizontal
  reference; gate decisions remain table/statistics derived.
- Prototype: one common-axis line plot with stable method markers and styles.
- Caption role: identify P-B common-head evaluation, nominal budgets, random band,
  and retrospective simulation scope.
- Placement: full-width immediately after the primary A4 result paragraph.
- Traceability: export every line/band point to `figures/source_data.csv`.

## Artifact 3: Image-task tradeoff

- Core claim: image reconstruction fidelity and CAI prediction utility are
  related measurements but not interchangeable objectives.
- Reviewer question: does the reconstruction-ranked mask improve RGB fidelity
  without necessarily minimizing CAI error?
- Evidence layer: mechanism and limitation.
- Source data: validated `image_curves.csv`, P-B rows for the three global masks.
- Statistics: equal-domain normalized RGB MSE, SSIM, and P-B MAE at each nominal
  checkpoint; no fitted trend or invented uncertainty.
- Prototype: (a) MSE-budget lines, (b) SSIM-budget lines, (c) CAI MAE versus MSE
  connected checkpoint paths.
- Caption role: define metric directions and prohibit interpreting RGB fidelity
  as mechanical validity.
- Placement: full-width after the image-versus-task analysis.
- Traceability: export all points to `figures/source_data.csv`.

## Shared visual grammar

- Global mechanical: Okabe-Ito blue `#0072B2`, solid line, circle marker.
- Global reconstruction: bluish green `#009E73`, dashed line, square marker.
- Global appearance: orange `#E69F00`, dash-dot line, triangle marker.
- Uniform: neutral gray `#666666`, solid line, diamond marker.
- Random median: reddish purple `#CC79A7`, dotted line, x marker.
- Mechanical oracle: black `#000000`, long-dashed line, star marker.
- Ranking heatmaps: perceptually uniform `cividis`, common [0, 1] limits, no
  rainbow palette.
- Critical distinctions use line style and marker in addition to color.
- SVG text remains editable; fonts and sizes are checked at final figure width.

## No-fabrication boundary

Figures consume only validated tables. Representative specimens, smoothing,
post-hoc axis truncation, selective domains, fitted trends, significance marks,
and unregistered uncertainty are forbidden.
