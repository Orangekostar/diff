# MVA A2 Visual Contract

Artifact: Figures O1-O5 and the Error-Budget Curve.

Target venue / format: full-width scientific figures; PNG at 300 dpi plus editable SVG.

Core claim: the retrospective mechanical-value oracle must be spatially inspectable and quantitatively compared with uniform, random, appearance, and reconstruction acquisition under equal measurement budgets.

Reviewer question: do the value maps select different locations, and does that difference translate into lower held-out-domain CAI error?

Evidence layer: O1-O5 are qualitative mechanism diagnostics; Error-Budget is the main quantitative result.

Source data: formal `state_metrics.parquet`, `oracle_values.parquet`, `oracle_trajectories.parquet`, `anchor_curve.csv`, and frozen A0/A1 acquisition metadata.

Statistics / uncertainty: P-B equal-domain MAE; random shows the 5th-95th seed quantile. Gate uncertainty remains in `bootstrap.csv`, not visually substituted by the random band.

Figure prototype: image plate, three aligned 8x8 percentile heatmaps, trajectory small multiples, and a multi-series line chart.

Panel map:

- O1: initial bilinear sparse reconstruction and measured-location mask.
- O2: initial reconstruction-value spatial ranking.
- O3: initial appearance-value spatial ranking.
- O4: initial CAI mechanical-value spatial ranking.
- O5: first acquisition step per cell for the three oracle trajectories.
- Error-Budget: P-B MAE against actual nominal checkpoint, with 50% uniform and 100% FULL report-only anchors.

Caption role: identify the retrospective normalized-raster simulation and avoid physical pitch, scan-time, or deployment claims.

Manuscript placement: O1-O5 beside the oracle mechanism audit; Error-Budget immediately after the A3 headroom question.

Output formats: `figures/*.png`, matching `figures/*.svg`, and `figures/source_data.csv`.

Traceability: the representative specimen is fixed as the first registered cohort row, `c8-2`; it is not selected by outcome. Every plotted value is exported to `source_data.csv`.

Palette and accessibility: Okabe-Ito categorical lines, neutral support colors, cividis percentile maps, distinct line styles and markers, no rainbow map, and grayscale-distinguishable method encodings.

No-fabrication status: render only after all formal shards and aggregate tables exist; no placeholder values enter final artifacts.
