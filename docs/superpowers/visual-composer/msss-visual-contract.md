# MSSS Visual Contract

Artifact: S1 Figures A--D
Target venue / format: engineering-informatics paper, vector PDF plus PNG preview
Core claim: locate and bound a CAI-sufficient spatial-information plateau without
equating distinct scale operators.
Reviewer question: Does each independent axis show a non-inferior plateau, a
source-stable boundary, and retained spatial specificity?
Evidence layer: main result plus mechanism gate and limitation.
Source data: `sampling_curve.csv`, `gaussian_curve.csv`, `wavelet_curve.csv`,
`spatial_specificity.csv`, `msss_selection.csv`.
Statistics / uncertainty: equal-domain MAE; synchronized within-domain specimen
bootstrap 95% intervals; 5% FULL-relative non-inferiority band.
Figure prototype or table type: small-multiple line curves and a combined
sufficiency/specificity decision plot.

Panel map:

- Figure A: sampling density versus CAI MAE; effective density remains tabular.
- Figure B: Gaussian sigma in pixels versus CAI MAE; no fabricated mm axis.
- Figure C: primary db2 low-pass curve, with haar/db4/detail variants as subdued
  sensitivity traces.
- Figure D: one row per axis showing FULL, global descriptive MSSS,
  OVER-COARSE, and SSG; normalized retention rank is visualization-only.

Caption role: identify fixed-candidate outer OOF evidence, distinguish global
descriptive boundaries from source-selected outer predictions, and state the
spatial-specificity direction.
Manuscript placement: full-width main-results figure after S1 protocol.
Output formats: PDF and PNG at 200 dpi; editable vector text in PDF.
Traceability: every plotted value is read from a mandatory CSV written before
rendering; `figure_manifest.json` binds source and output hashes.

Palette and accessibility:

- FULL: neutral `#222222`, circle marker, solid line.
- MSSS: Okabe-Ito blue `#0072B2`, square marker, solid line.
- OVER-COARSE: Okabe-Ito vermilion `#D55E00`, triangle marker, dashed line.
- Other candidates/sensitivity: `#A6A6A6` or Okabe-Ito sky blue with distinct
  line styles.
- No red/green-only encoding; marker and line style duplicate every status.

No-fabrication status: only validated S1 tables may be plotted. Missing Fourier
data is labelled not run and is not synthesized.
