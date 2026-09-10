# Reference Feasibility and Next Step

## Decision

Status: `COMPLETE_WITH_EVIDENCE_LIMITS`. The current evidence supports `AUTHOR_SCALARS_ONLY` and `SCREENSHOT_MEASUREMENT_ONLY`, not an author spatial mask.

- Physical TEST specimens: 24
- Author spatial references: 0
- Author damage scalars: 24
- Author C-scan screenshots: 24
- Quantitative/raw ultrasound records: 0
- Fixed cases with visually verified 75 x 75 mm screenshot axes: 3

## What the evidence supports

The author projected-delamination-area scalars can audit overall magnitude for like-staged CHARACTERIZE masks when the screenshot display scale is case-verified. The existing expert polygons remain the independent spatial reference for the current reviewed task. These roles are complementary and are not merged into a new GT.

- `ykhs7s2dck:q8-4`: author 55.2 mm^2; expert-region display-calibrated area 86.8 mm^2; full-input CHARACTERIZE Reader area 3029.7 mm^2. Status: `COMPARABLE_WITH_SCOPE_LIMIT`.
- `74t7kcdgkr:c8-24`: author 49.7 mm^2; expert-region display-calibrated area 90.8 mm^2; full-input CHARACTERIZE Reader area 130.1 mm^2. Status: `COMPARABLE_WITH_SCOPE_LIMIT`.
- `cgtnjyggtm:q24-40`: author 449.9 mm^2; expert-region display-calibrated area 483.0 mm^2; full-input CHARACTERIZE Reader area 787.1 mm^2. Status: `COMPARABLE_WITH_SCOPE_LIMIT`.

## What it does not support

A scalar area cannot identify location, shape, boundary, a required scan region, or an optimal action order. The author screenshots contain amplitude-coded observations but no inspected author contour/ROI. The 0.2 mm instrument pitch is not used to relabel screenshot pixels as physical probe points; native-raster cost remains screenshot-pixel acquisition cost, not time, travel, or path length.

For `q8-4`, the new author scalar (55.2 mm^2) and original screenshot do not explain why the expert region is compact while the Reader response is broad. They show that the full-input Reader's mask magnitude is inconsistent with the author scalar under the verified display scale, but they do not identify which pixels are the author's ImageJ region.

The reviewed negative result is unchanged. No threshold, reference polygon, report, trajectory, or success criterion was modified.

## Recommended next step

Request the per-specimen ImageJ ROI/contour and the area/length measurement definition for the current TEST identities, starting with q8-4, c8-24 and q24-40. Raw amplitude grids with coordinates/gating would improve measurement reproducibility but are secondary to the ROI required for spatial adjudication.
