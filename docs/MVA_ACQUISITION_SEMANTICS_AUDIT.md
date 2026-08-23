# MVA Acquisition Semantics Audit

Date: 2026-08-23
Decision: normalized retrospective raster acquisition only

## Registered observations

The frozen cohort contains 276 unique RGB PNG C-scan crops over six released
datasets. Crop shapes are:

| Native crop shape `(H,W)` | Count |
|---|---:|
| `(674,675)` | 240 |
| `(338,352)` | 19 |
| `(338,340)` | 17 |

The publisher materials verify a 75 x 75 mm scan field for the six-domain
cohort. They do not provide native A-scan samples, scanner coordinate logs, a
specimen-level point registry, or a proof that each raster pixel is one physical
ultrasonic measurement. The files are cropped/rasterized pseudocolor
screenshots. Seventeen raw screenshots contain two panels; the manifest supplies
separate specimen crops and panel indices. Full-field encoding preserves each
complete crop and later resizes it to 224 x 224 without content cropping.

## Required questions

1. **Original dimensions:** `(674,675)`, `(338,352)`, and `(338,340)` pixels.
2. **Published area:** 75 x 75 mm for every primary-cohort scan field.
3. **Nominal pitch:** not authoritative. Dividing field size by raster size is
   only a screenshot-coordinate spacing, not scanner pitch.
4. **Pixel-to-point identity:** not established.
5. **Resize/crop:** raw screenshots are split into specimen panels; the
   registered PNG is the complete panel crop. ResNet preprocessing resizes the
   complete crop to 224 x 224.
6. **P5 grid:** sampling occurs on the registered native RGB crop before encoder
   resize.
7. **Physical interpretation:** one sampled raster pixel cannot be claimed as
   one physical measurement point.
8. **Permitted physical claims:** the released field covers 75 x 75 mm and the
   experiment reduces observed locations on its normalized raster grid.
9. **Simulation-only claims:** measurement fraction, adaptive allocation,
   spatial observation reduction, and any acquisition trajectory.
10. **Orientation:** image axes are preserved, but no independent
    specimen-level impact-axis registration proves a common material or impact
    orientation.
11. **ROI:** every crop represents the complete released specimen panel, but
    native resolutions differ and a common scanner raster is not proven.
12. **Impact center:** no specimen-level impact-point coordinate exists. The
    geometric crop center is not an authoritative impact center.
13. **Missing/border/annotation pixels:** no missing-value mask or annotation
    overlay registry is supplied. Boundary-connected and unsupported color
    fields caused prior exclusions; four retained specimens have empty derived
    morphology masks but retain valid RGB crops.
14. **Pseudocolor:** RGB intensity is rendering appearance, not calibrated
    ultrasonic amplitude, depth, or damage severity.
15. **Restoration:** every simulated measured RGB triplet is overwritten back
    into the interpolated image exactly and verified byte-for-byte.

## Frozen MVA interpretation

All primary budgets use

`unique observed native-raster locations / native-raster locations`.

Action cells are normalized 8 x 8 spatial regions defined separately for each
native shape. Their boundaries are aligned to a nested endpoint-preserving
lattice. Refinement only adds native raster locations. The interpolation state
is deterministic and preserves measured RGB values exactly.

MVA may say `retrospective acquisition simulation`, `simulated measurement
budget`, `measurement-location reduction`, and `task-driven sampling design`.
It may not say `scanner pitch`, `inspection-time reduction`, `physical scanning
speedup`, `real-time scanner`, or `online deployment`.

## Center and appearance baselines

Center-first is excluded from confirmatory A2 because specimen-level impact
center authority is absent. A visual comparator is retained under the name
`full-image appearance-intensity oracle`; it ranks cells by RGB deviation from
the border median and is explicitly neither a damage mask nor deployable.
