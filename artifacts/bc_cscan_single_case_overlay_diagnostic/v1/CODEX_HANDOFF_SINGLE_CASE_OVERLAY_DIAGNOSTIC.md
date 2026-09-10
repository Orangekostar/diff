# Codex Handoff: Single-Case Overlay Diagnostic

## Repository Binding

- Base commit: `6954e37c8c6aa5fc8530d2668d689eae85604691`.
- Branch: `research/bc-cscan-single-case-overlay-diagnostic`.
- Export command: `python scripts/export_single_case_overlay_diagnostic.py --source-root <cmc_damage_inference-root>`.
- This export replays stored evidence only. It performs no training, model calls, replanning, endpoint selection, or reviewed formal rescore.

## 1. Selected Specimen

- Specimen: `ykhs7s2dck:q8-4` from the reviewed TEST split.
- Task: `LOCATE`.
- Policy: `BC_S3`, seed `3`.
- Actual calibrated first STOP: step `153`, final native-raster cost `0.7840312122211232`.
- Selection reason: among the preselected reviewed fixed cases with an actual calibrated first STOP, an available full-input Reader report, and distinct BC/proxy report identities, this case has the lowest reviewed EXPERT-vs-full-input Reader LOCATE IoU. It is a deliberately diagnostic failure case, not a favorable example selected for appearance.

## 2. Two Meanings of GT

- `EXPERT_GT`: the reviewed expert polygon reference, rasterized from `reference-023-4c81fe9c.json` in the registered C-scan frame.
- `PROXY_GT`: the legacy full-input Reader target. It is a Reader output and is not expert ground truth.

The current reference contains one certain polygon and no uncertain polygon.

## 3. Expert Versus Proxy

They are strongly different in this case. `EXPERT_GT` contains 7,024 pixels (1.54% of the image), while `PROXY_GT` contains 211,409 pixels (46.47%). Their pixel-mask IoU is `0.023517669881076217`. The existing reviewed table reports a LOCATE bounding-box IoU of `0.026349620054768468`; that number is not the pixel-mask IoU.

## 4. Which Reference BC Resembles

The BC first-STOP mask is substantially closer to `PROXY_GT`:

| Pair | Pixel-mask IoU |
|---|---:|
| EXPERT_GT vs PROXY_GT | 0.023518 |
| EXPERT_GT vs BC_FINAL | 0.031941 |
| PROXY_GT vs BC_FINAL | 0.733791 |

`BC_FINAL` contains 155,130 pixels (34.10% of the image). The figure therefore supports the statement that BC reproduces much of the legacy Reader target while remaining nearly disjoint from the compact expert region.

## 5. Stored BC Trajectory

The overlay uses the 153 actions actually executed before STOP; the action recorded on the stop row is not executed. The trajectory is broad rather than confined to the expert polygon:

- top-left: 36 actions;
- top-right: 48 actions;
- bottom-left: 21 actions;
- bottom-right: 48 actions;
- terminal grid state: 50 full cells, 3 coarse cells, 0 intermediate cells, and 11 unmeasured cells;
- measured native positions: 356,695 of 454,950 pixels (78.40%).

The action counts concentrate on the right half (96 of 153 actions), while still sampling all four quadrants.

## 6. Diagnostic Classification

The case is most consistent with a `PROXY_GT` definition / Reader extraction mismatch relative to the reviewed expert target. The full-input Reader already produces a broad texture-associated region, and BC at first STOP closely follows it. The overlay does not support treating STOP or planning as the primary explanation for the expert disagreement, although it cannot prove that they make no contribution.

The images alone cannot separate a conceptual proxy-target problem from a specific Reader threshold/background-prior extraction problem. They also do not establish that the expert annotation is erroneous.

## 7. Supported Findings Versus Hypotheses

Directly supported by stored data and overlays:

- the expert mask is compact around the central visible defect;
- the proxy and BC masks cover much broader registered-image texture;
- BC is far closer to the proxy mask than to the expert mask;
- the policy sampled broadly and stopped at the stored first-STOP state;
- the four source/coordinate representations are identical.

Still hypotheses requiring separate analysis:

- which Reader component causes the broad extraction;
- whether the expert polygon should be revised;
- whether a different stopping or planning rule would materially improve expert agreement;
- any causal attribution beyond this single diagnostic case.

## 8. Coordinate and Source Identity

- Registered size: width `675`, height `674`.
- Source image SHA-256: `f671d2558f1bd96d6977bb556e59d5b0dbdf28daca5821518e840bbc73a4e21d`.
- HTML display SHA-256: `7a41e62db515504fa2426576c537e5f9ff05e1425444192a5d8700bc2052db4b`.
- Reference frame: `registered_cscan`.
- Manifest source path, Reader record path, and authority source identify the same file.
- Decoded HTML packet pixels, reconstructed display pixels, source pixels, Reader full scan, and authority full scan are equal.
- Orientation relation: `IDENTICAL_NO_TRANSFORM`; no rotation, reflection, crop, or resize was applied.

No annotation/Reader base-image mismatch was found.

## Rendering Correction

The initial PNG export applied `origin="upper"` independently to both the mask image and its contour. With the registered y-axis already directed downward, the contour received a second y transform and was vertically mirrored while the translucent fill remained correct. The shared contour call now uses the registered array coordinates directly. An asymmetric-mask regression test checks exact contour bounds, and all PNGs containing mask boundaries were regenerated in place. Mask arrays, report hashes, overlap metrics, first-STOP identity, trajectory, and the base C-scan were unchanged.

## Stored Identity

- Reference version: `REVIEWED_02339eda86d486c3`.
- Full-input proxy report SHA-256: `382775ba836a551517567d8ef758b495dea576b7e76e8b6a1bbd42f9bf6dc063`.
- BC first-STOP report SHA-256: `474027ce54631e59064ce1a1f3c2d6dc66920b487860ca31cf85bd336a711b38`.
- Manifest: `single_case_overlay_manifest.json`.

## Exported Panels

- `panel_00_base_registered_cscan.png`: registered C-scan only.
- `panel_01_expert_gt_overlay.png`: green expert reference, with uncertainty status.
- `panel_02_proxy_gt_overlay.png`: blue full-input Reader proxy target.
- `panel_03_bc_final_mask_overlay.png`: red-orange BC first-STOP mask.
- `panel_04_bc_trajectory_and_mask_overlay.png`: actual 8 x 8 macro-action trajectory, measured cells, and BC mask.
- `panel_05_overlay_compare_expert_vs_bc.png`: expert and BC comparison.
- `panel_06_overlay_compare_expert_vs_proxy_vs_bc.png`: three filled overlays.
- `panel_07_overlay_boundary_only_compare.png`: three boundaries without fills.
- `figure_single_case_overlay_summary.png`: 2 x 3 browsing summary.
