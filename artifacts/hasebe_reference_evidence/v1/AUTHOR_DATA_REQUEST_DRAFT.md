# Author Data Request Draft

Subject: Request for specimen-level C-scan measurement provenance for the Hasebe CFRP datasets

We are using the six 2022 LVI datasets and the CAI v3 workbook for a task-oriented inspection study. To reproduce the published delamination measurements without interpreting screenshot colours as ground truth, could you provide or clarify the following for the matching specimen IDs?

1. The ImageJ ROI, polygon, contour, binary mask, or measurement table used to obtain projected delamination area and delamination length, with the file-to-specimen/version mapping.
2. The boundary rule used in ImageJ, including colour/amplitude threshold, gate/reference echo, treatment of separate components, holes, and whether area is the union across depth or one selected layer/gate.
3. The exact definition of delamination length (maximum Feret, principal-axis length, bounding-box dimension, or another quantity) and its unit.
4. If shareable, raw or exported quantitative C-scan amplitude data with x/y coordinates, pitch, gate settings, calibration/reference information, and orientation relative to the published screenshot.
5. Confirmation that the measurements are post-LVI and pre-CAI for `q8-4`, `c8-24`, and `q24-40`, and that the CAI v3 workbook rows correspond to the six v1 image datasets.

For `q8-4`, the workbook reports 55.2 mm^2, while different spatial interpretations of the public screenshot yield substantially different regions. The original ROI or boundary rule would let us distinguish a measurement-definition issue from a Reader or expert-interpretation issue without asking for new annotation.

This is a draft only; no message was sent and no author response is represented here.
