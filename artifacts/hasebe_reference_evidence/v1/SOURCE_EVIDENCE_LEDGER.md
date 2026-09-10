# Source Evidence Ledger

## Evidence classes

Author workbooks provide scalar measurements; author C-scan files are ultrasound screenshots, not segmentation ground truth. Expert reviewed polygons and local RGB/Reader masks remain separate evidence classes.

## Primary sources

| ID | Source and access | Direct support | Does not establish |
|---|---|---|---|
| O01 | [Hasebe et al., Data in Brief 42 (2022), 108462](https://pmc.ncbi.nlm.nih.gov/articles/PMC9294053/); FULL_TEXT | Internal-damage files are ultrasound C-scan screenshots; display colour encodes amplitude. Reported setup: 3.5 MHz probe, 0.200 x 0.200 mm pitch, and 75 x 75 mm scanning length. | A downloadable author mask or raw quantitative amplitude grid. |
| O02 | [Hasebe et al., Composites Part B 237 (2022), 109844](https://doi.org/10.1016/j.compositesb.2022.109844); ABSTRACT_AND_INTRO_ONLY | The prediction targets include delamination area and delamination length. | The inaccessible methods cannot be cited for a public mask protocol. |
| O03 | [Six Hasebe low-velocity-impact datasets, Mendeley Data v1](https://data.mendeley.com/datasets/74t7kcdgkr/1); RECORD_DESCRIPTIONS_VERIFIED | Each record describes post-impact C-scan images and an image scale. | A spatial contour or raw ultrasound array in the bound local files. |
| O04 | [Hasebe et al., Mendeley Data CAI dataset v3](https://data.mendeley.com/datasets/8scdmfdcfb/3); RECORD_DESCRIPTION_VERIFIED | The package separates LVI conditions/damage, specimen size, CAI images, raw logger records, and strength. | Treating post-CAI photographs or CAI logger traces as pre-CAI ultrasound. |
| O05 | [Hasebe et al., Data in Brief 58 (2025), 111509](https://pmc.ncbi.nlm.nih.gov/articles/PMC11999467/); FULL_TEXT | Projected delamination area was measured from C-scan images with ImageJ; dent depth is reported separately. | An exported ImageJ ROI/contour in the files inspected here. |

## Precise source locations

- O01, Data Description and Internal Damage Measurement/Table 2: screenshot-only export, amplitude-coded colour, 3.5 MHz probe, 0.200 x 0.200 mm pitch, 75 x 75 mm scan length.
- O02: only abstract/introduction-level access was available; no inaccessible method detail is asserted.
- O03: the six Mendeley v1 record descriptions identify post-impact internal C-scanning images and image scales; files inspected locally contain screenshots and impact-condition workbooks, not author masks.
- O03 records: [c8](https://data.mendeley.com/datasets/74t7kcdgkr/1), [c16](https://data.mendeley.com/datasets/yfxyg8jm46/1), [c24](https://data.mendeley.com/datasets/xcmzfsbd9t/1), [q8](https://data.mendeley.com/datasets/ykhs7s2dck/1), [q16](https://data.mendeley.com/datasets/w68dtmpfyf/1), and [q24](https://data.mendeley.com/datasets/cgtnjyggtm/1). The [ASTM record](https://data.mendeley.com/datasets/6zt73pcnxv/1) was checked only as out-of-scope context and was not added to TEST-24.
- O04/O05, Data Description: the CAI package separates LVI damage, size, post-CAI images/logger data and mechanical strength; projected area was measured from C-scan images using ImageJ.

## Local evidence result

- Bound workbooks opened and hash-verified: 9.
- Reviewed TEST specimens with author scalar damage records: 24/24.
- Reviewed TEST specimens with author C-scan screenshots: 24/24.
- Author spatial references found in the bound scope: 0/24.
- Raw/quantitative ultrasound arrays found in the bound scope: 0/24.
- The three exported screenshots retain the original 891 x 891 layout, axes, colour bar and annotations. They are CC BY 4.0 source evidence from the bound Hasebe Mendeley records.
