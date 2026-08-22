# MGMR Impact Center Audit

Date: 2026-08-22
Decision: specimen-level center alignment is not authoritative

## Evidence chain

- The source experiment instructs that specimens be centered in the impact
  fixture.
- The public C-scan condition reports a 75x75 mm scan length.
- The repository maps each complete registered crop to that 75x75 mm field and
  uses `(37.5, 37.5) mm` as a geometric center for morphology descriptors.
- Native crops have three resolutions and are resized without content cropping.
- Neither the workbooks, manifest, crop extraction records, nor registered masks
  contain a measured impact-point coordinate for each specimen.
- The source screenshots and crops provide no validated transform from fixture
  coordinates or the visible surface impact point to C-scan pixel coordinates.

The pipeline preserves the complete scan field and its pixel axes, so a geometric
center survives resize. That fact does not prove that the actual impact point is
the same pixel for every specimen. Mask centroids are damage-derived quantities
and cannot be substituted for impact coordinates without changing the claim.

## Consequence

An impact-centered polar graph is not authorized. If M0 passes, the first M1
implementation must use a 7x7 Cartesian grid aligned to the frozen layer3 map.
Polar alignment can be reconsidered only after a new, independently verified
specimen-level impact-center registry is supplied.
