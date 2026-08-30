# P0 Damage-to-Failure Response Data Audit

Status: `P0_GO`

## Authority

- Exact Git base: `3951f71f28b6efdf8c74eea0fe274b2a78a9cd57`
- Dataset: `8scdmfdcfb`, version `3`
- Official raw records: 446
- Official post-CAI image records: 892 (remote official hashes only;
  local image bytes were not downloaded or used)
- New training: NO

## Primary cohort

- Exact primary pairs: 276/276
- Per-domain pairs: {'74t7kcdgkr': 45, 'cgtnjyggtm': 49, 'w68dtmpfyf': 43, 'xcmzfsbd9t': 59, 'yfxyg8jm46': 42, 'ykhs7s2dck': 38}
- Primary raw traces decoded: 276/276
- Internal-title conflicts: ['c24-12']; each listed conflict is
  retained in QC and canonical identity is supported by official filename,
  dataset version, file SHA-256, and all three workbooks.

## Peak reconciliation

- Formula: `abs(Load[V] * 25 * 1000 / (width_mm * thickness_mm))`
- One workbook-derived absolute tolerance: 0.0050000000000000001 MPa
- Primary pass count: 276/276
- Maximum primary absolute error: 2.2737367544323206e-13 MPa

## Boundaries and expansion

- Strain status: `STRAIN_UNIT_UNRESOLVED`; JIS modulus,
  maximum strain, and all gauge-derived endpoints remain unauthorized.
- Stress-extension endpoints legal for P1 audit: extension_at_peak_load, uniform_pre_peak_stress_extension_slope, pre_peak_integrated_stress_extension_index, normalized_pre_peak_stress_extension_shape.
- Exact raw-file identity plus existing spatial-observation intersection:
  281; primary 276 plus
  5 additional identities.
- Decodable raw response plus spatial-observation intersection:
  280; primary 276 plus
  4 additional candidates.
- Additional impacted raw identities with scalar damage observations but no
  established spatial pair: 139.
- One non-primary raw source anomaly is retained in QC: `q8-17` declares 15,711
  rows but contains 3,840 data rows.

## Gate

- Decision: `P0_GO`
- Reasons: []
- P1-P5 execution state in this package: `NOT_RUN_NOT_AUTHORIZED`
