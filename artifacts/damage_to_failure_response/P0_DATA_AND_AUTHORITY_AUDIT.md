# P0 Data and Authority Audit

Audit date: 2026-08-30  
Repository base: `3951f71f28b6efdf8c74eea0fe274b2a78a9cd57`  
Dataset: Mendeley Data `8scdmfdcfb`, version 3  
Decision: `P0_GO`

## Executed authority chain

- The official inventory contains 1,341 records: three workbooks, 446 raw CAI
  CSVs, and 892 post-CAI front/back image records.
- Every local raw CSV matches the official byte size and SHA-256.
- The frozen feature bank matches SHA-256
  `f2a69f0da75e20880202d7fc4a6a92f979978406ec21f9d83e4bc8db07fb72a8`.
- The historical spatial manifest and eight discovery source candidates match
  their recorded SHA-256 values. The historical tree has no Git metadata, so
  those source files remain discovery evidence rather than Git authority.
- No private absolute path is serialized in the P0 package.

Machine-readable evidence:

- `results/damage_to_failure_response/p0_data_audit/source_hashes.csv`
- `results/damage_to_failure_response/p0_data_audit/artifact_manifest.json`
- `results/damage_to_failure_response/p0_data_audit/CHECKSUMS.sha256`

## Exact identity and raw QC

- Primary exact pairs: 276/276.
- Per-domain pairs: `45,49,43,59,42,38` in the frozen domain order.
- Identity rule: official canonical filename + feature-bank domain identity +
  official raw-file SHA-256. No row-order, image similarity, or visual guess is
  used.
- Strict raw decoding: 445/446 overall and 276/276 primary.
- `q8-17` is non-primary and fails closed because the file declares 15,711 rows
  but contains 3,840.
- `c24-12` is primary; its internal title is `c24-112`. Canonical `c24-12` is
  supported by the official filename, version, SHA-256, and all three
  workbooks. The mismatch is retained rather than rewritten.

Machine-readable evidence:

- `results/damage_to_failure_response/p0_data_audit/pairing_manifest.csv`
- `results/damage_to_failure_response/p0_data_audit/raw_trace_qc.csv`

## Published-peak reconciliation

The registered conversion is:

```text
load_kN = Load[V] * 25
stress_MPa = load_kN * 1000 / (measured_width_mm * measured_thickness_mm)
raw_peak_MPa = max(abs(stress_MPa))
```

The workbook uses two displayed decimal places, giving one global half-rounding
unit tolerance of 0.005 MPa. All 276 primary specimens pass. The maximum
primary absolute error is `2.2737367544323206e-13` MPa.

Machine-readable evidence:

- `results/damage_to_failure_response/p0_data_audit/published_peak_reconciliation.csv`

## Strain and endpoint boundary

All successfully decoded CSVs label the four gauge channels `με`, while the
Data in Brief prose uses a conflicting label and does not close sign,
averaging, or preload semantics. The final status is
`STRAIN_UNIT_UNRESOLVED`. JIS modulus, maximum strain, gauge dispersion, and
front/back or left/right gauge endpoints are unauthorized.

P1 may audit only stress-extension quantities: extension at peak load, one
uniform pre-peak stress-extension slope, one pre-peak integrated response
index, and normalized pre-peak stress-extension shape.

Machine-readable evidence:

- `results/damage_to_failure_response/p0_data_audit/strain_unit_audit.csv`

## Spatial expansion and post-CAI exclusion

- Existing spatial manifest plus raw-file identity intersection: 281.
- Existing spatial manifest plus valid decoded response: 280.
- Beyond the primary 276: five exact file identities, but four valid decoded
  response candidates because `q8-17` is truncated.
- Another 139 impacted raw identities have scalar damage observations but no
  established spatial pair.
- The 79 `r0/r45` raw identities have no exact spatial pair in the available
  manifest and are not an authorized extension.
- All 892 post-CAI image records are remotely size/SHA bound. Image bytes were
  not downloaded, and `POST_CAI_IMAGE_INPUT_FORBIDDEN = true` remains enforced.

Machine-readable evidence:

- `results/damage_to_failure_response/p0_data_audit/post_cai_image_manifest.csv`
- `results/damage_to_failure_response/p0_data_audit/summary.json`

## Execution evidence

- Focused P0 tests before execution: 82 passed.
- P0 replay: passed, eight payloads verified.
- Independent `sha256sum -c CHECKSUMS.sha256`: all entries passed.
- New model training: **NO**.
- P1/P2/P3/P4/P5 in the P0 package: `NOT_RUN_NOT_AUTHORIZED`.
