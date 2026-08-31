# P0R Author Registration Decision

## Decision

```text
status: P0R_AUTHOR_REGISTRATION_GO
downstream_registration_status: P0_REGISTRATION_GO
p1_authorized: true
authorized_registration_count: 276/276
excluded_specimens: none
```

The historical P0 decision remains
`P0_SPATIAL_REGISTRATION_NO_GO`. P0R is a separate stage based on later,
user-attested author correspondence; it does not revise the evidence available
to the historical audit.

## Gate Facts

- Exact historical identity and hash authority: PASS, 276/276.
- Author statement hash binding: PASS,
  `3560662d4509ea3e059d597cedca15950cce02f706a992330b161381acfba6ba`.
- Evidence type: `USER_ATTESTED_PERSONAL_COMMUNICATION_WITH_DATASET_AUTHOR`.
- Original author communication artifact: not provided; archival remains recommended.
- One global orientation fixed before result inspection: PASS, `ROT90`.
- Specimen panel resolved for every authorized row: PASS, 276/276.
- Historical raw-panel processing replay: PASS, 276/276 exact decoded RGB.
- Unsupported additional rotation or reflection: none.
- Composed normalized full-frame transform replay: PASS, 276/276.
- Registered 8x8 cell round trips: PASS, 17,664/17,664.
- Result-driven orientation inputs: none.
- Physical mm cross-modal mapping: not used.
- Per-domain coverage: PASS, `45/45`, `49/49`, `43/43`, `59/59`, `42/42`, `38/38`.
- Total coverage threshold: PASS, 276 >= 240.
- Author evidence conflict: false.
- Processing provenance unresolved: false.

The registration authority SHA-256 is
`38ab3cf32e866cda447a5edf2637fa502406c4c5c574bc966c13cc1cbbd2553a`.
The authorized roster SHA-256 is
`4fd8c6076dd3fcdf908a73739251db215fcb01f570f1a930b7faf250fe6d285a`.

## Diagnostic Inspection

Twelve hash-selected overlays were generated, two per primary domain. Every
overlay contains the original surface, author-fixed clockwise-90-degree
surface, inverse-mapped surface grid, and registered C-scan grid. All 12 were
inspected after the transforms were frozen. No preprocessing contradiction was
found. No orientation, crop, offset, scale, specimen roster, or transform was
changed after inspection.

The QC checksum ledger SHA-256 is
`f05bd76c5f21e656f61a35d44d8b5a61d4c620ebaa5247c1588c7604931e36e0`.

## Determinism And Integrity

The formal audit and an independent temporary rebuild both returned
`P0R_AUTHOR_REGISTRATION_GO`. Directory-level `diff -qr` was empty for both the
11-file formal P0R package and the 15-file QC directory. The formal package
checksum ledger SHA-256 is
`470a1df4ff19930f5924493fb6a51c49084c76a37376fa266a6499baa9ffbb82`.

No model was trained. No frozen scientific result was recomputed or modified.
P0R authorizes freezing and running the preregistered P1 visual-observability
experiment. P2-P4 remain unrun and are not authorized by P0R alone.
