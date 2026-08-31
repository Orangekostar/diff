# P0R Scan Processing Provenance

## Scope

This record binds the historical raw-screenshot to registered-crop processing
used by the separate P0R author-registration audit. It does not reinterpret or
overwrite the historical P0 decision.

## Authorities

| Authority | Logical path | SHA-256 |
| --- | --- | --- |
| Paired manifest | `data/public/hasebe/manifest/paired.csv` | `f81002981bf2f6aed84818b48da87cd57e6336f5f3da8d78df1a58d26dd8026f` |
| Historical implementation | `src/cmc_bbdm/hasebe.py` | `3bd0c56adb78b65cc09cec340b06c40c7ec6e73988e187067a76f1128ddf83f0` |
| Historical P0 checksums | `results/agentic_task_driven_nde/p0_registration/CHECKSUMS.sha256` | `2d6dcc9f09906379cae70cff1ba2e4ab79d421a36dda8b369d1923c0c35062db` |
| Historical P0 surface manifest | `results/agentic_task_driven_nde/p0_registration/surface_manifest.csv` | `31aaf123d6f7b684566ef19387d11a5ef2756e5bc51e666f8b1cdcc3b9ed5fe1` |

The historical external project copy contains no Git metadata. The
preprocessing source commit is therefore recorded as
`UNAVAILABLE_NO_GIT_METADATA`; no commit identity is inferred.

## Fixed Processing Recipes

The hash-bound `crop_cscan_panels` implementation is reproduced by three
closed recipes:

| Raw screenshot | Panel index | Axis-aligned crop box | Registered crop |
| --- | ---: | --- | --- |
| `891 x 891` | 0 | `(31, 33, 706, 707)` | `675 x 674` |
| `669 x 885` | 0 | `(39, 33, 469, 708)` | `430 x 675` |
| `996 x 581` | 0 | `(30, 33, 370, 371)` | `340 x 338` |
| `996 x 581` | 1 | `(464, 33, 816, 371)` | `352 x 338` |

The operation is RGB decode followed by axis-aligned crop. Resize,
interpolation, rotation, and reflection are all `NONE`/`IDENTITY`.

## Replay Result

- Frozen primary specimens: 276 across six domains.
- Unique raw screenshots: 259.
- Unique multi-panel screenshots: 19.
- Multi-panel screenshots with both panels selected: 17.
- Multi-panel screenshots with one panel selected: 2.
- Panel-count rows: 240 single-panel and 36 dual-panel.
- Registered crop geometries: 240 at `675 x 674`, 17 at `340 x 338`, and 19 at `352 x 338`.
- Exact recovered-panel versus registered-crop decoded RGB matches: 276/276.
- Unresolved panels: 0.
- Unsupported processing operations: 0.

The machine-readable record is
`results/agentic_task_driven_nde/p0r_author_registration/scan_processing_provenance.csv`
with SHA-256
`553153aeac0532b8083d2d9bbfc86d598714198d62ed6d369a6918a195dfe9b0`.

## Q24-7 Exemplar

`q24-7astm` in dataset `6zt73pcnxv` is not in the frozen 276-specimen roster.
It was checked separately because it is the author-supplied example:

- impacted surface SHA-256: `df05403626b5aae8a8dc3637ec5b95c808b5c71dd49a2b5af1f50f8d868c7d48`;
- raw scan SHA-256: `0b6a45259ffa8904dec678abdc738d08de86f5736609168f19ffd1f0f3235d30`;
- registered crop SHA-256: `9b44277ebf79b5106347d6633495e4f4b95ff645241839b89f902cc463508e32`;
- raw panel index: 0;
- surface geometry: `3147 x 2084`;
- registered crop geometry: `430 x 675`;
- decoded panel replay: exact match.

## Registration Boundary

The verified processing chain supplies the specimen-specific destination crop.
The author authority then fixes `ROT90` globally and maps normalized full-frame
surface coordinates edge-to-edge into that crop. It does not establish a
physical mm calibration. No CAI, oracle value, damage mask, damage centroid,
target-domain label, or manual target alignment was used to select orientation,
crop, scale, or offset.

All 17,664 cell mappings passed inverse/forward round-trip checks. The grid
mapping CSV SHA-256 is
`fb076daefbb0b748184d10b777268dfd13777a3e78cb7fed104252e6d62afd35`.

