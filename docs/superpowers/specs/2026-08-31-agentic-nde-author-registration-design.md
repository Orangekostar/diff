# Agentic NDE Author Registration Reopen Design

## Purpose

Reopen the stopped Agentic Task-Driven NDE route using a later, user-attested
personal communication with the Hasebe dataset author. The historical P0
decision remains frozen and correct for the evidence available at that time.
The new work creates a separate P0R stage and authorizes P1 only if P0R passes
its preregistered gate.

## Historical Boundary

- Exact base: `3cb63b544b6c13047773c0eda045558ff4466afa`.
- Historical decision: `P0_SPATIAL_REGISTRATION_NO_GO`.
- Historical P0 package and decision documents are immutable.
- Frozen MVA, MVD, MAVIS, damage-response, AEI, and acquisition-runtime paths
  remain unchanged.
- P0R is new evidence, not a reinterpretation of the old audit.

## New Authority

The controlling prompt records a user-attested personal communication with the
dataset author. The exact UTF-8 statement is hash-bound as
`3560662d4509ea3e059d597cedca15950cce02f706a992330b161381acfba6ba`.
It fixes one global rule: rotate every corresponding surface PNG clockwise by
90 degrees to obtain the scan-image orientation, with no additional crop of
the corresponding specimen outer frame. It permits differing pixel dimensions
or proportions.

The evidence type is
`USER_ATTESTED_PERSONAL_COMMUNICATION_WITH_DATASET_AUTHOR`, not an archived
direct message. No email address, account, timestamp, signature, or private
contact detail is inferred. A direct communication archive remains recommended
for later manuscript use but is not required for the internal P0R experiment.

## Registration Semantics

The author correspondence establishes normalized full-frame pixel
correspondence, not physical scanner calibration. The mapping basis is
`AUTHOR_FULL_FRAME_PIXEL_CORRESPONDENCE`, and
`physical_mm_used_for_cross_modal_mapping` is false.

The author transform is globally fixed as existing `Orientation.ROT90`, whose
image-coordinate mapping is `(u, v) -> (1 - v, u)`. No specimen-specific
orientation, mirror, offset, or target-driven alignment is permitted.

The destination is the specimen-specific registered C-scan crop, not a combined
raw screenshot. The full chain is:

```text
surface PNG
  -> fixed author CW90 normalized specimen frame
  -> known specimen panel within the raw scan screenshot
  -> historically registered specimen crop
  -> existing canonical 8x8 action grid
```

## Historical Processing Provenance

The historical implementation is
`external_hasebe/src/cmc_bbdm/hasebe.py`, function
`crop_cscan_panels`, SHA-256
`3bd0c56adb78b65cc09cec340b06c40c7ec6e73988e187067a76f1128ddf83f0`.
The external project copy has no Git metadata, so no source commit is claimed.

The verifier implements only the three fixed, source-proven crop recipes:

| Screenshot geometry | Panels | Crop boxes |
| --- | ---: | --- |
| `891x891` | 1 | `(31,33,706,707)` |
| `669x885` | 1 | `(39,33,469,708)` |
| `996x581` | 2 | `(30,33,370,371)`, `(464,33,816,371)` |

The operation is RGB decode followed by axis-aligned crop. It performs no
resize, interpolation, rotation, or reflection. Raw and processed C-scan pixels
are accepted only by this provenance verifier and can never enter the author
orientation API.

## Components

### `author_authority.py`

Defines the immutable/hashable author record, validates the exact evidence
type, statement hash, global ROT90 rule, no-crop specimen-frame claim, mapping
basis, optional archived-artifact hash, and absence of result-driven inputs.

### `scan_frame_provenance.py`

Defines the closed historical crop-recipe roster and replays raw screenshot to
registered crop extraction. It verifies panel index, input/output dimensions,
decoded RGB equality, and file hashes. It reports processing semantics without
exposing an orientation-selection interface.

### `p0r.py`

Replays the historical P0 authority, binds the author record and preprocessing
source, verifies all 276 processing chains, constructs one global CW90
full-frame transform per authorized specimen, maps the existing 64-cell grid,
applies the P0R gate, and writes a separate deterministic package.

### `p0r_qc.py`

Creates deterministic diagnostic overlays for two hash-selected specimens per
domain after registration is frozen. It never changes transforms or gate facts.

### Existing modules

- Extend `EvidenceRole` with `AUTHOR_CORRESPONDENCE`.
- Extend the existing deterministic artifact layer with a separate P0R writer
  and replay verifier.
- Extend the CLI with `audit-p0r` and `replay-p0r`; old commands remain exact.
- Reuse `SurfaceToCscanTransform`, `create_transform`, `Grid8x8`,
  `render_surface_grid`, `snapshot_file`, and the historical P0 package.

## P0R Data Flow

1. Validate the new config and refuse an existing output.
2. Replay the historical P0 package against compact and external authorities.
3. Require its historical status to remain the recorded NO-GO.
4. Build and validate the author authority from fixed config fields.
5. Snapshot the external paired manifest and historical preprocessing source.
6. For every historical 276-row surface authority record, bind raw screenshot,
   panel index, registered crop, and native surface geometry.
7. Replay the fixed panel crop and require exact decoded RGB equality.
8. Construct the fixed CW90 normalized full-frame transform to the registered
   crop and verify all 64 cell round trips.
9. Apply domain and total coverage gates.
10. Write deterministic package files atomically, then replay the package.
11. Generate overlays only after the transform and decision are fixed.

## P0R Gate

`P0R_AUTHOR_REGISTRATION_GO` requires all of the following:

- historical 276/276 identity/hash authority passes;
- author statement and source type are hash-bound;
- orientation is globally ROT90 and cannot be selected per specimen;
- every authorized row has a deterministic specimen panel;
- raw panel extraction exactly reproduces the registered crop pixels;
- no unsupported rotation/reflection or physical-mm claim is introduced;
- registration and 64-cell mapping replay exactly;
- no CAI, oracle, damage target, target domain label, or manual alignment enters
  orientation selection;
- each domain has at least 90% coverage and total coverage is at least 240.

Evidence contradiction yields `P0R_AUTHOR_EVIDENCE_CONFLICT`. An unavailable or
non-deterministic crop chain yields `P0R_PROCESSING_PROVENANCE_UNRESOLVED`.
Other gate failures yield `P0R_AUTHOR_REGISTRATION_NO_GO`. Every non-GO status
sets P1-P4 to `NOT_RUN_NOT_AUTHORIZED`.

## Artifacts And Replay

The P0R package has exact required membership from the controlling prompt.
Its manifest binds every payload file. Replay validates package membership,
checksums, source hashes, author statement identity, processing recipes,
decoded pixel equality, registration transform hashes, authorized roster,
64-cell mapping, and the recomputed gate. Committed artifacts contain only
logical relative paths.

QC overlays live outside the formal package under a dedicated artifact
directory with a hash manifest so they do not alter formal package membership.

## Error Handling

All authority, schema, path, hash, crop, transform, grid, and gate discrepancies
fail closed before output publication. Package writes use a staging directory
and atomic rename, refuse overwrite, reject symlinks and absolute paths, and
remove incomplete staging output on failure.

## Testing

Tests are written before production changes. They cover:

- exact author record and statement hash;
- inability to pass raw C-scan pixels or result targets into author authority;
- asymmetric CW90 corner semantics and numeric round trips;
- all three fixed crop layouts and dual-panel order;
- decoded-pixel equivalence and contradiction detection;
- separation of provenance pixels from orientation;
- 64-cell round trips for differing surface/destination dimensions;
- P0R GO, conflict, unresolved-provenance, and coverage gates;
- exact deterministic package membership and replay drift detection;
- CLI behavior and preservation of old P0 replay;
- unchanged historical/frozen paths.

## Conditional Downstream Work

This design ends at P0R. If P0R is GO, a separate committed P1 protocol and
implementation plan will freeze the authorized roster, registration hash,
surface preprocessing, ResNet18 feature identity, source-only nested selection,
controls, statistics, gates, seeds, and output paths before formal target-domain
evaluation. P2-P4 remain unauthorized until their respective upstream gates.

