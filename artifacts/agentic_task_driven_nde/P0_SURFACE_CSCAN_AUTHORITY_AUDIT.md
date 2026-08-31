# P0 Surface / C-scan Authority Audit

Date: 2026-08-31 UTC

Controlling prompt SHA-256:
`44debf1e9d98e4dd56e77409a01790d2eb08da5c3cb9936ad006e2fd3143764e`

Decision: `P0_SPATIAL_REGISTRATION_NO_GO`

This audit answers the twelve mandatory questions before visual-model
development. Machine-readable row authority is in
`results/agentic_task_driven_nde/p0_registration/`. No image, target, or
scientific endpoint was modified, and no model was trained.

## 1. Exact 276-specimen surface roster

**Answer: YES.** The compact frozen cohort selects exactly 276 unique primary
identities. Exact domain counts are:

| Domain | Count |
| --- | ---: |
| `74t7kcdgkr` | 45 |
| `cgtnjyggtm` | 49 |
| `w68dtmpfyf` | 43 |
| `xcmzfsbd9t` | 59 |
| `yfxyg8jm46` | 42 |
| `ykhs7s2dck` | 38 |

The external P0 roster has 278 primary-domain rows because it additionally
contains `74t7kcdgkr/c8-1` and `74t7kcdgkr/c8-10`. Those two rows are not in the
frozen compact cohort and were not silently admitted. The selected 276 rows
are listed in `surface_manifest.csv`; all have
`PASS_EXACT_SPECIMEN_ID_AND_HASH`.

Authority hashes:

- compact cohort: `d101fc7b9c762c8f650878c73a8666f824e07584f91f0aa4cc8e95fc88dfe9c4`
- external P0 roster: `84617cb12012ac9acc1b756b59161738a2a0a1089bd3eb432b7d01a066e65f9f`

## 2. Exact surface / C-scan / CAI matching

**Answer: YES for identity and separate file authority.** Pairing uses the
published specimen ID and domain, never row order, directory order, fuzzy
matching, or visual similarity. Every selected row binds the impacted surface,
raw C-scan screenshot, registered C-scan crop, and CAI identity.

The historical P0 roster places the registered-crop hash beside the raw
screenshot path for all 276 selected rows. Therefore those 276 path/hash pairs
do not hash-match the raw screenshots. This semantic defect is explicit as
`legacy_p0_raw_path_with_crop_hash_count = 276`. The separate paired manifest,
SHA-256
`f81002981bf2f6aed84818b48da87cd57e6336f5f3da8d78df1a58d26dd8026f`,
correctly distinguishes raw screenshot and registered crop paths/hashes and is
the binding authority used here.

`surface_manifest.csv` contains all exact identities and hashes. The package
contains 259 unique raw screenshots because some released screenshots contain
more than one specimen panel; registered crops and surface files remain 276
specimen-specific records each.

## 3. Surface formats, resolutions, and hashes

**Answer: fully enumerated.** All 276 selected surfaces are RGB PNG files with
native geometry preserved and no EXIF autorotation.

| Domain | Native resolution and count |
| --- | --- |
| `74t7kcdgkr` | `1500x1500` (16), `1679x1679` (4), `1680x1680` (1), `3357x3357` (24) |
| `cgtnjyggtm` | `3357x3357` (49) |
| `w68dtmpfyf` | `3357x3357` (43) |
| `xcmzfsbd9t` | `1500x1500` (20), `1679x1679` (6), `3357x3357` (33) |
| `yfxyg8jm46` | `3357x3357` (42) |
| `ykhs7s2dck` | `3357x3357` (38) |

Every specimen-level SHA-256, byte count, path, mode, format, and dimensions
are in `surface_manifest.csv` and `surface_qc.csv`. Their package hashes are:

- `surface_manifest.csv`:
  `31aaf123d6f7b684566ef19387d11a5ef2756e5bc51e666f8b1cdcc3b9ed5fe1`
- `surface_qc.csv`:
  `fe64077852ffea9af42e91f49b2aff03090ba9ba0900b5cacdc5454c95597b8c`

Visible annotations, specimen-boundary visibility, and physical image extent
are recorded as unknown rather than inferred from image content.

## 4. Surface physical-coordinate metadata

**Answer: insufficient for registration.** The source article documents an
`80 x 80 mm` CFRP specimen, a KEYENCE VR-5000 surface system, and a surface
profile reference plane determined from four specimen corners. It does not
publish the impacted RGB export crop, pixel-to-mm mapping, image-axis
directions, camera pose, impact coordinate, or a common coordinate frame with
the C-scan. The 80 mm specimen extent is source-declared specimen metadata; it
is not evidence that every image edge is the specimen boundary.

The article full-text XML used for this audit has SHA-256
`1994c62885ac9d2dc1a94b19e2e07c3ca0b9b895666d6cc547acc41070db010f`.

## 5. Registered C-scan physical-coordinate metadata

**Answer: insufficient for cross-instrument registration.** The source article
documents an HIS3 ultrasonic system, a nominal `75 x 75 mm` scan, and
`0.2 x 0.2 mm` scan pitch. The frozen registered crops have exact hashes and
pixel dimensions:

| Registered crop size | Count |
| --- | ---: |
| `340x338` | 17 |
| `352x338` | 19 |
| `675x674` | 240 |

No source authority records crop offsets, crop-axis directions, the mapping
from registered crop pixels to instrument coordinates, or a shared frame with
the surface export. Pixel size and published scan extent alone cannot resolve
that missing transform.

## 6. Orientation without hidden damage content

**Answer: NO.** No label, fiducial, export metadata, or geometry-only source
record resolves the eight legal axis-swap/reflection possibilities. All 276
rows therefore record `UNRESOLVED_8_WAY_AMBIGUITY`. Hidden C-scan pixels,
damage masks/centroids, CAI, oracle values, and manual target alignment were
not inspected or used to select an orientation.

## 7. Deterministic surface-region to legal 8x8 mapping

**Answer: NO across the two instrument frames.** The frozen C-scan action grid
is deterministic, row-major, and restricted to cells 0--63. Its SHA-256 is
`7a8b7b3c61b9f74597a5d1196b2e8561d10f6105b4df86a22cabbe3a5e9f56e2`.
Typed point, box, cell, and grid functions pass synthetic known-transform
tests, but deliberately reject an unresolved real transform. Normalizing both
images to `[0,1]` is not accepted as correspondence.

## 8. Frozen A2 initial mechanical teacher authority

**Answer: bound without exposing target values to P0.** The authority is
`results/mva/a2_oracle_value/oracle_values.parquet`, SHA-256
`6b289f2f6f74ac75dde47ea7cbfefcda1c49f025e74227bfb34ef269182ff963`.
The exact schema is:

```text
specimen_id, dataset_id, method, step, nominal_checkpoint, cell_index,
from_level, to_level, measured_count, native_count, effective_budget,
budget_before, candidate, primary_value, value, budget_after,
current_prediction, new_prediction, secondary_value, error_before,
error_after, current_error, new_error, selected,
p_a_predictor_state_sha256
```
Initial rows are `step = 0`, `from_level = 0`, `to_level = 1`, and nominal
checkpoint `0.0625`: 17,664 rows = 276 specimens x 64 legal cells. Candidate
identity equals cell identity. The aggregate identity/state SHA-256 is
`b8e55b426a7af124c3a0fc9ef30e2b5ad5a35b54264f7e5736b159178d5cc4d6`.
P0 reads schema and identity columns only; transform/QC code cannot accept the
teacher values.

## 9. Strongest compatible frozen static comparator

**Answer: `mvd_m1_o2`.** The frozen P7 summary explicitly names it as the
strongest deployable baseline. It is a static reference from this repository,
not an external published competitor.

- claim evidence SHA-256:
  `f37af43804d9c2b9a2021ca7f0fc43490257ca91bd8f6e917500a385480cf6e5`
- summary SHA-256:
  `966305c29bb6d0b13e408858a1db273df2b6971833dd678881bcf9d803b45302`

No new comparator or target evaluation was performed.

## 10. Local foundation visual / VLM model authority

| Model | Local status | Revision identity | License evidence |
| --- | --- | --- | --- |
| repository ResNet18 | `AVAILABLE` | torchvision `ResNet18_Weights.IMAGENET1K_V1`; weight SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` | BSD-3-Clause from installed torchvision 0.27.1; compatible |
| DINOv2 ViT-S/14 | `INCOMPLETE` | checkpoint only, 88,283,115 bytes; exact upstream revision not locally provable | `UNKNOWN` locally |
| `google/siglip2-giant-opt-patch16-384` | `ASSETS_COMPLETE_LICENSE_UNKNOWN` | snapshot `a713301b217d38485fb2204c808367d10bc3cc40` | no local model card/LICENSE evidence |
| `google/siglip2-so400m-patch16-naflex` | `ASSETS_COMPLETE_LICENSE_UNKNOWN` | snapshot `cc24074f717b612951c2dead130904ab9b65a81e` | no local model card/LICENSE evidence |
| `timm/ViT-SO400M-14-SigLIP-384` | `ASSETS_COMPLETE_LICENSE_UNKNOWN` | snapshot `ac16108d567c4389e6cd2b11c9b8585f7474435b` | no local model card/LICENSE evidence |
| Qwen2.5-VL | `NOT_FOUND` | none | not applicable |

Only the repository ResNet18 meets the strict local
asset + revision + license-evidence gate. This inventory does not authorize P1
because P0 failed first. No weight was loaded and no model inference ran.

## 11. External code reuse

**Answer: NO external code needs to be copied.** P0-P3 interfaces can be
implemented independently around the frozen local action/replay contracts.
The audited external projects remain architecture or prior-art references.
`EXTERNAL_REPOSITORY_AUDIT.md` records exact revisions, licenses, inspected
files, and `Code copied = NO` for every project. No external project can supply
the missing Hasebe cross-instrument transform.

## 12. Claims blocked by P0 or P1 failure

**P0 failure makes the following claim impossible:** that the released surface
image can legally guide specimen-specific selection of the frozen 8x8 C-scan
actions. Without a source-supported transform, surface regions and ultrasonic
actions do not share an authorized coordinate system; therefore surface action
observability, task conditioning, closed-loop surface fusion, and a VLM tool
agent cannot be evaluated under the stated protocol.

**A future P1 failure would make the following claim impossible:** that
specimen-specific surface visual evidence adds the missing held-out-domain
observability of frozen mechanical value beyond the preregistered no-image,
shuffled-image, spatial-derangement, and center controls. It would block P2-P4
under this route. Neither failure changes the already frozen retrospective MVA,
MVD, MAVIS, or AEI results.

## Authorization conclusion

All twelve questions have source-backed answers, but questions 6 and 7 are
negative. Formal visual-model development is therefore prohibited:

```text
P0 = P0_SPATIAL_REGISTRATION_NO_GO
P1 = NOT_RUN_NOT_AUTHORIZED
P2 = NOT_RUN_NOT_AUTHORIZED
P3 = NOT_RUN_NOT_AUTHORIZED
P4 = NOT_RUN_NOT_AUTHORIZED
```
