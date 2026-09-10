# Source Bindings and Field Lineage

## Workbook audit

| Binding | Path | Sheet(s) | Formula count |
|---|---|---|---:|
| impact_condition_74t7kcdgkr | `data/public/hasebe/raw/74t7kcdgkr/v1/root/impact condition_c8.xlsx` | condition | 0 |
| impact_condition_yfxyg8jm46 | `data/public/hasebe/raw/yfxyg8jm46/v1/root/impact condition_c16.xlsx` | condition | 0 |
| impact_condition_xcmzfsbd9t | `data/public/hasebe/raw/xcmzfsbd9t/v1/root/impact condition_c24.xlsx` | condition | 0 |
| impact_condition_ykhs7s2dck | `data/public/hasebe/raw/ykhs7s2dck/v1/root/impact condition_q8.xlsx` | condition | 0 |
| impact_condition_w68dtmpfyf | `data/public/hasebe/raw/w68dtmpfyf/v1/root/impact condition_q16.xlsx` | condition | 0 |
| impact_condition_cgtnjyggtm | `data/public/hasebe/raw/cgtnjyggtm/v1/root/impact condition_q24.xlsx` | condition | 0 |
| lvi_workbook | `data/public/hasebe_cai/raw/8scdmfdcfb/v3/1_Low velocity impact testing condition/Low velocity impact testing condition and damage_v0.2.xlsx` | LVI condition | 0 |
| size_workbook | `data/public/hasebe_cai/raw/8scdmfdcfb/v3/2_Specimen size/Specimen size_v0.2.xlsx` | Specimen size | 0 |
| cai_workbook | `data/public/hasebe_cai/raw/8scdmfdcfb/v3/5_Compression after impact strength/Compression after impact strength_v0.2.xlsx` | CAI strength | 0 |

All nine workbooks contain one visible sheet, no formula cells, no merged ranges, and no embedded drawing/image/chart objects detected by the workbook audit.

## Field bindings

| Sheet/cell | Raw field/unit | Normalized meaning |
|---|---|---|
| condition!A1 / - | c or q  | LAMINATE_FAMILY |
| condition!B1 / - | ply  | PLY_COUNT |
| condition!C1 / - | no.  | SPECIMEN_NUMBER |
| condition!D1 / - | impactor  | IMPACTOR_SHAPE |
| condition!E1 / E1 | J/mm J/mm | IMPACT_ENERGY_PER_THICKNESS |
| condition!F1 / - | Angle  | SCAN_ANGLE |
| condition!A1 / - | c or q  | LAMINATE_FAMILY |
| condition!B1 / - | ply  | PLY_COUNT |
| condition!C1 / - | no.  | SPECIMEN_NUMBER |
| condition!D1 / - | impactor  | IMPACTOR_SHAPE |
| condition!E1 / E1 | J/mm J/mm | IMPACT_ENERGY_PER_THICKNESS |
| condition!F1 / - | Angle  | SCAN_ANGLE |
| condition!A1 / - | c or q  | LAMINATE_FAMILY |
| condition!B1 / - | ply  | PLY_COUNT |
| condition!C1 / - | no.  | SPECIMEN_NUMBER |
| condition!D1 / - | impactor  | IMPACTOR_SHAPE |
| condition!E1 / E1 | J/mm J/mm | IMPACT_ENERGY_PER_THICKNESS |
| condition!F1 / - | Angle  | SCAN_ANGLE |
| condition!A1 / - | c or q  | LAMINATE_FAMILY |
| condition!B1 / - | ply  | PLY_COUNT |
| condition!C1 / - | no.  | SPECIMEN_NUMBER |
| condition!D1 / - | impactor  | IMPACTOR_SHAPE |
| condition!E1 / E1 | J/mm J/mm | IMPACT_ENERGY_PER_THICKNESS |
| condition!F1 / - | Angle  | SCAN_ANGLE |
| condition!A1 / - | c or q  | LAMINATE_FAMILY |
| condition!B1 / - | ply  | PLY_COUNT |
| condition!C1 / - | no.  | SPECIMEN_NUMBER |
| condition!D1 / - | impactor  | IMPACTOR_SHAPE |
| condition!E1 / E1 | J/mm J/mm | IMPACT_ENERGY_PER_THICKNESS |
| condition!F1 / - | Angle  | SCAN_ANGLE |
| condition!A1 / - | c or q  | LAMINATE_FAMILY |
| condition!B1 / - | ply  | PLY_COUNT |
| condition!C1 / - | no.  | SPECIMEN_NUMBER |
| condition!D1 / - | impactor  | IMPACTOR_SHAPE |
| condition!E1 / E1 | J/mm J/mm | IMPACT_ENERGY_PER_THICKNESS |
| condition!F1 / - | Angle  | SCAN_ANGLE |
| LVI condition!B2 / - | Specimen No.  | SPECIMEN_ID |
| LVI condition!C3 / - | Layup  | LAMINATE_FAMILY |
| LVI condition!D3 / - | Impactor shape  | IMPACTOR_SHAPE |
| LVI condition!E3 / E4 | Impact energy [J/mm] | TOTAL_IMPACT_ENERGY |
| LVI condition!E3 / F4 | Impact energy [J] | IMPACT_ENERGY_PER_THICKNESS |
| LVI condition!G3 / G4 | Projected delamination area [mm2] | PROJECTED_DELAMINATION_AREA |
| LVI condition!H3 / H4 | Dent depth [mm] | DENT_DEPTH |
| LVI condition!I3 / - | Is included  | CAI_INCLUDED_FLAG |
| Specimen size!B2 / - | Specimen No.  | SPECIMEN_ID |
| Specimen size!C3 / C2 | Height mm | SPECIMEN_HEIGHT |
| Specimen size!D3 / C2 | Width mm | SPECIMEN_WIDTH |
| Specimen size!E3 / C2 | Thickness mm | SPECIMEN_THICKNESS |
| CAI strength!B2 / - | Specimen No.  | SPECIMEN_ID |
| CAI strength!C2 / C3 | Compression after impact strength [Mpa] | CAI_STRENGTH |
| CAI strength!C2 / D3 | Compression after impact strength [%] | CAI_STRENGTH_REDUCTION |

## Damage-measurement lineage

`CAI LVI workbook, LVI condition!G/H (Excel row retained)` -> `cpb_published_measurements.py::read_published_damage_measurements` -> `build_matched_measurement_rows` -> `published_damage_measurements.csv` -> this task's exact TEST-24 crosswalk.

The historical CSV contains 278 rows and matched the bound workbook directly with 0 value mismatches. It is an author scalar transcription, not a mask.

## Local proxy lineage

Registered C-scan RGB -> threshold/morphology candidates -> calibration against author area scalars -> selected `bg40_rb20_close08` -> `physical_descriptors.csv`. Classification: `LOCAL_DERIVED_PROXY`. The friendly historical `descriptor_source` label does not make these rows author spatial ground truth.

The LVI workbook's E4/F4 unit labels are reversed relative to numeric values and the specimen-thickness relation. This extraction preserves raw labels and records corrected normalized meanings explicitly.
The relation `total J = J/mm x thickness mm` was audited for 420 bound CAI records: 298 exact, 419 within 5%, and 1 beyond 5%. The normalized semantics follow the bound source reader's explicit correction; this relation is supporting context, not a zero-error identity claim.
