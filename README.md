# CFRP C-scan observability and privileged-transfer results

This repository is a compact, analysis-oriented export of the registered G1,
G2, P1, and P2 experiments. It contains reports, configuration snapshots,
machine-readable metrics, held-out-domain predictions, negative controls, and
artifact manifests. It does not contain raw images, source DOCX files, model
checkpoints, bootstrap arrays, or large intermediate training caches.

## Decision sequence

| Stage | Question | Decision |
| --- | --- | --- |
| G1 | Can surface observations predict three scalar C-scan descriptors across datasets? | FAIL |
| G2 | Do measured or strict-OOF predicted scalar descriptors improve CAI prediction? | FAIL |
| P1 / G2b | Does the measured full C-scan field improve CAI prediction? | PASS |
| P2 | Can a surface-only student recover the full-field advantage by privileged distillation? | FAIL |

The primary response for G2, P1, and P2 is the published damaged-to-intact CAI
strength ratio, unit `1`. Confirmatory inference uses held-out datasets, not
individual specimens, as the inferential units.

## Directory guide

- `results/cross_stage/`: authoritative decision ledger and pre-V3 evidence.
- `results/g1_scalar_observability/`: 292 specimens, seven datasets, physical
  area/height/width targets.
- `results/g2_scalar_utility/`: 276 specimens, six datasets, four matched CAI
  pathways using surface, measured scalars, strict-OOF scalars, and deranged
  scalars.
- `results/p1_full_field_oracle/`: measured full-field representations,
  handcrafted/frozen/learned alternatives, shuffle controls, and strict nested
  leave-one-dataset-out results.
- `results/p2_privileged_transfer/`: equal-capacity surface student, MSPD,
  scalar teacher, shuffled teachers, random teacher, predictions, effects, and
  gate conditions.
- `analysis_tables/`: compact cross-stage and per-domain comparisons derived
  directly from the exported result tables.

## Start here

1. Read `results/cross_stage/V3_FINAL_GATE_STATUS.md`.
2. Read the four stage reports and their go/no-go files.
3. Inspect `analysis_tables/domain_effect_alignment.csv` before interpreting
   average improvements.
4. Inspect `analysis_tables/p2_teacher_prediction_similarity.csv` to determine
   whether the P2 gain is specific to the authentic teacher.
5. Use the questions in `ANALYSIS_PROMPT.md` for an independent analysis.

## Interpretation constraints

- A failed gate is not converted into a positive result by a favorable point
  estimate.
- G1 scalar failure does not prove that all internal information is absent from
  the surface.
- P1 establishes bounded predictive utility of a measured full-field
  representation; it does not establish surface observability or causal
  sufficiency.
- P2 must outperform the equal-capacity student and teacher-mismatch controls
  under the registered AND gate before any privileged-transfer claim is made.
- Positive deltas in the derived domain table mean lower MAE for the candidate.

`CHECKSUMS.sha256` binds every exported non-Git file.

## Export note

Two non-scientific `weight_path` strings in the P1 configuration and
preprocessing manifest were changed from a machine-specific absolute path to
`paper_v3/assets/resnet18-f37072fd.pth`. Numeric values, hashes of the frozen
weights, predictions, metrics, splits, and decisions were not changed. The
copied source artifact manifest therefore documents the original formal run,
while `CHECKSUMS.sha256` documents the bytes in this privacy-sanitized export.
