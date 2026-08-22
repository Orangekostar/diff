# CFRP C-scan observability, spatial specificity, and multi-view CAI results

This repository is a compact, analysis-oriented export of the registered G1,
G2, and P1-P7 experiments. It contains machine-readable metrics, held-out-domain
predictions, reports, configuration snapshots, artifact manifests, and
publication PDFs. It also includes the mechanics-consistent multi-view CAI
regression implementation and its formal E1-E3 result packages. It excludes raw
images, source DOCX files, model checkpoints, posterior image arrays, and large
feature caches.

## Decision sequence

| Stage | Registered question | Decision |
| --- | --- | --- |
| G1 | Can surface observations predict scalar C-scan descriptors across datasets? | FAIL |
| G2 | Do measured or strict-OOF predicted scalar descriptors improve CAI prediction? | FAIL |
| P1 | Does the measured full C-scan field improve CAI prediction? | PASS |
| P2 | Can a surface-only student recover that advantage by privileged distillation? | FAIL |
| P3 | Does specimen-specific spatial organization carry incremental information? | PASS |
| P4 | Do the registered dense spatial representations beat the frozen global baseline? | NO_GO |
| P5 | Does a 25% sparse C-scan retain at least 80% of the full-field gain? | PASS |
| P6 | Does diffusion reconstruction improve mechanical prediction over simple interpolation? | NO_MECHANICAL_GAIN |
| P7 | Does adding surface information improve the sparse-scan predictor? | NO_COMPLEMENTARITY |
| E1 | Are FULL, 50%, and 25% mechanics-valid views predictively equivalent or complementary? | GO_BY_PREDICTIVE_EQUIVALENCE |
| E2 | Does cooperative prediction agreement improve strict cross-domain CAI? | NO_GO |
| E3 | Does static consistency-plus-complementarity fusion improve CAI? | NO_GO |

The primary response for G2 and P1-P7 is the published damaged-to-intact CAI
strength ratio, unit `1`. Confirmatory inference uses held-out datasets, not
individual specimens, as the inferential units.

## Central result

The evidence is asymmetric. Measured full-field information was mechanically
useful (P1), and destroying specimen-specific spatial organization degraded
prediction (P3). However, the registered surface student (P2), dense learned
representations (P4), diffusion reconstruction (P6), and surface-plus-sparse
fusion (P7) did not improve their registered comparators. A simple 25% sparse
scan retained 89.90% of the full-field gain (P5), while classical interpolation
outperformed the learned P6 reconstructions. The E1 audit found highly similar
predictions across mechanics-valid views, but E2 cooperative regression and all
E3 fusion methods failed to improve the frozen FULL comparator. E4 dynamic
gating and E5 transport were therefore not authorized.

## Directory guide

- `results/cross_stage/`: the historical P2 decision snapshot and the current
  P3-P7 extension ledger.
- `results/g1_scalar_observability/` and `results/g2_scalar_utility/`: frozen
  scalar experiments.
- `results/p1_full_field_oracle/` and `results/p2_privileged_transfer/`: frozen
  full-field oracle and student-transfer experiments.
- `results/p3_spatial_specificity/`: original fields and spatial-destruction
  controls.
- `results/p4_dense_representation/`: global, DINOv2, DDPM, and residual-fusion
  representations.
- `results/p5_sparse_scan/`: three densities and interpolation sensitivities.
- `results/p6_diffusion_reconstruction/`: six reconstruction methods, image
  metrics, mechanical predictions, uncertainty summaries, and registered gates.
- `results/p7_surface_sparse/`: sparse-only and surface-plus-sparse comparisons.
- `results/multiview/`: checksum-bound E1 audit, E2 cooperative regression, and
  E3 complementarity results, including ratio and recovered-MPa metrics.
- `src/cmc_bbdm/aei_multiview_regression/`: multi-view implementation; the
  matching CLI, configuration, and tests are under `scripts/`,
  `paper_v3/configs/`, and `tests/`.
- `analysis_tables/`: compact cross-stage tables and explicitly marked post-hoc
  P6 uncertainty diagnostics.

## Start here

1. Read `results/cross_stage/V3_EXTENDED_GATE_STATUS.md`.
2. Inspect `analysis_tables/extended_stage_gate_summary.csv`.
3. Read each stage `REPORT.md` and `summary.json`.
4. Use `ANALYSIS_PROMPT.md` for an independent failure analysis.
5. Read `docs/AEI_MULTIVIEW_RESULT_REPORT.md` for the multi-view outcome and
   `docs/AEI_MULTIVIEW_CLAIM_EVIDENCE_MATRIX.md` for claim boundaries.

## Interpretation constraints

- A failed gate is not converted into a positive result by a favorable point
  estimate.
- G1 scalar failure does not imply that spatial internal information is absent;
  P1 and P3 directly show otherwise.
- P3 establishes predictive spatial specificity, not a unique causal damage map.
- P5 establishes retained predictive information at registered coordinates; it
  does not establish successful full-field reconstruction.
- P6 preserves every measured point exactly. Its failure concerns unmeasured
  reconstruction and downstream mechanical utility.
- P7 is evidence against complementarity under the frozen fusion estimator, not
  proof that all surface measurements are physically irrelevant.
- `analysis_tables/p6_posthoc_uncertainty_diagnostics.csv` is descriptive and
  was not a registered endpoint.
- The E1 oracle is non-deployable. E1 predictive equivalence does not override
  the E2/E3 no-go decisions or authorize unrun E4/E5 branches.

## Reproducibility

P3-P7 production and replay packages are byte-identical within each stage. P6
production and replay both validate with:

- scientific digest `fc8431597eadc9f1dff9b956d16810b6821888eef0b24c3db4df8a0dff50d505`;
- output-tree digest `433699f015914c166696d2c19feaf8fe452482ff61200d663afb76fae6f493e9`;
- artifact-manifest SHA-256 `01bb169f5648d0809466e26a8482c7025bc3471de306a0cc2bf8c2f4b0e08e6e`.

The copied full-package artifact manifests document the source runs, including
files intentionally omitted from this compact repository. The root
`CHECKSUMS.sha256` remains the authority for the original P0-P7 compact export;
new multi-view results use their stage-local checksum ledgers.

The three `results/multiview/` stage packages independently include
`CHECKSUMS.sha256` and `artifact_manifest.json`. Their formal decisions are E1
`GO`, E2 `NO_GO`, and E3 `NO_GO`; all stored ratio metrics remained byte-identical
when the recovered-MPa reporting columns were added.

## Privacy-sanitized export note

Two P1 `weight_path` strings were previously changed from a machine-specific
absolute path to `paper_v3/assets/resnet18-f37072fd.pth`. This extension also
changes the non-scientific P5 `p3_package` path to
`results/p3_spatial_specificity`. Numeric values, predictions, metrics, source
hashes, and decisions are unchanged. The copied artifact manifests document the
original full runs; `CHECKSUMS.sha256` documents the exported bytes.
