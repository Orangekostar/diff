# CFRP C-scan observability, spatial specificity, and multi-view CAI results

This repository is a compact, analysis-oriented export of the registered G1,
G2, and P1-P7 experiments. It contains machine-readable metrics, held-out-domain
predictions, reports, configuration snapshots, artifact manifests, and
publication PDFs. It also includes the mechanics-consistent multi-view, MSSS,
MGMR M0, and MVA A0-A3 implementations with their formal result packages. It
excludes raw images, source DOCX files, model checkpoints, posterior image
arrays, and unregistered feature caches.

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
| MSSS S1 | Is there a cross-axis mechanically sufficient and spatially specific scale? | NO_GO |
| MSSS NO-GO diagnostic | Does the descriptive scale vary with laminate or damage size? | EXPLORATORY_SIGNAL_NOT_VALIDATED |
| MSSS S2 | Does source-selected MSSS transfer across laminate structures? | NOT_RUN_NOT_AUTHORIZED |
| MGMR M0 | Are coarse and directional morphology complementary under strict LODO? | NO_GO |
| MVA A0-A3 | Does a CAI oracle reveal useful task-driven acquisition headroom? | MVA_ORACLE_GO |

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

The formal MSSS scale-discovery study found positive spatial-specificity effects
on all three axes, but no axis passed the complete sufficiency gate. Sampling
lacked a confirmed over-coarse boundary; Gaussian was unstable and failed
cross-fitted non-inferiority; wavelet also failed cross-fitted non-inferiority.
S1 therefore issued `NO_GO`, and the pre-registered S2 transfer branch was not
run. The required post-`NO_GO` diagnostic found 3/3-axis coarser directions for
quasi-isotropic layup, higher damage area, and higher damage width, but not for
ply count or damage height. These post-hoc signals do not validate coupling or
change the S1/S2 gates.

The registered MGMR M0 study also issued `NO_GO`. Coarse plus directional B3
was worse than both component baselines, while directional correction increased
the coarse and FULL residual MAEs. Gates A-D all failed, so graph and
laminate-aware M1 development was stopped.

The preregistered MVA A0-A3 study issued `MVA_ORACLE_GO`. At 12.5% simulated
measurement budget, the retrospective mechanical oracle reduced equal-domain
CAI MAE by 35.43% versus uniform in 6/6 held-out domains. Its AUEBC was
0.01062453, versus 0.01719904 for the reconstruction oracle and 0.01770459 for
appearance-first acquisition. This establishes diagnostic oracle headroom only;
it is not a deployable policy or evidence of physical inspection-time savings.

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
- `results/msss/`: byte-identical formal and replay S1 packages plus the
  post-`NO_GO` coupling diagnostic, with manifests and checksums.
- `results/mgmr/`: registered FULL/coarse/P3 feature banks and byte-identical
  formal/replay M0 packages.
- `results/mva/`: A0 acquisition audit, A1 nested simulator validation, and
  byte-identical formal/replay A2 oracle-value packages with figures and raw
  candidate/trajectory tables.
- `src/cmc_bbdm/aei_multiview_regression/`: multi-view implementation; the
  matching CLI, configuration, and tests are under `scripts/`,
  `paper_v3/configs/`, and `tests/`.
- `src/cmc_bbdm/msss/`: three-axis scale discovery, source-only selection,
  conditional transfer, artifacts, and replay implementation.
- `src/cmc_bbdm/mgmr/`: spatial feature extraction, feature-domain DWT,
  nested component evaluation, strict residual audit, gates, artifacts, and replay.
- `src/cmc_bbdm/mva/`: normalized-grid refinement, strict cross-fitting,
  acquisition controls/oracles, budget metrics, gates, artifacts, figures, and replay.
- `analysis_tables/`: compact cross-stage tables and explicitly marked post-hoc
  P6 uncertainty diagnostics.

## Start here

1. Read `results/cross_stage/V3_EXTENDED_GATE_STATUS.md`.
2. Inspect `analysis_tables/extended_stage_gate_summary.csv`.
3. Read each stage `REPORT.md` and `summary.json`.
4. Use `ANALYSIS_PROMPT.md` for an independent failure analysis.
5. Read `docs/AEI_MULTIVIEW_RESULT_REPORT.md` for the multi-view outcome and
   `docs/AEI_MULTIVIEW_CLAIM_EVIDENCE_MATRIX.md` for claim boundaries.
6. Read `results/msss/s1_scale_discovery/REPORT.md` and
   `docs/MSSS_CLAIM_EVIDENCE_RESULT.md` for the MSSS decision boundary; use
   `docs/MSSS_COMPLETION_AUDIT.md` for the requirement-by-requirement audit.
7. Read `results/mgmr/m0_component_gate/REPORT.md` and
   `docs/MGMR_COMPLETION_AUDIT.md` for the MGMR M0 stop decision.
8. Read `results/mva/a2_oracle_value/REPORT.md` and
   `docs/MVA_COMPLETION_AUDIT.md` for the MVA A0-A3 decision and claim boundary.

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
- MSSS component-level spatial specificity does not override the complete S1
  `NO_GO`, establish a transferable scale, or authorize S2.
- The post-`NO_GO` coupling signal is descriptive and reuses the inspected S1
  cohort; it is not independent validation of Scale-Laminate Coupling.
- MGMR M0 reused six previously inspected domains. Its failed complementarity
  and residual gates do not authorize M1 graph or laminate-aware claims.
- The MVA mechanical oracle uses true CAI and unobserved candidate RGB values.
  It is a retrospective normalized-raster upper bound, not a deployable
  acquisition method; A4-A7 were not implemented or executed.

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

The formal and replay MSSS S1 packages are byte-identical and validate with:

- scientific digest `6ac389b0a4e09487202f5a8a9273dfdf5b338ef40de705661c5877e3e9bd0152`;
- output-tree digest `e41c42bdd8cb022b2d7d3c286685ae2530f2c302a236cf2e0a77d76ecf6a365b`.

The formal and replay post-`NO_GO` diagnostic packages are byte-identical and
validate with:

- scientific digest `e59dff40aa1d588c6654795c8130b22fa1be66950824d6224afc673418897203`;
- output-tree digest `9a2c972055b364815cacfc0c31e4cf29f928645832f07918ddd4f8055ac03318`.

The formal and replay MGMR M0 packages are byte-identical and validate with:

- scientific digest `5f03f58f57543f18c06dc29ae6c7f9abe4534ddcce37096ddaa1b66856283c54`;
- output-tree digest `8294c4c108c74bd13d34c49078c1873da8b76037a5de8b9f720a2c1d2ff45aaf`;
- primary feature manifest `f8961b9c8cee8a37fb5f54b4c3a62298999f3a11bfcd5d7e588671389ddc0d2e`;
- P3 feature manifest `749d2acb5c9703eb7dea4f58ddac10316d93e8f41b4f7de7fcf64aefc0534b39`.

The formal and replay MVA A2 packages are byte-identical and validate with:

- output-tree digest `71d279dcd2dc1da9a09d08164669e9dc9432eea1da6476f5037fbad5aecc7595`;
- artifact-manifest SHA-256 `a5499078463dccd3092bcedb795dd872e5196da85527201a291883aea5fe545c`;
- synchronized-bootstrap index digest `732abea9fa90893d53c6a0620c020a2072cd97a2b7d8e48e16aa8feb10c5564c`.

## Privacy-sanitized export note

Two P1 `weight_path` strings were previously changed from a machine-specific
absolute path to `paper_v3/assets/resnet18-f37072fd.pth`. This extension also
changes the non-scientific P5 `p3_package` path to
`results/p3_spatial_specificity`. Numeric values, predictions, metrics, source
hashes, and decisions are unchanged. The copied artifact manifests document the
original full runs; `CHECKSUMS.sha256` documents the exported bytes.

The MGMR implementation plan's machine-specific compact-repository target was
also replaced with a generic phrase in this export. Its scientific protocol,
configuration snapshot, predictions, metrics, manifests, and decisions are
unchanged; stage-local checksum ledgers document the formal bytes.

The MVA implementation plan received the same path-only sanitization. Its
frozen protocol, configuration, code, formal/replay results, and stage-local
checksums are unchanged.
