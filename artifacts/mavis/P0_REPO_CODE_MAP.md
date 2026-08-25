# MAVIS P0 Repository Code Map

Frozen means MAVIS may call the implementation but must not alter its historical
code, configuration, authority, or result artifacts.

| Scientific role | Exact file | Exact symbol | Input | Output | Target label used? | Full future scan used? | Frozen? |
|---|---|---|---|---|---|---|---|
| Public cohort manifest | `/home/ww/paper3/cmc_damage_inference/src/cmc_bbdm/cpb_v3/data.py` | `load_data`, `V3Data`, `CscanRecord` | P1 config, paired manifest, public files | specimen/domain join, metadata13, CAI, C-scan records | Yes, authority only | Record binds full scan | Yes |
| Raw C-scan read and hash | `/home/ww/paper3/cmc_damage_inference/src/cmc_bbdm/cpb_v3/data.py` | `CscanRecord.read_bytes` | specimen-bound path and SHA256 | verified PNG bytes | No | Yes, privileged | Yes |
| Registered 276 authority | `src/cmc_bbdm/mgmr/authority.py` | `MGMRM0Authority`, `load_authority` | registered protocol and public root | 276 read-only RGB scans, CAI, metadata13 | Yes, isolated authority | Yes, privileged | Yes |
| MVA authority | `src/cmc_bbdm/mva/authority.py` | `MVAAuthority`, `load_mva_authority` | MGMR authority, registered B0 | scans, labels, metadata, full embeddings | Yes | Yes | Yes |
| Acquisition grid | `src/cmc_bbdm/mva/acquisition_grid.py` | `build_acquisition_grid` | native raster shape, initial budget | 8x8 hierarchical grid | No | No | Yes |
| Measurement state and cost | `src/cmc_bbdm/mva/measurement_state.py` | `initial_state`, `apply_action`, `budget_record`, `candidate_budget_record` | grid, acquired cells/levels, candidate action | immutable state, exact unique native-raster counts | No | No | Yes |
| Causal simulator primitive | `src/cmc_bbdm/mva/interpolation.py` | `reconstruct_measurement_state`, `refine_reconstruction` | privileged scan plus legal measurement state | reconstruction with measured pixels restored exactly | No | Yes, simulator-only | Reuse without modifying |
| Candidate/action bank | `src/cmc_bbdm/mva/a4_candidate_bank.py` | `CandidateBank`, `build_candidate_bank` | full scan, initial state, 64 candidates, encoder | initial/counterfactual embeddings, values, costs, hashes | No | Yes | Yes |
| Strict source teacher | `src/cmc_bbdm/mva/a4_source_labels.py` | `generate_source_labels` | source candidate bank, source CAI, fold roster | strict-OOF source mechanical value labels | Source only | Candidate outcomes only | Reuse without modifying |
| CAI estimator | `src/cmc_bbdm/mva/cai_evaluator.py` | `fit_cai_predictor`, `fit_sensitivity_cai_predictor` | source embeddings and CAI | fitted PCA/ridge predictor | Source training only | No after embeddings supplied | Reuse without modifying |
| Acquisition baselines | `src/cmc_bbdm/mva/oracle.py` | `choose_uniform_action`, `choose_random_action`, `choose_greedy_oracle_action` | legal actions and optional source teacher values | next action | Oracle: source only | Oracle: source counterfactuals | Reuse without modifying |
| MVA evaluation | `src/cmc_bbdm/mva/evaluation.py` | outer evaluation functions | trajectories, strict-OOF evaluator | specimen/domain budget curves | Evaluation only | Method-dependent | Yes |
| Compact MVD authority | `src/cmc_bbdm/mvd/authority.py` | `CompactMVDAuthority`, `load_compact_mvd_authority` | MVA artifacts and MVD config | embeddings/features/costs without raw scan | Source labels are separate | Candidate embeddings are full-scan-derived | Yes |
| MVD M0 one-shot plan | `src/cmc_bbdm/mvd/one_shot_oracle.py` | `score_initial_ranking`, `plan_frozen_ranking` | initial candidate scores and exact costs | one frozen ranking and checkpoint plan | Source oracle scoring only | Privileged oracle values | Yes |
| MVD M1 dataset | `src/cmc_bbdm/mvd/observability_dataset.py` | `build_outer_observability_examples` | deployable initial/candidate features plus source labels | outer-fold student examples | Source labels only | Student: no | Yes |
| MVD M1 models | `src/cmc_bbdm/mvd/observability_models.py` | `fit_ridge_scorer`, `fit_mlp_scorer` | source observability examples | static candidate scorer | Source only | No | Yes |
| Budget metric | `src/cmc_bbdm/mva/budget_metrics.py` | `auebc` | monotone exact-cost curve | scalar AUEBC | No | No | Yes |
| Domain uncertainty | `src/cmc_bbdm/mva/statistics.py` | `paired_domain_bootstrap` | paired domain effects | synchronized bootstrap interval | No | No | Yes |
| Artifact replay | `src/cmc_bbdm/mvd/replay.py` | `verify_checksums`, `replay_mvd_packages` | formal result package | byte/hash verification summary | No | No | Yes |

## Proven dependency chain

```text
paired.csv + hash-bound public CFRP files
  -> cpb_v3.load_data / CscanRecord
  -> mgmr.load_authority (276 specimens, 6 domains)
  -> mva.load_mva_authority
  -> MVA grid/state/candidate bank
  -> artifacts/mvd_authority/*.npz
  -> MVD M0 frozen one-shot evaluation
  -> MVD M1 static observability evaluation
  -> AUEBC/domain bootstrap
  -> checksummed formal and replay packages
```

The repository overlay imports the complete upstream package from
`/home/ww/paper3/cmc_damage_inference`. The pairing, hashes, cohort count, and
runtime loader were re-executed. Physical scanner path, physical point spacing,
and scanner time are `UNRESOLVED`; only native-raster measurement cost is
authoritative.
