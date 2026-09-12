# CAI Agent v3 Implementation Bindings

| ID | Existing authority read | Applicable fact | v3 implementation | Acceptance |
|---|---|---|---|---|
| C01 | `cai_active_image/statistics.py::normalized_error_area_mpa`; `learned_cscan/metrics.py::exact_step_integral` | V2 used trapezoids; old bounded loss type cannot carry MPa | `cai_agent_v3/metrics.py` | Golden `A=8`, tail `A=6`, Torch agreement |
| C02 | `cai_active_image/pipeline.py::_aggregate_prediction_rows/_effect_rows` | Seed predictions must not be ensembled before pricing | `legacy_rescore.py::rescore_v2` | 432 stored runs scored before specimen-level seed averaging |
| C03 | `cai_active_image/training.py::train_actor/train_predictor` | V2 post-action error, endpoint weight, fold sampling, batch stop are not reused | `metrics.py::torch_policy_cost_to_go`; v3 trainer is stage-gated | Synthetic return decomposition and unequal-termination checks |
| C04 | `cai_active_image/features.py::build_feature_bank`; `protocol.py` | Preserve per-cell frozen encoding, bypass old 60-person roster | `feature_bank.py::build_feature_bank` | 276-entry index, TEST labels redacted, each shard <50 MiB |
| C05 | P0R `surface_manifest.csv`; `hasebe_reference_evidence.py` | Exact 276 image identities, workbook MPa and capture sources | `cohort_export.py`; `cohort.py` | 276/276 label join, source-only union-find, no group split |
| C06 | `learned_cscan/perception.py`; `cai_active_image/perception.py` | Real frozen VLM, once-registered cells, highest reliable first-action C0 | `policy.py::vlm_first_action_mask` plus bound cache | Highest level only at action 0; all legal cells after action 1 |
| C07 | `learned_cscan/policies.py::LearnedCellActor` | Reuse 2-layer, 4-head, width-128 spatial pattern, not old weights or typed inputs | `models.py::SpatialCAIActor` | Forward hook, architecture attributes, feedback gradient to legal candidates |
| C08 | `mva/cai_evaluator.py`; `mva/encoder_session.py` | Fixed Ridge diagnostic and frozen ResNet18 execution | v3 predictor diagnostics; `feature_bank.py` | TRAIN-only transforms, alpha 10, full and partial VALID metrics |
| C09 | `V3_EXTENDED_GATE_STATUS.md`; G1 handoff | Prior spatial/sparse positives and fusion/policy negatives are motivation only | Predictor and policy gates in `gates.py` | Failed gate prevents downstream execution |
| C10 | `cai_active_image/environment.py`; `episodes.py` | Native 8x8 integer pixels, reveal after action, no repeated cells | V3 episode runner uses `NativeCellGrid` | Non-divisible raster, legality, per-episode stop tests |
| C11 | `tests/test_cai_active_image_v2.py` | Do not inherit trapezoid `6.0` or illegal-cell-only history tests | `tests/test_cai_agent_v3.py` | Independent golden metrics and unmeasured legal-logit feedback path |
| C12 | Supplied review and numeric oracle | Staged review, cumulative resource ledger and real GitHub state | `gates.py`, `REQUIREMENTS_REVIEW.md`, `compute_ledger.jsonl` | W0/W1, W2 and final reviews; local/upstream/remote SHA equality |

External method binding: `iancovert/dynamic-selection@e2b6f7403fdac4d217ac2ec5dea96acd60240b60` is limited to the optional W4 grouped-mask/Concrete training relaxation. The exact MIT text and adaptation notice are preserved in `third_party/dynamic-selection/`. The predictor is frozen, selection groups are 64 whole 512-dimensional cells, the selector sees only hard-visible state, and VALID/conditional TEST are hard acquisition. It is not used by W0-W3 and is not an exact reproduction.
