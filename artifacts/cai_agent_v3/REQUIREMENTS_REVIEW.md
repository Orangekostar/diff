# CAI Agent v3 Requirements Review

Review mode: `SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE`. No independent reviewer context was claimed.

## Stage A: W0/W1 pre-optimization review

| ID | Stage | Source | Actual code/symbol | Independent expected behavior | Evidence | Status | Severity | Approved deviation |
|---|---|---|---|---|---|---|---|---|
| A01 | W0/W1 | Execution 3.1/3.3 | `metrics.py::left_error_area_mpa`, `trajectory_objective_mpa` | Left-step A=8/J=8.5 and tail A=6/J=6.5 | Supplied `metric_reference.py --self-test`: 7 PASS; production tests match | PASS | BLOCKER | null |
| A02 | W0/W1 | Execution 2.2 | `legacy_rescore.py::rescore_v2` | Price each seed/run before averaging losses | 432 runs reconstructed; `per_run_metrics.csv`; ensemble table excluded | PASS | BLOCKER | null |
| A03 | W0/W1 | Execution 3.2 | `metrics.py::torch_policy_cost_to_go` | Pre-action local costs, .25 terminal term, matching returns | Golden return vectors pass; full Actor loss not applicable before W3 | NOT_APPLICABLE_YET | BLOCKER | null |
| A04 | W0/W1 | Execution 4.1/4.2 | `cohort.py`, `cohort_export.py` | 276 candidates; source-only groups; group-isolated split | 276 included, 0 excluded, 259 groups, 17 multi-specimen groups, no group split | PASS | BLOCKER | null |
| A05 | W0/W1 | Execution 4.3/6.2 | `policy.py::vlm_first_action_mask`; bound 60-record cache | Real cache identity; highest reliable C0 once; action 1 unlock | C0 behavioral test PASS; full 276 VLM completion deferred until predictor readiness | NOT_APPLICABLE_YET | BLOCKER | null |
| A06 | W0/W1 | Execution 5.2/6.1 | `models.py` predictors and Actor | Hidden C-scan changes cannot affect visible-state inference | Three predictor variants pass hidden-token invariance; open-loop invariance PASS | PASS | BLOCKER | null |
| A07 | W0/W1 | Execution 5.2/6.3 | `SpatialPredictor`, `SpatialCAIActor` | 2 layers, 4 heads, width 128 contextualizer executes | Forward hook and architecture assertions PASS | PASS | BLOCKER | null |
| A08 | W0/W1 | Execution 6.3 | `SpatialCAIActor.forward` | Measured content reaches an unmeasured candidate logit | Nonzero gradient from measured cell token to legal cell-1 score | PASS | BLOCKER | null |
| A09 | W0/W1 | Execution 6.4 | `SpatialCAIActor`, `TrueStaticActor` | no-VLM/open-loop/static preserve exact input permissions | VLM invariance, C-scan/prediction invariance and parameter-list tests PASS | PASS | BLOCKER | null |
| A10 | W0/W1 | Execution 6.1 | Reused `NativeCellGrid`; v3 episode runner pending | Native integer-pixel cost; repeated action illegal; per-episode stop | Full v3 episode execution not yet reached | NOT_APPLICABLE_YET | BLOCKER | null |
| A11 | W0/W1 | Execution 5.4/6.5 | `gates.py::execute_if_status` | NOT_READY must not enter downstream Actor stage | Injected `PREDICTOR_NOT_READY` leaves action callback uncalled | PASS | BLOCKER | null |
| A12 | W0/W1 | Execution 9-11 | CLI, review, ledger and future remote check | Honest cumulative resources and actual remote SHA | CLI help PASS; final resource/remote evidence not yet applicable | NOT_APPLICABLE_YET | BLOCKER | null |

Commands executed:

```text
python /home/ww/diff/docs/CODEX_HANDOFF_CAI_AGENT_V3/metric_reference.py --self-test
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_cai_agent_v3.py
python -m ruff check src/cmc_bbdm/cai_agent_v3 tests/test_cai_agent_v3.py scripts/run_cai_agent_v3.py
python scripts/run_cai_agent_v3.py --project-root . prepare
python scripts/run_cai_agent_v3.py --help
```

Observed: independent oracle 7/7 PASS; production-focused tests 18 PASS; Ruff PASS; W0 `LEGACY_RESCORE_COMPLETE`; W1 `COHORT_READY`; CLI help exits successfully.

## Stage B: W2 predictor and OOF gate review

| ID | Stage | Source | Actual code/symbol | Independent expected behavior | Evidence | Status | Severity | Approved deviation |
|---|---|---|---|---|---|---|---|---|
| A01 | W2 | Execution 3.1 | `metrics.py`; `evaluate_predictor` | Checkpoint selection and gate use the same left-step area | VALID prefix library hash fixed; all evaluations call production left-step function | PASS | BLOCKER | null |
| A04 | W2 | Execution 4.2/5.5 | `_oof_assignments` | Query capture groups excluded; all domains represented | Fold N=57/53/51, groups=55/50/47, each fold covers six domains, intersection empty | PASS | BLOCKER | null |
| A06 | W2 | Execution 5.2 | Three predictor forwards | Hidden C-scan tokens replaced before any global aggregation | Hidden-token perturbation tests pass for all three predictors | PASS | BLOCKER | null |
| A07 | W2 | Execution 5.2 | `SpatialPredictor.contextualizer` | Spatial candidate actually executes 2-layer/4-head/128-width Transformer | Forward hook PASS; recorded architecture and parameter counts | PASS | BLOCKER | null |
| A11 | W2 | Execution 5.4/5.5 | `predictor_gate.json`; `oof_readiness.json` | Actor allowed only after predictor and reward gates | `PREDICTOR_READY`, selected `MEAN_SC`; all three OOF models `PREDICTOR_READY`; aggregate `REWARD_MODELS_READY` | PASS | BLOCKER | null |
| A12 | W2 | Execution 10 | `compute_ledger.jsonl` | Failed/precheck/formal updates counted cumulatively | 3 precheck + 5,000 candidate + 5,750 OOF = 10,753 updates; failed precheck recorded at 0 | PASS | BLOCKER | null |

W2 scientific evidence: `MEAN_SC` was selected as `P_all` by the registered VALID prefix-area rule. Its exact-cost VALID full-input MAE is 41.1409 MPa, zero-input MAE is 58.2584 MPa, and prefix area is 48.3126 MPa. Ridge full surface+C-scan MAE is 52.9166 MPa. These are internal VALID results, not TEST evidence or engineering readiness.

Before W3, native cell costs were promoted from float32 storage to float64 pixel fractions. Re-evaluation changed three final GEOMETRY_SPREAD masks among 3,329 VALID state rows. The selected checkpoint area changed by -0.0000134 MPa; `P_all=MEAN_SC` and all three OOF readiness decisions were unchanged. No optimizer update or model parameter changed. Evidence: `cost_precision_audit.json` with status `EXACT_NATIVE_COST_REEVALUATION_COMPLETE`.

## Stage C: W3-W7 final protocol review

| ID | Stage | Actual code/symbol | Observed evidence | Status | Severity |
|---|---|---|---|---|---|
| A01 | Final | `metrics.py` | Supplied independent oracle remains 7/7 PASS; production golden tests use left-step A=8/J=8.5 and tail A=6/J=6.5 | PASS | BLOCKER |
| A02 | Final | `legacy_rescore.py` | 432 stored `(specimen, method, seed)` runs priced independently before specimen aggregation; ensemble rows excluded | PASS | BLOCKER |
| A03 | W3 | `_training_rollout_loss` | Pre-action cost-to-go test passes; terminal weight is .25; corrected W3 uses exact float64 state costs | PASS | BLOCKER |
| A04 | Final | `cohort.py`, cohort artifacts | 276 specimens, 259 source-derived groups, TRAIN/VALID/TEST 161/50/65; no capture group crosses a split | PASS | BLOCKER |
| A05 | W3 | `vlm_perception.py`, `policy.py` | Frozen Qwen cache covers 211 fit specimens: 205 valid, 6 explicitly unavailable after one repair; C0 applies once and then unlocks | PASS | BLOCKER |
| A06 | W3/W4 | Actor and GDFS visible-state paths | Hidden-token invariance tests pass; GDFS predictor state hashes are identical before/after | PASS | BLOCKER |
| A07 | W3/W4 | `SpatialCAIActor.contextualizer` | Forward hooks execute the registered 2-layer/4-head/128-width Transformer in Actor and GDFS paths | PASS | BLOCKER |
| A08 | W3/W4 | feedback/selector gradient tests | Measured content reaches an unmeasured legal score; the grouped soft-mask loss reaches the selector while predictors remain frozen | PASS | BLOCKER |
| A09 | W3 | fixed method matrix and ablations | no-VLM, open-loop and 64-logit true-static permissions pass behavioral tests; all required seed-1 pilots executed | PASS | BLOCKER |
| A10 | W3/W4 | `_legal_action_mask`, hard evaluation | 650 policy episodes/10,265 actions and 50 GDFS episodes/786 actions replay with exact native costs, no repeats and independent termination | PASS | BLOCKER |
| A11 | W5 | expansion, VLM and TEST gates | Conservative update upper bound leaves 5,836, below 9,000 required; expansion is `RESOURCE_LIMITED`, TEST VLM calls=0 and TEST labels accessed=false | PASS | BLOCKER |
| A12 | Final | ledger, artifacts, Git checks | 21,514 known updates plus at most 750 interrupted invalid updates; 22,264 upper bound <28,100. Final containing commit SHA is reported after push to avoid self-reference | PASS | BLOCKER |

Final status is separated as required:

- `implementation_status`: `COMPLETE_THROUGH_AUTHORIZED_RESOURCE_GATE`
- `protocol_conformance`: `CONFORMANT_AFTER_NATIVE_COST_PRECISION_REMEDIATION`
- `scientific_evidence`: `SEED1_INTERNAL_VALID_PILOT_SUPPORTED; FORMAL_THREE_SEED_TEST_NOT_EXECUTED`
- `engineering_status`: `ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED`
- `review_mode`: `SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE`

The first W3 attempt exposed a float32 cumulative-cost legality round trip. Three completed checkpoints were invalidated and removed, and the interrupted static run is conservatively charged at its 750-update upper bound. The corrected run is the only W3 result in `policy_pilot_gate.json`. Full details are in `w3_cost_legality_audit.json`.
