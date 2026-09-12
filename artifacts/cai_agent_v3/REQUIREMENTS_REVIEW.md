# CAI Agent v3 Requirements Review

Review mode: `SELF_CHECK_WITH_EXTERNAL_NUMERIC_ORACLE`. No independent reviewer context is claimed. The machine-readable final review is `requirements_review_final.json` and contains every required evidence field.

## Final finding

- Implementation: `IMPLEMENTATION_COMPLETE_EXECUTION_STOPPED_AT_W2_RESOURCE_LIMITED`.
- Protocol: `NONCONFORMANT_W2_CHECKPOINT_SELECTION_PRECISION`.
- Scientific evidence: W0 is valid; W2-W4 outputs are retained as invalidated diagnostics; formal three-seed TEST was not executed.
- Engineering status: `ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED`.

The earlier review incorrectly treated re-evaluation of the six retained W2 snapshots as proof that checkpoint selection was unchanged. The exact-cost refresh changed three VALID prefix masks. Because checkpoints from every 250-update selection point were not retained, the exact-cost ranking over training updates cannot be reconstructed. This violates the hard checkpoint-selection contract in Execution Sections 3.1, 5.3, and 9.2.

## Stage audit

| ID | Actual evidence | Status | Severity |
|---|---|---|---|
| A01 | Independent metric oracle 7/7 passes, but three exact-cost prefix masks changed after W2 checkpoint update selection | FAIL | BLOCKER |
| A02 | 432 v2 runs priced independently before aggregation; ensemble rows excluded | PASS | BLOCKER |
| A03 | Actor return math is correct, but W2 update selection used the pre-correction validation library | FAIL | BLOCKER |
| A04 | 276 specimens, 259 source groups, six domains, no group crosses split | PASS | BLOCKER |
| A05 | Frozen real Qwen cache: 205 valid and 6 unavailable over TRAIN/VALID; C0 applies once; TEST calls 0 | PASS | BLOCKER |
| A06 | Hidden-token and ablation invariance pass; TEST targets redacted; GDFS predictor hashes unchanged | PASS | BLOCKER |
| A07 | Forward hooks execute the registered two-layer, four-head, width-128 Transformer | PASS | BLOCKER |
| A08 | Measured tokens reach legal candidate scores; grouped soft-mask loss reaches the selector only | PASS | BLOCKER |
| A09 | Open-loop, no-VLM, and 64-logit true-static permission tests pass | PASS | BLOCKER |
| A10 | Corrected hard-route replay: 650 policy episodes/10,265 actions and 50 GDFS episodes/786 actions | PASS | BLOCKER |
| A11 | W3/GDFS were historically entered after an incomplete W2 refresh; corrected gate now blocks and invalidates them | FAIL | BLOCKER |
| A12 | 21,514 known updates plus at most 750 interrupted updates; TEST remained closed; Git verified after push | PASS | BLOCKER |

## Consequences

`predictor_gate.json` now reports `PREDICTOR_NOT_READY`; `oof_readiness.json` reports `REWARD_MODELS_NOT_READY`. The previous W3 values (main 44.627637 MPa, geometry 46.967150 MPa, open-loop 44.852927 MPa) and GDFS 46.335553 MPa remain reproducible diagnostics, but their status is `DIAGNOSTIC_ONLY_INVALIDATED` and they cannot support a continuation or performance claim.

The conservative remaining budget is 5,836 optimizer updates. A contract-compliant W2 replay alone has a registered upper bound of 12,000 updates, before replaying W3/GDFS. Per Execution Section 10, the run stops as `RESOURCE_LIMITED`; it does not shrink the cohort, lower the gate, or open TEST.

## Verification commands

```text
python /home/ww/diff/docs/CODEX_HANDOFF_CAI_AGENT_V3/metric_reference.py --self-test
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_cai_agent_v3.py
python -m ruff check src/cmc_bbdm/cai_agent_v3 tests/test_cai_agent_v3.py scripts/run_cai_agent_v3.py
PYTHONPATH=src python scripts/run_cai_agent_v3.py --project-root . refresh-cost-evaluation --device cpu
python scripts/run_cai_agent_v3.py --help
```
